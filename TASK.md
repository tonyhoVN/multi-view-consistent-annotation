# Task: Reachability-Filtered, Overlap-Aware Multi-View Scan

## Objective

Replace the fixed grid traversal in `scripts/multi_view_scan.py` with a
deterministic planning pipeline that minimizes coordinated dual-arm motion while
maintaining strong visual overlap between consecutive RGB-D frames:

1. generate mirrored spiral camera-pose pairs on the upper hemisphere;
2. convert camera poses to TCP targets using measured hand-eye TF;
3. call MoveIt IK without moving the robot and discard unreachable/colliding
   pose pairs;
4. build an open TSP path over the reachable joint configurations;
5. improve the path with overlap-constrained 2-opt;
6. execute and capture the optimized path, continuing past motion failures.

The left camera always owns the target with world `y > 0`; the right camera
owns the mirrored target with world `y < 0`. Both arms move together.

## Scope and constraints

- Preserve the current `RobotAPI` motion, TF, camera synchronization, simulated
  time, ready-state, manifest, cleanup, and failure-continuation behavior.
- IK filtering must be read-only: call `/compute_ik`; never use a motion service
  to test reachability.
- Set `avoid_collisions=True` in every IK request.
- Use the current `/joint_states` message as the full-state IK seed.
- Treat a paired left/right target as one TSP vertex. Reject the entire vertex
  if the multi-tip IK request fails.
- Make the motion distance selectable. Joint mode uses Euclidean distance
  between returned IK solutions. Pose mode uses coordinated left/right TCP
  translation and geodesic SO(3) rotation distance, including the measured
  current TCP pair when selecting the first vertex. Produce an open path in
  either mode (do not return to the start).
- Calculate neighbor overlap by projecting deterministic samples from the
  boundary of a configurable spherical scan volume into both camera frames
  using calibrated pinhole intrinsics. Treat each sample's outward radial vector
  as its surface normal; a sample is visible when it is front-facing, has
  positive depth, and lies inside the image bounds. This surface proxy prevents
  distant look-at views from incorrectly receiving 100% overlap merely because
  the whole bounding volume fits in both frustums. Compute temporal overlap
  independently for the left and right cameras and use the smaller value as the
  edge constraint.
- A 2-opt reversal is valid only when every adjacent edge in the resulting open
  path satisfies `minimum_neighbor_overlap`. Among valid reversals, accept only
  strict improvements to the combined joint-motion and overlap cost.
- Algorithms must be deterministic for identical CLI arguments and IK results.
- Keep pure geometry and path-optimization code independent from ROS so it can
  be unit tested without Isaac Sim or MoveIt.
- Long functions must contain short procedural comments at the relevant stages.

## Design

### Planning equations and execution order

For every reachable paired viewpoint `i`, retain one coordinated IK vector
`q_i` ordered by a stable, common joint-name list. Precompute each matrix only
once before motion:

- joint mode: `D_q(i,j) = ||q_i - q_j||_2`;
- pose mode: combine the root-sum-square left/right TCP translations with the
  weighted root-sum-square left/right geodesic rotation angles;
- per-arm visibility set: projected, front-facing sphere sample IDs;
- per-arm temporal overlap: `|V_i intersect V_j| / |V_i union V_j|`;
- paired overlap: `O(i,j) = min(O_left(i,j), O_right(i,j))`;
- feasible edge: `O(i,j) >= minimum_neighbor_overlap`;
- edge objective: `C(i,j) = D_selected(i,j) + overlap_weight * (1 - O(i,j))`;
- open-path objective: distance from the current/ready joint state to the first
  vertex plus the sum of `C` over consecutive vertices.

The implementation sequence is strict: generate all poses, measure hand-eye
TF, attempt `ready`, snapshot the current joint state, run IK on every pose pair,
cache projection/distance/cost matrices, build the open nearest-neighbor path,
run 2-opt, write planning diagnostics, and only then execute motion. Never mix
IK filtering with the capture loop because doing so changes the optimization
graph and wastes physical motion.

Try all possible starting vertices for nearest-neighbor initialization and keep
the lowest-cost complete feasible path. During 2-opt, reverse `[i:j]`, reject a
candidate if any resulting neighbor edge violates overlap, and accept only a
strict objective reduction. Cache symmetric pair metrics and use vectorized
NumPy projection so the expensive work is `O(N^2 * samples)` once; with a small
scan (`N` around 18), deterministic full-path validation during 2-opt is cheap
and safer than special-casing reversal boundaries.

### 1. Pure planning module

Create `scripts/scan_trajectory.py` with:

- dataclasses for camera intrinsics, spiral viewpoints, reachable viewpoints,
  and path metrics;
- equal-area golden-angle spiral generation over configurable azimuth and
  elevation bounds;
- mirrored look-at camera poses for left/right assignments;
- calibrated projection of a deterministic scan-volume sample set;
- pairwise overlap, joint-distance, and combined edge-cost matrices;
- nearest-neighbor open-TSP initialization seeded by the current joint state;
- overlap-constrained 2-opt refinement;
- path metrics for total distance under the selected motion metric,
  mean/minimum neighbor overlap, and total objective cost.

### 2. MoveIt IK adapter

Create `scripts/moveit_ik.py` with a small context-managed client that:

- owns an isolated ROS context, node, executor, `/joint_states` subscription,
  and `/compute_ik` client;
- honors `use_sim_time` before creating subscriptions;
- waits for a recent joint state using a wall-clock timeout;
- submits both TCP poses in the world/planning frame in one `GetPositionIK`
  request;
- returns a name-to-position mapping on success and a diagnostic string on
  failure;
- shuts down its executor and context reliably.

### 3. Scan integration

Update `scripts/multi_view_scan.py` to:

- load all scan defaults from `config/multi_view_scan.yaml`, while allowing
  explicit command-line options to override individual YAML values;
- load camera intrinsics from `camera.yaml`;
- expose CLI settings for spiral view count/bounds, scan-volume radius,
  projection sample count, IK service/topic/timeouts, minimum overlap,
  overlap cost weight, and 2-opt passes;
- generate the spiral before robot initialization;
- move to `ready`, acquire the measured current joint state, and filter all
  candidates through dual-tip collision-aware IK;
- optimize the reachable candidates before entering the motion/capture loop;
- retain each candidate's original spiral ID while assigning sequential capture
  step numbers to the optimized path;
- publish the selected target camera/TCP TFs before each motion;
- publish the complete optimized IK sequence as a MoveIt `DisplayTrajectory`
  and both world-frame camera routes as latched RViz `Path` messages before the
  first scan motion;
- store planner settings, rejected IK candidates, optimized source order, IK
  solutions, and before/after path metrics in `manifest.json`.

### 4. Verification

Add tests that cover:

- deterministic spiral generation and world-Y arm assignment;
- projection overlap identity, symmetry, bounds, and degradation with viewpoint
  separation;
- nearest-neighbor path permutation and start-state selection;
- 2-opt improvement without violating overlap constraints;
- unreachable-candidate removal through an injected/fake IK callback or client;
- manifest-relative output paths and existing scan cleanup behavior.

Do not execute robot motion during automated verification. Run syntax checks,
unit tests for the pure planner, and CLI help/import checks. A live IK-only check
may be run only when MoveIt is available and must not call a motion service.

## Acceptance criteria

- `multi_view_scan.py --help` documents every new planning parameter.
- No hard-coded azimuth/elevation grid remains in `run_scan`.
- Every executed pose pair previously passed collision-aware multi-tip IK.
- The final order is a permutation of reachable spiral candidates with no
  duplicates.
- Every final adjacent edge meets the configured overlap threshold, or planning
  stops before motion with a clear error explaining that no feasible path was
  found.
- The manifest is sufficient to reproduce and audit candidate generation,
  rejection, ordering, and path-quality metrics.
- Motion exceptions still skip the capture and continue to the next optimized
  viewpoint.
- Existing reconstruction and notebook consumers continue resolving image paths
  from `manifest.json`.
