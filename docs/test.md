# Reachability-Filtered, Overlap-Aware Dual-Arm Multiview Scanning

## Abstract

This document describes the multiview trajectory-planning method implemented in
this repository for coordinated scene acquisition with two hand-eye RGB-D
cameras. Candidate views are distributed over an upper hemispherical band using
a deterministic golden-angle sequence and paired by reflection across the
world (x)-(z) plane. Each camera-pose pair is converted to a pair of robot
tool-center-point (TCP) targets and tested using collision-aware, multi-tip
inverse kinematics (IK). Unreachable candidate pairs are discarded before any
scan motion occurs. The remaining candidates form the vertices of an open
traveling-salesperson problem (TSP). Its edge cost combines coordinated
joint-space displacement with a penalty for low predicted image overlap. A hard
lower bound on the overlap of both camera streams is enforced during greedy
path construction and deterministic 2-opt refinement. The resulting method
reduces unnecessary robot motion while preserving temporal overlap for RGB-D
registration and scene reconstruction.

This text documents the implemented algorithm precisely. Statements about
global optimality, exact scene visibility, or guaranteed path existence are
intentionally avoided because the implementation uses geometric proxies and
deterministic heuristics.

## 1. Problem definition

Let a dual-arm robot carry one calibrated RGB-D camera on each end effector.
Both cameras observe a scene centered at

$$
\mathbf{c} = [c_x,c_y,c_z]^\mathsf{T} \in \mathbb{R}^3.
$$

A scan candidate is a synchronized pair

$$
v_i = \left({}^{W}\mathbf{T}_{C_L,i},
             {}^{W}\mathbf{T}_{C_R,i}\right),
$$

where ({}^{W}\mathbf{T}_{C_L,i}) and
({}^{W}\mathbf{T}_{C_R,i}) are the left and right optical-frame poses in the
world frame. A pair is treated as one indivisible planning vertex because both
arms are commanded at the same time.

For the set of reachable vertices

$$
\mathcal{V}=\{v_1,\ldots,v_n\},
$$

the planner seeks an open permutation

$$
\boldsymbol{\pi}=(\pi_1,\pi_2,\ldots,\pi_n)
$$

that has low coordinated joint travel and high overlap between every two
consecutive RGB-D frames. The path is open: returning from (v_{\pi_n}) to
(v_{\pi_1}) is not part of the optimization objective. The implementation may
send the robot to its SRDF `ready` state after acquisition as a separate
operational step; that homing motion is not included in the reported scan-path
cost.

## 2. Notation and coordinate conventions

| Symbol | Definition |
| --- | --- |
| (W) | World coordinate frame. |
| (C_L,C_R) | Left and right camera optical frames. |
| (E_L,E_R) | Left and right TCP frames. |
| ({}^{A}\mathbf{T}_{B}) | Homogeneous pose of frame (B) expressed in frame (A). |
| (mathbf{R}_{WC}) | Camera-to-world rotation, stored in the upper-left block of ({}^{W}\mathbf{T}_C). |
| (mathbf{t}_{WC}) | Camera origin expressed in world coordinates. |
| (N) | Number of generated paired candidates before IK filtering. |
| (n\leq N) | Number of paired candidates retained after IK filtering. |
| (M) | Number of deterministic surface samples used to estimate overlap. |
| (	au) | Minimum permitted neighbor overlap. |
| (lambda) | Weight applied to the low-overlap penalty. |

The camera convention is the ROS optical convention: (+x) points right,
(+y) points down, and (+z) points forward. Homogeneous transforms have the
form

$$
{}^{W}\mathbf{T}_{C} =
\begin{bmatrix}
\mathbf{R}_{WC} & \mathbf{t}_{WC}\\
\mathbf{0}^\mathsf{T} & 1
\end{bmatrix}.
$$

The arm assignment is fixed in the world frame:

$$
y>0 \Rightarrow \text{left arm}, \qquad
y<0 \Rightarrow \text{right arm}.
$$

The generator validates this condition for every pose pair rather than
silently changing the arm assignment.

## 3. Deterministic paired spiral sampling

### 3.1 Equal-area elevation sampling

Let the permitted azimuth and elevation intervals be

$$
\alpha\in[\alpha_{\min},\alpha_{\max}], \qquad
\beta\in[\beta_{\min},\beta_{\max}].
$$

Angles are converted to radians before evaluating the equations. For candidate
index (i\in\{0,\ldots,N-1\}), define

$$
t_i=\frac{i+\tfrac{1}{2}}{N}.
$$

Elevation is sampled uniformly in (sin\beta):

$$
s_i = \sin\beta_{\min}
      + t_i\left(\sin\beta_{\max}-\sin\beta_{\min}\right),
$$

$$
\beta_i=\arcsin(s_i).
$$

The area element on a sphere under the elevation convention used here is
proportional to (d(\sin\beta)). Consequently, uniform spacing in
(sin\beta) avoids the concentration near the top of the hemisphere produced
by uniform elevation angles.

### 3.2 Golden-angle azimuth sampling

The conjugate golden-ratio fraction is

$$
g=\frac{\sqrt{5}-1}{2}.
$$

The normalized azimuth coordinate and physical azimuth are

$$
u_i=\left(\frac{1}{2}+ig\right)\bmod 1,
$$

$$
\alpha_i=\alpha_{\min}
         +u_i(\alpha_{\max}-\alpha_{\min}).
$$

This low-discrepancy ordering is deterministic and avoids aligned latitude and
longitude rows. It supplies candidate coverage only; the sequence is reordered
after IK filtering.

### 3.3 Mirrored camera positions

Given camera radius (r_c>0), the horizontal radius is

$$
\rho_i=r_c\cos\beta_i.
$$

The common (x) and (z) coordinates and the positive lateral offset are

$$
x_i=c_x+\rho_i\cos\alpha_i,
$$

$$
\Delta y_i=\rho_i\sin\alpha_i,
$$

$$
z_i=c_z+r_c\sin\beta_i.
$$

The synchronized left and right camera positions are

$$
\mathbf{p}_{L,i}=
\begin{bmatrix}x_i\\c_y+\Delta y_i\\z_i\end{bmatrix},
\qquad
\mathbf{p}_{R,i}=
\begin{bmatrix}x_i\\c_y-\Delta y_i\\z_i\end{bmatrix}.
$$

For the default center (c_y=0), these positions are exact reflections across
the world (x)-(z) plane.

### 3.4 Look-at camera orientation

For camera position (mathbf{p}), the optical forward direction is

$$
\mathbf{z}_C=
\frac{\mathbf{c}-\mathbf{p}}
     {\|\mathbf{c}-\mathbf{p}\|_2}.
$$

With nominal world-up vector

$$
\mathbf{u}_W=[0,0,1]^\mathsf{T},
$$

the remaining axes are

$$
\mathbf{x}_C=
\frac{\mathbf{z}_C\times\mathbf{u}_W}
     {\|\mathbf{z}_C\times\mathbf{u}_W\|_2},
\qquad
\mathbf{y}_C=\mathbf{z}_C\times\mathbf{x}_C.
$$

If (mathbf{z}_C) is parallel to the nominal up vector, the implementation
uses ([1,0,0]^\mathsf{T}) as the fallback reference. The camera rotation is

$$
\mathbf{R}_{WC}=
\begin{bmatrix}
\mathbf{x}_C & \mathbf{y}_C & \mathbf{z}_C
\end{bmatrix}.
$$

Fixed local optical-axis rotations preserve the camera position and look
direction:

$$
{}^{W}\mathbf{T}'_{C,a,i}
= {}^{W}\mathbf{T}_{C,a,i}\mathbf{R}_z(\delta_a),
\qquad a\in\{L,R\},
$$

with the current defaults (delta_L=-90^\circ) and
(delta_R=+90^\circ).

## 4. Camera-to-TCP conversion and reachability filtering

The rigid hand-eye transforms are read from TF at startup. Let
({}^{C_a}\mathbf{T}_{E_a}) denote the measured camera-to-TCP transform for arm
(a). Each desired camera pose becomes a TCP target through

$$
{}^{W}\mathbf{T}_{E_a,i}
= {}^{W}\mathbf{T}'_{C_a,i}\,{}^{C_a}\mathbf{T}_{E_a}.
$$

The robot first attempts to move to the named SRDF configuration
`dual_arm/ready`. Failure of this preliminary move is recorded but does not
prevent reachability analysis; the latest measured joint state is used in that
case.

For every paired candidate, one MoveIt `GetPositionIK` request contains both
TCP targets:

$$
\mathcal{P}_i=
\left\{{}^{W}\mathbf{T}_{E_L,i},
       {}^{W}\mathbf{T}_{E_R,i}\right\}.
$$

The request uses:

- planning group `dual_arm` by default;
- both TCP link names in `ik_link_names`;
- both target poses in `pose_stamped_vector`;
- the current complete `/joint_states` positions as the IK seed;
- the same simulated or wall-clock time domain as MoveIt;
- `avoid_collisions = true`.

Candidate (v_i) is retained only if the synchronized multi-tip request returns
MoveIt's success code. A failure for either arm therefore rejects the entire
paired vertex. This filtering step does not invoke trajectory planning or robot
motion.

For each retained vertex, the returned positions of all common non-finger
motion joints are placed in a stable name-sorted vector

$$
\mathbf{q}_i=
[q_{i,1},q_{i,2},\ldots,q_{i,J}]^\mathsf{T}.
$$

The same ordered joint list is used for every vertex and for the initial
measured configuration (mathbf{q}_0).

## 5. Projected temporal-overlap model

### 5.1 Spherical surface proxy

Overlap is estimated before image acquisition, so the implementation uses a
spherical proxy centered on the region of interest. Let (r_s) be the
configurable proxy radius and (M) the sample count. For
(k\in\{0,\ldots,M-1\}), define the spherical Fibonacci samples

$$
z_k=1-2\frac{k+\tfrac{1}{2}}{M},
$$

$$
\rho_k=\sqrt{\max(0,1-z_k^2)},
$$

$$
\theta_k=k\gamma, \qquad
\gamma=\pi(3-\sqrt{5}),
$$

$$
\mathbf{d}_k=
\begin{bmatrix}
\rho_k\cos\theta_k\\
\rho_k\sin\theta_k\\
z_k
\end{bmatrix},
\qquad
\mathbf{s}_k=\mathbf{c}+r_s\mathbf{d}_k.
$$

The implementation estimates the proxy center from the finite sample set,

$$
\bar{\mathbf{s}}=\frac{1}{M}\sum_{k=0}^{M-1}\mathbf{s}_k,
$$

and assigns each point an outward normal

$$
\mathbf{n}_k=
\frac{\mathbf{s}_k-\bar{\mathbf{s}}}
     {\|\mathbf{s}_k-\bar{\mathbf{s}}\|_2}.
$$

Using a front-facing surface proxy is important. A frustum-only test would
assign 100% overlap to widely separated cameras whenever the entire bounding
sphere happened to fit inside both images, even though they observe different
sides of the scene.

### 5.2 Pinhole projection

For camera pose ({}^{W}\mathbf{T}_{C}=(\mathbf{R}_{WC},
\mathbf{t}_{WC})), a proxy sample is transformed from world to camera
coordinates by

$$
\mathbf{s}^{C}_k
=\mathbf{R}_{WC}^{\mathsf{T}}
 (\mathbf{s}_k-\mathbf{t}_{WC})
= [x_k^C,y_k^C,z_k^C]^\mathsf{T}.
$$

With calibrated pinhole intrinsics

$$
\mathbf{K}=
\begin{bmatrix}
f_x & 0 & c_x^{\mathrm{img}}\\
0 & f_y & c_y^{\mathrm{img}}\\
0 & 0 & 1
\end{bmatrix},
$$

the projected pixel is

$$
u_k=f_x\frac{x_k^C}{z_k^C}+c_x^{\mathrm{img}},
\qquad
v_k=f_y\frac{y_k^C}{z_k^C}+c_y^{\mathrm{img}}.
$$

A sample is considered visible from camera (C) exactly when

$$
z_k^C>\epsilon_z,
$$

$$
0\leq u_k<W, \qquad 0\leq v_k<H,
$$

and

$$
\mathbf{n}_k^\mathsf{T}
(\mathbf{t}_{WC}-\mathbf{s}_k)>0.
$$

The last inequality is the front-facing test. The implementation uses
(epsilon_z=10^{-6}\) m. Let

$$
\mathcal{S}_{a,i}
=\{k\mid\mathbf{s}_k\text{ is visible from camera }a
\text{ at vertex }i\}.
$$

### 5.3 Per-arm and paired overlap

Temporal overlap for arm (a\in\{L,R\}) between vertices (i) and (j) is
the Jaccard index

$$
O^a_{ij}
=\frac{|\mathcal{S}_{a,i}\cap\mathcal{S}_{a,j}|}
       {|\mathcal{S}_{a,i}\cup\mathcal{S}_{a,j}|}.
$$

If the union is empty, the overlap is defined as zero. A dual-arm edge uses the
weaker camera overlap:

$$
O_{ij}=\min\left(O^L_{ij},O^R_{ij}\right).
$$

Thus, a high overlap from one hand-eye camera cannot compensate for a poor
transition on the other camera. The matrix is symmetric and
(O_{ii}=1).

## 6. Open-TSP formulation

### 6.1 Joint-motion distance

The motion proxy between two reachable paired vertices is

$$
D_{ij}=\|\mathbf{q}_i-\mathbf{q}_j\|_2.
$$

The cost of reaching the first vertex from the current or ready configuration
is

$$
D_{0i}=\|\mathbf{q}_0-\mathbf{q}_i\|_2.
$$

The current implementation uses an unweighted Euclidean norm over the retained
joints. Revolute-joint values are expressed in radians, while any prismatic
rail joint is expressed in meters. This convention must be reported when using
the metric in experiments because it is a numerical trajectory proxy rather
than mechanical energy, Cartesian path length, or execution time.

### 6.2 Overlap-aware edge objective

The soft edge cost is

$$
C_{ij}=D_{ij}+\lambda(1-O_{ij}), \qquad \lambda\geq0.
$$

Greater overlap lowers the edge cost. In addition, every temporal edge must
satisfy the hard constraint

$$
O_{ij}\geq\tau, \qquad 0\leq\tau\leq1.
$$

For a permutation (oldsymbol{\pi}), the optimized open-path objective is

$$
J(\boldsymbol{\pi})
=D_{0,\pi_1}
+\sum_{k=1}^{n-1}C_{\pi_k,\pi_{k+1}},
$$

subject to

$$
O_{\pi_k,\pi_{k+1}}\geq\tau,
\qquad k=1,\ldots,n-1.
$$

The start-to-first term contains only joint distance because there is no image
at the ready configuration for which temporal overlap could be defined.

## 7. Deterministic path construction

### 7.1 Multi-start constrained nearest neighbor

A greedy open path is constructed once for every possible starting vertex. At
current vertex (i), the feasible unvisited neighbor set is

$$
\mathcal{F}_i=
\{j\in\mathcal{U}\mid O_{ij}\geq\tau\},
$$

where (mathcal{U}) is the set of unvisited vertices. The next vertex is

$$
j^*=\arg\min_{j\in\mathcal{F}_i} C_{ij}.
$$

Vertex index is used as a deterministic tie breaker. A start is discarded if
the greedy procedure reaches a vertex with no feasible unvisited neighbor. Of
the complete greedy paths, the one with minimum (J) initializes 2-opt.

```text
best_path <- none
for start in vertices ordered by (D_0,start, vertex_id):
    path <- [start]
    unvisited <- all vertices except start
    while unvisited is not empty:
        feasible <- unvisited neighbors satisfying O >= tau
        if feasible is empty:
            discard this start
        append the feasible neighbor with minimum (C, vertex_id)
    retain path if its complete open-path objective is the lowest so far
if no complete path was formed:
    stop before robot scan motion
```

Trying every start reduces sensitivity to an unfortunate initial greedy choice,
but does not turn the procedure into an exact TSP solver.

### 7.2 Overlap-constrained 2-opt

Let the current path be

$$
\boldsymbol{\pi}=(\pi_1,\ldots,\pi_n).
$$

For all index pairs (1\leq a<b\leq n), the algorithm reverses the inclusive
subsequence:

$$
\boldsymbol{\pi}'=
(\pi_1,\ldots,\pi_{a-1},
 \pi_b,\pi_{b-1},\ldots,\pi_a,
 \pi_{b+1},\ldots,\pi_n).
$$

Reversals beginning at the first vertex and ending at the last vertex are
allowed. Therefore, every candidate is evaluated with the start-state cost
(D_{0,\pi'_1}). A candidate reversal is admissible only if

$$
O_{\pi'_k,\pi'_{k+1}}\geq\tau
\quad\forall k\in\{1,\ldots,n-1\}.
$$

During each pass, the implementation evaluates all reversals and retains the
lowest-cost strict improvement. The pass terminates without modification if no
candidate improves the objective by more than the numerical tolerance
(10^{-12}). Refinement stops at this local optimum or after a configurable
maximum number of passes.

```text
path <- best complete nearest-neighbor path
repeat for at most P passes:
    best_candidate <- path
    for every inclusive subsequence [a,b], a < b:
        candidate <- path with [a,b] reversed
        reject candidate if any neighbor overlap is below tau
        retain candidate if it strictly lowers the full open-path objective
    if no improving candidate exists:
        stop
    path <- best_candidate
return path
```

Because accepted operations are permutations, no reachable vertex is added,
removed, or duplicated during 2-opt.

## 8. End-to-end procedure

The complete implemented sequence is summarized below.

```text
Input:
    scene center, camera radius, angular bounds, candidate count
    calibrated camera intrinsics and hand-eye transforms
    overlap proxy radius, sample count, threshold, and weight
    MoveIt group, TCP frames, IK timeout, and 2-opt pass limit

1. Generate N deterministic left/right spiral camera-pose pairs.
2. Apply the fixed local optical-axis rotations.
3. Initialize both synchronized RGB-D clients and the read-only IK client.
4. Verify the time source, camera streams, and required TF transforms.
5. Attempt to move the dual-arm group to the SRDF ready configuration.
6. Read the current full joint state.
7. Convert each camera pair to TCP targets and call collision-aware multi-tip IK.
8. Reject every pair whose IK request fails; retain one joint vector per success.
9. Generate M deterministic samples on the spherical scene proxy.
10. Compute all pairwise overlap, joint-distance, and combined-cost matrices.
11. Construct a complete overlap-feasible multi-start nearest-neighbor path.
12. Refine the open path with overlap-constrained best-improvement 2-opt.
13. Write candidate rejection and path diagnostics to manifest.json.
14. For each optimized vertex:
      a. publish camera and TCP target TFs for RViz;
      b. command both arms simultaneously with move_l_dual;
      c. if motion raises RobotAPIError, record failure and continue;
      d. otherwise capture synchronized left/right RGB-D data and actual TF poses.
15. Report capture statistics and attempt to return to ready.
```

No camera-view motion is initiated until reachability filtering and trajectory
optimization have both completed.

## 9. Reported path metrics

For the optimized permutation, the implementation reports coordinated joint
travel

$$
L_q=D_{0,\pi_1}
+\sum_{k=1}^{n-1}D_{\pi_k,\pi_{k+1}},
$$

mean neighbor overlap

$$
\bar{O}=\frac{1}{n-1}
\sum_{k=1}^{n-1}O_{\pi_k,\pi_{k+1}},
$$

minimum neighbor overlap

$$
O_{\min}=\min_{k\in\{1,\ldots,n-1\}}
O_{\pi_k,\pi_{k+1}},
$$

and the full objective (J(\boldsymbol{\pi})). For a single-vertex path, mean
and minimum overlap are defined as 1 because the path has no temporal edges.

Both the initial greedy metrics and final 2-opt metrics are retained, allowing
the effect of local refinement to be reported directly.

## 10. Computational complexity

Let (J) be the number of retained motion joints and (P) the maximum number
of 2-opt passes.

| Stage | Time complexity | Dominant storage |
| --- | --- | --- |
| Spiral generation | (O(N)) | (O(N)) poses |
| Multi-tip IK filtering | (N) service calls | (O(nJ)) solutions |
| Pairwise joint distance | (O(n^2J)) | (O(n^2)) matrix |
| Pairwise projected overlap | (O(n^2M)) | (O(n^2+M)) |
| Multi-start nearest neighbor | (O(n^3)) | (O(n)) working path |
| Best-improvement 2-opt | (O(Pn^3)) with full-path validation | (O(n)) working path |

For the default (N=18), projection and heuristic search are small compared
with ROS communication and numerical IK. The implementation precomputes the
symmetric overlap, distance, and cost matrices once before path search.

## 11. Default parameters

| Parameter | Default | Role |
| --- | ---: | --- |
| Scan center | ([0.40,0.0,0.0]) m | Center of camera orbit and overlap proxy. |
| Camera radius (r_c) | (0.40) m | Camera distance from the scan center. |
| Candidate pairs (N) | 18 | Spiral samples before IK filtering. |
| Azimuth bounds | (10^\circ,135^\circ) | Left-side angular interval before mirroring. |
| Elevation bounds | (45^\circ,75^\circ) | Upper hemispherical band. |
| Surface radius (r_s) | (0.15) m | Spherical overlap proxy. |
| Surface samples (M) | 2048 | Resolution of discrete visibility overlap. |
| Minimum overlap (	au) | 0.35 | Hard feasibility threshold. |
| Overlap weight (lambda) | 0.5 | Soft preference for higher overlap. |
| 2-opt passes (P) | 30 | Refinement limit. |
| IK timeout | 0.25 s | Solver timeout per paired candidate. |
| Joint-state timeout | 2.0 s | Wall-clock state/service wait timeout. |
| Left/right local roll | (-90^\circ/+90^\circ) | Optical-frame orientation adjustment. |

Camera intrinsics default to the calibration file specified by
`--camera-yaml`. Parameter values, retained IK solutions, rejected candidates,
initial and optimized orders, per-edge optimized overlaps, and path metrics are
written to `scan_output/manifest.json`.

## 12. Failure handling

- If the preliminary `ready` motion fails, the error is recorded and planning
  continues from the measured current configuration.
- If the IK service or a recent joint state is unavailable, planning stops.
- If every candidate fails IK, planning stops.
- If the constrained greedy procedure cannot form a complete path, planning
  stops before scan motion and reports the requested overlap threshold.
- If an individual `move_l_dual` call raises `RobotAPIError`, that capture is
  marked `motion_failed` and the next optimized vertex is attempted.
- Successful captures retain both desired and measured camera poses so that
  reconstruction uses execution-time TF rather than the ideal target pose.

## 13. Reproducibility and publication reporting

For a reproducible experiment, report at least:

1. repository revision and ROS/MoveIt configuration;
2. robot model, planning group, kinematics plugin, and collision scene;
3. camera intrinsics, hand-eye calibration, and image resolution;
4. scan center, camera radius, angular bounds, and generated candidate count;
5. overlap proxy radius, sample count, (	au), and (lambda);
6. IK timeout, successful and rejected candidate counts, and rejection reasons;
7. initial greedy and optimized source-index orders;
8. (L_q), (J), (\bar O), and (O_{\min}) before and after 2-opt;
9. number of motion failures and successful RGB-D pairs;
10. whether time came from simulation `/clock` or the system clock.

The manifest records most run-dependent quantities. External configuration such
as the robot model, kinematics plugin, planning scene, and repository revision
should be archived alongside it.

For evaluation, suitable comparisons include the original spiral order, a
fixed azimuth/elevation grid, joint-distance-only TSP
((\lambda=0,\tau=0)), the greedy path without 2-opt, and the complete proposed
pipeline. Report both planning metrics and downstream reconstruction metrics;
low proxy cost alone does not establish improved reconstruction quality.

## 14. Limitations and claims boundary

The following limitations are important when presenting results:

1. **Approximate visibility.** The spherical proxy is not the reconstructed
   scene geometry. It models frustum inclusion and front-facing visibility but
   does not perform a depth-buffer test, account for inter-object occlusion, or
   predict sensor noise.
2. **One IK solution per vertex.** The planner retains one multi-tip IK result
   for each candidate. It does not jointly optimize over multiple IK branches.
3. **Heuristic path optimization.** Multi-start nearest neighbor and 2-opt do
   not guarantee the globally optimal TSP path. The greedy initializer can fail
   even if a different feasible Hamiltonian path exists.
4. **Joint-distance proxy.** The unweighted mixed-unit Euclidean norm is not a
   direct estimate of time, energy, or swept-volume collision risk.
5. **Execution may differ from scoring.** `move_l_dual` receives Cartesian TCP
   targets and may select an IK realization that differs from the solution used
   to construct the cost matrix.
6. **Static-scene assumption.** Pairwise overlap is precomputed once and assumes
   that the scene, camera calibration, and robot/world transforms do not change
   during acquisition.
7. **Shared intrinsic model.** The present overlap matrix uses one intrinsic
   calibration for both hand-eye cameras. Different physical calibrations
   require a per-camera extension.

These limitations should be stated explicitly in a publication. Appropriate
future extensions include joint-range normalization, per-joint motion weights,
multiple-IK-solution graph search, exact or beam-search initialization,
scene-derived visibility, per-camera intrinsics, and online path replanning.

## 15. Implementation correspondence

The publication method maps to the repository as follows:

- spiral generation, projection, overlap, and optimization:
  [`scripts/multi_view_scan/scan_trajectory.py`](../scripts/multi_view_scan/scan_trajectory.py);
- pose and look-at geometry:
  [`scripts/multi_view_scan/aux_math.py`](../scripts/multi_view_scan/aux_math.py);
- collision-aware MoveIt request construction:
  [`scripts/multi_view_scan/moveit_ik.py`](../scripts/multi_view_scan/moveit_ik.py);
- scan orchestration, TF visualization, capture, and manifest output:
  [`scripts/multi_view_scan/multi_view_scan.py`](../scripts/multi_view_scan/multi_view_scan.py);
- behavioral verification:
  [`tests/test_scan_trajectory.py`](../tests/test_scan_trajectory.py);
- concise implementation contract:
  [`TASK.md`](../TASK.md).
