# Reachability-Filtered, Overlap-Aware Dual-Arm Multiview Scanning

## Abstract

This document describes the multiview trajectory planner implemented in this
repository for coordinated scene acquisition with two hand-eye RGB-D cameras.
Candidate views are distributed over an upper hemispherical band using a
deterministic golden-angle sequence and paired by reflection across the world
$x$-$z$ plane. Every camera-pose pair is converted to two robot
tool-center-point (TCP) targets and tested using collision-aware multi-tip
inverse kinematics (IK). Unreachable pairs are discarded before scan motion.
The reachable candidates form an open traveling-salesperson problem (TSP)
whose cost combines coordinated joint displacement and predicted image
overlap. A hard overlap constraint is enforced during multi-start
nearest-neighbor construction and deterministic 2-opt refinement. The method
therefore reduces unnecessary arm motion while preserving temporal overlap
needed for RGB-D registration.

This document describes the implemented method rather than an idealized
variant. It does not claim global TSP optimality or exact scene visibility.

## 1. Problem formulation

Let $W$ be the world frame, $C_L$ and $C_R$ the left and right camera optical
frames, and $E_L$ and $E_R$ their associated TCP frames. Both cameras observe a
scene centered at

$$
\mathbf{c}=[c_x,c_y,c_z]^\mathsf{T}\in\mathbb{R}^3.
$$

A scan candidate is the synchronized pair

$$
v_i=
\left({}^{W}\mathbf{T}_{C_L,i},
      {}^{W}\mathbf{T}_{C_R,i}\right).
$$

The two camera poses are one indivisible planning vertex because the arms move
simultaneously. After IK filtering, let

$$
\mathcal{V}=\{v_1,\ldots,v_n\}
$$

be the reachable vertices. The planner seeks an open permutation

$$
\boldsymbol{\pi}=(\pi_1,\ldots,\pi_n)
$$

that minimizes coordinated joint movement while retaining sufficient visual
overlap between each pair of consecutive observations. The path is open:
returning from the final scan view to the first view is not included. A final
operational move to the SRDF **ready** state is also excluded from the reported
scan-path objective.

The homogeneous camera pose is

$$
{}^{W}\mathbf{T}_{C}=
\begin{bmatrix}
\mathbf{R}_{WC} & \mathbf{t}_{WC}\\
\mathbf{0}^\mathsf{T} & 1
\end{bmatrix},
$$

where the columns of $\mathbf{R}_{WC}$ are camera axes expressed in the world
frame. Camera optical axes follow the ROS convention: $+x$ points right, $+y$
points down, and $+z$ points forward.

## 2. Deterministic paired viewpoint generation

### 2.1 Equal-area elevation and golden-angle azimuth

Let the camera-orbit radius be $r_c>0$, the number of generated pairs be $N$,
and the allowed angular intervals be

$$
\alpha\in[\alpha_{\min},\alpha_{\max}],\qquad
\beta\in[\beta_{\min},\beta_{\max}].
$$

For candidate $i\in\{0,\ldots,N-1\}$, define

$$
t_i=\frac{i+\tfrac{1}{2}}{N}.
$$

Elevation is sampled uniformly in $\sin\beta$:

$$
s_i=\sin\beta_{\min}
+t_i\left(\sin\beta_{\max}-\sin\beta_{\min}\right),
$$

$$
\beta_i=\arcsin(s_i).
$$

Because the spherical area element under this elevation convention is
proportional to $d(\sin\beta)$, this construction avoids concentrating samples
near the top of the hemisphere.

Let

$$
g=\frac{\sqrt{5}-1}{2}
$$

be the conjugate golden-ratio fraction. The normalized and physical azimuths
are

$$
u_i=\left(\frac{1}{2}+ig\right)\bmod 1,
$$

$$
\alpha_i=\alpha_{\min}
+u_i\left(\alpha_{\max}-\alpha_{\min}\right).
$$

This deterministic low-discrepancy sequence supplies candidate coverage. Its
original order is not used as the final motion order.

### 2.2 Mirrored positions

The horizontal orbit radius is

$$
\rho_i=r_c\cos\beta_i.
$$

The common longitudinal and vertical coordinates and positive lateral offset
are

$$
x_i=c_x+\rho_i\cos\alpha_i,
$$

$$
\Delta y_i=\rho_i\sin\alpha_i,
$$

$$
z_i=c_z+r_c\sin\beta_i.
$$

The paired positions are

$$
\mathbf{p}_{L,i}=
\begin{bmatrix}
x_i\\c_y+\Delta y_i\\z_i
\end{bmatrix},\qquad
\mathbf{p}_{R,i}=
\begin{bmatrix}
x_i\\c_y-\Delta y_i\\z_i
\end{bmatrix}.
$$

For the default $c_y=0$, the positions are exact reflections across the world
$x$-$z$ plane. Every pair is validated against the fixed arm convention

$$
y_L>0,\qquad y_R<0.
$$

Invalid geometry raises an error instead of silently exchanging arms.

### 2.3 Look-at orientation

For a camera position $\mathbf{p}$, its forward optical direction is

$$
\mathbf{z}_C=
\frac{\mathbf{c}-\mathbf{p}}
{\|\mathbf{c}-\mathbf{p}\|_2}.
$$

With nominal world-up vector $\mathbf{u}_W=[0,0,1]^\mathsf{T}$, the other axes
are

$$
\mathbf{x}_C=
\frac{\mathbf{z}_C\times\mathbf{u}_W}
{\|\mathbf{z}_C\times\mathbf{u}_W\|_2},
$$

$$
\mathbf{y}_C=\mathbf{z}_C\times\mathbf{x}_C.
$$

If the forward axis is parallel to nominal world up, the implementation uses
$[1,0,0]^\mathsf{T}$ as a fallback reference. The camera rotation is

$$
\mathbf{R}_{WC}=
\begin{bmatrix}
\mathbf{x}_C & \mathbf{y}_C & \mathbf{z}_C
\end{bmatrix}.
$$

Fixed local rotations about optical $z$ preserve the position and look
direction:

$$
{}^{W}\mathbf{T}'_{C,a,i}
={}^{W}\mathbf{T}_{C,a,i}\mathbf{R}_z(\delta_a),
\qquad a\in\{L,R\}.
$$

The implementation uses $\delta_L=-90^\circ$ and
$\delta_R=+90^\circ$.

## 3. TCP conversion and collision-aware IK

The rigid camera-to-TCP transforms are measured from TF at startup. For arm
$a\in\{L,R\}$, the TCP target corresponding to a camera target is

$$
{}^{W}\mathbf{T}_{E_a,i}
={}^{W}\mathbf{T}'_{C_a,i}\,{}^{C_a}\mathbf{T}_{E_a}.
$$

Before filtering, the robot attempts to move to the named dual-arm **ready**
configuration. If this attempt fails, the failure is recorded and filtering
uses the latest measured configuration.

One MoveIt position-IK request is issued for every pair:

$$
\mathcal{P}_i=
\left\{{}^{W}\mathbf{T}_{E_L,i},
       {}^{W}\mathbf{T}_{E_R,i}\right\}.
$$

The request contains both TCP link names and target poses, uses the
**dual_arm** planning group by default, is seeded with all current joint
positions, and enables collision checking. Candidate $v_i$ is retained only
when the synchronized multi-tip problem returns MoveIt's success code. Thus,
failure of either target rejects the complete pair. This stage calls only the
IK service; it does not plan or execute a robot trajectory.

For each retained vertex, common non-finger joint positions are stored in a
stable name-sorted vector

$$
\mathbf{q}_i=[q_{i,1},\ldots,q_{i,J}]^\mathsf{T}.
$$

The initial measured configuration $\mathbf{q}_0$ uses the same joint ordering.
Candidate identifiers from the original spiral are preserved for diagnostics
even after filtering and reordering.

## 4. Projected temporal-overlap model

### 4.1 Spherical surface proxy

Overlap must be estimated before images are acquired. The planner therefore
uses a spherical scene proxy of radius $r_s$ centered on the region of interest.
For $M$ samples and $k\in\{0,\ldots,M-1\}$, spherical Fibonacci sampling is

$$
z_k=1-2\frac{k+\tfrac{1}{2}}{M},
$$

$$
\rho_k=\sqrt{\max(0,1-z_k^2)},
$$

$$
\theta_k=k\gamma,\qquad
\gamma=\pi(3-\sqrt{5}),
$$

$$
\mathbf{d}_k=
\begin{bmatrix}
\rho_k\cos\theta_k\\
\rho_k\sin\theta_k\\
z_k
\end{bmatrix},\qquad
\mathbf{s}_k=\mathbf{c}+r_s\mathbf{d}_k.
$$

The finite-set center and outward normal are

$$
\bar{\mathbf{s}}=\frac{1}{M}\sum_{k=0}^{M-1}\mathbf{s}_k,
$$

$$
\mathbf{n}_k=
\frac{\mathbf{s}_k-\bar{\mathbf{s}}}
{\|\mathbf{s}_k-\bar{\mathbf{s}}\|_2}.
$$

Front-facing surface samples are used because a frustum-only test would report
complete overlap whenever the whole proxy fit in both images, even for cameras
observing different sides of the scene.

### 4.2 Projection and visibility

For camera pose $(\mathbf{R}_{WC},\mathbf{t}_{WC})$, a sample in camera
coordinates is

$$
\mathbf{s}_k^C=
\mathbf{R}_{WC}^{\mathsf{T}}
(\mathbf{s}_k-\mathbf{t}_{WC})
=[x_k^C,y_k^C,z_k^C]^\mathsf{T}.
$$

With calibrated pinhole intrinsics

$$
\mathbf{K}=
\begin{bmatrix}
f_x&0&c_x^{\mathrm{img}}\\
0&f_y&c_y^{\mathrm{img}}\\
0&0&1
\end{bmatrix},
$$

the projected pixel is

$$
u_k=f_x\frac{x_k^C}{z_k^C}+c_x^{\mathrm{img}},\qquad
v_k=f_y\frac{y_k^C}{z_k^C}+c_y^{\mathrm{img}}.
$$

A sample is visible when all of the following conditions hold:

$$
z_k^C>\epsilon_z,
$$

$$
0\leq u_k<W,\qquad 0\leq v_k<H,
$$

$$
\mathbf{n}_k^\mathsf{T}
(\mathbf{t}_{WC}-\mathbf{s}_k)>0.
$$

The first two conditions enforce positive depth and image bounds; the third is
the front-facing condition. The implementation uses
$\epsilon_z=10^{-6}$ m.

Let $\mathcal{S}_{a,i}$ be the sample indices visible to camera arm $a$ at
vertex $i$. The temporal overlap for each arm is the Jaccard index

$$
O^a_{ij}=
\frac{|\mathcal{S}_{a,i}\cap\mathcal{S}_{a,j}|}
{|\mathcal{S}_{a,i}\cup\mathcal{S}_{a,j}|}.
$$

An empty union produces zero overlap. The paired edge uses the weaker result:

$$
O_{ij}=\min(O^L_{ij},O^R_{ij}).
$$

Consequently, high overlap from one camera cannot compensate for a poor
transition from the other camera. The overlap matrix is symmetric and has
$O_{ii}=1$.

## 5. Open-TSP objective

### 5.1 Selectable motion distance

Joint-distance mode uses the Euclidean distance between paired IK solutions:

$$
D^q_{ij}=\|\mathbf{q}_i-\mathbf{q}_j\|_2,
$$

with start distance

$$
D^q_{0i}=\|\mathbf{q}_0-\mathbf{q}_i\|_2.
$$

Pose-distance mode operates on the commanded left and right TCP poses. For arm
$a\in\{L,R\}$, translation distance is

$$
d^a_{t,ij}=
\|\mathbf{t}_{E_a,i}-\mathbf{t}_{E_a,j}\|_2.
$$

The shortest geodesic angle between TCP rotations is

$$
\theta^a_{ij}=
\arccos\left[
\operatorname{clip}\left(
\frac{
\operatorname{tr}\left(
(\mathbf{R}_{E_a,i})^\mathsf{T}\mathbf{R}_{E_a,j}
\right)-1}{2},
-1,1\right)\right].
$$

The coordinated translation and rotation distances are

$$
d_{t,ij}=
\sqrt{(d^L_{t,ij})^2+(d^R_{t,ij})^2},
$$

$$
d_{R,ij}=
\sqrt{(\theta^L_{ij})^2+(\theta^R_{ij})^2}.
$$

Pose distance is then

$$
D^p_{ij}=w_t d_{t,ij}+w_R d_{R,ij},
$$

where $w_t\geq0$, $w_R\geq0$, and the weights cannot both be zero. The start
distance $D^p_{0i}$ uses the measured current left/right TCP poses after the
attempted move to **ready**.

The selected motion matrix is

$$
D_{ij}=
\begin{cases}
D^q_{ij}, & \text{joint mode},\\
D^p_{ij}, & \text{pose mode}.
\end{cases}
$$

The same choice is used for $D_{0i}$. Pose mode is the default in the supplied
configuration. With $w_t=1$ and $w_R$ expressed in meters per radian, pose
distance has a meter-equivalent scale.

### 5.2 Overlap-aware edge cost

Selected motion distance and overlap are combined into

$$
C_{ij}=D_{ij}+\lambda(1-O_{ij}),\qquad\lambda\geq 0.
$$

For path $\boldsymbol{\pi}$, the open-path objective is

$$
J(\boldsymbol{\pi})=
D_{0,\pi_1}
+\sum_{k=1}^{n-1}C_{\pi_k,\pi_{k+1}},
$$

subject to the hard edge constraint

$$
O_{\pi_k,\pi_{k+1}}\geq\tau,
\qquad k=1,\ldots,n-1.
$$

The start term contains only joint distance because no image is acquired at
$\mathbf{q}_0$. Greater overlap reduces the soft cost, while $\tau$ provides an
independent minimum acceptable overlap.

In joint mode, the implementation uses an unweighted Euclidean norm. Revolute
joint values are in radians, while a prismatic rail is in meters. In pose mode,
translation and orientation changes are weighted geometric quantities. Neither
metric is a direct measure of energy or execution time.

## 6. Deterministic path construction

### 6.1 Multi-start constrained nearest neighbor

At current vertex $i$, the feasible unvisited set is

$$
\mathcal{F}_i=
\{j\in\mathcal{U}\mid O_{ij}\geq\tau\}.
$$

The next vertex is selected by

$$
j^*=\arg\min_{j\in\mathcal{F}_i}C_{ij}.
$$

The planner repeats this greedy construction from every possible start. A
partial path is rejected when no feasible unvisited neighbor exists. Among all
complete paths, the path with the smallest full objective $J$ initializes
2-opt. Vertex index is the deterministic tie breaker.

~~~text
best_path <- none
for every start, ordered by (start distance, vertex ID):
    path <- [start]
    unvisited <- all vertices except start
    while unvisited is not empty:
        feasible <- unvisited vertices satisfying overlap >= threshold
        if feasible is empty:
            discard this start
        append the feasible vertex with minimum (edge cost, vertex ID)
    keep path if its complete objective is the lowest observed
if no complete path exists:
    stop before scan motion
~~~

Trying all starts reduces sensitivity to a poor initial choice, but the result
remains heuristic.

### 6.2 Overlap-constrained 2-opt

For every pair $1\leq a<b\leq n$, 2-opt reverses the inclusive subsequence:

$$
\boldsymbol{\pi}'=
(\pi_1,\ldots,\pi_{a-1},
 \pi_b,\pi_{b-1},\ldots,\pi_a,
 \pi_{b+1},\ldots,\pi_n).
$$

Reversals may include the first and last vertices. The start-state term is
therefore recomputed for every candidate. A reversal is admissible only if

$$
O_{\pi'_k,\pi'_{k+1}}\geq\tau
\quad\forall k\in\{1,\ldots,n-1\}.
$$

Every pass evaluates all reversals and accepts the lowest-cost strict
improvement. Refinement stops when no candidate improves the objective by more
than $10^{-12}$ or when the pass limit is reached.

~~~text
path <- best complete nearest-neighbor path
repeat for at most P passes:
    best_candidate <- path
    for every inclusive subsequence [a,b] with a < b:
        candidate <- path with [a,b] reversed
        reject candidate if any neighbor overlap is below threshold
        retain candidate if it strictly lowers the full open-path cost
    if no improving candidate exists:
        stop
    path <- best_candidate
return path
~~~

Each accepted operation is a permutation, so no reachable vertex is added,
removed, or duplicated.

## 7. End-to-end procedure

The implemented execution order is:

1. Generate $N$ mirrored golden-angle camera-pose pairs.
2. Apply the fixed local optical-frame rotations.
3. Initialize both synchronized RGB-D clients and the read-only IK client.
4. Verify the time source, camera streams, and hand-eye TF transforms.
5. Attempt to move the dual-arm group to **ready**.
6. Read the current complete joint state.
7. Convert every camera pair to TCP targets.
8. Submit every pair to collision-aware multi-tip IK and remove failures.
9. Generate $M$ deterministic spherical surface samples.
10. Compute all symmetric overlap, joint-distance, and edge-cost matrices.
11. Construct the multi-start constrained nearest-neighbor path.
12. Refine the complete open path using overlap-constrained 2-opt.
13. Write all planning diagnostics to the scan manifest.
14. Publish the complete IK waypoint animation and both camera paths for RViz.
15. Publish each selected camera and TCP target TF before its motion.
16. Move both arms simultaneously and capture synchronized RGB-D images.
17. Record and skip individual motion failures without ending the scan.
18. Report capture statistics and attempt to return to **ready**.

No scan-view motion starts before IK filtering and path optimization finish.

## 8. Metrics, complexity, and defaults

The reported path distance under the selected metric is

$$
L_D=D_{0,\pi_1}
+\sum_{k=1}^{n-1}D_{\pi_k,\pi_{k+1}}.
$$

Mean and minimum neighbor overlap are

$$
\bar O=\frac{1}{n-1}
\sum_{k=1}^{n-1}O_{\pi_k,\pi_{k+1}},
$$

$$
O_{\min}=
\min_{k\in\{1,\ldots,n-1\}}
O_{\pi_k,\pi_{k+1}}.
$$

For a single-vertex path, both overlap metrics are defined as one. The manifest
stores these values and objective $J$ before and after 2-opt.

Let $J_q$ be the number of retained motion joints and $P$ the 2-opt pass limit.

| Stage | Time complexity |
| --- | ---: |
| Candidate generation | $O(N)$ |
| IK filtering | $N$ service calls |
| Joint-distance matrix | $O(n^2J_q)$ |
| Pose-distance matrix | $O(n^2)$ |
| Projected-overlap matrix | $O(n^2M)$ |
| Multi-start nearest neighbor | $O(n^3)$ |
| 2-opt with full-path validation | $O(Pn^3)$ |

The symmetric overlap, joint-distance, and combined-cost matrices are computed
once before search.

Default experimental values are:

| Parameter | Default |
| --- | ---: |
| Scene center | $[0.40,0.0,0.0]$ m |
| Camera radius $r_c$ | $0.40$ m |
| Candidate pairs $N$ | 30 |
| Azimuth bounds | $[10^\circ,135^\circ]$ |
| Elevation bounds | $[30^\circ,75^\circ]$ |
| Proxy radius $r_s$ | $0.15$ m |
| Proxy samples $M$ | 2048 |
| Minimum overlap $\tau$ | 0.35 |
| Overlap weight $\lambda$ | 0.5 |
| Distance metric | pose |
| Pose translation weight $w_t$ | 1.0 |
| Pose rotation weight $w_R$ | 0.10 m/rad |
| 2-opt passes $P$ | 30 |
| Per-candidate IK timeout | 0.25 s |
| Joint-state/service timeout | 2.0 s |
| Left/right camera roll | $-90^\circ/+90^\circ$ |

## 9. Failure semantics and reproducibility

- A failed preliminary **ready** motion is recorded; planning continues from
  the measured current state.
- Missing IK service or recent joint state terminates planning.
- If all candidates fail IK, planning terminates.
- If greedy initialization cannot form a complete constrained path, the
  program stops before scan motion.
- A robot-motion API error marks that view as failed and continues to the next
  optimized target.
- Successful captures store desired and measured camera poses. Reconstruction
  uses the measured pose.

For publication, report the repository revision, robot model, MoveIt planning
group and kinematics plugin, collision scene, camera and hand-eye calibration,
all sampling and optimization parameters, selected distance metric and weights,
candidate rejection counts, initial and final path orders, $L_D$, $J$,
$\bar O$, $O_{\min}$, motion-failure count, and the use of simulation or wall
time. The manifest contains the run-dependent planner settings, rejected
candidates, retained IK solutions, optimized order, neighbor overlaps, capture
status, and image paths.

Useful experimental baselines are the original spiral order, a fixed angular
grid, joint-distance-only optimization with $\lambda=0$ and $\tau=0$, and the
greedy path without 2-opt. Planning metrics should be accompanied by downstream
registration or reconstruction metrics; proxy overlap alone does not prove
improved reconstruction.

## 10. Limitations

1. **Approximate visibility:** the spherical proxy is not the true scene. It
   does not model inter-object occlusion, depth ordering, or sensor noise.
2. **One IK result per view:** the search does not optimize over multiple IK
   branches for the same camera target.
3. **Heuristic optimization:** nearest neighbor and 2-opt do not guarantee a
   globally optimal TSP path. Greedy initialization may fail even when another
   feasible Hamiltonian path exists.
4. **Motion proxy:** joint mode mixes unweighted radians and any prismatic-joint
   meters. Pose mode uses a manually selected meter-per-radian rotation weight.
   Neither directly models actuator time or energy.
5. **Execution mismatch:** the Cartesian dual-arm motion service may choose an
   IK realization different from the solution used for path scoring.
   The published robot animation connects the retained IK waypoints and is a
   visualization, not a continuously collision-validated executable plan.
6. **Static-scene assumption:** overlap is computed once and assumes that the
   scene and calibrations remain fixed.
7. **Shared intrinsics:** one intrinsic model is currently used for both
   cameras.

Potential extensions include joint-range normalization, per-joint time weights,
multiple-solution IK graph search, beam or exact path initialization,
scene-derived visibility, per-camera intrinsics, and online replanning.

## 11. Implementation correspondence

- Complete scan parameter configuration:
  [config/multi_view_scan.yaml](../config/multi_view_scan.yaml)
- Spiral generation, projection, and path optimization:
  [scripts/scan_trajectory.py](../scripts/scan_trajectory.py)
- Pose and look-at geometry:
  [scripts/aux_math.py](../scripts/aux_math.py)
- Collision-aware MoveIt IK requests:
  [scripts/moveit_ik.py](../scripts/moveit_ik.py)
- Scan execution and manifest generation:
  [scripts/multi_view_scan.py](../scripts/multi_view_scan.py)
- Automated behavioral tests:
  [tests/test_scan_trajectory.py](../tests/test_scan_trajectory.py)
- Implementation contract:
  [TASK.md](../TASK.md)
