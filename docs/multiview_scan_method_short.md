## Reachability-Filtered and Overlap-Aware Multiview Scan Planning

We plan synchronized trajectories for two wrist-mounted RGB-D cameras by
combining deterministic hemispherical sampling, collision-aware inverse
kinematics (IK), and overlap-constrained path optimization. A paired left/right
camera target is treated as one planning vertex because both robot arms move
simultaneously. Let $\mathbf{c}\in\mathbb{R}^3$ be the scene center, $r_c$ the
camera-orbit radius, and $N$ the number of candidate pairs. For candidate
$i\in\{0,\ldots,N-1\}$, elevation is sampled uniformly in spherical surface
area and azimuth follows a golden-angle sequence:

$$
t_i=\frac{i+\tfrac{1}{2}}{N},\qquad
\beta_i=\arcsin\!\left[
\sin\beta_{\min}
+t_i\left(\sin\beta_{\max}-\sin\beta_{\min}\right)\right],
$$

$$
g=\frac{\sqrt{5}-1}{2},\qquad
\alpha_i=\alpha_{\min}+
\left[\left(\tfrac{1}{2}+ig\right)\bmod 1\right]
\left(\alpha_{\max}-\alpha_{\min}\right).
$$

With $\rho_i=r_c\cos\beta_i$, the two camera positions are mirrored across the
world $x$-$z$ plane:

$$
\mathbf{p}_{L,i}=
\begin{bmatrix}
c_x+\rho_i\cos\alpha_i\\
c_y+\rho_i\sin\alpha_i\\
c_z+r_c\sin\beta_i
\end{bmatrix},\qquad
\mathbf{p}_{R,i}=
\begin{bmatrix}
c_x+\rho_i\cos\alpha_i\\
c_y-\rho_i\sin\alpha_i\\
c_z+r_c\sin\beta_i
\end{bmatrix}.
$$

The $y>0$ pose is assigned to the left arm and the $y<0$ pose to the right
arm. Each optical frame is oriented so that its $+z$ axis points toward
$\mathbf{c}$. For camera position $\mathbf{p}$, the look-at basis is

$$
\mathbf{z}_C=\frac{\mathbf{c}-\mathbf{p}}
{\|\mathbf{c}-\mathbf{p}\|_2},\qquad
\mathbf{x}_C=\frac{\mathbf{z}_C\times\mathbf{u}_W}
{\|\mathbf{z}_C\times\mathbf{u}_W\|_2},\qquad
\mathbf{y}_C=\mathbf{z}_C\times\mathbf{x}_C,
$$

where $\mathbf{u}_W=[0,0,1]^\mathsf{T}$. Fixed local optical-axis rotations of
$-90^\circ$ and $+90^\circ$ are applied to the left and right cameras,
respectively. The measured hand-eye transform converts every desired camera
pose to a TCP target:

$$
{}^W\mathbf{T}_{E_a,i}
={}^W\mathbf{T}_{C_a,i}\,{}^{C_a}\mathbf{T}_{E_a},
\qquad a\in\{L,R\}.
$$

Before scanning, the robot attempts to reach its predefined dual-arm **ready**
configuration. Every candidate pair is subsequently evaluated using one
collision-aware MoveIt multi-tip IK request containing both TCP targets. The
request is seeded with the current full robot state and enables collision
avoidance. If the synchronized IK problem fails, the complete left/right pair
is removed. For each retained vertex, the solution is stored as a consistently
ordered non-gripper joint vector $\mathbf{q}_i\in\mathbb{R}^J$.

Temporal image overlap is predicted using $M$ deterministic samples on a
spherical surface of radius $r_s$ around the scene. Spherical Fibonacci
sampling is defined by

$$
z_k=1-2\frac{k+\tfrac{1}{2}}{M},\qquad
\theta_k=k\pi(3-\sqrt{5}),\qquad
\mathbf{s}_k=\mathbf{c}+r_s
\begin{bmatrix}
\sqrt{1-z_k^2}\cos\theta_k\\
\sqrt{1-z_k^2}\sin\theta_k\\
z_k
\end{bmatrix}.
$$

For a camera pose $(\mathbf{R}_{WC},\mathbf{t}_{WC})$, a sample is transformed
and projected using calibrated pinhole intrinsics:

$$
\mathbf{s}_k^C=\mathbf{R}_{WC}^{\mathsf{T}}
(\mathbf{s}_k-\mathbf{t}_{WC}),\qquad
u_k=f_x\frac{x_k^C}{z_k^C}+c_x^{\mathrm{img}},\qquad
v_k=f_y\frac{y_k^C}{z_k^C}+c_y^{\mathrm{img}}.
$$

A sample is visible if it is in front of the camera, lies inside the image
bounds, and its outward spherical normal faces the camera. Let
$\mathcal{S}_{a,i}$ be the indices visible to arm $a$ at vertex $i$. The
per-arm temporal overlap is the Jaccard index, and the paired overlap uses the
weaker camera:

$$
O^a_{ij}=\frac{|\mathcal{S}_{a,i}\cap\mathcal{S}_{a,j}|}
{|\mathcal{S}_{a,i}\cup\mathcal{S}_{a,j}|},\qquad
O_{ij}=\min(O^L_{ij},O^R_{ij}).
$$

For the $n\leq N$ reachable vertices, the motion metric can be evaluated in
joint or TCP-pose space. Joint mode uses

$$
D^q_{ij}=\|\mathbf{q}_i-\mathbf{q}_j\|_2.
$$

For pose mode, let $d^a_{t,ij}$ be the Euclidean translation distance and
$\theta^a_{ij}$ the shortest SO(3) angle between TCP orientations for arm
$a\in\{L,R\}$. Coordinated pose distance is

$$
D^p_{ij}=
w_t\sqrt{(d^L_{t,ij})^2+(d^R_{t,ij})^2}
+w_R\sqrt{(\theta^L_{ij})^2+(\theta^R_{ij})^2},
$$

where $w_t$ and $w_R$ control the relative translation and rotation costs. The
selected distance $D_{ij}\in\{D^q_{ij},D^p_{ij}\}$ is combined with overlap:

$$
C_{ij}=D_{ij}+\lambda(1-O_{ij}),
$$

where $\lambda\geq 0$ controls the soft preference for overlap. We seek an open
permutation $\boldsymbol{\pi}=(\pi_1,\ldots,\pi_n)$ minimizing

$$
J(\boldsymbol{\pi})=
D_{0,\pi_1}+
\sum_{k=1}^{n-1}C_{\pi_k,\pi_{k+1}},
$$

subject to the hard neighbor constraint

$$
O_{\pi_k,\pi_{k+1}}\geq\tau,\qquad k=1,\ldots,n-1,
$$

where $D_{0,\pi_1}$ uses either the measured current joint vector or measured
current TCP pair, consistently with the selected metric, and $\tau$ is the
minimum permitted overlap. We initialize the path using constrained nearest
neighbor from every possible starting vertex and retain the lowest-cost
complete path. We then apply deterministic best-improvement 2-opt: every
subsequence reversal is considered, candidates violating any overlap constraint
are discarded, and the lowest-cost strict improvement is accepted. Refinement
ends when no improvement exists or the pass limit is reached. Before motion,
the greedy and refined IK waypoint sequences are published as separate MoveIt
display trajectories. Their dual-camera routes are simultaneously rendered as
wide, contrasting world-frame RViz line strips. Individual target TFs are also
published before their corresponding motions. Both arms are
then commanded together; a failed motion is recorded and skipped without
terminating the remaining scan.

Our default configuration uses $N=30$, $r_c=0.40$ m,
$[\alpha_{\min},\alpha_{\max}]=[10^\circ,135^\circ]$,
$[\beta_{\min},\beta_{\max}]=[30^\circ,75^\circ]$, $r_s=0.15$ m,
$M=2048$, $\tau=0.35$, $\lambda=0.5$, pose distance with $w_t=1.0$ and
$w_R=0.10$ m/rad, and 30 2-opt passes. The method is deterministic for fixed IK
results and parameters. It is a heuristic rather than an exact TSP solver:
spherical visibility approximates the unknown scene, one IK solution is
retained per candidate, and either selectable distance remains a proxy rather
than execution time or energy.
