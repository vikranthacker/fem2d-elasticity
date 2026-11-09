# Theory and formulation

This note documents the mathematics behind `felas`: the governing equations, the
linear-triangle finite element, global assembly, the Zienkiewicz–Zhu error
estimator, and the adaptive refinement algorithm.

## 1. Governing equations

For a 2-D body $\Omega$ of a linear-elastic isotropic material, the displacement
field $\mathbf{u} = (u, v)$ satisfies equilibrium

$$\nabla \cdot \boldsymbol{\sigma} + \mathbf{b} = \mathbf{0} \quad \text{in } \Omega,$$

with strain–displacement and constitutive relations

$$\boldsymbol{\varepsilon} = \begin{bmatrix}\varepsilon_{xx}\\\varepsilon_{yy}\\\gamma_{xy}\end{bmatrix}
= \begin{bmatrix}\partial_x u\\\partial_y v\\\partial_y u + \partial_x v\end{bmatrix},
\qquad \boldsymbol{\sigma} = \mathbf{D}\,\boldsymbol{\varepsilon}.$$

Boundary conditions are prescribed displacements on $\Gamma_D$ (Dirichlet) and
surface tractions $\bar{\mathbf{t}}$ on $\Gamma_N$ (Neumann).

The constitutive matrix $\mathbf{D}$ for **plane stress** is

$$\mathbf{D} = \frac{E}{1-\nu^2}\begin{bmatrix}1 & \nu & 0\\ \nu & 1 & 0\\ 0 & 0 & \tfrac{1-\nu}{2}\end{bmatrix},$$

and for **plane strain**

$$\mathbf{D} = \frac{E}{(1+\nu)(1-2\nu)}\begin{bmatrix}1-\nu & \nu & 0\\ \nu & 1-\nu & 0\\ 0 & 0 & \tfrac{1-2\nu}{2}\end{bmatrix}.$$

## 2. Weak form

Multiplying by a test function $\mathbf{w}$ and integrating by parts gives: find
$\mathbf{u}$ with $\mathbf{u} = \bar{\mathbf{u}}$ on $\Gamma_D$ such that

$$\int_\Omega \boldsymbol{\varepsilon}(\mathbf{w})^\top \mathbf{D}\,\boldsymbol{\varepsilon}(\mathbf{u})\,t\,\mathrm{d}\Omega
= \int_\Omega \mathbf{w}^\top \mathbf{b}\,t\,\mathrm{d}\Omega
+ \int_{\Gamma_N} \mathbf{w}^\top \bar{\mathbf{t}}\,t\,\mathrm{d}\Gamma$$

for all admissible $\mathbf{w}$, where $t$ is the thickness.

## 3. The linear triangle (CST)

On a triangle with nodes $(x_i, y_i)$, $i=0,1,2$, displacements are interpolated
with linear shape functions $N_i$. Because the $N_i$ are linear, their gradients
— and hence strains and stresses — are **constant** over the element. With
$2A = (x_1-x_0)(y_2-y_0) - (x_2-x_0)(y_1-y_0)$ and

$$b_i = \frac{\partial N_i}{\partial x}\,2A, \qquad c_i = \frac{\partial N_i}{\partial y}\,2A,$$

the strain–displacement matrix is

$$\mathbf{B} = \frac{1}{2A}\begin{bmatrix}
b_0 & 0 & b_1 & 0 & b_2 & 0\\
0 & c_0 & 0 & c_1 & 0 & c_2\\
c_0 & b_0 & c_1 & b_1 & c_2 & b_2
\end{bmatrix},$$

acting on the element displacement vector $\mathbf{u}_e = [u_0, v_0, \dots, v_2]^\top$.
The element stiffness matrix is

$$\mathbf{K}_e = t\,A\,\mathbf{B}^\top \mathbf{D}\,\mathbf{B}.$$

Consistent element loads for a constant body force $\mathbf{b}$ distribute
$tA/3$ to each node; for a traction on a 2-node edge of length $L$ the consistent
load is $\tfrac{tL}{6}(2\mathbf{t}_a + \mathbf{t}_b)$ at node $a$ (exact for a
traction varying linearly along the edge).

## 4. Assembly and solution

Element matrices and vectors are scattered into the global sparse system
$\mathbf{K}\mathbf{u} = \mathbf{f}$ using each node's two degrees of freedom
($2i$, $2i+1$). Dirichlet conditions are imposed by partitioning the system into
free ($f$) and constrained ($c$) sets and solving

$$\mathbf{K}_{ff}\,\mathbf{u}_f = \mathbf{f}_f - \mathbf{K}_{fc}\,\mathbf{u}_c,$$

with a sparse direct solver. Element stresses follow from
$\boldsymbol{\sigma}_e = \mathbf{D}\,\mathbf{B}\,\mathbf{u}_e$.

## 5. Stress recovery and the Zienkiewicz–Zhu estimator

The computed stress $\boldsymbol{\sigma}_h$ is piecewise constant and
discontinuous across element edges. A smoother, continuous field
$\boldsymbol{\sigma}^*$ is recovered by area-weighted averaging of element
stresses to the nodes and linear interpolation inside each element. The
**Zienkiewicz–Zhu** elementwise error indicator is the energy norm of the
difference,

$$\eta_K^2 = \int_K (\boldsymbol{\sigma}^* - \boldsymbol{\sigma}_h)^\top
\mathbf{C}^{-1} (\boldsymbol{\sigma}^* - \boldsymbol{\sigma}_h)\,t\,\mathrm{d}\Omega,
\qquad \mathbf{C} = \mathbf{D},$$

evaluated here with a 3-point (edge-midpoint) quadrature that is exact for the
quadratic integrand. The global estimate is $\eta = \big(\sum_K \eta_K^2\big)^{1/2}$,
and the relative error is $\eta / \big(\lVert\boldsymbol{\sigma}^*\rVert_E^2 + \eta^2\big)^{1/2}$.

> **This is a heuristic estimator, not a guaranteed bound.** The ZZ indicator
> gives *no* proven upper or lower bound on the true error $\lVert \boldsymbol{\sigma} - \boldsymbol{\sigma}_h \rVert_E$
> and no controlled effectivity index $\eta / \lVert \boldsymbol{\sigma} - \boldsymbol{\sigma}_h \rVert_E$.
> It is asymptotically exact only for *smooth* solutions on regular meshes,
> because it relies on the recovered field $\boldsymbol{\sigma}^*$ being a
> superconvergent approximation of $\boldsymbol{\sigma}$. At a singularity (the
> L-shape's re-entrant corner) or on the boundary (one-sided recovery patches)
> this assumption fails and $\eta$ becomes a proxy of unquantified fidelity. It
> is an excellent *refinement driver* — which elements to split — but should not
> be read as a certified error. Simple nodal averaging is used here; a
> superconvergent patch recovery (SPR) would improve the effectivity. For
> *guaranteed* a-posteriori bounds, see the verified-FEM literature on
> interpolation error constants and computer-assisted proofs.

## 6. Adaptive refinement

Each adaptive step performs **solve → estimate → mark → refine**:

- **Mark** (Dörfler / bulk): sort elements by $\eta_K^2$ and select the smallest
  set whose summed $\eta_K^2$ reaches a fraction $\theta$ of the total. Small
  $\theta$ marks few elements (aggressive grading); $\theta \to 1$ approaches
  uniform refinement.
- **Refine** (Rivara longest-edge bisection): a marked triangle is split across
  its longest edge by inserting the edge midpoint. If the neighbour sharing that
  edge does not have the same longest edge, it is refined first (recursively, via
  an explicit stack). This guarantees termination and produces a **conforming**
  mesh with no hanging nodes.

For problems with a singularity (e.g. the L-shaped domain's re-entrant corner),
this recovers a far better error-vs-dofs curve than uniform refinement, because
elements are added only where they reduce the error most.

## 7. Verification

The implementation is checked against:

- the **patch test** — a constant-stress (linear displacement) state is
  reproduced exactly on an arbitrary distorted mesh;
- the **Timoshenko cantilever** — the exact end-loaded beam solution, used to
  confirm the $O(h^2)$ L2 displacement convergence of linear triangles;
- the **Kirsch** plate-with-hole solution — the stress concentration factor of 3.
