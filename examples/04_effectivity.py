"""Effectivity of the Zienkiewicz-Zhu estimator: does eta track the true error?

This experiment measures the effectivity index  theta = eta / ||sigma - sigma_h||_E
on two benchmarks, computing the *true* error rigorously in each case:

* Cantilever (smooth): the exact Timoshenko stress is known analytically, so the
  true energy-norm error is integrated directly with a degree-4 (6-point) Gauss
  rule per element -- no reliance on Galerkin orthogonality.

* L-shape (singular, re-entrant corner): the applied traction is constant, so the
  consistent load is exact and Galerkin orthogonality holds; the true energy-norm
  error is obtained from an overkill reference solve via
  ||u - u_h||_E^2 = 2 (U_ref - U_h).

The point: theta tends to ~1 where the recovered stress is superconvergent (the
smooth problem) and departs from it at the singularity -- an empirical echo of the
Zienkiewicz-Zhu (SPR, Part II) superconvergence condition.

Run from the repository root:
    python examples/04_effectivity.py
"""

from __future__ import annotations

import os

import numpy as np

from felas import ElasticityProblem, Material, Mesh
from felas.analytical import timoshenko_cantilever
from felas.error_estimator import zz_error_estimate

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")

# Degree-4 symmetric triangle quadrature (6 points, barycentric; weights sum to 1).
_BARY = np.array([
    [0.445948490915965, 0.445948490915965, 0.108103018168070],
    [0.445948490915965, 0.108103018168070, 0.445948490915965],
    [0.108103018168070, 0.445948490915965, 0.445948490915965],
    [0.091576213509771, 0.091576213509771, 0.816847572980459],
    [0.091576213509771, 0.816847572980459, 0.091576213509771],
    [0.816847572980459, 0.091576213509771, 0.091576213509771],
])
_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


# ------------------------------------------------------------------ cantilever
def build_cantilever(mesh, sol, mat):
    prob = ElasticityProblem(mesh, mat)
    left = mesh.nodes_where(lambda x, y: np.isclose(x, 0.0))
    prob.prescribe(left, lambda x, y: tuple(v.item() for v in sol.displacement(x, y)))
    be = mesh.boundary_edges()
    on_right = np.all(np.isclose(mesh.nodes[be][:, :, 0], sol.L), axis=1)
    prob.add_edge_traction(be[on_right], sol.end_traction())
    return prob


def true_energy_norm_error_exact(mesh, prob, u, sol, compliance, thickness=1.0):
    """||sigma_exact - sigma_h||_E by direct per-element Gauss quadrature."""
    _, stress_h = prob.element_strain_stress(u)
    tris = mesh.nodes[mesh.elements]        # (n_el, 3, 2)
    areas = mesh.areas()
    e2 = 0.0
    for e in range(mesh.n_elements):
        tri = tris[e]
        sh = stress_h[e]
        acc = 0.0
        for bary, w in zip(_BARY, _W):
            p = bary @ tri                  # physical quadrature point
            d = sol.stress(p[0], p[1]) - sh
            acc += w * (d @ compliance @ d)
        e2 += thickness * areas[e] * acc
    return float(np.sqrt(max(e2, 0.0)))


def cantilever_study():
    sol = timoshenko_cantilever(E=1000.0, nu=0.3, P=1.0, L=8.0, c=2.0)
    mat = Material(E=sol.E, nu=sol.nu, plane="stress", thickness=sol.thickness)
    C = mat.compliance_matrix()
    rows = []
    for nx in (8, 16, 32, 64, 128):
        ny = max(2, nx // 2)
        mesh = Mesh.rectangle(sol.L, 2 * sol.c, nx=nx, ny=ny, x0=0.0, y0=-sol.c)
        prob = build_cantilever(mesh, sol, mat)
        u = prob.solve()
        _, stress = prob.element_strain_stress(u)
        est = zz_error_estimate(mesh, stress, C, mat.thickness)
        e_true = true_energy_norm_error_exact(mesh, prob, u, sol, C, mat.thickness)
        theta = est.global_error / e_true if e_true > 0 else np.nan
        rows.append((mesh.n_dofs, sol.L / nx, est.global_error, e_true, theta))
    return rows


# --------------------------------------------------------------------- L-shape
MAT = Material(E=1000.0, nu=0.3, plane="stress")


def build_lshape(mesh):
    prob = ElasticityProblem(mesh, MAT)
    top = mesh.nodes_where(lambda x, y: np.isclose(y, 1.0))
    prob.fix_nodes(top, ux=0.0, uy=0.0)
    be = mesh.boundary_edges()
    on_bottom = np.all(np.isclose(mesh.nodes[be][:, :, 1], -1.0), axis=1)
    prob.add_edge_traction(be[on_bottom], lambda x, y: (0.0, -1.0))
    return prob


def lshape_study():
    C = MAT.compliance_matrix()
    # Overkill reference strain energy.
    ref_mesh = Mesh.lshape(n=4).uniform_refine(5)
    ref_prob = build_lshape(ref_mesh)
    u_ref = ref_prob.solve()
    U_ref = ref_prob.strain_energy(u_ref)
    print(f"L-shape overkill reference: {ref_mesh.n_dofs} dofs, U_ref = {U_ref:.8f}")

    rows = []
    mesh = Mesh.lshape(n=4)
    for _ in range(5):                       # stay well below the reference
        prob = build_lshape(mesh)
        u = prob.solve()
        _, stress = prob.element_strain_stress(u)
        est = zz_error_estimate(mesh, stress, C, MAT.thickness)
        U_h = prob.strain_energy(u)
        e_true = np.sqrt(max(2.0 * (U_ref - U_h), 0.0))   # Galerkin orthogonality
        theta = est.global_error / e_true if e_true > 0 else np.nan
        rows.append((mesh.n_dofs, est.global_error, e_true, theta))
        mesh = mesh.uniform_refine(1)
    return rows, U_ref


def main():
    print("=== Cantilever (smooth): true error by direct Gauss quadrature ===")
    cant = cantilever_study()
    print(f"{'dofs':>7} {'h':>8} {'eta':>12} {'true_err':>12} {'theta':>8}")
    for dofs, h, eta, e, th in cant:
        print(f"{dofs:7d} {h:8.4f} {eta:12.4e} {e:12.4e} {th:8.4f}")

    print("\n=== L-shape (singular): true error from overkill reference ===")
    lsh, U_ref = lshape_study()
    print(f"{'dofs':>7} {'eta':>12} {'true_err':>12} {'theta':>8}")
    for dofs, eta, e, th in lsh:
        print(f"{dofs:7d} {eta:12.4e} {e:12.4e} {th:8.4f}")

    _plot(cant, lsh, FIGDIR)


def _plot(cant, lsh, figdir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(figdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    h = np.array([r[1] for r in cant])
    th_c = np.array([r[4] for r in cant])
    axes[0].semilogx(h, th_c, "o-", color="C0")
    axes[0].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    axes[0].set_xlabel("element size h")
    axes[0].set_ylabel(r"effectivity index $\theta = \eta / \|\sigma-\sigma_h\|_E$")
    axes[0].set_title("Cantilever (smooth)")
    axes[0].invert_xaxis()
    axes[0].grid(True, which="both", alpha=0.3)

    dofs = np.array([r[0] for r in lsh])
    th_l = np.array([r[3] for r in lsh])
    axes[1].semilogx(dofs, th_l, "s-", color="C3")
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    axes[1].set_xlabel("degrees of freedom")
    axes[1].set_ylabel(r"effectivity index $\theta$")
    axes[1].set_title("L-shape (re-entrant singularity)")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(figdir, "effectivity.png")
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
