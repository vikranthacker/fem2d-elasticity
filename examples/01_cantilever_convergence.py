"""Convergence study on the Timoshenko end-loaded cantilever.

Solves the cantilever on a sequence of uniformly refined meshes and compares the
FEM tip deflection and the displacement/energy errors against the exact
Timoshenko solution, confirming the expected convergence rates for linear
triangles (O(h^2) in the L2 displacement norm, O(h) in the energy norm).

Run from the repository root::

    python examples/01_cantilever_convergence.py
"""

from __future__ import annotations

import os

import numpy as np

from felas import ElasticityProblem, Material, Mesh
from felas.analytical import timoshenko_cantilever

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")


def build(mesh: Mesh, sol, mat: Material) -> ElasticityProblem:
    prob = ElasticityProblem(mesh, mat)
    L = sol.L
    # Prescribe the exact displacement on the built-in end x = 0.
    left = mesh.nodes_where(lambda x, y: np.isclose(x, 0.0))
    prob.prescribe(left, lambda x, y: tuple(v.item() for v in sol.displacement(x, y)))
    # Parabolic shear traction on the free end x = L.
    be = mesh.boundary_edges()
    on_right = np.all(np.isclose(mesh.nodes[be][:, :, 0], L), axis=1)
    prob.add_edge_traction(be[on_right], sol.end_traction())
    return prob


def l2_displacement_error(mesh: Mesh, u: np.ndarray, sol) -> float:
    """L2 norm of (u_h - u_exact), integrated with a 1-point (centroid) rule."""
    areas = mesh.areas()
    num = 0.0
    den = 0.0
    for e, tri in enumerate(mesh.elements):
        c = mesh.nodes[tri].mean(axis=0)
        uh = u[tri].mean(axis=0)
        ux, uy = sol.displacement(c[0], c[1])
        ue = np.array([ux, uy])
        num += areas[e] * np.sum((uh - ue) ** 2)
        den += areas[e] * np.sum(ue ** 2)
    return float(np.sqrt(num / den))


def main() -> None:
    sol = timoshenko_cantilever(E=1000.0, nu=0.3, P=1.0, L=8.0, c=2.0)
    mat = Material(E=sol.E, nu=sol.nu, plane="stress", thickness=sol.thickness)

    _, tip_exact = sol.displacement(sol.L, 0.0)   # vertical tip deflection
    exact_energy = _exact_energy(sol, mat)

    rows = []
    for nx in (4, 8, 16, 32, 64):
        ny = max(2, nx // 2)
        mesh = Mesh.rectangle(sol.L, 2.0 * sol.c, nx=nx, ny=ny, x0=0.0, y0=-sol.c)
        prob = build(mesh, sol, mat)
        u = prob.solve()

        tip_nodes = mesh.nodes_where(
            lambda x, y: np.isclose(x, sol.L) & np.isclose(y, 0.0))
        tip_fem = u[tip_nodes[0], 1] if len(tip_nodes) else np.nan
        h = sol.L / nx
        l2 = l2_displacement_error(mesh, u, sol)
        energy = prob.strain_energy(u)
        en_err = abs(energy - exact_energy) / exact_energy
        rows.append((mesh.n_dofs, h, tip_fem, l2, en_err))

    print(f"exact tip deflection u(L,0) = {float(tip_exact):.6f}")
    print(f"{'dofs':>7} {'h':>8} {'tip_fem':>12} {'L2_err':>12} {'energy_err':>12}")
    for dofs, h, tip, l2, en in rows:
        print(f"{dofs:7d} {h:8.4f} {tip:12.6f} {l2:12.4e} {en:12.4e}")

    rates = _rates([r[1] for r in rows], [r[3] for r in rows])
    print(f"\nobserved L2 convergence rate ~ {np.mean(rates):.2f} (expected ~2.0)")

    _plot(rows, FIGDIR)


def _exact_energy(sol, mat: Material) -> float:
    """Exact strain energy via fine Gauss integration of the analytic stress."""
    C = mat.compliance_matrix()
    n = 200
    xs = np.linspace(0, sol.L, n)
    ys = np.linspace(-sol.c, sol.c, n)
    gx, gy = np.meshgrid(xs, ys)
    s = sol.stress(gx.ravel(), gy.ravel())
    dens = np.einsum("ij,jk,ik->i", s, C, s)
    area_cell = (sol.L / (n - 1)) * (2 * sol.c / (n - 1))
    return 0.5 * mat.thickness * dens.sum() * area_cell


def _rates(h, err):
    h = np.asarray(h)
    err = np.asarray(err)
    return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])


def _plot(rows, figdir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(figdir, exist_ok=True)
    h = np.array([r[1] for r in rows])
    l2 = np.array([r[3] for r in rows])
    en = np.array([r[4] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(h, l2, "o-", label="L2 displacement error")
    ax.loglog(h, en, "s-", label="relative energy error")
    ax.loglog(h, l2[0] * (h / h[0]) ** 2, "k--", lw=1, label="slope 2")
    ax.loglog(h, en[0] * (h / h[0]) ** 1, "k:", lw=1, label="slope 1")
    ax.set_xlabel("element size h")
    ax.set_ylabel("relative error")
    ax.set_title("Cantilever convergence (linear triangles)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(figdir, "cantilever_convergence.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
