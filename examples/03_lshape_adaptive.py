"""L-shaped domain -- adaptive vs uniform refinement at a corner singularity.

The re-entrant corner of an L-shaped domain produces a stress singularity.
Uniform refinement wastes effort everywhere; ZZ-driven adaptive refinement
concentrates elements at the corner and reaches a smaller error for the same
number of degrees of freedom.  This script runs both and plots error vs dofs.

Run from the repository root::

    python examples/03_lshape_adaptive.py
"""

from __future__ import annotations

import os

import numpy as np

from felas import ElasticityProblem, Material, Mesh
from felas.adaptivity import adaptive_solve
from felas.error_estimator import zz_error_estimate

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")

MAT = Material(E=1000.0, nu=0.3, plane="stress")


def build(mesh: Mesh) -> ElasticityProblem:
    prob = ElasticityProblem(mesh, MAT)
    # Clamp the top edge (y = 1); pull the bottom edge (y = -1) downward.
    top = mesh.nodes_where(lambda x, y: np.isclose(y, 1.0))
    prob.fix_nodes(top, ux=0.0, uy=0.0)

    be = mesh.boundary_edges()
    on_bottom = np.all(np.isclose(mesh.nodes[be][:, :, 1], -1.0), axis=1)
    prob.add_edge_traction(be[on_bottom], lambda x, y: (0.0, -1.0))
    return prob


def _error(mesh: Mesh):
    prob = build(mesh)
    u = prob.solve()
    _, stress = prob.element_strain_stress(u)
    est = zz_error_estimate(mesh, stress, MAT.compliance_matrix(), MAT.thickness)
    return prob, u, est


def main() -> None:
    # Uniform refinement sequence.
    uni = []
    mesh = Mesh.lshape(n=4)
    for _ in range(5):
        _, _, est = _error(mesh)
        uni.append((mesh.n_dofs, est.global_error))
        mesh = mesh.uniform_refine(1)
    _, _, est = _error(mesh)
    uni.append((mesh.n_dofs, est.global_error))

    # Adaptive refinement sequence from the same coarse mesh.
    result = adaptive_solve(Mesh.lshape(n=4), build, MAT,
                            steps=14, theta=0.4, verbose=False)
    adap = [(s.n_dofs, s.global_error) for s in result.history]

    print(f"{'strategy':>10} {'dofs':>8} {'eta':>12}")
    for d, e in uni:
        print(f"{'uniform':>10} {d:8d} {e:12.4e}")
    for d, e in adap:
        print(f"{'adaptive':>10} {d:8d} {e:12.4e}")

    _plot(uni, adap, result, FIGDIR)


def _plot(uni, adap, result, figdir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    from felas.plotting import plot_mesh
    os.makedirs(figdir, exist_ok=True)

    uni = np.array(uni)
    adap = np.array(adap)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].loglog(uni[:, 0], uni[:, 1], "o-", label="uniform")
    axes[0].loglog(adap[:, 0], adap[:, 1], "s-", label="adaptive (ZZ)")
    axes[0].set_xlabel("degrees of freedom")
    axes[0].set_ylabel("estimated energy-norm error")
    axes[0].set_title("L-shape: adaptive vs uniform")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    plot_mesh(result.mesh, ax=axes[1])
    axes[1].set_title(f"Adapted mesh ({result.mesh.n_elements} elements)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    fig.tight_layout()
    out = os.path.join(figdir, "lshape_adaptive.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
