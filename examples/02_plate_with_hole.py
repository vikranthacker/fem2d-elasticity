"""Plate with a circular hole -- stress concentration (Kirsch benchmark).

Quarter model of a plate under remote uniaxial tension, with symmetry boundary
conditions.  The hoop stress at the hole edge approaches the classic stress
concentration factor of 3.  Adaptive refinement is driven towards the hole,
sharpening the peak-stress estimate.

Run from the repository root::

    python examples/02_plate_with_hole.py
"""

from __future__ import annotations

import os

import numpy as np

from felas import ElasticityProblem, Material, Mesh
from felas.adaptivity import adaptive_solve
from felas.recovery import recover_nodal_stress, von_mises

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")

RADIUS = 1.0
LENGTH = 5.0          # half-width of the plate (>> radius so Kirsch applies)
TENSION = 1.0


def build(mesh: Mesh) -> ElasticityProblem:
    mat = Material(E=1000.0, nu=0.3, plane="stress")
    prob = ElasticityProblem(mesh, mat)

    # Symmetry: bottom edge (y = 0) has uy = 0; left edge (x = 0) has ux = 0.
    bottom = mesh.nodes_where(lambda x, y: np.isclose(y, 0.0))
    left = mesh.nodes_where(lambda x, y: np.isclose(x, 0.0))
    prob.fix_nodes(bottom, ux=None, uy=0.0)
    prob.fix_nodes(left, ux=0.0, uy=None)

    # Remote tension applied on the right outer edge x = LENGTH.
    be = mesh.boundary_edges()
    on_right = np.all(np.isclose(mesh.nodes[be][:, :, 0], LENGTH), axis=1)
    prob.add_edge_traction(be[on_right], lambda x, y: (TENSION, 0.0))
    return prob


def peak_hoop_stress(mesh: Mesh, prob: ElasticityProblem, u) -> float:
    """Recovered sigma_xx at the top of the hole (0, RADIUS) ~ 3*TENSION."""
    _, stress = prob.element_strain_stress(u)
    nodal = recover_nodal_stress(mesh, stress)
    top = mesh.nodes_where(lambda x, y: np.isclose(x, 0.0) & np.isclose(y, RADIUS))
    if len(top) == 0:
        # fall back to the node nearest (0, RADIUS)
        d = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1] - RADIUS)
        top = [int(np.argmin(d))]
    return float(nodal[top[0], 0])


def main() -> None:
    mat = Material(E=1000.0, nu=0.3, plane="stress")
    mesh0 = Mesh.plate_with_hole(RADIUS, LENGTH, n_radial=6, n_circ=10)

    result = adaptive_solve(mesh0, build, mat, steps=6, theta=0.5, verbose=True)

    peak = peak_hoop_stress(result.mesh, result.problem, result.displacement)
    print(f"\npeak hoop stress at hole = {peak:.4f}  "
          f"(Kirsch factor for an infinite plate = 3.0)")

    _plot(result, FIGDIR)


def _plot(result, figdir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    from felas.plotting import plot_mesh, plot_nodal_field
    os.makedirs(figdir, exist_ok=True)

    mesh = result.mesh
    _, stress = result.problem.element_strain_stress(result.displacement)
    nodal = recover_nodal_stress(mesh, stress)
    vm = von_mises(nodal)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_mesh(mesh, ax=axes[0])
    axes[0].set_title(f"Adapted mesh ({mesh.n_elements} elements)")
    plot_nodal_field(mesh, vm, ax=axes[1], cmap="inferno",
                     title="von Mises stress")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    out = os.path.join(figdir, "plate_with_hole.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
