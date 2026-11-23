"""Effectivity of nodal-averaging vs. SPR recovery -- does better recovery
restore asymptotic exactness?

This is the controlled companion to ``04_effectivity.py``.  It measures the
effectivity index  theta = eta / ||sigma - sigma_h||_E  on the *same* two
benchmarks, but with two different recoveries driving the Zienkiewicz-Zhu
indicator:

* ``avg`` -- area-weighted nodal averaging (the simplest recovery);
* ``spr`` -- Superconvergent Patch Recovery (local least-squares linear fit over
  each node's element patch).

The true error is computed rigorously and identically for both (it does not
depend on the recovery), so the only thing that changes between the two columns
is sigma*.  The question is whether recovery *quality*, rather than the ZZ
framework itself, controls asymptotic exactness -- in particular at the
re-entrant corner, where plain averaging loses it.

Run from the repository root:
    python examples/05_spr_comparison.py
"""

from __future__ import annotations

import os

import numpy as np

from felas import (ElasticityProblem, Material, Mesh,
                   recover_nodal_stress, recover_nodal_stress_spr)
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


def _theta_both(mesh, element_stress, C, thickness, e_true):
    """Effectivity index for averaging and SPR against the same true error."""
    avg = zz_error_estimate(mesh, element_stress, C, thickness)
    spr_field = recover_nodal_stress_spr(mesh, element_stress)
    spr = zz_error_estimate(mesh, element_stress, C, thickness,
                            nodal_stress=spr_field)
    th_avg = avg.global_error / e_true if e_true > 0 else np.nan
    th_spr = spr.global_error / e_true if e_true > 0 else np.nan
    return avg.global_error, spr.global_error, th_avg, th_spr


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
    _, stress_h = prob.element_strain_stress(u)
    tris = mesh.nodes[mesh.elements]
    areas = mesh.areas()
    e2 = 0.0
    for e in range(mesh.n_elements):
        tri = tris[e]
        sh = stress_h[e]
        acc = 0.0
        for bary, w in zip(_BARY, _W):
            p = bary @ tri
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
        e_true = true_energy_norm_error_exact(mesh, prob, u, sol, C, mat.thickness)
        eta_a, eta_s, th_a, th_s = _theta_both(mesh, stress, C, mat.thickness, e_true)
        rows.append((mesh.n_dofs, sol.L / nx, e_true, th_a, th_s))
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
    ref_mesh = Mesh.lshape(n=4).uniform_refine(5)
    ref_prob = build_lshape(ref_mesh)
    u_ref = ref_prob.solve()
    U_ref = ref_prob.strain_energy(u_ref)
    print(f"L-shape overkill reference: {ref_mesh.n_dofs} dofs, U_ref = {U_ref:.8f}")

    rows = []
    mesh = Mesh.lshape(n=4)
    # Stop one level earlier than 04: the finest level there is reference-limited
    # (theta>1), so we report only the resolved range for a clean comparison.
    for _ in range(4):
        prob = build_lshape(mesh)
        u = prob.solve()
        _, stress = prob.element_strain_stress(u)
        U_h = prob.strain_energy(u)
        e_true = np.sqrt(max(2.0 * (U_ref - U_h), 0.0))
        eta_a, eta_s, th_a, th_s = _theta_both(mesh, stress, C, MAT.thickness, e_true)
        rows.append((mesh.n_dofs, e_true, th_a, th_s))
        mesh = mesh.uniform_refine(1)
    return rows, U_ref


def main():
    print("=== Cantilever (smooth): effectivity, averaging vs SPR ===")
    cant = cantilever_study()
    print(f"{'dofs':>7} {'h':>8} {'true_err':>12} {'theta_avg':>10} {'theta_spr':>10}")
    for dofs, h, e, ta, ts in cant:
        print(f"{dofs:7d} {h:8.4f} {e:12.4e} {ta:10.4f} {ts:10.4f}")

    print("\n=== L-shape (singular): effectivity, averaging vs SPR ===")
    lsh, _ = lshape_study()
    print(f"{'dofs':>7} {'true_err':>12} {'theta_avg':>10} {'theta_spr':>10}")
    for dofs, e, ta, ts in lsh:
        print(f"{dofs:7d} {e:12.4e} {ta:10.4f} {ts:10.4f}")

    _plot(cant, lsh, FIGDIR)


def _plot(cant, lsh, figdir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(figdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    h = np.array([r[1] for r in cant])
    axes[0].semilogx(h, [r[3] for r in cant], "o-", color="C0", label="nodal averaging")
    axes[0].semilogx(h, [r[4] for r in cant], "^-", color="C2", label="SPR")
    axes[0].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    axes[0].set_xlabel("element size h")
    axes[0].set_ylabel(r"effectivity index $\theta$")
    axes[0].set_title("Cantilever (smooth)")
    axes[0].invert_xaxis()
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    dofs = np.array([r[0] for r in lsh])
    axes[1].semilogx(dofs, [r[2] for r in lsh], "s-", color="C3", label="nodal averaging")
    axes[1].semilogx(dofs, [r[3] for r in lsh], "D-", color="C1", label="SPR")
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    axes[1].set_xlabel("degrees of freedom")
    axes[1].set_ylabel(r"effectivity index $\theta$")
    axes[1].set_title("L-shape (re-entrant singularity)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(figdir, "spr_comparison.png")
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
