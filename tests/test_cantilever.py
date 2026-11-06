"""FEM solution of the Timoshenko cantilever converges to the exact field."""

import numpy as np

from felas import ElasticityProblem, Material, Mesh
from felas.analytical import timoshenko_cantilever


def _solve(nx):
    sol = timoshenko_cantilever(E=1000.0, nu=0.3, P=1.0, L=8.0, c=2.0)
    mat = Material(E=sol.E, nu=sol.nu, plane="stress", thickness=sol.thickness)
    ny = max(2, nx // 2)
    mesh = Mesh.rectangle(sol.L, 2 * sol.c, nx=nx, ny=ny, x0=0.0, y0=-sol.c)

    prob = ElasticityProblem(mesh, mat)
    left = mesh.nodes_where(lambda x, y: np.isclose(x, 0.0))
    prob.prescribe(left, lambda x, y: tuple(v.item() for v in sol.displacement(x, y)))
    be = mesh.boundary_edges()
    on_right = np.all(np.isclose(mesh.nodes[be][:, :, 0], sol.L), axis=1)
    prob.add_edge_traction(be[on_right], sol.end_traction())
    u = prob.solve()

    areas = mesh.areas()
    num = den = 0.0
    for e, tri in enumerate(mesh.elements):
        c = mesh.nodes[tri].mean(axis=0)
        uh = u[tri].mean(axis=0)
        ue = np.array(sol.displacement(c[0], c[1]))
        num += areas[e] * np.sum((uh - ue) ** 2)
        den += areas[e] * np.sum(ue ** 2)
    return sol, mesh, u, np.sqrt(num / den)


def test_tip_deflection_accurate():
    sol, mesh, u, _ = _solve(48)
    # Tip deflection is the vertical displacement v at the free-end centreline.
    _, tip_exact = sol.displacement(sol.L, 0.0)
    tip_node = mesh.nodes_where(
        lambda x, y: np.isclose(x, sol.L) & np.isclose(y, 0.0))[0]
    assert abs(u[tip_node, 1] - tip_exact) / abs(tip_exact) < 0.02


def test_l2_convergence_rate():
    hs, errs = [], []
    for nx in (8, 16, 32, 64):
        sol, _, _, err = _solve(nx)
        hs.append(sol.L / nx)
        errs.append(err)
    hs = np.array(hs)
    errs = np.array(errs)
    rates = np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])
    # Linear triangles: expect ~O(h^2) in the L2 displacement norm.
    assert np.mean(rates) > 1.7
    assert errs[-1] < errs[0]
