import numpy as np

from felas import ElasticityProblem, Material, Mesh
from felas.adaptivity import adaptive_solve, dorfler_marking
from felas.error_estimator import zz_error_estimate

MAT = Material(E=1000.0, nu=0.3, plane="stress")


def test_dorfler_marking():
    ind = np.array([3.0, 1.0, 1.0, 1.0])     # eta^2 = [9, 1, 1, 1], total 12
    marked = dorfler_marking(ind, theta=0.5)  # need >= 6 -> first element only
    assert set(marked.tolist()) == {0}
    assert len(dorfler_marking(ind, theta=1.0)) == 4


def _lshape_problem(mesh: Mesh) -> ElasticityProblem:
    prob = ElasticityProblem(mesh, MAT)
    top = mesh.nodes_where(lambda x, y: np.isclose(y, 1.0))
    prob.fix_nodes(top, ux=0.0, uy=0.0)
    be = mesh.boundary_edges()
    on_bottom = np.all(np.isclose(mesh.nodes[be][:, :, 1], -1.0), axis=1)
    prob.add_edge_traction(be[on_bottom], lambda x, y: (0.0, -1.0))
    return prob


def test_adaptive_reduces_error_and_concentrates_at_corner():
    result = adaptive_solve(Mesh.lshape(n=4), _lshape_problem, MAT,
                            steps=10, theta=0.4)
    errs = [s.global_error for s in result.history]
    # Error decreases overall.
    assert errs[-1] < errs[0]
    # Refinement concentrates near the re-entrant corner at the origin.
    centroids = result.mesh.nodes[result.mesh.elements].mean(axis=1)
    dist = np.hypot(centroids[:, 0], centroids[:, 1])
    smallest = result.mesh.areas().argsort()[:max(1, result.mesh.n_elements // 20)]
    assert np.median(dist[smallest]) < 0.3


def test_adaptive_beats_uniform_at_similar_dofs():
    # Adaptive error at its final dofs should beat uniform refinement
    # evaluated at comparable or greater dofs.
    result = adaptive_solve(Mesh.lshape(n=4), _lshape_problem, MAT,
                            steps=12, theta=0.4)
    adap_dofs = result.history[-1].n_dofs
    adap_err = result.history[-1].global_error

    mesh = Mesh.lshape(n=4)
    uni_err = None
    for _ in range(6):
        prob = _lshape_problem(mesh)
        u = prob.solve()
        _, stress = prob.element_strain_stress(u)
        est = zz_error_estimate(mesh, stress, MAT.compliance_matrix())
        if mesh.n_dofs >= adap_dofs:
            uni_err = est.global_error
            break
        mesh = mesh.uniform_refine(1)
    assert uni_err is not None
    assert adap_err < uni_err
