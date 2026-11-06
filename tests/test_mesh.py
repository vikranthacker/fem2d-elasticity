import numpy as np

from felas import Mesh


def _is_conforming(mesh: Mesh) -> bool:
    """Every edge is used by exactly 1 (boundary) or 2 (interior) elements."""
    e = mesh.elements
    pairs = np.vstack([e[:, [0, 1]], e[:, [1, 2]], e[:, [2, 0]]])
    pairs = np.sort(pairs, axis=1)
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    return bool(np.all((counts == 1) | (counts == 2)))


def test_generators_valid():
    for mesh in (Mesh.rectangle(3.0, 2.0, 6, 4),
                 Mesh.lshape(3),
                 Mesh.plate_with_hole(1.0, 4.0, 5, 6)):
        assert np.all(mesh.areas() > 0)          # all CCW, non-degenerate
        assert _is_conforming(mesh)


def test_uniform_refine_conserves_area_and_conformity():
    mesh = Mesh.lshape(2)
    area0 = mesh.areas().sum()
    for _ in range(2):
        mesh = mesh.uniform_refine(1)
        assert abs(mesh.areas().sum() - area0) < 1e-9
        assert _is_conforming(mesh)


def test_uniform_refine_counts():
    mesh = Mesh.rectangle(1.0, 1.0, 2, 2)
    n0 = mesh.n_elements
    mesh2 = mesh.uniform_refine(1)
    assert mesh2.n_elements == 4 * n0


def test_longest_edge_bisection_is_conforming():
    mesh = Mesh.rectangle(2.0, 1.0, 6, 3)
    area0 = mesh.areas().sum()
    rng = np.random.default_rng(3)
    for _ in range(6):
        k = max(1, mesh.n_elements // 4)
        marked = rng.choice(mesh.n_elements, size=k, replace=False)
        refined = mesh.refine_marked(marked)
        assert _is_conforming(refined)
        assert abs(refined.areas().sum() - area0) < 1e-9
        assert refined.n_elements > mesh.n_elements
        mesh = refined


def test_refine_marked_minimum_bisection():
    # Refining one element adds at least one element and one node.
    mesh = Mesh.rectangle(1.0, 1.0, 2, 2)
    refined = mesh.refine_marked([0])
    assert refined.n_elements >= mesh.n_elements + 1
    assert refined.n_nodes >= mesh.n_nodes + 1
    assert _is_conforming(refined)
