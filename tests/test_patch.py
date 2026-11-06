"""The patch test: a conforming element must reproduce any constant-stress
(linear displacement) state exactly, on an arbitrary mesh."""

import numpy as np
import pytest

from felas import ElasticityProblem, Material, Mesh


@pytest.mark.parametrize("plane", ["stress", "strain"])
def test_patch_test(plane):
    mat = Material(E=1234.0, nu=0.3, plane=plane)
    mesh = Mesh.rectangle(2.0, 1.0, nx=5, ny=3)

    # Jiggle interior nodes so the mesh is irregular (boundary kept straight).
    rng = np.random.default_rng(1)
    interior = np.setdiff1d(np.arange(mesh.n_nodes), mesh.boundary_nodes())
    mesh.nodes[interior] += rng.uniform(-0.05, 0.05, size=(len(interior), 2))

    # Exact linear displacement field -> constant strain [a, b, c].
    a, b, c = 1e-3, -2e-3, 3e-3

    def field(x, y):
        return (a * x + 0.5 * c * y, 0.5 * c * x + b * y)

    prob = ElasticityProblem(mesh, mat)
    prob.prescribe(mesh.boundary_nodes(), field)
    u = prob.solve()

    # Interior displacements must match the exact field.
    exact = np.array([field(x, y) for x, y in mesh.nodes])
    assert np.allclose(u, exact, atol=1e-9)

    # Element stress must be constant and equal to D @ [a, b, c].
    _, stress = prob.element_strain_stress(u)
    expected = mat.constitutive_matrix() @ np.array([a, b, c])
    assert np.allclose(stress, expected, atol=1e-7)
