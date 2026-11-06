import numpy as np
import pytest

from felas.elements import (body_force_vector, element_stiffness,
                            strain_displacement_matrix, triangle_area)
from felas.materials import Material


def test_triangle_area():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    assert triangle_area(coords) == pytest.approx(3.0)
    # reversed orientation -> negative signed area
    assert triangle_area(coords[[0, 2, 1]]) == pytest.approx(-3.0)


def test_rigid_body_motion_gives_zero_strain():
    coords = np.array([[0.2, 0.1], [1.3, 0.0], [0.4, 1.1]])
    B, _ = strain_displacement_matrix(coords)
    # Pure translation
    trans = np.array([0.7, -0.3] * 3)
    assert np.allclose(B @ trans, 0.0)
    # Infinitesimal rotation u = (-y, x) about origin
    rot = np.array([-c[1] if k % 2 == 0 else c[0]
                    for c in coords for k in range(2)])
    assert np.allclose(B @ rot, 0.0, atol=1e-12)


def test_constant_strain_reproduced():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    B, _ = strain_displacement_matrix(coords)
    # Linear field u = a*x, v = b*y  ->  eps = [a, b, 0]
    a, b = 0.03, -0.02
    u = np.array([a * x + 0 * y if k == 0 else b * y
                  for (x, y) in coords for k in range(2)])
    eps = B @ u
    assert eps == pytest.approx([a, b, 0.0])


def test_stiffness_symmetric_and_singular():
    coords = np.array([[0.0, 0.0], [1.0, 0.2], [0.3, 1.1]])
    D = Material(E=210e3, nu=0.3).constitutive_matrix()
    Ke, B, area = element_stiffness(coords, D, thickness=1.0)
    assert np.allclose(Ke, Ke.T)
    # 3 rigid-body modes -> rank 3 -> 3 zero eigenvalues
    eig = np.linalg.eigvalsh(Ke)
    assert np.sum(np.abs(eig) < 1e-6 * np.max(eig)) == 3


def test_body_force_sums_to_total():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])  # area 2
    f = body_force_vector(coords, bx=1.0, by=-2.0, thickness=1.0)
    assert f[0::2].sum() == pytest.approx(2.0)      # 1.0 * area
    assert f[1::2].sum() == pytest.approx(-4.0)     # -2.0 * area
