import numpy as np
import pytest

from felas.analytical import (hoop_stress_at_hole, kirsch_stress,
                              timoshenko_cantilever)
from felas.materials import Material


def test_timoshenko_displacement_stress_consistency():
    """Differentiating the displacement field reproduces the stress field."""
    sol = timoshenko_cantilever(E=1000.0, nu=0.3, P=1.0, L=8.0, c=2.0)
    D = Material(E=sol.E, nu=sol.nu, plane="stress").constitutive_matrix()
    h = 1e-6
    rng = np.random.default_rng(0)
    for _ in range(100):
        x = rng.uniform(0.5, 7.5)
        y = rng.uniform(-1.8, 1.8)
        ux_p, vy_p = sol.displacement(x + h, y)
        ux_m, vy_m = sol.displacement(x - h, y)
        ux_py, vy_py = sol.displacement(x, y + h)
        ux_my, vy_my = sol.displacement(x, y - h)
        exx = (ux_p - ux_m) / (2 * h)
        eyy = (vy_py - vy_my) / (2 * h)
        gxy = (ux_py - ux_my) / (2 * h) + (vy_p - vy_m) / (2 * h)
        stress_fd = D @ np.array([exx, eyy, gxy])
        assert np.allclose(stress_fd, sol.stress(x, y), atol=1e-5)


def test_kirsch_concentration_factor():
    """Hoop stress at the top of the hole equals 3 * remote tension."""
    S, a = 2.0, 1.5
    # Point just at the hole edge, theta = pi/2 (top).
    stress = kirsch_stress(0.0, a, S=S, a=a)
    # sigma_xx there equals the hoop (tangential) stress = 3S.
    assert stress[0] == pytest.approx(3.0 * S, rel=1e-6)
    assert hoop_stress_at_hole(S) == pytest.approx(3.0 * S)


def test_kirsch_far_field():
    """Far from the hole the field tends to uniaxial tension (sxx -> S)."""
    S = 1.0
    stress = kirsch_stress(100.0, 0.0, S=S, a=1.0)
    assert stress[0] == pytest.approx(S, abs=1e-3)
    assert stress[1] == pytest.approx(0.0, abs=1e-3)
