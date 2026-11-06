"""Closed-form reference solutions used to verify the solver.

Two classic 2-D elasticity problems with known exact fields:

* :func:`timoshenko_cantilever` -- an end-loaded cantilever beam
  (Timoshenko & Goodier, *Theory of Elasticity*), used for convergence studies.
* :func:`kirsch_stress` -- the stress field around a circular hole in an
  infinite plate under uniaxial tension (stress concentration factor 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CantileverSolution:
    """Exact Timoshenko cantilever fields and parameters.

    Beam occupies ``0 <= x <= L`` and ``-c <= y <= c``.  A shear force of
    resultant ``P`` is applied (as a parabolic traction) on the free end
    ``x = L``; the beam is built in at ``x = 0`` where the exact displacement
    field is prescribed.  Plane stress.
    """

    E: float
    nu: float
    P: float
    L: float
    c: float
    thickness: float
    I: float

    def displacement(self, x, y):
        E, nu, P, L, c, I = self.E, self.nu, self.P, self.L, self.c, self.I
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        u = (P * y) / (6.0 * E * I) * ((6.0 * L - 3.0 * x) * x
                                       + (2.0 + nu) * (y * y - c * c))
        v = -(P) / (6.0 * E * I) * (3.0 * nu * y * y * (L - x)
                                    + (4.0 + 5.0 * nu) * c * c * x
                                    + (3.0 * L - x) * x * x)
        return u, v

    def stress(self, x, y):
        P, L, I, c = self.P, self.L, self.I, self.c
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        sxx = P * (L - x) * y / I
        syy = np.zeros_like(sxx)
        sxy = -P / (2.0 * I) * (c * c - y * y)
        return np.stack([sxx, syy, sxy], axis=-1)

    def end_traction(self) -> Callable[[float, float], tuple[float, float]]:
        """Traction (tx, ty) on the free end ``x = L`` (outward normal +x)."""
        P, I, c = self.P, self.I, self.c

        def traction(x, y):
            tx = 0.0                                   # sigma_xx = 0 at x = L
            ty = -P / (2.0 * I) * (c * c - y * y)      # = tau_xy on +x face
            return (tx, ty)

        return traction


def timoshenko_cantilever(E: float = 1000.0, nu: float = 0.3, P: float = 1.0,
                          L: float = 8.0, c: float = 2.0,
                          thickness: float = 1.0) -> CantileverSolution:
    """Build a :class:`CantileverSolution` with the given parameters."""
    I = thickness * (2.0 * c) ** 3 / 12.0
    return CantileverSolution(E, nu, P, L, c, thickness, I)


def kirsch_stress(x, y, S: float = 1.0, a: float = 1.0):
    """Kirsch solution: stress around a circular hole (radius ``a``) in an
    infinite plate under remote uniaxial tension ``S`` along x.

    Returns Voigt stress ``[sxx, syy, sxy]`` (same trailing shape as inputs).
    Points inside the hole (``r < a``) return NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        ar2 = (a / r) ** 2
        ar4 = ar2 * ar2
        c2 = np.cos(2.0 * th)
        s2 = np.sin(2.0 * th)
        srr = 0.5 * S * (1.0 - ar2) + 0.5 * S * (1.0 - 4.0 * ar2 + 3.0 * ar4) * c2
        sth = 0.5 * S * (1.0 + ar2) - 0.5 * S * (1.0 + 3.0 * ar4) * c2
        srt = -0.5 * S * (1.0 + 2.0 * ar2 - 3.0 * ar4) * s2
        # Rotate polar -> Cartesian.
        c = np.cos(th)
        s = np.sin(th)
        sxx = srr * c * c + sth * s * s - 2.0 * srt * s * c
        syy = srr * s * s + sth * c * c + 2.0 * srt * s * c
        sxy = (srr - sth) * s * c + srt * (c * c - s * s)
    out = np.stack([sxx, syy, sxy], axis=-1)
    out[r < a] = np.nan
    return out


def hoop_stress_at_hole(S: float = 1.0) -> float:
    """Peak hoop stress at the hole edge (theta = pi/2): the factor-3 result."""
    return 3.0 * S
