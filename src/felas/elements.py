"""3-node linear triangle (constant-strain triangle, CST) element.

For a linear triangle the strain-displacement matrix ``B`` and the element
stresses are constant over the element.  Degrees of freedom per element are
ordered ``[u0, v0, u1, v1, u2, v2]`` following the element's three nodes.
"""

from __future__ import annotations

import numpy as np


def triangle_area(coords: np.ndarray) -> float:
    """Signed area of a triangle given its 3x2 node coordinates.

    Positive when the nodes are ordered counter-clockwise.
    """
    (x0, y0), (x1, y1), (x2, y2) = coords
    return 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def strain_displacement_matrix(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the constant 3x6 B matrix and the (unsigned) area of the element.

    Parameters
    ----------
    coords : (3, 2) array
        Coordinates of the three triangle nodes.

    Returns
    -------
    B : (3, 6) array
        Strain-displacement matrix such that ``strain = B @ u_e`` with strain
        in Voigt form ``[eps_xx, eps_yy, gamma_xy]``.
    area : float
        Unsigned triangle area.
    """
    (x0, y0), (x1, y1), (x2, y2) = coords
    area_signed = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    area = abs(area_signed)
    if area < 1e-14:
        raise ValueError("Degenerate (zero-area) triangle encountered.")

    # Shape-function gradient coefficients.  b_i = dN_i/dx * 2A, c_i = dN_i/dy * 2A.
    two_area = 2.0 * area_signed
    b = np.array([y1 - y2, y2 - y0, y0 - y1]) / two_area
    c = np.array([x2 - x1, x0 - x2, x1 - x0]) / two_area

    B = np.zeros((3, 6))
    B[0, 0::2] = b          # eps_xx = du/dx
    B[1, 1::2] = c          # eps_yy = dv/dy
    B[2, 0::2] = c          # gamma_xy = du/dy + dv/dx
    B[2, 1::2] = b
    return B, area


def element_stiffness(coords: np.ndarray, D: np.ndarray, thickness: float = 1.0
                      ) -> tuple[np.ndarray, np.ndarray, float]:
    """Element stiffness matrix ``Ke = t * A * B^T D B``.

    Returns ``(Ke, B, area)`` so callers can reuse ``B`` for stress recovery.
    """
    B, area = strain_displacement_matrix(coords)
    Ke = thickness * area * (B.T @ D @ B)
    return Ke, B, area


def body_force_vector(coords: np.ndarray, bx: float, by: float,
                      thickness: float = 1.0) -> np.ndarray:
    """Consistent nodal load (length 6) for a constant body force (bx, by).

    Each node receives one third of the total body force on the element.
    """
    _, area = strain_displacement_matrix(coords)
    f = np.empty(6)
    w = thickness * area / 3.0
    f[0::2] = w * bx
    f[1::2] = w * by
    return f
