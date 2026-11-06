"""Zienkiewicz-Zhu (ZZ) recovery-based a-posteriori error estimation.

The element stress ``sigma_h`` is piecewise constant; a recovered field
``sigma*`` is obtained by averaging to the nodes (see :mod:`recovery`) and then
interpolated linearly inside each element.  The elementwise energy-norm error
indicator is

    eta_K^2 = integral_K (sigma* - sigma_h)^T C^{-1} (sigma* - sigma_h) dOmega

where ``C`` is the constitutive matrix.  The global estimate is the root of the
sum, and the relative error is ``eta / sqrt(||sigma*||_E^2 + eta^2)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh
from .recovery import recover_nodal_stress


@dataclass
class ErrorEstimate:
    """Result of a ZZ error estimation."""

    element_indicators: np.ndarray   # eta_K for each element
    global_error: float              # sqrt(sum eta_K^2), energy norm
    relative_error: float            # global_error / total energy norm
    recovered_nodal_stress: np.ndarray


# 3-point quadrature at edge midpoints (exact for quadratics on a triangle).
_MID_BARY = np.array([[0.5, 0.5, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5]])
_MID_W = 1.0 / 3.0


def zz_error_estimate(mesh: Mesh, element_stress: np.ndarray,
                      compliance: np.ndarray, thickness: float = 1.0
                      ) -> ErrorEstimate:
    """Compute the ZZ error estimate.

    Parameters
    ----------
    mesh : Mesh
    element_stress : (n_elem, 3) array
        Constant element stress (Voigt).
    compliance : (3, 3) array
        Inverse constitutive matrix ``C^{-1}`` (``Material.compliance_matrix``).
    thickness : float
    """
    element_stress = np.asarray(element_stress)
    nodal = recover_nodal_stress(mesh, element_stress)
    areas = mesh.areas()

    n_el = mesh.n_elements
    eta2 = np.zeros(n_el)
    energy2 = 0.0
    for e, tri in enumerate(mesh.elements):
        s_nodes = nodal[tri]          # (3, 3) recovered stress at the 3 nodes
        s_h = element_stress[e]       # constant
        area = areas[e]
        # Integrate (s* - s_h)^T C^{-1} (s* - s_h) with the 3 midpoint rule.
        err2 = 0.0
        rec2 = 0.0
        for bary in _MID_BARY:
            s_star = bary @ s_nodes   # linear interpolation at the quad point
            d = s_star - s_h
            err2 += _MID_W * (d @ compliance @ d)
            rec2 += _MID_W * (s_star @ compliance @ s_star)
        eta2[e] = thickness * area * err2
        energy2 += thickness * area * rec2

    eta = np.sqrt(np.maximum(eta2, 0.0))
    global_error = float(np.sqrt(eta2.sum()))
    denom = np.sqrt(energy2 + eta2.sum())
    relative = global_error / denom if denom > 0 else 0.0
    return ErrorEstimate(eta, global_error, relative, nodal)
