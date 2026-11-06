"""Nodal stress recovery and derived stress measures.

The linear-triangle solution gives a stress field that is *constant* within each
element and discontinuous across element boundaries.  Averaging the element
stresses to the nodes produces a smoother, continuous field that is both nicer
for plotting and the basis of the Zienkiewicz-Zhu error estimator.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh


def recover_nodal_stress(mesh: Mesh, element_stress: np.ndarray) -> np.ndarray:
    """Area-weighted averaging of element stresses to nodes.

    Parameters
    ----------
    mesh : Mesh
    element_stress : (n_elem, 3) array
        Constant stress per element in Voigt order ``[xx, yy, xy]``.

    Returns
    -------
    (n_nodes, 3) array
        Recovered continuous nodal stress field.
    """
    element_stress = np.asarray(element_stress)
    areas = mesh.areas()
    nodal = np.zeros((mesh.n_nodes, 3))
    weight = np.zeros(mesh.n_nodes)
    for e, tri in enumerate(mesh.elements):
        w = areas[e]
        nodal[tri] += w * element_stress[e]
        weight[tri] += w
    weight[weight == 0.0] = 1.0
    return nodal / weight[:, None]


def von_mises(stress: np.ndarray) -> np.ndarray:
    """Von Mises equivalent stress for Voigt stress ``[xx, yy, xy]`` (plane).

    Accepts a single stress or an ``(..., 3)`` array and returns the scalar(s).
    """
    stress = np.asarray(stress)
    sxx = stress[..., 0]
    syy = stress[..., 1]
    sxy = stress[..., 2]
    return np.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)
