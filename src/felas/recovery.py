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


def recover_nodal_stress_spr(mesh: Mesh, element_stress: np.ndarray) -> np.ndarray:
    """Superconvergent Patch Recovery (SPR) of the nodal stress.

    For each interior assembly node, a *linear* polynomial ``p(x, y) = a0 + a1 x
    + a2 y`` is fitted, component by component and in the least-squares sense, to
    the element stresses sampled at the centroids of the patch of elements
    surrounding the node (the centroid is the superconvergent sampling point of a
    constant-strain triangle).  The recovered nodal value is the polynomial
    evaluated at the node.  Boundary nodes---whose one-sided patches are
    rank-deficient or biased---take the value extrapolated from the polynomials
    of their adjacent interior nodes; where no such interior patch exists the
    method falls back to area-weighted averaging (:func:`recover_nodal_stress`).

    This is the recovery of Zienkiewicz & Zhu (1992, Part I); it is asymptotically
    exact under superconvergence and is the stronger sibling of plain nodal
    averaging.

    Parameters
    ----------
    mesh : Mesh
    element_stress : (n_elem, 3) array
        Constant stress per element in Voigt order ``[xx, yy, xy]``.

    Returns
    -------
    (n_nodes, 3) array
        SPR-recovered continuous nodal stress field.
    """
    element_stress = np.asarray(element_stress, dtype=float)
    nodes = mesh.nodes
    elements = mesh.elements
    n_nodes = mesh.n_nodes

    centroids = nodes[elements].mean(axis=1)          # (n_elem, 2)

    node_elems: list[list[int]] = [[] for _ in range(n_nodes)]
    for e, tri in enumerate(elements):
        for v in tri:
            node_elems[v].append(e)

    boundary = set(int(b) for b in mesh.boundary_nodes())

    # Start from area-weighted averaging as a universal fallback.
    recovered = recover_nodal_stress(mesh, element_stress).copy()

    # Interior patches: least-squares linear fit in node-local coordinates.
    poly: dict[int, np.ndarray] = {}                  # node -> (3, 3) coeffs [a0;a1;a2] x comp
    for n in range(n_nodes):
        if n in boundary:
            continue
        els = node_elems[n]
        if len(els) < 3:
            continue
        P = centroids[els]
        dx = P[:, 0] - nodes[n, 0]
        dy = P[:, 1] - nodes[n, 1]
        A = np.column_stack([np.ones(len(els)), dx, dy])   # (m, 3)
        if np.linalg.matrix_rank(A) < 3:
            continue
        coeffs, *_ = np.linalg.lstsq(A, element_stress[els], rcond=None)  # (3, 3)
        poly[n] = coeffs
        recovered[n] = coeffs[0]                       # value at the node (local origin)

    # Boundary nodes: extrapolate from adjacent interior patches.
    for n in boundary:
        neigh = set()
        for e in node_elems[n]:
            for v in elements[e]:
                if v != n and v in poly:
                    neigh.add(int(v))
        if not neigh:
            continue                                   # keep the averaging fallback
        acc = np.zeros(3)
        for v in neigh:
            c = poly[v]
            d = nodes[n] - nodes[v]
            acc += c[0] + c[1] * d[0] + c[2] * d[1]
        recovered[n] = acc / len(neigh)

    return recovered


def von_mises(stress: np.ndarray) -> np.ndarray:
    """Von Mises equivalent stress for Voigt stress ``[xx, yy, xy]`` (plane).

    Accepts a single stress or an ``(..., 3)`` array and returns the scalar(s).
    """
    stress = np.asarray(stress)
    sxx = stress[..., 0]
    syy = stress[..., 1]
    sxy = stress[..., 2]
    return np.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)
