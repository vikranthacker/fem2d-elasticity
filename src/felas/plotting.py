"""Matplotlib helpers for meshes, fields and deformed shapes.

These are convenience wrappers; matplotlib is only imported when a plotting
function is actually called, so the solver has no hard dependency on it.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh


def _triangulation(mesh: Mesh, displacement: np.ndarray | None = None,
                   scale: float = 1.0):
    import matplotlib.tri as mtri
    pts = mesh.nodes.copy()
    if displacement is not None:
        pts = pts + scale * np.asarray(displacement).reshape(-1, 2)
    return mtri.Triangulation(pts[:, 0], pts[:, 1], mesh.elements)


def plot_mesh(mesh: Mesh, ax=None, color="0.3", lw=0.5, **kwargs):
    """Draw the mesh wireframe."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    tri = _triangulation(mesh)
    ax.triplot(tri, color=color, lw=lw, **kwargs)
    ax.set_aspect("equal")
    return ax


def plot_nodal_field(mesh: Mesh, nodal_values: np.ndarray, ax=None,
                     cmap="viridis", title=None, show_mesh=False, colorbar=True):
    """Smooth (Gouraud) plot of a continuous nodal scalar field."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    tri = _triangulation(mesh)
    tpc = ax.tripcolor(tri, np.asarray(nodal_values), shading="gouraud", cmap=cmap)
    if show_mesh:
        ax.triplot(tri, color="k", lw=0.2, alpha=0.3)
    if colorbar:
        ax.figure.colorbar(tpc, ax=ax, shrink=0.8)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax


def plot_element_field(mesh: Mesh, element_values: np.ndarray, ax=None,
                       cmap="magma", title=None, colorbar=True):
    """Flat-shaded plot of a per-element scalar (e.g. the error indicator)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    tri = _triangulation(mesh)
    tpc = ax.tripcolor(tri, facecolors=np.asarray(element_values),
                       cmap=cmap, edgecolors="k", linewidth=0.15)
    if colorbar:
        ax.figure.colorbar(tpc, ax=ax, shrink=0.8)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax


def plot_deformed(mesh: Mesh, displacement: np.ndarray, scale: float = 1.0,
                  nodal_values: np.ndarray | None = None, ax=None,
                  cmap="viridis", title=None):
    """Plot the deformed mesh, optionally coloured by a nodal field."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.triplot(_triangulation(mesh), color="0.75", lw=0.3)
    tri_def = _triangulation(mesh, displacement, scale)
    if nodal_values is not None:
        tpc = ax.tripcolor(tri_def, np.asarray(nodal_values),
                           shading="gouraud", cmap=cmap)
        ax.figure.colorbar(tpc, ax=ax, shrink=0.8)
    else:
        ax.triplot(tri_def, color="C0", lw=0.5)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax
