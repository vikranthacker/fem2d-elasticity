"""Triangular mesh container, benchmark mesh generators and refinement.

The :class:`Mesh` stores node coordinates and 3-node triangle connectivity.  It
provides structured generators for the benchmark geometries used in the examples
(rectangle, L-shape, quarter plate with a circular hole), uniform (red)
refinement for convergence studies, and a conforming *longest-edge bisection*
(Rivara) refinement used by the adaptive solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .elements import triangle_area


@dataclass
class Mesh:
    """A 2-D triangular mesh.

    Attributes
    ----------
    nodes : (n_nodes, 2) float array
        Node coordinates.
    elements : (n_elems, 3) int array
        Triangle connectivity (node indices), ordered counter-clockwise.
    """

    nodes: np.ndarray
    elements: np.ndarray
    _edge_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float).reshape(-1, 2)
        self.elements = np.asarray(self.elements, dtype=np.int64).reshape(-1, 3)
        self._orient_ccw()

    # ------------------------------------------------------------------ basics
    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def n_dofs(self) -> int:
        return 2 * self.n_nodes

    def element_coords(self, e: int) -> np.ndarray:
        """Return the 3x2 coordinate array of element ``e``."""
        return self.nodes[self.elements[e]]

    def areas(self) -> np.ndarray:
        """Unsigned area of every element."""
        c = self.nodes[self.elements]
        x = c[:, :, 0]
        y = c[:, :, 1]
        return 0.5 * np.abs(
            (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
            - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
        )

    def _orient_ccw(self) -> None:
        """Ensure every triangle is oriented counter-clockwise (positive area)."""
        for i, tri in enumerate(self.elements):
            if triangle_area(self.nodes[tri]) < 0.0:
                self.elements[i, [1, 2]] = self.elements[i, [2, 1]]

    # -------------------------------------------------------------- topology
    def edges(self) -> np.ndarray:
        """Unique undirected edges as a sorted ``(n_edges, 2)`` int array."""
        e = self.elements
        pairs = np.vstack([e[:, [0, 1]], e[:, [1, 2]], e[:, [2, 0]]])
        pairs = np.sort(pairs, axis=1)
        return np.unique(pairs, axis=0)

    def boundary_edges(self) -> np.ndarray:
        """Edges that belong to exactly one element (the domain boundary)."""
        e = self.elements
        pairs = np.vstack([e[:, [0, 1]], e[:, [1, 2]], e[:, [2, 0]]])
        pairs = np.sort(pairs, axis=1)
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)
        return uniq[counts == 1]

    def boundary_nodes(self) -> np.ndarray:
        """Indices of nodes lying on the domain boundary."""
        return np.unique(self.boundary_edges())

    def nodes_where(self, predicate) -> np.ndarray:
        """Indices of nodes whose coordinates satisfy ``predicate(x, y)``."""
        x = self.nodes[:, 0]
        y = self.nodes[:, 1]
        return np.nonzero(predicate(x, y))[0]

    # -------------------------------------------------------------- refinement
    def uniform_refine(self, times: int = 1) -> "Mesh":
        """Red refinement: split every triangle into four similar triangles."""
        mesh = self
        for _ in range(times):
            mesh = _red_refine(mesh)
        return mesh

    def refine_marked(self, marked: Iterable[int]) -> "Mesh":
        """Conforming longest-edge bisection of the marked elements.

        Returns a new :class:`Mesh`.  The refinement propagates to neighbours
        as needed so that the result has no hanging nodes.
        """
        return _longest_edge_bisection(self, list(marked))

    # -------------------------------------------------------------- generators
    @classmethod
    def rectangle(cls, width: float, height: float, nx: int = 10, ny: int = 10,
                  x0: float = 0.0, y0: float = 0.0) -> "Mesh":
        """Structured triangulation of a rectangle ``[x0, x0+w] x [y0, y0+h]``."""
        xs = np.linspace(x0, x0 + width, nx + 1)
        ys = np.linspace(y0, y0 + height, ny + 1)
        gx, gy = np.meshgrid(xs, ys)
        nodes = np.column_stack([gx.ravel(), gy.ravel()])

        def nid(i, j):  # column i (x), row j (y)
            return j * (nx + 1) + i

        tris = []
        for j in range(ny):
            for i in range(nx):
                n00 = nid(i, j)
                n10 = nid(i + 1, j)
                n11 = nid(i + 1, j + 1)
                n01 = nid(i, j + 1)
                # Alternate the diagonal to avoid a biased mesh.
                if (i + j) % 2 == 0:
                    tris.append([n00, n10, n11])
                    tris.append([n00, n11, n01])
                else:
                    tris.append([n00, n10, n01])
                    tris.append([n10, n11, n01])
        return cls(nodes, np.array(tris, dtype=np.int64))

    @classmethod
    def lshape(cls, n: int = 8) -> "Mesh":
        """L-shaped domain ``[-1,1]^2 \\ (0,1]x[-1,0)`` (re-entrant corner at O).

        The classic singular benchmark: the solution has a stress singularity at
        the re-entrant corner, which adaptive refinement should resolve.
        ``n`` controls the resolution of each unit sub-square.
        """
        # Three unit squares tiling the L: top-left, top-right, bottom-left.
        blocks = [(-1.0, 0.0), (0.0, 0.0), (-1.0, -1.0)]
        meshes = [cls.rectangle(1.0, 1.0, n, n, bx, by) for bx, by in blocks]
        return _merge(meshes)

    @classmethod
    def plate_with_hole(cls, radius: float = 1.0, length: float = 4.0,
                        n_radial: int = 8, n_circ: int = 8) -> "Mesh":
        """Quarter model of a plate with a central circular hole.

        Models the first-quadrant quarter (``x >= 0, y >= 0``) of a square plate
        of half-width ``length`` with a quarter hole of ``radius`` at the origin.
        A transfinite map blends radially between the quarter arc and the outer
        square along rays of constant angle, so the two radial boundaries lie
        exactly on the symmetry axes ``y = 0`` (parameter ``u = 0``) and
        ``x = 0`` (parameter ``u = 1``).  Symmetry conditions complete the full
        plate; it is the standard Kirsch stress-concentration benchmark.
        """
        r, L = radius, length
        u = np.linspace(0.0, 1.0, n_circ + 1)      # angle:  0 -> pi/2
        v = np.linspace(0.0, 1.0, n_radial + 1)    # radial: hole -> outer square

        nodes = np.empty(((n_circ + 1) * (n_radial + 1), 2))

        def nid(i, j):
            return j * (n_circ + 1) + i

        for j, vv in enumerate(v):
            for i, uu in enumerate(u):
                ang = 0.5 * np.pi * uu                   # 0 (x-axis) -> pi/2 (y-axis)
                inner = np.array([r * np.cos(ang), r * np.sin(ang)])
                # Intersection of the ray at this angle with the outer square.
                if ang <= 0.25 * np.pi:
                    outer = np.array([L, L * np.tan(ang)])
                else:
                    outer = np.array([L / np.tan(ang), L])
                nodes[nid(i, j)] = (1.0 - vv) * inner + vv * outer

        tris = []
        for j in range(n_radial):
            for i in range(n_circ):
                n00 = nid(i, j)
                n10 = nid(i + 1, j)
                n11 = nid(i + 1, j + 1)
                n01 = nid(i, j + 1)
                tris.append([n00, n10, n11])
                tris.append([n00, n11, n01])
        return cls(nodes, np.array(tris, dtype=np.int64))


# --------------------------------------------------------------------- helpers
def _merge(meshes: list[Mesh], tol: float = 1e-9) -> Mesh:
    """Merge meshes, welding coincident nodes within ``tol``."""
    all_nodes = np.vstack([m.nodes for m in meshes])
    # Snap to a grid to find duplicates robustly.
    keys = np.round(all_nodes / tol).astype(np.int64)
    _, inverse, first = _unique_rows(keys)
    unique_nodes = all_nodes[first]

    elems = []
    offset = 0
    for m in meshes:
        elems.append(inverse[m.elements + offset])
        offset += m.n_nodes
    return Mesh(unique_nodes, np.vstack(elems))


def _unique_rows(a: np.ndarray):
    """Return (unique_rows, inverse_index, first_occurrence_index)."""
    order = np.lexsort(a.T[::-1])
    sorted_a = a[order]
    diff = np.ones(len(a), dtype=bool)
    diff[1:] = np.any(sorted_a[1:] != sorted_a[:-1], axis=1)
    unique_sorted = sorted_a[diff]
    group = np.cumsum(diff) - 1
    inverse = np.empty(len(a), dtype=np.int64)
    inverse[order] = group
    first = order[diff]
    return unique_sorted, inverse, first


def _red_refine(mesh: Mesh) -> Mesh:
    """Split every triangle into 4 by connecting edge midpoints."""
    nodes = [tuple(p) for p in mesh.nodes]
    index = {p: i for i, p in enumerate(nodes)}
    midpoint: dict[tuple[int, int], int] = {}

    def mid(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in midpoint:
            return midpoint[key]
        p = tuple((mesh.nodes[a] + mesh.nodes[b]) / 2.0)
        if p in index:
            idx = index[p]
        else:
            idx = len(nodes)
            nodes.append(p)
            index[p] = idx
        midpoint[key] = idx
        return idx

    new_elems = []
    for a, b, c in mesh.elements:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        new_elems.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
    return Mesh(np.array(nodes), np.array(new_elems, dtype=np.int64))


def _longest_edge_bisection(mesh: Mesh, marked: list[int]) -> Mesh:
    """Rivara longest-edge bisection producing a conforming mesh.

    Elements are bisected across their longest edge; a neighbour that does not
    share the same longest edge is refined first (recursively, via an explicit
    stack), which guarantees termination and conformity.
    """
    nodes: list[list[float]] = mesh.nodes.tolist()
    # Active element table: id -> (n0, n1, n2); None once replaced.
    elems: list[list[int] | None] = [list(t) for t in mesh.elements.tolist()]

    # edge (frozen sorted pair) -> set of active element ids using it.
    edge2elems: dict[tuple[int, int], set[int]] = {}

    def ekey(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def register(eid: int) -> None:
        t = elems[eid]
        assert t is not None
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge2elems.setdefault(ekey(a, b), set()).add(eid)

    def unregister(eid: int) -> None:
        t = elems[eid]
        assert t is not None
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge2elems[ekey(a, b)].discard(eid)

    for eid in range(len(elems)):
        register(eid)

    midpoint: dict[tuple[int, int], int] = {}

    def get_mid(a: int, b: int) -> int:
        key = ekey(a, b)
        if key in midpoint:
            return midpoint[key]
        pa, pb = nodes[a], nodes[b]
        idx = len(nodes)
        nodes.append([(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0])
        midpoint[key] = idx
        return idx

    def longest_edge(eid: int) -> tuple[int, int]:
        t = elems[eid]
        assert t is not None
        best_key = None
        best_len = -1.0
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            pa, pb = nodes[a], nodes[b]
            length = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
            key = ekey(a, b)
            # Deterministic tie-break so neighbours agree on "the" longest edge.
            if length > best_len + 1e-14 or (
                abs(length - best_len) <= 1e-14 and (best_key is None or key < best_key)
            ):
                best_len = length
                best_key = key
        return best_key  # type: ignore[return-value]

    def opposite_vertex(eid: int, edge: tuple[int, int]) -> int:
        t = elems[eid]
        assert t is not None
        for v in t:
            if v != edge[0] and v != edge[1]:
                return v
        raise RuntimeError("edge does not belong to element")

    def neighbour(eid: int, edge: tuple[int, int]) -> int | None:
        others = [x for x in edge2elems.get(edge, ()) if x != eid]
        return others[0] if others else None

    def split(eid: int, edge: tuple[int, int], m: int) -> None:
        """Replace element ``eid`` with two triangles across ``edge`` midpoint ``m``."""
        a, b = edge
        c = opposite_vertex(eid, edge)
        unregister(eid)
        elems[eid] = None
        for tri in ([a, m, c], [m, b, c]):
            # keep CCW orientation
            p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
            area2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
            if area2 < 0:
                tri = [tri[0], tri[2], tri[1]]
            new_id = len(elems)
            elems.append(tri)
            register(new_id)

    stack = [e for e in marked if elems[e] is not None]
    guard = 0
    max_iter = 10_000_000
    while stack:
        guard += 1
        if guard > max_iter:
            raise RuntimeError("longest-edge bisection failed to terminate")
        eid = stack[-1]
        if elems[eid] is None:
            stack.pop()
            continue
        edge = longest_edge(eid)
        nb = neighbour(eid, edge)
        if nb is not None and longest_edge(nb) != edge:
            stack.append(nb)      # refine neighbour first
            continue
        stack.pop()
        m = get_mid(*edge)
        split(eid, edge, m)
        if nb is not None and elems[nb] is not None:
            split(nb, edge, m)

    final = [t for t in elems if t is not None]
    return Mesh(np.array(nodes), np.array(final, dtype=np.int64))
