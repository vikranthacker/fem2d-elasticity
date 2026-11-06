"""Assembly and solution of the 2-D linear elasticity problem."""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .elements import body_force_vector, element_stiffness
from .materials import Material
from .mesh import Mesh


class ElasticityProblem:
    """Assemble and solve a plane linear-elasticity boundary value problem.

    Parameters
    ----------
    mesh : Mesh
        The triangular mesh.
    material : Material
        Isotropic linear-elastic material.
    """

    def __init__(self, mesh: Mesh, material: Material) -> None:
        self.mesh = mesh
        self.material = material
        self.D = material.constitutive_matrix()
        self.t = material.thickness

        n = mesh.n_dofs
        self.F = np.zeros(n)
        # Dirichlet constraints: dof index -> prescribed value.
        self._dirichlet: dict[int, float] = {}
        self._K: sp.csr_matrix | None = None
        self.u: np.ndarray | None = None

    # --------------------------------------------------------------- loads/BCs
    @staticmethod
    def _dofs(node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        node_ids = np.atleast_1d(np.asarray(node_ids, dtype=np.int64))
        return 2 * node_ids, 2 * node_ids + 1

    def fix_nodes(self, node_ids, ux: float | None = 0.0,
                  uy: float | None = 0.0) -> None:
        """Prescribe displacements at the given nodes.

        ``ux``/``uy`` of ``None`` leaves that component free.
        """
        dx, dy = self._dofs(node_ids)
        if ux is not None:
            for d in dx:
                self._dirichlet[int(d)] = float(ux)
        if uy is not None:
            for d in dy:
                self._dirichlet[int(d)] = float(uy)

    def prescribe(self, node_ids, values: Callable[[float, float], tuple[float, float]]
                  ) -> None:
        """Prescribe both displacement components from a field ``values(x, y)``."""
        node_ids = np.atleast_1d(np.asarray(node_ids, dtype=np.int64))
        for nid in node_ids:
            x, y = self.mesh.nodes[nid]
            ux, uy = values(x, y)
            self._dirichlet[int(2 * nid)] = float(ux)
            self._dirichlet[int(2 * nid + 1)] = float(uy)

    def add_point_load(self, node_id: int, fx: float = 0.0, fy: float = 0.0) -> None:
        self.F[2 * node_id] += fx
        self.F[2 * node_id + 1] += fy

    def add_body_force(self, bx: float, by: float) -> None:
        """Add a constant body force per unit volume over the whole domain."""
        for e in range(self.mesh.n_elements):
            coords = self.mesh.element_coords(e)
            fe = body_force_vector(coords, bx, by, self.t)
            dofs = self._element_dofs(e)
            np.add.at(self.F, dofs, fe)

    def add_edge_traction(self, edges: np.ndarray,
                          traction: Callable[[float, float], tuple[float, float]]
                          ) -> None:
        """Apply a surface traction on the given boundary ``edges``.

        ``edges`` is an ``(n, 2)`` array of node-index pairs.  ``traction(x, y)``
        returns ``(tx, ty)``.  The consistent nodal load for a 2-node edge with
        (possibly) varying traction is used.
        """
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        for a, b in edges:
            pa = self.mesh.nodes[a]
            pb = self.mesh.nodes[b]
            length = np.hypot(*(pb - pa))
            ta = np.array(traction(pa[0], pa[1]))
            tb = np.array(traction(pb[0], pb[1]))
            # Consistent load (exact for traction varying linearly along the edge).
            fa = self.t * length / 6.0 * (2.0 * ta + tb)
            fb = self.t * length / 6.0 * (ta + 2.0 * tb)
            self.F[2 * a:2 * a + 2] += fa
            self.F[2 * b:2 * b + 2] += fb

    def _element_dofs(self, e: int) -> np.ndarray:
        nodes = self.mesh.elements[e]
        dofs = np.empty(6, dtype=np.int64)
        dofs[0::2] = 2 * nodes
        dofs[1::2] = 2 * nodes + 1
        return dofs

    # --------------------------------------------------------------- assembly
    def assemble(self) -> sp.csr_matrix:
        """Assemble and cache the global stiffness matrix (unconstrained)."""
        n_el = self.mesh.n_elements
        rows = np.empty(36 * n_el, dtype=np.int64)
        cols = np.empty(36 * n_el, dtype=np.int64)
        data = np.empty(36 * n_el, dtype=float)

        for e in range(n_el):
            coords = self.mesh.element_coords(e)
            Ke, _, _ = element_stiffness(coords, self.D, self.t)
            dofs = self._element_dofs(e)
            idx = slice(36 * e, 36 * (e + 1))
            rows[idx] = np.repeat(dofs, 6)
            cols[idx] = np.tile(dofs, 6)
            data[idx] = Ke.ravel()

        K = sp.coo_matrix((data, (rows, cols)),
                          shape=(self.mesh.n_dofs, self.mesh.n_dofs)).tocsr()
        self._K = K
        return K

    # ---------------------------------------------------------------- solution
    def solve(self) -> np.ndarray:
        """Solve the constrained system; return displacements as ``(n_nodes, 2)``."""
        if self._K is None:
            self.assemble()
        K = self._K
        n = self.mesh.n_dofs

        if not self._dirichlet:
            raise ValueError("Problem is unconstrained (no Dirichlet BCs set); "
                             "the stiffness matrix is singular.")

        constrained = np.array(sorted(self._dirichlet), dtype=np.int64)
        g = np.array([self._dirichlet[int(d)] for d in constrained])
        free = np.setdiff1d(np.arange(n), constrained, assume_unique=False)

        Kff = K[free][:, free]
        Kfc = K[free][:, constrained]
        rhs = self.F[free] - Kfc @ g

        uf = spla.spsolve(Kff.tocsc(), rhs)

        u = np.zeros(n)
        u[free] = uf
        u[constrained] = g
        self.u = u
        return u.reshape(-1, 2)

    # --------------------------------------------------------------- recovery
    def element_strain_stress(self, u: np.ndarray | None = None
                              ) -> tuple[np.ndarray, np.ndarray]:
        """Constant strain and stress per element: two ``(n_elem, 3)`` arrays.

        Voigt order ``[xx, yy, xy]``.
        """
        if u is None:
            if self.u is None:
                raise ValueError("Solve the problem before requesting stresses.")
            u = self.u
        u = np.asarray(u).ravel()

        from .elements import strain_displacement_matrix
        n_el = self.mesh.n_elements
        strain = np.empty((n_el, 3))
        stress = np.empty((n_el, 3))
        for e in range(n_el):
            B, _ = strain_displacement_matrix(self.mesh.element_coords(e))
            ue = u[self._element_dofs(e)]
            eps = B @ ue
            strain[e] = eps
            stress[e] = self.D @ eps
        return strain, stress

    def strain_energy(self, u: np.ndarray | None = None) -> float:
        """Total strain energy ``0.5 * integral(stress : strain)``."""
        strain, stress = self.element_strain_stress(u)
        areas = self.mesh.areas()
        return 0.5 * self.t * float(np.sum(areas * np.sum(stress * strain, axis=1)))
