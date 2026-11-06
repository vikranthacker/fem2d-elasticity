"""felas -- a 2-D linear-elasticity finite element solver with adaptive mesh refinement.

A small, dependency-light (NumPy/SciPy) open-source finite element library for
plane-stress / plane-strain linear elasticity on triangular meshes.  It covers the
full workflow: mesh generation, element formulation (3-node constant-strain
triangles), global assembly, boundary conditions, solution, stress recovery,
Zienkiewicz-Zhu error estimation and adaptive (longest-edge bisection) mesh
refinement.

Typical use::

    from felas import Mesh, Material, ElasticityProblem
    mesh = Mesh.rectangle(8.0, 4.0, nx=16, ny=8)
    mat = Material(E=1000.0, nu=0.3, plane="stress")
    problem = ElasticityProblem(mesh, mat)
    ...
    u = problem.solve()
"""

from .materials import Material
from .mesh import Mesh
from .fem import ElasticityProblem
from .recovery import recover_nodal_stress, von_mises
from .error_estimator import zz_error_estimate
from .adaptivity import adaptive_solve, dorfler_marking

__all__ = [
    "Material",
    "Mesh",
    "ElasticityProblem",
    "recover_nodal_stress",
    "von_mises",
    "zz_error_estimate",
    "adaptive_solve",
    "dorfler_marking",
]

__version__ = "0.1.0"
