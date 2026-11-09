"""Adaptive mesh refinement driver (solve -> estimate -> mark -> refine)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .error_estimator import ErrorEstimate, zz_error_estimate
from .fem import ElasticityProblem
from .materials import Material
from .mesh import Mesh


def dorfler_marking(indicators: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Dörfler (bulk) marking.

    Return the indices of the smallest set of elements whose summed squared
    indicator reaches a fraction ``theta`` of the total.  ``theta`` near 1 marks
    almost everything (close to uniform), near 0 marks very few.
    """
    if not (0.0 < theta <= 1.0):
        raise ValueError("theta must be in (0, 1].")
    eta2 = np.asarray(indicators) ** 2
    order = np.argsort(eta2)[::-1]
    cumulative = np.cumsum(eta2[order])
    total = cumulative[-1] if len(cumulative) else 0.0
    if total <= 0.0:
        return np.array([], dtype=np.int64)
    k = int(np.searchsorted(cumulative, theta * total)) + 1
    return order[:k]


@dataclass
class AdaptiveStep:
    n_nodes: int
    n_elements: int
    n_dofs: int
    global_error: float
    relative_error: float


@dataclass
class AdaptiveResult:
    mesh: Mesh
    problem: ElasticityProblem
    displacement: np.ndarray
    estimate: ErrorEstimate
    history: list[AdaptiveStep] = field(default_factory=list)


def adaptive_solve(initial_mesh: Mesh,
                   build_problem: Callable[[Mesh], ElasticityProblem],
                   material: Material,
                   steps: int = 8,
                   theta: float = 0.5,
                   verbose: bool = False) -> AdaptiveResult:
    """Run an adaptive solve loop.

    Parameters
    ----------
    initial_mesh : Mesh
        Starting mesh.
    build_problem : callable
        ``build_problem(mesh) -> ElasticityProblem`` with all loads and boundary
        conditions applied.  It is called afresh on every refined mesh.
    material : Material
        Used to obtain the compliance matrix for the error estimate.
    steps : int
        Number of solve/refine iterations.
    theta : float
        Dörfler marking parameter.
    verbose : bool
        Print a one-line summary per step.
    """
    compliance = material.compliance_matrix()
    mesh = initial_mesh
    history: list[AdaptiveStep] = []
    problem = None
    u = None
    est = None

    for i in range(steps):
        problem = build_problem(mesh)
        u = problem.solve()
        _, stress = problem.element_strain_stress(u)
        est = zz_error_estimate(mesh, stress, compliance, material.thickness)

        step = AdaptiveStep(mesh.n_nodes, mesh.n_elements, mesh.n_dofs,
                            est.global_error, est.relative_error)
        history.append(step)
        if verbose:
            print(f"[{i:2d}] dofs={step.n_dofs:6d}  "
                  f"eta={step.global_error:.4e}  "
                  f"est.rel.err(ZZ)={step.relative_error:.4%}")

        if i == steps - 1:
            break
        marked = dorfler_marking(est.element_indicators, theta)
        if len(marked) == 0:
            break
        mesh = mesh.refine_marked(marked)

    return AdaptiveResult(mesh, problem, u, est, history)
