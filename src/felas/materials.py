"""Linear-elastic isotropic material and plane constitutive matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    """Isotropic linear-elastic material for 2-D elasticity.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio (must satisfy -1 < nu < 0.5).
    plane : {"stress", "strain"}
        Plane-stress (thin plate) or plane-strain (thick body) assumption.
    thickness : float
        Out-of-plane thickness (used for load/stiffness scaling).  For
        plane strain this is conventionally 1.0.
    """

    E: float
    nu: float
    plane: str = "stress"
    thickness: float = 1.0

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError("Young's modulus E must be positive.")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError("Poisson's ratio nu must lie in (-1, 0.5).")
        if self.plane not in ("stress", "strain"):
            raise ValueError("plane must be 'stress' or 'strain'.")
        if self.thickness <= 0:
            raise ValueError("thickness must be positive.")

    def constitutive_matrix(self) -> np.ndarray:
        """Return the 3x3 constitutive matrix D mapping strain -> stress.

        Strain/stress are in Voigt form ``[xx, yy, xy]`` with engineering
        shear strain ``gamma_xy = 2*eps_xy``.
        """
        E, nu = self.E, self.nu
        if self.plane == "stress":
            c = E / (1.0 - nu * nu)
            return c * np.array(
                [[1.0, nu, 0.0],
                 [nu, 1.0, 0.0],
                 [0.0, 0.0, (1.0 - nu) / 2.0]]
            )
        # plane strain
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return c * np.array(
            [[1.0 - nu, nu, 0.0],
             [nu, 1.0 - nu, 0.0],
             [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]]
        )

    def compliance_matrix(self) -> np.ndarray:
        """Inverse of the constitutive matrix (stress -> strain)."""
        return np.linalg.inv(self.constitutive_matrix())
