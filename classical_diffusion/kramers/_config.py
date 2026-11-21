from dataclasses import dataclass
from typing import override

from classical_diffusion.util import stable_hash


@dataclass(frozen=True, kw_only=True)
class KramersConfig:
    """The convergence of the classical solvers, of the Kramers equation and of its overdamped limit."""

    tol: float = 1e-6
    """The absolute error of the ISF, I(0) = 1, and the relative error of the diffusion constant and mean square displacement."""
    basis_tol: float = 1e-8
    """The norm of the components of the equilibrium amplitude e^{-V / 2} left out of the plane waves, relative to its norm (`get_plane_waves`)."""

    @override
    def __hash__(self) -> int:
        return stable_hash(self)


DEFAULT_CONFIG = KramersConfig()
