import zlib
from dataclasses import dataclass
from typing import override


@dataclass(frozen=True, kw_only=True)
class CaldeiraLeggettConfig:
    """The convergence parameters of the Caldeira-Leggett master equation.

    The basis is set by friction_tol, max_energy and plane_wave_energy, and the
    numerical solution by the tolerances of the linear solves and of the time
    propagation. Energies are in units of k_B T.
    """

    friction_tol: float = 5e-3
    """The relative error allowed in the friction, which sets the number of
    periods of the potential in the simulation cell.

    The smallest q is a reciprocal lattice vector of the cell, and the
    Lindblad operators e^{+-i q x} / q stand in for x only where q x is small.
    The error is taken at the coherence length of a free particle,
    hbar / sqrt(m k_B T).
    """
    max_energy: float = 12.0
    """The energy above the saddle of the potential, or the ground state if it is higher, up to which bands are kept."""
    plane_wave_energy: float | None = None
    """The kinetic energy up to which plane waves are kept to build the bands.

    By default, 2.5 times the largest kinetic energy that the highest band can
    classically have, its height above the minimum of the potential.
    """
    linear_rtol: float = 1e-12
    """The relative residual of each linear solve (GMRES), or its floor from round off if that is higher.

    The steady state relaxes slowly along some modes, such as hops between
    the sites of the cell, so its error can be much larger than its residual:
    for hydrogen on Cu(111) at 300 times its mass, a residual of 1e-8 leaves
    D wrong by 2e-3, and of 1e-12 by 1e-7.
    """
    linear_restart: int = 30
    """The number of GMRES iterations between restarts."""
    linear_maxiter: int = 50
    """The largest number of GMRES restarts in each pass."""
    linear_inner_rtol: float = 1e-4
    """The factor by which each pass of GMRES reduces the residual."""
    linear_refinements: int = 20
    """The largest number of passes of GMRES, each from the true residual."""
    ode_rtol: float = 1e-8
    """The relative tolerance of the adaptive time steps of the ISF."""
    ode_atol: float = 1e-10
    """The absolute tolerance of the adaptive time steps of the ISF."""

    @override
    def __hash__(self) -> int:
        # Stable between runs, so it can be used in cache paths
        return zlib.crc32(repr(self).encode("utf-8"))


DEFAULT_CONFIG = CaldeiraLeggettConfig()
