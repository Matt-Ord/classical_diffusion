import zlib
from dataclasses import dataclass
from typing import override


@dataclass(frozen=True, kw_only=True)
class CaldeiraLeggettConfig:
    """The convergence parameters of the steady state of the Caldeira-Leggett master equation.

    The basis is set by friction_tol (the cell) and basis_tol (the bands and
    plane waves), and the linear solves by linear_rtol. Every function
    starts from the steady state, so every function takes these, and the
    steady state is cached by them. The
    time steps of an ISF are set separately (`EvolutionConfig`). Energies are in
    units of k_B T.
    """

    friction_tol: float = 5e-3
    """The relative error allowed in the friction, which sets the number of
    periods of the potential in the simulation cell.

    The smallest q is a reciprocal lattice vector of the cell, and the
    Lindblad operators e^{+-i q x} / q stand in for x only where q x is small.
    The error is taken at the coherence length of a free particle,
    hbar / sqrt(m k_B T).
    """
    basis_tol: float = 1e-4
    """The error allowed from the basis of bands and plane waves, in D relative to itself and in the ISF.

    The bands and plane waves are chosen from the weight the states leave at
    the edge of the basis (`_caldeira_leggett._get_basis`), so no cutoff is
    set by hand: the error of D from bands up to an energy is about the
    population of the steady state in the highest band, times
    D_free / D (D_free = k_B T / m gamma), and that of the ISF at a large dK
    the part of rho e^{-i dK x} kicked above the highest band. The estimate
    is an upper bound on every system it was measured on, where the error
    was 1/300 to 1/3 of it.
    """
    linear_rtol: float = 1e-12
    """The relative residual of each linear solve (BiCGSTAB), or its floor from round off if that is higher.

    The steady state relaxes slowly along some modes, such as hops between
    the sites of the cell, so its error can be much larger than its residual:
    for hydrogen on Cu(111) at 300 times its mass, a residual of 1e-8 leaves
    D wrong by 2e-3, and of 1e-12 by 1e-7.
    """

    @override
    def __hash__(self) -> int:
        # Stable between runs, so it can be used in cache paths
        return zlib.crc32(repr(self).encode("utf-8"))


DEFAULT_CONFIG = CaldeiraLeggettConfig()


@dataclass(frozen=True, kw_only=True)
class EvolutionConfig:
    """The convergence of an ISF or mean square displacement (`get_reduced_evolution`), and of the slow modes (`get_slow_modes`).

    Each sets an accuracy. Which modes are found to save time steps is
    decided by fixed rules (`_evolution._PROBE`, `_CHECK`, `_COST_FACTOR`).
    """

    stop: float = 1e-6
    """The norm of what is left of X_0, relative to |X_0|, below which it is no longer stepped."""
    step_rtol: float = 1e-8
    """The relative tolerance of each step of the Krylov exponential stepper."""
    step_atol: float = 1e-9
    """The absolute tolerance of each step, relative to |X_0| (as the stop is).

    The ISF has |X_0| = 0.2-0.3 in the eigenvectors of K (H, Li and Na on Cu(111)),
    so 1e-9 is close to the absolute 1e-10 it replaces; an MSD has |X_0| = 5e-3 (H in
    2D), for which 1e-10 relative cost 11% more applications of L than 1e-9 and
    gave the same errors.
    """
    mode_rtol: float = 1e-6
    """The relative residual |L r - lambda r| / |lambda| of each slow mode."""


DEFAULT_EVOLUTION_CONFIG = EvolutionConfig()
