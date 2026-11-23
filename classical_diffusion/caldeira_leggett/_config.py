from dataclasses import dataclass
from typing import override

from classical_diffusion.util import stable_hash


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
    """The error allowed from the basis of bands and plane waves, in D relative to D and in the ISF.

    No cutoff is set by hand: the bands and plane waves are raised until
    the change of D over three larger bases shows that D converges, and
    bounds its error below it (`_caldeira_leggett._get_basis`), or a
    RuntimeError is raised; and for each ISF until rho e^{-i dK x} loses less
    than it above the highest band (`_caldeira_leggett._get_sector`).
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
        return stable_hash(self)


DEFAULT_CONFIG = CaldeiraLeggettConfig()


@dataclass(frozen=True, kw_only=True)
class EvolutionConfig:
    """The convergence of an ISF or mean square displacement (`get_reduced_evolution`), and of the slow modes (`get_slow_modes`)."""

    stop: float = 1e-6
    """The error allowed in the measurement at each time, relative to its value at t = 0.

    The evolution is by two-sided Lanczos, which stops once the measurement
    at the times changes by less than this over two blocks of steps, and
    its Laplace transform at 1 / span agrees with one found directly
    (`get_reduced_evolution`). Below its round off floor (1e-16 to 2e-9 on
    H, Li and Na on Cu(111)) it warns and returns what it reached.
    """
    mode_rtol: float = 1e-6
    """The relative residual |L r - lambda r| / |lambda| of each slow mode."""


DEFAULT_EVOLUTION_CONFIG = EvolutionConfig()
