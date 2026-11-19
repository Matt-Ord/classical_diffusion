import zlib
from dataclasses import dataclass
from typing import override


@dataclass(frozen=True, kw_only=True)
class CaldeiraLeggettConfig:
    """The convergence parameters of the steady state of the Caldeira-Leggett master equation.

    The basis is set by friction_tol, max_energy and plane_wave_energy, and the
    linear solves by the rest. Every function starts from the steady state, so
    every function takes these, and the steady state is cached by them. The
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
    max_energy: float = 12.0
    """The energy above the saddle of the potential, or the ground state if it is higher, up to which bands are kept: the states which are occupied."""
    extra_bands: int = 6
    """extra_bands**n_dim more bands are kept above max_energy: the states which complete the basis.

    The bands above the occupied ones are never occupied, but they are the
    intermediate states of the Lamb shift and of the response to the twist
    which gives D, whose sum restores the bare mass (the f-sum rule). D is the
    small difference of two terms which each depend on it, so it needs a
    number of them which grows with the dimension, not an energy: hydrogen on
    Cu(111) needs about 8 bands in 1D and 43 in 2D. With bands to 12 k_B T and
    6^2 more (52), D for hydrogen in 2D is within 1e-5 of converged, where
    with 4^2 or 5^2 more it is 6e-4 and 9e-4 high, and with bands to 16 k_B T
    and none more 0.5%: the error is erratic, as single bands enter. Lithium
    in 2D has 139 bands within 12 k_B T, for D 1.6e-4 low, and 175 with
    6^2 more; in 1D every system tested is within 1e-7 with 6 more. The high
    bands cannot be eliminated instead, as no gap separates them from those
    kept.
    """
    plane_wave_energy: float | None = None
    """The kinetic energy up to which plane waves are kept to build the bands.

    By default, 2.5 times the largest kinetic energy that the highest band can
    classically have, its height above the minimum of the potential.
    """
    linear_rtol: float = 1e-12
    """The relative residual of each linear solve (BiCGSTAB), or its floor from round off if that is higher.

    The steady state relaxes slowly along some modes, such as hops between
    the sites of the cell, so its error can be much larger than its residual:
    for hydrogen on Cu(111) at 300 times its mass, a residual of 1e-8 leaves
    D wrong by 2e-3, and of 1e-12 by 1e-7.
    """
    linear_maxiter: int = 1000
    """The largest number of BiCGSTAB iterations in each pass."""
    linear_inner_rtol: float = 1e-4
    """The factor by which each pass of BiCGSTAB reduces the residual."""
    linear_refinements: int = 20
    """The largest number of passes of BiCGSTAB, each from the true residual."""

    @override
    def __hash__(self) -> int:
        # Stable between runs, so it can be used in cache paths
        return zlib.crc32(repr(self).encode("utf-8"))


DEFAULT_CONFIG = CaldeiraLeggettConfig()


@dataclass(frozen=True, kw_only=True)
class EvolutionConfig:
    """The convergence of an ISF or mean square displacement (`get_reduced_evolution`)."""

    stop: float = 1e-6
    """The norm of what is left of X_0, relative to |X_0|, below which it is no longer stepped."""
    step_rtol: float = 1e-8
    """The relative tolerance of each step of the Krylov exponential stepper."""
    step_atol: float = 1e-10
    """The absolute tolerance of each step."""
    mode_rtol: float = 1e-6
    """The relative residual |L r - lambda r| / |lambda| of each slow mode."""
    probe: float = 2.0
    """The time stepped before the slow modes are found, in units of 1 / friction."""
    cost_factor: float = 3.0
    """The margin on the cost of a slow mode, estimated as the mean applications of L per mode found so far.

    A mode is found if cost_factor times that is below the stepping it saves
    (`CostCut`), or, once the remainder is stepped, below the stepping it
    still needs (`get_reduced_evolution`). The first mode found, the hop,
    costs most. 1.5 rather than 3 changes the total applications of L by
    +0.5% over H, Li and Na on Cu(111) in 1D and H in 2D (2 seeds).
    """


DEFAULT_EVOLUTION_CONFIG = EvolutionConfig()
