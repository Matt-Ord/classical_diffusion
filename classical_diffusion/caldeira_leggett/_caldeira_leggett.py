# cspell: ignore rint Hermiticity polyfit
import dataclasses
import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from classical_diffusion._results import (
    RmsResult,
    SlowModes,
    check_mode_count,
    fold_times,
    get_selection,
    get_unit_direction,
)
from classical_diffusion.caldeira_leggett._bands import (
    Bands,
    get_bands,
    get_basis_map,
    get_edge_weight,
    get_energies,
    get_next_cutoff,
    get_shift_operator,
    get_shifted_kappa,
    get_velocity,
)
from classical_diffusion.caldeira_leggett._config import (
    DEFAULT_CONFIG,
    DEFAULT_EVOLUTION_CONFIG,
    CaldeiraLeggettConfig,
    EvolutionConfig,
)
from classical_diffusion.caldeira_leggett._evolution import get_measurement
from classical_diffusion.caldeira_leggett._linear import get_eigen_liouvillian, to_eigen
from classical_diffusion.caldeira_leggett._liouvillian import (
    Operators,
    get_couplings,
    get_liouvillian,
    get_offset_operators,
    get_operators,
    get_twist_derivative,
)
from classical_diffusion.caldeira_leggett._model import Model, build_model
from classical_diffusion.caldeira_leggett._slow_modes import find_slow_modes
from classical_diffusion.caldeira_leggett._solver import (
    get_diffusion_coefficient,
    get_isf,
    get_rms,
)
from classical_diffusion.caldeira_leggett._solver import (
    get_steady_state as get_band_steady_state,
)
from classical_diffusion.simulation import IsfResult
from classical_diffusion.util import cached, timed

if TYPE_CHECKING:
    from classical_diffusion.caldeira_leggett._liouvillian import Liouvillian
    from classical_diffusion.caldeira_leggett._model import (
        ComplexArray,
        FloatArray,
    )
    from classical_diffusion.langevin._system import PeriodicSystem


@dataclass(frozen=True, kw_only=True)
class SteadyState:
    """The steady state of the master equation of a periodic system, with the bands it is found in, and its diffusion constant along the first axis.

    The state is diagonal in the crystal momentum. The bands are only defined
    up to a phase, and up to a rotation where they are degenerate, which can
    differ each time they are found, so the state keeps its own.
    """

    model: Model
    bands: Bands
    density: ComplexArray
    """The density matrix in the basis of the bands, (n_kappa, n_bands, n_bands)."""
    diffusion: float
    """D along the first axis (`get_diffusion_coefficient`), in the natural units."""
    response: ComplexArray
    """The response X_1 = L_0^+ L_1 rho_ss from which D is found, which the mean square displacement also needs."""
    error: float
    """The error of D from the basis, relative to D, measured by the change of D in larger bases (`_get_basis`)."""


def _get_steady_state_path(
    system: PeriodicSystem, *, config: CaldeiraLeggettConfig = DEFAULT_CONFIG
) -> Path:
    return Path(f"caldeira_leggett_steady_state_{hash(system)}_{hash(config)}.npz")


@cached(_get_steady_state_path, default_call="load_or_call_uncached")
@timed
def get_steady_state(
    system: PeriodicSystem, *, config: CaldeiraLeggettConfig = DEFAULT_CONFIG
) -> SteadyState:
    """Get the steady state of the master equation of a periodic system, in a basis of bands chosen to give D to config.basis_tol (`_get_basis`).

    Every ISF, and the diffusion constant, starts from the steady state. By
    default they load it from the cache, if it is there, but do not save it.
    Call ``get_steady_state.call_cached`` once, with the same configuration,
    to save it for them.
    """
    return _get_basis(system, config, None, None)


_START = 12.0
"""The first cutoff of the bands, in k_B T above the saddle or the ground state (`Model.cutoff`).

Each try costs a steady state and a linear solve, and the cutoffs chosen
for hydrogen, lithium and sodium on Cu(111) in 1D and 2D are 12-30 k_B T.
"""
_PLANE_WAVE_FACTOR = 2.5
"""The first plane wave factor (`Model.plane_wave_factor`), raised by 1.5 if the plane waves are what limit D."""
_STEP = 10.0
"""The least factor by which each raise aims to lower the population at the edge of the basis (`_get_basis`)."""
_RATE = 0.5
"""The largest ratio of the changes of D over the last two raises at which the convergence of D is trusted, and the least factor by which the last raise is then taken to have lowered the error of D (`_get_basis`)."""
_CONTRACTION = 3.0
"""The least factor by which each of the last two raises must have lowered the population at the edge of the basis for the changes of D to measure its convergence (`_get_basis`)."""
_DRIFT = 10.0
"""The most by which the change of D may fall faster than the population at the edge, over the last two raises, for the changes to measure the convergence of D (`_get_basis`).

The error of D falls with the population at the edge, at a ratio which
drifts by at most 3 as the population falls by 4 decades (`~/cd-agents/adaptive`).
A raise which adds only one of a pair of nearly degenerate bands lowers
the population at the edge but leaves D, and its error, nearly unchanged:
for lithium at 100 K over a barrier of 130 meV, D changed by 6e-8 from 21
to 22 bands where its error was 9e-7 (and 1e-8 at 23 bands).
"""
_MAX_TRIES = 20


def _get_edge(model: Model, bands: Bands, density: ComplexArray) -> tuple[float, float]:
    """Get the populations at the edge of the bands and of the plane waves.

    D is a small difference of terms of order D_free = k_B T / m gamma, each
    of which needs the bands above a state to complete it (the f-sum rule).
    The highest band has none, so its population leaves an error of about
    itself times D_free in D, as does the weight of each band on the
    outermost plane waves, which do not resolve it.
    """
    population = np.real(np.einsum("kaa->ka", density))
    top = abs(float(population[:, -1].sum()))
    return top, abs(float(np.sum(population * get_edge_weight(model, bands))))


def _get_growth(energies: FloatArray, population: FloatArray, factor: float) -> float:
    """Get how far to raise the cutoff (k_B T) to lower the population of the highest band by a factor.

    The population falls about exponentially with the energy of the band,
    as exp(-E) in a Gibbs state, but more slowly where the friction mixes
    the bands: about exp(-0.44 E) for hydrogen on Cu(111). The rate is fitted
    over the bands within 10 k_B T of the highest.
    """
    lowest = energies.min(axis=0)
    weight = np.abs(population.sum(axis=0))
    near = (lowest > lowest[-1] - 10) & (weight > 0)
    rate = 1.0
    if np.count_nonzero(near) > 2:  # ruff:ignore[magic-value-comparison]
        rate = -float(np.polyfit(lowest[near], np.log(weight[near]), 1)[0])
    return max(np.log(factor) / float(np.clip(rate, 0.5, 2.0)), 2.0)


def _get_basis(  # ruff:ignore[too-many-locals]
    system: PeriodicSystem,
    config: CaldeiraLeggettConfig,
    direction: tuple[float, ...] | None,
    start: SteadyState | None,
) -> SteadyState:
    """Get the steady state in bands whose error in D along the direction (by default the first axis) the change of D in larger bases shows to be below config.basis_tol.

    From a cutoff of _START above the saddle (or the basis of start), each
    try raises the basis and finds the steady state and D. The tries are
    checked on D itself, not on a model of its error. With three tries whose
    D change by c_1 and then c_2 (relative to D), D converges at the rate
    r = c_2 / c_1 if r <= _RATE, each raise lowered the population at the
    edge (`_get_edge`) by _CONTRACTION, and D did not converge more than
    _DRIFT times faster than that population fell (a raise which adds one
    of a pair of nearly degenerate bands leaves D unchanged, which is not
    convergence). The last raise is then taken to have lowered the error by
    _RATE or more (it lowered it by 0.27 where r was 0.005, for hydrogen at
    100 K), so the error of the middle D is at most c_2 / (1 - _RATE), and
    that of each try before it at most that plus its distance to the middle
    D. The smallest basis within the tolerance is returned with its error.
    Each raise lowers the population at the edge by _STEP, or by as much as
    the change of D per unit of population at the edge says the next basis
    needs to be within a quarter of the tolerance. The plane waves are
    raised if they hold more of the population at the edge, and otherwise
    the cutoff (`_get_growth`).

    Raises
    ------
    RuntimeError
        If no basis is confirmed within _MAX_TRIES tries.
    """
    cutoff = _START if start is None else start.model.cutoff
    factor = _PLANE_WAVE_FACTOR if start is None else start.model.plane_wave_factor
    tries: list[tuple[SteadyState, float]] = []
    for _ in range(_MAX_TRIES):
        if not tries and start is not None:
            model, bands, density = start.model, start.bands, start.density
            energies = get_energies(model)
        else:
            model = build_model(system, config, cutoff=cutoff, plane_wave_factor=factor)
            energies = get_energies(model)
            bands = get_bands(model, energies)
        operators, derivative, _ = _get_derivative(model, bands, direction)
        if tries or start is None:
            density = get_band_steady_state(operators, bands.energies, config)
        diffusion, response = get_diffusion_coefficient(
            operators, derivative, density, config
        )
        top, plane_waves = _get_edge(model, bands, density)
        tries.append(
            (
                SteadyState(
                    model=model,
                    bands=bands,
                    density=density,
                    diffusion=diffusion,
                    response=response,
                    error=np.inf,
                ),
                top + plane_waves,
            )
        )
        changes = [
            abs(b.diffusion - a.diffusion) / max(abs(diffusion), 1e-300)
            for (a, _), (b, _) in itertools.pairwise(tries[-3:])
        ]
        target = _STEP
        if len(changes) == 2:  # ruff:ignore[magic-value-comparison]
            first, second = changes
            edges = [edge for _, edge in tries[-3:]]
            if (
                second <= _RATE * first
                and edges[0] >= _CONTRACTION * edges[1] >= _CONTRACTION**2 * edges[2]
                and second * edges[0] >= first * edges[1] / _DRIFT
            ):
                # The error of the middle D, and of each try before it by its
                # distance to the middle D
                middle = tries[-2][0].diffusion
                bound = second / (1 - _RATE)
                for state, _ in tries[:-1]:
                    error = bound + abs(state.diffusion - middle) / abs(diffusion)
                    if error <= config.basis_tol:
                        return dataclasses.replace(state, error=error)
        if changes:
            # The error of this D, if it is the change per unit of edge
            error = changes[-1] * tries[-1][1] / max(tries[-2][1], 1e-300)
            target = max(_STEP, 4 * error / config.basis_tol)
        if plane_waves > top:
            factor *= 1.5
            continue
        cutoff = get_next_cutoff(
            model,
            energies,
            cutoff
            + _get_growth(
                bands.energies, np.real(np.einsum("kaa->ka", density)), target
            ),
        )
    msg = (
        f"No basis confirmed D to {config.basis_tol:.0e} within {_MAX_TRIES} tries:"
        " D did not converge, which it cannot if it is below the round off of"
        " the linear solves (linear_rtol D_free)"
    )
    raise RuntimeError(msg)


@dataclass(frozen=True, kw_only=True)
class _Sector:
    """The ISF at dK as Tr[O^dagger exp(L t) X_0], in the natural units of the model."""

    model: Model
    liouvillian: Liouvillian
    initial: ComplexArray
    """X_0."""
    observable: ComplexArray
    """O."""
    delta_k: FloatArray
    """dK, in the natural units."""
    steady_state: ComplexArray | None
    """The steady state in the bands of the sector, if dK is on the reciprocal lattice: the sector is then that of dK = 0, whose Liouvillian preserves the trace."""


_LOST_RATE = 0.25
"""The rate (per k_B T of the cutoff) at which the part of rho e^{-i dK x} kicked above the highest band falls, before two cutoffs measure it.

For hydrogen on Cu(111) in 1D at 2 |G| it falls as exp(-0.27 E).
"""


def _raise_cutoff(
    system: PeriodicSystem,
    config: CaldeiraLeggettConfig,
    model: Model,
    lost: float,
    last: tuple[float, float] | None,
) -> Model:
    """Get the model with the cutoff raised by as much as the loss of the ISF at t = 0 needs to fall to config.basis_tol, at the rate it fell from the last (cutoff, loss), or _LOST_RATE."""
    rate = _LOST_RATE
    if last is not None and lost < last[1]:
        rate = np.log(last[1] / lost) / (model.cutoff - last[0])
    growth = np.log(lost / config.basis_tol) / float(np.clip(rate, 0.1, 1.0))
    return build_model(
        system,
        config,
        cutoff=get_next_cutoff(
            model, get_energies(model), model.cutoff + max(growth + 1, 2.0)
        ),
        plane_wave_factor=model.plane_wave_factor,
    )


_ON_GRID = 1e-9
"""The distance from a point of the grid of crystal momenta, in units of its spacing, below which dK is on it (`_get_sector`)."""


def _get_sector(  # ruff:ignore[too-many-locals]
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    config: CaldeiraLeggettConfig,
) -> _Sector:
    """Get the sector of the ISF at dK, in bands raised until rho e^{-i dK x} loses at most config.basis_tol above the highest.

    The ISF at t = 0 is Tr[rho e^{-i dK x} e^{i dK x}] = 1, but in the bands
    e^{-i dK x} is truncated, and it loses the part of rho e^{-i dK x} kicked
    above the highest band, which is then the error of the whole ISF:
    for hydrogen on Cu(111) in 1D, in the bands of the steady state (to
    21 k_B T), 2e-6 at dK = |G| and 3e-3 at 2 |G|. Above the tolerance the
    cutoff is raised, by as much as the loss falls with it, and the steady
    state is projected onto the new bands, whose population above the
    old is negligible (`_get_basis`).

    Raises
    ------
    RuntimeError
        If the loss is not below the tolerance after _MAX_TRIES tries.
    """
    state = get_steady_state.load_or_call_uncached(system, config=config)
    model, bands, density = state.model, state.bands, state.density
    # dK is split into the nearest point n @ cell_reciprocal and an offset
    # theta. Every operator shifts k by a multiple of cell_reciprocal, so k mod
    # cell_reciprocal is conserved, and rho e^{-i dK x} is in the sector of
    # crystal momentum transfer s = -n. The offset is removed by the gauge
    # transformation X -> X e^{i theta x}, after which the bra evolves under
    # the operators of the Bloch states on the offset grid,
    # e^{-i theta x} O e^{i theta x} (`get_offset_operators`). The ISF is then
    # Tr[e^{i (dK - theta) x} X(t)], so any dK is reached with the same cell.
    scaled = np.asarray(delta_k) / system.units.length_into(1.0, model.units)
    fraction = np.linalg.solve(model.cell_reciprocal.T, scaled)
    n = np.rint(fraction).astype(int)
    shift = tuple(-int(i) for i in n)
    # An offset of round off, as dK = |G| has in the natural units, is none,
    # so that the sector of dK on the grid is exactly that of its point
    offset = np.where(np.abs(fraction - n) < _ON_GRID, 0.0, fraction - n) @ (
        model.cell_reciprocal
    )
    shifted_kappa = get_shifted_kappa(model, shift)
    last: tuple[float, float] | None = None
    for _ in range(_MAX_TRIES):
        # S[kappa] = <kappa - dK, m| e^{-i dK x} |kappa, n>, and the block
        # (kappa - dK, kappa) of rho e^{-i dK x} is rho[kappa - dK] S[kappa]
        scattering = get_shift_operator(model, bands, shift)
        initial = density[shifted_kappa] @ scattering
        lost = abs(1 - complex(np.sum(scattering.conj() * initial)))
        if lost <= config.basis_tol:
            break
        model, last = (
            _raise_cutoff(system, config, model, lost, last),
            (
                model.cutoff,
                lost,
            ),
        )
        bands = get_bands(model)
        overlap = get_basis_map(state.model, state.bands, model, bands)
        density = overlap @ state.density @ overlap.conj().swapaxes(-1, -2)
    else:
        msg = f"The bands did not hold rho e^(-i dK x) to {config.basis_tol:.0e}"
        raise RuntimeError(msg)
    couplings = get_couplings(model, bands)
    operators = get_operators(bands.energies, couplings)
    bra = get_offset_operators(model, bands, couplings, offset)
    steady_state = None
    if np.all(offset == 0) and np.all(shifted_kappa == np.arange(model.n_kappa)):
        steady_state = (
            state.density
            if model is state.model
            else get_band_steady_state(operators, bands.energies, config)
        )
    return _Sector(
        model=model,
        liouvillian=get_liouvillian(operators, bra, shifted_kappa),
        initial=initial,
        observable=scattering,
        delta_k=scaled,
        steady_state=steady_state,
    )


def _get_friction(system: PeriodicSystem, model: Model) -> float:
    """Get the friction in the natural units of the model."""
    return float(system.gamma) * model.units.time_into(1.0, system.units)


_MAX_FRICTION = 0.4
"""The largest hbar gamma / k_B T at which the universal Lindblad equation is within 1% of the Caldeira-Leggett model.

A Lindblad equation needs the bath to forget faster than the system relaxes
(the Born-Markov approximation): the correlation time of an Ohmic bath,
hbar / k_B T, short against the damping time 1 / gamma. The error is
measured on a free particle, whose diffusion constant in the Ohmic
Caldeira-Leggett model is k_B T / m gamma exactly at any temperature: the
universal Lindblad equation gives (k_B T / m gamma) (1 + (hbar gamma / 4 k_B T)^2),
to within the floor of 2e-4 of friction_tol = 1e-4, at hbar gamma / k_B T
from 0.1 to 2 (1.0027 at 0.2, 1.0101 at 0.4, 1.0158 at 0.5, 1.0626 at 1,
1.2500 at 2), for the masses of H, Li and Na at 100-300 K, which give the
same ratio at the same hbar gamma / k_B T. The error passes 1% at 0.4.
Hydrogen on Cu(111) at 15 ps^-1 and 150 K is at 0.76, 3.6%.
"""


def _warn_friction(system: PeriodicSystem) -> None:
    """Warn if the friction is beyond _MAX_FRICTION, where the model error of the universal Lindblad equation is over 1%."""
    ratio = float(system.gamma) * system.units.hbar / system.kbt
    if ratio > _MAX_FRICTION:
        msg = (
            f"hbar gamma / k_B T = {ratio:.2f} is above {_MAX_FRICTION}: the "
            "universal Lindblad equation is no longer Markovian, and overestimates "
            f"the diffusion constant of a free particle by {(ratio / 4) ** 2:.1%}"
        )
        # Reported at the caller of the public function, beyond its timer
        warnings.warn(msg, stacklevel=4)


@timed
def solve_isf(
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    times: FloatArray,
    *,
    config: CaldeiraLeggettConfig = DEFAULT_CONFIG,
    isf_config: EvolutionConfig = DEFAULT_EVOLUTION_CONFIG,
) -> IsfResult:
    """Solve the Caldeira-Leggett master equation of a periodic system, and get its ISF.

    The ISF I(delta_k, t) = Tr[e^{i delta_k x} V(t) (rho e^{-i delta_k x})]
    of the steady state rho is found at every time to isf_config.stop, by
    two-sided Lanczos (Pade via Lanczos, `get_reduced_evolution`). The
    solver is deterministic, so the ISF has no statistical error.

    At negative times the ISF of the steady state is, by the quantum
    regression theorem, I(-t) = <e^{-i delta_k x(t)} e^{i delta_k x(0)}> =
    Tr[e^{-i delta_k x} V(t) (e^{i delta_k x} rho)], and as V(t) preserves
    Hermiticity (the master equation is of Lindblad form), this is I(t)^*,
    as it is for the exact stationary correlation. Evolved explicitly for
    hydrogen on Cu(111) in 1D at 0.1, 0.3 and 1 |G|, it equals I(t)^* to
    7.6e-12, 2.1e-11 and 4.7e-9, the tolerance of the evolution, where the
    ISF has |Im I| up to 3.9e-4. The other order of the operators,
    Tr[e^{-i delta_k x} V(t) (rho e^{i delta_k x})], is I(-delta_k, t), which
    differs from I(t)^* by up to 7.8e-4.

    The bands are raised until they hold rho e^{-i delta_k x} to
    config.basis_tol (`_get_sector`), so any delta_k is reached. The Lindblad
    operators couple to e^{+-i q x} in place of x, which keeps the equation
    periodic, with the spectral function of the universal
    Lindblad equation as the temperature correction, so that the steady state
    is close to the Gibbs state at any temperature.

    Parameters
    ----------
    system
        The periodic system. Its friction is isotropic.
    delta_k
        The scattering wavevector, in the units of the system.
    times
        The times of the ISF, in the units of the system: any real times, in
        any order. The ISF is evolved once, to each |t|, so negative times
        cost nothing more.
    config
        The basis and linear solves, see `CaldeiraLeggettConfig`.
    isf_config
        The convergence of the evolution, see `EvolutionConfig`.

    Raises
    ------
    ValueError
        If the times are not a one dimensional array of finite numbers, or
        if a dK far beyond the thermal momentum needs more than
        MAX_PLANE_WAVES plane waves (`_get_sector`).

    Warns
    -----
    UserWarning
        If hbar gamma / k_B T > 0.4, where the universal Lindblad equation
        overestimates D of a free particle by over 1% (`_MAX_FRICTION`).
    """  # ruff:ignore[docstring-extraneous-exception]
    folded = fold_times(times)
    _warn_friction(system)
    sector = _get_sector(system, delta_k, config)
    result = get_isf(
        sector.liouvillian,
        sector.initial,
        sector.observable,
        system.units.time_into(folded.magnitudes, sector.model.units),
        delta_k=tuple(sector.delta_k),
        units=sector.model.units,
        isf_config=isf_config,
        steady_state=sector.steady_state,
    )
    # The ISF at exactly the given times, rather than at their round trip
    # through the natural units
    return IsfResult(
        times=folded.times,
        isf=folded.hermitian(result.isf),
        delta_k=delta_k,
        units=system.units,
    )


@timed
def get_slow_modes(
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    n_modes: int | None = None,
    *,
    config: CaldeiraLeggettConfig = DEFAULT_CONFIG,
    isf_config: EvolutionConfig = DEFAULT_EVOLUTION_CONFIG,
) -> SlowModes:
    """Get the slowest modes of the ISF of a periodic system at dK, by Jacobi-Davidson (`find_slow_modes`).

    The modes are the eigenvalues lambda_j of the Liouvillian, with
    the weights a_j of each in the ISF, so that
    I(t) = sum_j a_j exp(lambda_j t) once the faster modes have decayed
    (`get_long_time_isf`). Modes the ISF does not see are skipped. It needs
    no time steps: for lithium on Cu(110) in 2D, 600-700 applications of L,
    3 times fewer than a matrix pencil fit to 10 / friction of the ISF,
    which also misfits the hopping rate at large dK. The slow modes are
    also poles of the two-sided Lanczos recurrence of `solve_isf`, where
    their rates converge to 1e-8 early, but certifying each eigenvector to
    isf_config.mode_rtol from the recurrence took 2.6-7 times the
    applications of L of Jacobi-Davidson for H and Li on Cu(111) in 1D, and
    13-17 times for Na.

    Parameters
    ----------
    system
        The periodic system. Its friction is isotropic.
    delta_k
        The scattering wavevector, in the units of the system.
    n_modes
        The number of modes, by default every mode slower than SLOW_RATE of
        the friction, and at least one. At dK = 0 the only mode seen is the
        steady state, so fewer are found.
    config
        The basis and linear solves, see `CaldeiraLeggettConfig`.
    isf_config
        The convergence of the modes, see `EvolutionConfig`.

    Raises
    ------
    ValueError
        If n_modes is negative, or, as `solve_isf`, at a dK which needs more
        than MAX_PLANE_WAVES plane waves.

    Warns
    -----
    UserWarning
        If hbar gamma / k_B T > 0.4, where the universal Lindblad equation
        overestimates D of a free particle by over 1% (`_MAX_FRICTION`).
    """  # ruff:ignore[docstring-extraneous-exception]
    check_mode_count(n_modes)
    _warn_friction(system)
    sector = _get_sector(system, delta_k, config)
    # A rate in the natural units, over the natural unit of time in the system's
    natural = sector.model.units.time_into(1.0, system.units)
    if n_modes == 0:
        return SlowModes(
            system=system,
            delta_k=tuple(delta_k),
            rates=np.zeros(0, complex),
            amplitudes=np.zeros(0, complex),
        )
    eigen = get_eigen_liouvillian(sector.liouvillian)
    start = to_eigen(eigen, jnp.asarray(sector.initial))
    observable = jnp.asarray(sector.observable)
    measurement = get_measurement(
        eigen, lambda x: jnp.sum(observable.conj() * x), start
    )
    friction = _get_friction(system, sector.model)
    modes, _ = find_slow_modes(
        eigen,
        start,
        measurement,
        get_selection(n_modes, friction),
        config=isf_config,
    )
    return SlowModes(
        system=system,
        delta_k=tuple(delta_k),
        rates=np.array([m.rate for m in modes], dtype=complex) / natural,
        amplitudes=np.array([m.amplitude for m in modes], dtype=complex),
    )


@timed
def get_diffusion_constant(
    system: PeriodicSystem,
    *,
    direction: tuple[float, ...] | None = None,
    config: CaldeiraLeggettConfig = DEFAULT_CONFIG,
) -> float:
    """Get the diffusion constant of a periodic system under the Caldeira-Leggett master equation.

    The diffusion constant D along a direction is the limit of the dephasing
    rate of the ISF over delta_k^2 as delta_k -> 0. It is found exactly from
    perturbation theory in delta_k about the steady state, with one linear
    solve, rather than from an ISF, which decays ever more slowly as
    delta_k -> 0. Along the first axis it is found with the steady state,
    which checks the basis by it (`_get_basis`), so it costs nothing more.

    Parameters
    ----------
    system
        The periodic system. Its friction is isotropic.
    direction
        The direction along which to find D, by default the first axis.
    config
        The basis and linear solves, see `CaldeiraLeggettConfig`.

    Returns
    -------
    float
        The diffusion constant, in the units of the system.

    Warns
    -----
    UserWarning
        If hbar gamma / k_B T > 0.4, where the universal Lindblad equation
        overestimates D of a free particle by over 1% (`_MAX_FRICTION`).
    """
    _warn_friction(system)
    state = _get_state(system, config, direction)
    model = state.model
    # D has units of length^2 / time
    length = model.units.length_into(1.0, system.units)
    return state.diffusion * length**2 / model.units.time_into(1.0, system.units)


def _get_state(
    system: PeriodicSystem,
    config: CaldeiraLeggettConfig,
    direction: tuple[float, ...] | None,
) -> SteadyState:
    """Get the steady state with D and its response along the direction, in bands which give that D to config.basis_tol.

    The steady state finds them along the first axis. Along another, D can
    be far smaller (across the rows of an fcc(110) surface), so it is
    checked by a larger basis again, from the bands of the steady state
    (`_get_basis`).
    """
    state = get_steady_state.load_or_call_uncached(system, config=config)
    unit = np.eye(state.model.n_dim)[0]
    if direction is None or np.allclose(
        np.asarray(direction) / np.linalg.norm(direction), unit
    ):
        return state
    return _get_basis(system, config, direction, state)


def _get_derivative(
    model: Model,
    bands: Bands,
    direction: tuple[float, ...] | None,
) -> tuple[Operators, Operators, FloatArray]:
    """Get the operators, their twist derivative along the direction (by default the first axis), and its unit vector."""
    unit = get_unit_direction(model.n_dim, direction)
    operators, derivative = get_twist_derivative(
        bands.energies, get_couplings(model, bands), get_velocity(model, bands, unit)
    )
    return operators, derivative, unit


@timed
def solve_rms(
    system: PeriodicSystem,
    times: FloatArray,
    *,
    direction: tuple[float, ...] | None = None,
    config: CaldeiraLeggettConfig = DEFAULT_CONFIG,
    isf_config: EvolutionConfig = DEFAULT_EVOLUTION_CONFIG,
) -> RmsResult:
    """Solve the Caldeira-Leggett master equation of a periodic system, and get its mean square displacement along a direction.

    The mean square displacement <(x(t) - x(0))^2> is the limit dK -> 0 of
    -2 ln I(dK, t) / dK^2, the width of the Gaussian <e^{i dK x}> =
    e^{i dK <x>} e^{-dK^2 <dx^2> / 2}, but it is found exactly, from the
    second derivative of the ISF at dK = 0, so it is that of the infinite
    surface with no bias from a finite dK (`get_rms`). Its slope at long
    times is 2 D (`get_diffusion_constant`). It costs a linear solve and
    one evolution, as for an ISF, beyond the response which gives D.

    Parameters
    ----------
    system
        The periodic system. Its friction is isotropic.
    times
        The times, in the units of the system: any real times, in any order.
        The mean square displacement is even in t.
    direction
        The direction along which x is measured, by default the first axis.
    config
        The basis and linear solves, see `CaldeiraLeggettConfig`.
    isf_config
        The convergence of the evolution, see `EvolutionConfig`.

    Raises
    ------
    ValueError
        If the times are not a one dimensional array of finite numbers.

    Warns
    -----
    UserWarning
        If hbar gamma / k_B T > 0.4, where the universal Lindblad equation
        overestimates D of a free particle by over 1% (`_MAX_FRICTION`).
    """  # ruff:ignore[docstring-extraneous-exception]
    folded = fold_times(times)
    _warn_friction(system)
    state = _get_state(system, config, direction)
    model, bands, rho = state.model, state.bands, state.density
    operators, derivative, unit = _get_derivative(model, bands, direction)
    result = get_rms(
        operators,
        derivative,
        rho,
        system.units.time_into(folded.magnitudes, model.units),
        direction=tuple(unit),
        units=model.units,
        config=config,
        isf_config=isf_config,
        response=state.response,
    )
    length = model.units.length_into(1.0, system.units)
    return RmsResult(
        times=folded.times,
        # Even in t, as <(x(-t) - x(0))^2> = <(x(0) - x(t))^2> in the steady state
        mean_square=folded.even(result.mean_square) * length**2,
        direction=tuple(unit),
        units=system.units,
    )
