# cspell: ignore rint Hermiticity polyfit
import dataclasses
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
    """The estimated error of D from the basis, relative to D (`_get_basis`)."""


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
_SAFETY = 600.0
"""The error of D relative to D_free per unit of population left at the edge of the basis, before it is measured (`_get_basis`).

Against bands to 40-150 k_B T (`~/cd-agents/adaptive`), the error of D
over the population of the highest band times D_free / D is, where the
error is 1e-6 to 1e-2: 2-80 for hydrogen on Cu(111) in 1D, 0.3-20
lithium, 9-11 sodium, 2-140 hydrogen in 2D, 5-13 lithium in 2D, 6-12 a
nearly free particle, 11-160 lithium at 300 K, 27-90 xenon at 121 K,
200-310 CO at 150 K, 220-240 hydrogen at 50 K (deep tunnelling) and
170-580 potassium at 300 K. 600 bounds it on all, but is 50 times too
much for lithium, so it is measured for each system instead.
"""
_MARGIN = 3.0
"""The margin on the measured ratio of the error of D to the population at the edge (`_get_basis`).

Within a system the ratio grows by up to 3 as the population falls by 4
decades (2-10 times as it falls by 8).
"""
_FLOOR = 3.0
"""The smallest ratio of the error of D to the population at the edge, should D change by less between two bases."""
_FIRST_STEP = 10.0
"""The most the population at the edge is lowered by before the ratio is measured."""
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
    """Get the steady state in bands raised until D along the direction (by default the first axis) has an estimated error below config.basis_tol.

    From a cutoff of _START above the saddle (or that of start), each try
    finds the steady state and D. The error of D is the population at the
    edge of the basis (`_get_edge`), p, times D_free / D, times a ratio
    which differs from system to system by over 100, so it is measured:
    once two tries have found D_1 and D_2, the error of D_1 is about
    |D_2 - D_1|, which gives the ratio, and _MARGIN times the largest
    measured (and at least _FLOOR) estimates the error of each later try.
    Before that _SAFETY bounds it, and the population is lowered by at most
    _FIRST_STEP. The plane waves are raised if they hold more of the
    population at the edge, and otherwise the cutoff, by as much as the
    population falls with the energy of the bands needs (`_get_growth`).

    Raises
    ------
    RuntimeError
        If the error is not below the tolerance after _MAX_TRIES tries.
    """
    cutoff = _START if start is None else start.model.cutoff
    factor = _PLANE_WAVE_FACTOR if start is None else start.model.plane_wave_factor
    last: tuple[float, float] | None = None
    ratio = None
    for _ in range(_MAX_TRIES):
        model = build_model(system, config, cutoff=cutoff, plane_wave_factor=factor)
        energies = get_energies(model)
        bands = get_bands(model, energies)
        operators, derivative, _ = _get_derivative(model, bands, direction)
        density = get_band_steady_state(operators, bands.energies, config)
        diffusion, response = get_diffusion_coefficient(
            operators, derivative, density, config
        )
        top, plane_waves = _get_edge(model, bands, density)
        free = 1 / model.friction
        if last is not None:
            measured = _MARGIN * abs(diffusion - last[1]) / (last[0] * free)
            ratio = max(ratio or _FLOOR, measured)
        scale = (top + plane_waves) * free / max(abs(diffusion), 1e-300)
        error = (_SAFETY if ratio is None else ratio) * scale
        if error <= config.basis_tol:
            return SteadyState(
                model=model,
                bands=bands,
                density=density,
                diffusion=diffusion,
                response=response,
                error=error,
            )
        last = (top + plane_waves, diffusion)
        if plane_waves > top:
            factor *= 1.5
            continue
        lower = 2 * error / config.basis_tol
        cutoff = get_next_cutoff(
            model,
            energies,
            cutoff
            + _get_growth(
                bands.energies,
                np.real(np.einsum("kaa->ka", density)),
                lower if ratio is not None else min(lower, _FIRST_STEP),
            ),
        )
    msg = f"The basis did not reach an error of {config.basis_tol:.0e} in D"
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
    n = np.rint(np.linalg.solve(model.cell_reciprocal.T, scaled)).astype(int)
    shift = tuple(-int(i) for i in n)
    offset = scaled - n @ model.cell_reciprocal
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
    return _Sector(
        model=model,
        liouvillian=get_liouvillian(operators, bra, shifted_kappa),
        initial=initial,
        observable=scattering,
        delta_k=scaled,
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
    be far smaller (across the rows of an fcc(110) surface), and with it the
    population the basis may leave at its edge, so the bands are raised
    from there if they need to be (`_get_basis`).
    """
    state = get_steady_state.load_or_call_uncached(system, config=config)
    unit = np.eye(state.model.n_dim)[0]
    if direction is None or np.allclose(
        np.asarray(direction) / np.linalg.norm(direction), unit
    ):
        return state
    model, bands, density = state.model, state.bands, state.density
    operators, derivative, _ = _get_derivative(model, bands, direction)
    diffusion, response = get_diffusion_coefficient(
        operators, derivative, density, config
    )
    # The error of the steady state's D scaled to this D (`_get_basis`)
    error = state.error * abs(state.diffusion / diffusion)
    if error > config.basis_tol:
        return _get_basis(system, config, direction, state)
    return dataclasses.replace(
        state, diffusion=diffusion, response=response, error=error
    )


def _get_derivative(
    model: Model,
    bands: Bands,
    direction: tuple[float, ...] | None,
) -> tuple[Operators, Operators, FloatArray]:
    """Get the operators, their twist derivative along the direction (by default the first axis), and its unit vector."""
    couplings = get_couplings(model, bands)
    operators = get_operators(bands.energies, couplings)
    unit = get_unit_direction(model.n_dim, direction)
    velocity = get_velocity(model, bands, unit)
    derivative = get_twist_derivative(bands.energies, couplings, operators, velocity)
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
