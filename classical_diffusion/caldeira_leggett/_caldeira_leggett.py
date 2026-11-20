# cspell: ignore rint
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from classical_diffusion.caldeira_leggett._bands import (
    Bands,
    get_bands,
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
from classical_diffusion.caldeira_leggett._slow_modes import (
    ModeCount,
    RateCut,
    find_slow_modes,
)
from classical_diffusion.caldeira_leggett._solver import (
    RmsResult,
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
    from classical_diffusion.caldeira_leggett._model import ComplexArray, FloatArray
    from classical_diffusion.langevin._system import PeriodicSystem


@dataclass(frozen=True, kw_only=True)
class SteadyState:
    """The steady state of the master equation of a periodic system, with the bands it is found in.

    The state is diagonal in the crystal momentum. The bands are only defined
    up to a phase, and up to a rotation where they are degenerate, which can
    differ each time they are found, so the state keeps its own.
    """

    model: Model
    bands: Bands
    density: ComplexArray
    """The density matrix in the basis of the bands, (n_kappa, n_bands, n_bands)."""


def _get_steady_state_path(
    system: PeriodicSystem, *, config: CaldeiraLeggettConfig = DEFAULT_CONFIG
) -> Path:
    return Path(f"caldeira_leggett_steady_state_{hash(system)}_{hash(config)}.npz")


@cached(_get_steady_state_path, default_call="load_or_call_uncached")
@timed
def get_steady_state(
    system: PeriodicSystem, *, config: CaldeiraLeggettConfig = DEFAULT_CONFIG
) -> SteadyState:
    """Get the steady state of the master equation of a periodic system.

    Every ISF, and the diffusion constant, starts from the steady state. By
    default they load it from the cache, if it is there, but do not save it.
    Call ``get_steady_state.call_cached`` once, with the same configuration,
    to save it for them.
    """
    model = build_model(system, config)
    bands = get_bands(model)
    operators = get_operators(bands.energies, get_couplings(model, bands))
    density = get_band_steady_state(operators, bands.energies, config)
    return SteadyState(model=model, bands=bands, density=density)


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


_LOST_TOL = 1e-4
"""The part of the ISF at t = 0 which the bands may lose at a large dK.

It is an error of the basis, set by max_energy, not by the convergence of
the evolution, and the truncation of the bands loses a little at any dK:
1e-10 for lithium on Cu(111) in 1D at 0.3 |G|, 1.7e-6 for hydrogen in 2D
at 0.3 |G| with bands to 16 k_B T and no more. The tolerance catches a dK
far beyond the thermal momentum, where the loss grows fast: hydrogen in 1D
loses 2.5e-5 at 3 |G|, 1.4e-3 at 3.7 |G| and all of it at 10 |G|.
"""


def _get_sector(
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    config: CaldeiraLeggettConfig,
) -> _Sector:
    """Get the sector of the ISF at dK.

    The ISF at t = 0 is Tr[rho e^{-i dK x} e^{i dK x}] = 1, but in the bands
    e^{-i dK x} is truncated, and it loses the part of rho e^{-i dK x} kicked
    above the highest band: for hydrogen on Cu(111) in 1D, with the default
    bands, 1e-12 at dK = |G|, 7e-7 at 2.5 |G| and 1e-3 at 3.7 |G|, which is
    then the error of the whole ISF (against bands to 40 k_B T).

    Raises
    ------
    ValueError
        If more than _LOST_TOL of the ISF at t = 0 is lost.
    """
    state = get_steady_state.load_or_call_uncached(system, config=config)
    model, bands = state.model, state.bands
    couplings = get_couplings(model, bands)
    operators = get_operators(bands.energies, couplings)
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
    bra = get_offset_operators(model, bands, couplings, offset)
    shifted_kappa = get_shifted_kappa(model, shift)
    # S[kappa] = <kappa - dK, m| e^{-i dK x} |kappa, n>, and the block
    # (kappa - dK, kappa) of rho e^{-i dK x} is rho[kappa - dK] S[kappa]
    scattering = get_shift_operator(model, bands, shift)
    initial = state.density[shifted_kappa] @ scattering
    lost = abs(1 - complex(np.sum(scattering.conj() * initial)))
    if lost > _LOST_TOL:
        msg = (
            f"The bands hold only 1 - {lost:.1e} of rho e^(-i dK x) at this dK: "
            "raise CaldeiraLeggettConfig.max_energy"
        )
        raise ValueError(msg)
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
    of the steady state rho is found exactly at every time, as its slow
    modes plus a stepper on the rest (`get_reduced_evolution`). The solver
    is deterministic, so the ISF has no error.

    The Lindblad operators couple to e^{+-i q x} in place of x, which keeps
    the equation periodic, with the spectral function of the universal
    Lindblad equation as the temperature correction, so that the steady state
    is close to the Gibbs state at any temperature.

    Parameters
    ----------
    system
        The periodic system. Its friction is isotropic.
    delta_k
        The scattering wavevector, in the units of the system.
    times
        The times of the ISF, in the units of the system.
    config
        The basis and linear solves, see `CaldeiraLeggettConfig`.
    isf_config
        The convergence of the evolution, see `EvolutionConfig`.

    Raises
    ------
    ValueError
        If the times do not increase from zero, or if the bands lose more
        than 1e-4 of the ISF at t = 0, at a dK far beyond the thermal
        momentum (`_get_sector`).
    """
    times = np.asarray(times, dtype=np.float64)
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        msg = "The times must increase from zero"
        raise ValueError(msg)
    sector = _get_sector(system, delta_k, config)
    result = get_isf(
        sector.liouvillian,
        sector.initial,
        sector.observable,
        system.units.time_into(times, sector.model.units),
        delta_k=tuple(sector.delta_k),
        units=sector.model.units,
        friction=_get_friction(system, sector.model),
        isf_config=isf_config,
    )
    # The ISF at exactly the given times, rather than at their round trip
    # through the natural units
    return IsfResult(times=times, isf=result.isf, delta_k=delta_k, units=system.units)


@dataclass(frozen=True, kw_only=True)
class SlowModes:
    """The slowest modes of the ISF of a system at dK, I(t) = sum_j a_j exp(lambda_j t) once the faster modes have decayed."""

    system: PeriodicSystem
    delta_k: tuple[float, ...]
    """The scattering wavevector, in the units of the system."""
    rates: ComplexArray
    """lambda_j, in the units of the system (1 / time), slowest first."""
    amplitudes: ComplexArray
    """a_j."""


SLOW_RATE = 0.25
"""The decay rate below which a mode is slow, in units of the friction.

A quarter of the friction is clear of the in-well vibrations, which decay at
about half of it, and of the momentum relaxation, at about the friction:
the modes below it are the hops between sites, and the exchange between
sites of different kinds.
"""


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
    which also misfits the hopping rate at large dK.

    Parameters
    ----------
    system
        The periodic system. Its friction is isotropic.
    delta_k
        The scattering wavevector, in the units of the system.
    n_modes
        The number of modes, by default every mode slower than SLOW_RATE of
        the friction, and at least one. `solve_isf` may find more, any mode
        which is cheaper to find than to step through, but always these. At
        dK = 0 the only mode seen is the steady state, so fewer are found.
    config
        The basis and linear solves, see `CaldeiraLeggettConfig`.
    isf_config
        The convergence of the modes, see `EvolutionConfig`.

    Raises
    ------
    ValueError
        If n_modes is negative, or, as `solve_isf`, at a dK the bands cannot
        hold.
    """
    if n_modes is not None and n_modes < 0:
        msg = f"n_modes must not be negative, got {n_modes}"
        raise ValueError(msg)
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
        RateCut(rate=SLOW_RATE * friction)
        if n_modes is None
        else ModeCount(n_modes=n_modes),
        config=isf_config,
    )
    return SlowModes(
        system=system,
        delta_k=tuple(delta_k),
        rates=np.array([m.rate for m in modes], dtype=complex) / natural,
        amplitudes=np.array([m.amplitude for m in modes], dtype=complex),
    )


def get_long_time_isf(slow_modes: SlowModes, times: FloatArray) -> IsfResult:
    """Get the ISF of the slow modes alone, sum_j a_j exp(lambda_j t), at the times in the units of the system.

    It is the ISF once the faster modes have decayed: for the ISF at every
    time, use `solve_isf`.
    """
    times = np.asarray(times, dtype=np.float64)
    return IsfResult(
        times=times,
        isf=np.exp(np.outer(times, slow_modes.rates)) @ slow_modes.amplitudes,
        delta_k=slow_modes.delta_k,
        units=slow_modes.system.units,
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
    delta_k -> 0.

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
    """
    state = get_steady_state.load_or_call_uncached(system, config=config)
    model, bands, rho = state.model, state.bands, state.density
    operators, derivative, _ = _get_derivative(model, bands, direction)
    diffusion = get_diffusion_coefficient(operators, derivative, rho, config)
    # D has units of length^2 / time
    length = model.units.length_into(1.0, system.units)
    return diffusion * length**2 / model.units.time_into(1.0, system.units)


def _get_derivative(
    model: Model,
    bands: Bands,
    direction: tuple[float, ...] | None,
) -> tuple[Operators, Operators, FloatArray]:
    """Get the operators, their twist derivative along the direction (by default the first axis), and its unit vector."""
    couplings = get_couplings(model, bands)
    operators = get_operators(bands.energies, couplings)
    unit = np.eye(model.n_dim)[0] if direction is None else np.asarray(direction, float)
    unit /= np.linalg.norm(unit)
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
    times is 2 D (`get_diffusion_constant`). It costs two linear solves, as
    for D, and one evolution, as for an ISF.

    Parameters
    ----------
    system
        The periodic system. Its friction is isotropic.
    times
        The times, in the units of the system, increasing from zero.
    direction
        The direction along which x is measured, by default the first axis.
    config
        The basis and linear solves, see `CaldeiraLeggettConfig`.
    isf_config
        The convergence of the evolution, see `EvolutionConfig`.

    Raises
    ------
    ValueError
        If the times do not increase from zero.
    """
    times = np.asarray(times, dtype=np.float64)
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        msg = "The times must increase from zero"
        raise ValueError(msg)
    state = get_steady_state.load_or_call_uncached(system, config=config)
    model, bands, rho = state.model, state.bands, state.density
    operators, derivative, unit = _get_derivative(model, bands, direction)
    result = get_rms(
        operators,
        derivative,
        rho,
        system.units.time_into(times, model.units),
        direction=tuple(unit),
        units=model.units,
        config=config,
        friction=_get_friction(system, model),
        isf_config=isf_config,
    )
    length = model.units.length_into(1.0, system.units)
    return RmsResult(
        times=times,
        mean_square=result.mean_square * length**2,
        direction=tuple(unit),
        units=system.units,
    )
