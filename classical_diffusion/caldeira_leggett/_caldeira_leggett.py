import dataclasses
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np

from classical_diffusion.caldeira_leggett._bands import (
    BlochBands,
    get_bloch_bands,
    get_plane_wave_energy,
)
from classical_diffusion.caldeira_leggett._config import (
    DEFAULT_CONFIG,
    CaldeiraLeggettConfig,
)
from classical_diffusion.caldeira_leggett._liouvillian import (
    get_scattering_sector,
    get_sector_liouvillian,
)
from classical_diffusion.caldeira_leggett._model import BlochModel, build_model
from classical_diffusion.caldeira_leggett._solver import (
    get_diffusion_coefficient,
)
from classical_diffusion.caldeira_leggett._solver import get_isf as get_sector_isf
from classical_diffusion.caldeira_leggett._solver import (
    get_steady_state as get_sector_steady_state,
)
from classical_diffusion.simulation import IsfResult
from classical_diffusion.util import cached, timed

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.caldeira_leggett._model import ComplexArray, FloatArray
    from classical_diffusion.langevin._system import PeriodicSystem


@dataclass(frozen=True, kw_only=True)
class SteadyState:
    """The steady state of the master equation of a periodic system.

    The state is diagonal in the crystal momentum. It is stored in the basis
    of the plane waves, rather than of the bands, since the bands are only
    defined up to a phase, and up to a rotation where they are degenerate,
    which can differ each time they are found.
    """

    density: tuple[ComplexArray, ...]
    """The density matrix at each crystal momentum, shape (n_plane_waves, n_plane_waves)."""


def _with_x64[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    """Run f with 64 bit floats in jax, which the tolerances of the solver need."""

    @functools.wraps(f)
    def wrap(*args: P.args, **kw: P.kwargs) -> R:
        with jax.enable_x64(new_val=True):
            return f(*args, **kw)

    return wrap


def _get_model(system: PeriodicSystem, config: CaldeiraLeggettConfig) -> BlochModel:
    model = build_model(
        system,
        friction_tol=config.friction_tol,
        max_energy=config.max_energy,
        plane_wave_energy=config.max_energy
        if config.plane_wave_energy is None
        else config.plane_wave_energy,
    )
    if config.plane_wave_energy is None:
        return dataclasses.replace(
            model, plane_wave_energy=get_plane_wave_energy(model)
        )
    return model


def _get_bands(system: PeriodicSystem, config: CaldeiraLeggettConfig) -> BlochBands:
    return get_bloch_bands(_get_model(system, config))


def _get_steady_state_path(
    system: PeriodicSystem, *, config: CaldeiraLeggettConfig = DEFAULT_CONFIG
) -> Path:
    model = _get_model(system, config)
    return Path(
        f"caldeira_leggett_steady_state_{model.get_hash()}_{config.linear_rtol:.1e}.npz"
    )


@cached(_get_steady_state_path, default_call="load_or_call_uncached")
@timed
@_with_x64
def get_steady_state(
    system: PeriodicSystem, *, config: CaldeiraLeggettConfig = DEFAULT_CONFIG
) -> SteadyState:
    """Get the steady state of the master equation of a periodic system.

    Every ISF, and the diffusion constant, starts from the steady state. By
    default they load it from the cache, if it is there, but do not save it.
    Call ``get_steady_state.call_cached`` once, with the same configuration,
    to save it for them.
    """
    bands = _get_bands(system, config)
    rho = get_sector_steady_state(get_sector_liouvillian(bands), config)
    return SteadyState(
        density=tuple(
            u @ x @ u.conj().T for u, x in zip(bands.vectors, rho, strict=True)
        )
    )


def _in_bands(bands: BlochBands, steady_state: SteadyState) -> ComplexArray:
    """Get the steady state in the basis of the bands, shape (n_kappa, n_bands, n_bands)."""
    return np.array(
        [
            u.conj().T @ x @ u
            for u, x in zip(bands.vectors, steady_state.density, strict=True)
        ]
    )


@timed
@_with_x64
def solve_isf(
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    times: FloatArray,
    *,
    config: CaldeiraLeggettConfig = DEFAULT_CONFIG,
) -> IsfResult:
    """Solve the Caldeira-Leggett master equation of a periodic system, and get its ISF.

    The ISF I(delta_k, t) = Tr[e^{i delta_k x} V(t) (rho e^{-i delta_k x})]
    of the steady state rho is propagated in time, with adaptive steps, to the
    increasing times from zero. The solver is deterministic, so the ISF has
    no error.

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
        The convergence parameters, see `CaldeiraLeggettConfig`.

    Raises
    ------
    ValueError
        If the times do not increase from zero.
    """
    times = np.asarray(times, dtype=np.float64)
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        msg = "The times must increase from zero"
        raise ValueError(msg)
    bands = _get_bands(system, config)
    model = bands.model
    steady_state = get_steady_state.load_or_call_uncached(system, config=config)
    sector = get_scattering_sector(
        get_sector_liouvillian(bands), np.asarray(delta_k) * model.length
    )
    isf = get_sector_isf(
        sector, _in_bands(bands, steady_state), times / model.time, config
    )
    return IsfResult(times=times, isf=isf, delta_k=delta_k, units=system.units)


@timed
@_with_x64
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
        The convergence parameters, see `CaldeiraLeggettConfig`.

    Returns
    -------
    float
        The diffusion constant, in the units of the system.
    """
    bands = _get_bands(system, config)
    model = bands.model
    steady_state = get_steady_state.load_or_call_uncached(system, config=config)
    unit = np.eye(model.n_dim)[0] if direction is None else np.asarray(direction, float)
    diffusion = get_diffusion_coefficient(
        get_sector_liouvillian(bands),
        _in_bands(bands, steady_state),
        unit / np.linalg.norm(unit),
        config,
    )
    return diffusion * model.length**2 / model.time
