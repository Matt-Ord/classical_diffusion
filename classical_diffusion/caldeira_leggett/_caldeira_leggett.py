import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from classical_diffusion.caldeira_leggett._bands import (
    BlochBands,
    get_bloch_bands,
    get_plane_wave_energy,
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
    from classical_diffusion.caldeira_leggett._model import ComplexArray, FloatArray
    from classical_diffusion.langevin._system import PeriodicSystem

DEFAULT_FRICTION_TOL = 5e-3
DEFAULT_MAX_ENERGY = 12.0


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


def _get_model(
    system: PeriodicSystem,
    *,
    friction_tol: float,
    max_energy: float,
    plane_wave_energy: float | None,
) -> BlochModel:
    model = build_model(
        system,
        friction_tol=friction_tol,
        max_energy=max_energy,
        plane_wave_energy=max_energy
        if plane_wave_energy is None
        else plane_wave_energy,
    )
    if plane_wave_energy is None:
        return dataclasses.replace(
            model, plane_wave_energy=get_plane_wave_energy(model)
        )
    return model


def _get_bands(
    system: PeriodicSystem,
    *,
    friction_tol: float,
    max_energy: float,
    plane_wave_energy: float | None,
) -> BlochBands:
    return get_bloch_bands(
        _get_model(
            system,
            friction_tol=friction_tol,
            max_energy=max_energy,
            plane_wave_energy=plane_wave_energy,
        )
    )


def _get_steady_state_path(
    system: PeriodicSystem,
    *,
    friction_tol: float = DEFAULT_FRICTION_TOL,
    max_energy: float = DEFAULT_MAX_ENERGY,
    plane_wave_energy: float | None = None,
) -> Path:
    model = _get_model(
        system,
        friction_tol=friction_tol,
        max_energy=max_energy,
        plane_wave_energy=plane_wave_energy,
    )
    return Path(f"caldeira_leggett_steady_state_{model.get_hash()}.npz")


@cached(_get_steady_state_path, default_call="load_or_call_uncached")
@timed
def get_steady_state(
    system: PeriodicSystem,
    *,
    friction_tol: float = DEFAULT_FRICTION_TOL,
    max_energy: float = DEFAULT_MAX_ENERGY,
    plane_wave_energy: float | None = None,
) -> SteadyState:
    """Get the steady state of the master equation of a periodic system.

    Every ISF, and the diffusion constant, starts from the steady state. By
    default they load it from the cache, if it is there, but do not save it.
    Call ``get_steady_state.call_cached`` once, with the same parameters, to
    save it for them.
    """
    bands = _get_bands(
        system,
        friction_tol=friction_tol,
        max_energy=max_energy,
        plane_wave_energy=plane_wave_energy,
    )
    rho = get_sector_steady_state(get_sector_liouvillian(bands))
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
def solve_isf(  # ruff:ignore[too-many-arguments]
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    times: FloatArray,
    *,
    friction_tol: float = DEFAULT_FRICTION_TOL,
    max_energy: float = DEFAULT_MAX_ENERGY,
    plane_wave_energy: float | None = None,
) -> IsfResult:
    """Solve the Caldeira-Leggett master equation of a periodic system, and get its ISF.

    The ISF I(delta_k, t) = Tr[e^{i delta_k x} V(t) (rho e^{-i delta_k x})]
    of the steady state rho is propagated in time, at the increasing times
    from zero. The solver is deterministic, so the ISF has no error.

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
    friction_tol
        The relative error allowed in the friction, which sets the number of
        periods of the potential in the simulation cell. The smallest q is a
        reciprocal lattice vector of the cell, and e^{+-i q x} / q stands in
        for x only where q x is small. The error is taken at the coherence
        length of a free particle, hbar / sqrt(m k_B T).
    max_energy
        The energy, in units of k_B T, above the ground state up to which
        bands are kept.
    plane_wave_energy
        The kinetic energy, in units of k_B T, up to which plane waves are
        kept to build the bands. By default, 2.5 times the largest kinetic
        energy that the highest band can classically have: max_energy plus
        the zero point energy of the ground state.

    Raises
    ------
    ValueError
        If the times do not increase from zero.
    """
    times = np.asarray(times, dtype=np.float64)
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        msg = "The times must increase from zero"
        raise ValueError(msg)
    bands = _get_bands(
        system,
        friction_tol=friction_tol,
        max_energy=max_energy,
        plane_wave_energy=plane_wave_energy,
    )
    model = bands.model
    steady_state = get_steady_state.load_or_call_uncached(
        system,
        friction_tol=friction_tol,
        max_energy=max_energy,
        plane_wave_energy=plane_wave_energy,
    )
    sector = get_scattering_sector(
        get_sector_liouvillian(bands), np.asarray(delta_k) * model.length
    )
    isf = get_sector_isf(sector, _in_bands(bands, steady_state), times / model.time)
    return IsfResult(times=times, isf=isf, delta_k=delta_k, units=system.units)


@timed
def get_diffusion_constant(
    system: PeriodicSystem,
    *,
    direction: tuple[float, ...] | None = None,
    friction_tol: float = DEFAULT_FRICTION_TOL,
    max_energy: float = DEFAULT_MAX_ENERGY,
    plane_wave_energy: float | None = None,
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
    friction_tol, max_energy, plane_wave_energy
        The parameters of the basis, see `solve_isf`.

    Returns
    -------
    float
        The diffusion constant, in the units of the system.
    """
    bands = _get_bands(
        system,
        friction_tol=friction_tol,
        max_energy=max_energy,
        plane_wave_energy=plane_wave_energy,
    )
    model = bands.model
    steady_state = get_steady_state.load_or_call_uncached(
        system,
        friction_tol=friction_tol,
        max_energy=max_energy,
        plane_wave_energy=plane_wave_energy,
    )
    unit = np.eye(model.n_dim)[0] if direction is None else np.asarray(direction, float)
    diffusion = get_diffusion_coefficient(
        get_sector_liouvillian(bands),
        _in_bands(bands, steady_state),
        unit / np.linalg.norm(unit),
    )
    return diffusion * model.length**2 / model.time
