import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, override

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import sympy as sp

from classical_diffusion.jax.langevin import (
    solve_many as solve_many_jax,
)
from classical_diffusion.jax.langevin import (
    solve_many_overdamped as solve_many_overdamped_jax,
)
from classical_diffusion.langevin._sample import (
    _sample_initial_conditions,
)
from classical_diffusion.simulation import (
    SimulationResult,
    SingleSimulationResult,
    TimeSpan,
)
from classical_diffusion.system import UnitSystem
from classical_diffusion.util import _get_key, cached, hash_array, timed

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from classical_diffusion.langevin import System


@dataclass(frozen=True, kw_only=True)
class SingleLangevinSimulationResult[S: System](SingleSimulationResult[S]):
    """Results of a single simulation of the periodic Langevin equation."""

    p_points: np.ndarray[Any, np.dtype[np.floating]]


class LangevinSimulationResult[S: System](SimulationResult[S]):
    """Results of a simulation of the periodic Langevin equation."""

    _p_points: np.ndarray[Any, np.dtype[np.floating]]

    def __init__(
        self,
        *,
        system: S,
        x_points: np.ndarray[Any, np.dtype[np.floating]],
        p_points: np.ndarray,
        times: np.ndarray,
    ) -> None:
        self._system = system
        self._x_points = x_points
        self._p_points = p_points
        self._times = times

    @property
    def p_points(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        return self._p_points

    def __getitem__(self, idx: int) -> SingleLangevinSimulationResult[S]:
        """Return a single trajectory from the ensemble."""
        return SingleLangevinSimulationResult(
            system=self.system,
            times=self._times,
            x_points=self.x_points[idx],
            p_points=self.p_points[idx],
        )

    def __iter__(self) -> Iterator[SingleLangevinSimulationResult[S]]:
        """Iterate over the trajectories in the ensemble."""
        for i in range(self.x_points.shape[0]):
            yield self[i]

    @override
    def with_units(self, units: UnitSystem) -> Self:
        """Return the rescaled simulation of the system."""
        return type(self)(
            times=self.system.units.time_into(self.times, units),
            x_points=self.system.units.length_into(self.x_points, units),
            p_points=self.system.units.momentum_into(self.p_points, units),
            system=self.system.with_units(units),
        )

    @classmethod
    def from_iter(cls, results: Iterable[SingleLangevinSimulationResult[S]]) -> Self:
        results_list = list(results)
        return cls(
            times=results_list[0].times,
            x_points=np.stack([r.x_points for r in results_list]),
            p_points=np.stack([r.p_points for r in results_list]),
            system=results_list[0].system,
        )


def _convert_time_span(
    old: TimeSpan, old_units: UnitSystem, new_units: UnitSystem
) -> TimeSpan:
    return TimeSpan(
        t_start=old_units.time_into(old.t_start, new_units),
        t_end=old_units.time_into(old.t_end, new_units),
        n_steps=old.n_steps,
    )


def _get_max_force(system: System) -> float:
    """Find max ||F|| numerically using SciPy minimization starting at the origin."""
    if not system.force_expr or system.potential_expr == 0:
        return 0.0

    param_map = dict(zip(system.parameter_symbols, system.params, strict=False))
    force = sp.Matrix(system.force_expr).subs(param_map)

    # Convert the symbolic force vector into a fast numerical function
    force_fn = sp.lambdify(
        system.coordinate_symbols,
        force,
        modules=[{"DerivativeSafeMod": np.mod}, "numpy"],
    )

    def objective(coords: np.ndarray) -> float:
        # Evaluate force vector and return negative magnitude for minimization
        f_vec = np.array(force_fn(*coords), dtype=float)
        return -float(np.linalg.norm(f_vec))

    x0 = np.zeros(len(system.coordinate_symbols))
    res = scipy.optimize.minimize(objective, x0)

    return float(-res.fun) if res.success else 0.0


def _get_langevin_units(system: System) -> UnitSystem:
    """Units scaled purely via intrinsic physical scales of V(x), T, m, and gamma."""
    # Express rates in uniform units (s^-1)
    rate_force = _get_max_force(system) / np.sqrt(system.m * system.kbt)
    rate_force = 0.0 if rate_force == np.inf else float(rate_force)
    rate_gamma = system.gamma

    # Select dominant physical rate
    # If gamma and force are both small, then dont scale the units
    # Length is velocity * time, and time is 1 / nu_0
    v_th = np.sqrt(system.kbt / system.m)
    nu_0 = max(rate_force, rate_gamma, v_th / 1.0)
    characteristic_length = v_th / nu_0

    return UnitSystem(
        boltzmann=1 / system.temperature,
        atomic_mass=system.units.atomic_mass / system.m,
        angstrom=system.units.angstrom / characteristic_length,
    )


def get_random_initial_conditions(
    system: System,
    n_samples: int,
    *,
    minimum_energy: float = 0.0,
    _key: jax.Array,
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
]:
    """Get random initial conditions for a given system."""
    _key = _get_key(_key)

    x_points, p_points = _sample_initial_conditions(
        system.as_canonical(),
        n_samples,
        minimum_energy=minimum_energy,
        _key=_key,
    )
    x_points = np.array(x_points.reshape(-1, system.n_dim))
    p_points = np.array(p_points.reshape(-1, system.n_dim))
    return (x_points, p_points)


def get_random_initial_conditions_ext(
    system: System,
    n_samples: int,
    *,
    minimum_energy: float = 0.0,
    _key: jax.Array | None = None,
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
]:
    """Get random initial conditions for a given system."""
    _key = _get_key(_key)
    normalized_system = system.with_units(_get_langevin_units(system)).as_canonical()
    x_points, p_points = get_random_initial_conditions(
        normalized_system,
        n_samples,
        minimum_energy=system.units.energy_into(
            minimum_energy, normalized_system.units
        ),
        _key=_key,
    )
    return (
        normalized_system.units.length_into(x_points, system.units),
        normalized_system.units.momentum_into(p_points, system.units),
    )


def _solve_many_path[S: System](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[np.ndarray, np.ndarray],
    *,
    _key: jax.Array | None = None,
) -> Path:
    return Path(
        f"{hash(system)}_{hash(time_span)}_{hash_array(initial_conditions)}.npz"
    )


@cached(_solve_many_path)
@timed
def solve_many[S: System](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[
        np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
    ],
    *,
    _key: jax.Array | None = None,
) -> LangevinSimulationResult[S]:
    """Solve an ensemble of ULD Langevin trajectories in parallel via jax.vmap."""
    normalized_system = system.with_units(_get_langevin_units(system)).as_canonical()
    xs0_jax = jnp.asarray(
        system.units.length_into(initial_conditions[0], normalized_system.units)
    )
    ps0_jax = jnp.asarray(
        system.units.momentum_into(initial_conditions[1], normalized_system.units)
    )

    times, xs_batch, ps_batch = solve_many_jax(
        normalized_system,
        _convert_time_span(time_span, system.units, normalized_system.units),
        (xs0_jax, ps0_jax),
        _key=_get_key(_key),
    )

    return LangevinSimulationResult(
        times=normalized_system.units.time_into(np.array(times), system.units),
        x_points=normalized_system.units.length_into(np.array(xs_batch), system.units),
        p_points=normalized_system.units.momentum_into(
            np.array(ps_batch), system.units
        ),
        system=system,
    )


def _solve_single_path[S: System](
    system: S,
    time_span: TimeSpan,
    initial_condition: tuple[np.ndarray, np.ndarray],
    *,
    _key: jax.Array | None = None,
) -> Path:
    return Path(
        f"{system.__class__.__name__}_{hash(system)}_{hash(time_span)}_{hash_array(initial_condition)}.npz"
    )


@cached(_solve_single_path)
@timed
def solve_single[S: System](
    system: S,
    time_span: TimeSpan,
    initial_condition: tuple[
        np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
    ],
    *,
    _key: jax.Array | None = None,
) -> SingleLangevinSimulationResult[S]:
    """Solve the ULD Langevin equation for a single trajectory via vmap."""
    return solve_many.call_uncached(
        system,
        time_span,
        (
            np.array([initial_condition[0]]),
            np.array([initial_condition[1]]),
        ),
        _key=_key,
    )[0]


def _solve_ensemble_path[S: System](
    system: S,
    time_span: TimeSpan,
    n_samples: int,
    *,
    minimum_energy: float = 0.0,
    _key: jax.Array | None = None,
) -> Path:
    return Path(
        f"solve_ensemble_{system.__class__.__name__}_{hash(system)}_{hash(time_span)}_{n_samples}_{minimum_energy}.npz"
    )


@cached(_solve_ensemble_path)
@timed
def solve_ensemble[S: System](
    system: S,
    time_span: TimeSpan,
    n_samples: int,
    *,
    minimum_energy: float = 0.0,
    _key: jax.Array | None = None,
) -> LangevinSimulationResult[S]:
    """Solve an ensemble of trajectories."""
    _key = _get_key(_key)

    simulated_system = system.with_units(_get_langevin_units(system)).as_canonical()
    out = solve_many.call_uncached(
        simulated_system,
        _convert_time_span(time_span, system.units, simulated_system.units),
        get_random_initial_conditions(
            simulated_system,
            n_samples,
            minimum_energy=system.units.energy_into(
                minimum_energy, simulated_system.units
            ),
            _key=_key,
        ),
        _key=_key,
    ).with_units(system.units)
    return LangevinSimulationResult[S](
        times=out.times, x_points=out.x_points, p_points=out.p_points, system=system
    )


@timed
def solve_single_ballistic[S: System](
    system: S,
    time_span: TimeSpan,
    initial_condition: tuple[
        np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
    ],
    *,
    _key: jax.Array | None = None,
) -> SingleLangevinSimulationResult[S]:
    """Solve the ULD Langevin equation for a single trajectory via vmap."""
    out = solve_single.call_uncached(
        dataclasses.replace(system.as_canonical(), gamma=0.0),
        time_span,
        initial_condition,
        _key=_key,
    )

    return SingleLangevinSimulationResult(
        times=out.times,
        x_points=out.x_points,
        p_points=out.p_points,
        system=system,
    )


@timed
def solve_many_ballistic[S: System](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[
        np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
    ],
    *,
    _key: jax.Array | None = None,
) -> LangevinSimulationResult[S]:
    """Solve the ULD Langevin equation for a single trajectory via vmap."""
    simulated_system = system.as_canonical()
    out = solve_many.call_uncached(
        dataclasses.replace(simulated_system, gamma=0.0),
        time_span,
        initial_conditions,
        _key=_key,
    )

    return LangevinSimulationResult(
        times=out.times,
        x_points=out.x_points,
        p_points=out.p_points,
        system=system,
    )


def _solve_ensemble_ballistic_path[S: System](
    system: S,
    time_span: TimeSpan,
    n_samples: int,
    *,
    minimum_energy: float = 0.0,
    _key: jax.Array | None = None,
) -> Path:
    return Path(
        f"solve_ensemble_ballistic_{system.__class__.__name__}_{hash(system)}_{hash(time_span)}_{n_samples}_{minimum_energy}.npz"
    )


@cached(_solve_ensemble_ballistic_path)
@timed
def solve_ensemble_ballistic[S: System](
    system: S,
    time_span: TimeSpan,
    n_samples: int,
    *,
    minimum_energy: float = 0.0,
    _key: jax.Array | None = None,
) -> LangevinSimulationResult[S]:
    """Solve an ensemble of ballistic trajectories in parallel via jax.vmap."""
    _key = _get_key(_key)

    simulated_system = system.with_units(_get_langevin_units(system)).as_canonical()
    out = solve_ensemble.call_uncached(
        dataclasses.replace(simulated_system, gamma=0.0),
        _convert_time_span(time_span, system.units, simulated_system.units),
        n_samples=n_samples,
        minimum_energy=system.units.energy_into(minimum_energy, simulated_system.units),
        _key=_key,
    ).with_units(system.units)

    return LangevinSimulationResult[S](
        times=out.times, x_points=out.x_points, p_points=out.p_points, system=system
    )


def _get_overdamped_langevin_units(system: System) -> UnitSystem:
    """Units scaled for the overdamped Langevin equation."""
    # dx = (F(x) / gamma) dt + sqrt(2 kB T / gamma) dW
    # scale so the noise is of order 1, i.e. sqrt(2 kB T / gamma) * sqrt(dt) ~ 1
    # so we want gamma approx 1 in the new units
    characteristic_length = np.sqrt(system.kbt * system.m) / system.gamma
    characteristic_length = np.sqrt(system.kbt / system.m) / system.gamma
    return UnitSystem(
        boltzmann=1 / system.temperature,
        atomic_mass=system.units.atomic_mass / system.m,
        angstrom=system.units.angstrom / characteristic_length,
    )


def _solve_many_overdamped_path[S: System](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[
        np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
    ],
    *,
    _key: jax.Array | None = None,
) -> Path:
    return Path(
        f"overdamped_{hash(system)}_{hash(time_span)}_{hash_array(initial_conditions)}.npz"
    )


@cached(_solve_many_overdamped_path)
@timed
def solve_many_overdamped[S: System](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[
        np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
    ],
    *,
    _key: jax.Array | None = None,
) -> LangevinSimulationResult[S]:
    """Solve an ensemble of overdamped Langevin trajectories in parallel via jax.vmap."""
    simulated_system = system.with_units(
        _get_overdamped_langevin_units(system)
    ).as_canonical()
    times, x_points, p_points = solve_many_overdamped_jax(
        simulated_system,
        _convert_time_span(time_span, system.units, simulated_system.units),
        initial_conditions=(
            jnp.asarray(
                system.units.length_into(initial_conditions[0], simulated_system.units)
            ),
            jnp.asarray(
                system.units.momentum_into(
                    initial_conditions[1], simulated_system.units
                )
            ),
        ),
        _key=_get_key(_key),
    )

    return LangevinSimulationResult(
        times=simulated_system.units.time_into(np.array(times), system.units),
        x_points=simulated_system.units.length_into(np.array(x_points), system.units),
        p_points=simulated_system.units.momentum_into(
            np.zeros_like(p_points), system.units
        ),
        system=system,
    )


def _solve_ensemble_overdamped_path[S: System](
    system: S,
    time_span: TimeSpan,
    n_samples: int,
    *,
    _key: jax.Array | None = None,
) -> Path:
    return Path(f"overdamped_ensemble_{hash(system)}_{hash(time_span)}_{n_samples}.npz")


@cached(_solve_ensemble_overdamped_path)
@timed
def solve_ensemble_overdamped[S: System](
    system: S,
    time_span: TimeSpan,
    n_samples: int,
    *,
    _key: jax.Array | None = None,
) -> LangevinSimulationResult[S]:
    """Solve an ensemble of overdamped Langevin trajectories in parallel via jax.vmap."""
    _key = _get_key(_key)
    simulated_system = system.with_units(
        _get_overdamped_langevin_units(system)
    ).as_canonical()
    result = solve_many_overdamped.call_uncached(
        simulated_system,
        _convert_time_span(time_span, system.units, simulated_system.units),
        get_random_initial_conditions(simulated_system, n_samples, _key=_key),
        _key=_key,
    ).with_units(system.units)

    return LangevinSimulationResult(
        times=result.times,
        x_points=result.x_points,
        p_points=result.p_points,
        system=system,
    )
