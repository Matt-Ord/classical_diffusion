import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, override

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from classical_diffusion.langevin._sample import (
    make_free_point_sampler,
    make_initial_conditions_sampler,
)
from classical_diffusion.simulation import SimulationResult, SingleSimulationResult
from classical_diffusion.system import UnitSystem
from classical_diffusion.util import cached, hash_array, timed

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from classical_diffusion.langevin import CanonicalSystem, System
    from classical_diffusion.simulation import TimeSpan


def _get_key(key: jax.Array | None) -> jax.Array:
    return key if key is not None else jax.random.PRNGKey(0)


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
    def with_si_units(self) -> Self:
        """Return the rescaled simulation of the system."""
        si_units = UnitSystem()
        length_factor = si_units.angstrom / self.system.units.angstrom
        mass_factor = si_units.atomic_mass / self.system.units.atomic_mass
        energy_factor = si_units.Boltzmann / self.system.units.Boltzmann
        time_factor = np.sqrt(length_factor**2 * mass_factor / energy_factor)
        momentum_factor = mass_factor * length_factor / time_factor

        return type(self)(
            times=self.times * time_factor,
            x_points=self.x_points * length_factor,
            p_points=self._p_points * momentum_factor,
            system=self.system.with_si_units(),
        )


def _get_force_fn(
    system: System,
) -> Callable[[jnp.ndarray, tuple[float, ...]], jnp.ndarray]:
    """Compute a callable force function, taking and returning an array."""
    raw_fn = sp.lambdify(
        system.lambda_symbols,
        system.force_expr,
        modules=[{"DerivativeSafeMod": jnp.mod}, "jax"],
    )
    return lambda x_array, params: jnp.array(raw_fn(*x_array, *params))


@jax.jit
def _run_deterministic_ensemble_jit(
    system: "CanonicalSystem",  # ruff:ignore[quoted-annotation]
    xs0: jnp.ndarray,
    ps0: jnp.ndarray,
    times: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    force_fn = _get_force_fn(system)

    def vector_field(
        _t: Any,  # ruff:ignore[any-type]
        y: jnp.ndarray,
        _args: Any,  # ruff:ignore[any-type]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x, v = y
        return (v, force_fn(x, system.params) / system.m)

    term = dfx.ODETerm(vector_field)

    # Core solver for a single particle pair (x0, p0)
    def solve_one(x0: jnp.ndarray, p0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        sol = dfx.diffeqsolve(
            term,
            solver=dfx.Tsit5(),  # cspell: disable-line
            t0=0,
            t1=times[-1],
            dt0=times[1] - times[0],
            y0=(x0, p0),
            args=None,
            saveat=dfx.SaveAt(ts=times),
            stepsize_controller=dfx.PIDController(
                rtol=1e-6,  # cspell: disable-line
                atol=1e-8,
            ),  # cspell: disable-line
            max_steps=100_000_000,
        )
        return sol.ys

    return jax.vmap(solve_one, in_axes=(0, 0))(xs0, ps0)


@jax.jit
def _run_langevin_ensemble_jit(
    system: "CanonicalSystem",  # ruff:ignore[quoted-annotation]
    xs0: jnp.ndarray,
    ps0: jnp.ndarray,
    keys: jax.Array,
    times: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    gamma = jnp.broadcast_to(system.gamma, (system.n_dim,))
    u = jnp.broadcast_to(system.kbt / system.m, (system.n_dim,))
    force_fn = _get_force_fn(system)

    def grad_f(x: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return -force_fn(x, system.params) / system.kbt

    def solve_one(
        x0: jnp.ndarray, p0: jnp.ndarray, key: jax.Array
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        bm = dfx.VirtualBrownianTree(
            t0=0,
            t1=times[-1],
            tol=1e-3,
            shape=(system.n_dim,),
            key=key,
            levy_area=dfx.SpaceTimeTimeLevyArea,
        )

        drift_term = dfx.UnderdampedLangevinDriftTerm(gamma, u, grad_f)
        diffusion_term = dfx.UnderdampedLangevinDiffusionTerm(gamma, u, bm)
        terms = dfx.MultiTerm(drift_term, diffusion_term)

        sol = dfx.diffeqsolve(
            terms,
            solver=dfx.ALIGN(),
            t0=0,
            t1=times[-1],
            dt0=times[1] - times[0],
            y0=(x0, p0),
            args=None,
            stepsize_controller=dfx.PIDController(
                rtol=1e-2,  # cspell: disable-line
                atol=1e-3,
            ),
            saveat=dfx.SaveAt(ts=times),
            max_steps=100_000_000,
        )
        return sol.ys

    return jax.vmap(solve_one, in_axes=(0, 0, 0))(xs0, ps0, keys)


def _solve_ensemble_path[S: System](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[np.ndarray, np.ndarray],
    *,
    _key: jax.Array | None = None,
) -> Path:
    filename = f"{hash(system)}_{hash(time_span)}_{hash_array(initial_conditions)}.npz"
    return Path("examples/data") / filename


@cached(_solve_ensemble_path)
@timed
def solve_ensemble[S: System](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[
        np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
    ],
    *,
    _key: jax.Array | None = None,
) -> LangevinSimulationResult[S]:
    """Solve an ensemble of ULD Langevin trajectories in parallel via jax.vmap."""
    xs0_jax = jnp.asarray(initial_conditions[0])
    ps0_jax = jnp.asarray(initial_conditions[1])
    n_run = xs0_jax.shape[0]

    times = jnp.linspace(
        time_span.t_start, time_span.t_end, time_span.n_steps + 1, endpoint=True
    )

    if np.isclose(system.gamma, 0.0):
        xs_batch, ps_batch = _run_deterministic_ensemble_jit(
            system.as_canonical(), xs0_jax, ps0_jax, times
        )
    else:
        keys = jax.random.split(_get_key(_key), n_run)

        xs_batch, ps_batch = _run_langevin_ensemble_jit(
            system.as_canonical(), xs0_jax, ps0_jax, keys, times
        )

    # --- SHAPE TRANSFORMATION ---
    # Diffrax + vmap naturally outputs: (n_run, n_time, n_dim)
    # We transpose axes 1 and 2 to match your target layout: (n_run, n_dim, n_time)
    xs_batch = jnp.transpose(xs_batch, (0, 2, 1))
    ps_batch = jnp.transpose(ps_batch, (0, 2, 1))

    return LangevinSimulationResult(
        times=np.array(times),
        x_points=np.array(xs_batch),
        p_points=np.array(ps_batch),
        system=system,
    )


def _solve_single_path[S: System](
    system: S,
    time_span: TimeSpan,
    initial_condition: tuple[np.ndarray, np.ndarray],
    *,
    _key: jax.Array | None = None,
) -> Path:
    filename = f"{system.__class__.__name__}_{hash(system)}_{hash(time_span)}_{hash_array(initial_condition)}.npz"
    return Path("examples/data") / filename


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
    return solve_ensemble.load_or_call_uncached(
        system,
        time_span,
        (
            np.array([initial_condition[0]]),
            np.array([initial_condition[1]]),
        ),
        _key=_key,
    )[0]


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
    out = solve_ensemble.load_or_call_uncached(
        dataclasses.replace(system.as_canonical(), gamma=0.0),
        time_span,
        (
            np.array([initial_condition[0]]),
            np.array([initial_condition[1]]),
        ),
        _key=_key,
    )[0]

    return SingleLangevinSimulationResult(
        times=out.times,
        x_points=out.x_points,
        p_points=out.p_points,
        system=system,
    )


def _solve_ballistic_ensemble_path[S: System](
    system: S,
    time_span: TimeSpan,
    n_samples: int,
    *,
    _key: jax.Array | None = None,
) -> Path:
    filename = (
        f"{system.__class__.__name__}_{hash(system)}_{hash(time_span)}_{n_samples}.npz"
    )
    return Path("examples/data") / filename


def _get_initial_conditions(
    system: System, n_samples: int, *, _key: jax.Array | None = None
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
]:
    sampler = make_initial_conditions_sampler(system)
    keys = jax.random.split(_get_key(_key), n_samples)
    x_initial, p_initial = sampler(keys)
    x_initial = x_initial.reshape(-1, system.n_dim)
    p_initial = p_initial.reshape(-1, system.n_dim)
    return x_initial, p_initial


@cached(_solve_ballistic_ensemble_path)
@timed
def solve_ballistic_ensemble[S: System](
    system: S,
    time_span: TimeSpan,
    n_samples: int,
    *,
    _key: jax.Array | None = None,
) -> LangevinSimulationResult[S]:
    """Solve an ensemble of ballistic trajectories in parallel via jax.vmap."""
    _key = _get_key(_key)
    out = solve_ensemble.load_or_call_uncached(
        dataclasses.replace(system.as_canonical(), gamma=0.0),
        time_span,
        _get_initial_conditions(system, n_samples, _key=_key),
        _key=_key,
    )
    return LangevinSimulationResult(
        times=out.times,
        x_points=out.x_points,
        p_points=out.p_points,
        system=system,
    )


def _get_over_barrier_initial_conditions(
    system: System,
    barrier_energy: float,
    n_samples: int,
    *,
    _key: jax.Array | None = None,
) -> tuple:
    _key = _get_key(_key)
    sampler = make_free_point_sampler(system, barrier_energy)
    keys = jax.random.split(_key, n_samples)
    free_x_initial, free_p_initial = sampler(keys)
    free_x_initial = free_x_initial.reshape(-1, system.n_dim)
    free_p_initial = free_p_initial.reshape(-1, system.n_dim)
    return free_x_initial, free_p_initial


def _solve_over_barrier_ballistic_ensemble_path[S: System](
    system: S,
    time_span: TimeSpan,
    n_samples: int,
    barrier_energy: float,
    *,
    _key: jax.Array | None = None,
) -> Path:
    filename = f"over_barrier_{system.__class__.__name__}_{hash(system)}_{hash(time_span)}_{n_samples}_{barrier_energy}.npz"
    return Path("examples/data") / filename


@cached(_solve_over_barrier_ballistic_ensemble_path)
@timed
def solve_over_barrier_ballistic_ensemble[S: System](
    system: S,
    time_span: TimeSpan,
    n_samples: int,
    barrier_energy: float,
    *,
    _key: jax.Array | None = None,
) -> LangevinSimulationResult[S]:
    """Solve an ensemble of ballistic trajectories in parallel via jax.vmap."""
    _key = _get_key(_key)
    out = solve_ensemble.load_or_call_uncached(
        dataclasses.replace(system.as_canonical(), gamma=0.0),
        time_span,
        _get_over_barrier_initial_conditions(
            system,
            barrier_energy=barrier_energy,
            n_samples=n_samples,
            _key=_key,
        ),
        _key=_key,
    )
    return LangevinSimulationResult(
        times=out.times,
        x_points=out.x_points,
        p_points=out.p_points,
        system=system,
    )


@jax.jit
def _run_overdamped_ensemble_jit(
    system: "CanonicalSystem",  # ruff:ignore[quoted-annotation]
    xs0: jnp.ndarray,
    keys: jax.Array,
    times: jnp.ndarray,
) -> jnp.ndarray:
    gamma = jnp.broadcast_to(system.gamma, (system.n_dim,))
    force_fn = _get_force_fn(system)
    diffusion_matrix = jnp.diag(jnp.sqrt(2.0 * system.kbt / gamma))

    def solve_one(x0: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        bm = dfx.VirtualBrownianTree(
            t0=0,
            t1=times[-1],
            tol=1e-4,
            shape=(system.n_dim,),
            key=key,
            levy_area=dfx.SpaceTimeTimeLevyArea,
        )

        # dx = (F(x) / gamma) dt + sqrt(2 kB T / gamma) dW
        drift_term = dfx.ODETerm(
            lambda _t, x, _args: force_fn(x, system.params) / gamma
        )
        diffusion_term = dfx.ControlTerm(lambda _t, _x, _args: diffusion_matrix, bm)
        terms = dfx.MultiTerm(drift_term, diffusion_term)

        sol = dfx.diffeqsolve(
            terms,
            solver=dfx.ShARK(),
            t0=0,
            t1=times[-1],
            dt0=times[1] - times[0],
            y0=x0,
            args=None,
            stepsize_controller=dfx.ClipStepSizeController(
                dfx.PIDController(
                    rtol=1e-2,  # cspell: disable-line
                    atol=1e-3,
                ),
                step_ts=times,
            ),
            saveat=dfx.SaveAt(ts=times),
            max_steps=100_000_000,
        )
        return sol.ys

    return jax.vmap(solve_one, in_axes=(0, 0))(xs0, keys)


def _solve_overdamped_ensemble_path[S: System](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[
        np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
    ],
    *,
    _key: jax.Array | None = None,
) -> Path:
    filename = f"overdamped_{hash(system)}_{hash(time_span)}_{hash_array(initial_conditions)}.npz"
    return Path("examples/data") / filename


@cached(_solve_overdamped_ensemble_path)
@timed
def solve_overdamped_ensemble[S: System](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[
        np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
    ],
    *,
    _key: jax.Array | None = None,
) -> LangevinSimulationResult[S]:
    """Solve an ensemble of overdamped Langevin trajectories in parallel via jax.vmap."""
    n_run = initial_conditions[0].shape[0]

    times = jnp.linspace(
        time_span.t_start, time_span.t_end, time_span.n_steps + 1, endpoint=True
    )

    keys = jax.random.split(_get_key(_key), n_run)
    xs_batch = _run_overdamped_ensemble_jit(
        system.as_canonical(), initial_conditions[0], keys, times
    )

    xs_batch = jnp.transpose(xs_batch, (0, 2, 1))

    return LangevinSimulationResult(
        times=np.array(times),
        x_points=np.array(xs_batch),
        p_points=np.zeros_like(xs_batch),
        system=system,
    )
