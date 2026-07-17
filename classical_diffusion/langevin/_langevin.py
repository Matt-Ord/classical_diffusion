import zlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from classical_diffusion.util import cached, timed

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.system import System


@dataclass(frozen=True, kw_only=True)
class TimeSpan:
    """Time-stepping parameters, bundled together."""

    t0: float
    t1: float
    dt: float
    dt_step: float | None = None


@dataclass(frozen=True, kw_only=True)
class SingleSimulationResult[S: System[Any] = System[Any]]:
    """Results of a simulation of the periodic Langevin equation."""

    times: np.ndarray
    x_points: np.ndarray[Any, np.dtype[np.floating]]
    p_points: np.ndarray[Any, np.dtype[np.floating]]
    system: S


@dataclass(frozen=True, kw_only=True)
class SimulationResult[S: System[Any] = System[Any]]:
    """Results of a simulation of the periodic Langevin equation."""

    times: np.ndarray
    x_points: np.ndarray[Any, np.dtype[np.floating]]
    p_points: np.ndarray[Any, np.dtype[np.floating]]
    system: S

    def __getitem__(self, idx: int) -> SingleSimulationResult[S]:
        """Return a single trajectory from the ensemble."""
        return SingleSimulationResult(
            times=self.times,
            x_points=self.x_points[idx],
            p_points=self.p_points[idx],
            system=self.system,
        )


def _get_force_fn(
    system: System,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute a callable force function, taking and returning an array."""
    raw_fn = sp.lambdify(system.symbolic_coordinates, system.force_expr, "jax")
    return lambda x_array: jnp.array(raw_fn(*x_array))


def sample_results[S: System](
    result: SimulationResult[S],
    *,
    stride_time: float,
) -> SimulationResult[S]:
    """Subsample all trajectories at the parameters' stride, along the saved-time axis."""
    stride = int(stride_time / (result.times[1] - result.times[0]))
    return SimulationResult(
        times=result.times[::stride],
        x_points=result.x_points[:, :, ::stride],
        p_points=result.p_points[:, :, ::stride],
        system=result.system,
    )


@partial(jax.jit, static_argnames=("system"))
def _run_deterministic_ensemble_jit(
    system: "System",  # noqa: UP037
    xs0: jnp.ndarray,
    ps0: jnp.ndarray,
    times: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    force_fn = _get_force_fn(system)

    def vector_field(
        _t: Any,  # noqa: ANN401
        y: jnp.ndarray,
        _args: Any,  # noqa: ANN401
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x, v = y
        return (v, force_fn(x) / system.m)

    term = dfx.ODETerm(vector_field)

    # Core solver for a single particle pair (x0, p0)
    def solve_one(x0: jnp.ndarray, p0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        sol = dfx.diffeqsolve(
            term,
            solver=dfx.Tsit5(),
            t0=0,
            t1=times[-1],
            dt0=times[1] - times[0],
            y0=(x0, p0),
            args=None,
            saveat=dfx.SaveAt(ts=times),
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-5),
            max_steps=100_000_000,
        )
        return sol.ys

    return jax.vmap(solve_one, in_axes=(0, 0))(xs0, ps0)


@partial(jax.jit, static_argnames=("system", "dt_step"))
def _run_langevin_ensemble_jit(  # noqa: PLR0913, PLR0917
    system: "System",  # noqa: UP037
    xs0: jnp.ndarray,
    ps0: jnp.ndarray,
    keys: jax.Array,
    times: jnp.ndarray,
    dt_step: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    gamma = jnp.broadcast_to(system.gamma, (system.n_dim,))
    u = jnp.broadcast_to(system.kbt / system.m, (system.n_dim,))
    force_fn = _get_force_fn(system)

    def grad_f(x: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return -force_fn(x) / system.kbt

    def solve_one(
        x0: jnp.ndarray, p0: jnp.ndarray, key: jax.Array
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        bm = dfx.VirtualBrownianTree(
            t0=0,
            t1=times[-1],
            tol=dt_step / 2,
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
            dt0=dt_step,
            y0=(x0, p0),
            args=None,
            saveat=dfx.SaveAt(ts=times),
            max_steps=100_000_000,
        )
        return sol.ys

    return jax.vmap(solve_one, in_axes=(0, 0, 0))(xs0, ps0, keys)


def _hash_initial_conditions(initial_conditions: tuple[np.ndarray, np.ndarray]) -> int:
    chk = 0
    for arr in initial_conditions:
        # Chain the CRC32 checksums of the raw float bytes
        chk = zlib.crc32(arr.tobytes(), chk)
        # Include shape just in case the same floats are reshaped
        chk = zlib.crc32(str(arr.shape).encode(), chk)
    return chk


def _solve_ensemble_path[S: System](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[np.ndarray, np.ndarray],
    _key: jax.Array,
) -> Path:
    filename = f"{hash(system)}_{hash(time_span)}_{_hash_initial_conditions(initial_conditions)}.npz"
    return Path("examples/data") / filename


@cached(_solve_ensemble_path, default_call="load_or_call_uncached")
@timed
def solve_ensemble[S: System](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[
        np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
    ],
    _key: jax.Array,
) -> SimulationResult[S]:
    """Solve an ensemble of ULD Langevin trajectories in parallel via jax.vmap."""
    xs0_jax = jnp.asarray(initial_conditions[0])
    ps0_jax = jnp.asarray(initial_conditions[1])
    n_run = xs0_jax.shape[0]

    times = jnp.linspace(
        time_span.t0,
        time_span.t1,
        int((time_span.t1 - time_span.t0) / time_span.dt) + 1,
    )

    if np.isclose(system.gamma, 0.0):
        # Deterministic batch run
        xs_batch, ps_batch = _run_deterministic_ensemble_jit(
            system, xs0_jax, ps0_jax, times
        )
    else:
        # Stochastic batch run: Calculate safe physical timestep criteria
        dt_friction_limit = 0.05 / system.gamma if system.gamma > 0 else time_span.dt
        dt_step = (
            jnp.minimum(time_span.dt, dt_friction_limit)
            if time_span.dt_step is None
            else time_span.dt_step
        )

        # Vectorized generation of independent noise seeds per run
        keys = jax.random.split(_key, n_run)

        xs_batch, ps_batch = _run_langevin_ensemble_jit(
            system, xs0_jax, ps0_jax, keys, times, float(dt_step)
        )

    # --- SHAPE TRANSFORMATION ---
    # Diffrax + vmap naturally outputs: (n_run, n_time, n_dim)
    # We transpose axes 1 and 2 to match your target layout: (n_run, n_dim, n_time)
    xs_batch = jnp.transpose(xs_batch, (0, 2, 1))
    ps_batch = jnp.transpose(ps_batch, (0, 2, 1))

    return SimulationResult[S](
        times=np.array(times),
        x_points=np.array(xs_batch),
        p_points=np.array(ps_batch),
        system=system,
    )


def _solve_single_path[S: System](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[np.ndarray, np.ndarray],
    _key: jax.Array,
) -> Path:
    filename = f"{hash(system)}_{hash(time_span)}_{_hash_initial_conditions(initial_conditions)}.npz"
    return Path("examples/data") / filename


@cached(_solve_single_path, default_call="load_or_call_uncached")
@timed
def solve_single[S: System](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[
        np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
    ],
    _key: jax.Array,
) -> SingleSimulationResult[S]:
    """Solve the ULD Langevin equation for an ensemble of trajectories via vmap."""
    return solve_ensemble.load_or_call_uncached(
        system,
        time_span,
        (
            np.array([[initial_conditions[0][0]]]),
            np.array([[initial_conditions[1][0]]]),
        ),
        _key,
    )[0]
