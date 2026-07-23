import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from scipy.stats.sampling import NumericalInversePolynomial

from classical_diffusion.util import cached

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.system import CanonicalSystem, System


@dataclass(frozen=True, kw_only=True)
class TimeSpan:
    """Time-stepping parameters, bundled together."""

    t0: float
    t1: float
    dt: float
    dt_step: float | None = None

    def __post_init__(self) -> None:
        if self.t1 <= self.t0:
            msg = f"t1 must be greater than t0, got t0={self.t0}, t1={self.t1}"
            raise ValueError(msg)
        if self.dt <= 0:
            msg = f"dt must be positive, got dt={self.dt}"
            raise ValueError(msg)
        if self.n_steps <= 1:
            msg = f"Time span must have at least 2 steps, got n_steps={self.n_steps}"
            raise ValueError(msg)

    @property
    def n_steps(self) -> int:
        """The number of steps in the time span."""
        return int((self.t1 - self.t0) / self.dt) + 1


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
) -> Callable[[jnp.ndarray, tuple[float, ...]], jnp.ndarray]:
    """Compute a callable force function, taking and returning an array."""
    raw_fn = sp.lambdify(system.lambda_symbols, system.force_expr, "jax")
    return lambda x_array, params: jnp.array(raw_fn(*x_array, *params))


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
def _run_langevin_ensemble_jit(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    system: "CanonicalSystem",  # ruff:ignore[quoted-annotation]
    xs0: jnp.ndarray,
    ps0: jnp.ndarray,
    keys: jax.Array,
    times: jnp.ndarray,
    dt_step: jnp.ndarray,
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
        chk = zlib.crc32(arr.tobytes(), chk)  # cspell: disable-line
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


@cached(_solve_ensemble_path)
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
        time_span.n_steps,
        endpoint=True,
    )

    if np.isclose(system.gamma, 0.0):
        xs_batch, ps_batch = _run_deterministic_ensemble_jit(
            system.as_canonical(), xs0_jax, ps0_jax, times
        )
    else:
        dt_friction_limit = 0.05 / system.gamma if system.gamma > 0 else time_span.dt
        dt_step = (
            jnp.minimum(time_span.dt, dt_friction_limit)
            if time_span.dt_step is None
            else time_span.dt_step
        )

        # Vectorized generation of independent noise seeds per run
        keys = jax.random.split(_key, n_run)

        xs_batch, ps_batch = _run_langevin_ensemble_jit(
            system.as_canonical(), xs0_jax, ps0_jax, keys, times, jnp.asarray(dt_step)
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
    initial_condition: tuple[np.ndarray, np.ndarray],
    _key: jax.Array,
) -> Path:
    filename = f"{hash(system)}_{hash(time_span)}_{_hash_initial_conditions(initial_condition)}.npz"
    return Path("examples/data") / filename


@cached(_solve_single_path)
def solve_single[S: System](
    system: S,
    time_span: TimeSpan,
    initial_condition: tuple[
        np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
    ],
    _key: jax.Array,
) -> SingleSimulationResult[S]:
    """Solve the ULD Langevin equation for a single trajectory via vmap."""
    return solve_ensemble.load_or_call_uncached(
        system,
        time_span,
        (
            np.array([initial_condition[0]]),
            np.array([initial_condition[1]]),
        ),
        _key,
    )[0]


def _solve_ballistic_ensemble_path[S: System](
    system: S,
    time_span: TimeSpan,
    n_trajectories: int,
    _key: jax.Array,
) -> Path:
    filename = f"{hash(system)}_{hash(time_span)}_{n_trajectories}.npz"
    return Path("examples/data") / filename


def sample_x_initial(system: System, n_trajectories: int) -> np.ndarray:
    x0, *_ = system.coordinate_symbols
    potential_fn = sp.lambdify(
        (x0, *system.parameter_symbols), system.potential_expr, "numpy"
    )
    kbt = system.kbt
    params = system.params

    class XDensity:
        @staticmethod
        def pdf(x: float) -> float:
            return np.exp(-potential_fn(x, *params) / kbt)

    x_sampler = NumericalInversePolynomial(XDensity(), domain=system.sampling_domain)  # ty:ignore[invalid-argument-type]
    return x_sampler.rvs(size=n_trajectories).reshape(n_trajectories, system.n_dim)


def sample_p_initial(system: System, n_trajectories: int) -> np.ndarray:

    rng = np.random.default_rng()
    p_std = np.sqrt(system.kbt * system.m)

    return rng.normal(0.0, p_std, size=(n_trajectories, system.n_dim))


@cached(_solve_ballistic_ensemble_path)
def solve_ballistic_ensemble[S: System](
    system: S,
    time_span: TimeSpan,
    n_trajectories: int,
    _key: jax.Array,
) -> SimulationResult[S]:
    """Solve an ensemble of ballistic trajectories in parallel via jax.vmap."""
    return solve_ensemble.load_or_call_uncached(
        system.with_gamma(0.0),
        time_span,
        (
            sample_x_initial(system=system, n_trajectories=n_trajectories),
            sample_p_initial(system=system, n_trajectories=n_trajectories),
        ),
        _key,
    )
