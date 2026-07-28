import dataclasses
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from scipy.stats.sampling import NumericalInversePolynomial

from classical_diffusion.system import (
    System,
    UnitSystem,
    get_energy,
)
from classical_diffusion.util import cached, timed

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.system import (
        CanonicalSystem,
    )

rng = np.random.default_rng()


@dataclass(frozen=True, kw_only=True)
class TimeSpan:
    """Time-stepping parameters, bundled together."""

    t0: float
    t1: float
    n_steps: int

    def __post_init__(self) -> None:
        if self.t1 <= self.t0:
            msg = f"t1 must be greater than t0, got t0={self.t0}, t1={self.t1}"
            raise ValueError(msg)
        if self.n_steps <= 1:
            msg = f"Time span must have at least 2 steps, got n_steps={self.n_steps}"
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class SingleSimulationResult[S: System[Any] = System[Any]]:
    """Results of a simulation of the periodic Langevin equation."""

    times: np.ndarray
    x_points: np.ndarray[Any, np.dtype[np.floating]]
    p_points: np.ndarray[Any, np.dtype[np.floating]]
    system: S

    def with_si_units(self) -> Self:
        """Return the rescaled simulation of the system."""
        si_units = UnitSystem()
        length_factor = si_units.angstrom / self.system.units.angstrom
        mass_factor = si_units.atomic_mass / self.system.units.atomic_mass
        energy_factor = si_units.Boltzmann / self.system.units.kb
        time_factor = np.sqrt(length_factor**2 * mass_factor / energy_factor)
        momentum_factor = mass_factor * length_factor / time_factor
        return dataclasses.replace(
            self,
            times=self.times * time_factor,
            x_points=self.x_points * length_factor,
            p_points=self.p_points * momentum_factor,
            system=self.system.with_si_units(),
        )


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

    def with_si_units(self) -> Self:
        """Return the rescaled simulation of the system."""
        si_units = UnitSystem()
        length_factor = si_units.angstrom / self.system.units.angstrom
        mass_factor = si_units.atomic_mass / self.system.units.atomic_mass
        energy_factor = si_units.Boltzmann / self.system.units.Boltzmann
        time_factor = np.sqrt(length_factor**2 * mass_factor / energy_factor)
        momentum_factor = mass_factor * length_factor / time_factor
        return dataclasses.replace(
            self,
            times=self.times * time_factor,
            x_points=self.x_points * length_factor,
            p_points=self.p_points * momentum_factor,
            system=self.system.with_si_units(),
        )


def _get_force_fn(
    system: System,
) -> Callable[[jnp.ndarray, tuple[float, ...]], jnp.ndarray]:
    """Compute a callable force function, taking and returning an array."""
    raw_fn = sp.lambdify(system.lambda_symbols, system.force_expr, "jax")
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
        time_span.n_steps,
        endpoint=True,
    )

    if np.isclose(system.gamma, 0.0):
        xs_batch, ps_batch = _run_deterministic_ensemble_jit(
            system.as_canonical(), xs0_jax, ps0_jax, times
        )
    else:
        # Vectorized generation of independent noise seeds per run
        keys = jax.random.split(_key, n_run)

        xs_batch, ps_batch = _run_langevin_ensemble_jit(
            system.as_canonical(), xs0_jax, ps0_jax, keys, times
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
@timed
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
    n_samples: int,
    _key: jax.Array,
) -> Path:
    filename = f"{hash(system)}_{hash(time_span)}_{n_samples}.npz"
    return Path("examples/data") / filename


def sample_x_initial(
    system: System, n_samples: int
) -> np.ndarray[Any, np.dtype[np.floating]]:
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

        @staticmethod
        def cdf(x: float) -> float:
            msg = "CDF is not implemented for XDensity."
            raise NotImplementedError(msg)

        @staticmethod
        def logpdf(x: float) -> float:
            return -potential_fn(x, *params) / kbt

    x_sampler = NumericalInversePolynomial(XDensity(), domain=system.sampling_domain)
    return x_sampler.rvs(size=n_samples).reshape(n_samples, system.n_dim)


def sample_p_initial(
    system: System, n_samples: int
) -> np.ndarray[Any, np.dtype[np.floating]]:

    p_std = np.sqrt(system.kbt * system.m)

    return rng.normal(0.0, p_std, size=(n_samples, system.n_dim))


def make_free_point_sampler(system: System, barrier_energy: float):
    x0, *_ = system.coordinate_symbols
    potential_fn = sp.lambdify(
        (x0, *system.parameter_symbols), system.potential_expr, "jax"
    )
    sigma = jnp.sqrt(system.m * system.kbt)
    x_lo, x_hi = system.sampling_domain

    def sample_one(key):
        def cond_fn(state):
            _, _, _, accepted = state
            return ~accepted

        def body_fn(state):
            key, _x, _p, _ = state
            key, kx, ku, kp = jax.random.split(key, 4)

            x_candidate = jax.random.uniform(kx, minval=x_lo, maxval=x_hi)
            V = potential_fn(x_candidate, *system.params)
            x_ok = jax.random.uniform(ku) < jnp.exp(-V / system.kbt)

            p_candidate = jax.random.normal(kp) * sigma
            energy = p_candidate**2 / (2 * system.m) + V

            accept = x_ok & (energy > barrier_energy)
            return key, x_candidate, p_candidate, accept

        init = (key, jnp.array(0.0), jnp.array(0.0), jnp.array(False))
        _, x_final, p_final, _ = jax.lax.while_loop(cond_fn, body_fn, init)
        return x_final, p_final

    return jax.jit(jax.vmap(sample_one))


@cached(_solve_ballistic_ensemble_path)
@timed
def solve_ballistic_ensemble[S: System](
    system: S,
    time_span: TimeSpan,
    n_samples: int,
    _key: jax.Array,
) -> SimulationResult[S]:
    """Solve an ensemble of ballistic trajectories in parallel via jax.vmap."""
    return solve_ensemble.load_or_call_uncached(
        system.with_gamma(0.0),
        time_span,
        (
            sample_x_initial(system=system, n_samples=n_samples),
            sample_p_initial(system=system, n_samples=n_samples),
        ),
        _key,
    )


def split_escaped_and_trapped(result: SimulationResult, barrier_energy: float) -> tuple:
    """Split result into trajectories trapped within or free to move over the barrier."""
    energy = get_energy(result.system, result.x_points, result.p_points)[:, 0]

    free_mask = energy > barrier_energy
    free_x_points, free_p_points = (
        result.x_points[free_mask],
        result.p_points[free_mask],
    )

    trapped_mask = energy <= barrier_energy
    trapped_x_points, trapped_p_points = (
        result.x_points[trapped_mask],
        result.p_points[trapped_mask],
    )

    free_result = dataclasses.replace(
        result, x_points=free_x_points, p_points=free_p_points
    )
    trapped_result = dataclasses.replace(
        result, x_points=trapped_x_points, p_points=trapped_p_points
    )

    return free_result, trapped_result


def solve_free_ballistic_ensemble[S: System](
    system: System,
    time_span: TimeSpan,
    n_samples: int,
    _key: jax.Array,
    barrier_energy: float,
) -> SimulationResult[S]:
    """Take an ensemble of uniformly distributed ballistic trajectories and solve free trajectories in parallel via jax.vmap. Also returns probability of being above the barrier."""
    sampler = make_free_point_sampler(system, barrier_energy)
    keys = jax.random.split(jax.random.PRNGKey(0), n_samples)
    free_x_initial, free_p_initial = sampler(keys)
    free_x_initial = free_x_initial.reshape(-1, system.n_dim)
    free_p_initial = free_p_initial.reshape(-1, system.n_dim)

    return solve_ensemble.load_or_call_uncached(
        system.with_gamma(0.0),
        time_span,
        (free_x_initial, free_p_initial),
        _key,
    )
