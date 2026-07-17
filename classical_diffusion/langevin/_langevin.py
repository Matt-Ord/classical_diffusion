from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import sympy as sp
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.system import System


from pathlib import Path
from typing import (
    TYPE_CHECKING,
)

from classical_diffusion.util import cached


@dataclass(frozen=True, kw_only=True)
class TimeSpan:
    """Time-stepping parameters, bundled together."""

    t0: float
    t1: float
    dt: float


_EXPECTED_NDIM = 2


@dataclass(frozen=True, kw_only=True)
class InitialConditions:
    """initial conditions, bundled together."""

    x0: np.ndarray[Any, np.dtype[np.floating]]
    p0: np.ndarray[Any, np.dtype[np.floating]]

    def __post_init__(self) -> None:
        if self.x0.ndim != _EXPECTED_NDIM or self.p0.ndim != _EXPECTED_NDIM:
            msg = (
                f"x0/p0 must be 2D (n_trajectories, n_dims), got "
                f"x0.shape={self.x0.shape}, p0.shape={self.p0.shape}"
            )
            raise ValueError(msg)
        if self.x0.shape != self.p0.shape:
            msg = f"x0 shape {self.x0.shape} != p0 shape {self.p0.shape}"
            raise ValueError(msg)


class SimulationParameters[S: System = System]:
    """Parameters for the Langevin equation."""

    def __init__(
        self,
        *,
        system: S,
        time_span: TimeSpan,
        initial_conditions: InitialConditions,
    ) -> None:
        self.system = system

        assert system.n_dim == initial_conditions.x0.shape[1], (
            f"system.n_dim={system.n_dim} != initial_conditions.x0.shape[1]={initial_conditions.x0.shape[1]}"
        )
        self.time_span = time_span
        self.initial_conditions = initial_conditions

    @cached_property
    def n_trajectories(self) -> int:
        """Return the number of simulated trajectories."""
        return self.initial_conditions.x0.shape[0]

    @cached_property
    def n_dimensions(self) -> int:
        """Return the dimensions of the system."""
        return self.initial_conditions.x0.shape[1]

    @cached_property
    def n_save(self) -> int:
        """Return the number of saved steps."""
        return int((self.time_span.t1 - self.time_span.t0) / self.time_span.dt)

    def with_initial_conditions(
        self, initial_conditions: InitialConditions
    ) -> SimulationParameters:
        return SimulationParameters(
            system=self.system,
            time_span=self.time_span,
            initial_conditions=initial_conditions,
        )


@dataclass(frozen=True, kw_only=True)
class SimulationResult[S: System[Any] = System[Any]]:
    """Results of a simulation of the periodic Langevin equation."""

    times: np.ndarray
    xs: np.ndarray[Any, np.dtype[np.floating]]
    ps: np.ndarray[Any, np.dtype[np.floating]]
    system: S


def _my_path[S: System](params: SimulationParameters[S], _key: jax.Array) -> Path:
    ts = params.time_span

    parts = [
        f"{hash(params.system)}",
        f"t0-{ts.t0:.4g}",
        f"t1-{ts.t1:.4g}",
        f"dt{ts.dt:.4g}",
        f"n_trajectory{params.n_trajectories}",
        f"n_dims{params.n_dimensions}",
    ]

    filename = "_".join(parts) + ".npz"
    return Path("examples/data") / filename


def _get_force_fn(
    params: SimulationParameters[Any],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute a callable force function, taking and returning an array."""
    raw_fn = sp.lambdify(
        params.system.symbolic_coordinates, params.system.force_expr, "jax"
    )
    return lambda x_array: jnp.array(raw_fn(*x_array))


def solve_batch[S: System](
    params: SimulationParameters[S], _key: jax.Array
) -> SimulationResult[S]:
    """Solve the ULD Langevin equation for an ensemble of trajectories via vmap."""
    n_dims = params.n_dimensions
    gamma = jnp.broadcast_to(params.system.gamma, (n_dims,))
    u = jnp.broadcast_to(params.system.kbt / params.system.m, (n_dims,))
    force_fn = _get_force_fn(params)

    def grad_f(x: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return -force_fn(x) / params.system.kbt

    def solve_single(
        key: jax.Array, x0: jnp.ndarray, p0: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        bm = dfx.VirtualBrownianTree(
            params.time_span.t0,
            params.time_span.t1,
            tol=1e-4,
            shape=(n_dims,),
            key=key,
            levy_area=dfx.SpaceTimeTimeLevyArea,
        )
        drift_term = dfx.UnderdampedLangevinDriftTerm(gamma, u, grad_f)
        diffusion_term = dfx.UnderdampedLangevinDiffusionTerm(gamma, u, bm)
        terms = dfx.MultiTerm(drift_term, diffusion_term)

        sol = dfx.diffeqsolve(
            terms,
            solver=dfx.ALIGN(),
            t0=params.time_span.t0,
            t1=params.time_span.t1,
            dt0=params.time_span.dt,
            y0=(x0, p0),
            args=None,
            saveat=dfx.SaveAt(
                ts=jnp.linspace(params.time_span.t0, params.time_span.t1, params.n_save)
            ),
            max_steps=100_000_000,
        )
        return sol.ts, sol.ys  # ys is (x, v) tuple pytree

    keys = jrandom.split(_key, params.n_trajectories)
    times, (xs_batch, ps_batch) = jax.vmap(solve_single)(
        keys, params.initial_conditions.x0, params.initial_conditions.p0
    )

    return SimulationResult[S](
        times=np.array(times[0]),
        xs=np.array(xs_batch).transpose(0, 2, 1),  # -> (n_trajectories, n_dims, n_save)
        ps=np.array(ps_batch).transpose(0, 2, 1),
        system=params.system,
    )


def sample_result[S: System](
    result: SimulationResult[S],
    *,
    stride_time: float,
) -> SimulationResult[S]:
    """Subsample all trajectories at the parameters' stride, along the saved-time axis."""
    stride = int(stride_time / (result.times[1] - result.times[0]))
    return SimulationResult(
        times=result.times[::stride],
        xs=result.xs[:, :, ::stride],
        ps=result.ps[:, :, ::stride],
        system=result.system,
    )


def fold_result[S: System](
    result: SimulationResult[S],
    delta: float,
) -> SimulationResult[S]:
    """Fold x into first BZ zone."""
    return SimulationResult(
        times=result.times,
        xs=result.xs % delta,
        ps=result.ps,
        system=result.system,
    )


def truncate_trajectories[S: System](
    params: SimulationParameters[S], chunk_i: int, chunk_size: int
) -> SimulationParameters[S]:
    """Return a copy of params restricted to one chunk of trajectories."""
    start = chunk_i * chunk_size
    end = (chunk_i + 1) * chunk_size
    return SimulationParameters(
        system=params.system,
        time_span=params.time_span,
        initial_conditions=InitialConditions(
            x0=params.initial_conditions.x0[start:end],
            p0=params.initial_conditions.p0[start:end],
        ),
    )


def concatenate_results[S: System](
    results: list[SimulationResult[S]],
) -> SimulationResult[S]:
    """Concatenate a list of chunked results, merging trajectories and initial conditions."""
    return SimulationResult(
        times=results[0].times,
        xs=np.concatenate([r.xs for r in results], axis=0),
        ps=np.concatenate([r.ps for r in results], axis=0),
        system=results[0].system,
    )


@cached(_my_path, default_call="load_or_call_uncached")
def solve_ensemble[S: System](
    params: SimulationParameters[S], _key: jax.Array, chunk_size: int = 20
) -> SimulationResult[S]:
    """Batch simulation to retain some parallelisation while adding a progress bar."""
    if chunk_size > params.n_trajectories:
        return solve_batch(params=params, _key=_key)

    results = []
    n_chunks = (params.n_trajectories + chunk_size - 1) // chunk_size
    for i in tqdm(range(n_chunks), desc="Solving trajectories"):
        chunk_params = truncate_trajectories(params, chunk_size=chunk_size, chunk_i=i)
        chunk_result = solve_batch(params=chunk_params, _key=_key)
        results.append(chunk_result)
    return concatenate_results(results)
