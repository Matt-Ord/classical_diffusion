from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.system import System


from pathlib import Path
from typing import (
    TYPE_CHECKING,
)

from classical_diffusion.util import cached


# TODO: split into "physical system" code, and "simulation" code.
@dataclass(frozen=True, kw_only=True)
class TimeSpan:
    """Time-stepping parameters, bundled together."""

    t0: float
    t1: float
    dt: float
    # TODO: burn in??
    burn_in: int


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
    def burn_in_time(self) -> float:
        """Return the number of saved steps."""
        return self.time_span.burn_in

    @cached_property
    def n_save(self) -> int:
        """Return the number of saved steps."""
        return int((self.time_span.t1 - self.burn_in_time) / self.time_span.dt)


@dataclass(frozen=True, kw_only=True)
class SimulationResult[P: SimulationParameters[Any] = SimulationParameters[Any]]:
    """Results of a simulation of the periodic Langevin equation."""

    # TODO: duplication here!
    times: np.ndarray
    xs: np.ndarray[Any, np.dtype[np.floating]]
    ps: np.ndarray[Any, np.dtype[np.floating]]
    params: P


# TODO: discuss __hash__ as a nice way to do this
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


@cached(_my_path, default_call="load_or_call_uncached")
def solve_ensemble[S: System](
    params: SimulationParameters[S], _key: jax.Array
) -> SimulationResult[SimulationParameters[S]]:
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
                ts=jnp.linspace(params.burn_in_time, params.time_span.t1, params.n_save)
            ),
            max_steps=100_000_000,
        )
        return sol.ts, sol.ys  # ys is (x, v) tuple pytree

    keys = jrandom.split(_key, params.n_trajectories)
    ts_batch, (xs_batch, ps_batch) = jax.vmap(solve_single)(
        keys, params.initial_conditions.x0, params.initial_conditions.p0
    )

    return SimulationResult[SimulationParameters[S]](
        # TODO: discuss duplicating data
        times=np.array(ts_batch[0]),
        xs=np.array(xs_batch).transpose(0, 2, 1),  # -> (n_trajectories, n_dims, n_save)
        ps=np.array(ps_batch).transpose(0, 2, 1),
        params=params,
    )


def sample_result[P: SimulationParameters](
    result: SimulationResult[P],
    *,
    stride: int,
) -> SimulationResult[P]:
    """Subsample all trajectories at the parameters' stride, along the saved-time axis."""
    return SimulationResult(
        times=result.times[::stride],
        xs=result.xs[:, :, ::stride],
        ps=result.ps[:, :, ::stride],
        params=result.params,
    )


def fold_result[P: SimulationParameters](
    result: SimulationResult[P],
    delta: float,
) -> SimulationResult[P]:
    """Fold x into first BZ zone."""
    return SimulationResult(
        times=result.times,
        xs=result.xs % delta,
        ps=result.ps,
        params=result.params,
    )
