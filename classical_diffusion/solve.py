import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable


from typing import (
    TYPE_CHECKING,
)


@dataclass(frozen=True, kw_only=True)
class TimeSpan:
    """Time-stepping parameters, bundled together."""

    t0: float
    t1: float
    dt: float
    n_save: int


@dataclass(frozen=True, kw_only=True)
class InitialConditions:
    """initial conditions, bundled together."""

    x0: np.ndarray
    p0: np.ndarray


@dataclass(frozen=True, kw_only=True)
class CharacteristicValues:
    """characteristic values, bundled together."""

    length: float
    time: float


@dataclass(frozen=True, kw_only=True)
class PhysicalParams:
    """physical parameters, bundled together."""

    gamma: float
    temp: float
    m: float

    @cached_property
    def kbt(self) -> float:
        """Return the kbt of system."""
        return 1.0 * self.temp  # k_B set to 1 for now

    @cached_property
    def sigma(self) -> float:
        """Return the noise coefficient of the system."""
        return np.sqrt(2 * self.gamma * self.kbt)


class SimulationParams:
    """Parameters for the Langevin equation."""

    def __init__(
        self,
        *,
        physical_parameters: PhysicalParams,
        time_span: TimeSpan,
        initial_conditions: InitialConditions,
        potential: sp.Expr,
        characteristic_values: CharacteristicValues,
    ) -> None:
        self.physical = physical_parameters
        self.time_span = time_span
        self.initial_conditions = initial_conditions
        self.potential = potential
        self.characteristic_values = characteristic_values

    @cached_property
    def n_dimensions(self) -> int:
        """Return the dimensions of the system."""
        return len(self.initial_conditions.x0)

    @cached_property
    def sigma(self) -> float:
        """Return the noise coefficient of the system."""
        return np.sqrt(
            2 * self.physical.gamma * 1.0 * self.physical.temp
        )  # k_B set to 1 for now

    @cached_property
    def symbolic_coordinates(self) -> tuple:
        """Return the symbolic coordinates of the system."""
        return sp.symbols(f"x0:{self.n_dimensions}")

    @cached_property
    def symbolic_force(self) -> list[sp.Expr]:
        """Return the symbolic force of the system."""
        return [-sp.diff(self.potential, c) for c in self.symbolic_coordinates]

    @cached_property
    def force_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Compute a callable force function, taking and returning an array."""
        raw_fn = sp.lambdify(self.symbolic_coordinates, self.symbolic_force, "jax")
        return lambda x_array: jnp.array(raw_fn(*x_array))

    @cached_property
    def n_sample(self) -> int:
        """Compute the number of steps to obtain correct sample spacing."""
        return math.floor(
            self.time_span.t1 - self.time_span.t0 / self.characteristic_values.time
        )


@dataclass(frozen=True, kw_only=True)
class SimulationResult:
    """Results of a simulation of the periodic Langevin equation."""

    times: np.ndarray
    # Maybe x_points, p_points is clearer?
    xs: np.ndarray[Any, np.dtype[np.floating]]
    ps: np.ndarray[Any, np.dtype[np.floating]]

    params: SimulationParams


def make_solver(params: SimulationParams) -> tuple:
    """Generate drift and diffusion functions tailored to a specific potential."""
    n_dims = params.n_dimensions

    def drift(_t: float, y: jnp.ndarray, args: tuple) -> jnp.ndarray:
        """Drift function for the Langevin equation."""
        x, p = y[:n_dims], y[n_dims:]
        m, gamma, _sigma = args
        force = params.force_fn(x)
        return jnp.concatenate([p / m, force - gamma * p])

    def diffusion(_t: float, _y: jnp.ndarray, args: tuple) -> jnp.ndarray:
        _m, _gamma, sigma = args
        # creates 2*n_dims by n_dims matrix matching diffrax requirements
        sigma_arr = jnp.broadcast_to(sigma, (n_dims,))
        top = jnp.zeros((n_dims, n_dims))
        bottom = jnp.diag(sigma_arr)
        return jnp.concatenate([top, bottom], axis=0)

    return drift, diffusion


def solve_langevin(params: SimulationParams, key: jax.Array) -> SimulationResult:
    """Solve the Langevin equation using diffrax."""
    drift, diffusion = make_solver(params)
    bm = dfx.VirtualBrownianTree(
        params.time_span.t0,
        params.time_span.t1,
        tol=1e-4,
        shape=(params.n_dimensions,),
        key=key,
    )

    terms = dfx.MultiTerm(
        dfx.ODETerm(drift),
        dfx.ControlTerm(diffusion, bm),
    )

    sol = dfx.diffeqsolve(  # cspell: disable-line
        terms,
        solver=dfx.EulerHeun(),  # cspell: disable-lin,
        t0=params.time_span.t0,
        t1=params.time_span.t1,
        dt0=params.time_span.dt,
        y0=jnp.concatenate(
            [params.initial_conditions.x0, params.initial_conditions.p0]
        ),
        args=(
            params.physical.m,
            params.physical.gamma,
            params.physical.sigma,
        ),
        saveat=dfx.SaveAt(
            ts=jnp.linspace(
                params.time_span.t0, params.time_span.t1, params.time_span.n_save
            )
        ),  # cspell: disable-line
        max_steps=100_000_000,
    )

    return SimulationResult(
        times=np.array(sol.ts),
        xs=np.array(sol.ys[:, : params.n_dimensions]).T,
        ps=np.array(sol.ys[:, params.n_dimensions :]).T,
        params=params,
    )
