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


from pathlib import Path
from typing import (
    TYPE_CHECKING,
)

from .util import cached


# TODO: split into "physical system" code, and "simulation" code.
@dataclass(frozen=True, kw_only=True)
class TimeSpan:
    """Time-stepping parameters, bundled together."""

    t0: float
    t1: float
    dt: float
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


@dataclass(frozen=True, kw_only=True)
class CharacteristicValues:
    """characteristic values, bundled together."""

    length: float
    time: float


# TODO: should include V(x)??
@dataclass(frozen=True, kw_only=True)
class PhysicalParameters:
    """physical parameters, bundled together."""

    gamma: float
    temperature: float
    m: float

    @cached_property
    def kbt(self) -> float:
        """Return the kbt of system."""
        return 1.0 * self.temperature  # k_B set to 1 for now


class SimulationParameters:
    """Parameters for the Langevin equation."""

    def __init__(
        self,
        *,
        physical_parameters: PhysicalParameters,
        time_span: TimeSpan,
        initial_conditions: InitialConditions,
        potential: sp.Expr,  # TODO: what are characteristic values
        characteristic_values: CharacteristicValues,
    ) -> None:
        self.physical_parameters = physical_parameters
        self.time_span = time_span
        self.initial_conditions = initial_conditions
        self.potential = potential
        self.characteristic_values = characteristic_values

    @cached_property
    def n_trajectories(self) -> int:
        """Return the number of simulated trajectories."""
        return self.initial_conditions.x0.shape[0]

    @cached_property
    def n_dimensions(self) -> int:
        """Return the dimensions of the system."""
        return self.initial_conditions.x0.shape[1]

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
    def burn_in_time(self) -> float:
        """Return the number of saved steps."""
        return self.time_span.burn_in * self.characteristic_values.time

    @cached_property
    def n_save(self) -> int:
        """Return the number of saved steps."""
        return int((self.time_span.t1 - self.burn_in_time) / self.time_span.dt)

    @cached_property
    def stride(self) -> int:
        """Return the sampling stride."""
        return max(1, int(self.characteristic_values.time / self.time_span.dt))


@dataclass(frozen=True, kw_only=True)
class SimulationResult[P: SimulationParameters = SimulationParameters]:
    """Results of a simulation of the periodic Langevin equation."""

    times: np.ndarray
    xs: np.ndarray[Any, np.dtype[np.floating]]
    ps: np.ndarray[Any, np.dtype[np.floating]]
    params: P


# TODO: discuss __hash__ as a nice way to do this
def _my_path(params: SimulationParameters, _key: jax.Array) -> Path:
    pp = params.physical_parameters
    ts = params.time_span

    parts = [
        f"g{pp.gamma:.4g}",
        f"T{pp.temperature:.4g}",
        f"m{pp.m:.4g}",
        f"t0-{ts.t0:.4g}",
        f"t1-{ts.t1:.4g}",
        f"dt{ts.dt:.4g}",
        f"n_trajectory{params.n_trajectories}",
        f"n_dims{params.n_dimensions}",
    ]

    # TODO: discuss hash
    if isinstance(params, SHOParameters):
        parts.append(f"omega{params.omega:.4g}")
    elif isinstance(params, PeriodicParameters):
        parts.extend(
            (f"lattice{params.lattice_spacing:.4g}", f"amp{params.amplitude:.4g}")
        )

    filename = "_".join(parts) + ".npz"
    return Path("examples/data") / filename


@cached(_my_path, default_call="load_or_call_uncached")
def solve_ensemble[P: SimulationParameters](
    params: P, _key: jax.Array
) -> SimulationResult[P]:
    """Solve the ULD Langevin equation for an ensemble of trajectories via vmap."""
    n_dims = params.n_dimensions
    gamma = jnp.broadcast_to(params.physical_parameters.gamma, (n_dims,))
    u = jnp.broadcast_to(
        params.physical_parameters.kbt / params.physical_parameters.m, (n_dims,)
    )

    def grad_f(x: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return -params.force_fn(x) / params.physical_parameters.kbt

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

    return SimulationResult(
        # TODO: discuss duplicating data
        times=np.array(ts_batch[0]),
        xs=np.array(xs_batch).transpose(0, 2, 1),  # -> (n_trajectories, n_dims, n_save)
        ps=np.array(ps_batch).transpose(0, 2, 1),
        params=params,
    )


class SHOParameters(SimulationParameters):
    """Parameters for the harmonic Langevin equation."""

    def __init__(
        self,
        *,
        omega: float,
        physical_parameters: PhysicalParameters,
        time_span: TimeSpan,
        initial_conditions: InitialConditions,
    ) -> None:
        potential = 0.5 * omega**2 * sp.symbols("x0") ** 2

        characteristic_length = np.sqrt(
            1.0 * physical_parameters.temperature / (physical_parameters.m * omega**2)
        )  # k_B set to 1 for now

        characteristic_time = 1 / omega

        super().__init__(
            physical_parameters=physical_parameters,
            time_span=time_span,
            initial_conditions=initial_conditions,
            potential=potential,
            characteristic_values=CharacteristicValues(
                length=characteristic_length,
                time=characteristic_time,
            ),
        )
        self.omega = omega
        self.delta_k: tuple[float, ...] = (1 / self.characteristic_values.length,)


class PeriodicParameters(SimulationParameters):
    """Parameters for the harmonic Langevin equation."""

    def __init__(
        self,
        *,
        lattice_spacing: float,
        amplitude: float,
        physical_parameters: PhysicalParameters,
        time_span: TimeSpan,
        initial_conditions: InitialConditions,
    ) -> None:
        potential = amplitude * sp.cos(2 * np.pi * sp.symbols("x0") / lattice_spacing)

        characteristic_length = lattice_spacing
        if physical_parameters.gamma == 0:
            characteristic_time = lattice_spacing * np.sqrt(
                physical_parameters.m / physical_parameters.kbt
            )
        else:
            characteristic_time = 1 / physical_parameters.gamma

        super().__init__(
            physical_parameters=physical_parameters,
            time_span=time_span,
            initial_conditions=initial_conditions,
            potential=potential,
            characteristic_values=CharacteristicValues(
                length=characteristic_length,
                time=characteristic_time,
            ),
        )
        self.lattice_spacing = lattice_spacing
        self.amplitude = amplitude
        self.delta_k: tuple[float, ...] = (
            0.1 * 2 * np.pi / self.characteristic_values.length,
        )


class FlatParameters(SimulationParameters):
    """Parameters for the harmonic Langevin equation."""

    def __init__(
        self,
        *,
        physical_parameters: PhysicalParameters,
        time_span: TimeSpan,
        initial_conditions: InitialConditions,
    ) -> None:
        potential = 0 * sp.symbols("x0")
        # TODO: why the maths here?
        if physical_parameters.gamma == 0:
            characteristic_time = 1.0
            characteristic_length = 1.0
        else:
            characteristic_time = 1 / physical_parameters.gamma
            characteristic_length = (
                np.sqrt(physical_parameters.kbt / physical_parameters.m)
                / physical_parameters.gamma
            )

        super().__init__(
            physical_parameters=physical_parameters,
            time_span=time_span,
            initial_conditions=initial_conditions,
            potential=potential,
            characteristic_values=CharacteristicValues(
                length=characteristic_length,
                time=characteristic_time,
            ),
        )
        self.direction = np.ones(self.n_dimensions)
        unit_direction = self.direction / np.linalg.norm(self.direction)
        self.delta_k: tuple[float, ...] = tuple(
            unit_direction / self.characteristic_values.length
        )


# TODO: can you have a generic "periodic" class?
# TODO: discuss what goes in __init__


class PeriodicParameters2D(SimulationParameters):
    """Parameters for the 2d periodic."""

    def __init__(
        self,
        *,
        delta_x: float,
        height: float,
        physical_parameters: PhysicalParameters,
        time_span: TimeSpan,
        initial_conditions: InitialConditions,
    ) -> None:
        characteristic_length = delta_x
        if physical_parameters.gamma == 0:
            characteristic_time = delta_x * np.sqrt(
                physical_parameters.m / physical_parameters.kbt
            )
        else:
            characteristic_time = 1 / physical_parameters.gamma

        c = 2.0 * np.pi / (np.sqrt(3.0) * delta_x)

        kx0 = c * (-1.0 / np.sqrt(3.0))
        kx1 = c * 1.0

        ky0 = c * (2.0 / np.sqrt(3.0))
        ky1 = 0.0

        arg1 = sp.symbols("x0") * kx0 + sp.symbols("x1") * kx1
        arg2 = sp.symbols("x0") * ky0 + sp.symbols("x1") * ky1
        arg3 = arg1 + arg2

        cos_sum = sp.cos(arg1) + sp.cos(arg2) + sp.cos(arg3)

        super().__init__(
            physical_parameters=physical_parameters,
            time_span=time_span,
            initial_conditions=initial_conditions,
            potential=2.0 * height * cos_sum,
            characteristic_values=CharacteristicValues(
                length=characteristic_length,
                time=characteristic_time,
            ),
        )

        @cached_property
        def kx(delta_x: float) -> tuple[float, float]:
            """Calculate the reciprocal lattice vectors."""
            c = 2.0 * np.pi / (np.sqrt(3.0) * delta_x)

            return (c * (-1.0 / np.sqrt(3.0)), c * 1.0)

        @cached_property
        def ky(delta_x: float) -> tuple[float, float]:
            """Calculate the reciprocal lattice vectors."""
            c = 2.0 * np.pi / (np.sqrt(3.0) * delta_x)

            return (c * (2.0 / np.sqrt(3.0)), 0.0)
