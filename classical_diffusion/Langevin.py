from dataclasses import astuple, dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


# Plotting functions


def _plot_x_evolution(
    result: Any, *, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = plt.subplots(figsize=(10, 6))

    times = result.times - result.times[0]
    (line,) = ax.plot(times, result.xs)
    line.set_label("$x$")

    ax.set_xlabel("Time")
    ax.set_ylabel("Position ($x$)")

    return fig, ax, line


def _plot_p_evolution(
    result: Any,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = plt.subplots(figsize=(10, 6))

    times = result.times - result.times[0]
    (line,) = ax.plot(times, result.ps)
    line.set_label("$p$")

    ax.set_xlabel("Time")
    ax.set_ylabel("Momentum ($p$)")

    return fig, ax, line


@dataclass(frozen=True, kw_only=True)
class TimeSpan:
    """Time-stepping parameters, bundled together since n_sav derives from all three."""

    t0: float
    t1: float
    dt0: float
    n_sav: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_sav", int((self.t1 - self.t0) / self.dt0))


@dataclass(kw_only=True)
class SimulationParams:
    """Parameters for the Langevin equation."""

    gamma: float
    sigma: float
    m: float
    potential: sp.Expr
    time: TimeSpan
    y0: np.ndarray

    @cached_property
    def symbolic_gradient(self) -> sp.Expr:
        """Compute the symbolic gradient of the potential."""
        return sp.diff(self.potential, sp.symbols("x"))

    @cached_property
    def gradient_fn(self) -> Callable[[float], float]:
        """Compute the symbolic gradient of the potential."""
        return sp.lambdify(sp.symbols("x"), self.symbolic_gradient, "numpy")

    def force(self, x: float) -> float:
        """Compute the force at position x."""
        return -self.gradient_fn(x)


def make_solver(params: SimulationParams) -> tuple:
    """Generate drift and diffusion functions tailored to a specific potential."""

    def drift(t: float, y: jnp.ndarray, args: tuple) -> jnp.ndarray:
        """Drift function for the Langevin equation."""
        x, p = y
        m, gamma, _sigma = args
        force = -params.gradient_fn(x)
        return jnp.array([p / m, force - gamma * p])

    def diffusion(t: float, y: jnp.ndarray, args: tuple) -> jnp.ndarray:
        """Diffusion function for the harmonic Langevin equation."""
        _x, _p = y
        (
            _m,
            _gamma,
            sigma,
        ) = args
        return jnp.array([0.0, sigma])

    return drift, diffusion


def solve_langevin(params: SimulationParams, key: jax.Array) -> dfx.Solution:
    """Solve the Langevin equation using diffrax."""
    t0, t1, dt0, n_sav = astuple(params.time)
    y0 = jnp.array(params.y0)
    args = (params.m, params.gamma, params.sigma)
    # Brownian noise
    bm = dfx.VirtualBrownianTree(t0, t1, tol=1e-4, shape=(), key=key)

    drift, diffusion = make_solver(params)

    terms = dfx.MultiTerm(
        dfx.ODETerm(drift),
        dfx.ControlTerm(diffusion, bm),
    )
    solver = dfx.EulerHeun()
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, n_sav))

    return dfx.diffeqsolve(
        terms, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, args=args, saveat=saveat
    )
