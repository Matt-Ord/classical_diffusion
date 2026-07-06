from typing import TYPE_CHECKING, Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


# Plotting functions


def plot_x_evolution(
    result: Any, *, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = plt.subplots(figsize=(10, 6))

    times = result.times - result.times[0]
    (line,) = ax.plot(times, result.xs)
    line.set_label("$x$")

    ax.set_xlabel("Time")
    ax.set_ylabel("Position ($x$)")

    return fig, ax, line


def plot_p_evolution(
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


def drift(t, y, args) -> jnp.ndarray:
    """Drift function for the periodic Langevin equation."""
    x, p = y
    m, gamma, _sigma, pot_args, potential_func = args
    force = -jax.grad(potential_func, argnums=0)(x, pot_args)
    return jnp.array([p / m, force - gamma * p])


def diffusion(t, y, args):
    """Diffusion function for the harmonic Langevin equation."""
    _x, _p = y
    _m, _gamma, sigma, _pot_args, _potential_func = args
    return jnp.array([0.0, sigma])


def solve_langevin(t0, t1, dt0, y0, args, n, *, key):
    # Brownian noise
    bm = dfx.VirtualBrownianTree(t0, t1, tol=1e-4, shape=(), key=key)

    terms = dfx.MultiTerm(
        dfx.ODETerm(drift),
        dfx.ControlTerm(diffusion, bm),
    )
    solver = dfx.EulerHeun()
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, n))

    return dfx.diffeqsolve(
        terms, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, args=args, saveat=saveat
    )
