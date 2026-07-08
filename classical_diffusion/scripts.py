from dataclasses import astuple, dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# ???
from typing import (
    TYPE_CHECKING,
    Never,
    cast,
)

from matplotlib.animation import ArtistAnimation

try:
    from matplotlib import pyplot as plt
    from matplotlib.animation import ArtistAnimation
    from matplotlib.colors import LogNorm, SymLogNorm
    from matplotlib.colors import Normalize as BaseNorm
    from matplotlib.scale import LinearScale, LogScale, SymmetricalLogScale
except ImportError:
    plt = cast("Never", None)
    LogNorm, BaseNorm, SymLogNorm = cast("Never", (None, None, None))
    LinearScale, LogScale, SymmetricalLogScale = cast("Never", (None, None, None))
    SquaredScale = cast("Never", None)
    ArtistAnimation = cast("Never", None)


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.axes import Axes as MPLAxesBase
    from matplotlib.figure import Figure


def get_figure(ax: MPLAxesBase | None = None) -> tuple[Figure, Axes]:
    """Get the figure of the given axis.

    If no figure exists, a new figure is created
    """  # noqa: DOC501
    if plt is None:
        msg = "Matplotlib is not installed. Please install it with the 'plot' extra."
        raise ImportError(msg)  # noqa: RUF100

    if ax is None:
        return cast("tuple[Figure, Axes]", plt.subplots())  # type: ignore plt.subplots Unknown type

    fig = cast("Figure|None", ax.get_figure())
    if fig is None:
        fig = cast("Figure", plt.figure())  # type: ignore plt.figure Unknown type
        ax.set_figure(fig)
    return fig, ax


@dataclass(frozen=True, kw_only=True)
class TimeSpan:
    """Time-stepping parameters, bundled together since n_sav derives from all three."""

    t0: float
    t1: float
    dt0: float


@dataclass(kw_only=True)
class SimulationParams:
    """Parameters for the Langevin equation."""

    gamma: float
    sigma: float
    m: float
    potential: sp.Expr
    time: TimeSpan
    y0: np.ndarray
    delta_x: np.ndarray[Any, np.dtype[np.float64]]
    n_dims: int

    @cached_property
    def symbolic_coords(self) -> tuple:
        """Return the symoblic coodinates of the system."""
        return sp.symbols(f"x0:{self.n_dims}")

    @cached_property
    def symbolic_gradient(self) -> list:
        """Compute the symbolic gradient of the potential."""
        return [sp.diff(self.potential, c) for c in self.symbolic_coords]

    @cached_property
    def gradient_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Compute a callable gradient function, taking and returning an array."""
        raw_fn = sp.lambdify(self.symbolic_coords, self.symbolic_gradient, "jax")
        # converts inputed array into N seperate scalars to feed into raw_fn
        return lambda x_array: jnp.array(raw_fn(*x_array))


@dataclass(frozen=True, kw_only=True)
class SimulationResult:
    """Results of a simulation of the periodic Langevin equation."""

    times: np.ndarray
    xs: np.ndarray[Any, np.dtype[np.floating]]
    ps: np.ndarray[Any, np.dtype[np.floating]]

    params: SimulationParams


def make_solver(params: SimulationParams) -> tuple:
    """Generate drift and diffusion functions tailored to a specific potential."""
    n_dims = params.n_dims

    def drift(y: jnp.ndarray, args: tuple) -> jnp.ndarray:
        """Drift function for the Langevin equation."""
        x, p = y[:n_dims], y[n_dims:]
        m, gamma, _sigma = args
        force = -params.gradient_fn(x)
        return jnp.concatenate([p / m, force - gamma * p])

    def diffusion(args: tuple) -> jnp.ndarray:
        _m, _gamma, sigma = args
        # creates 2*n_dims by n_dims matrix matching diffrax requirements
        sigma_arr = jnp.broadcast_to(sigma, (n_dims,))
        top = jnp.zeros((n_dims, n_dims))
        bottom = jnp.diag(sigma_arr)
        return jnp.concatenate([top, bottom], axis=0)

    return drift, diffusion


def solve_langevin(params: SimulationParams, key: jax.Array) -> SimulationResult:
    """Solve the Langevin equation using diffrax."""
    t0, t1, dt0 = astuple(params.time)
    n_sav = int((t1 - t0) / dt0)
    y0 = jnp.array(params.y0)
    args = (params.m, params.gamma, params.sigma)
    # Brownian noise
    bm = dfx.VirtualBrownianTree(t0, t1, tol=1e-4, shape=(params.n_dims,), key=key)

    drift, diffusion = make_solver(params)

    terms = dfx.MultiTerm(
        dfx.ODETerm(drift),
        dfx.ControlTerm(diffusion, bm),
    )
    solver = dfx.EulerHeun()
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, n_sav))

    sol = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        max_steps=100_000,
    )

    return SimulationResult(
        times=np.array(sol.ts),
        xs=np.array(sol.ys[:, : params.n_dims]).T,
        ps=np.array(sol.ys[:, params.n_dims :]).T,
        params=params,
    )
