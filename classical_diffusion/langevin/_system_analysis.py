from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from classical_diffusion.langevin._langevin import _get_langevin_units
from classical_diffusion.plot import CAM_BLUE_CMAP, get_figure
from classical_diffusion.util import _get_key

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Annotation

    from classical_diffusion.langevin._system import (
        HarmonicSystem,
        PeriodicSystem1D,
        PeriodicSystemFCC,
        System,
    )

    from ._system import CanonicalSystem


def plot_potential_1d(
    system: System,
    start: float,
    end: float,
    *,
    n_points: int = 1000,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the potential energy surface for a 1D or 2D system.

    For 1D systems, plots V(x) as a line. For 2D systems, plots V(x, y)
    as a filled heatmap.

    """
    fig, ax = get_figure(ax)

    delta = np.array(start) - np.array(end)

    t = np.linspace(0, 1, n_points)
    points = np.array(start) + t[:, np.newaxis] * delta

    potential_func = sp.lambdify(
        system.lambda_symbols,
        system.potential_expr,
        modules=[{"DerivativeSafeMod": np.mod}, "numpy"],
    )
    potential = np.broadcast_to(potential_func(*points.T, *system.params), (n_points,))

    distances = np.linalg.norm(start) + t * np.linalg.norm(delta)

    (line,) = ax.plot(distances, potential)

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"$V(x)$")
    ax.set_xlim(distances[0], distances[-1])

    return fig, ax, line


def plot_force_1d(
    params: System,
    start: float,
    end: float,
    *,
    n_points: int = 1000,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the force for a 1D or 2D system.

    For 1D systems, plots F(x) as a line. For 2D systems, plots F(x, y)
    as a filled heatmap.

    """
    fig, ax = get_figure(ax)

    delta = np.array(start) - np.array(end)

    t = np.linspace(0, 1, n_points)
    points = np.array(start) + t[:, np.newaxis] * delta

    force_func = sp.lambdify(
        params.lambda_symbols,
        params.force_expr[0],
        modules=[{"DerivativeSafeMod": np.mod}, "numpy"],
    )
    force = np.broadcast_to(force_func(*points.T, *params.params), (n_points,))

    distances = np.linalg.norm(start) + t * np.linalg.norm(delta)

    (line,) = ax.plot(distances, force)

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"$F(x)$")
    ax.set_xlim(distances[0], distances[-1])

    return fig, ax, line


def plot_periodic_potential_1d(
    system: PeriodicSystem1D, *, n_points: int = 1000, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    """Plot the periodic potential in 1D."""
    return plot_potential_1d(
        system, 0, 3 * system.delta_x * 2, n_points=n_points, ax=ax
    )


def _add_unit_cell(
    ax: Axes,
    system: PeriodicSystemFCC,
) -> tuple[Line2D, Annotation]:
    a1, a2 = np.asarray(system.lattice_vectors)

    corner_points = [(0, 0), a1, a1 + a2, a2, (0, 0)]

    (boundary,) = ax.plot(*np.array(corner_points).T)
    boundary.set_marker("o")

    annotation = ax.annotate(
        f"{np.linalg.norm(a1):.3g} m".strip(),
        xy=np.array(a1) / 2,
        ha="left",
        va="center",
    )

    return boundary, annotation


def plot_potential_2d(
    system: System,
    start: tuple[float, ...],
    end: tuple[float, ...],
    *,
    n_points: tuple[int, int] = (100, 100),
    ax: Axes | None = None,
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot the potential energy surface for a 2D system as a filled heatmap.

    Parameters
    ----------
    system : System
        The system for which to plot the potential.
    start : tuple[float, ...]
        The lower-bound coordinates (x_min, y_min).
    end : tuple[float, ...]
        The upper-bound coordinates (x_max, y_max).
    n_points : tuple[int, int], optional
        The number of grid points in the x and y directions, by default (100, 100).
    ax : Axes | None, optional
        The matplotlib Axes to plot on, by default None.

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
        The figure, axes, and the generated QuadMesh.
    """
    fig, ax = get_figure(ax)

    x = np.linspace(start[0], end[0], n_points[0])
    y = np.linspace(start[1], end[1], n_points[1])
    x_grid, y_grid = np.meshgrid(x, y)

    potential_func = sp.lambdify(
        system.lambda_symbols,
        system.potential_expr,
        modules=[{"DerivativeSafeMod": np.mod}, "numpy"],
    )
    potential = np.broadcast_to(
        potential_func(x_grid, y_grid, *system.params), x_grid.shape
    )

    mesh = ax.pcolormesh(x_grid, y_grid, potential, cmap=CAM_BLUE_CMAP)

    color_bar = fig.colorbar(mesh, ax=ax)
    color_bar.set_label(r"$V(x, y)$")

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    ax.set_xlim(start[0], end[0])
    ax.set_ylim(start[1], end[1])
    ax.set_aspect("equal", adjustable="box")

    return fig, ax, mesh


def plot_periodic_potential_fcc(
    system: PeriodicSystemFCC,
    *,
    n_points: tuple[int, int] = (100, 100),
    ax: Axes | None = None,
    width: float = 4,
    height: float = 4,
) -> tuple[Figure, Axes, QuadMesh, tuple]:
    """Plot the periodic potential in 2D."""
    fig, ax, mesh = plot_potential_2d(
        system,
        (-width / 2 * system.delta_x, -height / 2 * system.delta_x),
        (
            width / 2 * system.delta_x,
            height / 2 * system.delta_x,
        ),
        n_points=n_points,
        ax=ax,
    )

    unit_cell = _add_unit_cell(ax=ax, system=system)
    unit_cell[0].set_color("C2")
    return fig, ax, mesh, unit_cell


def add_bridge_site_energy(
    ax: Axes,
    system: PeriodicSystemFCC,
    origin_site: tuple[int, int] = (0, 0),
    *,
    label_offset: tuple[float, float] = (18.0, 12.0),
) -> tuple:
    a1, a2 = np.asarray(system.lattice_vectors)
    origin = origin_site[0] * a1 + origin_site[1] * a2
    bridge_point = origin + a2 + a1 / 2  # midpoint of the top edge

    potential_func = sp.lambdify(
        system.lambda_symbols,
        system.potential_expr,
        modules=[{"DerivativeSafeMod": np.mod}, "numpy"],
    )
    bridge_energy = float(
        potential_func(bridge_point[0], bridge_point[1], *system.params)
    )

    site = ax.plot(
        *bridge_point,
        marker="x",
        markersize=9,
        markeredgewidth=2,
        zorder=10,
    )
    annotation = ax.annotate(
        f"{bridge_energy:.3g} J".strip(),
        xy=bridge_point,
        xytext=label_offset,
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        zorder=10,
    )

    return ax, site, annotation


def add_top_site_energy(
    ax: Axes,
    system: PeriodicSystemFCC,
    origin_site: tuple[int, int] = (0, 0),
    *,
    label_offset: tuple[float, float] = (12.0, -12.0),
) -> tuple:
    a1, a2 = np.asarray(system.lattice_vectors)
    top_point = origin_site[0] * a1 + origin_site[1] * a2

    potential_func = sp.lambdify(
        system.lambda_symbols,
        system.potential_expr,
        modules=[{"DerivativeSafeMod": np.mod}, "numpy"],
    )
    top_energy = float(potential_func(top_point[0], top_point[1], *system.params))

    site = ax.plot(
        *top_point,
        marker="x",
        markersize=9,
        markeredgewidth=2,
        zorder=10,
    )
    annotation = ax.annotate(
        f"{top_energy:.3g} J".strip(),
        xy=top_point,
        xytext=label_offset,
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=9,
        zorder=10,
    )

    return ax, site, annotation


def add_hollow_site_distance(
    ax: Axes,
    system: PeriodicSystemFCC,
    origin_site: tuple[int, int] = (0, 0),
    *,
    label_offset: tuple[float, float] = (12.0, 0.0),
) -> tuple:

    a1, a2 = np.asarray(system.lattice_vectors)
    origin = origin_site[0] * a1 + origin_site[1] * a2

    hollow_a = origin + (a1 + a2) / 3
    hollow_b = origin + 2 * (a1 + a2) / 3
    distance = float(np.linalg.norm(hollow_b - hollow_a))

    site_a = ax.plot(*hollow_a, marker="o", markersize=8, zorder=10)
    site_b = ax.plot(*hollow_b, marker="o", markersize=8, zorder=10)
    line = ax.plot(
        [hollow_a[0], hollow_b[0]],
        [hollow_a[1], hollow_b[1]],
        linestyle="--",
        linewidth=1.2,
        zorder=10,
    )

    midpoint = (hollow_a + hollow_b) / 2
    annotation = ax.annotate(
        f"{distance:.3g} m".strip(),
        xy=midpoint,
        xytext=label_offset,
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        zorder=10,
    )

    return distance, (site_a, site_b), line, annotation


def get_exact_harmonic_isf(
    system: HarmonicSystem,
    delta_k: tuple[float,],
    times: np.ndarray[tuple[int], np.dtype[np.floating[Any]]],
) -> np.ndarray[tuple[int], np.dtype[np.floating[Any]]]:
    """Return the exact ISF for simulation."""
    gamma, _temp, m = system.gamma, system.temperature, system.m
    f = np.sqrt(system.omega**2 - gamma**2 / 4)

    return np.exp(
        -(delta_k[0] ** 2)
        * (system.kbt / (m * system.omega**2))
        * (
            1
            - np.exp(-gamma * times / 2)
            * (np.cos(f * times) + (gamma / (2 * f)) * np.sin(f * times))
        )
    )


def plot_exact_harmonic_isf(
    system: HarmonicSystem,
    delta_k: tuple[float,],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the state occupations of a quantum simulation result."""
    fig, ax = get_figure(ax)

    times = times if times is not None else np.linspace(0, 30, 1000)

    isf_exact = get_exact_harmonic_isf(system, delta_k, times)
    (line,) = ax.plot(times, isf_exact)
    line.set_label("ISF")

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")
    ax.legend()

    return fig, ax, line


def get_exact_flat_isf(
    system: System,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> np.ndarray:
    """Return the exact ISF for a 1D flat (potential-free) surface."""
    kbt, m, gamma = system.kbt, system.m, system.gamma
    k_squared = np.sum(np.array(delta_k) ** 2)
    return np.exp(
        ((k_squared**2) * kbt / (gamma**2 * m))
        * (1 - gamma * times - np.exp(-gamma * times))
    )


def plot_exact_flat_isf(
    system: System,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the exact ISF for a 1D flat (potential-free) surface."""
    fig, ax = get_figure(ax)

    times = times if times is not None else np.linspace(0, 10, 1000)
    isf_exact = get_exact_flat_isf(system, delta_k=delta_k, times=times)

    (line,) = ax.plot(times, isf_exact)
    line.set_label("Exact Flat ISF")

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")
    ax.legend()

    return fig, ax, line


def get_exact_flat_ballistic_isf(
    system: System,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> np.ndarray:
    """Return the exact ballistic ISF for a 1D flat (potential-free) surface."""
    kbt, m = system.kbt, system.m
    m = np.atleast_2d(m)
    inv_m = np.linalg.inv(m)
    inner_product = np.einsum("i,ij,j->", delta_k, inv_m, delta_k)
    return np.exp(-(inner_product * kbt / 2) * times**2)


def plot_exact_flat_ballistic_isf(
    system: System,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    *,
    ax: Axes | None = None,
    offset: float = 0,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the exact ISF for a 1D flat (potential-free) surface."""
    fig, ax = get_figure(ax)

    times = times if times is not None else np.linspace(0, 30, 1000)
    isf_exact = offset + (1 - offset) * get_exact_flat_ballistic_isf(
        system=system, delta_k=delta_k, times=times
    )

    (line,) = ax.plot(times, isf_exact)

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")
    ax.legend()

    return fig, ax, line


def get_characteristic_friction_time(system: System) -> float:
    """Return characteristic time for a flat system."""
    if system.gamma == 0:
        return 1.0
    return 1 / system.gamma


SAMPLE_REGION = 10


@jax.jit(static_argnames=("n_samples"))
def _get_under_barrier_probability_jax(
    system: "CanonicalSystem",  # ruff: ignore[quoted-annotation]
    barrier_energy: float,
    n_samples: int,
    key: jax.Array,
) -> jax.Array:
    # To find the under barrier probability, the V(x) is sampled at x
    # points centered around the origin, and the momentum is sampled from the Maxwell-Boltzmann distribution.
    # The fraction of phase space under the barrier is then computed using importance sampling:
    # P(under barrier) = sum(w(x) * I(V(x) + K(p) < barrier)) / sum(w(x))
    # where w(x) = exp(-V(x)/kBT) / q(x) is the importance weight, and q(x) is the sampling distribution.
    key_x, key_p = jax.random.split(key)

    # Sample positions according to q(x)
    x_samples = (
        jax.random.normal(key_x, shape=(n_samples, system.n_dim)) * SAMPLE_REGION
    )
    p_standard_deviation = jnp.sqrt(system.m * system.kbt)
    p_samples = (
        jax.random.normal(key_p, shape=(n_samples, system.n_dim)) * p_standard_deviation
    )

    potential_fn = sp.lambdify(system.lambda_symbols, system.potential_expr, "jax")
    v_energies = jax.vmap(lambda x: potential_fn(*x, *system.params))(x_samples)
    kinetic_energies = jnp.sum(p_samples**2, axis=-1) / (2.0 * system.m)
    total_energies = v_energies + kinetic_energies

    # Importance weights for x: w(x) = exp(-V(x)/kBT) / q(x)
    log_weights = -v_energies / system.kbt + jnp.sum(x_samples**2, axis=-1) / (
        2.0 * SAMPLE_REGION**2
    )
    weights = jnp.exp(log_weights - jnp.max(log_weights))
    return jnp.average(total_energies < barrier_energy, weights=weights)


N_SAMPLES = 1_000_000


def get_under_barrier_probability(
    system: System, barrier_energy: float, *, _key: jax.Array | None = None
) -> float:
    _key = _get_key(_key)

    canonical_system = system.with_units(_get_langevin_units(system)).as_canonical()
    barrier_energy = system.units.energy_into(barrier_energy, canonical_system.units)
    return float(
        _get_under_barrier_probability_jax(
            key=_key,
            system=canonical_system,
            barrier_energy=barrier_energy,
            n_samples=N_SAMPLES,
        )
    )
