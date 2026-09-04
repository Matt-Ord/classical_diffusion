from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
import sympy as sp

from classical_diffusion.langevin import (
    LangevinSimulationResult,
)
from classical_diffusion.langevin._analysis import _get_energy
from classical_diffusion.langevin._langevin import get_random_initial_conditions_ext
from classical_diffusion.plot import get_figure
from classical_diffusion.util import _get_key, timed

if TYPE_CHECKING:
    import jax
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from classical_diffusion.langevin._system import (
        HarmonicSystem,
        PeriodicSystem1D,
        PeriodicSystemFCC,
        System,
    )


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

    mesh = ax.pcolormesh(
        x_grid, y_grid, potential, cmap=mpl.rcParams["image.cmap"], shading="auto"
    )

    color_bar = fig.colorbar(mesh, ax=ax)
    color_bar.set_label(r"$V(x, y)$")

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    ax.set_xlim(start[0], end[0])
    ax.set_ylim(start[1], end[1])
    ax.set_aspect("equal", adjustable="box")

    return fig, ax, mesh


def _plot_unit_cell(
    ax: Axes,
    system: PeriodicSystemFCC,
) -> Line2D:
    a1, a2 = system.lattice_vectors

    corner_points = [(0, 0), a1, a1 + a2, a2, (0, 0)]

    (line,) = ax.plot(*np.array(corner_points).T)
    line.set_marker("o")

    return line


def plot_periodic_potential_fcc(
    system: PeriodicSystemFCC,
    *,
    n_points: tuple[int, int] = (1000, 1000),
    ax: Axes | None = None,
    shape: tuple[int, int] = (3, 3),
) -> tuple[Figure, Axes, QuadMesh, Line2D]:
    """Plot the periodic potential in 2D."""
    fig, ax, mesh = plot_potential_2d(
        system,
        (-shape[0] / 2 * system.delta_x, -shape[1] / 2 * system.delta_x),
        (
            shape[0] / 2 * system.delta_x,
            shape[1] / 2 * system.delta_x,
        ),
        n_points=n_points,
        ax=ax,
    )

    unit_cell = _plot_unit_cell(ax=ax, system=system)
    unit_cell.set_color("C2")
    return fig, ax, mesh, unit_cell


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
    ax.set_xlabel("Time")
    ax.set_ylabel("ISF")
    ax.legend()

    return fig, ax, line


def get_exact_flat_isf(
    system: System,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> np.ndarray:
    """Return the exact ISF for a flat (potential-free) surface."""
    kbt, m, gamma = system.kbt, system.m, system.gamma
    k_squared = np.sum(np.array(delta_k) ** 2)
    return np.exp(
        ((k_squared) * kbt / (gamma**2 * m))
        * (1 - gamma * times - np.exp(-gamma * times))
    )


def plot_exact_flat_isf(
    system: System,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the exact ISF for a flat (potential-free) surface."""
    fig, ax = get_figure(ax)

    times = times if times is not None else np.linspace(0, 10, 1000)
    isf_exact = get_exact_flat_isf(system, delta_k=delta_k, times=times)

    (line,) = ax.plot(times, isf_exact)
    line.set_label("Exact Flat ISF")

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("ISF")
    ax.legend()

    return fig, ax, line


def get_exact_flat_ballistic_isf(
    system: System,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> np.ndarray:
    """Return the exact ballistic ISF for a flat (potential-free) surface."""
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
    """Plot the exact ISF for a flat (potential-free) surface."""
    fig, ax = get_figure(ax)

    times = times if times is not None else np.linspace(0, 30, 1000)
    isf_exact = offset + (1 - offset) * get_exact_flat_ballistic_isf(
        system=system, delta_k=delta_k, times=times
    )

    (line,) = ax.plot(times, isf_exact)

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("ISF")
    ax.legend()

    return fig, ax, line


def get_characteristic_friction_time(system: System) -> float:
    """Return characteristic time for a flat system."""
    if system.gamma == 0:
        return 1.0
    return 1 / system.gamma


N_SAMPLES = 1_000_000


@timed
def get_under_barrier_probability(
    system: System, barrier_energy: float, *, _key: jax.Array | None = None
) -> float:
    _key = _get_key(_key)

    x_points, p_points = get_random_initial_conditions_ext(
        system, n_samples=N_SAMPLES, _key=_key
    )
    energies = _get_energy(system, x_points, p_points)
    return np.mean(energies < barrier_energy)


def shift_origin_to_unit_cell_1d[S: PeriodicSystem1D](
    result: LangevinSimulationResult[S], *, origin_idx: int | None = None
) -> LangevinSimulationResult[S]:
    origin_idx = (
        origin_idx if origin_idx is not None else np.argmin(np.abs(result.times)).item()
    )
    x0 = result.x_points[:, :, origin_idx]

    n_shift = np.floor(x0 / result.system.delta_x + 0.5)
    x_folded = result.x_points - n_shift[:, :, None] * result.system.delta_x
    return LangevinSimulationResult(
        times=result.times,
        x_points=x_folded,
        p_points=result.p_points,
        system=result.system,
    )


def shift_origin_to_unit_cell_fcc[S: PeriodicSystemFCC](
    result: LangevinSimulationResult[S], *, origin_idx: int | None = None
) -> LangevinSimulationResult[S]:
    lattice_vectors = result.system.lattice_vectors

    origin_idx = (
        origin_idx if origin_idx is not None else np.argmin(np.abs(result.times)).item()
    )
    x0 = result.x_points[:, :, origin_idx]
    n_shift = np.round(x0 @ np.linalg.inv(lattice_vectors))
    cartesian_shifts = np.round(n_shift) @ lattice_vectors

    # 6. Broadcast subtract the shift across all time points (shape: n_samples, 2, n_time_points)
    x_folded = result.x_points - cartesian_shifts[:, :, np.newaxis]

    return LangevinSimulationResult(
        times=result.times,
        x_points=x_folded,
        p_points=result.p_points,
        system=result.system,
    )
