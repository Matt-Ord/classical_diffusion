from typing import TYPE_CHECKING, Any

import numpy as np
import sympy as sp
from scipy import integrate
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ellipk, ellipkinc

from classical_diffusion.langevin._sample import sample_energy_1d_periodic
from classical_diffusion.plot import get_figure

if TYPE_CHECKING:
    from collections.abc import Callable

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
    params: System,
    start: tuple[float, ...],
    end: tuple[float, ...],
    *,
    n_points: tuple[int, int] = (100, 100),
    ax: Axes | None = None,
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot the potential energy surface for a 2D system as a filled heatmap.

    Parameters
    ----------
    params : System
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
        params.lambda_symbols,
        params.potential_expr,
        modules=[{"DerivativeSafeMod": np.mod}, "numpy"],
    )
    potential = np.broadcast_to(
        potential_func(x_grid, y_grid, *params.params), x_grid.shape
    )

    mesh = ax.pcolormesh(x_grid, y_grid, potential)

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    ax.set_xlim(start[0], end[0])
    ax.set_ylim(start[1], end[1])

    return fig, ax, mesh


def plot_periodic_potential_fcc(
    params: PeriodicSystemFCC,
    *,
    n_points: tuple[int, int] = (100, 100),
    ax: Axes | None = None,
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot the periodic potential in 2D."""
    # TODO: fix up  PeriodicParameters2D to make lattice directions explicit # ruff:ignore[line-contains-todo]
    return plot_potential_2d(
        params,
        (-2 * params.delta_x, -2 * params.delta_x),
        (
            2 * params.delta_x,
            2 * params.delta_x,
        ),
        n_points=n_points,
        ax=ax,
    )


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
    k_squared = sum(k_i**2 for k_i in delta_k)
    return np.exp(-((k_squared) * system.kbt / (2 * system.m)) * times**2)


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


def _set_up_integral_1d(
    system: System,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    potential_func = sp.lambdify(system.lambda_symbols, system.potential_expr, "numpy")
    params = system.params

    def integrand(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.exp(
            -1 / system.kbt * (p**2 / (2 * system.m) + potential_func(x, *params))
        )

    return integrand


def _calculate_partition_function_1d(system: System) -> float:
    integrand = _set_up_integral_1d(system)
    z, _ = integrate.dblquad(
        integrand,
        -np.inf,
        np.inf,  # p limits (outer)
        lambda _p: system.sampling_domain[0][0],  # x lower (inner) — fixed, one period
        lambda _p: system.sampling_domain[0][1],  # x upper (inner)
    )
    return z


def _get_x_domian_given_p_1d(
    system: System, barrier_energy: float
) -> Callable[[float], float]:
    potential_func = sp.lambdify(system.lambda_symbols, system.potential_expr, "numpy")
    params = system.params

    def x_t(p: float) -> float:
        ke = p**2 / (2 * system.m)
        if ke >= barrier_energy:
            return 0.0
        target = barrier_energy - ke  # V(x_t) = this
        return brentq(
            lambda x: potential_func(x, *params) - target,
            0.0,
            system.sampling_domain[0][1],
        )

    return x_t


def calculate_probability_under_barrier_1d(
    system: System, barrier_energy: float
) -> float:

    x_t = _get_x_domian_given_p_1d(system, barrier_energy)
    integrand = _set_up_integral_1d(system)

    integral_below, _ = integrate.dblquad(
        integrand,
        -np.inf,
        np.inf,
        lambda p: -x_t(p),
        x_t,
    )

    z = _calculate_partition_function_1d(system)

    return integral_below / z


def _get_elastic_p_exact_1d_periodic(
    system: PeriodicSystem1D, n_samples: int
) -> np.ndarray:
    energy = sample_energy_1d_periodic(
        system=system, n_samples=n_samples, domain=(system.barrier_energy, np.inf)
    )
    epsilon = energy / system.barrier_energy

    return (
        np.pi
        * np.sqrt(2 * system.barrier_energy * epsilon * system.m)
        / ellipkinc(np.pi, 1 / epsilon)
    )


def get_full_effective_mass_exact_1d_periodic(
    system: PeriodicSystem1D, n_samples: int
) -> float:

    elastic_ps = _get_elastic_p_exact_1d_periodic(
        system=system,
        n_samples=n_samples,
    )
    avg_p2_given_escaped = np.average(elastic_ps**2, axis=0)

    prob_escape = 1 - calculate_probability_under_barrier_1d(
        system=system, barrier_energy=system.barrier_energy
    )

    return (system.kbt * system.m**2) / (prob_escape * avg_p2_given_escaped)


def get_free_effective_mass_exact_1d_periodic(
    system: PeriodicSystem1D, n_samples: int
) -> float:

    elastic_ps = _get_elastic_p_exact_1d_periodic(system=system, n_samples=n_samples)

    return (system.kbt * system.m**2) / np.average(elastic_ps**2, axis=0)


def get_full_effective_mass_exact_1d_periodic_directly(
    system: PeriodicSystem1D,
) -> float:
    u0 = system.barrier_energy / (2 * system.kbt)

    def integrand_denominator(epsilon: float) -> float:
        return np.sqrt(epsilon) / ellipk(1 / epsilon) * np.exp(-2 * u0 * epsilon)

    def integrand_trapped(epsilon: float) -> float:
        return ellipk(epsilon) * np.exp(-2 * u0 * epsilon)

    def integrand_running(epsilon: float) -> float:
        return (
            2 * (1 / np.sqrt(epsilon)) * ellipk(1 / epsilon) * np.exp(-2 * u0 * epsilon)
        )

    denominator_integral, _ = quad(integrand_denominator, 1, np.inf)
    trapped_integral, _ = quad(integrand_trapped, 0, 1)
    running_integral, _ = quad(integrand_running, 1, np.inf)

    partition = 2 * trapped_integral + running_integral

    return system.m * partition / (denominator_integral * 2 * u0 * np.pi**2)


def get_free_effective_mass_exact_1d_periodic_directly(
    system: PeriodicSystem1D,
) -> float:
    u0 = system.barrier_energy / (2 * system.kbt)

    def integrand_denominator(epsilon: float) -> float:
        return np.sqrt(epsilon) / ellipk(1 / epsilon) * np.exp(-2 * u0 * epsilon)

    def integrand_running(epsilon: float) -> float:
        return (
            2 * (1 / np.sqrt(epsilon)) * ellipk(1 / epsilon) * np.exp(-2 * u0 * epsilon)
        )

    denominator_integral, _ = quad(integrand_denominator, 1, np.inf)
    running_integral, _ = quad(integrand_running, 1, np.inf)

    partition = running_integral

    return system.m * partition / (denominator_integral * 2 * u0 * np.pi**2)
