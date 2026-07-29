from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from scipy import integrate
from scipy.optimize import brentq
from scipy.special import ellipkinc

from classical_diffusion.plot import get_figure
from classical_diffusion.system._system import get_energy

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from classical_diffusion.system import (
        HarmonicSystem,
        PeriodicSystem1D,
        PeriodicSystemFCC,
        System,
    )


def plot_potential_1d(
    params: System,
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

    potential_func = sp.lambdify(params.lambda_symbols, params.potential_expr, "numpy")
    potential = np.broadcast_to(potential_func(*points.T, *params.params), (n_points,))

    distances = np.linalg.norm(start) + t * np.linalg.norm(delta)

    (line,) = ax.plot(distances, potential)

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"$V(x)$")
    ax.set_xlim(distances[0], distances[-1])

    return fig, ax, line


def plot_periodic_potential_1d(
    system: PeriodicSystem1D, *, n_points: int = 1000, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    """Plot the periodic potential in 1D."""
    return plot_potential_1d(
        system, -system.delta_x * 2, system.delta_x * 2, n_points=n_points, ax=ax
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

    potential_func = sp.lambdify(params.lambda_symbols, params.potential_expr, "numpy")
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
        (0, 0),
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
    times: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> np.ndarray:
    """Return the exact ISF for simulation."""
    gamma, temp, m = system.gamma, system.temperature, system.m
    f = np.sqrt(system.omega**2 - gamma**2 / 4)

    return np.exp(
        -(delta_k[0] ** 2)
        * ((1.0 * temp) / (m * system.omega**2))
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
    """Return the exact ballistic ISF for a 1D flat (potential-free) surface."""
    kbt, m, gamma = system.kbt, system.m, system.gamma
    k_squared = sum(k_i**2 for k_i in delta_k)
    return np.exp(
        ((k_squared) * kbt / (gamma**2 * m))
        * (1 - gamma * times - np.exp(-gamma * times))
    )


def get_exact_gaussian_isf(
    system: System,
    effective_mass: float,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> np.ndarray:
    """Return the exact ballistic ISF for a 1D flat (potential-free) surface."""
    kbt, _, _ = system.kbt, system.m, system.gamma
    k_squared = sum(k_i**2 for k_i in delta_k)
    return np.exp(-((k_squared) * kbt / (2 * effective_mass)) * times**2)


def plot_exact_gaussian_isf(
    system: System,
    effective_mass: float,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the exact ISF for a 1D flat (potential-free) surface."""
    fig, ax = get_figure(ax)

    times = times if times is not None else np.linspace(0, 1e-11, 1000)
    isf_exact = get_exact_gaussian_isf(
        system=system, effective_mass=effective_mass, delta_k=delta_k, times=times
    )

    (line,) = ax.plot(times, isf_exact)

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")
    ax.legend()

    return fig, ax, line


def plot_exact_offset_gaussian_isf(  # ruff:ignore[too-many-arguments]
    system: System,
    effective_mass: float,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    *,
    ax: Axes | None = None,
    offset: float = 0,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the exact ISF for a 1D flat (potential-free) surface."""
    fig, ax = get_figure(ax)

    times = times if times is not None else np.linspace(0, 1e-11, 1000)
    isf_exact = offset + (1 - offset) * get_exact_gaussian_isf(
        system=system, effective_mass=effective_mass, delta_k=delta_k, times=times
    )

    (line,) = ax.plot(times, isf_exact)

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")
    ax.legend()

    return fig, ax, line


def plot_exact_flat_isf(
    system: System,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the exact ISF for a 1D flat (potential-free) surface."""
    fig, ax = get_figure(ax)

    times = times if times is not None else np.linspace(0, 30, 1000)
    isf_exact = get_exact_flat_isf(system, delta_k=delta_k, times=times)

    (line,) = ax.plot(times, isf_exact)
    line.set_label("Exact Flat ISF")

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


def set_up_integral(system: System) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    potential_func = sp.lambdify(system.lambda_symbols, system.potential_expr, "numpy")
    params = system.params

    def integrand(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.exp(
            -1 / system.kbt * (p**2 / (2 * system.m) + potential_func(x, *params))
        )

    return integrand


def calculate_partition_function(system: System) -> float:
    integrand = set_up_integral(system)
    z, _ = integrate.dblquad(
        integrand,
        -np.inf,
        np.inf,  # p limits (outer)
        lambda p: system.sampling_domain[0],  # x lower (inner) — fixed, one period
        lambda p: system.sampling_domain[1],  # x upper (inner)
    )
    return z


def get_x_domian_given_p(
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
            system.sampling_domain[1],
        )

    return x_t


def calculate_probability_under_barrier(system: System, barrier_energy: float) -> float:

    x_t = get_x_domian_given_p(system, barrier_energy)
    integrand = set_up_integral(system)

    integral_below, _ = integrate.dblquad(
        integrand,
        -np.inf,
        np.inf,
        lambda p: -x_t(p),
        x_t,
    )

    z = calculate_partition_function(system)

    return integral_below / z


def make_free_point_sampler(system: System, barrier_energy: float) -> Callable:
    x0, *_ = system.coordinate_symbols
    potential_fn = sp.lambdify(
        (x0, *system.parameter_symbols), system.potential_expr, "jax"
    )
    sigma = jnp.sqrt(system.m * system.kbt)
    x_lo, x_hi = system.sampling_domain

    def _sample_one(key: jax.Array) -> tuple:
        def _cond_fn(state: tuple) -> bool:
            _, _, _, accepted = state
            return ~accepted

        def _body_fn(state: tuple) -> tuple:
            key, _x, _p, _ = state
            key, kx, ku, kp = jax.random.split(key, 4)

            x_candidate = jax.random.uniform(kx, minval=x_lo, maxval=x_hi)
            V = potential_fn(x_candidate, *system.params)
            x_ok = jax.random.uniform(ku) < jnp.exp(-V / system.kbt)

            p_candidate = jax.random.normal(kp) * sigma
            energy = p_candidate**2 / (2 * system.m) + V

            accept = x_ok & (energy > barrier_energy)
            return key, x_candidate, p_candidate, accept

        init = (key, jnp.array(0.0), jnp.array(0.0), jnp.array(False))
        _, x_final, p_final, _ = jax.lax.while_loop(_cond_fn, _body_fn, init)
        return x_final, p_final

    return jax.jit(jax.vmap(_sample_one))


def get_elastic_p_exact_1d_periodic(
    system: PeriodicSystem1D, x_initial: np.ndarray, p_initial: np.ndarray
) -> np.ndarray:
    energy = get_energy(system, x_initial, p_initial)
    q2 = system.barrier_energy / energy
    phi0 = np.pi * x_initial[:, 0] / system.delta_x
    phi = np.pi * (x_initial[:, 0] / system.delta_x + 1)
    omega = (2 * np.pi / system.delta_x) * np.sqrt(energy / (2 * system.m))

    t = 1 / omega * (ellipkinc(phi, q2) - ellipkinc(phi0, q2))
    return system.m * system.delta_x / t


def get_full_effective_mass_exact_1d_periodic(
    system: PeriodicSystem1D, initial_conditions: tuple
) -> float:

    x_initial, p_initial = initial_conditions
    elastic_ps = get_elastic_p_exact_1d_periodic(
        system=system, x_initial=x_initial, p_initial=p_initial
    )
    avg_p2_given_escaped = np.average(elastic_ps**2, axis=0)

    prob_escape = 1 - calculate_probability_under_barrier(
        system=system, barrier_energy=system.barrier_energy
    )

    return (system.kbt * system.m) / (prob_escape * avg_p2_given_escaped)


def get_free_effective_mass_exact_1d_periodic(
    system: PeriodicSystem1D, initial_conditions: tuple
) -> float:

    x_initial, p_initial = initial_conditions
    elastic_ps = get_elastic_p_exact_1d_periodic(
        system=system, x_initial=x_initial, p_initial=p_initial
    )

    return (system.kbt * system.m) / np.average(elastic_ps**2, axis=0)
