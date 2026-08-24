from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from scipy import integrate
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ellipk, ellipkinc
from scipy.stats.sampling import NumericalInversePolynomial

from classical_diffusion.plot import get_figure

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from classical_diffusion.langevin import LangevinSimulationResult
    from classical_diffusion.system import (
        HarmonicSystem,
        PeriodicSquareSystem1D,
        PeriodicSquareSystem2D,
        PeriodicSystem1D,
        PeriodicSystemFCC,
        System,
    )


def get_energy(
    system: System,
    x_points: np.ndarray,
    p_points: np.ndarray,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Return the energy of the system."""
    potential = sp.lambdify(
        (*system.coordinate_symbols, *system.parameter_symbols),
        system.potential_expr,
        "numpy",
    )

    x_components = [x_points[:, d] for d in range(system.n_dim)]
    potential = potential(*x_components, *system.params)

    kinetic = np.sum(p_points**2, axis=1) / (2 * system.m)

    return kinetic + potential


def plot_energy(
    result: LangevinSimulationResult, n_trajectories: int = 1, *, ax: Axes
) -> tuple[Figure, Axes]:
    """Plot the energy of the system with time."""
    fig, ax = get_figure(ax)
    energy = get_energy(
        system=result.system, x_points=result.x_points, p_points=result.p_points
    )
    for trajectory in range(n_trajectories):
        ax.plot(
            result.times,
            energy[trajectory, :],
            label=f"trajectory {trajectory}",
        )

    ax.set_xlabel("time")
    ax.set_ylabel("energy")

    return fig, ax


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

    potential_func = sp.lambdify(system.lambda_symbols, system.potential_expr, "numpy")
    potential = np.broadcast_to(potential_func(*points.T, *system.params), (n_points,))

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
        system, 0, 3 * system.delta_x * 2, n_points=n_points, ax=ax
    )


def plot_periodic_square_potential_1d(
    system: PeriodicSquareSystem1D, *, n_points: int = 1000, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    """Plot the square periodic potential in 1D."""
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
        (-2 * params.delta_x, -2 * params.delta_x),
        (
            2 * params.delta_x,
            2 * params.delta_x,
        ),
        n_points=n_points,
        ax=ax,
    )


def plot_periodic_potential_square_2d(
    params: PeriodicSquareSystem2D,
    *,
    n_points: tuple[int, int] = (100, 100),
    ax: Axes | None = None,
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot the periodic potential in 2D."""
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
    effective_mass: np.ndarray,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> np.ndarray:
    """Return the exact ballistic ISF for a 1D flat (potential-free) surface."""
    kbt, _, _ = system.kbt, system.m, system.gamma
    inv_m = np.linalg.inv(effective_mass)
    inner_product = np.einsum("i,tij,j->t", delta_k, inv_m, delta_k)
    return np.exp(-(inner_product * kbt / 2) * times**2)


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

    times = times if times is not None else np.linspace(0, 2e-11, 1000)
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
        lambda _p: system.sampling_domain[0],  # x lower (inner) — fixed, one period
        lambda _p: system.sampling_domain[1],  # x upper (inner)
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
    potential_fn = sp.lambdify(
        (*system.coordinate_symbols, *system.parameter_symbols),
        system.potential_expr,
        "jax",
    )
    sigma = jnp.sqrt(system.m * system.kbt)
    x_lo, x_hi = system.sampling_domain

    grid_1d = jnp.linspace(x_lo, x_hi, 50)
    grids = jnp.meshgrid(*([grid_1d] * system.n_dim), indexing="ij")
    v_grid = potential_fn(*[g.ravel() for g in grids], *system.params)
    pdf_max = jnp.exp(-jnp.min(v_grid) / system.kbt)

    def _sample_one(key: jax.Array) -> tuple:
        def _cond_fn(state: tuple) -> bool:
            _, _, _, accepted = state
            return ~accepted

        def _body_fn(state: tuple) -> tuple:
            key, _x, _p, _ = state
            key, kx, ku, kp = jax.random.split(key, 4)

            x_candidate = jax.random.uniform(
                kx, minval=x_lo, maxval=x_hi, shape=(system.n_dim,)
            )
            v = potential_fn(*x_candidate, *system.params)
            x_ok = jax.random.uniform(ku) < jnp.exp(-v / system.kbt) / pdf_max

            p_candidate = jax.random.normal(kp, (system.n_dim,)) * sigma
            energy = jnp.sum(p_candidate**2) / (2 * system.m) + v

            accept = x_ok & (energy > barrier_energy)
            return key, x_candidate, p_candidate, accept

        init = (key, jnp.zeros(system.n_dim), jnp.zeros(system.n_dim), jnp.array(False))  # ruff:ignore[boolean-positional-value-in-call]
        _, x_final, p_final, _ = jax.lax.while_loop(_cond_fn, _body_fn, init)
        return x_final, p_final

    return jax.jit(jax.vmap(_sample_one))


def make_initial_conditions_sampler(system: System) -> Callable:

    potential_fn = sp.lambdify(
        (*system.coordinate_symbols, *system.parameter_symbols),
        system.potential_expr,
        "jax",
    )

    x_lo, x_hi = system.sampling_domain

    grid_1d = jnp.linspace(x_lo, x_hi, 50)
    grids = jnp.meshgrid(*([grid_1d] * system.n_dim), indexing="ij")
    v_grid = potential_fn(*[g.ravel() for g in grids], *system.params)
    pdf_max = jnp.exp(-jnp.min(v_grid) / system.kbt)

    def _sample_one(key: jax.Array) -> tuple:
        def _cond_fn(state: tuple) -> bool:
            _, _, accepted = state
            return ~accepted

        def _body_fn(state: tuple) -> tuple:
            key, _x, _ = state
            key, kx, ku = jax.random.split(key, 3)
            x_candidates = jax.random.uniform(
                kx, minval=x_lo, maxval=x_hi, shape=(system.n_dim,)
            )
            v = potential_fn(*x_candidates, *system.params)
            accept = jax.random.uniform(ku) < jnp.exp(-v / system.kbt) / pdf_max

            return key, x_candidates, accept

        key, p_key = jax.random.split(key, 2)
        init = (key, jnp.zeros(system.n_dim), jnp.array(False))  # ruff:ignore[boolean-positional-value-in-call]
        _, x_finals, _ = jax.lax.while_loop(_cond_fn, _body_fn, init)
        p_std = jnp.sqrt(system.kbt * system.m)
        p_finals = jax.random.normal(p_key, (system.n_dim,)) * p_std

        return x_finals, p_finals

    return jax.jit(jax.vmap(_sample_one))


def period(energy: float, system: PeriodicSystem1D) -> float:
    omega = (2 * np.pi / system.delta_x) * np.sqrt(energy / (2 * system.m))
    q2 = system.barrier_energy / energy
    return 2 * ellipkinc(np.pi, q2) / omega


def sample_energy_1d_periodic(
    system: PeriodicSystem1D, n_samples: int, domain: tuple
) -> np.ndarray[Any, np.dtype[np.floating]]:
    kbt = system.kbt

    class EnergyDensity:
        @staticmethod
        def pdf(energy: float) -> float:
            return np.exp(-energy / kbt) * period(energy, system)

        @staticmethod
        def cdf(energy: float) -> float:
            msg = "CDF is not implemented for EnergyDensity."
            raise NotImplementedError(msg)

        @staticmethod
        def logpdf(energy: float) -> float:
            return -energy / kbt + np.log(period(energy, system))

    energy_sampler = NumericalInversePolynomial(
        EnergyDensity(),
        domain=domain,
        center=domain[0],
    )
    return energy_sampler.rvs(size=n_samples)


def get_elastic_p_exact_1d_periodic(
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

    elastic_ps = get_elastic_p_exact_1d_periodic(
        system=system,
        n_samples=n_samples,
    )
    avg_p2_given_escaped = np.average(elastic_ps**2, axis=0)

    prob_escape = 1 - calculate_probability_under_barrier(
        system=system, barrier_energy=system.barrier_energy
    )

    return (system.kbt * system.m**2) / (prob_escape * avg_p2_given_escaped)


def get_free_effective_mass_exact_1d_periodic(
    system: PeriodicSystem1D, n_samples: int
) -> float:

    elastic_ps = get_elastic_p_exact_1d_periodic(system=system, n_samples=n_samples)

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


def get_elastic_p_exact_1d_square_periodic(
    system: PeriodicSystem1D, n_samples: int
) -> np.ndarray:
    energy = sample_energy_1d_periodic(
        system=system, n_samples=n_samples, domain=(system.barrier_energy, np.inf)
    )
    epsilon = energy / system.barrier_energy

    return np.sqrt(8 * system.barrier_energy * system.m * (epsilon**2 - epsilon)) / (
        np.sqrt(epsilon) + np.sqrt(epsilon - 1)
    )


def get_full_effective_mass_exact_1d_square_periodic(
    system: PeriodicSystem1D, n_samples: int
) -> float:

    elastic_ps = get_elastic_p_exact_1d_square_periodic(
        system=system,
        n_samples=n_samples,
    )
    avg_p2_given_escaped = np.average(elastic_ps**2, axis=0)

    prob_escape = 1 - calculate_probability_under_barrier(
        system=system, barrier_energy=system.barrier_energy
    )

    return (system.kbt * system.m**2) / (prob_escape * avg_p2_given_escaped)


def get_free_effective_mass_exact_1d_square_periodic(
    system: PeriodicSystem1D, n_samples: int
) -> float:

    elastic_ps = get_elastic_p_exact_1d_square_periodic(
        system=system, n_samples=n_samples
    )

    return (system.kbt * system.m**2) / np.average(elastic_ps**2, axis=0)


def get_full_effective_mass_exact_1d_square_periodic_directly(
    system: PeriodicSystem1D,
) -> float:

    u0 = system.barrier_energy / (2 * system.kbt)

    def integrand(epsilon: float) -> np.ndarray:
        return (
            np.exp(-2 * u0 * epsilon)
            * np.sqrt(epsilon**2 - epsilon)
            / (np.sqrt(epsilon) - np.sqrt(epsilon + 1))
        )

    integral, _ = quad(integrand, 1, np.inf)
    dimensionless_factor = np.sqrt(np.pi / u0**3) * np.exp(u0) / np.cosh(u0)

    return system.m * dimensionless_factor / integral


def add_periodic_grid(
    system: PeriodicSquareSystem2D,
    x_range: tuple,
    y_range: tuple,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Draw gridlines matching the periodic cell boundaries, centered on (0,0)."""
    fig, ax = get_figure(ax)
    period = system.delta_x

    n_start = int(np.floor((x_range[0] + period / 2) / period))
    n_end = int(np.ceil((x_range[1] + period / 2) / period))
    x_lines = (np.arange(n_start, n_end + 1) - 0.5) * period

    n_start = int(np.floor((y_range[0] + period / 2) / period))
    n_end = int(np.ceil((y_range[1] + period / 2) / period))
    y_lines = (np.arange(n_start, n_end + 1) - 0.5) * period

    ax.set_xticks(x_lines, minor=True)
    ax.set_yticks(y_lines, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.6)

    return fig, ax
