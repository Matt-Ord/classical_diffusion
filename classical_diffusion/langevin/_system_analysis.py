from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
import sympy as sp
from scipy.integrate import quad, quad_vec
from scipy.optimize import least_squares
from scipy.special import ellipk, i0e, kve

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


def _get_elastic_weight(
    energy: float | np.ndarray, barrier_energy: float
) -> float | np.ndarray:
    """Return the phase-space weight T(E) exp(-(E - E_b)) of an orbit above the barrier.

    Energies are given in units of kT, and the crossing time T(E) is
    proportional to K(E_b / E) / sqrt(E) for the cosine potential.
    """
    return (
        ellipk(barrier_energy / energy)
        / np.sqrt(energy)
        * np.exp(barrier_energy - energy)
    )


def _get_elastic_momentum(
    energy: float | np.ndarray, barrier_energy: float
) -> float | np.ndarray:
    """Return the elastic momentum p_e(E) = m delta_x / T(E), in units of sqrt(m kT)."""
    return np.pi * np.sqrt(2 * energy) / (2 * ellipk(barrier_energy / energy))


def _get_elastic_partition(barrier_energy: float) -> float:
    """Return the integral of the phase-space weight over states above the barrier."""
    return quad(_get_elastic_weight, barrier_energy, np.inf, args=(barrier_energy,))[0]


def get_exact_elastic_free_probability(system: PeriodicSystem1D) -> float:
    """Return the probability that a particle in a 1D cosine potential is above the barrier."""
    barrier_energy = system.barrier_energy / system.kbt
    # Z = delta_x sqrt(2 pi m kT) exp(-E_b / 2) I_0(E_b / 2), where each direction
    # above the barrier contributes delta_x sqrt(2 m kT) / pi * weight
    normalization = np.pi**1.5 / 2 * i0e(barrier_energy / 2) * np.exp(barrier_energy)
    return _get_elastic_partition(barrier_energy) / normalization


def get_exact_elastic_effective_mass(system: PeriodicSystem1D) -> float:
    """Return the effective mass m kT / <p_e^2> of the states above the barrier."""
    barrier_energy = system.barrier_energy / system.kbt
    p_squared = quad(
        lambda e: (
            _get_elastic_weight(e, barrier_energy)
            * _get_elastic_momentum(e, barrier_energy) ** 2
        ),
        barrier_energy,
        np.inf,
    )[0] / _get_elastic_partition(barrier_energy)
    return system.m / p_squared


def get_exact_elastic_isf(
    system: PeriodicSystem1D,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> np.ndarray:
    """Return the exact elastic ISF of a ballistic particle in a 1D cosine potential.

    The dynamic contribution is the characteristic function of the elastic
    momentum p_e(E), averaged over the orbits above the barrier.
    """
    barrier_energy = system.barrier_energy / system.kbt
    scaled_times = np.linalg.norm(delta_k) * times * np.sqrt(system.kbt / system.m)

    dynamic = quad_vec(
        lambda e: (
            _get_elastic_weight(e, barrier_energy)
            * np.cos(scaled_times * _get_elastic_momentum(e, barrier_energy))
        ),
        barrier_energy,
        # The weight is negligible beyond E_b + 60 kT
        barrier_energy + 60,
    )[0] / _get_elastic_partition(barrier_energy)

    free_probability = get_exact_elastic_free_probability(system)
    return (1 - free_probability) + free_probability * dynamic


def get_exact_elastic_mean_velocity(system: PeriodicSystem1D) -> float:
    """Return the mean elastic speed <|p_e|> / m of the states above the barrier.

    The mean is fixed by transition state theory, A <|p_e|> = m delta_x Gamma_TST,
    so <|p_e|> = sqrt(2 m kT / pi) exp(-E_b / 2kT) / (A I_0(E_b / 2kT)).
    """
    barrier_energy = system.barrier_energy / system.kbt
    free_probability = get_exact_elastic_free_probability(system)
    return (
        np.sqrt(2 * system.kbt / (np.pi * system.m))
        * np.exp(-barrier_energy)
        / (free_probability * i0e(barrier_energy / 2))
    )


def get_barrier_frequency(system: PeriodicSystem1D) -> float:
    """Return the frequency omega_b of the inverted harmonic oscillator at the barrier top."""
    return np.pi / system.delta_x * np.sqrt(2 * system.barrier_energy / system.m)


def get_exact_elastic_p_distribution(
    system: PeriodicSystem1D, *, n_points: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """Return the probability density of |p_e| for the states above the barrier.

    The density follows from the phase-space weight T(E) exp(-E / kT) dE,
    and the elastic momentum p_e(E) = m delta_x / T(E).
    """
    barrier_energy = system.barrier_energy / system.kbt
    # Sample densely close to the barrier, where p_e varies rapidly
    energies = barrier_energy + np.geomspace(1e-12, 60, n_points)
    momenta = _get_elastic_momentum(energies, barrier_energy)
    density = (
        _get_elastic_weight(energies, barrier_energy)
        * np.gradient(energies, momenta)
        / _get_elastic_partition(barrier_energy)
    )

    sigma = np.sqrt(system.m * system.kbt)
    return momenta * sigma, density / sigma


def _get_gig_moment(n: int, lam: float, a: float, b: float) -> float:
    """Return <p^n> of the GIG distribution p^(lam - 1) exp(-(a p + b / p) / 2)."""
    z = np.sqrt(a * b)
    return (b / a) ** (n / 2) * kve(lam + n, z) / kve(lam, z)


def _get_gig_parameters(system: PeriodicSystem1D) -> tuple[float, float, float]:
    """Return the GIG parameters (lam, a, b) of |p_e|, in units of sqrt(m kT).

    The parameters are set by physical quantities of the system. The slowest
    crossings of the barrier top are exponentially rare, at a rate set by the
    barrier frequency, which fixes b = 2 m delta_x omega_b. The parameters lam
    and a are chosen to reproduce the mean speed, fixed by transition state
    theory, and the effective mass.
    """
    sigma = np.sqrt(system.m * system.kbt)
    mean = system.m * get_exact_elastic_mean_velocity(system) / sigma
    p_squared = system.m / get_exact_elastic_effective_mass(system)
    b = 2 * system.m * system.delta_x * get_barrier_frequency(system) / sigma

    # For large sqrt(ab), sqrt(ab) = <p>^2 / Var(p)
    z0 = mean**2 / (p_squared - mean**2)
    lam0 = z0 * (mean * z0 / b - 1) - 0.5

    def _residual(x: np.ndarray) -> list[float]:
        a = np.exp(2 * x[1]) / b
        return [
            np.log(_get_gig_moment(1, x[0], a, b) / mean),
            np.log(_get_gig_moment(2, x[0], a, b) / p_squared),
        ]

    solution = least_squares(_residual, [lam0, np.log(z0)], xtol=1e-14, ftol=1e-14)
    return solution.x[0], np.exp(2 * solution.x[1]) / b, b


def get_gig_elastic_p_distribution(
    system: PeriodicSystem1D,
    momenta: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> np.ndarray:
    """Return the GIG model of the probability density of |p_e| for escaping states."""
    lam, a, b = _get_gig_parameters(system)
    sigma = np.sqrt(system.m * system.kbt)
    # The density vanishes at p = 0, where the slowest crossings are suppressed
    p = np.maximum(np.abs(momenta) / sigma, np.finfo(float).tiny)
    with np.errstate(over="ignore"):
        log_density = (lam - 1) * np.log(p) - (a * p + b / p) / 2
    log_norm = (
        np.log(2 * kve(lam, np.sqrt(a * b))) - np.sqrt(a * b) - lam / 2 * np.log(a / b)
    )
    return np.exp(log_density - log_norm) / sigma


def get_gig_elastic_isf(
    system: PeriodicSystem1D,
    delta_k: tuple[float, ...],
    times: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> np.ndarray:
    """Return the GIG closed form of the elastic ISF in a 1D cosine potential.

    The distribution of |p_e| of the escaping states is modelled by a
    generalised inverse Gaussian p^(lam - 1) exp(-(a p + b / p) / 2), whose
    parameters are set by four physical quantities: the fraction A of escaping
    states, the transition state theory jump rate, the effective mass and the
    barrier frequency. The ISF is
    (1 - A) + A Re[(a / (a - 2iq))^(lam / 2) K_lam(sqrt(b (a - 2iq))) / K_lam(sqrt(ab))],
    where q = delta_k t / m.
    """
    lam, a, b = _get_gig_parameters(system)
    q = np.linalg.norm(delta_k) * times * np.sqrt(system.kbt / system.m)

    z0 = np.sqrt(a * b)
    z1 = np.sqrt(b * (a - 2j * q))
    ratio = kve(lam, z1) / kve(lam, z0) * np.exp(-(z1 - z0))
    dynamic = np.real((a / (a - 2j * q)) ** (lam / 2) * ratio)

    free_probability = get_exact_elastic_free_probability(system)
    return (1 - free_probability) + free_probability * dynamic


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
