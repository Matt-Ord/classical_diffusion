from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
import numpy as np
import scipy.stats
import sympy as sp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.signal import butter, sosfiltfilt

from classical_diffusion.langevin import (
    LangevinSimulationResult,
    SingleLangevinSimulationResult,
    get_energy,
)
from classical_diffusion.langevin._system import PeriodicSystem1D, System
from classical_diffusion.plot import get_figure
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


def _get_sampled_kinetic_energies[T: LangevinSimulationResult](
    result: T,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    return (np.sum(result.p_points**2, axis=1)) / (
        2 * result.system.m * result.system.kbt
    )


def _get_all_kinetic_energies[T: LangevinSimulationResult](
    result: T | list[T],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    result: list[T] = (
        cast("list[T]", [result])
        if isinstance(result, LangevinSimulationResult)
        else result
    )
    return np.concatenate([_get_sampled_kinetic_energies(r) for r in result]).ravel()


def plot_kinetic_probability[T: LangevinSimulationResult](
    result: T | list[T],
    *,
    ax: Axes | None = None,
    bins: int = 100,
    max_energy: float = 4.0,
) -> tuple[Figure, Axes, tuple[Line2D, BarContainer]]:
    """Plot the kinetic probabilities for the sample."""
    fig, ax = get_figure(ax)

    kinetic_energy = _get_all_kinetic_energies(result)

    energy_range = (np.min(kinetic_energy), max_energy + np.min(kinetic_energy))
    _bin_counts, bin_edges, bars = ax.hist(
        kinetic_energy,
        bins=bins,
        density=True,
        alpha=0.6,
        color="C0",
        label="Simulation Data",
        range=energy_range,
    )
    (bin_edges[:-1] + bin_edges[1:]) / 2

    def classical_pdf(
        energies: np.ndarray[tuple[int], np.dtype[np.float64]], mu: float
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return (1.0 / np.sqrt(2 * np.pi * energies * mu)) * np.exp(-energies / (2 * mu))

    energies = np.linspace(0.001, bin_edges[-1], 500)

    (line0,) = ax.plot(energies, classical_pdf(energies, mu=0.5))
    line0.set_color("C1")
    line0.set_linestyle("-")
    line0.set_linewidth(2)
    line0.set_label("Theoretical PDF (Mean = 0.5)")

    ax.set_xlim(0, max_energy)

    exponent = np.floor(np.log10(classical_pdf(np.array([max_energy]), mu=0.5)[0]))
    ax.set_ylim(10 ** (exponent - 1), None)
    ax.set_xlabel(r"Kinetic Energy / $k_B T$")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.set_yscale("log")

    return fig, ax, (line0, cast("BarContainer", bars))


def split_result(
    result: LangevinSimulationResult,
) -> tuple[LangevinSimulationResult, LangevinSimulationResult]:
    """Split a simulation result in half along the time axis, each restarting at t=0."""
    xs1, xs2 = np.split(result.x_points, 2, axis=-1)
    ps1, ps2 = np.split(result.p_points, 2, axis=-1)
    times1, times2 = np.split(result.times, 2)

    times1 -= times1[0]
    times2 -= times2[0]

    first = LangevinSimulationResult(
        times=times1, x_points=xs1, p_points=ps1, system=result.system
    )
    second = LangevinSimulationResult(
        times=times2, x_points=xs2, p_points=ps2, system=result.system
    )
    return first, second


def x_exact_pdf(result: LangevinSimulationResult, *, n_grid: int = 10_000) -> tuple:
    """Return x boltzman pdf for given potential."""
    potential = sp.lambdify(
        (*result.system.coordinate_symbols, *result.system.parameter_symbols),
        result.system.potential_expr,
        modules=[{"DerivativeSafeMod": np.mod}, "numpy"],
    )
    x_grid = np.linspace(result.x_points.min(), result.x_points.max(), n_grid)
    v_grid = np.broadcast_to(potential(x_grid, *result.system.params), x_grid.shape)

    kbt = result.system.kbt

    v_shifted = v_grid - v_grid.min()
    unnormalised = np.exp(-v_shifted / kbt)

    z = np.trapezoid(unnormalised, x_grid)

    return x_grid, unnormalised / z


def plot_x_distribution(
    result: LangevinSimulationResult,
    *,
    ax: Axes | None = None,
    x_points: np.ndarray[Any, np.dtype[np.floating]] | None = None,
) -> tuple[Figure, Axes, list[Line2D]]:
    fig, ax = get_figure(ax)

    # Determine evaluation grid for x if not provided
    if x_points is None:
        x_points = np.linspace(np.min(result.x_points), np.max(result.x_points), 200)

    norm = Normalize(vmin=float(result.times[0]), vmax=float(result.times[-1]))
    sm = ScalarMappable(cmap="viridis", norm=norm)
    colors = sm.to_rgba(result.times)
    lines: list[Line2D] = []

    for i in range(len(result.times)):
        sample_points = result.x_points[:, 0, i]
        # Add additional jitter to avoid singularities
        if np.std(sample_points) < 1e-8:  # ruff: ignore[magic-value-comparison]
            rng = np.random.default_rng()
            sample_points += rng.normal(
                0, 1e-3 * np.max(x_points), size=sample_points.shape
            )
        kde = scipy.stats.gaussian_kde(sample_points)
        density = kde(x_points)

        (line,) = ax.plot(x_points, density, color=colors[i])
        lines.append(line)

    fig.colorbar(sm, ax=ax, label="Time / $s$")

    ax.set_xlabel("$x$ / $m$")
    ax.set_ylabel("$P(x)$")
    ax.set_xlim(x_points[0], x_points[-1])

    return fig, ax, lines


def plot_x_histogram(
    result: LangevinSimulationResult,
    *,
    ax: Axes | None = None,
    bins: int = 100,
) -> tuple[Figure, Axes, tuple[Line2D, BarContainer]]:
    """Plot a fancy histogram of periodically sampled position or momentum.

    Subsamples the trajectory every `sample_every` steps (to reduce
    autocorrelation between adjacent time points) before histogramming.
    """
    fig, ax = get_figure(ax)

    _bin_counts, _bin_edges, bars = ax.hist(
        result.x_points[1:].reshape(-1),
        bins=bins,
        density=True,
        alpha=1.0,
    )

    x_grid, x_pdf = x_exact_pdf(result)
    ax.plot(x_grid, x_pdf, lw=1.5)

    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")

    return fig, ax, cast("BarContainer", bars)


def p_exact_pdf(result: LangevinSimulationResult, *, n_grid: int = 10_000) -> tuple:
    """Return p boltzman pdf."""
    p_grid = np.linspace(result.p_points.min(), result.p_points.max(), n_grid)
    m, kbt = (
        result.system.m,
        result.system.kbt,
    )
    pdf_theory = np.sqrt(1 / (2 * np.pi * m * kbt)) * np.exp(
        -(p_grid**2) / (2 * m * kbt)
    )

    return p_grid, pdf_theory


def plot_p_histogram(
    result: LangevinSimulationResult,
    *,
    ax: Axes | None = None,
    bins: int = 100,
) -> tuple[Figure, Axes, tuple[Line2D, BarContainer]]:
    """Plot a fancy histogram of periodically sampled position or momentum.

    Subsamples the trajectory every `sample_every` steps (to reduce
    autocorrelation between adjacent time points) before histogramming.
    """
    fig, ax = get_figure(ax)

    _bin_counts, _bin_edges, bars = ax.hist(
        result.p_points.reshape(-1),
        bins=bins,
        density=True,
        alpha=1.0,
    )

    p_grid, p_pdf = p_exact_pdf(result=result)
    ax.plot(p_grid, p_pdf, lw=1.5)

    ax.set_xlabel("p")
    ax.set_ylabel("Probability Density")

    return fig, ax, cast("BarContainer", bars)


def plot_phase_space_density(
    result: LangevinSimulationResult,
    *,
    ax: Axes | None = None,
    bins: int = 100,
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot 2D density map of (x, p) phase space."""
    fig, ax = get_figure(ax)

    _counts, _xedges, _yedges, mesh = ax.hist2d(
        result.x_points[1:].reshape(-1),
        result.p_points[1:].reshape(-1),
        bins=bins,
        density=True,
        cmap="viridis",
    )

    fig.colorbar(mesh, ax=ax, label="Probability Density")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title("Phase Space Density")

    return fig, ax, mesh


def _get_elastic_velocity_estimates(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the overall average gradient (velocity) across all provided sample points."""
    n = np.arange(1, len(t) + 1)
    st = np.cumsum(t)
    stt = np.cumsum(t * t)
    sy = np.cumsum(y, axis=-1)
    sty = np.cumsum(y * t[None, :], axis=-1)

    denom = n * stt - st**2
    return (n * sty - st * sy) / denom


def _get_elastic_p_estimates(
    result: LangevinSimulationResult, *, max_samples: int = 100
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
]:
    """Return the elastic (ballistic straight-line) momentum estimate per trajectory across all dimensions."""
    n_times = len(result.times)
    n_samples = min(max_samples, n_times)
    sample_indices = jnp.linspace(0, n_times - 1, n_samples, dtype=int)

    t_sampled = result.times[sample_indices]
    x_sampled = result.x_points[:, :, sample_indices]

    v_elastic = _get_elastic_velocity_estimates(t_sampled, x_sampled)
    return v_elastic * result.system.m, t_sampled


def plot_elastic_p(
    result: LangevinSimulationResult,
    *,
    n_trajectories: int,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot convergence of elastic momenta over all trajectories.

    Raises
    ------
    ValueError
        If `n_trajectories` exceeds the number of trajectories available in `result`.
    """
    if n_trajectories > result.p_points.shape[0]:
        msg = f"n_trajectories={n_trajectories} exceeds available trajectories ({result.p_points.shape[0]})"
        raise ValueError(msg)

    fig, ax = get_figure(ax)

    ps, sample_times = _get_elastic_p_estimates(result)
    for trajectory in range(n_trajectories):
        ax.plot(
            sample_times,
            ps[trajectory, :],
            label=f"trajectory {trajectory}",
        )

    ax.set_xlabel("time")
    ax.set_ylabel("p_elastic")

    return fig, ax


def plot_initial_p(
    result: LangevinSimulationResult, *, n_trajectories: int, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """Plot convergence of elastic momenta over all trajectories.

    Raises
    ------
    ValueError
        If `n_trajectories` exceeds the number of trajectories available in `result`.
    """
    if n_trajectories > result.p_points.shape[0]:
        msg = f"n_trajectories={n_trajectories} exceeds available trajectories ({result.p_points.shape[0]})"
        raise ValueError(msg)

    fig, ax = get_figure(ax)

    for trajectory in range(n_trajectories):
        ax.axhline(
            result.p_points[trajectory, 0, 0],
            label=f"trajectory {trajectory}",
            linestyle=":",
        )

    return fig, ax


_EXPECTED_NDIM = 2


def plot_2d_trajectory(
    result: LangevinSimulationResult,
    *,
    ax: Axes | None = None,
    n_trajectories: int = 1,
) -> tuple[Figure, Axes, list[Line2D]]:
    """Plot x against y for 2d trajectory."""
    fig, ax = get_figure(ax)

    lines = []
    for traj in range(n_trajectories):
        (line,) = ax.plot(result.x_points[traj, 0, :], result.x_points[traj, 1, :])
        lines.append(line)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    return fig, ax, lines


def plot_2d_trajectory_single(
    result: LangevinSimulationResult,
    *,
    ax: Axes | None = None,
    start_step: float = 0,
    end_step: float = 0,
) -> tuple[Figure, Axes, Line2D]:
    """Plot x against y for 2d trajectory."""
    end_step = len(result.x_points) - end_step
    fig, ax = get_figure(ax)

    (line,) = ax.plot(
        result.x_points[0, start_step:end_step], result.x_points[1, start_step:end_step]
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    return fig, ax, line


def _partition_result(
    result: LangevinSimulationResult, mask: np.ndarray[Any, np.dtype[np.bool_]]
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
]:
    return result.x_points[mask], result.p_points[mask]


def get_evolution_trapped_probability(result: LangevinSimulationResult) -> np.ndarray:
    """Retrun the evolution of the probability of a particle being trapped as sample size increases."""
    energies = get_energy(result.system, result.x_points, result.p_points)
    is_over_barrier = energies < result.system.barrier_energy
    return np.cumsum(is_over_barrier, axis=-1) / (
        np.arange(is_over_barrier.shape[-1]) + 1
    )


def get_under_barrier_probability_ballistic(
    system: System, x_points: np.ndarray, p_points: np.ndarray, barrier_energy: float
) -> float:
    """Return the probability of a particle being trapped under barrier."""
    energies = get_energy(system, x_points, p_points)
    energies = energies[:, 0]
    is_under_barrier = energies < barrier_energy
    return np.sum(is_under_barrier) / is_under_barrier.size


def plot_probability_over_barrier(
    result: LangevinSimulationResult, n_trajectories: int, *, ax: Axes
) -> tuple[Figure, Axes]:
    """Plot the convergence of the probability of a trajectory having sufficient energy to cross barrier."""
    fig, ax = get_figure(ax)
    probability_evolution = get_evolution_trapped_probability(result)
    for trajectory in range(n_trajectories):
        ax.plot(
            result.times,
            probability_evolution[trajectory, :],
            label=f"trajectory {trajectory}",
        )

    ax.set_xlabel("times")
    ax.set_ylabel("probability")

    return fig, ax


def get_effective_mass(elastic_result: LangevinSimulationResult) -> np.ndarray:
    """Return the effective mass matrix averaged over a full simulation."""
    elastic_ps = elastic_result.p_points
    elastic_ps_squared = np.einsum("nit,njt->nijt", elastic_ps, elastic_ps)
    avg_elastic_ps_squared = np.average(elastic_ps_squared, axis=(0, 3))

    return (elastic_result.system.kbt * elastic_result.system.m**2) * np.linalg.inv(
        avg_elastic_ps_squared
    )


def get_full_effective_mass_from_free(
    elastic_result: LangevinSimulationResult,
    prob_under_barrier: float,
) -> np.ndarray:
    """Return the effective mass, correcting for trapped trajectories analytically."""
    elastic_ps = elastic_result.p_points
    elastic_ps_squared = np.einsum("nit,njt->nijt", elastic_ps, elastic_ps)
    avg_elastic_ps_squared_given_escaped = np.average(elastic_ps_squared, axis=(0, 3))

    prob_escape = 1 - prob_under_barrier

    return (elastic_result.system.kbt * elastic_result.system.m**2) * np.linalg.inv(
        avg_elastic_ps_squared_given_escaped * prob_escape
    )


def plot_2d_gradient(
    x_values: np.ndarray[Any, np.dtype[np.floating]],
    y_values: np.ndarray[Any, np.dtype[np.floating]],
    z_values: np.ndarray[Any, np.dtype[np.floating]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot the effective mass against inertial mass and barrier energy."""
    fig, ax = get_figure(ax)

    mesh = ax.pcolormesh(
        x_values,
        y_values,
        z_values,
        shading="auto",
        cmap="viridis",
    )

    return fig, ax, mesh


def split_escaped_and_trapped(
    result: LangevinSimulationResult[PeriodicSystem1D],
) -> tuple[
    LangevinSimulationResult[PeriodicSystem1D],
    LangevinSimulationResult[PeriodicSystem1D],
]:
    """Split result into trajectories trapped within or free to move over the barrier."""
    energy = get_energy(
        system=result.system, x_points=result.x_points, p_points=result.p_points
    )[:, 0]

    free_x_points, free_p_points = _partition_result(
        result, energy > result.system.barrier_energy
    )
    trapped_x_points, trapped_p_points = _partition_result(
        result, energy <= result.system.barrier_energy
    )

    free = LangevinSimulationResult[PeriodicSystem1D](
        times=result.times,
        x_points=free_x_points,
        p_points=free_p_points,
        system=result.system,
    )
    trapped = LangevinSimulationResult[PeriodicSystem1D](
        times=result.times,
        x_points=trapped_x_points,
        p_points=trapped_p_points,
        system=result.system,
    )
    return free, trapped


@timed
def breakdown_ballistic_trajectory[S: System](
    result: SingleLangevinSimulationResult[S], *, minimum_timescale: float = 0
) -> tuple[
    SingleLangevinSimulationResult[S],
    SingleLangevinSimulationResult[S],
]:
    """Split a ballistic simulation into its elastic (slow) and inelastic (fast) components."""
    times = result.times
    dt = times[1] - times[0]

    # Changes slower than minimum_timescale correspond to frequencies f < 1 / minimum_timescale.
    # High frequencies are filtered out to yield the elastic (slow) component.
    fs = 1.0 / dt
    cutoff_freq = 1.0 / max(minimum_timescale, 1e-5 * dt)
    nyquist = 0.5 * fs

    if cutoff_freq < nyquist:
        sos = butter(N=4, Wn=cutoff_freq / nyquist, btype="low", output="sos")

        # Low-pass filter both momentum and position along the time axis (axis=-1)
        # cspell: disable-next-line  # ruff: ignore[commented-out-code]
        p_elastic_points = sosfiltfilt(sos, result.p_points, axis=-1)
        # Since the filter is a linear operation, it commutes with integration
        # So, filtering the position is equivalent to integrating the filtered momentum
        # cspell: disable-next-line  # ruff: ignore[commented-out-code]
        x_elastic_points = sosfiltfilt(sos, result.x_points, axis=-1)
    else:
        p_elastic_points = result.p_points.copy()
        x_elastic_points = result.x_points.copy()

    elastic = SingleLangevinSimulationResult(
        times=result.times,
        x_points=x_elastic_points,
        p_points=p_elastic_points,
        system=result.system,
    )

    inelastic = SingleLangevinSimulationResult(
        times=result.times,
        x_points=result.x_points - x_elastic_points,
        p_points=result.p_points - p_elastic_points,
        system=result.system,
    )
    return elastic, inelastic


def plot_effective_mass_ratio(
    barrier_energy: np.ndarray[Any, np.dtype[np.floating[Any]]],
    mass_ratio: np.ndarray[Any, np.dtype[np.floating[Any]]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the ratio of effective mass to inertial mass against barrier energy."""
    fig, ax = get_figure(ax)

    (line,) = ax.plot(barrier_energy, mass_ratio[0])
    line.set_label("Effective mass ratio")

    ax.set_title("Effective Mass Ratio vs Barrier Energy")
    ax.set_xlabel("Barrier Energy")
    ax.set_ylabel(r"$m_{\mathrm{eff}} / m$")
    ax.legend()

    return fig, ax, line
