from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, Unpack, cast

import matplotlib as mpl
import numpy as np
import scipy
import sympy as sp

from classical_diffusion.langevin import (
    SimulationResult,
)
from classical_diffusion.langevin._langevin import sample_results
from classical_diffusion.plot import get_figure, get_measured_data
from classical_diffusion.system._system import System

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection, QuadMesh
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from classical_diffusion.langevin._langevin import SingleSimulationResult
    from classical_diffusion.plot import Measure


def _calculate_total_offsset_multiplications_complex(
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    # scipy.signal.correlate handles complex numbers and conjugation automatically
    # Note: correlate(a, b) conjugates the first argument by default
    return scipy.signal.correlate(lhs, rhs, mode="full")[: lhs.size][::-1]


def _time_average[DT: np.floating](
    time_sum: np.ndarray[Any, np.dtype[DT]],
) -> np.ndarray[Any, np.dtype[DT]]:
    """Apply the time-averaging denominator."""
    size = time_sum.shape[-1]
    return time_sum / np.arange(1, size + 1)[::-1]


def get_isf(
    positions: np.ndarray[Any, np.dtype[np.floating]],
    delta_k: tuple[float, ...],
    *,
    pairwise: bool = True,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """Get the restored displacement of a wavepacket."""
    if not pairwise:
        phase = np.einsum(
            "i,...ij->...j",
            delta_k,
            positions - positions[..., 0].reshape((*positions.shape[:-1], 1)),
        )
        return np.exp(1j * phase)

    scatter = np.exp(-1j * np.einsum("i,...ij->...j", delta_k, positions))

    # convolution_j = \sum_i^N-j e^(ik.x_i+j) e^(-ik.x_i)
    convolution = np.apply_along_axis(
        lambda m: _calculate_total_offsset_multiplications_complex(m, m),
        axis=-1,
        arr=scatter,
    )
    return _time_average(convolution)


class ISFKwargs(TypedDict):
    """Settings controlling how the ISF is computed from trajectory data."""

    delta_k: tuple[float, ...]
    pairwise: NotRequired[bool]


def plot_isf(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    **kwargs: Unpack[ISFKwargs],
) -> tuple[Figure, Axes, Line2D, PolyCollection]:
    """Plot the ensemble-averaged ISF over time, with a shaded ±1 SEM band.

    `time_scale` overrides the result's own characteristic time for
    x-axis normalization — pass the same value to multiple calls to
    compare curves on a shared timescale.
    """
    fig, ax = get_figure(ax)

    isf = get_isf(result.x_points, **kwargs)

    n_trajectories = isf.shape[0]
    avg_isf = np.mean(isf, axis=0)
    sem_isf = np.std(isf, axis=0) / np.sqrt(n_trajectories)

    avg_data = get_measured_data(avg_isf, measure)
    sem_data = get_measured_data(sem_isf, measure)

    (line,) = ax.plot(result.times, avg_data)
    line.set_label("ISF")

    fill = ax.fill_between(
        result.times,
        avg_data - sem_data,
        avg_data + sem_data,
        alpha=0.3,
        label="SEM",
    )
    fill.set_color(line.get_color())

    line.set_label("SEM")

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    return fig, ax, line, fill


def plot_isf_with_delta_k(
    result: SimulationResult,
    delta_k_values: np.ndarray[Any, np.dtype[np.floating]],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    pairwise: bool = True,
) -> tuple[Figure, Axes]:
    """Plot the ensemble-averaged ISF over time, with a shaded ±1 SEM band.

    `time_scale` overrides the result's own characteristic time for
    x-axis normalization — pass the same value to multiple calls to
    compare curves on a shared timescale.
    """
    fig, ax = get_figure(ax)

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(
        vmin=np.min(delta_k_values).item(), vmax=np.max(delta_k_values).item()
    )

    for dk in delta_k_values:
        dk_tuple = (dk,)
        isf = get_isf(result.x_points, delta_k=dk_tuple, pairwise=pairwise)
        avg_isf = np.mean(isf, axis=0)
        avg_data = get_measured_data(avg_isf, measure)
        ax.plot(result.times, avg_data, color=cmap(norm(dk)))

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r"$\Delta k$"
    )

    return fig, ax


def plot_x_evolution(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
    idx: int = 0,
    n_trajectories: int = 1,
    time_scale: float = 1.0,
) -> tuple[Figure, Axes, list[Line2D]]:
    """Plot x against t for the first n_trajectories trajectories.

    Raises
    ------
    ValueError
        If `n_trajectories` exceeds the number of trajectories available in `result`.
    """
    fig, ax = get_figure(ax)

    if n_trajectories > result.x_points.shape[0]:
        msg = f"n_trajectories={n_trajectories} exceeds available trajectories ({result.x_points.shape[0]})"
        raise ValueError(msg)

    scaled_times = result.times / time_scale

    lines = []
    for trajectory in range(n_trajectories):
        (line,) = ax.plot(scaled_times, result.x_points[trajectory, idx])
        lines.append(line)

    ax.set_xlabel("$t / characteristic time$")
    ax.set_ylabel("$x$")

    return fig, ax, lines


def plot_p_evolution(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
    idx: int = 0,
    n_trajectories: int = 1,
    time_scale: float,
) -> tuple[Figure, Axes, list[Line2D]]:
    """Plot p against t for the first n_trajectories trajectories.

    Raises
    ------
    ValueError
        If `n_trajectories` exceeds the number of trajectories available in `result`.
    """
    fig, ax = get_figure(ax)

    if n_trajectories > result.p_points.shape[0]:
        msg = f"n_trajectories={n_trajectories} exceeds available trajectories ({result.p_points.shape[0]})"
        raise ValueError(msg)

    scaled_times = result.times / time_scale

    lines = []
    for trajectory in range(n_trajectories):
        (line,) = ax.plot(scaled_times, result.p_points[trajectory, idx])
        lines.append(line)

    ax.set_xlabel("$t / characteristic time$")
    ax.set_ylabel("$p$")

    return fig, ax, lines


def _get_sampled_kinetic_energies[T: SimulationResult](
    result: T,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    return (np.sum(result.p_points**2, axis=1)) / (
        2 * result.system.m * result.system.kbt
    )


def _get_all_kinetic_energies[T: SimulationResult](
    result: T | list[T],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    result: list[T] = (
        cast("list[T]", [result]) if isinstance(result, SimulationResult) else result
    )
    return np.concatenate([_get_sampled_kinetic_energies(r) for r in result]).ravel()


def plot_kinetic_probability[T: SimulationResult](
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


def split_result(result: SimulationResult) -> tuple[SimulationResult, SimulationResult]:
    """Split a simulation result in half along the time axis, each restarting at t=0."""
    xs1, xs2 = np.split(result.x_points, 2, axis=-1)
    ps1, ps2 = np.split(result.p_points, 2, axis=-1)
    times1, times2 = np.split(result.times, 2)

    times1 -= times1[0]
    times2 -= times2[0]

    first = SimulationResult(
        times=times1, x_points=xs1, p_points=ps1, system=result.system
    )
    second = SimulationResult(
        times=times2, x_points=xs2, p_points=ps2, system=result.system
    )
    return first, second


def x_exact_pdf(result: SimulationResult, *, n_grid: int = 10_000) -> tuple:
    """Return x boltzman pdf for given potential."""
    potential = sp.lambdify(
        (*result.system.coordinate_symbols, *result.system.parameter_symbols),
        result.system.potential_expr,
        "numpy",
    )
    x_grid = np.linspace(result.x_points.min(), result.x_points.max(), n_grid)
    v_grid = np.broadcast_to(potential(x_grid, *result.system.params), x_grid.shape)

    kbt = result.system.kbt

    v_shifted = v_grid - v_grid.min()
    unnormalised = np.exp(-v_shifted / kbt)

    z = np.trapezoid(unnormalised, x_grid)

    return x_grid, unnormalised / z


def plot_x_histogram(
    result: SimulationResult,
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
        result.x_points.reshape(-1),
        bins=bins,
        density=True,
        alpha=1.0,
    )

    x_grid, x_pdf = x_exact_pdf(result)
    ax.plot(x_grid, x_pdf, lw=1.5)

    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")
    ax.legend()

    return fig, ax, cast("BarContainer", bars)


def p_exact_pdf(result: SimulationResult, *, n_grid: int = 10_000) -> tuple:
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
    result: SimulationResult,
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
    ax.legend()

    return fig, ax, cast("BarContainer", bars)


def plot_phase_space_density(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
    bins: int = 100,
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot 2D density map of (x, p) phase space."""
    fig, ax = get_figure(ax)

    _counts, _xedges, _yedges, mesh = ax.hist2d(
        result.x_points.reshape(-1),
        result.p_points.reshape(-1),
        bins=bins,
        density=True,
        cmap="viridis",
    )

    fig.colorbar(mesh, ax=ax, label="Probability Density")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title("Phase Space Density")

    return fig, ax, mesh


def expanding_slope_ensemble(
    times: np.ndarray[Any, np.dtype[np.floating]],
    x_points: np.ndarray[Any, np.dtype[np.floating]],
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Compute the expanding slope of x_points over time for an ensemble of trajectories."""
    n_trajectories, n_timepoints = x_points.shape
    slopes = np.zeros((n_trajectories, n_timepoints))

    for i in range(n_trajectories):
        for j in range(1, n_timepoints):
            slopes[i, j] = (x_points[i, j] - x_points[i, 0]) / (times[j] - times[0])
    # TODO: replace with whatever this was meant to be!
    return slopes


def get_elastic_p(
    result: SimulationResult, *, idx: int = 0
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Return the elastic (ballistic straight-line) momentum estimate per trajectory."""
    x_points = result.x_points[:, idx, :]
    v_elastic = expanding_slope_ensemble(result.times, x_points)
    return v_elastic * result.system.m


def plot_elastic_p(
    result: SimulationResult,
    stride_time: float,
    *,
    n_trajectories: int,
    ax: Axes | None = None,
    time_scale: float = 1.0,
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

    sampled_result = sample_results(result, stride_time=stride_time)
    ps = get_elastic_p(sampled_result)
    for trajectory in range(n_trajectories):
        ax.plot(
            sampled_result.times / time_scale,
            ps[trajectory, :],
            label=f"trajectory {trajectory}",
        )

    ax.set_xlabel("time")
    ax.set_ylabel("p_elastic")

    return fig, ax


def plot_initial_p(
    result: SimulationResult, *, n_trajectories: int, ax: Axes | None = None
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


def breakdown_ballistic_trajectory[S: System](
    result: SimulationResult[S], stride_time: float, idx: int = 0
) -> tuple[SimulationResult[S], SimulationResult[S]]:
    """Split a ballistic simulation into its elastic and inelastic components."""
    sampled_result = sample_results(result, stride_time=stride_time)
    p_elastic = get_elastic_p(sampled_result)
    p_final = p_elastic[..., -1:]
    p_elastic_points = (np.broadcast_to(p_final, result.p_points[:, idx, :].shape))[
        :, None, :
    ]
    x_elastic_points = (
        p_final * result.times / result.system.m + result.x_points[:, :, 0]
    )[:, None, :]

    elastic = SimulationResult(
        times=result.times,
        x_points=x_elastic_points,
        p_points=p_elastic_points,
        system=result.system,
    )
    inelastic = SimulationResult(
        times=result.times,
        x_points=result.x_points - x_elastic_points,
        p_points=result.p_points - p_elastic_points,
        system=result.system,
    )
    return elastic, inelastic


_EXPECTED_NDIM = 2


def plot_2d_trajectory(
    result: SingleSimulationResult, *, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    """Plot x against y for 2d trajectory.

    Raises
    ------
    ValueError
        If n_dimensions is not 2.
    """
    fig, ax = get_figure(ax)

    if result.x_points.shape[0] != _EXPECTED_NDIM:
        msg = "incorrect number of system dimensions, must be 2)"
        raise ValueError(msg)

    (line,) = ax.plot(result.x_points[0], result.x_points[1])

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    return fig, ax, line


def get_energy(result: SimulationResult) -> np.ndarray:
    """Return the energy of the system."""
    potential = sp.lambdify(
        (*result.system.coordinate_symbols, *result.system.parameter_symbols),
        result.system.potential_expr,
        "numpy",
    )

    potential = potential(result.x_points, *result.system.params).squeeze(axis=1)

    kinetic = np.sum(result.p_points**2, axis=1) / (2 * result.system.m)

    return kinetic + potential


def plot_energy(
    result: SimulationResult, n_trajectories: int, *, ax: Axes, time_scale: float = 1.0
) -> tuple[Figure, Axes]:
    """Plot the energy of the system with time."""
    fig, ax = get_figure(ax)
    energy = get_energy(result)
    for trajectory in range(n_trajectories):
        ax.plot(
            result.times / time_scale,
            energy[trajectory, :],
            label=f"trajectory {trajectory}",
        )

    ax.set_xlabel("time")
    ax.set_ylabel("energy")

    return fig, ax


def _partition_result(
    result: SimulationResult, mask: np.ndarray[Any, np.dtype[np.bool_]]
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
]:
    return result.x_points[mask], result.p_points[mask]


def split_escaped_and_trapped(
    result: SimulationResult,
) -> tuple[SimulationResult, SimulationResult]:
    """Split result into trajectories trapped within or free to move over the barrier."""
    energy = get_energy(result)[:, 0]

    free_x_points, free_p_points = _partition_result(
        result, energy > result.system.barrier_energy
    )
    trapped_x_points, trapped_p_points = _partition_result(
        result, energy <= result.system.barrier_energy
    )

    free = SimulationResult(
        times=result.times,
        x_points=free_x_points,
        p_points=free_p_points,
        system=result.system,
    )
    trapped = SimulationResult(
        times=result.times,
        x_points=trapped_x_points,
        p_points=trapped_p_points,
        system=result.system,
    )
    return free, trapped


def get_evolution_trapped_probability(result: SimulationResult) -> np.ndarray:
    """Retrun the evolution of the probability of a particle being trapped as sample size increases."""
    energies = get_energy(result)
    is_over_barrier = energies < result.system.barrier_energy
    return np.cumsum(is_over_barrier, axis=-1) / (
        np.arange(is_over_barrier.shape[-1]) + 1
    )


def get_under_barrier_probability_ballistic(result: SimulationResult) -> np.ndarray:
    """Return the probability of a particle being trapped under barrier."""
    energies = get_energy(result)[:, 0]
    is_over_barrier = energies < result.system.barrier_energy
    return np.sum(is_over_barrier) / is_over_barrier.size


def plot_probability_over_barrier(
    result: SimulationResult, n_trajectories: int, *, ax: Axes
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

    ax.set_xlabel("time")
    ax.set_ylabel("probability")

    return fig, ax


def get_effective_mass(result: SimulationResult) -> float:
    """Return the effective mass averaged over a full simulation."""
    elastic_ps = get_elastic_p(result=result)[:, -1]
    return (result.system.kbt * result.system.m**2) / np.average(elastic_ps**2, axis=0)


def plot_effective_mass_periodic_1D(  # ruff:ignore[invalid-function-name]
    effective_mass: np.ndarray[Any, np.dtype[np.floating]],
    inertial_mass: np.ndarray[Any, np.dtype[np.floating]],
    barrier_energy: np.ndarray[Any, np.dtype[np.floating]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot the effective mass against inertial mass and barrier energy."""
    fig, ax = get_figure(ax)

    scaled_inertial_mass = inertial_mass
    scaled_barrier_energy = barrier_energy

    mesh = ax.pcolormesh(
        scaled_barrier_energy,
        scaled_inertial_mass,
        effective_mass,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax, label="effective mass")

    ax.set_xlabel("Dimensionless Barrier energy")
    ax.set_ylabel("Dimenesionless Inertial mass")

    return fig, ax, mesh
