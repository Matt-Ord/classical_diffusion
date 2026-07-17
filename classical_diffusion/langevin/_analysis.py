from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, Unpack, cast

import numpy as np
import scipy
import sympy as sp

from classical_diffusion.langevin import (
    SimulationResult,
)
from classical_diffusion.plot import get_figure, get_measured_data

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

    line.set_label("SEM")

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    return fig, ax, line, fill


def plot_x_evolution(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
    idx: int = 0,
    n_trajectories: int = 1,
    time_scale: float,
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
        result.system.symbolic_coordinates,
        result.system.potential_expr,
        "numpy",
    )
    x_grid = np.linspace(result.x_points.min(), result.x_points.max(), n_grid)
    v_grid = np.broadcast_to(potential(x_grid), x_grid.shape)

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


def get_elastic_p(
    result: SimulationResult,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Return the elastic (ballistic straight-line) momentum estimate per trajectory."""
    x0 = result.x_points[:, :, 0]
    v_elastic = (result.x_points - x0) / result.times
    return v_elastic * result.system.m


def plot_elastic_p(
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

    ps = get_elastic_p(result)
    for trajectory in range(n_trajectories):
        ax.plot(result.times, ps[trajectory, 0], label=f"trajectory {trajectory}")

    ax.set_xlabel("time")
    ax.set_ylabel("p_elastic")

    return fig, ax


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
