from typing import TYPE_CHECKING, Any, cast, overload

import matplotlib as mpl
import numpy as np
import scipy.stats
import sympy as sp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.signal import butter, sosfiltfilt  # cspell: disable-line

from classical_diffusion.langevin._langevin import (
    LangevinSimulationResult,
    SingleLangevinSimulationResult,
)
from classical_diffusion.langevin._system import System
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


def _get_energy(
    system: System,
    x_points: np.ndarray,
    p_points: np.ndarray,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Return the energy of the system."""
    potential = sp.lambdify(
        (*system.coordinate_symbols, *system.parameter_symbols),
        system.potential_expr,
        modules=[{"DerivativeSafeMod": np.mod}, "numpy"],
    )

    x_components = [x_points[:, d] for d in range(system.n_dim)]
    potential = potential(*x_components, *system.params)

    kinetic = np.sum(p_points**2, axis=1) / (2 * system.m)

    return kinetic + potential


def plot_energy(
    result: LangevinSimulationResult | SingleLangevinSimulationResult, *, ax: Axes
) -> tuple[Figure, Axes]:
    """Plot the energy of the system with time."""
    fig, ax = get_figure(ax)

    for res in result if isinstance(result, LangevinSimulationResult) else [result]:
        energy = _get_energy(
            system=res.system, x_points=res.x_points, p_points=res.p_points
        )
        ax.plot(result.times, energy)

    ax.set_xlabel("time")
    ax.set_ylabel("energy")

    return fig, ax


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


def _get_exact_x_distribution_pdf(
    result: LangevinSimulationResult, *, n_grid: int = 10_000
) -> tuple:
    """Return x boltzman pdf for given potential."""
    potential = sp.lambdify(
        (*result.system.coordinate_symbols, *result.system.parameter_symbols),
        result.system.potential_expr,
        modules=[{"DerivativeSafeMod": np.mod}, "numpy"],
    )
    x_grid = np.linspace(result.x_points.min(), result.x_points.max(), n_grid)
    v_grid = np.broadcast_to(potential(x_grid, *result.system.params), x_grid.shape)

    v_shifted = v_grid - v_grid.min()
    unnormalised = np.exp(-v_shifted / result.system.kbt)

    z = np.trapezoid(unnormalised, x_grid)

    return x_grid, unnormalised / z


def plot_x_distribution_histogram(
    result: LangevinSimulationResult,
    *,
    ax: Axes | None = None,
    bins: int | None = None,
) -> tuple[Figure, Axes, tuple[Line2D, BarContainer]]:
    """Plot a fancy histogram of periodically sampled position or momentum.

    Subsamples the trajectory every `sample_every` steps (to reduce
    autocorrelation between adjacent time points) before histogramming.
    """
    fig, ax = get_figure(ax)

    _bin_counts, _bin_edges, bars = ax.hist(
        result.x_points[1:].reshape(-1),
        bins=bins or int(np.sqrt(result.x_points.size)),
        density=True,
        alpha=1.0,
    )

    x_grid, x_pdf = _get_exact_x_distribution_pdf(result)
    ax.plot(x_grid, x_pdf, lw=1.5)

    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")

    return fig, ax, cast("BarContainer", bars)


def plot_x_distribution_kde(
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
    sm = ScalarMappable(cmap=mpl.rcParams["image.cmap"], norm=norm)
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
        result.x_points[..., 1:].reshape(-1),
        result.p_points[..., 1:].reshape(-1),
        bins=bins,
        density=True,
        cmap=mpl.rcParams["image.cmap"],
    )

    fig.colorbar(mesh, ax=ax, label="Probability Density")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title("Phase Space Density")

    return fig, ax, mesh


def _get_elastic_p_estimate(
    result: SingleLangevinSimulationResult, *, filter_timescale: float = 0
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
]:
    """Return the elastic (ballistic straight-line) momentum estimate per trajectory across all dimensions."""
    elastic, _inelastic = breakdown_ballistic_trajectory(
        result,
        filter_timescale=filter_timescale,
    )
    return elastic.p_points, elastic.times


def plot_elastic_p(
    result: LangevinSimulationResult | SingleLangevinSimulationResult,
    *,
    ax: Axes | None = None,
    filter_timescale: float = 0,
) -> tuple[Figure, Axes]:
    """Plot elastic momenta over all trajectories."""
    fig, ax = get_figure(ax)

    for res in result if isinstance(result, LangevinSimulationResult) else [result]:
        ps, sample_times = _get_elastic_p_estimate(
            res, filter_timescale=filter_timescale
        )
        ax.plot(sample_times, ps)

    ax.set_xlabel("time")
    ax.set_ylabel("p_elastic")

    return fig, ax


def get_under_barrier_occupation(
    system: System, x_points: np.ndarray, p_points: np.ndarray, barrier_energy: float
) -> float:
    """Return the probability of a particle being trapped under barrier."""
    energies = _get_energy(system, x_points, p_points)
    is_under_barrier = energies < barrier_energy
    return np.sum(is_under_barrier) / is_under_barrier.size


def _truncate_results[S: System](
    result: LangevinSimulationResult[S],
    times: tuple[float, float],
) -> LangevinSimulationResult[S]:
    mask = (result.times >= times[0]) & (result.times <= times[1])
    return LangevinSimulationResult(
        system=result.system,
        times=result.times[mask],
        x_points=result.x_points[:, :, mask],
        p_points=result.p_points[:, :, mask],
    )


def get_effective_mass(
    result: LangevinSimulationResult,
    *,
    under_barrier_probability: float = 0,
    filter_timescale: float = 0,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Return the effective mass, correcting for trapped trajectories analytically."""
    elastic_result = breakdown_ballistic_trajectory(
        result, filter_timescale=filter_timescale
    )[0]

    elastic_result = _truncate_results(
        elastic_result, times=(filter_timescale, result.times[-1] - filter_timescale)
    )

    elastic_p_squared = np.einsum(  # cspell: disable-next-line
        "nit,njt->nijt", elastic_result.p_points, elastic_result.p_points
    )
    escaped_p_squared = np.average(elastic_p_squared, axis=(0, 3))

    escape_probability = 1 - under_barrier_probability

    return (elastic_result.system.kbt * elastic_result.system.m**2) * np.linalg.inv(
        escaped_p_squared * escape_probability
    )


def _breakdown_langevin_simulation_result[S: System](
    result: LangevinSimulationResult[S], *, filter_timescale: float = 0
) -> tuple[LangevinSimulationResult[S], LangevinSimulationResult[S]]:
    times = result.times
    dt = times[1] - times[0]

    # Changes slower than filter_timescale correspond to frequencies f < 1 / filter_timescale.
    # High frequencies are filtered out to yield the elastic (slow) component.
    fs = 1.0 / dt
    cutoff_freq = 1.0 / max(filter_timescale, 1e-5 * dt)
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

    elastic = LangevinSimulationResult(
        times=result.times,
        x_points=x_elastic_points,
        p_points=p_elastic_points,
        system=result.system,
    )

    inelastic = LangevinSimulationResult(
        times=result.times,
        x_points=result.x_points - x_elastic_points,
        p_points=result.p_points - p_elastic_points,
        system=result.system,
    )
    return elastic, inelastic


@overload
def breakdown_ballistic_trajectory[S: System](
    result: SingleLangevinSimulationResult[S],
    cutoff: float = 0,
    *,
    filter_timescale: float = 0,
) -> tuple[SingleLangevinSimulationResult[S], SingleLangevinSimulationResult[S]]: ...


@overload
def breakdown_ballistic_trajectory[S: System](
    result: LangevinSimulationResult[S],
    cutoff: float = 0,
    *,
    filter_timescale: float = 0,
) -> tuple[LangevinSimulationResult[S], LangevinSimulationResult[S]]: ...


@timed
def breakdown_ballistic_trajectory[S: System](
    result: SingleLangevinSimulationResult[S] | LangevinSimulationResult[S],
    *,
    filter_timescale: float = 0,
) -> (
    tuple[SingleLangevinSimulationResult[S], SingleLangevinSimulationResult[S]]
    | tuple[LangevinSimulationResult[S], LangevinSimulationResult[S]]
):
    """Split a ballistic simulation into its elastic (slow) and inelastic (fast) components."""
    if isinstance(result, SingleLangevinSimulationResult):
        elastic_batch, inelastic_batch = _breakdown_langevin_simulation_result(
            LangevinSimulationResult.from_iter([result]),
            filter_timescale=filter_timescale,
        )
        return elastic_batch[0], inelastic_batch[0]

    return _breakdown_langevin_simulation_result(
        result, filter_timescale=filter_timescale
    )
