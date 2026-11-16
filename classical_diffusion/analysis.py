from typing import TYPE_CHECKING, Any, cast, overload

import jax.numpy as jnp
import matplotlib as mpl
import numpy as np

from classical_diffusion._isf import get_isf
from classical_diffusion.jax.langevin import (
    get_trajectory_breakpoints as get_trajectory_breakpoints_jax,
)
from classical_diffusion.jax.langevin import (
    partition_trajectory as partition_trajectory_jax,
)
from classical_diffusion.langevin._langevin import LangevinSimulationResult
from classical_diffusion.plot import get_figure, get_measured_data
from classical_diffusion.simulation import SimulationResult, SingleSimulationResult
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    from classical_diffusion.langevin import SingleLangevinSimulationResult
    from classical_diffusion.plot import Measure
    from classical_diffusion.simulation import IsfResult


__all__ = [
    "get_hop_times",
    "get_isf",
    "partition_trajectory",
    "plot_hop_time_distribution_histogram",
    "plot_isf",
    "plot_isf_with_delta_k",
    "plot_momentum_histogram",
    "plot_p_evolution_1d",
    "plot_p_evolution_2d",
    "plot_root_mean_square_x",
    "plot_x_evolution_1d",
    "plot_x_evolution_2d",
]


def plot_isf(
    result: IsfResult,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D, PolyCollection]:
    """Plot an ISF over time, with a shaded band of one standard error."""
    fig, ax = get_figure(ax)

    data = get_measured_data(result.isf, measure)

    (line,) = ax.plot(result.times, data)
    line.set_label("ISF")

    fill = ax.fill_between(result.times, data - result.error, data + result.error)
    fill.set_alpha(0.3)
    fill.set_color(line.get_color())

    ax.set_xlabel("Time")
    ax.set_ylabel("ISF")
    ax.set_xlim(result.times[0], result.times[-1])

    return fig, ax, line, fill


def plot_isf_with_delta_k(
    results: Iterable[IsfResult],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
) -> tuple[Figure, Axes]:
    """Plot the ISF at each scattering wavevector, colored by its magnitude."""
    fig, ax = get_figure(ax)

    results = list(results)
    delta_k = [float(np.linalg.norm(r.delta_k)) for r in results]
    norm = mpl.colors.Normalize(vmin=min(delta_k), vmax=max(delta_k))
    cmap = mpl.colormaps[mpl.rcParams["image.cmap"]]
    for result, dk in zip(results, delta_k, strict=True):
        _, _, line, poly = plot_isf(result, ax=ax, measure=measure)
        poly.set_alpha(0)
        line.set_color(cmap(norm(dk)))

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r"$\Delta k$"
    )

    return fig, ax


def plot_x_evolution_1d(
    result: SimulationResult | SingleSimulationResult,
    *,
    ax: Axes | None = None,
    idx: int = 0,
) -> tuple[Figure, Axes, list[Line2D]]:
    """Plot x against t for each trajectory."""
    fig, ax = get_figure(ax)

    lines = []
    for res in result if isinstance(result, SimulationResult) else [result]:
        (line,) = ax.plot(res.times, res.x_points[idx])
        lines.append(line)

    if len(lines) > 1:
        for line in lines[:-1]:
            line.set_color("C1")
            line.set_alpha(0.5)

        lines[-1].set_color("C0")

    ax.set_xlabel("$time$")
    ax.set_ylabel("$x$")
    ax.set_xlim(res.times[0], res.times[-1])

    return fig, ax, lines


def plot_x_evolution_2d(
    result: SimulationResult | SingleSimulationResult,
    *,
    ax: Axes | None = None,
    idx: tuple[int, int] = (0, 1),
) -> tuple[Figure, Axes, list[Line2D]]:
    """Plot x against y for each trajectory."""
    fig, ax = get_figure(ax)

    lines = []
    for res in result if isinstance(result, SimulationResult) else [result]:
        (line,) = ax.plot(res.x_points[idx[0]], res.x_points[idx[1]])
        lines.append(line)

    if len(lines) > 1:
        for line in lines[:-1]:
            line.set_color("C1")
            line.set_alpha(0.2)

        lines[-1].set_color("C0")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", adjustable="datalim")

    return fig, ax, lines


def plot_p_evolution_1d(
    result: LangevinSimulationResult | SingleLangevinSimulationResult,
    *,
    ax: Axes | None = None,
    idx: int = 0,
) -> tuple[Figure, Axes, list[Line2D]]:
    """Plot p against t for each trajectory."""
    fig, ax = get_figure(ax)

    lines = []
    for res in result if isinstance(result, LangevinSimulationResult) else [result]:
        (line,) = ax.plot(res.times, res.p_points[idx])
        lines.append(line)

    ax.set_xlabel("$t / characteristic time$")
    ax.set_ylabel("$p$")

    return fig, ax, lines


def plot_p_evolution_2d(
    result: LangevinSimulationResult | SingleLangevinSimulationResult,
    *,
    ax: Axes | None = None,
    idx: tuple[int, int] = (0, 1),
) -> tuple[Figure, Axes, list[Line2D]]:
    """Plot p_{idx[0]} against p_{idx[1]} for each trajectory."""
    fig, ax = get_figure(ax)

    lines = []
    for res in result if isinstance(result, LangevinSimulationResult) else [result]:
        (line,) = ax.plot(res.p_points[idx[0]], res.p_points[idx[1]])
        lines.append(line)

    ax.set_xlabel("$t / characteristic time$")
    ax.set_ylabel("$p$")

    return fig, ax, lines


def _get_delta_x2(
    positions: np.ndarray[Any, np.dtype[np.floating]],
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Calculate <Delta x^2> for all lag times, returning an NxM array."""
    return np.square(positions - positions[:, 0].reshape(-1, 1))


def plot_root_mean_square_x[S: Any](
    result: SimulationResult[S] | SingleSimulationResult[S],
    *,
    ax: Axes | None = None,
    idx: int = 0,
) -> tuple[Figure, Axes, tuple[Line2D, Line2D]]:
    """Plot the root mean square of the displacement over time."""
    fig, ax = get_figure(ax)

    delta_x2 = _get_delta_x2(result.x_points[:, idx])
    average_rms = np.sqrt(np.mean(delta_x2, axis=0))
    (line,) = ax.plot(result.times, average_rms)
    line.set_label(r"$\sqrt{ \Delta x^2 }$")

    std_rms = np.std(np.sqrt(delta_x2), axis=0) / (1 + delta_x2.shape[0])
    fill = ax.fill_between(result.times, average_rms - std_rms, average_rms + std_rms)
    fill.set_color(line.get_color())

    thermal_v = np.sqrt(result.system.kbt / result.system.m)

    (line_b,) = ax.plot(result.times, thermal_v * result.times, linestyle="--")
    line_b.set_label("Ballistic")

    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$\sqrt{ \Delta x^2 }$ /m")
    ax.legend()
    ax.set_xlim(result.times[0], result.times[-1])
    ax.set_ylim(0, np.max(average_rms) * 1.1)

    return fig, ax, (line, line_b)


def _partition_single_trajectory[S: Any](
    result: SingleSimulationResult[S],
    *,
    process_points: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> SingleSimulationResult[S]:
    assert result.x_points.shape[0] == 1, (
        "Only 1d trajectory partitioning is supported."
    )
    data = partition_trajectory_jax(
        jnp.array(result.x_points[0]), process_points=process_points
    )

    return SingleSimulationResult(
        times=result.times,
        x_points=np.array(data.reshape(1, -1)),
        system=result.system,
    )


@overload
def partition_trajectory[S: Any](
    result: SingleSimulationResult[S],
    *,
    process_points: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> SingleSimulationResult[S]: ...


@overload
def partition_trajectory[S: Any](
    result: SimulationResult[S],
    *,
    process_points: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> SimulationResult[S]: ...


@timed
def partition_trajectory[S: Any](
    result: SingleSimulationResult[S] | SimulationResult[S],
    *,
    process_points: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> SingleSimulationResult[S] | SimulationResult[S]:
    """Filter a trajectory using the Kalafut-Visscher step detection algorithm."""  # cspell: disable-line
    if isinstance(result, SingleSimulationResult):
        return _partition_single_trajectory(result, process_points=process_points)
    return SimulationResult.from_iter(
        _partition_single_trajectory(r, process_points=process_points) for r in result
    )


def _get_hop_times_single_trajectory[S: Any](
    result: SingleSimulationResult[S],
    *,
    process_points: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    assert result.x_points.shape[0] == 1, (
        "Only 1d trajectory partitioning is supported."
    )

    breakpoints = get_trajectory_breakpoints_jax(
        jnp.array(result.x_points[0]), process_points=process_points
    )

    true_indices = np.flatnonzero(breakpoints)
    # jnp.diff computes sequence lengths; [:-1] excludes the final sequence
    dt = result.times[1] - result.times[0]
    return np.diff(true_indices)[:-1] * dt


@overload
def get_hop_times[S: Any](
    result: SingleSimulationResult[S],
    *,
    process_points: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> np.ndarray[Any, np.dtype[np.floating]]: ...
@overload
def get_hop_times[S: Any](
    result: SimulationResult[S],
    *,
    process_points: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> np.ndarray[Any, np.dtype[np.floating]]: ...


@timed
def get_hop_times[S: Any](
    result: SingleSimulationResult[S] | SimulationResult[S],
    *,
    process_points: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Get the intervals between hops in a trajectory."""
    if isinstance(result, SingleSimulationResult):
        return _get_hop_times_single_trajectory(result, process_points=process_points)

    intervals = [
        _get_hop_times_single_trajectory(r, process_points=process_points)
        for r in result
    ]
    return np.concatenate(intervals) if intervals else np.array([], dtype=float)


def plot_hop_time_distribution_histogram(
    result: SingleSimulationResult | SimulationResult,
    process_points: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    *,
    ax: Axes | None = None,
    n_bins: int | None = None,
) -> tuple[Figure, Axes, Polygon]:
    """Plot a histogram of the hop times."""
    fig, ax = get_figure(ax)

    hop_times = get_hop_times(result, process_points=process_points)

    n_bins = int(np.sqrt(hop_times.size) / 4) if n_bins is None else n_bins
    bins = np.quantile(hop_times, np.linspace(0, 1, n_bins + 1))
    _bin_counts, _bin_edges, bars = ax.hist(
        hop_times,
        bins=bins,  # ty: ignore[invalid-argument-type]
        density=True,
        histtype="stepfilled",
    )
    (histogram,) = cast("list[Polygon]", bars)
    ax.set_xlabel("hop time")
    ax.set_ylabel("Probability Density")

    return fig, ax, histogram


def plot_momentum_histogram(
    p: np.ndarray[Any, np.dtype[np.floating]],
    *,
    ax: Axes | None = None,
    n_bins: int | None = None,
    p_range: tuple[float, float] | None = None,
) -> tuple[Figure, Axes, Polygon]:
    """Plot a histogram of sampled momenta, in the units of the system."""
    fig, ax = get_figure(ax)

    p_range = p_range or (float(np.min(p)), float(np.max(p)))
    _bin_counts, _bin_edges, bars = ax.hist(
        p.ravel(),
        bins=n_bins if n_bins is not None else int(np.sqrt(p.size)),
        density=True,
        range=p_range,
        histtype="stepfilled",
    )
    (histogram,) = cast("list[Polygon]", bars)
    ax.set_xlabel("p")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(*p_range)

    return fig, ax, histogram
