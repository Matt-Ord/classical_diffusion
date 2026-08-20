from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, Unpack

import matplotlib as mpl
import numpy as np
import scipy

from classical_diffusion.langevin._langevin import LangevinSimulationResult
from classical_diffusion.plot import get_figure, get_measured_data
from classical_diffusion.simulation import SimulationResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from classical_diffusion.langevin import SingleLangevinSimulationResult
    from classical_diffusion.plot import Measure
    from classical_diffusion.simulation import SingleSimulationResult


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
    """Plot the ensemble-averaged ISF over time, with a shaded ±1 SEM band."""
    fig, ax = get_figure(ax)

    isf = get_isf(result.x_points, **kwargs)
    n_trajectories = isf.shape[0]
    avg_isf = np.mean(isf, axis=0)
    sem_isf = np.std(isf, axis=0) / np.sqrt(n_trajectories)

    avg_data = get_measured_data(avg_isf, measure)
    sem_data = get_measured_data(sem_isf, measure)

    (line,) = ax.plot(result.times, avg_data)
    line.set_label("ISF")

    fill = ax.fill_between(result.times, avg_data - sem_data, avg_data + sem_data)
    fill.set_alpha(0.3)
    fill.set_label("SEM")
    fill.set_color(line.get_color())

    line.set_label("SEM")

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
    """Plot the ensemble-averaged ISF over time, with a shaded ±1 SEM band."""
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
        line.set_color("C0")
        lines.append(line)

    lines[-1].set_color("C1")

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

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

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
