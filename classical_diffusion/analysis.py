from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import matplotlib as mpl
import numpy as np

from classical_diffusion.jax.analysis import get_pairwise_isf as get_pairwise_isf_jax
from classical_diffusion.langevin._langevin import LangevinSimulationResult
from classical_diffusion.plot import CAM_BLUE_CMAP, get_figure, get_measured_data
from classical_diffusion.simulation import SimulationResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from classical_diffusion.langevin import SingleLangevinSimulationResult
    from classical_diffusion.plot import Measure
    from classical_diffusion.simulation import SingleSimulationResult


def get_isf(
    positions: np.ndarray[Any, np.dtype[np.floating]],
    delta_k: tuple[float, ...],
    *,
    origin_idx: int = 0,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """Get the isf, calculated from with respect to a fixed reference position."""
    phase = np.einsum(
        "i,...ij->...j",
        delta_k,
        positions - positions[..., origin_idx].reshape((*positions.shape[:-1], 1)),
    )
    return np.exp(1j * phase)


def get_pairwise_isf(
    positions: np.ndarray[Any, np.dtype[np.floating]],
    delta_k: tuple[float, ...],
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """Get the ISF, calculated from the pairwise correlation of the positions."""
    return np.array(get_pairwise_isf_jax(jnp.array(positions), jnp.asarray(delta_k)))


def plot_isf(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    delta_k: tuple[float, ...],
    pairwise: bool = True,
) -> tuple[Figure, Axes, Line2D, PolyCollection]:
    """Plot the ensemble-averaged ISF over time, with a shaded ±1 SEM band."""
    fig, ax = get_figure(ax)

    if pairwise:
        isf = get_pairwise_isf(result.x_points, delta_k=delta_k)
        times = result.times - result.times[0]
    else:
        origin_idx = np.argmin(np.abs(result[0].times)).item()
        isf = get_isf(result.x_points, delta_k=delta_k, origin_idx=origin_idx)
        times = result.times - result.times[origin_idx]

    avg_isf = np.mean(isf, axis=0)
    sem_isf = np.std(isf, axis=0) / np.sqrt(isf.shape[0])

    avg_data = get_measured_data(avg_isf, measure)
    sem_data = get_measured_data(sem_isf, measure)

    (line,) = ax.plot(times, avg_data)
    line.set_label("ISF")

    fill = ax.fill_between(times, avg_data - sem_data, avg_data + sem_data)
    fill.set_alpha(0.3)
    fill.set_color(line.get_color())

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")
    ax.set_xlim(times[0], times[-1])

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

    norm = mpl.colors.Normalize(
        vmin=np.min(delta_k_values).item(), vmax=np.max(delta_k_values).item()
    )
    cmap = CAM_BLUE_CMAP
    for dk in delta_k_values:
        _, _, line, poly = plot_isf(
            result,
            ax=ax,
            measure=measure,
            delta_k=(dk,),
            pairwise=pairwise,
        )
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
    x_scale: float = 1,
    y_scale: float = 1,
) -> tuple[Figure, Axes, list[Line2D]]:
    """Plot x against t for each trajectory."""
    fig, ax = get_figure(ax)

    lines = []
    for res in result if isinstance(result, SimulationResult) else [result]:
        (line,) = ax.plot(res.times / x_scale, res.x_points[idx] / y_scale)
        lines.append(line)

    if len(lines) > 1:
        for line in lines[:-1]:
            line.set_color("C1")
            line.set_alpha(0.5)

        lines[-1].set_color("C0")
        lines[-1].set_zorder(10)
        lines[-1].set_alpha(1.0)

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
    ax.set_aspect("equal")

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
