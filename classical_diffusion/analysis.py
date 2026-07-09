import contextlib
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy

with contextlib.suppress(ImportError):
    pass


from .solve import SimulationResult, get_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


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
    delta_x: np.ndarray[Any, np.dtype[np.float64]],
    pairwise: bool = True,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """Get the restored displacement of a wavepacket."""
    if not pairwise:
        if np.count_nonzero(delta_x) > 0:
            warnings.warn(
                "the width of the wavepacket is ignored for non-pairwise ISF calculation",
                stacklevel=2,
            )
        # If not pairwise, we just calculate the ISF from the initial
        # time
        phase = np.einsum(
            "i,...ij->...j",
            delta_k,
            positions - positions[..., 0].reshape((*positions.shape[:-1], 1)),
        )
        return np.exp(1j * phase)

    scatter = np.exp(-1j * np.einsum("i,...ij->...j", delta_k, positions))
    scatter *= np.exp(-0.5 * np.einsum("i,...ij->...j", delta_k, delta_x) ** 2)

    # convolution_j = \sum_i^N-j e^(ik.x_i+j) e^(-ik.x_i)
    convolution = np.apply_along_axis(
        lambda m: _calculate_total_offsset_multiplications_complex(m, m),
        axis=-1,
        arr=scatter,
    )
    return _time_average(convolution)


def plot_isf(
    result: SimulationResult,
    *,
    delta_k: tuple[float, ...],
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the state occupations of a quantum simulation result."""
    fig, ax = get_figure(ax)

    isf = get_isf(result.xs, delta_k, delta_x=result.params.delta_x)
    (line,) = ax.plot(result.times, isf)
    line.set_label("ISF")

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time /au")
    ax.set_ylabel("ISF")
    ax.legend()

    return fig, ax, line


def plot_x_evolution(
    result: SimulationResult, *, ax: Axes | None = None, idx: int = 0
) -> tuple[Figure, Axes, Line2D]:
    """Plot x against t."""
    fig, ax = get_figure(ax)

    times = result.times - result.times[0]
    (line,) = ax.plot(times, result.xs[idx])
    line.set_label("$x$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")

    return fig, ax, line


def plot_p_evolution(
    result: SimulationResult, *, ax: Axes | None = None, idx: int = 0
) -> tuple[Figure, Axes, Line2D]:
    """Plot p against t."""
    fig, ax = get_figure(ax)

    times = result.times - result.times[0]
    (line,) = ax.plot(times, result.ps[idx])
    line.set_label("$p$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$p$")

    return fig, ax, line
