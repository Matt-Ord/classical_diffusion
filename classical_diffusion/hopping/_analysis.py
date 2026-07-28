from typing import TYPE_CHECKING, Any

import numpy as np
from nfft import nfft

from classical_diffusion.plot import get_figure, get_measured_data

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from classical_diffusion.hopping._hopping import HoppingSimulationResult
    from classical_diffusion.plot import Measure


def hopping_isf(
    positions: np.ndarray[Any, np.dtype[np.floating]],
    times: np.ndarray,
    delta_k: np.ndarray,
) -> np.ndarray:
    """Calculate the average ISF of hopping trajectories with uneven times steps."""
    amplitude = np.exp(-1j * np.dot(positions, delta_k))

    interpolated_fts = []
    for traj_index in range(len(amplitude)):
        single_interpolated_ft = nfft(times[traj_index], amplitude[traj_index])
        interpolated_fts.append(single_interpolated_ft)

    interpolated_ft = np.mean(np.array(interpolated_fts), axis=0)
    isf = np.fft.ifft(interpolated_ft * np.conj(interpolated_ft))

    return isf[: len(isf) // 2] / max(isf)


def plot_hopping_isf(
    result: HoppingSimulationResult,
    delta_k: np.ndarray,
    ax: Axes | None = None,
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """Plot the ensemble-averaged ISF over time."""
    fig, ax = get_figure(ax)

    avg_isf = hopping_isf(
        result.x_points,
        result.times,
        delta_k,
    )

    avg_data = get_measured_data(avg_isf, measure)

    # Find the interpolated times steps
    start = 0
    1 / result.lattice.diff_time
    stop = max(max(traj) for traj in result.times)

    (line,) = ax.plot(np.linspace(start, stop, len(avg_data)), avg_data)
    line.set_label("ISF")

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    return fig, ax, line
