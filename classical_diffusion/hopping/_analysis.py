from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

import classical_diffusion.jax as jx
from classical_diffusion.hopping._system import Lattice
from classical_diffusion.plot import get_figure
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from classical_diffusion.hopping._hopping import DeterministicSolverResult


@timed
def get_deterministic_isf[L: Lattice](
    result: DeterministicSolverResult[L],
    delta_k: tuple[float, ...],
) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    return np.array(
        jx.get_deterministic_isf(
            result.system.as_canonical(), jnp.array(result.probabilities), delta_k
        )
    )


@timed
def plot_deterministic_isf[L: Lattice](
    result: DeterministicSolverResult[L],
    delta_k: tuple[float, ...],
    *,
    ax: Axes | None = None,
    amplitude: float = 1.0,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the ensemble-averaged ISF over time, with a shaded ±1 SEM band."""
    fig, ax = get_figure(ax)

    isf = get_deterministic_isf(result, delta_k)
    (line,) = ax.plot(np.array(result.times), amplitude * np.array(isf))
    line.set_label("ISF")

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    return fig, ax, line
