from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

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
    system: L,
    probabilities: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    delta_k: float,
) -> np.ndarray:
    distances = system.x_points_from_indices(np.arange(probabilities.shape[1]))
    phase_factors = np.exp(1j * delta_k * distances)
    return np.abs(np.dot(probabilities, phase_factors))


@jax.jit
def get_deterministic_isf_jax[L: Lattice](
    probabilities: jnp.ndarray,
    delta_k: float,
) -> jnp.ndarray:
    distances = jnp.arange(probabilities.shape[1])
    phase_factors = jnp.exp(1j * delta_k * distances)
    return jnp.abs(jnp.dot(probabilities, phase_factors))


@timed
def plot_deterministic_isf[L: Lattice](
    system: L,
    result: DeterministicSolverResult,
    delta_k: float,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the ensemble-averaged ISF over time, with a shaded ±1 SEM band."""
    fig, ax = get_figure(ax)

    isf = get_deterministic_isf(system, result.probabilities, delta_k)
    (line,) = ax.plot(np.array(result.times), np.array(isf))
    line.set_label("ISF")

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    return fig, ax, line
