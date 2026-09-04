import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import ticker
from scipy.constants import Boltzmann

from classical_diffusion.analysis import (
    partition_trajectory,
    plot_hop_time_distribution_histogram,
    plot_x_evolution_1d,
)
from classical_diffusion.hopping import get_kramers_rate
from classical_diffusion.langevin import (
    KramersParameters,
    KramersSystem1D,
    solve_ensemble_overdamped,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import cache_base_path, timed


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ProcessPeriodicPoints:
    """Callable class to process points for periodic systems."""

    delta_x: float

    def __init__(self, delta_x: float) -> None:
        self.delta_x = delta_x

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        """Process points for periodic systems."""
        return jnp.round(u / self.delta_x) * self.delta_x


@timed
def _plot_filtered_trajectory() -> None:

    system = KramersSystem1D(
        params=KramersParameters(
            omega_well=1.0,
            omega_barrier=1.0,
            barrier_energy=1.0,
            m=1.0,
            temperature=0.5 / Boltzmann,
            gamma=0.1,
        )
    )

    result = solve_ensemble_overdamped(
        system, TimeSpan(t_start=0.0, t_end=100.0, n_steps=1000), n_samples=1
    )

    fig, ax = get_fancy_figure()
    _, _, line = plot_x_evolution_1d(result, ax=ax)
    line[0].set_label("Full Trajectory")
    filtered_result = partition_trajectory(
        result, process_points=ProcessPeriodicPoints(system.delta_x)
    )

    _, _, line = plot_x_evolution_1d(filtered_result, ax=ax)
    line[0].set_label("Filtered Trajectory")

    # Show interval lines at each minima of V(x)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(system.delta_x))
    ax.grid(visible=True, axis="y", color="gray", linestyle="--", linewidth=0.7)

    fig.savefig("./examples/hopping_model/discrete_trajectory.trajectory.pdf")


@timed
def _plot_hop_intervals_histogram() -> None:

    system = KramersSystem1D(
        params=KramersParameters(
            omega_well=1.0,
            omega_barrier=1.0,
            barrier_energy=1.0,
            m=1.0,
            temperature=0.5 / Boltzmann,
            gamma=0.1,
        )
    )

    result = solve_ensemble_overdamped(
        system, TimeSpan(t_start=0.0, t_end=100.0, n_steps=10000), n_samples=1000
    )

    fig, ax = get_fancy_figure()

    average = 1 / (2 * get_kramers_rate(system.kramers_params))
    times = np.linspace(0, 2 * average, 100)
    kramers_pdf = (1 / average) * np.exp(-times / average)
    (line,) = ax.plot(times, kramers_pdf)
    line.set_label("Kramers Theory")
    line.set_linestyle("--")

    plot_hop_time_distribution_histogram(
        result,
        process_points=ProcessPeriodicPoints(system.delta_x),
        ax=ax,
    )
    ax.set_xlim(times[0], times[-1])
    ax.legend()

    ax.set_xlabel("Hop Time")
    ax.set_ylabel("Frequency")

    fig.savefig("./examples/hopping_model/discrete_trajectory.hop_intervals.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _plot_filtered_trajectory()
        _plot_hop_intervals_histogram()
