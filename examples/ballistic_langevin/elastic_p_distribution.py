from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    breakdown_ballistic_trajectory,
    get_exact_elastic_p_distribution,
    get_gamma_elastic_p_distribution,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import CAM_BLUE, CAM_CHERRY, get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import cache_base_path

if TYPE_CHECKING:
    from matplotlib.container import BarContainer

# States are trapped when elastic momentum is
# small compared to the thermal momentum
TRAPPED_MOMENTUM = 1e-2


def _get_simulated_elastic_p() -> np.ndarray:
    """Return |p_e| / sqrt(m kT) of the simulated states above the barrier."""
    system = SODIUM_COPPER_SYSTEM_1D
    result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_end=8 / system.gamma, n_steps=4000),
        n_samples=10000,
    )
    elastic_result, _ = breakdown_ballistic_trajectory(
        result,
        filter_timescale=1 / system.gamma,
    )
    mask = (elastic_result.times >= 2 / system.gamma) & (
        elastic_result.times <= 6 / system.gamma
    )
    elastic_p = np.abs(elastic_result.p_points[:, 0, mask]).reshape(-1)
    elastic_p /= np.sqrt(system.m * system.kbt)
    return elastic_p[elastic_p > TRAPPED_MOMENTUM]


def _plot_elastic_p_distribution_1d() -> None:
    """Plot the distribution of |p_e| above the barrier, compared to a free particle."""
    system = SODIUM_COPPER_SYSTEM_1D
    sigma = np.sqrt(system.m * system.kbt)

    fig, ax = get_fancy_figure()

    _, _, bars = ax.hist(_get_simulated_elastic_p(), bins=40, density=True)
    bars = cast("BarContainer", bars)
    bars.set_label("simulation")
    for bar in bars:
        bar.set_facecolor(CAM_BLUE.light)
        bar.set_edgecolor(CAM_BLUE.light)

    momenta, density = get_exact_elastic_p_distribution(system)
    (exact_line,) = ax.plot(momenta / sigma, density * sigma)
    exact_line.set_label("exact")
    exact_line.set_color(CAM_BLUE.dark)

    p = np.linspace(0, 5, 500)
    (gamma_line,) = ax.plot(
        p, get_gamma_elastic_p_distribution(system, p * sigma) * sigma
    )
    gamma_line.set_label("gamma")
    gamma_line.set_linestyle("--")
    gamma_line.set_color(CAM_BLUE.warm)

    (free_line,) = ax.plot(p, np.sqrt(2 / np.pi) * np.exp(-(p**2) / 2))
    free_line.set_label("free particle")
    free_line.set_linestyle(":")
    free_line.set_color(CAM_CHERRY.dark)

    ax.set_xlabel(r"$|p_e| / \sqrt{m k_B T}$")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, None)
    ax.legend(handles=[bars, exact_line, gamma_line, free_line])
    fig.savefig("examples/ballistic_langevin/elastic_p_distribution.1d.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _plot_elastic_p_distribution_1d()
