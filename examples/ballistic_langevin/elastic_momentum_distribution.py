from pathlib import Path

import numpy as np
from scipy.stats import gamma as gamma_distribution
from scipy.stats import halfnorm

from classical_diffusion.analysis import plot_momentum_histogram
from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    LangevinSimulationResult,
    PeriodicSystem1D,
    breakdown_ballistic_trajectory,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import CAM_BLUE, get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import unit_with_trajectory_scale
from classical_diffusion.util import cache_base_path


def _with_trajectory_units(system: PeriodicSystem1D) -> PeriodicSystem1D:
    """Measure momentum in units of the thermal momentum sqrt(m k_B T)."""
    return system.with_units(
        unit_with_trajectory_scale(
            system.units,
            length=system.delta_x,
            time=1 / system.gamma,
            momentum=np.sqrt(system.m * system.kbt),
        )
    )


def _get_elastic_momentum(
    result: LangevinSimulationResult,
    *,
    filter_timescale: float,
    window: tuple[float, float],
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Return the magnitude of the elastic momentum of each trajectory.

    The elastic momentum is averaged over a window away from the edges of the
    simulation, where the filter is unaffected by edge effects.
    """
    elastic, _ = breakdown_ballistic_trajectory(
        result, filter_timescale=filter_timescale
    )
    mask = (elastic.times >= window[0]) & (elastic.times <= window[1])
    return np.abs(np.average(elastic.p_points[:, 0, mask], axis=-1))


def _get_gamma_parameters(
    p: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> tuple[float, float]:
    """Return the gamma (shape, scale) which match <p> and <p^2>."""
    mean = float(np.average(p))
    variance = float(np.average(p**2)) - mean**2
    return mean**2 / variance, variance / mean


def _simulate_elastic_momentum(
    system: PeriodicSystem1D, *, n_samples: int
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    # Only trajectories above the barrier escape, and contribute a
    # non-zero elastic momentum
    result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_start=-4 / system.gamma, t_end=4 / system.gamma, n_steps=2000),
        n_samples=n_samples,
        energy_range=(system.barrier_energy, np.inf),
    )
    return _get_elastic_momentum(
        result,
        filter_timescale=1 / system.gamma,
        window=(-2 / system.gamma, 2 / system.gamma),
    )


def _plot_elastic_momentum_distribution_flat() -> None:
    # On a flat surface, the elastic momentum is the momentum, which is
    # distributed according to a half-normal distribution
    system = _with_trajectory_units(
        PeriodicSystem1D(
            gamma=SODIUM_COPPER_SYSTEM_1D.gamma,
            temperature=SODIUM_COPPER_SYSTEM_1D.temperature,
            m=SODIUM_COPPER_SYSTEM_1D.m,
            delta_x=SODIUM_COPPER_SYSTEM_1D.delta_x,
            barrier_energy=0,
        )
    )
    p_elastic = _simulate_elastic_momentum(system, n_samples=30000)

    fig, ax = get_fancy_figure()
    _, _, histogram = plot_momentum_histogram(p_elastic, ax=ax, p_range=(0, 4))
    histogram.set_color(CAM_BLUE.warm)
    histogram.set_label("simulation")

    p = np.linspace(0, 4, 400)
    (line,) = ax.plot(p, halfnorm.pdf(p), color=CAM_BLUE.dark)
    line.set_linestyle("--")
    line.set_label("Maxwell-Boltzmann")

    ax.set_xlabel(r"$p_e / \sqrt{m k_B T}$")
    ax.set_ylabel(r"$\rho(p_e)$")
    ax.legend(handles=[histogram, line])
    fig.savefig("examples/ballistic_langevin/elastic_momentum_distribution.flat.pdf")


def _plot_elastic_momentum_distribution_corrugated() -> None:
    # On a corrugated surface, slow crossings are exponentially rare, so the
    # distribution is peaked away from zero. The gamma distribution is chosen
    # to match the first two moments, <p_e> and <p_e^2>
    system = _with_trajectory_units(SODIUM_COPPER_SYSTEM_1D)
    p_elastic = _simulate_elastic_momentum(system, n_samples=30000)

    fig, ax = get_fancy_figure()
    _, _, histogram = plot_momentum_histogram(p_elastic, ax=ax, p_range=(0, 4))
    histogram.set_color(CAM_BLUE.warm)
    histogram.set_label("simulation")

    shape, scale = _get_gamma_parameters(p_elastic)
    p = np.linspace(0, 4, 400)
    (line,) = ax.plot(
        p, gamma_distribution.pdf(p, shape, scale=scale), color=CAM_BLUE.dark
    )
    line.set_linestyle("--")
    line.set_label("gamma")

    ax.set_xlabel(r"$p_e / \sqrt{m k_B T}$")
    ax.set_ylabel(r"$\rho(p_e)$")
    ax.legend(handles=[histogram, line])
    fig.savefig(
        "examples/ballistic_langevin/elastic_momentum_distribution.corrugated.pdf"
    )


if __name__ == "__main__":
    # These examples plot the distribution of the elastic momentum p_e among
    # trajectories which escape the adsorption site, for sodium on a flat and
    # a corrugated 1D surface. The elastic momentum is extracted by applying a
    # filter with timescale 1 / gamma to ballistic trajectories (see filtered_paths).
    #
    # The corrugated distribution is compared to a gamma distribution, with
    # parameters chosen to match <p_e> and <p_e^2>.
    with cache_base_path(Path("examples/data")):
        _plot_elastic_momentum_distribution_flat()
        _plot_elastic_momentum_distribution_corrugated()
