from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.stats import gamma as gamma_distribution
from scipy.stats import halfnorm

from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    LangevinSimulationResult,
    PeriodicSystem1D,
    breakdown_ballistic_trajectory,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import CAM_BLUE, get_fancy_figure, get_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import cache_base_path

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon


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
    system: PeriodicSystem1D, *, n_samples: int, n_batches: int
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    # Only trajectories above the barrier escape, and contribute a
    # non-zero elastic momentum. Trajectories are simulated in batches
    # to limit the memory required to store each trajectory. The cache
    # does not depend on the random key, so each batch is simulated uncached.
    p_elastic = list[np.ndarray[tuple[int], np.dtype[np.floating]]]()
    for _ in range(n_batches):
        result = solve_ensemble_ballistic.call_uncached(
            system,
            TimeSpan(t_start=-4 / system.gamma, t_end=4 / system.gamma, n_steps=4000),
            n_samples=n_samples,
            energy_range=(system.barrier_energy, np.inf),
        )
        p_elastic.append(
            _get_elastic_momentum(
                result,
                filter_timescale=1 / system.gamma,
                window=(-2 / system.gamma, 2 / system.gamma),
            )
        )
    return np.concatenate(p_elastic)


def _plot_elastic_momentum_histogram(
    p: np.ndarray[tuple[int], np.dtype[np.floating]],
    *,
    ax: Axes | None = None,
    p_range: tuple[float, float],
    n_bins: int | None = None,
) -> tuple[Figure, Axes, Polygon]:
    """Plot a histogram of the elastic momentum distribution."""
    fig, ax = get_figure(ax)
    # A single filled outline avoids the anti-aliased seams between
    # adjacent bars, which appear as gaps of varying width
    _, _, patches = ax.hist(
        p,
        bins=n_bins if n_bins is not None else int(np.sqrt(p.size)),
        density=True,
        alpha=1,
        range=p_range,
        color=CAM_BLUE.warm,
        histtype="stepfilled",
    )
    (histogram,) = cast("list[Polygon]", patches)
    histogram.set_label("simulation")

    ax.set_xlabel(r"$p_e / \sqrt{m k_B T}$")
    ax.set_ylabel(r"$\rho(p_e)$")
    ax.set_xlim(*p_range)
    return fig, ax, histogram


def _plot_elastic_momentum_distribution_1d() -> None:
    corrugated = SODIUM_COPPER_SYSTEM_1D
    flat = PeriodicSystem1D(
        gamma=corrugated.gamma,
        temperature=corrugated.temperature,
        m=corrugated.m,
        delta_x=corrugated.delta_x,
        barrier_energy=0,
    )
    p_thermal = np.sqrt(corrugated.m * corrugated.kbt)
    p_range = (0.0, 4.0)
    p_plot = np.linspace(*p_range, 400)

    # On a flat surface, the elastic momentum is the momentum, which is
    # distributed according to a half-normal distribution
    p_flat = _simulate_elastic_momentum(flat, n_samples=10000, n_batches=3) / p_thermal
    fig, ax = get_fancy_figure()
    _, _, histogram = _plot_elastic_momentum_histogram(p_flat, ax=ax, p_range=p_range)
    (line,) = ax.plot(p_plot, halfnorm.pdf(p_plot), color=CAM_BLUE.dark)
    line.set_linestyle("--")
    line.set_label("Maxwell-Boltzmann")
    ax.legend(handles=[histogram, line])
    fig.savefig("examples/ballistic_langevin/elastic_momentum_distribution.flat.pdf")

    # On a corrugated surface, slow crossings are exponentially rare, so the
    # distribution is peaked away from zero. The gamma distribution is chosen
    # to match the first two moments, <p_e> and <p_e^2>
    p_corrugated = (
        _simulate_elastic_momentum(corrugated, n_samples=10000, n_batches=3) / p_thermal
    )
    shape, scale = _get_gamma_parameters(p_corrugated)
    fig, ax = get_fancy_figure()
    _, _, histogram = _plot_elastic_momentum_histogram(
        p_corrugated, ax=ax, p_range=p_range
    )
    (line,) = ax.plot(
        p_plot,
        gamma_distribution.pdf(p_plot, shape, scale=scale),
        color=CAM_BLUE.dark,
    )
    line.set_linestyle("--")
    line.set_label("gamma")
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
        _plot_elastic_momentum_distribution_1d()
