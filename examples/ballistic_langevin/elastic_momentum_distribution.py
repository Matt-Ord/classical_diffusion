from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ellipk
from scipy.stats import gamma as gamma_distribution
from scipy.stats import halfnorm

from classical_diffusion.analysis import plot_isf, plot_momentum_histogram
from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    LangevinSimulationResult,
    PeriodicSystem1D,
    System,
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
    elastic: LangevinSimulationResult,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Return the magnitude of the time averaged elastic momentum of each trajectory."""
    return np.abs(np.average(elastic.p_points[:, 0, :], axis=-1))


def _get_exact_parameters(system: PeriodicSystem1D) -> tuple[float, float, float]:
    """Return the exact dynamic fraction A, TST rate Gamma_TST and effective mass.

    Each energy E = epsilon E_b is weighted by e^(-beta E) tau(E), where tau(E)
    is the crossing time of a unit cell above the barrier and the half period
    below it. In units of (Delta x / pi) sqrt(m / 2 E_b), the crossing time is
    2 K(1 / epsilon) / sqrt(epsilon) and the half period is 2 K(epsilon).
    """
    u0 = system.barrier_energy / system.kbt
    time_scale = (
        system.delta_x / np.pi * np.sqrt(system.m / (2 * system.barrier_energy))
    )

    def crossing_time(epsilon: float) -> float:
        return 2 * ellipk(1 / epsilon) / np.sqrt(epsilon)

    def trapped(epsilon: float) -> float:
        return np.exp(u0 * (1 - epsilon)) * 2 * ellipk(epsilon)

    def escaping(x: float, power: int) -> float:
        return np.exp(-u0 * x) * crossing_time(1 + x) ** power

    # Each integral is scaled by E_b e^(-beta E_b), which cancels in every ratio
    trapped_integral, _ = quad(trapped, 0, 1, limit=500)
    escaping_integral, _ = quad(escaping, 0, np.inf, args=(1,), limit=500)
    inverse_integral, _ = quad(escaping, 0, np.inf, args=(-1,), limit=500)
    partition = escaping_integral + trapped_integral

    dynamic_fraction = escaping_integral / partition
    # Gamma_TST = k_B T e^(-beta E_b) / 2 Z_E
    rate = system.kbt / (2 * system.barrier_energy * time_scale * partition)
    # k_B T / m_eff = Delta x^2 / Z_E integral_(E_b)^infinity e^(-beta E) / tau(E)
    effective_mass = (
        system.kbt * time_scale**2 * partition / (system.delta_x**2 * inverse_integral)
    )
    return dynamic_fraction, rate, effective_mass


def _get_gamma_parameters(system: PeriodicSystem1D) -> tuple[float, float, float]:
    """Return A and the gamma (shape, scale) of the escaping trajectories.

    The mean p_bar = 2 m Gamma_TST Delta x / A and the shape kappa are fixed by
    the exact parameters, such that <p_e> = 2 m Gamma_TST Delta x and
    <p_e^2> = m^2 k_B T / m_eff over every trajectory.
    """
    dynamic_fraction, rate, effective_mass = _get_exact_parameters(system)
    mean = 2 * system.m * rate * system.delta_x / dynamic_fraction
    shape = 1 / (
        system.m**2 * system.kbt / (dynamic_fraction * effective_mass * mean**2) - 1
    )
    return dynamic_fraction, shape, mean / shape


def _truncate_results[S: System](
    result: LangevinSimulationResult[S],
    times: tuple[float, float],
) -> LangevinSimulationResult[S]:
    mask = (result.times >= times[0]) & (result.times <= times[1])
    return LangevinSimulationResult(
        system=result.system,
        times=result.times[mask],
        x_points=result.x_points[:, :, mask],
        p_points=result.p_points[:, :, mask],
    )


def _simulate_elastic_trajectories[S: PeriodicSystem1D](
    system: S, *, n_samples: int, escaping_only: bool = True
) -> LangevinSimulationResult[S]:
    # Only trajectories above the barrier escape, and contribute a
    # non-zero elastic momentum
    result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_start=-4 / system.gamma, t_end=4 / system.gamma, n_steps=2000),
        n_samples=n_samples,
        energy_range=(system.barrier_energy if escaping_only else -np.inf, np.inf),
    )
    elastic, _ = breakdown_ballistic_trajectory(
        result, filter_timescale=1 / system.gamma
    )
    # Keep a window away from the edges of the simulation, where the
    # filter is unaffected by edge effects
    return _truncate_results(elastic, times=(-2 / system.gamma, 2 / system.gamma))


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
    p_elastic = _get_elastic_momentum(
        _simulate_elastic_trajectories(system, n_samples=30000)
    )

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
    # distribution is peaked away from zero. The gamma distribution is
    # calculated from the exact A, Gamma_TST and m_eff
    system = _with_trajectory_units(SODIUM_COPPER_SYSTEM_1D)
    p_elastic = _get_elastic_momentum(
        _simulate_elastic_trajectories(system, n_samples=30000)
    )

    fig, ax = get_fancy_figure()
    _, _, histogram = plot_momentum_histogram(p_elastic, ax=ax, p_range=(0, 4))
    histogram.set_color(CAM_BLUE.warm)
    histogram.set_label("simulation")

    _, shape, scale = _get_gamma_parameters(system)
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


def _plot_elastic_isf_corrugated() -> None:
    # Trapped trajectories have no elastic momentum, and contribute a constant
    # 1 - A to the ISF. The ISF of the escaping trajectories is the cosine
    # transform of the distribution of elastic momentum. The gamma distribution
    # calculated from the exact A, Gamma_TST and m_eff has the analytic transform
    # Re[(1 - i dk t theta / m)^-k], so the ISF is approximately
    # (1 - A) + A Re[(1 - i dk t theta / m)^-k]
    system = _with_trajectory_units(SODIUM_COPPER_SYSTEM_1D)
    elastic = _simulate_elastic_trajectories(
        system, n_samples=30000, escaping_only=False
    )
    # 0.43 A^-1, measured in units of 1 / delta_x
    delta_k = 0.43e10 * SODIUM_COPPER_SYSTEM_1D.delta_x

    fig, ax = get_fancy_figure()
    _, _, line, fill = plot_isf(
        elastic, ax=ax, delta_k=(delta_k,), measure="real", pairwise=False
    )
    line.set_color(CAM_BLUE.warm)
    fill.set_color(CAM_BLUE.warm)
    line.set_label("simulation")

    escaping_fraction, shape, scale = _get_gamma_parameters(system)
    times = np.linspace(0, 2 / system.gamma, 400)
    gamma_isf = (1 - escaping_fraction) + escaping_fraction * np.real(
        (1 - 1j * delta_k * times * scale / system.m) ** (-shape)
    )
    (gamma_line,) = ax.plot(times, gamma_isf, color=CAM_BLUE.dark)
    gamma_line.set_linestyle("--")
    gamma_line.set_label("approximate")

    ax.set_xlim(0, 2 / system.gamma)
    ax.set_ylim(0.6, 1)
    ax.set_xlabel(r"$t \gamma$")
    ax.set_ylabel(r"$I(\Delta k, t)$")
    ax.legend(handles=[line, gamma_line])
    fig.savefig("examples/ballistic_langevin/elastic_momentum_distribution.isf.pdf")


if __name__ == "__main__":
    # These examples plot the distribution of the elastic momentum p_e among
    # trajectories which escape the adsorption site, for sodium on a flat and
    # a corrugated 1D surface. The elastic momentum is extracted by applying a
    # filter with timescale 1 / gamma to ballistic trajectories (see filtered_paths).
    #
    # The corrugated distribution is compared to a gamma distribution, with
    # parameters calculated from the exact dynamic fraction A, TST rate
    # Gamma_TST and effective mass m_eff, and the ISF of the elastic
    # trajectories is compared to (1 - A) + A times the cosine transform of
    # the gamma distribution, where A is the fraction of escaping trajectories.
    with cache_base_path(Path("examples/data")):
        _plot_elastic_momentum_distribution_flat()
        _plot_elastic_momentum_distribution_corrugated()
        _plot_elastic_isf_corrugated()
