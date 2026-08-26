from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
from scipy.special import ellipk

from classical_diffusion.plot import (
    CAM_BLUE,
    CAM_BLUE_CMAP,
    CAM_CHERRY,
    get_fancy_figure,
    get_figure,
    get_two_panel_figure,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import FillBetweenPolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from classical_diffusion.langevin import PeriodicSystem1D
from typing import TYPE_CHECKING

import scipy
from scipy.constants import atomic_mass, hbar

from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    PeriodicSystem1D,
)


def _with_mass(system: PeriodicSystem1D, m: float) -> PeriodicSystem1D:
    """Return a copy of the system with a new barrier energy."""
    return PeriodicSystem1D(
        gamma=system.gamma,
        temperature=system.temperature,
        m=m,
        delta_x=system.delta_x,
        barrier_energy=system.barrier_energy,
        units=system.units,
        n_dim=system.n_dim,
    )


def _calculate_action(system: PeriodicSystem1D, energy: float) -> float:
    epsilon = energy / system.barrier_energy
    if epsilon <= 1:

        def integrand(u: float) -> float:
            return np.sqrt((u - epsilon) / (u * (1 - u)))

        integral, _ = scipy.integrate.quad(integrand, epsilon, 1)

        return (
            system.delta_x
            / np.pi
            * np.sqrt(2 * system.m * system.barrier_energy)
            * integral
        )

    def integrand(u: float) -> float:
        return np.sqrt((epsilon - u) / (u * (u - 1)))

    integral, _ = scipy.integrate.quad(integrand, 1, epsilon)

    return (
        -system.delta_x
        / np.pi
        * np.sqrt(2 * system.m * system.barrier_energy)
        * integral
    )


def _calculate_tunneling_time(system: PeriodicSystem1D, energy: float) -> float:
    """Exact closed-form -d(action)/dE, via the elliptic-integral derivative identity."""
    epsilon = energy / system.barrier_energy
    if epsilon > 1:
        return 0
    return (
        system.delta_x
        / np.pi
        * np.sqrt(2 * system.m / system.barrier_energy)
        * ellipk(1 - epsilon)
    )


def _calculate_tunneling_probability(system: PeriodicSystem1D, energy) -> float:
    s = _calculate_action(system, energy)
    exponent = 2 * s / hbar
    return 1 / (1 + np.exp(exponent))


def _calculate_time_between_hits(system, energy) -> float:
    epsilon = energy / system.barrier_energy

    def integrand(u: float):
        return 1 / np.sqrt(u * (1 - u) * (epsilon - u))

    if energy > system.barrier_energy:
        integral, _ = scipy.integrate.quad(integrand, 0, 1)
    else:
        integral, _ = scipy.integrate.quad(integrand, 0, epsilon)
    return (
        system.delta_x
        / np.pi
        * np.sqrt(system.m / (2 * system.barrier_energy))
        * integral
    )


def _calculate_quantum_traversal_time(system, energy) -> float:
    prob = _calculate_tunneling_probability(system, energy)
    return _calculate_time_between_hits(
        system, energy
    ) / prob + _calculate_tunneling_time(system, energy)


def _calculate_quantum_p_elastic(system, energy) -> float:
    return system.m * system.delta_x / _calculate_quantum_traversal_time(system, energy)


_ELEMENT_MASSES = {"H": 1.008, "He": 4.003, "Li": 6.94, "Na": 22.990}  # amu


def _add_element_markers(
    cbar: mpl.colorbar.Colorbar,
    elements: dict[str, float] = _ELEMENT_MASSES,
) -> None:
    """Mark specific element masses on a mass colorbar, labeled on the left
    so they don't collide with the colorbar's own tick numbers on the right.
    """
    for label, mass in elements.items():
        if not (cbar.norm.vmin <= mass <= cbar.norm.vmax):
            continue
        cbar.ax.plot(
            0.5,
            mass,
            marker="x",
            markersize=8,
            markeredgewidth=2,
            clip_on=False,
            color=CAM_CHERRY.dark,
        )
        cbar.ax.annotate(
            label,
            xy=(0.0, mass),
            xytext=(-6, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            fontsize=8,
        )


def _plot_period_against_energy_color_map(
    period: np.ndarray[Any, np.dtype[np.floating]],
    energy: np.ndarray[Any, np.dtype[np.floating]],
    masses: np.ndarray[Any, np.dtype[np.floating]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the ensemble-averaged ISF over time, with a shaded ±1 SEM band."""
    fig, ax = get_figure(ax)

    norm = mpl.colors.Normalize(vmin=np.min(masses).item(), vmax=np.max(masses).item())
    cmap = CAM_BLUE_CMAP
    for idx, m in enumerate(masses):
        _, _, line = _plot_period_against_energy(
            energy,
            period[idx],
            ax=ax,
        )
        line.set_color(tuple(cmap(norm(m))))

    ax.set_xlabel("Energy / barrier energy")
    ax.set_ylabel(r"$T_{quantum}(E) / s$")
    color_bar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="mass / au"
    )
    _add_element_markers(color_bar)
    return fig, ax


def _plot_period_against_energy(
    energy: np.ndarray[Any, np.dtype[np.floating[Any]]],
    period: np.ndarray[Any, np.dtype[np.floating[Any]]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the ratio of effective mass to inertial mass against barrier energy."""
    fig, ax = get_figure(ax)

    (line,) = ax.plot(energy, period)

    ax.set_xlabel("Energy / barrier energy")
    ax.set_ylabel(r"$T_{quantum}(E) / s$")  # cspell: disable-line

    return fig, ax, line


def _plot_p_elastic_squared_over_m_against_energy(
    energy: np.ndarray[Any, np.dtype[np.floating[Any]]],
    elastic_p_squared_over_m: np.ndarray[Any, np.dtype[np.floating[Any]]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the ratio of effective mass to inertial mass against barrier energy."""
    fig, ax = get_figure(ax)

    (line,) = ax.plot(energy, elastic_p_squared_over_m)

    ax.set_xlabel("Energy / barrier energy")
    ax.set_ylabel(r"$p_{e}(E)^2 / m$")  # cspell: disable-line

    return fig, ax, line


def _plot_p_elastic_squared_over_m_against_m(
    m: np.ndarray[Any, np.dtype[np.floating[Any]]],
    elastic_p_squared_over_m: np.ndarray[Any, np.dtype[np.floating[Any]]],
    *,
    ax: Axes | None = None,
    classical: float | None = None,
) -> tuple[Figure, Axes, Line2D] | tuple[Figure, Axes, Line2D, Line2D]:
    """Plot the ratio of effective mass to inertial mass against barrier energy."""
    fig, ax = get_figure(ax)

    (line,) = ax.plot(m, elastic_p_squared_over_m)

    ax.set_xlabel("mass / au")
    ax.set_ylabel(r"$p_{e}(E)^2 / m$")  # cspell: disable-line

    if classical is None:
        return fig, ax, line

    line_classical = ax.axhline(
        y=classical, color="k", linestyle=":", label="classical"
    )
    return fig, ax, line, line_classical


def _plot_p_elastic_squared_over_m_against_energy_color_map(
    period: np.ndarray[Any, np.dtype[np.floating]],
    energy: np.ndarray[Any, np.dtype[np.floating]],
    masses: np.ndarray[Any, np.dtype[np.floating]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the ensemble-averaged ISF over time, with a shaded ±1 SEM band."""
    fig, ax = get_figure(ax)

    norm = mpl.colors.Normalize(vmin=np.min(masses).item(), vmax=np.max(masses).item())
    cmap = CAM_BLUE_CMAP
    for idx, m in enumerate(masses):
        _, _, line = _plot_p_elastic_squared_over_m_against_energy(
            energy,
            period[idx],
            ax=ax,
        )
        line.set_color(tuple(cmap(norm(m))))

    ax.set_xlabel("Energy / barrier energy")
    ax.set_ylabel(r"$p_{e}(E)^2 / m$")
    color_bar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="mass / au"
    )
    _add_element_markers(color_bar)
    return fig, ax


def _calculate_partition_function(system: PeriodicSystem1D) -> float:
    """Full partition function integrated over all energy states, epsilon in [0, inf)."""
    u0 = system.barrier_energy / (system.kbt)

    def integrand_trapped(epsilon: float) -> float:
        return ellipk(epsilon) * np.exp(-u0 * epsilon)

    def integrand_running(epsilon: float) -> float:
        return 1 / np.sqrt(epsilon) * ellipk(1 / epsilon) * np.exp(-u0 * epsilon)

    trapped, _ = scipy.integrate.quad(integrand_trapped, 0, 1)
    running, _ = scipy.integrate.quad(integrand_running, 1, np.inf)

    return trapped + running


def _plot_boltzmann_distribution(
    normalized_energy: np.ndarray[Any, np.dtype[np.floating[Any]]],
    system: PeriodicSystem1D,
    ax: Axes,
) -> tuple[Figure, Axes, FillBetweenPolyCollection]:
    fig, ax = get_figure(ax)

    density_of_states = np.where(
        normalized_energy <= 1,
        ellipk(normalized_energy),
        1 / np.sqrt(normalized_energy) * ellipk(1 / normalized_energy),
    )
    weight = density_of_states * np.exp(
        -system.barrier_energy * normalized_energy / system.kbt
    )
    z = _calculate_partition_function(system)
    probability = weight / z

    ax_boltz = ax.twinx()
    dist = ax_boltz.fill_between(normalized_energy, probability, alpha=0.2, zorder=0)
    ax_boltz.set_ylabel("Probability density")
    ax_boltz.set_ylim(bottom=0)
    ax_boltz.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    return fig, ax, dist


def _plot_quantum_classical_comparison() -> None:
    system = SODIUM_COPPER_SYSTEM_1D

    normalized_energies = np.linspace(0.75, 1.5, 2000)
    quantum_period = np.zeros_like(normalized_energies)
    tunneling_time = np.zeros_like(normalized_energies)
    time_between = np.zeros_like(normalized_energies)

    for idx, _ in enumerate(normalized_energies):
        energy = normalized_energies[idx] * system.barrier_energy
        quantum_period[idx] = _calculate_quantum_traversal_time(system, energy)
        tunneling_time[idx] = _calculate_tunneling_time(system, energy)
        time_between[idx] = _calculate_time_between_hits(system, energy)

    classical_period = np.full_like(normalized_energies, 10000)
    above_threshold = np.flatnonzero(normalized_energies > 1)
    for energy_idx in above_threshold:
        energy = normalized_energies[energy_idx] * system.barrier_energy
        classical_period[energy_idx] = _calculate_time_between_hits(system, energy)

    fig, ax = get_fancy_figure()
    _, ax, quantum_line = _plot_period_against_energy(
        ax=ax, period=quantum_period, energy=normalized_energies
    )
    quantum_line.set_label("quantum")
    _, ax, classical_line = _plot_period_against_energy(
        ax=ax, period=classical_period, energy=normalized_energies
    )
    classical_line.set_label("classical")
    classical_line.set_linestyle(":")

    _, ax, dist = _plot_boltzmann_distribution(normalized_energies, system, ax)

    dist.set_color(CAM_CHERRY.dark)
    dist.set_label("boltzman distribution")
    ax.set_yscale("log")

    ax.legend(handles=[classical_line, quantum_line, dist])
    ax.set_ylim(1e-13, 1e-10)
    fig.savefig(
        "examples/ballistic_langevin/quantum_period.classical_comparison.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    fig, ax = get_fancy_figure()
    _, ax, tunneling_line = _plot_period_against_energy(
        ax=ax, period=tunneling_time, energy=normalized_energies
    )
    _, ax, traveling_line = _plot_period_against_energy(
        ax=ax, period=time_between, energy=normalized_energies
    )
    tunneling_line.set_label("tunneling time")
    traveling_line.set_label("travelling line")
    ax.legend(handles=[tunneling_line, traveling_line])
    fig.savefig(
        "examples/ballistic_langevin/quantum_period.tunneling_time.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _plot_quantum_time_period() -> None:
    system = SODIUM_COPPER_SYSTEM_1D

    normalized_masses = np.linspace(1, 28, 8)
    normalized_energies = np.linspace(0.75, 1.5, 2000)

    quantum_periods = []
    for idx, mass in enumerate(normalized_masses):
        quantum_period = np.zeros_like(normalized_energies)
        system = _with_mass(system, m=mass * atomic_mass)
        for idx, _ in enumerate(normalized_energies):
            energy = normalized_energies[idx] * system.barrier_energy
            quantum_period[idx] = _calculate_quantum_traversal_time(system, energy)
        quantum_periods.append(quantum_period)

    classical_period = np.full_like(normalized_energies, 10000)
    above_threshold = np.flatnonzero(normalized_energies > 1)
    for energy_idx in above_threshold:
        energy = normalized_energies[energy_idx] * system.barrier_energy
        classical_period[energy_idx] = _calculate_time_between_hits(system, energy)

    fig, ax = get_fancy_figure()
    _, ax = _plot_period_against_energy_color_map(
        ax=ax,
        period=np.array(quantum_periods),
        energy=normalized_energies,
        masses=normalized_masses,
    )
    ax.set_yscale("log")

    ax.set_ylim(1e-13, 1e-10)
    fig.savefig(
        "examples/ballistic_langevin/quantum_period.m_range.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _calculate_classical_elastic_p_squared_over_m(
    system: PeriodicSystem1D, epsilon: float
) -> float:
    """p_e^2(epsilon) = E_b * epsilon / (2 * pi^2 * K(1/epsilon)^2), valid for epsilon >= 1."""
    if epsilon > 1:
        return (
            system.barrier_energy * epsilon * np.pi**2 / (2 * ellipk(1 / epsilon) ** 2)
        )
    return 0


def _p_elastic_squared_against_energy() -> None:
    system = SODIUM_COPPER_SYSTEM_1D

    normalized_masses = np.linspace(1, 50, 8)
    normalized_energies = np.linspace(0.75, 1.5, 2000)

    classical_p_elastic_squared_over_m = np.array(
        [
            _calculate_classical_elastic_p_squared_over_m(system, eps)
            for eps in normalized_energies
        ]
    )

    quantum_p_elastic_squared_over_ms = []
    for idx, mass in enumerate(normalized_masses):
        p_elastic_squared_over_m = np.zeros_like(normalized_energies)
        system = _with_mass(system, m=mass * atomic_mass)
        for idx, _ in enumerate(normalized_energies):
            energy = normalized_energies[idx] * system.barrier_energy
            p_elastic_squared_over_m[idx] = (
                _calculate_quantum_p_elastic(system, energy)
            ) ** 2 / system.m
        quantum_p_elastic_squared_over_ms.append(p_elastic_squared_over_m)

    fig, ax = get_fancy_figure()
    _, ax = _plot_p_elastic_squared_over_m_against_energy_color_map(
        ax=ax,
        period=np.array(quantum_p_elastic_squared_over_ms),
        energy=normalized_energies,
        masses=normalized_masses,
    )
    fig, ax, classical_line = _plot_p_elastic_squared_over_m_against_energy(
        ax=ax,
        elastic_p_squared_over_m=np.array(classical_p_elastic_squared_over_m),
        energy=normalized_energies,
    )
    classical_line.set_label("classical")
    classical_line.set_linestyle(":")
    fig.savefig(
        "examples/ballistic_langevin/quantum_elastic_p.energy.comparison.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _p_elastic_squared_against_mass() -> None:
    system = SODIUM_COPPER_SYSTEM_1D

    fig, ax = get_two_panel_figure()
    normalized_masses = np.linspace(1, 100, 50)

    epsilon = 1.1
    quantum_p_elastic_squared_over_m = np.zeros_like(normalized_masses)
    for idx, _ in enumerate(normalized_masses):
        mass = normalized_masses[idx] * atomic_mass
        energy = epsilon * system.barrier_energy
        system = _with_mass(system, m=mass)
        quantum_p_elastic_squared_over_m[idx] = (
            _calculate_quantum_p_elastic(system, energy)
        ) ** 2 / system.m
    fig, ax[0], quantum_line, classical_line = _plot_p_elastic_squared_over_m_against_m(  # ty: ignore[invalid-assignment]
        ax=ax[0],
        elastic_p_squared_over_m=np.array(quantum_p_elastic_squared_over_m),
        m=normalized_masses,
        classical=_calculate_classical_elastic_p_squared_over_m(system, epsilon),
    )
    classical_line.set_label("classical")
    classical_line.set_color(CAM_BLUE.dark)
    classical_line.set_linestyle(":")
    quantum_line.set_label("quantum")
    quantum_line.set_color(CAM_BLUE.warm)
    ax[0].set_title(rf"$\epsilon = {epsilon}$")
    ax[0].legend(handles=[quantum_line, classical_line])

    epsilon = 0.9
    quantum_p_elastic_squared_over_m = np.zeros_like(normalized_masses)
    for idx, _ in enumerate(normalized_masses):
        mass = normalized_masses[idx] * atomic_mass
        energy = epsilon * system.barrier_energy
        system = _with_mass(system, m=mass)
        quantum_p_elastic_squared_over_m[idx] = (
            _calculate_quantum_p_elastic(system, energy)
        ) ** 2 / system.m
    fig, ax[1], quantum_line, classical_line = _plot_p_elastic_squared_over_m_against_m(  # ty: ignore[invalid-assignment]
        ax=ax[1],
        elastic_p_squared_over_m=np.array(quantum_p_elastic_squared_over_m),
        m=normalized_masses,
        classical=_calculate_classical_elastic_p_squared_over_m(system, epsilon),
    )
    classical_line.set_label("classical")
    classical_line.set_color(CAM_BLUE.dark)
    classical_line.set_linestyle(":")
    quantum_line.set_label("quantum")
    quantum_line.set_color(CAM_BLUE.warm)
    ax[1].set_title(rf"$\epsilon = {epsilon}$")
    ax[1].set_ylabel("")
    ax[1].legend(handles=[quantum_line, classical_line])

    fig.savefig(
        "examples/ballistic_langevin/quantum_elastic_p.mass.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _calculate_dynamic_energy_cutoff_interp(
    system: PeriodicSystem1D,
    observation_time: float,
    *,
    epsilon_grid: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
) -> float:
    """Find the dynamic/static cutoff energy by interpolating a precomputed
    sweep rather than root-finding -- avoids ever evaluating near threshold.
    """
    if epsilon_grid is None:
        epsilon_grid = np.linspace(1e-3, 0.99, 500)

    traversal_times = np.array(
        [
            _calculate_quantum_traversal_time(system, eps * system.barrier_energy)
            for eps in epsilon_grid
        ]
    )

    # traversal_time spans many orders of magnitude and decreases with epsilon
    # (away from the near-threshold region, excluded via the 0.99 upper bound),
    # so interpolate in log-time for accuracy, and reverse both arrays since
    # np.interp needs its x-array ascending
    log_times = np.log(traversal_times)
    epsilon_cutoff = np.interp(
        np.log(observation_time), log_times[::-1], epsilon_grid[::-1]
    )
    return epsilon_cutoff * system.barrier_energy


def _plot_dynamic_energy_cutoff(
    system: PeriodicSystem1D,
    observation_time: float,
    *,
    ax: Axes,
) -> Line2D:
    """Draw a vertical line at the dynamic/static energy cutoff, normalized by
    barrier energy to match the x-axis convention used elsewhere.
    """  # ruff: ignore[missing-blank-line-after-summary]
    energy_cutoff = _calculate_dynamic_energy_cutoff_interp(system, observation_time)
    epsilon_cutoff = energy_cutoff / system.barrier_energy

    return ax.axvline(x=epsilon_cutoff, linestyle="--", label="dynamic cutoff")


def _classify_dynamic_states() -> None:
    system = SODIUM_COPPER_SYSTEM_1D

    normalized_energies = np.linspace(0.85, 1.1, 2000)

    quantum_p_elastic_squared_over_m = np.zeros_like(normalized_energies)
    for idx, _ in enumerate(normalized_energies):
        energy = normalized_energies[idx] * system.barrier_energy
        quantum_p_elastic_squared_over_m[idx] = (
            _calculate_quantum_p_elastic(system, energy)
        ) ** 2 / system.m

    fig, ax = get_fancy_figure()
    _, ax, quantum_line = _plot_p_elastic_squared_over_m_against_energy(
        ax=ax,
        elastic_p_squared_over_m=np.array(quantum_p_elastic_squared_over_m),
        energy=normalized_energies,
    )
    quantum_line.set_label("quantum")
    _, ax, dist = _plot_boltzmann_distribution(normalized_energies, system, ax)

    observation_time = 1 / system.gamma
    line_cutoff = _plot_dynamic_energy_cutoff(system, observation_time, ax=ax)
    dist.set_color(CAM_CHERRY.dark)
    dist.set_label("boltzman distribution")

    ax.legend(handles=[dist, line_cutoff])
    ax.set_ylim(0)
    fig.savefig(
        "examples/ballistic_langevin/quantum_elastic_p.energy.single.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _calculate_restricted_partition_function(
    system: PeriodicSystem1D, epsilon_min: float
) -> float:
    """Partition function integrated only over dynamic states, epsilon in [epsilon_min, inf)."""
    u0 = system.barrier_energy / system.kbt

    def integrand_trapped(epsilon: float) -> float:
        return ellipk(epsilon) * np.exp(-u0 * epsilon)

    def integrand_running(epsilon: float) -> float:
        return 1 / np.sqrt(epsilon) * ellipk(1 / epsilon) * np.exp(-u0 * epsilon)

    if epsilon_min < 1:
        trapped, _ = scipy.integrate.quad(integrand_trapped, epsilon_min, 1)
        running, _ = scipy.integrate.quad(integrand_running, 1, np.inf)
        return trapped + running

    running, _ = scipy.integrate.quad(integrand_running, epsilon_min, np.inf)
    return running


def _calculate_mean_p_elastic_squared_dynamic(
    system: PeriodicSystem1D, observation_time: float
) -> float:
    """Boltzmann average of p_e^2 restricted to dynamic states -- energies above
    the cutoff where quantum_traversal_time <= observation_time -- so states too
    slow to hop within the observed window are excluded from both the weighting
    integral and the partition function normalizing it.
    """
    epsilon_min = (
        _calculate_dynamic_energy_cutoff_interp(system, observation_time)
        / system.barrier_energy
    )
    u0 = system.barrier_energy / system.kbt

    def weighted_trapped(epsilon: float) -> float:
        energy = epsilon * system.barrier_energy
        p_e = _calculate_quantum_p_elastic(system, energy)
        return p_e**2 * ellipk(epsilon) * np.exp(-u0 * epsilon)

    def weighted_running(epsilon: float) -> float:
        energy = epsilon * system.barrier_energy
        p_e = _calculate_quantum_p_elastic(system, energy)
        return p_e**2 / np.sqrt(epsilon) * ellipk(1 / epsilon) * np.exp(-u0 * epsilon)

    if epsilon_min < 1:
        trapped, _ = scipy.integrate.quad(weighted_trapped, epsilon_min, 1)
        running, _ = scipy.integrate.quad(weighted_running, 1, np.inf)
        weighted_integral = trapped + running
    else:
        weighted_integral, _ = scipy.integrate.quad(
            weighted_running, epsilon_min, np.inf
        )

    z_restricted = _calculate_restricted_partition_function(system, epsilon_min)
    return weighted_integral / z_restricted


def _quantum_effective_mass(system: PeriodicSystem1D, observation_time: float) -> float:
    return (
        system.kbt
        * system.m**2
        / _calculate_mean_p_elastic_squared_dynamic(system, observation_time)
    )


if __name__ == "__main__":
    _plot_quantum_classical_comparison()
    _classify_dynamic_states()
    _plot_quantum_time_period()
    _p_elastic_squared_against_energy()
    _p_elastic_squared_against_mass()
