from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.random as jrandom
import numpy as np
from scipy.integrate import quad
from scipy.special import ellipk
from tqdm import tqdm

from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    PeriodicSystem1D,
    get_effective_mass,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import CAM_BLUE, CAM_CHERRY, get_fancy_figure, get_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import cached, disabled_timing, hash_array

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


def with_barrier_energy(
    system: PeriodicSystem1D, barrier_energy: float
) -> PeriodicSystem1D:
    """Return a copy of the system with a new barrier energy."""
    return PeriodicSystem1D(
        gamma=system.gamma,
        temperature=system.temperature,
        m=system.m,
        delta_x=system.delta_x,
        barrier_energy=barrier_energy,
        units=system.units,
        n_dim=system.n_dim,
    )


def _add_effective_mass_low_u0_asymptote(
    u0_max: float,
    u0_min: float,
    *,
    ax: Axes | None = None,
    n_points: int = 200,
) -> tuple[Figure, Axes, Line2D]:
    """Overlay the u0 -> 0 asymptote 1 - (4/pi^1.5) sqrt(u0)."""
    fig, ax = get_figure(ax)

    u0 = np.logspace(np.log10(u0_min), np.log10(u0_max), n_points)
    asymptote = 1 - (4 / np.pi**1.5) * np.sqrt(u0)

    (line,) = ax.plot(
        u0,
        asymptote,
        linestyle="--",
        label=r"$1 - (4/\pi^{3/2})\sqrt{u_0}$",
    )
    return fig, ax, line


def _get_exact_effective_mass_ratio(
    system: PeriodicSystem1D,
) -> float:
    u0 = system.barrier_energy / (system.kbt)

    def integrand_denominator(epsilon: float) -> float:
        return np.sqrt(epsilon) / ellipk(1 / epsilon) * np.exp(-u0 * epsilon)

    def integrand_partition(epsilon: float) -> float:
        return 1 / np.sqrt(epsilon) * ellipk(1 / epsilon) * np.exp(-u0 * epsilon)

    denominator_integral, _ = quad(integrand_denominator, 1, np.inf)
    partition_integral, _ = quad(integrand_partition, 1, np.inf)

    return 2 * partition_integral / (denominator_integral * u0 * np.pi**2)


def plot_effective_mass_ratio_against_energy(
    barrier_energy: np.ndarray[Any, np.dtype[np.floating[Any]]],
    mass_ratio: np.ndarray[Any, np.dtype[np.floating[Any]]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the ratio of effective mass to inertial mass against barrier energy."""
    fig, ax = get_figure(ax)

    (line,) = ax.plot(barrier_energy, mass_ratio)

    ax.set_xlabel("Barrier Energy / kbt")
    ax.set_ylabel(r"$m_{\mathrm{eff}} / m$")  # cspell: disable-line

    return fig, ax, line


def _plot_effective_mass_ratio() -> None:

    def _solve_effective_mass_path(
        system: PeriodicSystem1D,
        normalized_barrier_energies: np.ndarray,
        n_samples: np.ndarray,
        t_end: float,
        cutoff: float,
    ) -> Path:
        filename = f"{t_end}_{cutoff}_{t_end}_{hash_array((n_samples,))}_{t_end}_{hash_array((normalized_barrier_energies,))}_{hash(system)}.npz"
        return Path("examples/data") / filename

    @cached(_solve_effective_mass_path)
    def _effective_mass_simulation(
        system: PeriodicSystem1D,
        normalized_barrier_energies: np.ndarray,
        n_samples: np.ndarray,
        t_end: float,
        cutoff: float,
    ) -> np.ndarray:
        keys = jrandom.split(jrandom.PRNGKey(100), normalized_barrier_energies.size)
        simulated_effective_mass_ratio = np.zeros_like(normalized_barrier_energies)

        barrier_energy = normalized_barrier_energies * system.kbt

        with disabled_timing():
            for idx, _ in enumerate(
                tqdm(np.ndindex(barrier_energy.shape), total=barrier_energy.size)
            ):
                result = solve_ensemble_ballistic(
                    with_barrier_energy(
                        system,
                        barrier_energy=barrier_energy[idx],
                    ).as_canonical(),
                    TimeSpan(
                        t_end=t_end,
                        n_steps=1000,
                    ),
                    minimum_energy=barrier_energy[idx],
                    n_samples=n_samples[idx],
                    _key=keys[idx],
                )

                simulated_effective_mass_ratio[idx] = (
                    get_effective_mass(
                        result, cutoff=cutoff, filter_timescale=1 / system.gamma
                    ).item()
                    / system.m
                )

            return simulated_effective_mass_ratio

    def _effective_mass_exact(
        system: PeriodicSystem1D, normalized_barrier_energies_fine: np.ndarray
    ) -> np.ndarray:
        exact_effective_mass_ratio = np.zeros_like(normalized_barrier_energies_fine)
        for idx, _ in enumerate(
            tqdm(
                np.ndindex(normalized_barrier_energies_fine.shape),
                total=normalized_barrier_energies_fine.size,
            )
        ):
            system = with_barrier_energy(
                system, normalized_barrier_energies_fine[idx] * system.kbt
            )
            exact_effective_mass_ratio[idx] = _get_exact_effective_mass_ratio(system)
        return exact_effective_mass_ratio

    normalized_barrier_energies = np.logspace(-3, 1, 10)
    simulated_effective_mass_ratio = _effective_mass_simulation(
        system=SODIUM_COPPER_SYSTEM_1D,
        normalized_barrier_energies=normalized_barrier_energies,
        t_end=40e-12,
        cutoff=10e-12,
        n_samples=(1000 / np.sqrt(normalized_barrier_energies)).astype(int),
    )

    fig, ax = get_fancy_figure()
    _, ax, simulation_line = plot_effective_mass_ratio_against_energy(
        barrier_energy=normalized_barrier_energies,
        mass_ratio=simulated_effective_mass_ratio,
        ax=ax,
    )
    simulation_line.set_label("simulation")
    simulation_line.set_marker("x")
    simulation_line.set_linestyle("")
    simulation_line.set_color(CAM_CHERRY.dark)

    normalized_barrier_energies_fine = np.logspace(
        np.log10(normalized_barrier_energies[0]),
        np.log10(normalized_barrier_energies[-1]),
        1000,
    )
    exact_effective_mass_ratio = _effective_mass_exact(
        system=SODIUM_COPPER_SYSTEM_1D,
        normalized_barrier_energies_fine=normalized_barrier_energies_fine,
    )

    _, ax, exact_line = plot_effective_mass_ratio_against_energy(
        barrier_energy=normalized_barrier_energies_fine,
        mass_ratio=exact_effective_mass_ratio,
        ax=ax,
    )
    exact_line.set_label("exact")
    exact_line.set_color(CAM_BLUE.dark)

    _, ax, asymptote_line = _add_effective_mass_low_u0_asymptote(
        u0_min=normalized_barrier_energies_fine[0], u0_max=1, ax=ax
    )
    asymptote_line.set_label("asymptote")
    asymptote_line.set_color(CAM_BLUE.warm)

    ax.legend(handles=[simulation_line, exact_line, asymptote_line])

    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1e1)
    fig.savefig(
        "examples/ballistic_langevin/1d_periodic.effective_mass_trend.pdf",
        dpi=1000,
    )

    fig, ax = get_fancy_figure()


if __name__ == "__main__":
    _plot_effective_mass_ratio()
