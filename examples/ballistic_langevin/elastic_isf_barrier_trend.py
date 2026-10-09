import dataclasses

import numpy as np

from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    PeriodicSystem1D,
    get_exact_elastic_effective_mass,
    get_exact_elastic_free_probability,
    get_exact_elastic_isf,
    get_exact_flat_ballistic_isf,
    get_gamma_elastic_isf,
)
from classical_diffusion.plot import CAM_BLUE, CAM_CHERRY, get_three_panel_figure


def _with_barrier_energy(
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


def _plot_elastic_isf_barrier_trend() -> None:
    """Plot the dynamic part of the elastic ISF, (ISF - (1 - A)) / A, against barrier energy."""
    fig, axes = get_three_panel_figure()
    delta_k = (2 * np.pi / SODIUM_COPPER_SYSTEM_1D.delta_x * 0.2,)

    for ax, barrier_energy_ratio in zip(axes, (0.5, 2.0, 5.0), strict=True):
        system = _with_barrier_energy(
            SODIUM_COPPER_SYSTEM_1D,
            barrier_energy=barrier_energy_ratio * SODIUM_COPPER_SYSTEM_1D.kbt,
        )
        # Time in units of the free particle decay, delta_k t sqrt(kT / m)
        scaled_times = np.linspace(0, 8, 400)
        times = scaled_times / (delta_k[0] * np.sqrt(system.kbt / system.m))

        free_probability = get_exact_elastic_free_probability(system)

        def _dynamic(isf: np.ndarray, p: float = free_probability) -> np.ndarray:
            return (isf - (1 - p)) / p

        (exact_line,) = ax.plot(
            scaled_times, _dynamic(get_exact_elastic_isf(system, delta_k, times))
        )
        exact_line.set_label("exact")
        exact_line.set_color(CAM_BLUE.dark)

        effective_mass = get_exact_elastic_effective_mass(system)
        (effective_mass_line,) = ax.plot(
            scaled_times,
            get_exact_flat_ballistic_isf(
                dataclasses.replace(system.as_canonical(), m=effective_mass),
                delta_k,
                times,
            ),
        )
        effective_mass_line.set_label("effective mass")
        effective_mass_line.set_linestyle(":")
        effective_mass_line.set_color(CAM_CHERRY.dark)

        (gamma_line,) = ax.plot(
            scaled_times,
            _dynamic(get_gamma_elastic_isf(system, delta_k, times)),
        )
        gamma_line.set_label("gamma")
        gamma_line.set_linestyle("-.")
        gamma_line.set_color(CAM_BLUE.warm)

        ax.set_title(f"$E_b / k_B T = {barrier_energy_ratio}$")
        ax.set_xlabel(r"$\Delta k t \sqrt{k_B T / m}$")
        ax.set_xlim(0, 8)

    axes[0].set_ylabel("Dynamic ISF")
    axes[0].legend(handles=[exact_line, effective_mass_line, gamma_line])
    fig.savefig("examples/ballistic_langevin/elastic_isf_barrier_trend.pdf")


if __name__ == "__main__":
    _plot_elastic_isf_barrier_trend()
