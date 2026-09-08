from pathlib import Path
from typing import Any

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
from classical_diffusion.plot import CAM_BLUE, CAM_CHERRY, get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import (
    cache_base_path,
    cached,
    disabled_timing,
    hash_array,
)


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


def _get_single_exact_effective_mass_ratio(
    barrier_energy_ratio: float,
) -> float:
    u0 = barrier_energy_ratio

    def integrand_denominator(epsilon: float) -> float:
        return np.sqrt(epsilon) / ellipk(1 / epsilon) * np.exp(-u0 * epsilon)

    def integrand_partition(epsilon: float) -> float:
        return 1 / np.sqrt(epsilon) * ellipk(1 / epsilon) * np.exp(-u0 * epsilon)

    denominator_integral, _ = quad(integrand_denominator, 1, np.inf)
    partition_integral, _ = quad(integrand_partition, 1, np.inf)

    return 2 * partition_integral / (denominator_integral * u0 * np.pi**2)


def _get_exact_effective_mass_ratio(
    barrier_energy_ratio_fine: np.ndarray,
) -> np.ndarray[tuple[int], np.dtype[np.floating[Any]]]:

    return np.array(
        [_get_single_exact_effective_mass_ratio(m) for m in barrier_energy_ratio_fine],
    )


def _get_low_barrier_effective_mass_ratio(
    dimensionless_barrier_energy: float,
) -> float:
    """Calculate the low barrier effective mass asymptote."""
    return 1 - (4 / np.pi**1.5) * np.sqrt(dimensionless_barrier_energy)


def _solve_effective_mass_path(
    system: PeriodicSystem1D,
    barrier_energy_ratio: np.ndarray,
    n_samples: np.ndarray,
) -> Path:
    filename = f"effective_mass_{hash_array((n_samples,))}_{hash_array((barrier_energy_ratio,))}_{hash(system)}.npz"
    return Path(filename)


@cached(_solve_effective_mass_path)
def _get_simulated_effective_mass(
    system: PeriodicSystem1D,
    barrier_energy_ratio: np.ndarray,
    n_samples: np.ndarray,
) -> np.ndarray:
    keys = jrandom.split(jrandom.PRNGKey(100), barrier_energy_ratio.size)
    out = np.zeros_like(barrier_energy_ratio)

    barrier_energy = barrier_energy_ratio * system.kbt

    with disabled_timing():
        for idx, _ in enumerate(
            tqdm(np.ndindex(barrier_energy.shape), total=barrier_energy.size)
        ):
            result = solve_ensemble_ballistic.call_uncached(
                _with_barrier_energy(system, barrier_energy=barrier_energy[idx]),
                TimeSpan(t_end=8 / system.gamma, n_steps=1000),
                energy_range=(barrier_energy[idx], np.inf),
                n_samples=n_samples[idx],
                _key=keys[idx],
            )

            mass = get_effective_mass(result, filter_timescale=1 / system.gamma)
            out[idx] = mass.item() / system.m

        return out


def _plot_effective_mass_ratio() -> None:

    barrier_energy_ratio = np.logspace(-3, 1, 10)

    simulated_effective_mass_ratio = _get_simulated_effective_mass(
        system=SODIUM_COPPER_SYSTEM_1D,
        barrier_energy_ratio=barrier_energy_ratio,
        n_samples=(1000 / np.sqrt(barrier_energy_ratio)).astype(int),
    )

    fig, ax = get_fancy_figure()

    (simulation_line,) = ax.plot(barrier_energy_ratio, simulated_effective_mass_ratio)
    simulation_line.set_label("simulation")
    simulation_line.set_marker("x")
    simulation_line.set_linestyle("")
    simulation_line.set_color(CAM_CHERRY.dark)

    barrier_energy_ratio = np.logspace(
        np.log10(barrier_energy_ratio[0]),
        np.log10(barrier_energy_ratio[-1]),
        1000,
    )

    (exact_line,) = ax.plot(
        barrier_energy_ratio,
        _get_exact_effective_mass_ratio(barrier_energy_ratio),
    )

    exact_line.set_label("exact")
    exact_line.set_color(CAM_BLUE.dark)

    (asymptote_line,) = ax.plot(
        barrier_energy_ratio,
        _get_low_barrier_effective_mass_ratio(barrier_energy_ratio),
    )
    asymptote_line.set_label("asymptote")
    asymptote_line.set_color(CAM_BLUE.warm)

    ax.legend(handles=[simulation_line, exact_line, asymptote_line])

    ax.set_xscale("log")  # cspell: disable-line
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(0, None)
    fig.savefig(
        "examples/ballistic_langevin/effective_mass_trend.pdf",
    )


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _plot_effective_mass_ratio()
