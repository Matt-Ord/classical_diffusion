# cspell: ignore mathrm xscale yscale ydata
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np
from scipy.constants import Boltzmann, hbar

from classical_diffusion.analysis import get_isf, plot_isf
from classical_diffusion.caldeira_leggett import (
    get_diffusion_constant,
    solve_isf,
    solve_rms,
)
from classical_diffusion.langevin import (
    HYDROGEN_COPPER_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_1D,
    CosineSystem1D,
    FixedStep,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import unit_with_trajectory_scale
from classical_diffusion.util import cache_base_path

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

# Every system has the same temperature and friction, and a barrier of
# 1.5 k_B T, low enough that the classical particle hops often, so that a few
# minutes of trajectories give its ISF and D to a few percent. Only the mass
# differs. H and Na share the lattice, the spacing of hollows on Cu(111)
_TEMPERATURE = 150.0
_FRICTION = 1e12
_BARRIER = 1.5
"""In units of k_B T."""
# Langevin steps of 0.05 in its natural units resolve the vibrations of H and
# Na: steps of 0.1 change D by less than its 2% noise
_STEP_CONFIG = FixedStep(step_size=0.05)
_CLASSICAL_FACTOR = 300
"""The mass, over that of H, from which D is classical to within a few percent."""


def _get_system(mass: float) -> CosineSystem1D:
    """Get the system of a mass, in units of the lattice spacing and the friction time 1 / gamma."""
    system = CosineSystem1D(
        gamma=_FRICTION,
        temperature=_TEMPERATURE,
        m=mass,
        delta_x=HYDROGEN_COPPER_SYSTEM_1D.delta_x,
        barrier_energy=_BARRIER * Boltzmann * _TEMPERATURE,
    )
    return system.with_units(
        unit_with_trajectory_scale(
            system.units, length=system.delta_x, time=1 / system.gamma
        )
    )


def _get_pairwise_msd(x: np.ndarray, lags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get <(x(t + tau) - x(t))^2> of each trajectory at the lags (in samples), averaged over every origin t, and its standard error."""
    per_sample = np.stack(
        [np.zeros(len(x))]
        + [np.mean((x[:, lag:] - x[:, :-lag]) ** 2, axis=1) for lag in lags[1:]],
        axis=1,
    )
    return per_sample.mean(axis=0), per_sample.std(axis=0) / np.sqrt(len(x))


def _finish(ax: Axes, lines: list[Line2D], t_end: float) -> None:
    """Mark the friction time, and label the time axis in units of it."""
    friction = ax.axvline(1, color="grey", linestyle=":", label=r"$1 / \gamma$")
    ax.set_xlim(0, t_end)
    ax.set_xlabel(r"Time $t \gamma$")
    ax.legend(handles=[*lines, friction])


def _plot_isf_and_msd(t_end: float, name: str, mass: float) -> None:
    """Plot the ISF and mean square displacement of the quantum and classical particle, to t_end in units of 1 / gamma."""
    system = _get_system(mass)
    times = np.linspace(0, t_end, 201)
    # The trajectories run to twice t_end, so that every lag up to t_end
    # averages many pairs of times
    classical = solve_ensemble(
        system,
        TimeSpan(t_end=2 * t_end, n_steps=2000),
        n_samples=1000,
        step_config=_STEP_CONFIG,
    )

    # The quantum ISF and mean square displacement are deterministic, and
    # have no error
    delta_k = (0.4 * 2 * np.pi / system.delta_x,)
    fig, ax = get_fancy_figure()
    _, ax, quantum, _ = plot_isf(
        solve_isf(system, delta_k, times), ax=ax, measure="real"
    )
    _, ax, langevin, _ = plot_isf(get_isf(classical, delta_k), ax=ax, measure="real")
    quantum.set_label("Caldeira-Leggett")
    langevin.set_label("Langevin")
    # The quantum ISF of H dips below zero, as it is a sum of oscillating modes
    ax.set_ylim(
        min(0.0, 1.1 * float(np.asarray(quantum.get_ydata(), dtype=float).min())), 1
    )
    _finish(ax, [quantum, langevin], t_end)
    fig.savefig(f"examples/caldeira_leggett.isf_{name}.pdf")

    fig, ax = get_fancy_figure()
    (quantum,) = ax.plot(times, solve_rms(system, times).mean_square)
    quantum.set_label("Caldeira-Leggett")
    lags = np.arange(0, len(classical.times) // 2 + 1, 5)
    msd, error = _get_pairwise_msd(np.asarray(classical.x_points)[:, 0, :], lags)
    lag_times = lags * (classical.times[1] - classical.times[0])
    (langevin,) = ax.plot(lag_times, msd)
    langevin.set_label("Langevin")
    fill = ax.fill_between(lag_times, msd - error, msd + error, alpha=0.3)
    fill.set_color(langevin.get_color())
    ax.set_ylim(0, None)
    ax.set_ylabel(r"$\langle (x(t) - x(0))^2 \rangle / \Delta x^2$")
    _finish(ax, [quantum, langevin], t_end)
    fig.savefig(f"examples/caldeira_leggett.msd_{name}.pdf")


def _plot_tunnelling_diffusion() -> None:
    """Plot D against the mass, from H up, in a family of systems which are the same classically.

    The barrier of H on Cu(111) at 100 K, 15.8 k_B T, is below the crossover
    temperature of H, hbar omega_b / 2 pi k_B = 132 K, where tunnelling
    through the barrier is faster than hopping over it. Classically, D =
    a^2 omega_0 f(beta E_b, gamma / omega_0), with omega_0 the frequency of
    the well, so with the friction scaled as 1 / sqrt(m) every mass has the
    same D / omega_0. The quantum D depends on the mass only through
    hbar omega_0 / k_B T, and its classical value is its limit at large mass,
    which differs from the classical one by order hbar^2. A Langevin
    simulation cannot reach the barrier, with one hop in about 10^5 ps, and
    the rate theories of Kramers and Lifson and Jackson do not hold at
    gamma / omega_0 = 0.01, where a hop crosses many cells (for H at
    3 k_B T they are 70 times low and 5 times high). At 3 and 4.5 k_B T this
    limit is within 2% of Langevin.
    """
    hydrogen = HYDROGEN_COPPER_SYSTEM_1D
    temperature = 100.0
    # Up to 100 times the mass of H on the plot, and its classical limit from
    # 300 and 1000 times, linear in (hbar omega_0 / k_B T)^2 ~ 1 / m
    factors = np.concatenate([np.geomspace(1, 100, 11), [_CLASSICAL_FACTOR, 1000]])
    # D / omega_0, in units of that of H
    scaled = np.array(
        [
            get_diffusion_constant(
                CosineSystem1D(
                    gamma=_FRICTION / np.sqrt(factor),
                    temperature=temperature,
                    m=factor * hydrogen.m,
                    delta_x=hydrogen.delta_x,
                    barrier_energy=hydrogen.barrier_energy,
                )
            )
            * np.sqrt(factor)
            for factor in factors
        ]
    )
    heavy = factors >= _CLASSICAL_FACTOR
    classical = np.polyfit(1 / factors[heavy], scaled[heavy], 1)[-1]

    omega_barrier = (
        2
        * np.pi
        / hydrogen.delta_x
        * np.sqrt(hydrogen.barrier_energy / (2 * hydrogen.m))
    )
    # T_c = T where hbar omega_b / k_B T = 2 pi, and omega_b ~ 1 / sqrt(m)
    crossover = (hbar * omega_barrier / (2 * np.pi * Boltzmann * temperature)) ** 2

    fig, ax = get_fancy_figure()
    (line,) = ax.plot(factors[~heavy], scaled[~heavy] / classical, marker="o")
    line.set_label("Caldeira-Leggett")
    ax.axhline(1, color="black", linestyle="--", label="classical")
    ax.axvline(crossover, color="grey", linestyle=":", label=r"$T_c = T$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Mass / mass of H, at $\gamma \propto m^{-1/2}$")
    ax.set_ylabel(r"$D_\mathrm{quantum} / D_\mathrm{classical}$")
    ax.legend()
    fig.savefig("examples/caldeira_leggett.diffusion_tunnelling.pdf")


if __name__ == "__main__":
    # The tolerances of the master equation need 64 bit floats
    jax.config.update("jax_enable_x64", True)  # ruff: ignore[boolean-positional-value-in-call]
    with cache_base_path(Path("./examples/data")):
        # H is light enough that its ISF differs from the classical one
        _plot_isf_and_msd(3, "hydrogen", HYDROGEN_COPPER_SYSTEM_1D.m)
        # Na is heavy enough to be classical
        _plot_isf_and_msd(10, "sodium", SODIUM_COPPER_SYSTEM_1D.m)
        _plot_tunnelling_diffusion()
