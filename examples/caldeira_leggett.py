# cspell: ignore mathrm xscale
from pathlib import Path

import jax
import numpy as np

from classical_diffusion.analysis import get_isf, plot_isf
from classical_diffusion.caldeira_leggett import get_diffusion_constant, solve_isf
from classical_diffusion.hopping import get_lifson_jackson_diffusion
from classical_diffusion.langevin import (
    HYDROGEN_COPPER_SYSTEM_1D,
    LITHIUM_COPPER_SYSTEM_1D,
    CosineSystem1D,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import cache_base_path


def _plot_isf() -> None:
    system = LITHIUM_COPPER_SYSTEM_1D
    delta_k = (0.25 * 2 * np.pi / system.delta_x,)
    t_end = 10 / system.gamma

    fig, ax = get_fancy_figure()

    # The quantum ISF is deterministic, and has no error
    quantum = solve_isf(system, delta_k, np.linspace(0, t_end, 201))
    _, ax, line_0, _ = plot_isf(quantum, ax=ax, measure="real")
    line_0.set_label("Caldeira-Leggett")

    classical = solve_ensemble(
        system, TimeSpan(t_end=t_end, n_steps=4000), n_samples=1000
    )
    _, ax, line_1, _ = plot_isf(get_isf(classical, delta_k), ax=ax, measure="real")
    line_1.set_label("Langevin")

    ax.set_xlim(0, t_end)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1])
    fig.savefig("examples/caldeira_leggett.isf.pdf")


def _plot_diffusion_against_mass() -> None:
    base = HYDROGEN_COPPER_SYSTEM_1D
    # From hydrogen to about the mass of the heaviest adsorbates, such as xenon
    masses = base.m * np.geomspace(1, 100, 9)

    ratios = []
    for m in masses:
        # Only the mass changes: the potential, friction and temperature are those of H
        system = CosineSystem1D(
            gamma=base.gamma,
            temperature=base.temperature,
            m=m,
            delta_x=base.delta_x,
            barrier_energy=base.barrier_energy,
        )
        diffusion = get_diffusion_constant(system)
        ratios.append(
            diffusion / get_lifson_jackson_diffusion(system, delta_x=system.delta_x)
        )

    fig, ax = get_fancy_figure()
    (line,) = ax.plot(masses / base.m, ratios, marker="o")
    line.set_label("Caldeira-Leggett")
    ax.axhline(1, color="black", linestyle="--", label="Lifson-Jackson")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mass / mass of H")
    ax.set_ylabel(r"$D / D_\mathrm{LJ}$")
    ax.legend()
    fig.savefig("examples/caldeira_leggett.diffusion_mass.pdf")


if __name__ == "__main__":
    # The tolerances of the master equation need 64 bit floats
    jax.config.update("jax_enable_x64", True)  # ruff: ignore[boolean-positional-value-in-call]
    with cache_base_path(Path("./examples/data")):
        _plot_isf()
        _plot_diffusion_against_mass()
