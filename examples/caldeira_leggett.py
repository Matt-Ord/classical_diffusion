from pathlib import Path

import numpy as np

from classical_diffusion.analysis import get_isf, plot_isf
from classical_diffusion.caldeira_leggett import (
    get_diffusion_constant,
    get_steady_state,
    solve_isf,
)
from classical_diffusion.langevin import LITHIUM_COPPER_SYSTEM_1D, solve_ensemble
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


def _print_diffusion_constant() -> None:
    system = LITHIUM_COPPER_SYSTEM_1D
    # The diffusion constant is the limit of the dephasing rate over delta_k^2
    # as delta_k -> 0, found without simulating an ISF
    diffusion = get_diffusion_constant(system)
    print(f"D = {diffusion:.3e} m^2 / s")

    # A looser tolerance on the friction uses fewer periods of the potential
    # in the simulation cell, and a lower energy fewer bands
    diffusion = get_diffusion_constant(system, friction_tol=2e-2, max_energy=10.0)
    print(f"D = {diffusion:.3e} m^2 / s, in a smaller basis")


if __name__ == "__main__":
    with cache_base_path(Path("./examples/data")):
        # Every ISF, and the diffusion constant, starts from the steady state.
        # They load it from the cache, but only this call saves it
        get_steady_state.call_cached(LITHIUM_COPPER_SYSTEM_1D)
        _plot_isf()
        _print_diffusion_constant()
