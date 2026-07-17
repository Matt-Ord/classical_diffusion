import jax.random as jrandom
import numpy as np

from classical_diffusion.langevin import (
    TimeSpan,
    plot_isf,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import (
    PeriodicSystem1D,
    plot_periodic_potential_1d,
)


def _plot_periodic_system() -> None:
    system = PeriodicSystem1D(
        gamma=0.1, temperature=1.0, m=1.0, delta_x=5.0, barrier_energy=1.5
    )
    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_1d(system, ax=ax)
    fig.savefig("examples/1d_system.potential.pdf")


def _plot_1d_periodic_isf() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=0.1, temperature=0.5, m=1.0, delta_x=5, barrier_energy=1.5
    )

    result = solve_ensemble(
        system,
        TimeSpan(
            t0=0,
            t1=50 / system.gamma,
            dt=0.01 / system.gamma,
            dt_step=0.05 / system.gamma,
        ),
        (np.full((2000, 1), 0.0), np.full((2000, 1), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()

    delta_k = (0.5 * 2 * np.pi / system.delta_x,)
    _, ax, line_simulated, _ = plot_isf(
        result=result,
        ax=ax,
        delta_k=delta_k,
    )
    line_simulated.set_label("simulation")

    ax.set_xlim(0, 4 / system.gamma)
    ax.set_ylim(0, 1)
    fig.savefig("./examples/1d_system.isf.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    _plot_periodic_system()
    _plot_1d_periodic_isf()
