import jax.random as jrandom
import numpy as np

from classical_diffusion.langevin import (
    TimeSpan,
    plot_isf,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import HarmonicSystem, plot_exact_harmonic_isf


def _plot_harmonic_isf() -> None:
    key = jrandom.PRNGKey(100)

    system = HarmonicSystem(gamma=0.1, temperature=0.5, m=1.0, omega=1.0)

    result = solve_ensemble(
        system,
        TimeSpan(
            t0=0,
            t1=50 / system.gamma,
            dt=0.01 / system.gamma,
            dt_step=0.01 / system.gamma,
        ),
        (np.full((200, 1), 0.0), np.full((200, 1), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()

    delta_k = (2 * np.pi / 5,)
    _, ax, line_simulated, _ = plot_isf(
        result=result,
        ax=ax,
        delta_k=delta_k,
    )
    line_simulated.set_label("simulation")

    ax.axhline(
        y=np.exp(
            -(delta_k[0] ** 2) * system.temperature / (system.m * system.omega**2)
        ),
        linestyle=":",
        color="black",
    )

    _, ax, line_exact = plot_exact_harmonic_isf(system, delta_k, result.times, ax=ax)
    line_exact.set_label("exact")
    ax.legend(
        loc="upper right",
        handles=[line_simulated, line_exact],
        labels=["Simulation", "Exact"],
    )
    ax.set_xlim(0, 2 / system.gamma)
    ax.set_ylim(0, 1)
    fig.savefig("./examples/harmonic_isf.isf.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    _plot_harmonic_isf()
