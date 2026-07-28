import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import plot_isf, plot_x_evolution
from classical_diffusion.hopping import (
    Lattice1D,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan


def _plot_1d_hopping_isf() -> None:

    lattice = Lattice1D(lattice_spacing=5, hop_time=15)
    total_time = 400
    results = solve_ensemble(
        lattice=lattice,
        time_span=TimeSpan(t_end=total_time, n_steps=total_time),
        initial_condition=np.full((100, 1), 0.0),
        key=jrandom.PRNGKey(seed=100),
    )

    fig, ax = get_fancy_figure()

    _, _, _ = plot_x_evolution(
        result=results,
        ax=ax,
    )
    print("saving trajectory plot")
    fig.savefig("./examples/1d_lattice.trajectory.pdf")

    fig, ax = get_fancy_figure()

    delta_k = (0.5 * 2 * np.pi / lattice.lattice_spacing,)
    _, ax, line_0, _ = plot_isf(
        result=results,
        delta_k=delta_k,
        ax=ax,
    )
    line_0.set_label("Hopping simulation")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1)

    fig.savefig("./examples/1d_lattice.isf.pdf")


if __name__ == "__main__":
    _plot_1d_hopping_isf()
