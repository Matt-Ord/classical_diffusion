import numpy as np

from classical_diffusion.analysis import plot_isf, plot_x_evolution
from classical_diffusion.hopping import (
    Lattice1D,
    get_ensemble_probabilities,
    plot_deterministic_isf,
    solve_ensemble,
)
from classical_diffusion.plot import (
    get_fancy_figure,
)
from classical_diffusion.simulation import TimeSpan


def _plot_1d_hopping_isf() -> None:

    results = solve_ensemble(
        system=Lattice1D(lattice_spacing=5, hop_time=15),
        time_span=TimeSpan(t_end=400, n_steps=4000),
        initial_condition=np.full((1, 1), 0.0),
    )

    fig, ax = get_fancy_figure()

    fig, ax, _ = plot_x_evolution(result=results, ax=ax)
    ax.set_xlim(0, results.times[-1])
    fig.savefig("./examples/1d_hopping.trajectory.pdf")

    results = solve_ensemble(
        system=Lattice1D(lattice_spacing=5, hop_time=15),
        time_span=TimeSpan(t_end=4000, n_steps=8000),
        initial_condition=np.full((4000, 1), 0.0),
    )

    fig, ax = get_fancy_figure()
    delta_k = (0.5 * 2 * np.pi / results.system.lattice_spacing,)
    _, ax, line_0, _ = plot_isf(result=results, delta_k=delta_k, ax=ax)
    line_0.set_label("Hopping simulation")
    ax.set_xlim(0, right=25)
    ax.set_ylim(0, 1)

    fig.savefig("./examples/1d_hopping.isf.pdf")

    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig("./examples/1d_hopping.isf.log.pdf")


def _plot_1d_deterministic_isf() -> None:

    system = Lattice1D(lattice_spacing=5, hop_time=15)

    result = get_ensemble_probabilities(
        system,
        (500001,),
        TimeSpan(t_end=25, n_steps=200),
        500,
    )

    fig, ax = get_fancy_figure()
    delta_k = 0.5 * 2 * np.pi / system.lattice_spacing
    _, ax, line_0 = plot_deterministic_isf(system, result, delta_k, ax=ax)

    line_0.set_label("Deterministic Hopping")
    ax.set_xlim(0, right=25)
    ax.set_ylim(0, 1)

    fig.savefig("./examples/1d_hopping.isf.deterministic.pdf")

    ax.set_yscale("symlog", linthresh=1e-4)
    fig.savefig("./examples/1d_hopping.isf.deterministic.log.pdf")


if __name__ == "__main__":
    _plot_1d_deterministic_isf()
    _plot_1d_hopping_isf()
