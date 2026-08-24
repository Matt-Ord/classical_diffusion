import numpy as np
from scipy.constants import Boltzmann

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.hopping import (
    get_deterministic_probabilities,
    get_kramers_parameters_cosine,
    lattice_1d_from_kramers_parameters,
    plot_deterministic_isf,
)
from classical_diffusion.langevin import (
    KramersSystem1D,
    PeriodicSystem1D,
    plot_force_1d,
    plot_potential_1d,
    solve_many_overdamped,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan


def _plot_kramers_potential() -> None:
    system = PeriodicSystem1D(
        gamma=0.1, temperature=0.5 / Boltzmann, m=2.0, delta_x=5, barrier_energy=3
    )

    fig, ax = get_fancy_figure()
    ax_force = ax.twinx()
    _, _, line = plot_potential_1d(system, 0, 2 * system.delta_x, ax=ax)
    _, _, line = plot_force_1d(system, 0, 2 * system.delta_x, ax=ax_force)
    line.set_linestyle("--")

    kramers_system = KramersSystem1D(params=get_kramers_parameters_cosine(system))
    _, _, line = plot_potential_1d(kramers_system, 0, 2 * kramers_system.delta_x, ax=ax)
    _, _, line = plot_force_1d(
        kramers_system, 0, 2 * kramers_system.delta_x, ax=ax_force
    )
    line.set_linestyle("--")

    fig.savefig("examples/kramers_model.potential.pdf")


def _plot_kramers_periodic_comparison() -> None:
    system = PeriodicSystem1D(
        gamma=0.1, temperature=0.5 / Boltzmann, m=1.0, delta_x=5, barrier_energy=3
    )

    result = solve_many_overdamped(
        system,
        TimeSpan(t_end=40 / system.gamma, n_steps=4000),
        initial_conditions=(np.zeros(shape=(20, 1)), np.zeros(shape=(20, 1))),
    )

    fig, ax = get_fancy_figure()

    delta_k = (0.7 * 2 * np.pi / system.delta_x,)
    _, _, line_0, _ = plot_isf(result=result, ax=ax, delta_k=delta_k)
    line_0.set_label("cosine model")

    system = KramersSystem1D(params=get_kramers_parameters_cosine(system))
    result = solve_many_overdamped(
        system,
        TimeSpan(t_end=40 / system.gamma, n_steps=4000),
        initial_conditions=(np.zeros(shape=(20, 1)), np.zeros(shape=(20, 1))),
    )
    delta_k = (0.7 * 2 * np.pi / system.delta_x,)
    _, _, line_1, _ = plot_isf(result=result, ax=ax, delta_k=delta_k)
    line_1.set_label("kramers model")

    ax.set_xlim(0, 4 / system.gamma)
    ax.set_ylim(0, 1)

    ax.legend(handles=[line_0, line_1])
    fig.savefig("examples/kramers_model.vs_cosine.isf.pdf")


def _plot_kramers_hopping_comparison() -> None:
    system = PeriodicSystem1D(
        gamma=0.1, temperature=0.5 / Boltzmann, m=1.0, delta_x=5, barrier_energy=3
    )
    system = KramersSystem1D(params=get_kramers_parameters_cosine(system))

    result = solve_many_overdamped.call_uncached(
        system,
        TimeSpan(t_end=40 / system.gamma, n_steps=4000),
        initial_conditions=(np.zeros(shape=(20, 1)), np.zeros(shape=(20, 1))),
    )

    fig, ax = get_fancy_figure()

    delta_k = (0.7 * 2 * np.pi / system.delta_x,)
    _, _, line_0, _ = plot_isf(result=result, ax=ax, delta_k=delta_k)
    line_0.set_label("kramers model")
    ax.set_xlim(0, 4 / system.gamma)

    hop_system = lattice_1d_from_kramers_parameters(system.kramers_params)
    result = get_deterministic_probabilities(
        hop_system,
        (501,),
        TimeSpan(t_end=4 / system.gamma, n_steps=200),
    )
    _, _, line_1 = plot_deterministic_isf(result, ax=ax, delta_k=delta_k, amplitude=0.8)
    line_1.set_label("hopping model")

    ax.set_ylim(0, 1)

    ax.legend(handles=[line_0, line_1])
    fig.savefig("examples/kramers_model.vs_hopping.isf.pdf")


if __name__ == "__main__":
    # The kramers potential is a foundational model in rate theory.
    # Classically, in the overdamped regime, the rate of hopping
    # should depend only on the frequency of the top of the barrier.
    # Here, the two different model pes are compared
    _plot_kramers_potential()
    # The rates are approximately equal, and therefore the overdamped
    # ISF has approximately the same decay rate for both models (when
    # the choice of delta_k is suitably adjusted).
    _plot_kramers_periodic_comparison()
    # We can also compare the kramers model to a hopping model,
    # fro which there is an analytical equation which gives the hopping
    # rates.
    _plot_kramers_hopping_comparison()
