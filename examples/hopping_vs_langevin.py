import jax
import numpy as np

from classical_diffusion.analysis import plot_isf
from classical_diffusion.hopping import Lattice1D, solve_ensemble
from classical_diffusion.hopping._system import (
    get_kramers_lattice_harmonic,
)
from classical_diffusion.langevin import PeriodicSystem1D, solve_overdamped_ensemble
from classical_diffusion.langevin._system import DoubleHarmonicSystem
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan


def _plot_hop_vs_periodic_langevin_isf(
    lattice: Lattice1D,
    system: PeriodicSystem1D,
    time_span: TimeSpan = TimeSpan(t_end=400, n_steps=4000),
    initial_condition: np.ndarray = np.full((4000, 1), 0.0),
    xlim: float = 5,
) -> None:

    hopping_results = solve_ensemble(
        system=lattice,
        time_span=time_span,
        initial_condition=initial_condition,
        key=jax.random.PRNGKey(seed=100),
    )

    # Now Langevin

    langevin_result = solve_overdamped_ensemble(
        system,
        time_span,
        (initial_condition, np.full(initial_condition.shape, 0.0)),
        _key=jax.random.PRNGKey(seed=100),
    )

    fig, ax = get_fancy_figure()
    delta_k = (0.5 * 2 * np.pi / system.delta_x,)

    _, ax, line_0, _ = plot_isf(result=hopping_results, delta_k=delta_k, ax=ax)
    line_0.set_label("Hopping model")

    _, _, line, _ = plot_isf(
        result=langevin_result, ax=ax, delta_k=delta_k, pairwise=True, measure="real"
    )
    line.set_label("Overdamped Langevin")

    ax.set_xlim(0, right=40)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"Hopping vs Langevin for barrier energy = {system.barrier_energy}")
    fig.savefig(f"./examples/1d_comparison_{system.barrier_energy}.isf.pdf")
    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig(f"./examples/1d_comparison_{system.barrier_energy}.isf.log.pdf")


def _plot_hop_vs_harmonic_langevin_isf(
    lattice: Lattice1D,
    system: DoubleHarmonicSystem,
    time_span: TimeSpan = TimeSpan(t_end=400, n_steps=4000),
    initial_condition: np.ndarray = np.full((4000, 1), 0.0),
    xlim: float = 5,
) -> None:

    hopping_results = solve_ensemble(
        system=lattice,
        time_span=time_span,
        initial_condition=initial_condition,
        key=jax.random.PRNGKey(seed=100),
    )

    # Now Langevin

    langevin_result = solve_overdamped_ensemble(
        system,
        time_span,
        (initial_condition, np.full(initial_condition.shape, 0.0)),
        _key=jax.random.PRNGKey(seed=100),
    )

    fig, ax = get_fancy_figure()
    delta_k = (0.5 * 2 * np.pi / system.delta_x,)

    _, ax, line_0, _ = plot_isf(result=hopping_results, delta_k=delta_k, ax=ax)
    line_0.set_label("Hopping model")

    _, _, line, _ = plot_isf(
        result=langevin_result, ax=ax, delta_k=delta_k, pairwise=True, measure="real"
    )
    line.set_label("Overdamped Langevin")

    ax.set_xlim(0, right=40)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"Hopping vs Langevin for barrier energy = {system.barrier_energy}")
    fig.savefig(f"./examples/1d_comparison_{system.barrier_energy}.isf.pdf")
    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig(f"./examples/1d_comparison_{system.barrier_energy}.isf.log.pdf")


def _kramers_test() -> None:

    initial_position: np.ndarray = np.full((50, 1), 0.0)
    time_span = TimeSpan(t_end=100, n_steps=200)

    system = DoubleHarmonicSystem(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        barrier_energy=3,
    )

    lattice = get_kramers_lattice_harmonic(system)
    print(f"hop time = {lattice.hop_time}")

    _plot_hop_vs_harmonic_langevin_isf(
        lattice=lattice,
        system=system,
        time_span=time_span,
        initial_condition=initial_position,
        xlim=10 / system.gamma,
    )


def benchmark(
    system: PeriodicSystem1D,
) -> None:

    time_span = TimeSpan(t_end=4, n_steps=400)
    initial_condition: np.ndarray = np.full((2, 1), 0.0)

    solve_overdamped_ensemble.call_uncached(
        system,
        time_span,
        (initial_condition, np.full(initial_condition.shape, 0.0)),
        _key=jax.random.PRNGKey(seed=100),
    )

    solve_overdamped_ensemble.call_uncached(
        system,
        time_span,
        (initial_condition, np.full(initial_condition.shape, 0.0)),
        _key=jax.random.PRNGKey(seed=100),
    )

    initial_condition: np.ndarray = np.full((50, 1), 0.0)
    time_span = TimeSpan(t_end=400, n_steps=100)

    solve_overdamped_ensemble.call_uncached(
        system,
        time_span,
        (initial_condition, np.full(initial_condition.shape, 0.0)),
        _key=jax.random.PRNGKey(seed=100),
    )

    solve_overdamped_ensemble.call_uncached(
        system,
        time_span,
        (initial_condition, np.full(initial_condition.shape, 0.0)),
        _key=jax.random.PRNGKey(seed=100),
    )

    time_span = TimeSpan(t_end=400, n_steps=4000)
    initial_condition: np.ndarray = np.full((2, 1), 0.0)

    solve_overdamped_ensemble.call_uncached(
        system,
        time_span,
        (initial_condition, np.full(initial_condition.shape, 0.0)),
        _key=jax.random.PRNGKey(seed=100),
    )

    solve_overdamped_ensemble.call_uncached(
        system,
        time_span,
        (initial_condition, np.full(initial_condition.shape, 0.0)),
        _key=jax.random.PRNGKey(seed=100),
    )


if __name__ == "__main__":
    print("running")
    # benchmark(
    #     system=PeriodicSystem1D(
    #         gamma=0.1,
    #         temperature=0.5,
    #         m=1.0,
    #         delta_x=5,
    #         barrier_energy=3,
    #     )
    # )
    _kramers_test()
