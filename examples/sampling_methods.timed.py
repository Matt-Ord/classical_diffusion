import jax.random as jrandom

from classical_diffusion.langevin import (
    TimeSpan,
    plot_x_evolution,
    solve_free_ballistic_ensemble,
    split_escaped_and_trapped,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import (
    PeriodicSystem1D,
    UnitSystem,
)
from classical_diffusion.util import timed

key = jrandom.PRNGKey(100)

system = PeriodicSystem1D(
    gamma=4e11,
    temperature=105,
    m=8e-27,
    delta_x=3e-10,
    barrier_energy=1.6e-21,
    units=UnitSystem(),
)

normalized_system = system.with_normalized_units()


@timed
def _run_simulation(system: PeriodicSystem1D) -> None:
    result = solve_free_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t0=0,
            t1=normalized_system.units.time_into(10e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        n_samples=10_000,
        _key=key,
        barrier_energy=normalized_system.barrier_energy,
    )

    _free_result, trapped_result = split_escaped_and_trapped(
        result, barrier_energy=system.barrier_energy
    )
    print(trapped_result.x_points.shape)
    fig, ax = get_fancy_figure()
    fig, ax, _ = plot_x_evolution(result, n_trajectories=1000, ax=ax)
    fig.savefig(
        "./examples/sampling_methods.trajectories.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _run_simulation(system)
