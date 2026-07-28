import jax.random as jrandom

from classical_diffusion.langevin import (
    TimeSpan,
    get_effective_mass,
    plot_x_evolution,
    solve_ballistic_ensemble,
    solve_free_ballistic_ensemble,
)
from classical_diffusion.langevin._analysis import get_effective_mass_weighted
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import PeriodicSystem1D, UnitSystem, get_diffusion_time

key = jrandom.PRNGKey(4)

system = PeriodicSystem1D(
    gamma=4e11,
    temperature=100,
    m=8e-27,
    delta_x=3e-10,
    barrier_energy=1.6e-21,
    units=UnitSystem(),
)

normalized_system = system.with_normalized_units()

result_full = solve_ballistic_ensemble(
    normalized_system,
    TimeSpan(
        t0=0,
        t1=2e-12
        / get_diffusion_time(system=system, characteristic_length=system.delta_x),
        n_steps=1000,
    ),
    n_samples=30000,
    _key=key,
)

effective_mass_full = get_effective_mass(result_full)
print(effective_mass_full)

result_free = solve_free_ballistic_ensemble(
    normalized_system,
    TimeSpan(
        t0=0,
        t1=2e-12
        / get_diffusion_time(system=system, characteristic_length=system.delta_x),
        n_steps=1000,
    ),
    n_samples=20000,
    _key=key,
    barrier_energy=normalized_system.barrier_energy,
)

fig, ax = get_fancy_figure()
_, ax, _ = plot_x_evolution(result_free, n_trajectories=1000, ax=ax)
fig.savefig(
    "./examples/test.effective_mass.free.trajectories.only.pdf",
    dpi=300,
    bbox_inches="tight",
)

effective_mass_free = get_effective_mass_weighted(result_free)
print(effective_mass_free)
