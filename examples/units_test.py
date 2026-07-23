import jax.random as jrandom
import numpy as np

from classical_diffusion.langevin import (
    TimeSpan,
    plot_x_evolution,
    solve_ensemble,
)
from classical_diffusion.system import (
    PeriodicSystem1D,
    UnitSystem,
)

system = PeriodicSystem1D(
    gamma=4e12,
    temperature=110,
    m=8e-27,
    delta_x=3e-10,
    barrier_energy=1.6e-22,
    units=UnitSystem(),
)
print(system.units)
# Convert to the system's units
print(system.with_system_units(system.simulation_units()))

# Convert back to regular units
print(system.with_system_units(UnitSystem()))

key = jrandom.PRNGKey(100)


result = solve_ensemble(
    system.with_system_units(system.simulation_units()),
    TimeSpan(
        t0=0,
        t1=40,
        dt=0.01,
        dt_step=0.01,
    ),
    (np.full((1, 1), 0.0), np.full((1, 1), 0.0)),
    _key=key,
)

fig, ax, _ = plot_x_evolution(result)
fig.savefig("examples/units_test.evolution.output.pdf")

fig, ax, _ = plot_x_evolution(result.with_units(UnitSystem()))
fig.savefig("examples/units_test.evolution.SI.pdf")
