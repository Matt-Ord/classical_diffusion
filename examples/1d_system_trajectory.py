import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np

from classical_diffusion.langevin import (
    TimeSpan,
    plot_energy,
    plot_p_evolution,
    plot_x_evolution,
    solve_ensemble,
)
from classical_diffusion.system import (
    PeriodicSystem1D,
    UnitSystem,
)

key = jrandom.PRNGKey(100)

system = PeriodicSystem1D(
    gamma=4e12,
    temperature=110,
    m=8e-27,
    delta_x=3e-10,
    barrier_energy=1.6e-22,
    units=UnitSystem(),
)

result = solve_ensemble(
    system,
    TimeSpan(
        t0=0,
        t1=40,
        dt=0.01,
        dt_step=0.01,
    ),
    (np.full((1, 1), 0.0), np.full((1, 1), 0.0)),
    _key=key,
)

fig, axes = plt.subplots(3, 1)
_, ax0, _ = plot_x_evolution(result.with_units(UnitSystem()), ax=axes[0])

_, ax1, _ = plot_p_evolution(result.with_units(UnitSystem()), ax=axes[1])

_, ax2 = plot_energy(result.with_units(UnitSystem()), ax=axes[2])

axes[0].set_xticks([])
axes[1].set_xticks([])

axes[0].set_xlabel(" ")
axes[1].set_xlabel(" ")

fig.savefig("examples/1d_system_trajectory.pdf", dpi=300, bbox_inches="tight")
