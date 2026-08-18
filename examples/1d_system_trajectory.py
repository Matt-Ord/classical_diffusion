import matplotlib.pyplot as plt

from classical_diffusion.analysis import plot_p_evolution, plot_x_evolution
from classical_diffusion.langevin import (
    PeriodicSystem1D,
    plot_energy,
    solve_ballistic_ensemble,
)
from classical_diffusion.simulation import TimeSpan

system = PeriodicSystem1D(
    gamma=0,
    temperature=110,
    m=3e-27,
    delta_x=3e-10,
    barrier_energy=1.6e-21,
)

normalized_system = system.with_normalized_units()


result = solve_ballistic_ensemble(
    normalized_system,
    TimeSpan(
        t_end=normalized_system.units.time_into(5e-12),
        n_steps=1000,
    ),
    n_samples=1,
)

fig, axes = plt.subplots(3, 1)
_, ax0, _ = plot_x_evolution(result.with_si_units(), ax=axes[0])

_, ax1, _ = plot_p_evolution(result.with_si_units(), ax=axes[1])

_, ax2 = plot_energy(result, ax=axes[2])
ax2.set_ylabel("Energy / kbt")


axes[0].set_xticks([])
axes[1].set_xticks([])

axes[0].set_xlabel(" ")
axes[1].set_xlabel(" ")

fig.savefig("examples/1d_system_trajectory.pdf", dpi=300, bbox_inches="tight")
