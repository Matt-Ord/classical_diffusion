import jax.random as jrandom
import numpy as np

from classical_diffusion.langevin import (
    InitialConditions,
    PeriodicParameters2D,
    PhysicalParameters,
    TimeSpan,
    plot_2d_trajectory,
    solve_ensemble,
)
from classical_diffusion.plot import CAM_BLUE, get_fancy_figure

if __name__ == "__main__":
    params = PeriodicParameters2D(
        physical_parameters=PhysicalParameters(m=1.0, temperature=1.0, gamma=1.0),
        time_span=TimeSpan(t0=0, t1=1000, dt=0.1, burn_in=0),
        delta_x=1.0,
        height=2.0,
        initial_conditions=InitialConditions(
            x0=np.array([[0.0, 0.0]]), p0=np.array([[0.0, 0.0]])
        ),
    )
    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_periodic_potential_2d(params=params, ax=ax)
    fig.savefig("examples/2d.periodic.potential.pdf")
    input("aa")
    key = jrandom.PRNGKey(100)
    result = solve_ensemble.load_or_call_cached(params=params, _key=key)

    fig, ax = get_fancy_figure()
    _, ax, line = plot_2d_trajectory(result=result, ax=ax)
    line.set_color(CAM_BLUE.warm)
    fig.savefig("examples/2d.periodic.trajectory.pdf")
