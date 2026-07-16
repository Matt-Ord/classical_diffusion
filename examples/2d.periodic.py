from typing import TYPE_CHECKING

import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    plot_2d_trajectory,
    plot_potential_2d,
)
from classical_diffusion.solve import (
    InitialConditions,
    PeriodicParameters2D,
    PhysicalParameters,
    TimeSpan,
    solve_ensemble,
)
from classical_diffusion.theme import CAM_BLUE, get_fancy_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure


def plot_periodic_potential_2d(
    params: PeriodicParameters2D,
    *,
    n_points: tuple[int, int] = (100, 100),
    ax: Axes | None = None,
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot the periodic potential in 2D."""
    # TODO: fix up  PeriodicParameters2D to make lattice directions explicit
    return plot_potential_2d(
        params,
        (0, 0),
        (
            2 * params.characteristic_values.length,
            2 * params.characteristic_values.length,
        ),
        n_points=n_points,
        ax=ax,
    )


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
