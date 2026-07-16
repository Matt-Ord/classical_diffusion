from dataclasses import dataclass

import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    get_fancy_figure,
    plot_2d_trajectory,
    plot_potential,
)
from classical_diffusion.solve import (
    InitialConditions,
    PeriodicParams2D,
    PhysicalParams,
    TimeSpan,
    solve_ensemble,
)


@dataclass(frozen=True, kw_only=True)
class CamColor:
    """A class to hold CAM color palettes."""

    light: str
    warm: str
    base: str
    dark: str


CAM_BLUE = CamColor(
    light="#D1F9F1",
    warm="#00BDB6",
    base="#8EE8D8",
    dark="#133844",
)

CAM_SLATE_4 = "#232830"

params = PeriodicParams2D(
    physical_parameters=PhysicalParams(m=1.0, temp=1.0, gamma=1.0),
    time_span=TimeSpan(t0=0, t1=1000, dt=0.1, burn_in=0),
    delta_x=1.0,
    height=2.0,
    initial_conditions=InitialConditions(
        x0=np.array([[0.0, 0.0]]), p0=np.array([[0.0, 0.0]])
    ),
)

key = jrandom.PRNGKey(100)
result = solve_ensemble.load_or_call_cached(params=params, _key=key)

fig, ax = get_fancy_figure(fig_size=(6, 4))
_, ax, line = plot_2d_trajectory(result=result, ax=ax)
line.set_color(CAM_BLUE.warm)
fig.savefig("/workspaces/classical_diffusion/examples/2d.periodic.trajectory.pdf")

fig, ax = get_fancy_figure()
plot_potential(params=params, ax=ax)
fig.savefig("/workspaces/classical_diffusion/examples/2d.periodic.potential.pdf")
