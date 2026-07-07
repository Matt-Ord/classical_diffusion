from dataclasses import dataclass, field

import jax
import jax.random as jrandom
import numpy as np
import sympy as sp

from classical_diffusion.Langevin import (
    SimulationParams,
    TimeSpan,
    _plot_p_evolution,
    _plot_x_evolution,
    solve_langevin,
)


@dataclass(kw_only=True)  # no frozen=True
class SHOParameters(SimulationParams):
    """Parameters for the harmonic Langevin equation."""

    omega: float
    potential: sp.Expr = field(init=False)

    def __post_init__(self) -> None:
        self.potential = 0.5 * self.omega**2 * sp.symbols("x") ** 2


@dataclass(frozen=True, kw_only=True)
class SHOSimulationResult:
    """Results of a simulation of the harmonic Langevin equation."""

    times: np.ndarray
    xs: np.ndarray
    ps: np.ndarray

    params: SimulationParams


def solve_single_trajectory_sho(
    params: SimulationParams, key: jax.Array
) -> SHOSimulationResult:
    """Solve a single trajectory of the harmonic Langevin equation."""
    sol = solve_langevin(params, key)
    assert sol.ys is not None, "solve failed to produce output"

    return SHOSimulationResult(
        times=np.array(sol.ts),
        xs=np.array(sol.ys[:, 0]),
        ps=np.array(sol.ys[:, 1]),
        params=params,
    )


# Parameters
m = 1.0
gamma = 0.5
sigma = 1.0

omega = 1.0


# Initial conditions
x0 = 1.0
p0 = 0.0
y0 = np.array([x0, p0])

# Time span
t0 = 0.0
t1 = 10.0
dt0 = 0.01

params = SHOParameters(
    gamma=gamma,
    sigma=sigma,
    m=m,
    omega=omega,
    time=TimeSpan(t0=t0, t1=t1, dt0=dt0),
    y0=y0,
)

# Solve a single trajectory and plot the results
key = jrandom.PRNGKey(100)  # Change seed to generate new result
result = solve_single_trajectory_sho(params=params, key=key)
fig_x, ax, line = _plot_x_evolution(result)
fig_p, ax, line = _plot_p_evolution(result)

fig_x.savefig("position_evolution.png", dpi=300, bbox_inches="tight")
fig_p.savefig("momentum_evolution.png", dpi=300, bbox_inches="tight")

print(
    "Figures successfully saved as 'position_evolution.png' and 'momentum_evolution.png'!"
)
