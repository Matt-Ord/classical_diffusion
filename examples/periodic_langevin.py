from dataclasses import dataclass, field

import jax
import jax.random as jrandom
import numpy as np
import sympy as sp

from classical_diffusion.Langevin import (
    SimulationParams,
    solve_langevin,
)


@dataclass(kw_only=True)
class PerParameters(SimulationParams):
    """Parameters for the harmonic Langevin equation."""

    V0: float
    k: float
    potential: sp.Expr = field(init=False)

    def __post_init__(self) -> None:
        self.potential = self.V0 * sp.cos(sp.symbols("x") * self.k)


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


def solve_single_trajectory_periodic(key, args) -> PeriodicSimulationResult:
    """Solve a single trajectory of the periodic Langevin equation."""
    sol = solve_langevin(time_args, y0, args, key)

    return PeriodicSimulationResult(
        times=np.array(sol.ts),
        xs=np.array(sol.ys[:, 0]),
        ps=np.array(sol.ys[:, 1]),
        V0=V0,
        k=k,
        gamma=gamma,
        sigma=sigma,
        m=m,
    )


# Parameters
m = 1.0
gamma = 0.5
sigma = 1.0

V0 = 2.0
k = 1.0
pot_args = (V0, k)

args = (m, gamma, sigma, pot_args, periodic_potential)

# Initial conditions
x0 = 1.0
p0 = 0.0
y0 = jnp.array([x0, p0])

# Time span
t0 = 0.0
t1 = 10.0
dt0 = 0.01
n_sav = int((t1 - t0) / dt0)
time_args = (t0, t1, dt0, n_sav)

# Solve a single trajectory and plot the results
key = jrandom.PRNGKey(42)  # Change seed to generate new result
result = solve_single_trajectory_periodic(key=key, args=args)
fig_x, ax, line = plot_x_evolution(result)
fig_p, ax, line = plot_p_evolution(result)

fig_x.savefig("position_evolution.png", dpi=300, bbox_inches="tight")
fig_p.savefig("momentum_evolution.png", dpi=300, bbox_inches="tight")

print(
    "Figures successfully saved as 'position_evolution.png' and 'momentum_evolution.png'!"
)
