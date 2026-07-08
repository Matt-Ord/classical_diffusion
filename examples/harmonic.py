from dataclasses import dataclass, field

import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from classical_diffusion.analysis import plot_isf, plot_p_evolution, plot_x_evolution
from classical_diffusion.Langevin import (
    SimulationParams,
    TimeSpan,
    solve_langevin,
)


@dataclass(kw_only=True)  # no frozen=True
class SHOParameters(SimulationParams):
    """Parameters for the harmonic Langevin equation."""

    omega: float
    potential: sp.Expr = field(init=False)

    def __post_init__(self) -> None:
        self.potential = 0.5 * self.omega**2 * sp.symbols("x") ** 2


# Parameters
m = 1.0
gamma = 1.0
sigma = 1.0

omega = 1.0


# Initial conditions
x0 = 1.0
p0 = 0.0
y0 = np.array([x0, p0])

# Time span
t0 = 0.0
t1 = 200.0
dt0 = 0.01

# Burn in?
burn_in_time = 0
burn_in_steps = int(burn_in_time / dt0)

n_times = round((t1 - t0) / dt0)
delta_x = np.zeros((1, n_times))

params = SHOParameters(
    gamma=gamma,
    sigma=sigma,
    m=m,
    omega=omega,
    time=TimeSpan(t0=t0, t1=t1, dt0=dt0),
    y0=y0,
    delta_x=delta_x,
)

key = jrandom.PRNGKey(100)  # Change seed to generate new result
result = solve_langevin(params=params, key=key)


x_equilibriated = result.xs[:, burn_in_steps:]

x_std = float(np.std(x_equilibriated))
delta_k = (1.0 / x_std,)


fig, (ax_x, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

_, ax_x, line_x = plot_x_evolution(result=result, ax=ax_x)
_, ax_p, line_p = plot_p_evolution(result=result, ax=ax_p)

ax_x.set_xlabel("")  # top plot doesn't need its own x-label, shares with bottom
fig.suptitle("Position and Momentum Evolution")
fig.tight_layout()
fig.savefig("trajectory_x_p_harmonic.png", dpi=300, bbox_inches="tight")

fig, ax = None, None

fig, _, _ = plot_isf(result=result, delta_k=delta_k, ax=ax)

fig.savefig("isf_harmonic", dpi=300, bbox_inches="tight")
