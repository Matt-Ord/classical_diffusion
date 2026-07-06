from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jrandom

from classical_diffusion.Langevin import (
    plot_p_evolution,
    plot_x_evolution,
    solve_langevin,
)


@dataclass(frozen=True, kw_only=True)
class SHOSimulationResult:
    """Results of a simulation of the harmonic Langevin equation."""

    times: jnp.ndarray
    xs: jnp.ndarray
    ps: jnp.ndarray
    omega: float
    gamma: float
    sigma: float
    m: float


def sho_potential(x, pot_args):
    omega = pot_args[0]
    return 0.5 * omega**2 * x**2


def solve_single_trajectory_sho(key) -> SHOSimulationResult:
    """Solve a single trajectory of the harmonic Langevin equation."""
    # Parameters
    m = 1.0
    gamma = 0.5
    sigma = 1.0

    omega = 1.0
    pot_args = (omega,)

    args = (m, gamma, sigma, pot_args, sho_potential)

    # Initial conditions
    x0 = 1.0
    p0 = 0.0
    y0 = jnp.array([x0, p0])

    # Time span
    t0 = 0.0
    t1 = 10.0
    dt0 = 0.01
    n = int((t1 - t0) / dt0)

    sol = solve_langevin(t0, t1, dt0, y0, args, n, key=key)

    return SHOSimulationResult(
        times=sol.ts,
        xs=sol.ys[:, 0],
        ps=sol.ys[:, 1],
        omega=omega,
        gamma=gamma,
        sigma=sigma,
        m=m,
    )


# Solve a single trajectory and plot the results
key = jrandom.PRNGKey(42)  # Change seed to generate new result
result = solve_single_trajectory_sho(key=key)
fig_x, ax, line = plot_x_evolution(result)
fig_p, ax, line = plot_p_evolution(result)

fig_x.savefig("position_evolution.png", dpi=300, bbox_inches="tight")
fig_p.savefig("momentum_evolution.png", dpi=300, bbox_inches="tight")

print(
    "Figures successfully saved as 'position_evolution.png' and 'momentum_evolution.png'!"
)
