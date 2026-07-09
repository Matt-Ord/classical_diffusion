from dataclasses import dataclass, field

import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from classical_diffusion.analysis import plot_isf, plot_p_evolution, plot_x_evolution
from classical_diffusion.solve import (
    SimulationParams,
    TimeSpan,
    solve_langevin,
)


@dataclass(kw_only=True)  # no frozen=True # TODO: why?
class SHOParameters(SimulationParams):
    """Parameters for the harmonic Langevin equation."""

    omega: float
    potential: sp.Expr = field(init=False)

    def __post_init__(self) -> None:
        # TODO:  Dont post __post_init__ here
        self.potential = 0.5 * self.omega**2 * sp.symbols("x0") ** 2


if __name__ == "__main__":
    # Parameters

    # Time span
    times = TimeSpan(t0=0, t1=20, dt0=0.01)

    # Burn in?
    burn_in_time = 0
    burn_in_steps = int(burn_in_time / times.dt0)

    n_times = round((times.t1 - times.t0) / times.dt0)
    delta_x = np.zeros((1, n_times))

    params = SHOParameters(
        gamma=0.5,
        sigma=1.0,
        m=1.0,
        omega=1.0,
        # do you mean dt rather than dt0?
        # TODO: times rather than time?
        time=times,
        # TODO: from this API it is not clear what is x and what is p
        y0=np.array([1.0, 0.0]),
        delta_x=delta_x,
        n_dims=1,
    )

    key = jrandom.PRNGKey(100)  # Change seed to generate new result
    result = solve_langevin(params=params, key=key)

    # TODO: Define burn-in steps here - also why is this not part of t0?
    x_equilibriated = result.xs[:, burn_in_steps:]

    # TODO: use your physics knowledge here!
    x_std = float(np.std(x_equilibriated))
    delta_k = (1.0 / x_std,)

    fig, (ax_x, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    _, ax_x, line_x = plot_x_evolution(result=result, ax=ax_x)
    _, ax_p, line_p = plot_p_evolution(result=result, ax=ax_p)

    # TODO: would be nice if these plots were motivated by some physics.
    # For simple x-p plots, plot alongside the ballistic motion and maybe
    # give examples of different friction. Think how you would present this
    # at the end of the project.
    # Would also be nice to see
    ax_x.set_xlabel("")  # top plot doesn't need its own x-label, shares with bottom
    fig.suptitle("Position and Momentum Evolution")
    fig.tight_layout()
    fig.savefig(
        "./examples/trajectory_x_p_harmonic.png",
        dpi=300,
        bbox_inches="tight",
    )
    # TODO: remove this here!
    fig, ax = None, None
    # TODO: it is clear that the last half of the ISF is not correct
    # you need to do a longer simulation.
    # Also, what value do you expect it to decay to? What frequency will it oscillate
    # at? this should be a nice way to demonstrate inelastic processes!
    fig, _, _ = plot_isf(result=result, delta_k=delta_k, ax=ax)

    fig.savefig(
        # TODO: pdf is always better!
        "./examples/isf_harmonic.png",
        dpi=300,
        bbox_inches="tight",
    )
