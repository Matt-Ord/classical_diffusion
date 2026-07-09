import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from classical_diffusion.analysis import plot_isf, plot_p_evolution, plot_x_evolution
from classical_diffusion.solve import (
    InitialConditions,
    SimulationParams,
    TimeSpan,
    solve_langevin,
)


class SHOParameters(SimulationParams):
    """Parameters for the harmonic Langevin equation."""

    def __init__(self, omega: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.omega = omega
        self.delta_k = np.sqrt(
            self.m * self.omega**2 / (1.0 * self.temp)
        )  # k_B set to 1 for now

    @property
    def potential(self) -> sp.Expr:
        """Return symbolic function for sho potential."""
        return 0.5 * self.omega**2 * sp.symbols("x0") ** 2


if __name__ == "__main__":
    params = SHOParameters(
        gamma=0.0,
        temp=1.0,
        m=1.0,
        omega=1.0,
        time_span=TimeSpan(t0=0, t1=20, dt=0.01, n_save=2000),
        initial_conditions=InitialConditions(x0=np.array([3.0]), p0=np.array([0.0])),
    )

    key = jrandom.PRNGKey(100)  # Change seed to generate new result
    result = solve_langevin(params=params, key=key)

    fig, (ax_x, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    _, ax_x, line_x = plot_x_evolution(result=result, ax=ax_x)
    _, ax_p, line_p = plot_p_evolution(result=result, ax=ax_p)

    ax_x.set_xlabel("")
    fig.suptitle("Position and Momentum Evolution")
    fig.tight_layout()
    fig.savefig(
        "./examples/harmonic.trajectory.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    x_std = float(np.std(result.xs))
    delta_k = (1.0 / x_std,)

    fig, _, _ = plot_isf(result=result, delta_k=params.delta_k)

    fig.savefig(
        "./examples/harmonic.isf.pdf",
        dpi=300,
        bbox_inches="tight",
    )
