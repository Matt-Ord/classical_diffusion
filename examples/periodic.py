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


class PeriodicParameters(SimulationParams):
    """Parameters for the periodic Langevin equation."""

    def __init__(self, amplitude: float, lattice_spacing: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.lattice_spacing = lattice_spacing
        self.delta_x = 2 * np.pi / lattice_spacing

    @property
    def potential(self) -> sp.Expr:
        """Return symbolic function for sho potential."""
        return self.amplitude * sp.cos(sp.symbols("x0") * self.delta_x)


if __name__ == "__main__":
    params = PeriodicParameters(
        gamma=1.0,
        temp=1.0,
        m=1.0,
        lattice_spacing=1,
        amplitude=2.0,
        time_span=TimeSpan(t0=0, t1=20, dt=0.01, n_save=2000),
        initial_conditions=InitialConditions(x0=np.array([1.0]), p0=np.array([0.0])),
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
        "./examples/periodic.trajectory.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    delta_k = 1 / lattice_spacing

    fig, _, _ = plot_isf(result=result, delta_k=delta_k)

    fig.savefig(
        "./examples/periodic.isf.pdf",
        dpi=300,
        bbox_inches="tight",
    )
