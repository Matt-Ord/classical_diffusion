import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from classical_diffusion.analysis import plot_isf, plot_p_evolution, plot_x_evolution
from classical_diffusion.solve import (
    CharacteristicValues,
    InitialConditions,
    PhysicalParams,
    SimulationParams,
    TimeSpan,
    solve_langevin,
)


class PeriodicParams(SimulationParams):
    """Parameters for the harmonic Langevin equation."""

    def __init__(
        self,
        *,
        lattice_spacing: float,
        amplitude: float,
        physical_parameters: PhysicalParams,
        time_span: TimeSpan,
        initial_conditions: InitialConditions,
    ) -> None:
        potential = amplitude * sp.cos(2 * np.pi * sp.symbols("x0") / lattice_spacing)

        characteristic_length = 1 / lattice_spacing

        characteristic_time = np.sqrt(
            1.0 * physical_parameters.temp / (physical_parameters.m * lattice_spacing)
        )

        super().__init__(
            physical_parameters=physical_parameters,
            time_span=time_span,
            initial_conditions=initial_conditions,
            potential=potential,
            characteristic_values=CharacteristicValues(
                length=characteristic_length,
                time=characteristic_time,
            ),
        )
        self.lattice_spacing = lattice_spacing
        self.amplitude = amplitude
        self.delta_k: tuple[float, ...] = (1 / self.characteristic_values.length,)


if __name__ == "__main__":
    gammas = [0.0, 0.1, 1.0]
    fig_trajectory, (ax_x, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    fig_isf, ax_isf = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
    for gamma in gammas:
        params = PeriodicParams(
            physical_parameters=PhysicalParams(gamma=gamma, temp=5.0, m=1.0),
            lattice_spacing=1.0,
            amplitude=1.0,
            time_span=TimeSpan(t0=0, t1=10, dt=0.001, n_save=1000),
            initial_conditions=InitialConditions(
                x0=np.array([0.25]), p0=np.array([0.0])
            ),
        )

        key = jrandom.PRNGKey(100)  # Change seed to generate new result
        result = solve_langevin(params=params, key=key)

        _, ax_x, line_x = plot_x_evolution(result=result, ax=ax_x)
        _, ax_p, line_p = plot_p_evolution(result=result, ax=ax_p)

        line_x.set_label(f"gamma = {gamma}")
        line_p.set_label(f"gamma = {gamma}")

        _, ax_isf, line_isf = plot_isf(
            result=result,
            ax=ax_isf,
            delta_k=params.delta_k,
        )

        line_isf.set_label(f"gamma = {gamma}")

    ax_x.set_xlabel("")
    ax_x.legend()
    ax_p.legend()
    fig_trajectory.suptitle("Position and Momentum Evolution")
    fig_trajectory.tight_layout()
    fig_trajectory.savefig(
        "./examples/periodic.trajectory.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    ax_isf.legend()
    fig_isf.savefig(
        "./examples/periodic.isf.pdf",
        dpi=300,
        bbox_inches="tight",
    )
