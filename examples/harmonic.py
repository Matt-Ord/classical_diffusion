from dataclasses import astuple, dataclass

import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from classical_diffusion.analysis import (
    get_fancy_figure,
    plot_exact_isf,
    plot_isf,
    plot_kinetic_probability,
    plot_p_evolution,
    plot_x_evolution,
)
from classical_diffusion.solve import (
    CharacteristicValues,
    InitialConditions,
    PhysicalParams,
    SimulationParams,
    TimeSpan,
    solve_langevin,
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


class SHOParams(SimulationParams):
    """Parameters for the harmonic Langevin equation."""

    def __init__(
        self,
        *,
        omega: float,
        physical_parameters: PhysicalParams,
        time_span: TimeSpan,
        initial_conditions: InitialConditions,
    ) -> None:
        potential = 0.5 * omega**2 * sp.symbols("x0") ** 2

        characteristic_length = np.sqrt(
            1.0 * physical_parameters.temp / (physical_parameters.m * omega**2)
        )  # k_B set to 1 for now

        characteristic_time = 1 / omega

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
        self.omega = omega
        self.delta_k: tuple[float, ...] = (1 / self.characteristic_values.length,)


def get_exact_isf_sho(params: SHOParams) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact ISF for simulation."""
    times = np.arange(params.time_span.t0, params.time_span.t1, params.time_span.dt)
    gamma, temp, m = astuple(params.physical)
    f = np.sqrt(params.omega**2 - gamma**2 / 4)
    delta_k = params.delta_k[0]  # unpack from tuple

    return np.exp(
        -(delta_k**2)
        * ((1.0 * temp) / (m * params.omega**2))
        * (
            1
            - np.exp(-gamma * times / 2)
            * (np.cos(f * times) + (gamma / (2 * f)) * np.sin(f * times))
        )
    ), times


if __name__ == "__main__":
    params = SHOParams(
        physical_parameters=PhysicalParams(gamma=0.1, temp=1.0, m=1.0),
        omega=2,
        time_span=TimeSpan(t0=0, t1=10000, dt=0.001, n_save=100000),
        initial_conditions=InitialConditions(x0=np.array([1.0]), p0=np.array([0.0])),
    )

    key = jrandom.PRNGKey(100)  # Change seed to generate new result
    result = solve_langevin(params=params, key=key)

    fig_trajectory, (ax_x, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    _, ax_x, line_x = plot_x_evolution(result=result, ax=ax_x)
    _, ax_p, line_p = plot_p_evolution(result=result, ax=ax_p)

    fig_isf, ax_isf = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
    _, ax_isf, line_isf_sim = plot_isf(result=result, ax=ax_isf, delta_k=params.delta_k)
    line_isf_sim.set_label("simulation")

    isf_exact, times = get_exact_isf_sho(params)
    _, ax_isf, line_isf_exact = plot_exact_isf(isf=isf_exact, times=times, ax=ax_isf)
    line_isf_exact.set_label("exact")

    delta_k = params.delta_k[0]
    ax_isf.axhline(
        y=np.exp(
            -(delta_k**2) * params.physical.temp / (params.physical.m * params.omega**2)
        ),
        linestyle=":",
        color="black",
    )

ax_x.set_xlabel("")
ax_x.legend()
ax_p.legend()
fig_trajectory.suptitle("Position and Momentum Evolution")
fig_trajectory.tight_layout()
fig_trajectory.savefig(
    "./examples/harmonic.trajectory.pdf",
    dpi=300,
    bbox_inches="tight",
)

ax_isf.legend()
ax_isf.set_xlim(0, 30)
fig_isf.savefig(
    "./examples/harmonic.isf.pdf",
    dpi=300,
    bbox_inches="tight",
)

fig, ax = get_fancy_figure()
_, _, (line0, line1, line2, bars) = plot_kinetic_probability(
    result=result, max_energy=6, ax=ax
)
line0.set_color(CAM_BLUE.dark)
line1.set_alpha(0)
line2.set_alpha(0)
for patch in bars.patches:
    patch.set_color(CAM_BLUE.warm)
    patch.set_alpha(1)
ax.legend(
    frameon=False,
    loc="upper right",
    fontsize=9,
    handles=[line0],
    labels=["Theoretical"],
    labelcolor=CAM_SLATE_4,
)
fig.savefig(
    "/workspaces/classical_diffusion/examples/kinetic_probability.classical.pdf"
)
