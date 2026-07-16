from dataclasses import dataclass

import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    IsfConfig,
    get_fancy_figure,
    plot_exact_isf_sho,
    plot_isf,
    plot_kinetic_probability,
)
from classical_diffusion.solve import (
    InitialConditions,
    PhysicalParams,
    SHOParams,
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


if __name__ == "__main__":
    params = SHOParams(
        physical_parameters=PhysicalParams(gamma=0.1, temp=1.0, m=1.0),
        omega=2,
        time_span=TimeSpan(t0=0, t1=10_000, dt=0.01, burn_in=0),
        initial_conditions=InitialConditions(
            x0=np.array([[1.0]]), p0=np.array([[0.0]])
        ),
    )

    key = jrandom.PRNGKey(100)  # Change seed to generate new result
    result = solve_ensemble.load_or_call_cached(params=params, _key=key)

    fig_isf, ax_isf = get_fancy_figure(fig_size=(6, 4))
    _, ax_isf, line_isf_sim = plot_isf(
        result=result,
        ax=ax_isf,
        config=IsfConfig(delta_k=params.delta_k),
        color=CAM_BLUE.warm,
    )
    line_isf_sim.set_label("simulation")

    delta_k = params.delta_k[0]
    ax_isf.axhline(
        y=np.exp(
            -(delta_k**2) * params.physical.temp / (params.physical.m * params.omega**2)
        ),
        linestyle=":",
        color="black",
    )

    _, ax_isf, line_isf_exact = plot_exact_isf_sho(
        result=result, ax=ax_isf, color=CAM_BLUE.dark
    )
    line_isf_exact.set_label("exact")
    ax_isf.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
        handles=[line_isf_sim, line_isf_exact],
        labels=["Simulation", "Exact"],
        labelcolor=CAM_SLATE_4,
    )
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
        "/workspaces/classical_diffusion/examples/harmonic.kinetic.probability.pdf"
    )
