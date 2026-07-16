from dataclasses import dataclass

import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    IsfConfig,
    get_fancy_figure,
    plot_exact_isf_sho,
    plot_isf,
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

CAM_CHERRY = CamColor(
    light="#F2CAD8",
    warm="#E18AAC",
    base="#CD3572",
    dark="#911449",
)

CAM_SLATE_4 = "#232830"


if __name__ == "__main__":
    t1s = [100, 1000, 10000]
    colors = [CAM_BLUE.warm, CAM_BLUE.warm, CAM_BLUE.base]

    fig_isf, ax_isf = get_fancy_figure(fig_size=(6, 4))
    sim_lines = []

    for t1, color in zip(t1s, colors, strict=True):
        params = SHOParams(
            physical_parameters=PhysicalParams(gamma=0.1, temp=1.0, m=1.0),
            omega=2,
            time_span=TimeSpan(t0=0, t1=t1, dt=0.01, burn_in=0),
            initial_conditions=InitialConditions(
                x0=np.array([[1.0]]), p0=np.array([[0.0]])
            ),
        )
        key = jrandom.PRNGKey(100)

        result = solve_ensemble(params=params, _key=key)
        _, ax_isf, line = plot_isf(
            result=result,
            ax=ax_isf,
            color=color,
            config=IsfConfig(delta_k=params.delta_k),
        )
        line.set_label(f"t = {t1 / params.characteristic_values.time}")
        sim_lines.append(line)
        print(f"Simulation t = {t1} complete")

    delta_k = params.delta_k[0]
    ax_isf.axhline(
        y=np.exp(
            -(delta_k**2) * params.physical.temp / (params.physical.m * params.omega**2)
        ),
        linestyle=":",
        color="black",
    )

    _, ax_isf, line_isf_exact = plot_exact_isf_sho(
        result=result, ax=ax_isf, color=CAM_CHERRY.dark
    )
    line_isf_exact.set_label("exact")

    ax_isf.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
        handles=[*sim_lines, line_isf_exact],
        labels=[line.get_label() for line in sim_lines] + ["Exact"],
        labelcolor=CAM_SLATE_4,
    )
    ax_isf.set_xlim(0, 50)
    fig_isf.savefig(
        "./examples/harmonic.isf.comparison.pdf",
        dpi=300,
        bbox_inches="tight",
    )
