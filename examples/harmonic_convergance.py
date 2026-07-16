import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    IsfConfig,
    plot_exact_isf_sho,
    plot_isf,
)
from classical_diffusion.solve import (
    InitialConditions,
    PhysicalParameters,
    SHOParameters,
    TimeSpan,
    solve_ensemble,
)
from classical_diffusion.theme import (
    CAM_BLUE,
    CAM_SLATE_4,
    get_fancy_figure,
)

if __name__ == "__main__":
    t1s = [100, 1000, 10000]
    colors = [CAM_BLUE.warm, CAM_BLUE.warm, CAM_BLUE.base]

    fig_isf, ax_isf = get_fancy_figure()
    sim_lines = []

    for t1, _color in zip(t1s, colors, strict=True):
        params = SHOParameters(
            physical_parameters=PhysicalParameters(gamma=0.1, temperature=1.0, m=1.0),
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
            config=IsfConfig(delta_k=params.delta_k),
        )
        line.set_label(f"t = {t1 / params.characteristic_values.time}")
        sim_lines.append(line)
        print(f"Simulation t = {t1} complete")

    delta_k = params.delta_k[0]
    ax_isf.axhline(
        y=np.exp(
            -(delta_k**2)
            * params.physical_parameters.temperature
            / (params.physical_parameters.m * params.omega**2)
        ),
        linestyle=":",
        color="black",
    )

    _, ax_isf, line_isf_exact = plot_exact_isf_sho(result=result, ax=ax_isf)
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
