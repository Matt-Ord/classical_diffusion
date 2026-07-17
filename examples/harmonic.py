import jax.random as jrandom
import numpy as np

from classical_diffusion.langevin import (
    InitialConditions,
    IsfConfig,
    TimeSpan,
    plot_isf,
    solve_ensemble,
)
from classical_diffusion.plot import (
    get_fancy_figure,
)

if __name__ == "__main__":
    params = SHOParameters(
        physical_parameters=PhysicalParameters(gamma=0.1, temperature=1.0, m=1.0),
        omega=2,
        time_span=TimeSpan(t0=0, t1=10_000, dt=0.01, burn_in=0),
        initial_conditions=InitialConditions(
            x0=np.array([[1.0]]), p0=np.array([[0.0]])
        ),
    )

    key = jrandom.PRNGKey(100)  # Change seed to generate new result
    result = solve_ensemble.load_or_call_cached(params=params, _key=key)

    fig_isf, ax_isf = get_fancy_figure()
    # TODO: laggy?
    _, ax_isf, line_isf_sim = plot_isf(
        result=result,
        ax=ax_isf,
        # TODO: overkill abstractions?
        config=IsfConfig(delta_k=params.delta_k),
    )
    line_isf_sim.set_label("simulation")

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
        loc="upper right",
        handles=[line_isf_sim, line_isf_exact],
        labels=["Simulation", "Exact"],
    )
    ax_isf.set_xlim(0, 30)
    fig_isf.savefig(
        "./examples/harmonic.isf.pdf",
        dpi=300,
        bbox_inches="tight",
    )
