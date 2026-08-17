import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    plot_isf_with_delta_k,
)
from classical_diffusion.langevin import (
    PeriodicSystem1D,
    breakdown_ballistic_trajectory,
    get_diffusion_time,
    solve_ballistic_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan

system = PeriodicSystem1D(
    gamma=4e11,
    temperature=100,
    m=6e-27,
    delta_x=1.48e-10,
    barrier_energy=4e-21,
)

normalized_system = system.with_normalized_units()

key = jrandom.PRNGKey(100)


def _plot_inelastic_trends() -> None:

    ballistic_result = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(6e-12),
            n_steps=1000,
        ),
        n_samples=2000,
        _key=key,
    )

    _, inelastic_result = breakdown_ballistic_trajectory(
        ballistic_result,
        minimum_timescale=get_diffusion_time(
            normalized_system, characteristic_length=normalized_system.delta_x / 0.5
        ),
    )
    delta_k_values = np.linspace(0.1, 2.0, 9) * (2e10)

    fig, ax = get_fancy_figure()
    _, ax = plot_isf_with_delta_k(
        result=inelastic_result.with_si_units(),
        ax=ax,
        delta_k_values=delta_k_values,
        pairwise=False,
    )
    ax.set_xlim(
        0,
        1e-12,
    )
    fig.savefig(
        "examples/ballistic_langevin/inelastic_isf_trends/1d_periodic.trend.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _plot_inelastic_trends()
