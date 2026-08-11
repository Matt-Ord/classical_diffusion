import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    plot_isf_with_delta_k,
)
from classical_diffusion.langevin import (
    breakdown_filtered_ballistic_trajectory_butterworth,
    get_initial_conditions,
    solve_ballistic_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import (
    PeriodicSystemFCC,
    UnitSystem,
    get_diffusion_time,
)

system = PeriodicSystemFCC(
    gamma=4e11,
    temperature=100,
    m=6e-27,
    delta_x=1.48e-10,
    barrier_energy=4e-21,
    units=UnitSystem(),
)

normalized_system = system.with_normalized_units()

key = jrandom.PRNGKey(100)


def _plot_inelastic_trends() -> None:

    initial_conditions = get_initial_conditions(normalized_system, n_samples=2000)

    ballistic_result = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(6e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        initial_conditions,
        _key=key,
    )

    _, inelastic_result = breakdown_filtered_ballistic_trajectory_butterworth(
        ballistic_result,
        minimum_timescale=get_diffusion_time(
            normalized_system, characteristic_length=normalized_system.delta_x / 0.5
        ),
    )
    direction = np.array([1, 1])
    delta_k_values = np.linspace(0.1, 2.0, 9) * tuple(
        7e9 * direction / np.linalg.norm(direction)
    )

    fig, ax = get_fancy_figure()
    _, ax = plot_isf_with_delta_k(
        result=inelastic_result.with_si_units(),
        ax=ax,
        delta_k_values=delta_k_values,
        pairwise=False,
    )
    ax.set_xlim(
        0,
        2e-12,
    )
    ax.set_ylim(
        0.4,
        1.0,
    )
    fig.savefig(
        "examples/ballistic_langevin/inelastic_isf_trends/2d_fcc.trend.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _plot_inelastic_trends()
