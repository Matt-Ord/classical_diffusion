import numpy as np

from classical_diffusion.analysis import (
    plot_isf_with_delta_k,
)
from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    breakdown_ballistic_trajectory,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import CAM_CHERRY_CMAP, get_two_panel_figure
from classical_diffusion.simulation import TimeSpan


def _plot_inelastic_trends() -> None:

    system = SODIUM_COPPER_SYSTEM_1D

    ballistic_result = solve_ensemble_ballistic(
        system,
        TimeSpan(
            t_start=-10e-12,
            t_end=10e-12,
            n_steps=1000,
        ),
        n_samples=2000,
    )

    elastic_result, inelastic_result = breakdown_ballistic_trajectory(
        ballistic_result,
        filter_timescale=1 / system.gamma,
    )
    delta_k_values = np.linspace(0.1, 2.0, 9) * 2 * np.pi / system.delta_x * 0.5

    fig, ax = get_two_panel_figure()
    _, ax[0] = plot_isf_with_delta_k(
        result=elastic_result,
        ax=ax[0],
        delta_k_values=delta_k_values,
        pairwise=False,
    )

    _, ax[1] = plot_isf_with_delta_k(
        result=inelastic_result,
        ax=ax[1],
        delta_k_values=delta_k_values,
        pairwise=False,
        cmap=CAM_CHERRY_CMAP,
    )

    ax[0].set_xlim(0, 1e-12)
    ax[0].set_ylim(0.8, 1.0)
    ax[1].set_xlim(0, 1e-12)
    fig.savefig(
        "examples/ballistic_langevin/1d_periodic.isf_trend.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _plot_inelastic_trends()
