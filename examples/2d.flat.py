from dataclasses import dataclass

import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    IsfConfig,
    get_fancy_figure,
    plot_2d_trajectory,
    plot_exact_isf_flat,
    plot_isf,
    plot_p_histogram,
    sample_result,
)
from classical_diffusion.solve import (
    FlatParams,
    InitialConditions,
    PhysicalParams,
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
    rng = np.random.default_rng(seed=0)
    key = jrandom.PRNGKey(100)

    temp = 3.0
    m = 1.0

    # Distribution
    dist_params = FlatParams(
        physical_parameters=PhysicalParams(gamma=0.5, temp=temp, m=m),
        time_span=TimeSpan(t0=0, t1=30_000, dt=0.01, burn_in=1),
        initial_conditions=InitialConditions(
            x0=np.array([[0.0, 0.0]]), p0=np.array([[0.0, 0.0]])
        ),
    )

    dist_result = solve_ensemble.load_or_call_cached(params=dist_params, _key=key)

    fig, ax = get_fancy_figure(fig_size=(6, 4))
    _, ax, line = plot_2d_trajectory(result=dist_result, ax=ax)
    line.set_color(CAM_BLUE.warm)
    fig.savefig("/workspaces/classical_diffusion/examples/2d.flat.trajectory.pdf")

    dist_result_sampled = sample_result(dist_result)
    fig, ax = get_fancy_figure(fig_size=(6, 4))
    _, ax, line = plot_p_histogram(result=dist_result_sampled, ax=ax)
    fig.savefig("/workspaces/classical_diffusion/examples/2d.flat.distribution.p.pdf")

    # Ballistic
    gamma = 0.0
    n_trajectories = 2000
    cutoff = 4.0

    n_save_flat = dist_result_sampled.xs.shape[-1]
    sample_idx = rng.choice(n_save_flat, size=n_trajectories)

    x0 = dist_result_sampled.xs[0, :, sample_idx]
    p0 = dist_result_sampled.ps[0, :, sample_idx]

    initial_conditions = InitialConditions(x0=x0, p0=p0)

    ballistic_params = FlatParams(
        physical_parameters=PhysicalParams(gamma=0.0, temp=temp, m=m),
        time_span=TimeSpan(t0=0, t1=cutoff, dt=0.01, burn_in=0),
        initial_conditions=initial_conditions,
    )

    ballistic_result = solve_ensemble.load_or_call_cached(
        params=ballistic_params, _key=key
    )

    fig_isf, ax = get_fancy_figure(fig_size=(6, 4))
    _, ax, line_ballistic = plot_isf(
        result=ballistic_result,
        ax=ax,
        color=CAM_BLUE.warm,
        config=IsfConfig(delta_k=ballistic_params.delta_k, pairwise=False),
        time_scale=ballistic_params.characteristic_values.time,
    )

    _, ax, line_exact = plot_exact_isf_flat(
        params=ballistic_params,
        delta_k=ballistic_params.delta_k,
        color=CAM_BLUE.dark,
        ax=ax,
    )
    ax.set_xlim(0, cutoff / ballistic_params.characteristic_values.time)
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
        handles=[line_ballistic, line_exact],
        labels=["Simulation", "Theory"],
        labelcolor=CAM_SLATE_4,
    )
    fig_isf.savefig("/workspaces/classical_diffusion/examples/2d.flat.isf.pdf")
