from dataclasses import dataclass

import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    IsfConfig,
    fold_result,
    get_fancy_figure,
    plot_elastic_p,
    plot_isf,
    plot_p_histogram,
    plot_phase_space_density,
    plot_x_evolution,
    plot_x_histogram,
    sample_result,
)
from classical_diffusion.solve import (
    InitialConditions,
    PeriodicParams,
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

    temp = 1.0
    m = 1.0
    lattice_spacing = 5.0
    amplitude = 1.5

    # Distribution
    gamma = 0.1
    n_trajectories = 200
    dist_params = PeriodicParams(
        physical_parameters=PhysicalParams(gamma=gamma, temp=temp, m=m),
        lattice_spacing=lattice_spacing,
        amplitude=amplitude,
        time_span=TimeSpan(t0=0, t1=1000, dt=0.01, burn_in=1),
        initial_conditions=InitialConditions(
            x0=np.full((n_trajectories, 1), 2.5),
            p0=np.full((n_trajectories, 1), 0.0),
        ),
    )

    dist_result = solve_ensemble.load_or_call_cached(params=dist_params, _key=key)

    dist_result_sampled = sample_result(dist_result)

    fig_phase_space_density, ax = get_fancy_figure(fig_size=(6, 4))
    _, ax, mesh = plot_phase_space_density(result=dist_result_sampled, ax=ax)
    fig_phase_space_density.savefig(
        "/workspaces/classical_diffusion/examples/1d.periodic.phase.space.density.pdf"
    )

    fig_x_hist, ax = get_fancy_figure(fig_size=(6, 4))
    _, ax, bars = plot_x_histogram(result=dist_result_sampled, ax=ax)
    fig_x_hist.savefig(
        "/workspaces/classical_diffusion/examples/1d.periodic.distribution.x.pdf"
    )

    dist_result_folded = fold_result(result=dist_result_sampled)

    fig_x_hist, ax = get_fancy_figure(fig_size=(6, 4))
    _, ax, bars = plot_x_histogram(result=dist_result_folded, ax=ax)
    fig_x_hist.savefig(
        "/workspaces/classical_diffusion/examples/1d.periodic.distribution.x.folded.pdf"
    )

    fig_p_hist, ax = get_fancy_figure(fig_size=(6, 4))
    _, ax, bars = plot_p_histogram(result=dist_result_folded, ax=ax)
    fig_p_hist.savefig(
        "/workspaces/classical_diffusion/examples/1d.periodic.distribution.p.pdf"
    )

    fig_phase_space_density, ax = get_fancy_figure(fig_size=(6, 4))
    _, ax, mesh = plot_phase_space_density(result=dist_result_folded, ax=ax)
    fig_phase_space_density.savefig(
        "/workspaces/classical_diffusion/examples/1d.periodic.phase.space.density.folded.pdf"
    )

    # Full ISF
    n_trajectories = 200

    initial_conditions = InitialConditions(
        x0=rng.choice(dist_result_folded.xs.reshape(-1), size=n_trajectories)[:, None],
        p0=rng.choice(dist_result_folded.ps.reshape(-1), size=n_trajectories)[:, None],
    )

    full_params = PeriodicParams(
        physical_parameters=PhysicalParams(gamma=gamma, temp=temp, m=m),
        lattice_spacing=lattice_spacing,
        amplitude=amplitude,
        time_span=TimeSpan(t0=0, t1=1000, dt=0.01, burn_in=0),
        initial_conditions=initial_conditions,
    )

    shared_time_scale = full_params.characteristic_values.time
    full_result = solve_ensemble.load_or_call_cached(params=full_params, _key=key)

    full_isf, ax = get_fancy_figure(fig_size=(6, 4))
    _, ax, line_full = plot_isf(
        result=full_result,
        ax=ax,
        color=CAM_BLUE.dark,
        time_scale=shared_time_scale,
        config=IsfConfig(delta_k=full_params.delta_k),
    )

    # Ballistic
    key = jrandom.PRNGKey(200)
    gamma = 0.0
    n_trajectories = 50_000
    cutoff = 20

    initial_conditions = InitialConditions(
        x0=rng.choice(dist_result_folded.xs.reshape(-1), size=n_trajectories)[:, None],
        p0=rng.choice(dist_result_folded.ps.reshape(-1), size=n_trajectories)[:, None],
    )

    ballistic_params = PeriodicParams(
        physical_parameters=PhysicalParams(gamma=gamma, temp=temp, m=m),
        lattice_spacing=lattice_spacing,
        amplitude=amplitude,
        time_span=TimeSpan(t0=0, t1=cutoff, dt=0.01, burn_in=0),
        initial_conditions=initial_conditions,
    )

    ballistic_result = solve_ensemble.load_or_call_cached(
        params=ballistic_params, _key=key
    )

    _, ax, line_ballistic = plot_isf(
        result=ballistic_result,
        ax=ax,
        color=CAM_BLUE.warm,
        config=IsfConfig(delta_k=full_params.delta_k, pairwise=False),
        time_scale=shared_time_scale,
    )
    ax.set_xlim(0, cutoff / shared_time_scale)
    ax.set_ylim(0.7, 1.0)
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
        handles=[line_full, line_ballistic],
        labels=["Full ISF", "Ballistic ISF"],
        labelcolor=CAM_SLATE_4,
    )
    full_isf.savefig(
        "/workspaces/classical_diffusion/examples/1d.periodic.full.isf.pdf"
    )

    fig_x_trajectories, ax = get_fancy_figure(fig_size=(6, 4))
    _, ax = plot_elastic_p(
        result=ballistic_result, ax=ax, n_trajectories=n_trajectories
    )
    fig_x_trajectories.savefig(
        "/workspaces/classical_diffusion/examples/1d.periodic.p.elastic.convergence.pdf"
    )

    fig_trajectories, ax = get_fancy_figure(fig_size=(6, 4))
    _, ax, lines = plot_x_evolution(
        result=ballistic_result, ax=ax, n_trajectories=n_trajectories
    )
    fig_trajectories.savefig(
        "/workspaces/classical_diffusion/examples/1d.periodic.x.trajectories.pdf"
    )
