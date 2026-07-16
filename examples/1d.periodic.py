import dataclasses

import jax.random as jrandom
import numpy as np

from classical_diffusion.langevin import (
    InitialConditions,
    IsfConfig,
    TimeSpan,
    fold_result,
    plot_elastic_p,
    plot_isf,
    plot_p_histogram,
    plot_phase_space_density,
    plot_x_evolution,
    plot_x_histogram,
    sample_result,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import plot_periodic_potential_1d

if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    key = jrandom.PRNGKey(100)

    distribution_params = PeriodicParameters(
        physical_parameters=PhysicalParameters(gamma=0.1, temperature=1.0, m=1.0),
        lattice_spacing=5.0,
        amplitude=1.5,
        time_span=TimeSpan(t0=0, t1=1000, dt=0.01, burn_in=1),
        initial_conditions=InitialConditions(
            x0=np.full((200, 1), 2.5),
            p0=np.full((200, 1), 0.0),
        ),
    )

    fig, ax = get_fancy_figure()
    fig, _, _ = plot_periodic_potential_1d(distribution_params, ax=ax)
    fig.savefig("examples/1d.periodic.potential.pdf")

    dist_result = solve_ensemble.load_or_call_cached(
        params=distribution_params, _key=key
    )

    dist_result_sampled = sample_result(dist_result)

    fig_phase_space_density, ax = get_fancy_figure()
    _, ax, mesh = plot_phase_space_density(result=dist_result_sampled, ax=ax)
    fig_phase_space_density.savefig("examples/1d.periodic.phase.space.density.pdf")

    fig_x_hist, ax = get_fancy_figure()
    _, ax, bars = plot_x_histogram(result=dist_result_sampled, ax=ax)
    fig_x_hist.savefig("examples/1d.periodic.distribution.x.pdf")

    dist_result_folded = fold_result(result=dist_result_sampled)

    fig_x_hist, ax = get_fancy_figure()
    _, ax, bars = plot_x_histogram(result=dist_result_folded, ax=ax)
    fig_x_hist.savefig("examples/1d.periodic.distribution.x.folded.pdf")

    fig_p_hist, ax = get_fancy_figure()
    _, ax, bars = plot_p_histogram(result=dist_result_folded, ax=ax)
    fig_p_hist.savefig("examples/1d.periodic.distribution.p.pdf")

    fig_phase_space_density, ax = get_fancy_figure()
    _, ax, mesh = plot_phase_space_density(result=dist_result_folded, ax=ax)
    fig_phase_space_density.savefig(
        "examples/1d.periodic.phase.space.density.folded.pdf"
    )

    # Full ISF
    n_trajectories = 200

    initial_conditions = InitialConditions(
        x0=rng.choice(dist_result_folded.xs.reshape(-1), size=n_trajectories)[:, None],
        p0=rng.choice(dist_result_folded.ps.reshape(-1), size=n_trajectories)[:, None],
    )

    full_params = PeriodicParameters(
        physical_parameters=distribution_params.physical_parameters,
        lattice_spacing=distribution_params.lattice_spacing,
        amplitude=distribution_params.amplitude,
        time_span=TimeSpan(t0=0, t1=1000, dt=0.01, burn_in=0),
        initial_conditions=initial_conditions,
    )

    shared_time_scale = full_params.characteristic_values.time
    full_result = solve_ensemble.load_or_call_cached(params=full_params, _key=key)

    full_isf, ax = get_fancy_figure()
    _, ax, line_full = plot_isf(
        result=full_result,
        ax=ax,
        time_scale=shared_time_scale,
        config=IsfConfig(delta_k=full_params.delta_k),
    )

    # Ballistic
    key = jrandom.PRNGKey(200)

    n_trajectories = 50_000
    cutoff = 20

    initial_conditions = InitialConditions(
        x0=rng.choice(dist_result_folded.xs.reshape(-1), size=n_trajectories)[:, None],
        p0=rng.choice(dist_result_folded.ps.reshape(-1), size=n_trajectories)[:, None],
    )

    ballistic_params = PeriodicParameters(
        physical_parameters=dataclasses.replace(
            distribution_params.physical_parameters, gamma=0
        ),
        lattice_spacing=distribution_params.lattice_spacing,
        amplitude=distribution_params.amplitude,
        time_span=TimeSpan(t0=0, t1=cutoff, dt=0.01, burn_in=0),
        initial_conditions=initial_conditions,
    )

    ballistic_result = solve_ensemble.load_or_call_cached(
        params=ballistic_params, _key=key
    )

    _, ax, line_ballistic = plot_isf(
        result=ballistic_result,
        ax=ax,
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
    )
    full_isf.savefig("examples/1d.periodic.full.isf.pdf")

    fig_x_trajectories, ax = get_fancy_figure()
    _, ax = plot_elastic_p(
        result=ballistic_result, ax=ax, n_trajectories=n_trajectories
    )
    fig_x_trajectories.savefig("examples/1d.periodic.p.elastic.convergence.pdf")

    fig_trajectories, ax = get_fancy_figure()
    _, ax, lines = plot_x_evolution(
        result=ballistic_result, ax=ax, n_trajectories=n_trajectories
    )
    fig_trajectories.savefig("examples/1d.periodic.x.trajectories.pdf")
