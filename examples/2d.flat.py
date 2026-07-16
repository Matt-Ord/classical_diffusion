import dataclasses

import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    IsfConfig,
    plot_2d_trajectory,
    plot_exact_isf_flat,
    plot_isf,
    plot_p_histogram,
    sample_result,
)
from classical_diffusion.solve import (
    FlatParameters,
    InitialConditions,
    PhysicalParameters,
    TimeSpan,
    solve_ensemble,
)
from classical_diffusion.theme import get_fancy_figure

if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    key = jrandom.PRNGKey(100)

    # Distribution
    dist_params = FlatParameters(
        physical_parameters=PhysicalParameters(gamma=0.5, temperature=3, m=1),
        time_span=TimeSpan(t0=0, t1=30_000, dt=0.01, burn_in=1),
        initial_conditions=InitialConditions(
            x0=np.array([[0.0, 0.0]]), p0=np.array([[0.0, 0.0]])
        ),
    )

    dist_result = solve_ensemble.load_or_call_cached(params=dist_params, _key=key)

    fig, ax = get_fancy_figure()
    _, ax, line = plot_2d_trajectory(result=dist_result, ax=ax)
    fig.savefig("examples/2d.flat.trajectory.pdf")

    dist_result_sampled = sample_result(dist_result)
    fig, ax = get_fancy_figure()
    _, ax, line = plot_p_histogram(result=dist_result_sampled, ax=ax)
    fig.savefig("examples/2d.flat.distribution.p.pdf")

    # Ballistic

    n_trajectories = 2000
    cutoff = 4.0

    n_save_flat = dist_result_sampled.xs.shape[-1]
    sample_idx = rng.choice(n_save_flat, size=n_trajectories)

    x0 = dist_result_sampled.xs[0, :, sample_idx]
    p0 = dist_result_sampled.ps[0, :, sample_idx]

    initial_conditions = InitialConditions(x0=x0, p0=p0)

    ballistic_params = FlatParameters(
        physical_parameters=dataclasses.replace(
            dist_params.physical_parameters, gamma=0
        ),
        # TODO: separate simulation from physical, as different args
        time_span=TimeSpan(t0=0, t1=cutoff, dt=0.01, burn_in=0),
        initial_conditions=initial_conditions,
    )

    ballistic_result = solve_ensemble.load_or_call_cached(
        params=ballistic_params, _key=key
    )

    fig_isf, ax = get_fancy_figure()
    _, ax, line_ballistic = plot_isf(
        result=ballistic_result,
        ax=ax,
        config=IsfConfig(delta_k=ballistic_params.delta_k, pairwise=False),
        time_scale=ballistic_params.characteristic_values.time,
    )

    _, ax, line_exact = plot_exact_isf_flat(
        params=ballistic_params,
        delta_k=ballistic_params.delta_k,
        ax=ax,
    )
    ax.set_xlim(0, cutoff / ballistic_params.characteristic_values.time)
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
        handles=[line_ballistic, line_exact],
        labels=["Simulation", "Theory"],
    )
    fig_isf.savefig("examples/2d.flat.isf.pdf")
