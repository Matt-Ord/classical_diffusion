import jax.random as jrandom
import numpy as np

from classical_diffusion.langevin import (
    InitialConditions,
    IsfConfig,
    SimulationParameters,
    TimeSpan,
    plot_isf,
    plot_p_histogram,
    sample_result,
    solve_ensemble,
)
from classical_diffusion.langevin._analysis import plot_2d_trajectory
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system._system import FlatSystem2D

if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    key = jrandom.PRNGKey(100)

    # Distribution
    dist_params = SimulationParameters(
        system=FlatSystem2D(gamma=0.5, temperature=3, m=1),
        time_span=TimeSpan(t0=0, t1=30_000, dt=0.01),
        initial_conditions=InitialConditions(
            x0=np.array([[0.0, 0.0]]), p0=np.array([[0.0, 0.0]])
        ),
    )

    dist_result = solve_ensemble.load_or_call_cached(params=dist_params, _key=key)

    fig, ax = get_fancy_figure()
    _, ax, line = plot_2d_trajectory(result=dist_result, ax=ax)
    fig.savefig("examples/2d.flat.trajectory.pdf")

    dist_result_sampled = sample_result(
        dist_result, stride_time=2 / dist_params.system.gamma
    )
    fig, ax = get_fancy_figure()
    _, ax, line = plot_p_histogram(result=dist_result_sampled, ax=ax)
    fig.savefig("examples/2d.flat.distribution.p.pdf")

    # Ballistic
    gamma = 0.0
    n_trajectories = 2000
    cutoff = 4.0

    n_save_flat = dist_result_sampled.xs.shape[-1]
    sample_idx = rng.choice(n_save_flat, size=n_trajectories)

    x0 = dist_result_sampled.xs[0, :, sample_idx]
    p0 = dist_result_sampled.ps[0, :, sample_idx]

    initial_conditions = InitialConditions(x0=x0, p0=p0)

    ballistic_params = SimulationParameters(
        system=dist_params.system.with_gamma(gamma=gamma),
        time_span=TimeSpan(t0=0, t1=10, dt=0.01),
        initial_conditions=initial_conditions,
    )

    ballistic_result = solve_ensemble.load_or_call_cached(
        params=ballistic_params, _key=key
    )

    delta_k_magnitude = 0.1 * 2 * np.pi
    direction = np.array([1, 1])
    unit_direction = dir
    delta_k = delta_k_magnitude * direction / np.linalg.norm(direction)

    fig_isf, ax = get_fancy_figure()
    _, ax, line_ballistic, fill = plot_isf(
        result=ballistic_result,
        ax=ax,
        config=IsfConfig(delta_k=delta_k, pairwise=False),
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
