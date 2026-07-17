import dataclasses

import jax.random as jrandom
import numpy as np

from classical_diffusion.langevin import (
    InitialConditions,
    IsfConfig,
    SimulationParameters,
    TimeSpan,
    plot_isf,
    plot_p_histogram,
    plot_x_histogram,
    sample_result,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import HarmonicSystem
from classical_diffusion.system._analysis import plot_exact_flat_isf  # noqa: PLC2701

if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    key = jrandom.PRNGKey(100)

    dist_params = SimulationParameters(
        system=HarmonicSystem(omega=0, gamma=0.5, temperature=1.5, m=1),
        time_span=TimeSpan(t0=0, t1=100, dt=0.01),
        initial_conditions=InitialConditions(
            x0=np.array([[0.0]]), p0=np.array([[0.0]])
        ),
    )

    dist_result = solve_ensemble.load_or_call_cached(params=dist_params, _key=key)
    dist_result_sampled = sample_result(
        dist_result, stride_time=1 / dist_params.system.gamma
    )

    fig_p_hist, ax = get_fancy_figure()
    _, ax, bars = plot_p_histogram(result=dist_result_sampled, ax=ax)
    fig_p_hist.savefig("examples/1d.flat.distribution.p.pdf")

    fig_x_hist, ax = get_fancy_figure()
    _, ax, bars = plot_x_histogram(result=dist_result_sampled, ax=ax)
    fig_x_hist.savefig("examples/1d.flat.distribution.x.pdf")

    # Ballistic
    gamma = 0.0
    n_trajectories = 5000

    initial_conditions = InitialConditions(
        x0=rng.choice(dist_result_sampled.xs.reshape(-1), size=n_trajectories)[:, None],
        p0=rng.choice(dist_result_sampled.ps.reshape(-1), size=n_trajectories)[:, None],
    )

    ballistic_params = SimulationParameters(
        system=dataclasses.replace(dist_params.system, gamma=gamma),
        time_span=TimeSpan(t0=0, t1=100, dt=0.01),
        initial_conditions=initial_conditions,
    )

    ballistic_result = solve_ensemble.load_or_call_cached(
        params=ballistic_params, _key=key
    )

    delta_k = (0.1 * 2 * np.pi,)
    fig_isf, ax = get_fancy_figure()
    _, ax, line_ballistic, fill = plot_isf(
        result=ballistic_result,
        ax=ax,
        config=IsfConfig(delta_k=delta_k, pairwise=False),
        time_scale=1 / ballistic_params.system.gamma,
    )

    _, ax, line_exact = plot_exact_flat_isf(
        system=ballistic_params.system,
        delta_k=delta_k,
        ax=ax,
        times=np.arange(
            ballistic_params.time_span.t0, ballistic_params.time_span.t1, 0.1
        ),
    )

    ax.set_xlim(0, ballistic_params.time_span.t1)
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
        handles=[line_ballistic, line_exact],
        labels=["Simulation", "Theory"],
    )
    fig_isf.savefig("examples/1d.flat.isf.pdf")
