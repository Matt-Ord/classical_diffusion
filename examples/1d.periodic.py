import jax.random as jrandom
import numpy as np

from classical_diffusion.langevin import (
    InitialConditions,
    IsfConfig,
    SimulationParameters,
    TimeSpan,
    plot_isf,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import PeriodicSystem1D, plot_periodic_potential_1d


def _plot_periodic_system() -> None:
    system = PeriodicSystem1D(
        gamma=0.1, temperature=1.0, m=1.0, delta_x=5.0, barrier_energy=1.5
    )
    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_1d(system, ax=ax)
    fig.savefig("examples/1d.periodic.potential.pdf")


def _plot_1d_isf() -> None:
    key = jrandom.PRNGKey(100)
    # Full ISF
    n_trajectories = 200

    initial_conditions = InitialConditions(
        x0=rng.choice(dist_result_folded.xs.reshape(-1), size=n_trajectories)[:, None],
        p0=rng.choice(dist_result_folded.ps.reshape(-1), size=n_trajectories)[:, None],
    )

    full_parameters = parameters.with_initial_conditions(initial_conditions)
    full_result = solve_ensemble.load_or_call_cached(params=full_parameters, _key=key)

    time_scale = 1 / parameters.system.gamma
    delta_k = (0.1 * 2 * np.pi / full_result.system.delta_x,)
    full_isf, ax = get_fancy_figure()
    _, ax, line_full, fill = plot_isf(
        result=full_result,
        ax=ax,
        time_scale=time_scale,
        config=IsfConfig(delta_k=delta_k),
    )

    # Ballistic
    key = jrandom.PRNGKey(200)

    n_trajectories = 50_000
    cutoff = 20

    initial_conditions = InitialConditions(
        x0=rng.choice(dist_result_folded.xs.reshape(-1), size=n_trajectories)[:, None],
        p0=rng.choice(dist_result_folded.ps.reshape(-1), size=n_trajectories)[:, None],
    )

    ballistic_params = SimulationParameters(
        system=full_parameters.system.with_gamma(gamma=0.0),
        time_span=TimeSpan(t0=1, t1=cutoff, dt=0.01),
        initial_conditions=initial_conditions,
    )

    ballistic_result = solve_ensemble.load_or_call_cached(
        params=ballistic_params, _key=key
    )

    _, ax, line_ballistic, _fill = plot_isf(
        result=ballistic_result,
        ax=ax,
        config=IsfConfig(delta_k=delta_k, pairwise=False),
        time_scale=time_scale,
    )
    ax.set_xlim(0, cutoff)
    ax.set_ylim(0.7, 1.0)
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
        handles=[line_full, line_ballistic],
        labels=["Full ISF", "Ballistic ISF"],
    )
    full_isf.savefig("examples/1d.periodic.full.isf.pdf")


if __name__ == "__main__":
    # _plot_periodic_system()
    _plot_xp_distributions()
    # _plot_1d_isf()
