import itertools
from typing import TYPE_CHECKING

import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from classical_diffusion.analysis import plot_isf
from classical_diffusion.langevin import (
    PeriodicSystemFCC,
    SingleLangevinSimulationResult,
    System,
    plot_2d_trajectory,
    plot_periodic_potential_fcc,
    solve_ballistic_ensemble,
    solve_ensemble,
    solve_single,
)
from classical_diffusion.plot import (
    get_fancy_figure,
    setup_fancy_figure,
    setup_rc_params,
)
from classical_diffusion.simulation import TimeSpan

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _plot_periodic_system() -> None:
    system = PeriodicSystemFCC(
        gamma=0.1, temperature=1.0, m=1.0, delta_x=5.0, barrier_energy=1.5
    )
    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_fcc(system, ax=ax)
    fig.savefig("examples/2d_system.potential.pdf")


def _plot_2d_periodic_isf() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystemFCC(
        gamma=0.1, temperature=0.5, m=1.0, delta_x=5, barrier_energy=1.5
    )

    result = solve_ensemble(
        system,
        TimeSpan(
            t_end=50 / system.gamma,
            n_steps=5000,
        ),
        (np.full((2000, 2), 0.0), np.full((2000, 2), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()

    delta_k = (0.5 * 2 * np.pi / system.delta_x,)
    _, ax, line_0, _ = plot_isf(
        result=result,
        ax=ax,
        delta_k=delta_k,
    )
    line_0.set_label("simulation")

    result = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t_end=4 / system.gamma,
            n_steps=400,
        ),
        n_samples=2000,
        _key=key,
    )
    _, ax, line_1, _ = plot_isf(result=result, ax=ax, delta_k=delta_k, pairwise=False)
    line_1.set_label("ballistic simulation")

    ax.set_xlim(0, 4 / system.gamma)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1])
    fig.savefig("./examples/2d_system.isf.pdf", dpi=300, bbox_inches="tight")


def _plot_2d_trajectory() -> None:
    # TODO: add elastic and inelastic trajectories to the plot
    key = jrandom.PRNGKey(100)
    system = PeriodicSystemFCC(
        gamma=0.1, temperature=0.5, m=1.0, delta_x=5, barrier_energy=1.5
    )

    result = solve_single(
        system,
        TimeSpan(
            t_end=100 / system.gamma,
            n_steps=10000,
        ),
        (np.full((2,), 0.0), np.full((2,), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()
    _, ax, _line = plot_2d_trajectory(result=result, ax=ax)

    fig.savefig("examples/2d_system.trajectory.pdf")


def _get_two_panel_figure() -> tuple[Figure, list[Axes]]:
    setup_rc_params()
    fig, ax = plt.subplots(layout="constrained", ncols=2, figsize=(6, 2.5))
    setup_fancy_figure(fig, ax)
    return fig, ax


MIN_SPLIT_POINTS = 4
MIN_VARIANCE = 1e-15


def filter_trajectory_kv(
    x: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    *,
    min_split_points: int = MIN_SPLIT_POINTS,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Discretize a 1D or 2D signal using the objective Kalafut-Visscher step detection algorithm."""
    x = x[:, np.newaxis] if x.ndim == 1 else x

    n_samples = x.shape[0]
    breakpoints = [0, n_samples]
    stack = [(0, n_samples)]

    while stack:
        start, end = stack.pop()
        n_seg = end - start

        if n_seg <= min_split_points:
            continue

        segment = x[start:end]
        total_sum = np.sum(segment, axis=0)  # Shape (D,)
        total_sum_x2 = np.sum(segment**2)  # Scalar sum of all elements squared

        # Calculate local baseline variance across dimensions before splitting
        base_mu = total_sum / n_seg  # Shape (D,)
        base_ssr = (
            total_sum_x2 - 2 * np.sum(base_mu * total_sum) + n_seg * np.sum(base_mu**2)
        )
        base_variance = base_ssr / n_seg
        if base_variance <= MIN_VARIANCE:
            continue

        indices = np.arange(2, n_seg - 1)
        n_right_arr = n_seg - indices

        # Vectorized segment cumulative sums across time for all dimensions: shape (K, D)
        cum_sum_left = np.cumsum(segment, axis=0)[indices - 1]
        cum_sum_right = total_sum - cum_sum_left

        # Mean vectors for left and right partitions: shape (K, D)
        mu_left = cum_sum_left / indices[:, None]
        mu_right = cum_sum_right / n_right_arr[:, None]

        # Vectorized sum of squared residuals reduced across spatial dimensions (axis=1)
        split_ssr = (
            total_sum_x2
            - 2 * np.sum(mu_left * cum_sum_left, axis=1)
            + indices * np.sum(mu_left**2, axis=1)
            - 2 * np.sum(mu_right * cum_sum_right, axis=1)
            + n_right_arr * np.sum(mu_right**2, axis=1)
        )
        split_variances = split_ssr / n_seg
        split_variances = np.maximum(split_variances, MIN_VARIANCE)

        # Objective Criterion: delta_sic > 0 means the split is justified by the data
        delta_sic = n_seg * np.log(base_variance / split_variances) - np.log(n_seg)
        best_idx = np.argmax(delta_sic)

        if delta_sic[best_idx] > 0:
            global_split_point = start + indices[best_idx]
            breakpoints.append(global_split_point)
            stack.extend([(start, global_split_point), (global_split_point, end)])

    # Reconstruct the final de-noised piece-wise constant path
    breakpoints = sorted(set(breakpoints))
    fitted_trajectory = np.zeros_like(x, dtype=np.float64)

    for b_start, b_end in itertools.pairwise(breakpoints):
        fitted_trajectory[b_start:b_end] = np.mean(x[b_start:b_end], axis=0)

    return fitted_trajectory


def breakdown_filtered_ballistic_trajectory[S: System](
    result: SingleLangevinSimulationResult[S], *, minimum_timescale: float = 0
) -> tuple[
    SingleLangevinSimulationResult[S],
    SingleLangevinSimulationResult[S],
]:
    """Split a ballistic simulation into its elastic and inelastic components across all dimensions."""
    dt = result.times[1] - result.times[0]
    min_split_points = max(4, int(minimum_timescale / dt))

    p_points = result.p_points
    p_elastic_points = filter_trajectory_kv(
        p_points.T, min_split_points=min_split_points
    ).T

    # Initial positions x_0: shape (n_trajectories, n_dimensions, 1)
    x_0 = result.x_points[..., :1]

    # Calculate x_elastic(t) = x_0 + (p_elastic / m) * t across all spatial components
    dt = result.times[1] - result.times[0]
    displacements = (p_elastic_points / result.system.m) * dt

    # Prepend zero at t=0 so initial position x_elastic(t_0) == x_0
    step_displacements = np.concatenate(
        [np.zeros_like(x_0), displacements[..., :-1]], axis=-1
    )
    x_elastic_points = x_0 + np.cumsum(step_displacements, axis=-1)

    elastic = SingleLangevinSimulationResult(
        times=result.times,
        x_points=x_elastic_points,
        p_points=p_elastic_points,
        system=result.system,
    )

    inelastic = SingleLangevinSimulationResult(
        times=result.times,
        x_points=result.x_points - x_elastic_points,
        p_points=result.p_points - p_elastic_points,
        system=result.system,
    )
    return elastic, inelastic


def breakdown_filtered_ballistic_trajectory_butterworth[S: System](
    result: SingleLangevinSimulationResult[S], *, minimum_timescale: float = 0
) -> tuple[
    SingleLangevinSimulationResult[S],
    SingleLangevinSimulationResult[S],
]:
    """Split a ballistic simulation into its elastic (slow) and inelastic (fast) components."""
    times = result.times
    dt = times[1] - times[0]

    # Changes slower than minimum_timescale correspond to frequencies f < 1 / minimum_timescale.
    # High frequencies are filtered out to yield the elastic (slow) component.
    fs = 1.0 / dt
    cutoff_freq = 1.0 / max(minimum_timescale, 1e-5 * dt)
    nyquist = 0.5 * fs

    if cutoff_freq < nyquist:
        sos = butter(N=4, Wn=cutoff_freq / nyquist, btype="low", output="sos")

        # Low-pass filter both momentum and position along the time axis (axis=-1)
        # cspell: disable-next-line  # ruff: ignore[commented-out-code]
        p_elastic_points = sosfiltfilt(sos, result.p_points, axis=-1)
        # Since the filter is a linear operation, it commutes with integration
        # So, filtering the position is equivalent to integrating the filtered momentum
        # cspell: disable-next-line  # ruff: ignore[commented-out-code]
        x_elastic_points = sosfiltfilt(sos, result.x_points, axis=-1)
    else:
        p_elastic_points = result.p_points.copy()
        x_elastic_points = result.x_points.copy()

    elastic = SingleLangevinSimulationResult(
        times=result.times,
        x_points=x_elastic_points,
        p_points=p_elastic_points,
        system=result.system,
    )

    inelastic = SingleLangevinSimulationResult(
        times=result.times,
        x_points=result.x_points - x_elastic_points,
        p_points=result.p_points - p_elastic_points,
        system=result.system,
    )
    return elastic, inelastic


def _plot_2d_ballistic_trajectory() -> None:

    key = jrandom.PRNGKey(100)
    system = PeriodicSystemFCC(
        gamma=0, temperature=0.5, m=1.0, delta_x=5, barrier_energy=1.5
    )

    result = solve_single(
        system,
        TimeSpan(t_end=1000, n_steps=1000),
        # TODO: replace with random "free" initial condition?
        (np.full((2,), 0.0), np.full((2,), 1.0)),
        _key=key,
    )
    elastic, inelastic = breakdown_filtered_ballistic_trajectory_butterworth(
        result, minimum_timescale=2 / 0.1
    )

    fig, ax = _get_two_panel_figure()

    _, _ax_0, line = plot_2d_trajectory(result=result, ax=ax[0])
    _, _ax_0, line_e = plot_2d_trajectory(result=elastic, ax=ax[0])

    _, _ax_1, line_i = plot_2d_trajectory(result=inelastic, ax=ax[1])
    line_i.set_color("C2")

    ax[0].legend(
        handles=[line, line_e],
        labels=["full", "elastic"],
    )
    ax[1].legend(
        handles=[line_i],
        labels=["inelastic"],
    )

    fig.savefig("examples/2d_system.ballistic_trajectory.pdf")


if __name__ == "__main__":
    # _plot_periodic_system()
    # _plot_2d_periodic_isf()
    # _plot_2d_trajectory()
    _plot_2d_ballistic_trajectory()
