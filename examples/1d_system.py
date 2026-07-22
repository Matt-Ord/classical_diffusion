from dataclasses import dataclass

import jax.random as jrandom
import numpy as np
from classical_diffusion.langevin.analysis import (
    breakdown_ballistic_trajectory,
    get_effective_mass,
    plot_isf_with_delta_k,
    split_escaped_and_trapped,
    under_barrier_probability_ballistic,
)
from classical_diffusion.system.analysis import (
    plot_exact_gaussian_isf,
    plot_exact_offset_gaussian_isf,
)

from classical_diffusion.langevin import (
    TimeSpan,
    plot_isf,
    solve_ballistic_ensemble,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import (
    PeriodicSystem1D,
    plot_periodic_potential_1d,
)
from classical_diffusion.system._system import UnitSystem


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
CAM_CHERRY = CamColor(
    light="#F2CAD8",
    warm="#E18AAC",
    base="#CD3572",
    dark="#911449",
)


def _plot_periodic_system() -> None:
    system = PeriodicSystem1D(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        delta_x=5,
        barrier_energy=0.5,
        units=UnitSystem(),
    )
    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_1d(system, ax=ax)
    fig.savefig("examples/1d_system.potential.pdf")


def _plot_1d_periodic_isf() -> None:  # ruff:ignore[too-many-locals]
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        delta_x=5,
        barrier_energy=0.5,
        units=UnitSystem(),
    )

    result = solve_ensemble(
        system,
        TimeSpan(
            t0=0,
            t1=40 / system.gamma,
            dt=0.01 / system.gamma,
            dt_step=0.01 / system.gamma,
        ),
        (np.full((20, 1), 0.0), np.full((20, 1), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()

    delta_k = (0.7 * 2 * np.pi / system.delta_x,)
    _, ax, line_0, fill_0 = plot_isf(
        result=result,
        ax=ax,
        delta_k=delta_k,
    )
    line_0.set_label("full simulation")
    line_0.set_color(CAM_BLUE.dark)
    fill_0.set_color(CAM_BLUE.dark)

    result = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t0=0,
            t1=10 / system.gamma,
            dt=0.01 / system.gamma,
            dt_step=0.01 / system.gamma,
        ),
        n_trajectories=2000,
        _key=key,
    )

    _, ax, line_1, fill_1 = plot_isf(
        result=result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_1.set_label("ballistic simulation")
    line_1.set_color(CAM_BLUE.warm)
    fill_1.set_color(CAM_BLUE.warm)

    elastic_result, inelastic_result = breakdown_ballistic_trajectory(
        result, stride_time=1 / system.gamma
    )

    _, ax, line_2, fill_2 = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_2.set_label("elastic")
    line_2.set_linestyle(":")
    line_2.set_color(CAM_CHERRY.light)
    fill_2.set_color(CAM_CHERRY.light)

    _, ax, line_3, fill_3 = plot_isf(
        result=inelastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_3.set_label("inelastic")
    line_3.set_linestyle(":")
    line_3.set_color(CAM_CHERRY.dark)
    fill_3.set_color(CAM_CHERRY.dark)

    ax.set_xlim(0, 4 / system.gamma)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig("./examples/1d_system.isf.pdf", dpi=300, bbox_inches="tight")

    delta_k_values = np.linspace(0.1, 2.0, 9) * (0.5 * 2 * np.pi / system.delta_x)

    fig, ax = get_fancy_figure()
    _, ax = plot_isf_with_delta_k(
        result=inelastic_result, ax=ax, delta_k_values=delta_k_values, pairwise=False
    )
    ax.set_xlim(0, 4 / system.gamma)
    fig.savefig(
        "./examples/1d_system.isf.ballsitic.inelastic.delta_k.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _plot_effective_mass_isf() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        delta_x=5,
        barrier_energy=0.5,
        units=UnitSystem(),
    )

    fig, ax = get_fancy_figure()

    delta_k = (0.5 * 2 * np.pi / system.delta_x,)

    result = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t0=0,
            t1=100 / system.gamma,
            dt=0.01 / system.gamma,
            dt_step=0.01 / system.gamma,
        ),
        n_trajectories=1,
        _key=key,
    )

    elastic_result, _ = breakdown_ballistic_trajectory(
        result, stride_time=1 / system.gamma
    )

    _, ax, line_0, _ = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")

    _, ax, line_1 = plot_exact_gaussian_isf(
        system=system, ax=ax, delta_k=delta_k, effective_mass=system.m
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")
    line_1.set_color(CAM_BLUE.dark)

    effective_mass = get_effective_mass(result)

    _, ax, line_2 = plot_exact_gaussian_isf(
        system=system, ax=ax, delta_k=delta_k, effective_mass=effective_mass
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")
    line_2.set_color(CAM_CHERRY.dark)

    ax.set_xlim(0, 0.3 / system.gamma)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1, line_2])
    fig.savefig(
        "./examples/1d_system.isf.effective.mass.pdf", dpi=300, bbox_inches="tight"
    )


def _plot_effective_mass_offset_isf() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        delta_x=5,
        barrier_energy=0.5,
        units=UnitSystem(),
    )

    fig, ax = get_fancy_figure()

    result = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t0=0,
            t1=100 / system.gamma,
            dt=0.01 / system.gamma,
            dt_step=0.01 / system.gamma,
        ),
        n_trajectories=1,
        _key=key,
    )

    elastic_result, _ = breakdown_ballistic_trajectory(
        result, stride_time=1 / system.gamma
    )

    delta_k = (0.5 * 2 * np.pi / system.delta_x,)

    _, ax, line_0, _ = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")

    _, ax, line_1 = plot_exact_offset_gaussian_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        effective_mass=system.m,
        offset=np.average(under_barrier_probability_ballistic(result)),
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")
    line_1.set_color(CAM_BLUE.dark)

    free_result, _ = split_escaped_and_trapped(result)

    effective_mass = get_effective_mass(free_result)

    _, ax, line_2 = plot_exact_offset_gaussian_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        effective_mass=effective_mass,
        offset=np.average(under_barrier_probability_ballistic(result)),
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")
    line_2.set_color(CAM_CHERRY.dark)

    ax.set_xlim(0, 1.0 / system.gamma)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1, line_2])
    fig.savefig(
        "./examples/1d_system.isf.effective.mass.offset.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _plot_periodic_system()
    _plot_1d_periodic_isf()
    _plot_effective_mass_isf()
    _plot_effective_mass_offset_isf()
