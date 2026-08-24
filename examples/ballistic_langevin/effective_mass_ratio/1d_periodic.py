from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.random as jrandom
import numpy as np
from scipy.constants import Boltzmann, hbar
from scipy.integrate import quad
from scipy.special import ellipk
from tqdm import tqdm

from classical_diffusion.langevin import (
    PeriodicSystem1D,
    get_effective_mass,
    get_under_barrier_probability,
    solve_ballistic_ensemble,
)
from classical_diffusion.plot import get_fancy_figure, get_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import cached, disabled_timing, hash_array

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

system = PeriodicSystem1D(
    gamma=4e11,
    temperature=100,
    m=6e-27,
    delta_x=1.48e-10,
    barrier_energy=4e-21,
)
key = jrandom.PRNGKey(100)


def _get_free_effective_mass_exact_1d_periodic_directly(
    system: PeriodicSystem1D,
) -> float:
    u0 = system.barrier_energy / (2 * system.kbt)

    def integrand_denominator(epsilon: float) -> float:
        return np.sqrt(epsilon) / ellipk(1 / epsilon) * np.exp(-2 * u0 * epsilon)

    def integrand_running(epsilon: float) -> float:
        return (
            2 * (1 / np.sqrt(epsilon)) * ellipk(1 / epsilon) * np.exp(-2 * u0 * epsilon)
        )

    denominator_integral, _ = quad(integrand_denominator, 1, np.inf)
    running_integral, _ = quad(integrand_running, 1, np.inf)

    partition = running_integral

    return system.m * partition / (denominator_integral * 2 * u0 * np.pi**2)


def _get_full_effective_mass_exact_1d_periodic_directly(
    system: PeriodicSystem1D,
) -> float:
    u0 = system.barrier_energy / (2 * system.kbt)

    def integrand_denominator(epsilon: float) -> float:
        return np.sqrt(epsilon) / ellipk(1 / epsilon) * np.exp(-2 * u0 * epsilon)

    def integrand_trapped(epsilon: float) -> float:
        return ellipk(epsilon) * np.exp(-2 * u0 * epsilon)

    def integrand_running(epsilon: float) -> float:
        return (
            2 * (1 / np.sqrt(epsilon)) * ellipk(1 / epsilon) * np.exp(-2 * u0 * epsilon)
        )

    denominator_integral, _ = quad(integrand_denominator, 1, np.inf)
    trapped_integral, _ = quad(integrand_trapped, 0, 1)
    running_integral, _ = quad(integrand_running, 1, np.inf)

    partition = 2 * trapped_integral + running_integral

    return system.m * partition / (denominator_integral * 2 * u0 * np.pi**2)


def plot_2d_gradient(
    x_values: np.ndarray[Any, np.dtype[np.floating]],
    y_values: np.ndarray[Any, np.dtype[np.floating]],
    z_values: np.ndarray[Any, np.dtype[np.floating]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot the effective mass against inertial mass and barrier energy."""
    fig, ax = get_figure(ax)

    mesh = ax.pcolormesh(
        x_values,
        y_values,
        z_values,
        shading="auto",
        cmap="viridis",
    )

    return fig, ax, mesh


def plot_effective_mass_ratio(
    barrier_energy: np.ndarray[Any, np.dtype[np.floating[Any]]],
    mass_ratio: np.ndarray[Any, np.dtype[np.floating[Any]]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the ratio of effective mass to inertial mass against barrier energy."""
    fig, ax = get_figure(ax)

    (line,) = ax.plot(barrier_energy, mass_ratio[0])
    line.set_label("Effective mass ratio")

    ax.set_title("Effective Mass Ratio vs Barrier Energy")
    ax.set_xlabel("Barrier Energy")
    ax.set_ylabel(r"$m_{\mathrm{eff}} / m$")  # cspell: disable-line
    ax.legend()

    return fig, ax, line


def _plot_effective_mass_ratio() -> None:  # ruff:ignore[too-many-statements]

    def _solve_effective_mass_path(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
        temperature: float,
        delta_x: float,
        end_time: float,
        n_samples: int,
        _key: jax.Array,
        barrier_energy: np.ndarray,
        inertial_mass: np.ndarray,
    ) -> Path:
        filename = f"{temperature}_{delta_x}_{end_time}_{n_samples}_{key}_{hash_array((barrier_energy,))}_{hash_array((inertial_mass,))}.npz"
        return Path("examples/data") / filename

    @cached(_solve_effective_mass_path)
    def _effective_mass_simulation(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
        temperature: float,
        delta_x: float,
        end_time: float,
        n_samples: int,
        _key: jax.Array,
        barrier_energy: np.ndarray,
        inertial_mass: np.ndarray,
    ) -> tuple:
        keys = jrandom.split(jrandom.PRNGKey(100), barrier_energy.size)
        barrier_energy, inertial_mass = np.meshgrid(barrier_energy, inertial_mass)
        prob_under_barrier = np.zeros_like(barrier_energy)
        full_effective_mass_ratio = np.zeros_like(barrier_energy)
        free_effective_mass_ratio = np.zeros_like(barrier_energy)
        full_effective_mass_exact_ratio = np.zeros_like(barrier_energy)
        free_effective_mass_exact_ratio = np.zeros_like(barrier_energy)

        kbt = temperature * Boltzmann
        m0 = 4 * (np.pi) ** 2 * (hbar) ** 2 / (kbt * (delta_x) ** 2)

        m_grid = inertial_mass * m0
        barrier_energy_grid = barrier_energy * kbt

        with disabled_timing():
            for idx, (i, j) in enumerate(
                tqdm(np.ndindex(barrier_energy.shape), total=barrier_energy.size)
            ):
                system = PeriodicSystem1D(
                    gamma=0,
                    temperature=temperature,
                    m=m_grid[i, j],
                    delta_x=delta_x,
                    barrier_energy=barrier_energy_grid[i, j],
                )

                result = solve_ballistic_ensemble(
                    system,
                    TimeSpan(
                        t_end=system.units.time_into(end_time),
                        n_steps=1000,
                    ),
                    n_samples=n_samples,
                    minimum_energy=system.barrier_energy,
                    _key=keys[idx],
                )

                prob_under_barrier_val = get_under_barrier_probability(
                    system=system,
                    barrier_energy=system.barrier_energy,
                )
                prob_under_barrier[i, j] = prob_under_barrier_val

                full_effective_mass_ratio[i, j] = get_effective_mass(
                    result,
                    under_barrier_probability=prob_under_barrier_val,
                ).item()

                free_effective_mass_ratio[i, j] = get_effective_mass(result).item()

                full_effective_mass_exact_ratio[i, j] = (
                    _get_full_effective_mass_exact_1d_periodic_directly(system)
                )

                free_effective_mass_exact_ratio[i, j] = (
                    _get_free_effective_mass_exact_1d_periodic_directly(system)
                )

            return (
                prob_under_barrier,
                full_effective_mass_ratio,
                free_effective_mass_ratio,
                full_effective_mass_exact_ratio,
                free_effective_mass_exact_ratio,
            )

    barrier_energy = np.linspace(1, 4, 20)
    inertial_mass = np.linspace(1, 20, 20)
    (
        _prob_under_barrier,
        full_effective_mass_ratio,
        free_effective_mass_ratio,
        full_effective_mass_exact_ratio,
        free_effective_mass_exact_ratio,
    ) = _effective_mass_simulation(
        temperature=115,
        delta_x=1.48e-10,
        end_time=2e-12,
        n_samples=50,
        _key=key,
        barrier_energy=barrier_energy,
        inertial_mass=inertial_mass,
    )

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_2d_gradient(
        x_values=barrier_energy,
        y_values=inertial_mass,
        z_values=full_effective_mass_ratio,
        ax=ax,
    )
    fig.colorbar(mesh, ax=ax, label="effective mass ratio")
    ax.set_xlabel("Dimensionless Barrier energy")
    ax.set_ylabel("Dimenesionless Inertial mass")
    mesh.set_rasterized(True)
    fig.savefig(
        "examples/ballistic_langevin/effective_mass_ratio/1d_periodic.2d_plot.full.simulation.pdf",
        dpi=1000,
    )

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_2d_gradient(
        x_values=barrier_energy,
        y_values=inertial_mass,
        z_values=free_effective_mass_ratio,
        ax=ax,
    )
    fig.colorbar(mesh, ax=ax, label="effective mass ratio")
    ax.set_xlabel("Dimensionless Barrier energy")
    ax.set_ylabel("Dimenesionless Inertial mass")
    mesh.set_rasterized(True)
    fig.savefig(
        "examples/ballistic_langevin/effective_mass_ratio/1d_periodic.2d_plot.free.simulation.pdf",
        dpi=1000,
    )

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_2d_gradient(
        x_values=barrier_energy,
        y_values=inertial_mass,
        z_values=free_effective_mass_exact_ratio,
        ax=ax,
    )
    fig.colorbar(mesh, ax=ax, label="effective mass ratio")
    ax.set_xlabel("Dimensionless Barrier energy")
    ax.set_ylabel("Dimenesionless Inertial mass")
    mesh.set_rasterized(True)
    fig.savefig(
        "examples/ballistic_langevin/effective_mass_ratio/1d_periodic.effective_mass.2d_plot.free.exact.pdf",
        dpi=1000,
    )

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_2d_gradient(
        x_values=barrier_energy,
        y_values=inertial_mass,
        z_values=full_effective_mass_exact_ratio,
        ax=ax,
    )
    fig.colorbar(mesh, ax=ax, label="effective mass ratio")
    ax.set_xlabel("Dimensionless Barrier energy")
    ax.set_ylabel("Dimenesionless Inertial mass")
    mesh.set_rasterized(True)
    fig.savefig(
        "examples/ballistic_langevin/effective_mass_ratio/1d_periodic.effective_mass.2d_plot.full.exact.pdf",
        dpi=1000,
    )

    fig, ax = get_fancy_figure()
    _, ax, _line = plot_effective_mass_ratio(
        barrier_energy=barrier_energy,
        mass_ratio=free_effective_mass_exact_ratio,
        ax=ax,
    )
    fig.savefig(
        "examples/ballistic_langevin/effective_mass_ratio/1d_periodic.effective_mass.free.1d_plot.exact.pdf",
        dpi=1000,
    )

    fig, ax = get_fancy_figure()
    _, ax, _ = plot_effective_mass_ratio(
        barrier_energy=barrier_energy,
        mass_ratio=full_effective_mass_exact_ratio,
        ax=ax,
    )
    fig.savefig(
        "examples/ballistic_langevin/effective_mass_ratio/1d_periodic.effective_mass.full.1d_plot.exact.pdf",
        dpi=1000,
    )

    full_error = (
        abs(full_effective_mass_exact_ratio - full_effective_mass_ratio)
        / full_effective_mass_exact_ratio
    )
    free_error = (
        abs(free_effective_mass_exact_ratio - free_effective_mass_ratio)
        / free_effective_mass_exact_ratio
    )

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_2d_gradient(
        x_values=barrier_energy,
        y_values=inertial_mass,
        z_values=full_error,
        ax=ax,
    )
    fig.colorbar(mesh, ax=ax, label="error")
    ax.set_xlabel("Dimensionless Barrier energy")
    ax.set_ylabel("Dimenesionless Inertial mass")
    mesh.set_rasterized(True)
    fig.savefig(
        "examples/ballistic_langevin/effective_mass_ratio/1d_periodic.effective_mass.2d_plot.free.error.pdf",
        dpi=1000,
    )

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_2d_gradient(
        x_values=barrier_energy,
        y_values=inertial_mass,
        z_values=free_error,
        ax=ax,
    )
    fig.colorbar(mesh, ax=ax, label="error")
    ax.set_xlabel("Dimensionless Barrier energy")
    ax.set_ylabel("Dimenesionless Inertial mass")
    mesh.set_rasterized(True)
    fig.savefig(
        "examples/ballistic_langevin/effective_mass_ratio/1d_periodic.effective_mass.2d_plot.full.error.pdf",
        dpi=1000,
    )


if __name__ == "__main__":
    _plot_effective_mass_ratio()
