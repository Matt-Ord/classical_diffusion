"""The classical ISF of lithium on Cu(111) in 1D, from the Kramers equation, from zero friction to the overdamped limit."""

import numpy as np

from classical_diffusion.kramers import (
    get_diffusion_constant,
    get_diffusion_constant_overdamped,
    solve_isf,
    solve_isf_overdamped,
)
from classical_diffusion.langevin import LITHIUM_COPPER_SYSTEM_1D, PeriodicSystem
from classical_diffusion.plot import get_fancy_figure


def _with_friction(system: PeriodicSystem, gamma: float) -> PeriodicSystem:
    return PeriodicSystem(
        gamma=gamma,
        temperature=system.temperature,
        m=system.m,
        lattice_vectors=system.lattice_vectors,
        terms=system.terms,
        units=system.units,
    )


def main() -> None:
    """Plot the ISF at a quarter of the first reciprocal lattice vector."""
    # The built-in systems are in SI units
    system = LITHIUM_COPPER_SYSTEM_1D
    spacing = float(np.linalg.norm(system.lattice_vectors))
    delta_k = (0.25 * 2 * np.pi / spacing,)
    times = np.linspace(0, 20e-12, 401)

    fig, ax = get_fancy_figure()
    for factor in (0.0, 0.1, 1.0, 10.0):
        frictional = _with_friction(system, factor * system.gamma)
        result = solve_isf(frictional, delta_k, times)
        (line,) = ax.plot(times * 1e12, result.isf.real)
        line.set_label(rf"Kramers, $\gamma = {factor:g}\,\gamma_0$")
        if factor > 0:
            diffusion = get_diffusion_constant(frictional)
            print(f"gamma = {factor:g} gamma_0: D = {diffusion:.6e} m^2/s")
    overdamped = _with_friction(system, 10 * system.gamma)
    result = solve_isf_overdamped(overdamped, delta_k, times)
    (line,) = ax.plot(times * 1e12, result.isf.real, linestyle="--")
    line.set_label(r"Smoluchowski, $\gamma = 10\,\gamma_0$")
    print(
        "overdamped at 10 gamma_0: "
        f"D = {get_diffusion_constant_overdamped(overdamped):.6e} m^2/s"
    )
    ax.set_xlabel("Time / ps")
    ax.set_ylabel(r"$I(\Delta K, t)$")
    ax.set_xlim(0, times[-1] * 1e12)
    ax.legend()
    fig.savefig("examples/kramers.isf.pdf")


if __name__ == "__main__":
    main()
