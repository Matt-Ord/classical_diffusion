import numpy as np
import pytest
import sympy as sp

from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
    XENON_PLATINUM_SYSTEM_2D,
    PeriodicSystem1D,
    PeriodicSystemFCC,
    System,
)
from classical_diffusion.langevin._langevin import (  # ruff: ignore[import-private-name]
    _get_langevin_units,
    _get_max_force,
)


def test_units() -> None:
    system = PeriodicSystem1D(
        gamma=4e12,
        temperature=110,
        m=8e-27,
        delta_x=3e-10,
        barrier_energy=1.6e-22,
    )

    units = _get_langevin_units(system)
    normalized_system = system.with_units(units)

    si_system = normalized_system.with_si_units()

    for field in ("gamma", "temperature", "m", "delta_x", "barrier_energy"):
        np.testing.assert_allclose(
            getattr(si_system, field), getattr(system, field), rtol=1e-3, err_msg=field
        )


@pytest.mark.parametrize(
    "system",
    [
        SODIUM_COPPER_SYSTEM_1D,
        SODIUM_COPPER_SYSTEM_2D,
        XENON_PLATINUM_SYSTEM_2D,
        PeriodicSystemFCC(
            gamma=1e12,
            temperature=155,
            m=3.8e-26,
            delta_x=2.5e-10,
            barrier_energy=4e-21,
        ),
    ],
)
def test_max_force(system: System) -> None:
    param_map = dict(zip(system.parameter_symbols, system.params, strict=False))
    force_fn = sp.lambdify(
        system.coordinate_symbols, sp.Matrix(system.force_expr).subs(param_map)
    )
    rng = np.random.default_rng(0)
    points = rng.random((system.n_dim, 20000)) * 2 * system.delta_x  # ty: ignore[unresolved-attribute]
    forces = np.array(force_fn(*points), dtype=float).reshape(system.n_dim, -1)
    sampled = np.max(np.linalg.norm(forces, axis=0))

    max_force = _get_max_force(system)
    assert max_force >= sampled * (1 - 1e-6)
    assert max_force <= sampled * 1.01
