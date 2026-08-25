import numpy as np

from classical_diffusion.langevin import (
    PeriodicSystem1D,
)
from classical_diffusion.langevin._langevin import (  # ruff: ignore[import-private-name]
    _get_langevin_units,
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
