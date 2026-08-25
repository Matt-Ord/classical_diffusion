from dataclasses import field
from typing import final

import jax

from classical_diffusion.hopping import KramersParameters
from classical_diffusion.langevin._system import (
    CanonicalSystem,
    System,
    _build_kramers_potential,
)
from classical_diffusion.system import CanonicalUnitSystem


@jax.tree_util.register_dataclass
@final
class KramersParametersJax(KramersParameters):
    barrier_energy: float | jax.Array
    units: CanonicalUnitSystem = field(default_factory=CanonicalUnitSystem)


class KramersSystem1D(System):
    """A one-dimensional Kramers system."""

    def __new__(cls, *, params: KramersParameters, n_dim: int = 1) -> CanonicalSystem:
        return CanonicalSystem(
            gamma=params.gamma,
            temperature=params.temperature,
            m=params.m,
            potential=(n_dim, _build_kramers_potential()),
            params=(params.omega_well, params.omega_barrier, params.barrier_energy),
            units=params.units.as_canonical(),
        )
