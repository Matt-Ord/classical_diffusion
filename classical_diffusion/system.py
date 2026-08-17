from dataclasses import dataclass
from typing import Any, final, overload

import jax
import numpy as np
from scipy.constants import (
    Boltzmann as Boltzmann_si,
)
from scipy.constants import (
    angstrom as angstrom_si,
)
from scipy.constants import (
    atomic_mass as atomic_mass_si,
)


@dataclass(frozen=True, kw_only=True)
class UnitSystem:
    """Defines the units used in the scattering calculation."""

    Boltzmann: float = Boltzmann_si
    atomic_mass: float = atomic_mass_si
    angstrom: float = angstrom_si

    @classmethod
    def si(cls) -> UnitSystem:
        """Get the SI units."""
        return cls()

    @overload
    def time_into(self, value: float, units: UnitSystem) -> float: ...
    @overload
    def time_into[DT: np.dtype[np.number]](
        self, value: np.ndarray[Any, DT], units: UnitSystem
    ) -> np.ndarray[Any, DT]: ...
    def time_into(
        self, value: float | np.ndarray[Any, np.dtype[np.number]], units: UnitSystem
    ) -> float | np.ndarray[Any, np.dtype[np.number]]:
        """Convert a time from SI units to the system's units."""
        length_factor = self.angstrom / units.angstrom
        mass_factor = self.atomic_mass / units.atomic_mass
        energy_factor = self.Boltzmann / units.Boltzmann
        time_factor = np.sqrt(length_factor**2 * mass_factor / energy_factor)
        return value * time_factor

    @overload
    def mass_into(self, value: float, units: UnitSystem) -> float: ...
    @overload
    def mass_into[DT: np.dtype[np.number]](
        self, value: np.ndarray[Any, DT], units: UnitSystem
    ) -> np.ndarray[Any, DT]: ...
    def mass_into(
        self, value: float | np.ndarray[Any, np.dtype[np.number]], units: UnitSystem
    ) -> float | np.ndarray[Any, np.dtype[np.number]]:
        """Convert a time from SI units to the system's units."""
        mass_factor = self.atomic_mass / units.atomic_mass
        return value * mass_factor

    def as_canonical(self) -> CanonicalUnitSystem:
        """Get a jax-compatible unit system."""
        return CanonicalUnitSystem(
            Boltzmann=self.Boltzmann,
            atomic_mass=self.atomic_mass,
            angstrom=self.angstrom,
        )


@jax.tree_util.register_dataclass
@final
@dataclass(frozen=True, kw_only=True)
class CanonicalUnitSystem(UnitSystem):
    """A jax-compatible unit system."""
