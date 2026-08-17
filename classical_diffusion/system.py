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

    @property
    def time_factor(self) -> float:
        """The time factor for the unit system."""
        return np.sqrt(self.atomic_mass * self.angstrom**2 / self.Boltzmann)

    @overload
    def time_into(self, value: float, units: UnitSystem | None = None) -> float: ...
    @overload
    def time_into[DT: np.dtype[np.number]](
        self, value: np.ndarray[Any, DT], units: UnitSystem | None = None
    ) -> np.ndarray[Any, DT]: ...
    def time_into(
        self,
        value: float | np.ndarray[Any, np.dtype[np.number]],
        units: UnitSystem | None = None,
    ) -> float | np.ndarray[Any, np.dtype[np.number]]:
        """Convert a time from SI units to the system's units."""
        units = units or UnitSystem.si()
        time_factor = self.time_factor / units.time_factor
        return value * time_factor

    @property
    def mass_factor(self) -> float:
        """The mass factor of the system."""
        return self.atomic_mass / atomic_mass_si

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
        mass_factor = self.mass_factor / units.mass_factor
        return value * mass_factor

    def as_canonical(self) -> CanonicalUnitSystem:
        """Get a jax-compatible unit system."""
        return CanonicalUnitSystem(
            Boltzmann=self.Boltzmann,
            atomic_mass=self.atomic_mass,
            angstrom=self.angstrom,
        )

    @property
    def length_factor(self) -> float:
        """The length factor of the system."""
        return self.angstrom / angstrom_si

    def length_into(
        self, value: float | np.ndarray[Any, np.dtype[np.number]], units: UnitSystem
    ) -> float | np.ndarray[Any, np.dtype[np.number]]:
        """Convert a length from SI units to the system's units."""
        length_factor = self.length_factor / units.length_factor
        return value * length_factor

    @property
    def energy_factor(self) -> float:
        """The energy factor of the system."""
        return self.Boltzmann / Boltzmann_si

    def energy_into(
        self, value: float | np.ndarray[Any, np.dtype[np.number]], units: UnitSystem
    ) -> float | np.ndarray[Any, np.dtype[np.number]]:
        """Convert an energy from SI units to the system's units."""
        energy_factor = self.energy_factor / units.energy_factor
        return value * energy_factor


@jax.tree_util.register_dataclass
@final
@dataclass(frozen=True, kw_only=True)
class CanonicalUnitSystem(UnitSystem):
    """A jax-compatible unit system."""
