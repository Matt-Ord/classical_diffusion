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

    boltzmann: float = Boltzmann_si
    atomic_mass: float = atomic_mass_si
    angstrom: float = angstrom_si

    @classmethod
    def si(cls) -> UnitSystem:
        """Get the SI units."""
        return cls()

    @property
    def time_factor(self) -> float:
        """The time factor for the unit system."""
        return np.sqrt(self.atomic_mass * self.angstrom**2 / self.boltzmann)

    @property
    def mass_factor(self) -> float:
        """The mass factor of the system."""
        return self.atomic_mass / atomic_mass_si

    @property
    def length_factor(self) -> float:
        """The length factor of the system."""
        return self.angstrom / angstrom_si

    @property
    def momentum_factor(self) -> float:
        """The momentum factor of the system."""
        return self.mass_factor * self.length_factor / self.time_factor

    @property
    def energy_factor(self) -> float:
        """The energy factor of the system."""
        return self.boltzmann / Boltzmann_si

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
        """Convert time from self units into target units (defaults to SI)."""
        units = units or UnitSystem.si()
        return value * (units.time_factor / self.time_factor)

    @overload
    def frequency_into(
        self, value: float, units: UnitSystem | None = None
    ) -> float: ...
    @overload
    def frequency_into[DT: np.dtype[np.number]](
        self, value: np.ndarray[Any, DT], units: UnitSystem | None = None
    ) -> np.ndarray[Any, DT]: ...
    def frequency_into(
        self,
        value: float | np.ndarray[Any, np.dtype[np.number]],
        units: UnitSystem | None = None,
    ) -> float | np.ndarray[Any, np.dtype[np.number]]:
        """Convert frequency from self units into target units (defaults to SI)."""
        units = units or UnitSystem.si()
        return value * (self.time_factor / units.time_factor)

    @overload
    def mass_into(self, value: float, units: UnitSystem | None = None) -> float: ...
    @overload
    def mass_into[DT: np.dtype[np.number]](
        self, value: np.ndarray[Any, DT], units: UnitSystem | None = None
    ) -> np.ndarray[Any, DT]: ...
    def mass_into(
        self,
        value: float | np.ndarray[Any, np.dtype[np.number]],
        units: UnitSystem | None = None,
    ) -> float | np.ndarray[Any, np.dtype[np.number]]:
        """Convert mass from self units into target units (defaults to SI)."""
        units = units or UnitSystem.si()
        return value * (units.mass_factor / self.mass_factor)

    @overload
    def length_into(self, value: float, units: UnitSystem | None = None) -> float: ...
    @overload
    def length_into[DT: np.dtype[np.number]](
        self, value: np.ndarray[Any, DT], units: UnitSystem | None = None
    ) -> np.ndarray[Any, DT]: ...
    def length_into(
        self,
        value: float | np.ndarray[Any, np.dtype[np.number]],
        units: UnitSystem | None = None,
    ) -> float | np.ndarray[Any, np.dtype[np.number]]:
        """Convert length from self units into target units (defaults to SI)."""
        units = units or UnitSystem.si()
        return value * (units.length_factor / self.length_factor)

    @overload
    def momentum_into(self, value: float, units: UnitSystem | None = None) -> float: ...
    @overload
    def momentum_into[DT: np.dtype[np.number]](
        self, value: np.ndarray[Any, DT], units: UnitSystem | None = None
    ) -> np.ndarray[Any, DT]: ...
    def momentum_into(
        self,
        value: float | np.ndarray[Any, np.dtype[np.number]],
        units: UnitSystem | None = None,
    ) -> float | np.ndarray[Any, np.dtype[np.number]]:
        """Convert momentum from self units into target units (defaults to SI)."""
        units = units or UnitSystem.si()
        momentum_factor = units.momentum_factor / self.momentum_factor
        return value * momentum_factor

    @overload
    def energy_into(self, value: float, units: UnitSystem | None = None) -> float: ...
    @overload
    def energy_into[DT: np.dtype[np.number]](
        self, value: np.ndarray[Any, DT], units: UnitSystem | None = None
    ) -> np.ndarray[Any, DT]: ...
    def energy_into(
        self,
        value: float | np.ndarray[Any, np.dtype[np.number]],
        units: UnitSystem | None = None,
    ) -> float | np.ndarray[Any, np.dtype[np.number]]:
        """Convert energy from self units into target units (defaults to SI)."""
        units = units or UnitSystem.si()
        return value * (units.energy_factor / self.energy_factor)

    def as_canonical(self) -> CanonicalUnitSystem:
        """Get a jax-compatible unit system."""
        return CanonicalUnitSystem(
            boltzmann=self.boltzmann,
            atomic_mass=self.atomic_mass,
            angstrom=self.angstrom,
        )


@jax.tree_util.register_dataclass
@final
@dataclass(frozen=True, kw_only=True)
class CanonicalUnitSystem(UnitSystem):
    """A jax-compatible unit system."""
