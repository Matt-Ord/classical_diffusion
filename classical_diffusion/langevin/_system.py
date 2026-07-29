import dataclasses
import zlib
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Self, final, overload, override

import jax
import numpy as np
import sympy as sp
from scipy.constants import (  # type: ignore[import-untyped]
    Boltzmann as Boltzmann_si,
)
from scipy.constants import (  # type: ignore[import-untyped]
    angstrom as angstrom_si,
)
from scipy.constants import (  # type: ignore[import-untyped]
    atomic_mass as atomic_mass_si,
)


def _hash_sympy_expr(expr: sp.Expr) -> int:
    stable_string = sp.srepr(expr)
    return zlib.crc32(stable_string.encode("utf-8"))


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




@dataclass(frozen=True, kw_only=True)
class System:
    """Parameters representing a physical system."""

    gamma: float
    temperature: float
    m: float
    potential: tuple[int, sp.Expr]
    params: tuple[float, ...] = ()
    units: UnitSystem

    @property
    def n_dim(self) -> int:
        """The number of dimensions of the system."""
        return self.potential[0]

    @cached_property
    def coordinate_symbols(self) -> tuple[sp.Symbol, ...]:
        """The symbols of each coordinate of the system."""
        return sp.symbols(f"x0:{self.n_dim}")

    @property
    def n_params(self) -> int:
        """The number of parameters of the system."""
        return len(self.params)

    @cached_property
    def parameter_symbols(self) -> tuple[sp.Symbol, ...]:
        """The symbols of each parameter of the system."""
        return sp.symbols(f"s0:{self.n_params}")

    @cached_property
    def lambda_symbols(self) -> tuple[sp.Symbol, ...]:
        """The symbols of the system, including coordinates and parameters."""
        return (*self.coordinate_symbols, *self.parameter_symbols)

    @property
    def potential_expr(self) -> sp.Expr:
        """The potential expression of the system."""
        return self.potential[1]

    @cached_property
    def force_expr(self) -> list[sp.Expr]:
        """The symbolic force of the system."""
        return [
            -sp.simplify(sp.diff(self.potential_expr, c))
            for c in self.coordinate_symbols
        ]

    def with_gamma(self, gamma: float) -> System:
        """Return the system with exchanged gamma."""
        return System(
            gamma=gamma,
            m=self.m,
            temperature=self.temperature,
            potential=self.potential,
            units=self.units,
        )

    def as_canonical(self) -> CanonicalSystem:
        """Return the canonical form of the system."""
        return CanonicalSystem(
            gamma=self.gamma,
            temperature=self.temperature,
            m=self.m,
            potential=self.potential,
            params=self.params,
            units=self.units,
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.gamma,
                self.temperature,
                self.m,
                self.params,
                (
                    self.potential[0],
                    _hash_sympy_expr(self.potential[1]),
                ),
            )
        )

    @property
    def kbt(self) -> float:
        """Retrurn kbt."""
        return self.units.Boltzmann * self.temperature

    @property
    def sampling_domain(self) -> tuple[float, float]:
        """The domain over which the equilibrium x-density should be sampled."""
        return (-np.inf, np.inf)

    def with_normalized_units(self) -> Self:
        si_units = UnitSystem()
        length_factor = si_units.angstrom / self.units.angstrom
        mass_factor = si_units.atomic_mass / self.units.atomic_mass
        energy_factor = si_units.Boltzmann / self.units.Boltzmann
        time_factor = np.sqrt(length_factor**2 * mass_factor / energy_factor)
        return dataclasses.replace(
            self,
            gamma=self.gamma / time_factor,
            temperature=self.temperature,
            m=self.m * mass_factor,
            units=si_units,
            potential=self.potential,
        )

    def with_si_units(self) -> Self:
        si_units = UnitSystem()
        length_factor = si_units.angstrom / self.units.angstrom
        mass_factor = si_units.atomic_mass / self.units.atomic_mass
        energy_factor = si_units.Boltzmann / self.units.Boltzmann
        time_factor = np.sqrt(length_factor**2 * mass_factor / energy_factor)
        return dataclasses.replace(
            self,
            gamma=self.gamma / time_factor,
            temperature=self.temperature,
            m=self.m * mass_factor,
            units=si_units,
            potential=self.potential,
        )


@jax.tree_util.register_dataclass
@final
@dataclass(frozen=True, kw_only=True)
class CanonicalSystem(System):
    """Parameters representing a physical system."""

    potential: tuple[int, sp.Expr] = field(metadata={"static": True})
    units: UnitSystem = field(metadata={"static": True})


class HarmonicSystem(System):
    """Parameters representing a simple harmonic oscillator system."""

    _omega: float

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        omega: float,
        units: UnitSystem,
        n_dim: int = 1,
    ) -> None:
        s0 = sp.Symbol("s0")
        potential = 0.5 * s0**2 * sp.symbols("x0") ** 2

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(n_dim, potential),
            params=(omega,),
            units=units,
        )

    @property
    def omega(self) -> float:
        """The angular frequency of the system."""
        return self.params[0]

    @override
    def with_gamma(self, gamma: float) -> HarmonicSystem:
        return HarmonicSystem(
            gamma=gamma,
            temperature=self.temperature,
            m=self.m,
            omega=self.omega,
            n_dim=self.n_dim,
            units=self.units,
        )


class PeriodicSystem1D(System):
    """Parameters for a 1D cosine potential system."""

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        delta_x: float,
        barrier_energy: float,
        units: UnitSystem,
        n_dim: int = 1,
    ) -> None:
        s0 = sp.Symbol("s0")
        s1 = sp.Symbol("s1")
        potential = 0.5 * s1 * (1 - sp.cos(2 * sp.pi * sp.symbols("x0") / s0))

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(n_dim, potential),
            params=(delta_x, barrier_energy),
            units=units,
        )

    @property
    def delta_x(self) -> float:
        """The delta x of the system."""
        return self.params[0]

    @property
    def barrier_energy(self) -> float:
        """The barrier energy of the system."""
        return self.params[1]

    @override
    def with_gamma(self, gamma: float) -> PeriodicSystem1D:
        return PeriodicSystem1D(
            gamma=gamma,
            temperature=self.temperature,
            m=self.m,
            delta_x=self.delta_x,
            barrier_energy=self.barrier_energy,
            n_dim=self.n_dim,
            units=self.units,
        )

    @override
    @property
    def sampling_domain(self) -> tuple[float, float]:
        """The domain over which the equilibrium x-density should be sampled."""
        return (-self.delta_x / 2, self.delta_x / 2)

    def get_normalized_units(self) -> UnitSystem:
        """Return the units suited to a simulation of the system."""
        return UnitSystem(
            Boltzmann=1 / self.temperature,
            angstrom=self.units.angstrom / self.delta_x,
            atomic_mass=self.units.atomic_mass / self.m,
        )

    def with_normalized_units(self) -> PeriodicSystem1D:
        """Return the normalized parameters of the simulation."""
        normalized_units = self.get_normalized_units()
        length_factor = normalized_units.angstrom / self.units.angstrom
        mass_factor = normalized_units.atomic_mass / self.units.atomic_mass
        energy_factor = normalized_units.Boltzmann / self.units.Boltzmann
        time_factor = np.sqrt(length_factor**2 * mass_factor / energy_factor)
        return PeriodicSystem1D(
            gamma=self.gamma / time_factor,
            temperature=self.temperature,
            m=self.m * mass_factor,
            delta_x=self.delta_x * length_factor,
            barrier_energy=self.barrier_energy * energy_factor,
            n_dim=self.n_dim,
            units=normalized_units,
        )

    def with_si_units(self) -> PeriodicSystem1D:
        """Return the si parameters of the system."""
        si_units = UnitSystem()
        length_factor = si_units.angstrom / self.units.angstrom
        mass_factor = si_units.atomic_mass / self.units.atomic_mass
        energy_factor = si_units.Boltzmann / self.units.Boltzmann
        time_factor = np.sqrt(length_factor**2 * mass_factor / energy_factor)
        return PeriodicSystem1D(
            gamma=self.gamma / time_factor,
            temperature=self.temperature,
            m=self.m * mass_factor,
            delta_x=self.delta_x * length_factor,
            barrier_energy=self.barrier_energy * energy_factor,
            n_dim=self.n_dim,
            units=si_units,
        )


def _get_potential_expr_fcc() -> sp.Expr:
    """Return the potential energy expression for a 2D FCC lattice."""
    x0, x1 = sp.symbols("x0 x1")
    s0, s1 = sp.symbols("s0 s1")
    c = 2.0 * sp.pi / (sp.sqrt(3.0) * s0)

    kx0 = c * (-1.0 / sp.sqrt(3.0))
    kx1 = c * 1.0

    ky0 = c * (2.0 / sp.sqrt(3.0))
    ky1 = 0.0

    arg1 = x0 * kx0 + x1 * kx1
    arg2 = x0 * ky0 + x1 * ky1
    arg3 = arg1 + arg2

    cos_sum = sp.cos(arg1) + sp.cos(arg2) + sp.cos(arg3)

    return 2.0 * s1 * cos_sum


class PeriodicSystemFCC(System):
    """Parameters for the face-centered cubic periodic system."""

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        delta_x: float,
        barrier_energy: float,
        units: UnitSystem,
    ) -> None:
        potential = _get_potential_expr_fcc()

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(2, potential),
            params=(delta_x, barrier_energy),
            units=units,
        )

    @property
    def delta_x(self) -> float:
        """The delta x of the system."""
        return self.params[0]

    @property
    def barrier_energy(self) -> float:
        """The barrier energy of the system."""
        return self.params[1]


def get_energy(
    system: System, x_points: np.ndarray, p_points: np.ndarray
) -> np.ndarray:
    """Return the energy of the system."""
    potential = sp.lambdify(
        (*system.coordinate_symbols, *system.parameter_symbols),
        system.potential_expr,
        "numpy",
    )

    potential = potential(x_points, *system.params).squeeze(axis=1)

    kinetic = np.sum(p_points**2, axis=1) / (2 * system.m)

    return kinetic + potential


def get_diffusion_time(system: System, characteristic_length: float) -> float:
    """Return the average time for a particle to traverse a characteristic length."""
    return np.sqrt(system.m * characteristic_length**2 / system.kbt)


def get_characteristic_periodic_mass(system: PeriodicSystem1D) -> float:
    """Return the characteristic mass for a 1D periodic system."""
    return system.kbt * system.delta_x**2 / system.gamma**2
