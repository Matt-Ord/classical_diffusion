from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, final

import jax
import jax.numpy as jnp
import numpy as np

from classical_diffusion.system import UnitSystem

if TYPE_CHECKING:
    from classical_diffusion.langevin import PeriodicSystem1D


@dataclass(frozen=True, kw_only=True)
class Lattice(ABC):
    """Parameters representing a simplified, discrete lattice representing a physical potential."""

    units: UnitSystem = field(default_factory=UnitSystem)

    @abstractmethod
    def x_points_from_indices(
        self, indices: np.ndarray[Any, np.dtype[np.int_]]
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        pass

    @abstractmethod
    def get_rates(
        self, positions: np.ndarray[Any, np.dtype[np.int_]]
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.int_]],
        np.ndarray[Any, np.dtype[np.float64]],
    ]:
        pass

    @abstractmethod
    def as_canonical(self) -> CanonicalLattice: ...


@jax.tree_util.register_dataclass
@final
@dataclass(frozen=True)
class CanonicalLattice:
    """A canonical representation of a lattice, with units of time and length."""

    lattice_spacing: float
    hop_time: float

    def get_rates(self, positions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        delta_site = jnp.array([-1, 1])
        hop_sites = positions[:, jnp.newaxis] + delta_site[jnp.newaxis, :]
        single_hop_rates = jnp.array([1 / self.hop_time, 1 / self.hop_time])
        hop_rates = jnp.tile(single_hop_rates, (len(positions), 1))
        return (hop_sites, hop_rates)

    def x_points_from_indices(self, indices: jnp.ndarray) -> jnp.ndarray:
        return indices * self.lattice_spacing


class Lattice1D(Lattice):
    """A Hopping model for a 1D system."""

    def __init__(self, lattice_spacing: float, hop_time: float) -> None:
        self._lattice_spacing = lattice_spacing
        self._hop_time = hop_time

    @property
    def hop_time(self) -> float:
        return self._hop_time

    @property
    def lattice_spacing(self) -> float:
        return self._lattice_spacing

    def x_points_from_indices(
        self, indices: np.ndarray[Any, np.dtype[np.int_]]
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        return indices.astype(np.float64) * self.lattice_spacing

    def get_rates(
        self, positions: np.ndarray[Any, np.dtype[np.int_]]
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.int_]],
        np.ndarray[Any, np.dtype[np.float64]],
    ]:
        delta_site = np.array([-1, 1])
        hop_sites = positions[:, np.newaxis] + delta_site[np.newaxis, :]
        single_hop_rates = np.array(
            [
                1 / self.hop_time,
                1 / self.hop_time,
            ]
        )
        hop_rates = np.tile(single_hop_rates, (len(positions), 1))
        return (hop_sites, hop_rates)

    def as_canonical(self) -> CanonicalLattice:
        return CanonicalLattice(
            hop_time=(self.hop_time),
            lattice_spacing=(self.lattice_spacing),
        )


@dataclass(kw_only=True, frozen=True)
class KramersParameters:
    omega_well: float
    omega_barrier: float
    barrier_energy: float
    m: float
    temperature: float
    gamma: float
    units: UnitSystem = field(default_factory=UnitSystem)

    @property
    def delta_x(self) -> float:
        """The delta x of the system."""
        omegas_ss = self.omega_well**2 + self.omega_barrier**2
        return 2 * np.sqrt(
            (2 * omegas_ss * self.barrier_energy)
            / (self.omega_barrier**2 * self.omega_well**2)
        )

    @property
    def kbt(self) -> float:
        return self.temperature * self.units.boltzmann

    def with_units(self, units: UnitSystem) -> KramersParameters:
        """Return the parameters of the system in the specified units."""
        return KramersParameters(
            omega_well=self.units.frequency_into(self.omega_well, units),
            omega_barrier=self.units.frequency_into(self.omega_barrier, units),
            barrier_energy=self.units.energy_into(self.barrier_energy, units),
            m=self.units.mass_into(self.m, units),
            temperature=self.temperature,
            gamma=self.units.frequency_into(self.gamma, units),
            units=units,
        )


def get_kramers_rate(params: KramersParameters) -> float:
    return (
        (params.omega_well * params.omega_barrier) / (2 * np.pi * params.gamma)
    ) * np.exp(-params.barrier_energy / (params.kbt))


def get_kramers_parameters_cosine(system: PeriodicSystem1D) -> KramersParameters:
    """Potential must be cosine."""
    barrier_energy = system.barrier_energy
    delta_x = system.delta_x
    # Effective omega, approximating as a harmonic potential
    omega = np.sqrt(2 * (np.pi**2) * (barrier_energy / delta_x**2))

    return KramersParameters(
        omega_barrier=omega,
        omega_well=omega,
        barrier_energy=barrier_energy,
        m=system.m,
        temperature=system.temperature,
        gamma=system.gamma,
        units=system.units,
    )


def lattice_1d_from_kramers_parameters(params: KramersParameters) -> Lattice1D:
    """Get a 1D lattice from Kramers parameters."""
    hop_time = 1 / get_kramers_rate(params)
    return Lattice1D(lattice_spacing=params.delta_x, hop_time=hop_time)
