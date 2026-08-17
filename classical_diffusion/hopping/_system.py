from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from classical_diffusion.system import UnitSystem
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from classical_diffusion.langevin import PeriodicSystem1D


class CanonicalLattice(Protocol):
    """Protocol for JAX-compatible canonical PyTree lattices."""

    def get_rates(self, positions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]: ...


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
@dataclass(frozen=True)
class CanonicalLattice1D:
    hop_time: float
    lattice_spacing: float

    def get_rates(self, positions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        hop_sites = positions + jnp.array([[1], [-1]])
        hop_rates = jnp.array([1.0 / self.hop_time, 1.0 / self.hop_time])
        return hop_sites, hop_rates


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
        return indices * self.lattice_spacing

    @timed
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

    def as_canonical(self) -> CanonicalLattice1D:
        return CanonicalLattice1D(
            hop_time=(self.hop_time),
            lattice_spacing=(self.lattice_spacing),
        )


@dataclass(kw_only=True, frozen=True)
class KramersParameters:
    omega_well: float
    omega_barrier: float
    barrier_energy: float
    kbt: float
    gamma: float

    @property
    def delta_x(self) -> float:
        """The delta x of the system."""
        omegas_ss = self.omega_well**2 + self.omega_barrier**2
        return float(
            2
            * np.sqrt(
                (2 * omegas_ss * self.barrier_energy)
                / (self.omega_barrier**2 * self.omega_well**2)
            )
        )


def get_kramers_rate(params: KramersParameters) -> float:
    return (
        (params.omega_well * params.omega_barrier) / (2 * np.pi * params.gamma)
    ) * np.exp(-params.barrier_energy / params.kbt)


def get_kramers_parameters_cosine(system: PeriodicSystem1D) -> KramersParameters:
    """Potential must be cosine."""
    mass = system.m
    barrier_energy = system.barrier_energy
    delta_x = system.delta_x
    # Effective omega, approximating as a harmonic potential
    omega = np.sqrt(2 * (np.pi**2) * (barrier_energy / delta_x**2) / mass)

    return KramersParameters(
        omega_barrier=omega,
        omega_well=omega,
        barrier_energy=barrier_energy,
        kbt=system.kbt,
        gamma=system.gamma,
    )


def lattice_1d_from_kramers_parameters(params: KramersParameters) -> Lattice1D:
    """Get a 1D lattice from Kramers parameters."""
    hop_time = 1 / get_kramers_rate(params)
    return Lattice1D(lattice_spacing=params.delta_x, hop_time=hop_time)
