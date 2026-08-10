from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from classical_diffusion.util import timed

if TYPE_CHECKING:
    from classical_diffusion.langevin import PeriodicSystem1D, System


class CanonicalLattice(Protocol):
    """Protocol for JAX-compatible canonical PyTree lattices."""

    def get_rates(self, positions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]: ...


@dataclass(frozen=True, kw_only=True)
class Lattice(ABC):
    """Parameters representing a simplified, discrete lattice representing a physical potential."""

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


# This is not general - hop destinations are specified
class Lattice1D_4hop(Lattice):
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
        delta_site = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        hop_sites = positions[:, np.newaxis] + delta_site[np.newaxis, :]
        single_hop_rates = np.array(
            [
                0.125 / self.hop_time,
                0.25 / self.hop_time,
                0.5 / self.hop_time,
                1 / self.hop_time,
                1 / self.hop_time,
                0.5 / self.hop_time,
                0.25 / self.hop_time,
                0.125 / self.hop_time,
            ]
        )
        hop_rates = np.tile(single_hop_rates, (len(positions), 1))
        return (hop_sites, hop_rates)

    def as_canonical(self) -> CanonicalLattice1D:
        return CanonicalLattice1D(
            hop_time=(self.hop_time),
            lattice_spacing=(self.lattice_spacing),
        )


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


def get_kramers_lattice(system: System) -> Lattice:
    """Potential must be periodic."""
    gamma = system.gamma
    kBT = system.kbt
    mass = system.m

    potential = system.potential

    [sp.diff(potential, c) for c in coordinat]

    # 2. Hessian Matrix (N x N 2nd derivatives)
    sp.Matrix([[sp.diff(V, c1, c2) for c2 in coords] for c1 in coords])

    omega_max = jnp.sqrt(maximum / mass)
    omega_min = jnp.sqrt(minimum / mass)
    amplitude = maximum - minimum

    # @jax.jit
    def characterise_potential_jit() -> tuple[float, float, float]:

        print(f"min: {min}, max: {max}")

        omega_min = jnp.sqrt(jax.hessian(potential)(min) / mass)
        omega_max = jnp.sqrt(-jax.hessian(potential)(max) / mass)
        print(f"omega_min: {omega_min}, omega_max: {omega_max}")

        amp = potential(max) - potential(min)
        print(f"amp: {amp}")

        return omega_min, omega_max, amp

    omega_min, omega_max, amplitude = characterise_potential_jit()

    rate = (
        (omega_min * omega_max)
        / (2 * jnp.pi * mass * gamma)
        * jnp.exp(-amplitude / kBT)
    )

    return Lattice1D(lattice_spacing=1.0, hop_time=1.0 / rate)


def get_kramers_lattice_cosine(system: PeriodicSystem1D) -> Lattice1D:
    """Potential must be periodic."""
    system.potential[0]

    gamma = system.gamma
    kBT = system.kbt
    mass = system.m
    barrier_energy = system.barrier_energy
    delta_x = system.delta_x

    amplitude = barrier_energy
    omega_max = jnp.sqrt(2 * (jnp.pi**2) * (barrier_energy / delta_x**2) / mass)
    rate = float(
        (omega_max**2) / (2 * jnp.pi * mass * gamma) * jnp.exp(-amplitude / kBT)
    )

    return Lattice1D(lattice_spacing=1.0, hop_time=1.0 / rate)
