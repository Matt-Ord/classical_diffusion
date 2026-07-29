from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np


class CanonicalLattice(Protocol):
    """Protocol for JAX-compatible canonical PyTree lattices."""

    def get_rates(self, pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]: ...


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
        self, pos: jnp.ndarray[Any, jnp.dtype[jnp.int_]]
    ) -> tuple[
        jnp.ndarray[Any, jnp.dtype[jnp.int_]],
        jnp.ndarray[Any, jnp.dtype[jnp.float_]],
    ]:
        pass

    @abstractmethod
    def as_canonical(self) -> CanonicalLattice: ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CanonicalLattice1D:
    hop_time: float
    lattice_spacing: float

    def get_rates(self, pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        hop_sites = pos + jnp.array([[1], [-1]])
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

    def get_rates(
        self, pos: np.ndarray[Any, np.dtype[np.int_]]
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.int_]],
        np.ndarray[Any, np.dtype[np.float64]],
    ]:
        hop_sites = pos + np.array([[1], [-1]])
        hop_rates = np.array([1 / self.hop_time, 1 / self.hop_time])
        return (hop_sites, hop_rates)

    def as_canonical(self) -> CanonicalLattice1D:
        return CanonicalLattice1D(
            hop_time=(self.hop_time),
            lattice_spacing=(self.lattice_spacing),
        )
