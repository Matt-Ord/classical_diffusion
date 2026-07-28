from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, kw_only=True)
class Lattice(ABC):
    """Parameters representing a simplified, discrete lattice representing a physical potential."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """The number of dimensions of the lattice."""

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
        self, pos: jnp.ndarray[Any, jnp.dtype[jnp.int_]]
    ) -> tuple[
        jnp.ndarray[Any, jnp.dtype[jnp.int_]],
        jnp.ndarray[Any, jnp.dtype[jnp.float_]],
    ]:
        hop_sites = pos + jnp.array([[1], [-1]])
        simple_hop_rate = 1 / self.hop_time
        hop_rates = jnp.array([simple_hop_rate, simple_hop_rate])
        return (hop_sites, hop_rates)
