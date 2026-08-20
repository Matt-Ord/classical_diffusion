import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from classical_diffusion.system import UnitSystem

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True, kw_only=True)
class SingleSimulationResult[S: Any]:
    """Results of a simulation of the periodic Langevin equation."""

    system: S
    times: np.ndarray
    x_points: np.ndarray[tuple[int, int], np.dtype[np.floating]]

    def with_si_units(self) -> Self:
        """Return the rescaled simulation of the system."""
        si_units = UnitSystem()
        length_factor = si_units.angstrom / self.system.units.angstrom
        mass_factor = si_units.atomic_mass / self.system.units.atomic_mass
        energy_factor = si_units.Boltzmann / self.system.units.kb
        time_factor = np.sqrt(length_factor**2 * mass_factor / energy_factor)
        mass_factor * length_factor / time_factor
        return dataclasses.replace(
            self,
            times=self.times * time_factor,
            x_points=self.x_points * length_factor,
            system=self.system.with_si_units(),
        )


class SimulationResult[S: Any]:
    """Results of a simulation ensemble."""

    _times: np.ndarray[Any, np.dtype[np.floating]]
    _x_points: np.ndarray[Any, np.dtype[np.floating]]
    _system: S

    def __init__(
        self,
        *,
        times: np.ndarray,
        x_points: np.ndarray[Any, np.dtype[np.floating]],
        system: S,
    ) -> None:
        _times = times
        _x_points = x_points
        _system = system

    @property
    def times(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """The time points at which the simulation was sampled."""
        return self._times

    @property
    def x_points(self) -> np.ndarray[tuple[int, int, int], np.dtype[np.floating]]:
        """The positions of the particles at each time point."""
        return self._x_points

    @property
    def system(self) -> S:
        """The system used for the simulation."""
        return self._system

    def __getitem__(self, idx: int) -> SingleSimulationResult[S]:
        """Get a single trajectory from the ensemble."""
        return SingleSimulationResult[S](
            system=self.system,
            times=self._times,
            x_points=self.x_points[idx],
        )

    def __iter__(self) -> Iterator[SingleSimulationResult[S]]:
        """Iterate over the trajectories in the ensemble."""
        for i in range(self.x_points.shape[0]):
            yield self[i]

    def with_si_units(self) -> Self:
        """Return the rescaled simulation of the system."""
        si_units = UnitSystem()
        length_factor = si_units.angstrom / self.system.units.angstrom
        mass_factor = si_units.atomic_mass / self.system.units.atomic_mass
        energy_factor = si_units.Boltzmann / self.system.units.Boltzmann
        time_factor = np.sqrt(length_factor**2 * mass_factor / energy_factor)
        return type(self)(
            times=self.times * time_factor,
            x_points=self.x_points * length_factor,
            system=self.system.with_si_units(),
        )


@dataclass(frozen=True, kw_only=True)
class TimeSpan:
    """Time-stepping parameters, bundled together."""

    t_start: float = 0
    t_end: float
    n_steps: int

    def __post_init__(self) -> None:
        if self.t_end <= self.t_start:
            msg = f"t_end must be greater than t_start, got t_start={self.t_start}, t_end={self.t_end}"
            raise ValueError(msg)
        if self.n_steps <= 1:
            msg = f"Time span must have at least 2 steps, got n_steps={self.n_steps}"
            raise ValueError(msg)
