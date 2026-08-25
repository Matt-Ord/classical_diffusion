import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, final

import jax

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from classical_diffusion.system import UnitSystem


@dataclass(frozen=True, kw_only=True)
class SingleSimulationResult[S: Any]:
    """Results of a simulation of the periodic Langevin equation."""

    system: S
    times: np.ndarray
    x_points: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    """The positions of the particle at each time point.

    stored as a 2D array of shape (n_dimensions, n_time_points).
    """

    def with_units(self, units: UnitSystem) -> Self:
        """Return the rescaled simulation of the system."""
        return dataclasses.replace(
            self,
            times=self.system.units.time_into(self.times, units),
            x_points=self.system.units.length_into(self.x_points, units),
            system=self.system.with_units(units),
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
        """The positions of the particles at each time point.

        stored as a 3D array of shape (n_samples, n_dimensions, n_time_points).
        """
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

    def with_units(self, units: UnitSystem) -> Self:
        """Return the rescaled simulation of the system."""
        return type(self)(
            times=self.system.units.time_into(self.times, units),
            x_points=self.system.units.length_into(self.x_points, units),
            system=self.system.with_units(units),
        )


@final
@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class TimeSpan:
    """Time-stepping parameters, bundled together."""

    t_start: float = 0
    t_end: float
    n_steps: int = field(metadata={"static": True})
