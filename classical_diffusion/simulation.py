import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, final

import jax

from classical_diffusion.system import UnitSystem

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np


@dataclass(frozen=True, kw_only=True)
class SingleSimulationResult[S: Any]:
    """Results of a simulation of the periodic Langevin equation."""

    system: S
    times: np.ndarray
    x_points: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    """The positions of the particle at each time point.

    stored as a 2D array of shape (n_dimensions, n_time_points).
    """

    def with_si_units(self) -> Self:
        """Return the rescaled simulation of the system."""
        si_units = UnitSystem()

        return dataclasses.replace(
            self,
            times=self.system.units.time_into(self.times, si_units),
            x_points=self.system.units.length_into(self.x_points, si_units),
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

    def with_si_units(self) -> Self:
        """Return the rescaled simulation of the system."""
        return type(self)(
            times=self.system.units.time_into(self.times, UnitSystem()),
            x_points=self.system.units.length_into(self.x_points, UnitSystem()),
            system=self.system.with_si_units(),
        )


@final
@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class TimeSpan:
    """Time-stepping parameters, bundled together."""

    t_start: float = field(default=0, metadata={"static": True})
    t_end: float = field(metadata={"static": True})
    n_steps: int = field(metadata={"static": True})

    def __post_init__(self) -> None:
        if self.t_end <= self.t_start:
            msg = f"t_end must be greater than t_start, got t_start={self.t_start}, t_end={self.t_end}"
            raise ValueError(msg)
        if self.n_steps <= 1:
            msg = f"Time span must have at least 2 steps, got n_steps={self.n_steps}"
            raise ValueError(msg)
