from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, kw_only=True)
class SingleSimulationResult[S: Any]:
    """Results of a simulation of the periodic Langevin equation."""

    system: S
    times: np.ndarray[tuple[int], np.dtype[np.floating]]
    x_points: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    """
    The positions of the particle at each time point.

    Stored as a 2d array of shape (n_dimensions, n_times).
    """


class SimulationResult[S: Any]:
    """Results of a simulation ensemble."""

    _times: np.ndarray[tuple[int], np.dtype[np.floating]]
    _x_points: np.ndarray[tuple[int, int, int], np.dtype[np.floating]]
    _system: S

    def __init__(
        self,
        *,
        times: np.ndarray[tuple[int], np.dtype[np.floating]],
        x_points: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
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

        Stored as a 3d array of shape (n_samples, n_dimensions, n_times).
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
