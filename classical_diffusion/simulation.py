from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, kw_only=True)
class SingleSimulationResult[S: Any]:
    """Results of a simulation of the periodic Langevin equation."""

    system: S
    times: np.ndarray
    x_points: np.ndarray[Any, np.dtype[np.floating]]


class SimulationResult[S: Any]:
    """Results of a simulation ensemble."""

    _times: np.ndarray
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
    def times(self) -> np.ndarray:
        return self._times

    @property
    def x_points(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        return self._x_points

    @property
    def system(self) -> S:
        return self._system

    def __getitem__(self, idx: int) -> SingleSimulationResult[S]:
        """Return a single trajectory from the ensemble."""
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
