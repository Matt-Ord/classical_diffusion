import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, final

import jax
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

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

    @property
    def units(self) -> UnitSystem:
        """The units used for the simulation."""
        return self.system.units


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
        self._times = times
        self._x_points = x_points
        self._system = system

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

    @property
    def units(self) -> UnitSystem:
        """The units used for the simulation."""
        return self.system.units

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

    @classmethod
    def from_iter(cls, results: Iterable[SingleSimulationResult[S]]) -> Self:
        """Create a SimulationResult from an iterable of SingleSimulationResults."""
        results_list = list(results)
        return cls(
            times=results_list[0].times,
            x_points=np.stack([r.x_points for r in results_list]),
            system=results_list[0].system,
        )

    def with_units(self, units: UnitSystem) -> Self:
        """Return the rescaled simulation of the system."""
        return type(self)(
            times=self.system.units.time_into(self.times, units),
            x_points=self.system.units.length_into(self.x_points, units),
            system=self.system.with_units(units),
        )


class IsfResult:
    """An intermediate scattering function I(delta_k, t) = <e^{i delta_k . (x(t) - x(0))}>.

    Every method which finds an ISF returns this, whether from an ensemble of
    trajectories, where the ISF is the mean and the error its standard error, or
    from a deterministic solver, where the error is zero.
    """

    def __init__(
        self,
        *,
        times: np.ndarray[Any, np.dtype[np.floating]],
        isf: np.ndarray[Any, np.dtype[np.number]],
        delta_k: tuple[float, ...],
        units: UnitSystem,
        error: np.ndarray[Any, np.dtype[np.floating]] | None = None,
    ) -> None:
        self._times = np.asarray(times, dtype=np.float64)
        self._isf = np.asarray(isf, dtype=np.complex128)
        self._error = np.zeros(self._isf.shape) if error is None else np.asarray(error)
        self._delta_k = tuple(float(k) for k in delta_k)
        self._units = units

    @property
    def times(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """The time since the origin, in the units."""
        return self._times

    @property
    def isf(self) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """The ISF at each time."""
        return self._isf

    @property
    def error(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """The standard error of the ISF at each time."""
        return self._error

    @property
    def delta_k(self) -> tuple[float, ...]:
        """The scattering wavevector, in the units."""
        return self._delta_k

    @property
    def units(self) -> UnitSystem:
        """The units of the times and the scattering wavevector."""
        return self._units

    def with_units(self, units: UnitSystem) -> IsfResult:
        """Return the ISF, with its times and wavevector in the given units."""
        length = self.units.length_into(1.0, units)
        return IsfResult(
            times=self.units.time_into(self.times, units),
            isf=self.isf,
            error=self.error,
            delta_k=tuple(k / length for k in self.delta_k),
            units=units,
        )


@final
@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class TimeSpan:
    """Time-stepping parameters, bundled together."""

    t_start: float = 0
    t_end: float
    n_steps: int = field(metadata={"static": True})
