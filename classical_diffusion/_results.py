"""The results and time handling shared by the quantum (`caldeira_leggett`) and classical (`kramers`) solvers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from classical_diffusion.simulation import IsfResult

if TYPE_CHECKING:
    from classical_diffusion.langevin._system import PeriodicSystem
    from classical_diffusion.system import UnitSystem

type FloatArray = np.ndarray[Any, np.dtype[np.float64]]
type IntArray = np.ndarray[Any, np.dtype[np.int64]]
type ComplexArray = np.ndarray[Any, np.dtype[np.complex128]]


@dataclass(frozen=True, kw_only=True)
class RmsResult:
    """The mean square displacement <(x(t) - x(0))^2> along a direction."""

    times: FloatArray
    mean_square: FloatArray
    direction: tuple[float, ...]
    """The unit vector along which x is measured."""
    units: UnitSystem

    @property
    def rms(self) -> FloatArray:
        """The root mean square displacement."""
        return np.sqrt(self.mean_square)


@dataclass(frozen=True, kw_only=True)
class SlowModes:
    """The slowest modes of the ISF of a system at dK, I(t) = sum_j a_j exp(lambda_j t) once the faster modes have decayed."""

    system: PeriodicSystem
    delta_k: tuple[float, ...]
    """The scattering wavevector, in the units of the system."""
    rates: ComplexArray
    """lambda_j, in the units of the system (1 / time), slowest first."""
    amplitudes: ComplexArray
    """a_j."""


SLOW_RATE = 0.25
"""The decay rate below which a mode is slow, in units of the friction.

A quarter of the friction is clear of the in-well vibrations, which decay at
about half of it, and of the momentum relaxation, at about the friction:
the modes below it are the hops between sites, and the exchange between
sites of different kinds.
"""


@dataclass(frozen=True, kw_only=True)
class ModeCount:
    """Exactly n_modes modes."""

    n_modes: int


@dataclass(frozen=True, kw_only=True)
class RateCut:
    """Every mode with Re lambda > -rate, and at least one."""

    rate: float


type Selection = ModeCount | RateCut


def check_mode_count(n_modes: int | None) -> None:
    """Check the number of slow modes asked for.

    Raises
    ------
    ValueError
        If n_modes is negative.
    """
    if n_modes is not None and n_modes < 0:
        msg = f"n_modes must not be negative, got {n_modes}"
        raise ValueError(msg)


def get_selection(n_modes: int | None, friction: float) -> Selection:
    """Get n_modes modes, or by default every mode slower than SLOW_RATE of the friction (in any units of 1 / time).

    Raises
    ------
    ValueError
        If n_modes is negative.
    """  # ruff:ignore[docstring-extraneous-exception]
    check_mode_count(n_modes)
    if n_modes is None:
        return RateCut(rate=SLOW_RATE * friction)
    return ModeCount(n_modes=n_modes)


def select_modes(
    rates: ComplexArray, amplitudes: ComplexArray, selection: Selection
) -> tuple[ComplexArray, ComplexArray]:
    """Get the modes the selection asks for, slowest first."""
    order = np.argsort(-rates.real, kind="stable")
    rates, amplitudes = rates[order], amplitudes[order]
    if isinstance(selection, ModeCount):
        count = selection.n_modes
    else:
        count = max(1, int(np.count_nonzero(rates.real > -selection.rate)))
    return rates[:count], amplitudes[:count]


def get_long_time_isf(slow_modes: SlowModes, times: FloatArray) -> IsfResult:
    """Get the ISF of the slow modes alone, sum_j a_j exp(lambda_j t), at the times in the units of the system.

    It is the ISF once the faster modes have decayed: for the ISF at every
    time, use `solve_isf`.
    """
    folded = fold_times(times)
    isf = np.exp(np.outer(folded.magnitudes, slow_modes.rates)) @ slow_modes.amplitudes
    return IsfResult(
        times=folded.times,
        isf=folded.hermitian(isf),
        delta_k=slow_modes.delta_k,
        units=slow_modes.system.units,
    )


def get_unit_direction(n_dim: int, direction: tuple[float, ...] | None) -> FloatArray:
    """Get the unit vector along a direction, by default the first axis."""
    unit = np.eye(n_dim)[0] if direction is None else np.asarray(direction, float)
    return unit / np.linalg.norm(unit)


@dataclass(frozen=True, kw_only=True)
class FoldedTimes:
    """Any real times, in any order, as the magnitudes |t| to evolve to, unique and increasing from zero."""

    times: FloatArray
    """The times asked for."""
    magnitudes: FloatArray
    """The unique |t| and zero, increasing."""
    index: IntArray
    """The index in magnitudes of each |t|."""

    def even(self, values: np.ndarray) -> np.ndarray:
        """Get a quantity even in t at the times, from its values at the magnitudes."""
        return values[self.index]

    def hermitian(self, values: np.ndarray) -> np.ndarray:
        """Get a correlation with f(-t) = f(t)^* at the times, from its values at the magnitudes."""
        out = values[self.index]
        return np.where(self.times < 0, np.conj(out), out)


def fold_times(times: FloatArray) -> FoldedTimes:
    """Fold any real times onto the magnitudes from zero (`FoldedTimes`).

    For times which increase from zero the magnitudes are the times themselves.

    Raises
    ------
    ValueError
        If the times are not a one dimensional array of finite numbers.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        msg = "The times must be a one dimensional array of finite numbers"
        raise ValueError(msg)
    magnitudes, index = np.unique(
        np.abs(np.concatenate([[0.0], times])), return_inverse=True
    )
    return FoldedTimes(times=times, magnitudes=magnitudes, index=index[1:])
