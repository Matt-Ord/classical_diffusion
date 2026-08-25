from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from classical_diffusion.hopping._system import Lattice
from classical_diffusion.jax.hopping import (
    get_deterministic_probabilities as get_deterministic_probabilities_jax,
)
from classical_diffusion.jax.hopping import solve_ensemble as solve_ensemble_jax
from classical_diffusion.simulation import SimulationResult, TimeSpan
from classical_diffusion.util import _get_key, timed


class HoppingSimulationResult[L: Lattice](SimulationResult[L]):
    def __init__(
        self,
        *,
        system: L,
        x_indices: np.ndarray[Any, np.dtype[np.int_]],
        times: np.ndarray[Any, np.dtype[np.floating]],
    ) -> None:
        self._system = system
        self._x_indices = x_indices
        self._times = times

    @cached_property
    def x_points(self) -> np.ndarray:
        return self.system.x_points_from_indices(self._x_indices)


@dataclass(frozen=True, kw_only=True)
class DeterministicSolverResult[L: Lattice]:
    system: L
    times: np.ndarray[tuple[int], np.dtype[np.floating]]
    probabilities: np.ndarray[tuple[int, int], np.dtype[np.floating]]


@timed
def solve_ensemble[L: Lattice = Lattice](
    system: L,
    time_span: TimeSpan,
    initial_condition: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    *,
    _key: jax.Array | None = None,
) -> HoppingSimulationResult[L]:
    """Solve the hopping ensemble."""
    _key = _get_key(_key)

    x_indices, times = solve_ensemble_jax(
        system.as_canonical(), time_span, jnp.array(initial_condition), _key=_key
    )

    return HoppingSimulationResult[L](
        system=system,
        times=np.array(times),
        x_indices=np.array(jnp.transpose(x_indices, (0, 2, 1))),
    )


@timed
def get_deterministic_probabilities[L: Lattice](
    system: L,
    shape: tuple[int, ...],
    time_span: TimeSpan,
) -> DeterministicSolverResult:
    """Use a deterministic PDE to find the ensemble probabilities at all times."""
    probabilities, times = get_deterministic_probabilities_jax(
        system.as_canonical(), time_span, shape=shape
    )
    return DeterministicSolverResult(
        system=system, times=np.array(times), probabilities=np.array(probabilities)
    )
