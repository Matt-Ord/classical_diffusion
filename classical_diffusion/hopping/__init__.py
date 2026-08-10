"""Simulations using a hopping model."""

from ._analysis import plot_deterministic_isf
from ._hopping import (
    DeterministicSolverResult,
    HoppingSimulationResult,
    get_ensemble_probabilities,
    solve_ensemble,
)
from ._system import Lattice, Lattice1D, Lattice1D_4hop

__all__ = [
    "DeterministicSolverResult",
    "HoppingSimulationResult",
    "Lattice",
    "Lattice1D",
    "Lattice1D_4hop",
    "get_ensemble_probabilities",
    "plot_deterministic_isf",
    "solve_ensemble",
]
