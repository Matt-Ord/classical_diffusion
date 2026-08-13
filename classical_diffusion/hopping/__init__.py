"""Simulations using a hopping model."""

from classical_diffusion.hopping._analysis import plot_deterministic_isf
from classical_diffusion.hopping._hopping import (
    DeterministicSolverResult,
    HoppingSimulationResult,
    get_ensemble_probabilities,
    solve_ensemble,
)
from classical_diffusion.hopping._system import (
    KramersParameters,
    Lattice,
    Lattice1D,
    get_kramers_lattice,
    get_kramers_parameters_cosine,
    get_kramers_rate,
)

__all__ = [
    "DeterministicSolverResult",
    "HoppingSimulationResult",
    "KramersParameters",
    "Lattice",
    "Lattice1D",
    "get_ensemble_probabilities",
    "get_kramers_lattice",
    "get_kramers_parameters_cosine",
    "get_kramers_rate",
    "plot_deterministic_isf",
    "solve_ensemble",
]
