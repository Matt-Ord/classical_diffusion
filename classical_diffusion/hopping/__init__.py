"""Simulations using a hopping model."""

from classical_diffusion.hopping._analysis import (
    get_deterministic_isf,
    get_deterministic_isf_jax,
    plot_deterministic_isf,
)
from classical_diffusion.hopping._hopping import (
    DeterministicSolverResult,
    HoppingSimulationResult,
    deterministic_probabilities_jax,
    get_deterministic_probabilities,
    solve_ensemble,
)
from classical_diffusion.hopping._system import (
    CanonicalLattice,
    CanonicalLattice1D,
    KramersParameters,
    Lattice,
    Lattice1D,
    get_kramers_parameters_cosine,
    get_kramers_rate,
    lattice_1d_from_kramers_parameters,
)

__all__ = [
    "CanonicalLattice",
    "CanonicalLattice1D",
    "DeterministicSolverResult",
    "HoppingSimulationResult",
    "KramersParameters",
    "Lattice",
    "Lattice1D",
    "deterministic_probabilities_jax",
    "get_deterministic_isf",
    "get_deterministic_isf_jax",
    "get_deterministic_probabilities",
    "get_kramers_parameters_cosine",
    "get_kramers_rate",
    "lattice_1d_from_kramers_parameters",
    "plot_deterministic_isf",
    "solve_ensemble",
]
