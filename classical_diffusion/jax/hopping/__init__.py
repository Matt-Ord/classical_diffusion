"""Jax-native simulations using a hopping model."""

from classical_diffusion.jax.hopping._analysis import get_deterministic_isf
from classical_diffusion.jax.hopping._hopping import (
    get_deterministic_probabilities,
    solve_ensemble,
)

__all__ = ["get_deterministic_isf", "get_deterministic_probabilities", "solve_ensemble"]
