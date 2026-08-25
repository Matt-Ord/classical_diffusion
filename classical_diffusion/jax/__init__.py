"""JAX compatible classes and functions."""

from classical_diffusion.jax._analysis import (
    get_deterministic_isf_jax as get_deterministic_isf,
)

__all__ = [
    "get_deterministic_isf",
]
