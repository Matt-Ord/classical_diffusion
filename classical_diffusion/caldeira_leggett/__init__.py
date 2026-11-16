"""Simulations of a periodic system using the Caldeira-Leggett master equation.

The solver needs 64 bit floats in jax, `jax.config.update("jax_enable_x64", True)`.
"""

from classical_diffusion.caldeira_leggett._caldeira_leggett import (
    SteadyState,
    get_diffusion_constant,
    get_steady_state,
    solve_isf,
)
from classical_diffusion.caldeira_leggett._config import (
    DEFAULT_CONFIG,
    CaldeiraLeggettConfig,
)

__all__ = [
    "DEFAULT_CONFIG",
    "CaldeiraLeggettConfig",
    "SteadyState",
    "get_diffusion_constant",
    "get_steady_state",
    "solve_isf",
]
