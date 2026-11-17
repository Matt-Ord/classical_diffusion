"""Simulations of a periodic system using the Caldeira-Leggett master equation.

The solver needs 64 bit floats in jax, `jax.config.update("jax_enable_x64", True)`.
"""

from classical_diffusion.caldeira_leggett._caldeira_leggett import (
    DEFAULT_ISF_STEP_CONFIG,
    SteadyState,
    get_diffusion_constant,
    get_steady_state,
    solve_isf,
    solve_rms,
)
from classical_diffusion.caldeira_leggett._config import (
    DEFAULT_CONFIG,
    CaldeiraLeggettConfig,
)
from classical_diffusion.caldeira_leggett._solver import RmsResult

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_ISF_STEP_CONFIG",
    "CaldeiraLeggettConfig",
    "RmsResult",
    "SteadyState",
    "get_diffusion_constant",
    "get_steady_state",
    "solve_isf",
    "solve_rms",
]
