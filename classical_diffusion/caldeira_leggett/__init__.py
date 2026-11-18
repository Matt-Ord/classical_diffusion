"""Simulations of a periodic system using the Caldeira-Leggett master equation.

The solver needs 64 bit floats in jax, `jax.config.update("jax_enable_x64", True)`.
"""

from classical_diffusion.caldeira_leggett._caldeira_leggett import (
    SLOW_RATE,
    SlowModes,
    SteadyState,
    get_diffusion_constant,
    get_long_time_isf,
    get_slow_modes,
    get_steady_state,
    solve_isf,
    solve_rms,
)
from classical_diffusion.caldeira_leggett._config import (
    DEFAULT_CONFIG,
    CaldeiraLeggettConfig,
)
from classical_diffusion.caldeira_leggett._evolution import (
    DEFAULT_EVOLUTION_CONFIG,
    EvolutionConfig,
)
from classical_diffusion.caldeira_leggett._solver import RmsResult

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_EVOLUTION_CONFIG",
    "SLOW_RATE",
    "CaldeiraLeggettConfig",
    "EvolutionConfig",
    "RmsResult",
    "SlowModes",
    "SteadyState",
    "get_diffusion_constant",
    "get_long_time_isf",
    "get_slow_modes",
    "get_steady_state",
    "solve_isf",
    "solve_rms",
]
