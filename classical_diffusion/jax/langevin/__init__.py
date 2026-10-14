"""Jax-native Simulations of a Physical System."""

from classical_diffusion.jax.langevin._analysis import (
    get_trajectory_breakpoints,
    partition_trajectory,
)
from classical_diffusion.jax.langevin._langevin import (
    DEFAULT_STEP_SIZE,
    Adaptive,
    FixedStep,
    StepSize,
    solve_many,
    solve_many_overdamped,
)
from classical_diffusion.jax.langevin._system import (
    KramersParametersJax as KramersParameters,
)
from classical_diffusion.jax.langevin._system import KramersSystem1D

__all__ = [
    "DEFAULT_STEP_SIZE",
    "Adaptive",
    "FixedStep",
    "KramersParameters",
    "KramersSystem1D",
    "StepSize",
    "get_trajectory_breakpoints",
    "partition_trajectory",
    "solve_many",
    "solve_many_overdamped",
]
