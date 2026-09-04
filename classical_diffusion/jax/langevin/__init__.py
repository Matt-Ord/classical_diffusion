"""Jax-native Simulations of a Physical System."""

from classical_diffusion.jax.langevin._analysis import (
    get_trajectory_breakpoints,
    partition_trajectory,
)
from classical_diffusion.jax.langevin._langevin import (
    solve_many,
    solve_many_overdamped,
)
from classical_diffusion.jax.langevin._system import (
    KramersParametersJax as KramersParameters,
)
from classical_diffusion.jax.langevin._system import KramersSystem1D

__all__ = [
    "KramersParameters",
    "KramersSystem1D",
    "get_trajectory_breakpoints",
    "partition_trajectory",
    "solve_many",
    "solve_many_overdamped",
]
