"""Deterministic solutions of the classical Langevin equation of a periodic system: the Kramers equation, and its overdamped limit, the Smoluchowski equation.

The solvers give the ISF, mean square displacement and diffusion constant of
the Langevin ensembles of `classical_diffusion.langevin` with no sampling
error, to a tolerance (`KramersConfig`).
"""

from classical_diffusion._results import RmsResult, SlowModes, get_long_time_isf
from classical_diffusion.kramers._config import DEFAULT_CONFIG, KramersConfig
from classical_diffusion.kramers._kramers import (
    get_diffusion_constant,
    get_slow_modes,
    solve_isf,
    solve_rms,
)
from classical_diffusion.kramers._overdamped import (
    get_diffusion_constant_overdamped,
    get_slow_modes_overdamped,
    solve_isf_overdamped,
    solve_rms_overdamped,
)

__all__ = [
    "DEFAULT_CONFIG",
    "KramersConfig",
    "RmsResult",
    "SlowModes",
    "get_diffusion_constant",
    "get_diffusion_constant_overdamped",
    "get_long_time_isf",
    "get_slow_modes",
    "get_slow_modes_overdamped",
    "solve_isf",
    "solve_isf_overdamped",
    "solve_rms",
    "solve_rms_overdamped",
]
