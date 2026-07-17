"""Representation of a Physical System."""

from ._analysis import (
    get_characteristic_friction_time,
    plot_exact_flat_isf,
    plot_exact_harmonic_isf,
    plot_periodic_potential_1d,
    plot_periodic_potential_fcc,
    plot_potential_1d,
    plot_potential_2d,
)
from ._system import (
    FlatSystem2D,
    HarmonicSystem,
    PeriodicSystem1D,
    PeriodicSystemFCC,
    System,
)

__all__ = [
    "FlatSystem2D",
    "HarmonicSystem",
    "HarmonicSystem",
    "PeriodicSystem1D",
    "PeriodicSystemFCC",
    "System",
    "get_characteristic_friction_time",
    "plot_exact_flat_isf",
    "plot_exact_harmonic_isf",
    "plot_periodic_potential_1d",
    "plot_periodic_potential_fcc",
    "plot_potential_1d",
    "plot_potential_2d",
]
