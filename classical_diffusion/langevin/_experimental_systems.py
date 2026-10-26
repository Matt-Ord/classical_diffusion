import numpy as np
from scipy.constants import Avogadro, atomic_mass, electron_volt

from classical_diffusion.langevin._system import (
    PeriodicSystem1D,
    PeriodicSystemFCC,
    PeriodicSystemTriangular,
)

SODIUM_COPPER_BRIDGE_ENERGY = (416.78 - 414.24) * 1e3 / Avogadro

SODIUM_COPPER_SYSTEM_2D = PeriodicSystemFCC(
    gamma=2e11,
    temperature=155,
    barrier_energy=SODIUM_COPPER_BRIDGE_ENERGY,
    delta_x=2.558e-10,
    m=3.8175458e-26,
)

SODIUM_COPPER_SYSTEM_1D = PeriodicSystem1D(
    gamma=2e11,
    temperature=155,
    barrier_energy=SODIUM_COPPER_BRIDGE_ENERGY,
    delta_x=(1 / np.sqrt(3)) * SODIUM_COPPER_SYSTEM_2D.delta_x,
    m=3.8175458e-26,
)

XENON_PLATINUM_BRIDGE_ENERGY = 31.3e-3 * electron_volt
"""The top to bridge barrier of xenon on Pt(111).

The barrier matches the measured activation energy for diffusion, 32 +- 8 meV.
"""

XENON_PLATINUM_SYSTEM_2D = PeriodicSystemTriangular(
    gamma=5e10,
    temperature=121,
    barrier_energy=XENON_PLATINUM_BRIDGE_ENERGY,
    delta_x=2.775e-10,
    m=131.293 * atomic_mass,
)

XENON_PLATINUM_SYSTEM_1D = PeriodicSystem1D(
    gamma=XENON_PLATINUM_SYSTEM_2D.gamma,
    temperature=XENON_PLATINUM_SYSTEM_2D.temperature,
    barrier_energy=XENON_PLATINUM_BRIDGE_ENERGY,
    delta_x=XENON_PLATINUM_SYSTEM_2D.delta_x,
    m=XENON_PLATINUM_SYSTEM_2D.m,
)
"""The cut through XENON_PLATINUM_SYSTEM_2D along a close-packed row."""
