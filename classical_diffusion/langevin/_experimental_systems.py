import numpy as np
from scipy.constants import Avogadro

from classical_diffusion.langevin import PeriodicSystem1D, PeriodicSystemFCC

SODIUM_COPPER_BRIDGE_ENERGY = (416.78 - 414.24) * 1e3 / Avogadro

SODIUM_COPPER_SYSTEM_2D = PeriodicSystemFCC(
    gamma=5e11,
    temperature=155,
    barrier_energy=SODIUM_COPPER_BRIDGE_ENERGY,
    delta_x=2.558e-10,
    m=3.8175458e-26,
)

SODIUM_COPPER_SYSTEM_1D = PeriodicSystem1D(
    gamma=5e11,
    temperature=155,
    barrier_energy=SODIUM_COPPER_BRIDGE_ENERGY,
    delta_x=(1 / np.sqrt(3)) * SODIUM_COPPER_SYSTEM_2D.delta_x,
    m=3.8175458e-26,
)
