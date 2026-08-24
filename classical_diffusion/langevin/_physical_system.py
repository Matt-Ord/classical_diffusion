import numpy as np
from scipy.constants import Avogadro, atomic_mass

from classical_diffusion.langevin._system import PeriodicSystem1D, PeriodicSystemFCC

SODIUM_MASS = 22.99 * atomic_mass
# see <https://www.sciencedirect.com/science/article/pii/S0039602897000897>
SODIUM_COPPER_BRIDGE_ENERGY = 4.22 * 10 ** (-21)
LITHIUM_COPPER_BRIDGE_ENERGY = (477.16 - 471.41) * 1e3 / Avogadro


SODIUM_COPPER_1D = PeriodicSystem1D(
    gamma=2e11,
    temperature=150,
    m=SODIUM_MASS,
    delta_x=(2.558e-10 / np.sqrt(3)),
    barrier_energy=SODIUM_COPPER_BRIDGE_ENERGY,
)


SODIUM_COPPER_2D = PeriodicSystemFCC(
    gamma=SODIUM_COPPER_1D.gamma,
    temperature=SODIUM_COPPER_1D.temperature,
    m=SODIUM_MASS,
    delta_x=2.558e-10,  # TODO: need to check this.
    barrier_energy=SODIUM_COPPER_1D.barrier_energy,
)
