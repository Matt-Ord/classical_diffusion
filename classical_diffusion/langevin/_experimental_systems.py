import numpy as np
from scipy.constants import atomic_mass, electron_volt

from classical_diffusion.langevin._system import (
    PeriodicSystem1D,
    PeriodicSystemFCC,
    PeriodicSystemTriangular,
)

SODIUM_MASS = 22.990 * atomic_mass

SODIUM_COPPER_BRIDGE_ENERGY = 12e-3 * electron_volt
"""The hollow to bridge barrier of sodium on Cu(111).

The DFT (PBE) energy of S. P. Rittmeyer et al., Phys. Rev. Lett. 117, 196001
(2016), Table S1, used in the Langevin simulations of D. J. Ward et al.,
Phys. Chem. Chem. Phys. 23, 7799 (2021). It agrees with the activation
energy of 12 +- 3 meV measured by helium spin-echo.
"""

SODIUM_COPPER_TOP_ENERGY = 96e-3 * electron_volt
"""The hollow to top energy of sodium on Cu(111), from the same DFT as the bridge."""

SODIUM_COPPER_GAMMA = 10 * atomic_mass / 1e-12 / SODIUM_MASS
"""The friction of sodium on Cu(111), eta = 10 amu ps^-1 (about 0.43 ps^-1).

Fitted by Rittmeyer et al. to the helium spin-echo dephasing rates, with
Langevin dynamics m r'' = -grad V - m gamma r' + noise on the DFT potential.
"""

SODIUM_COPPER_SYSTEM_2D = PeriodicSystemFCC(
    gamma=SODIUM_COPPER_GAMMA,
    temperature=155,
    barrier_energy=SODIUM_COPPER_BRIDGE_ENERGY,
    top_energy=SODIUM_COPPER_TOP_ENERGY,
    delta_x=2.558e-10,
    m=SODIUM_MASS,
)

SODIUM_COPPER_SYSTEM_1D = PeriodicSystem1D(
    gamma=SODIUM_COPPER_GAMMA,
    temperature=155,
    barrier_energy=SODIUM_COPPER_BRIDGE_ENERGY,
    delta_x=(1 / np.sqrt(3)) * SODIUM_COPPER_SYSTEM_2D.delta_x,
    m=SODIUM_MASS,
)
"""The 1D analogue of SODIUM_COPPER_SYSTEM_2D, with the same hollow to bridge barrier."""

XENON_PLATINUM_BRIDGE_ENERGY = 32e-3 * electron_volt
"""The top to bridge barrier of xenon on Pt(111).

The barrier is the activation energy for diffusion, 32 +- 8 meV, measured
by helium spin-echo at 90 to 140 K in D. Ward, "A study of spin-echo
lineshapes in helium atom scattering from adsorbates", PhD thesis, University
of Cambridge (2013).
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
