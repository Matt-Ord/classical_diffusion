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

LITHIUM_MASS = 6.94 * atomic_mass

LITHIUM_COPPER_BRIDGE_ENERGY = 8e-3 * electron_volt
"""The hollow to bridge barrier of lithium on Cu(111).

The DFT (PBE) climbing image nudged elastic band barrier of G. Lai et al.,
arXiv:2208.04089 (2022), Fig. S10a, read from the figure to about +- 2 meV.
The two hollow sites are degenerate within this precision.
"""

LITHIUM_COPPER_TOP_ENERGY = 200e-3 * electron_volt
"""The hollow to top energy of lithium on Cu(111).

Read from the neural network potential energy map of Lai et al., Fig. 3c, to
about +- 20 meV. With the bridge barrier it gives a potential whose variance,
3.5e-3 eV^2, is close to the 3.0e-3 eV^2 Lai et al. quote for the map.
"""

LITHIUM_COPPER_GAMMA = 1.2e12
"""The friction of lithium on Cu(111)."""

LITHIUM_COPPER_SYSTEM_2D = PeriodicSystemFCC(
    gamma=LITHIUM_COPPER_GAMMA,
    temperature=140,
    barrier_energy=LITHIUM_COPPER_BRIDGE_ENERGY,
    top_energy=LITHIUM_COPPER_TOP_ENERGY,
    delta_x=2.556e-10,
    m=LITHIUM_MASS,
)
"""Lithium on Cu(111), with the Cu-Cu distance of the slab of Lai et al."""

LITHIUM_COPPER_SYSTEM_1D = PeriodicSystem1D(
    gamma=LITHIUM_COPPER_GAMMA,
    temperature=140,
    barrier_energy=LITHIUM_COPPER_BRIDGE_ENERGY,
    delta_x=(1 / np.sqrt(3)) * LITHIUM_COPPER_SYSTEM_2D.delta_x,
    m=LITHIUM_MASS,
)
"""The 1D analogue of LITHIUM_COPPER_SYSTEM_2D, with the same hollow to bridge barrier."""

HYDROGEN_MASS = 1.008 * atomic_mass

HYDROGEN_COPPER_BRIDGE_ENERGY = 135e-3 * electron_volt
"""The fcc to hcp hollow barrier of hydrogen on Cu(111).

The DFT (PBE) nudged elastic band barrier of W. Fang et al., Phys. Rev. Lett.
119, 126001 (2017), Fig. 1a, read from the figure to about +- 5 meV. The hcp
site lies 14 meV above the fcc site (Table S3), which is neglected here.
"""

HYDROGEN_COPPER_GAMMA = 1.5e13
"""The friction of hydrogen on Cu(111), about 15 ps^-1.

An upper bound from the HREELS linewidth of the parallel vibration, P. S. M.
Townsend, PhD thesis, University of Cambridge (2017), Section 6.5.6.
"""

HYDROGEN_COPPER_SYSTEM_2D = PeriodicSystemFCC(
    gamma=HYDROGEN_COPPER_GAMMA,
    temperature=150,
    barrier_energy=HYDROGEN_COPPER_BRIDGE_ENERGY,
    delta_x=2.558e-10,
    m=HYDROGEN_MASS,
)
"""Hydrogen on Cu(111). Fang et al. give no top site energy, so it is 9 times the bridge."""

HYDROGEN_COPPER_SYSTEM_1D = PeriodicSystem1D(
    gamma=HYDROGEN_COPPER_GAMMA,
    temperature=150,
    barrier_energy=HYDROGEN_COPPER_BRIDGE_ENERGY,
    delta_x=(1 / np.sqrt(3)) * HYDROGEN_COPPER_SYSTEM_2D.delta_x,
    m=HYDROGEN_MASS,
)
"""The 1D analogue of HYDROGEN_COPPER_SYSTEM_2D, with the same hollow to hollow barrier."""

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
