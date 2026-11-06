import numpy as np
from scipy.constants import atomic_mass, electron_volt

from classical_diffusion.langevin._system import (
    CosineSystem1D,
    build_2d_fcc,
    build_2d_rectangular,
    build_2d_square,
    build_2d_triangular,
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

SODIUM_COPPER_SPACING = 2.558e-10
"""The nearest-neighbor Cu-Cu spacing of Cu(111) used for sodium."""

SODIUM_COPPER_SYSTEM_2D = build_2d_fcc(
    gamma=SODIUM_COPPER_GAMMA,
    temperature=155,
    barrier_energy=SODIUM_COPPER_BRIDGE_ENERGY,
    top_energy=SODIUM_COPPER_TOP_ENERGY,
    delta_x=SODIUM_COPPER_SPACING,
    m=SODIUM_MASS,
)

SODIUM_COPPER_SYSTEM_1D = CosineSystem1D(
    gamma=SODIUM_COPPER_GAMMA,
    temperature=155,
    barrier_energy=SODIUM_COPPER_BRIDGE_ENERGY,
    delta_x=(1 / np.sqrt(3)) * SODIUM_COPPER_SPACING,
    m=SODIUM_MASS,
)
"""The 1D analogue of SODIUM_COPPER_SYSTEM_2D, with the same hollow to bridge barrier."""

LITHIUM_MASS = 6.94 * atomic_mass

LITHIUM_COPPER_BRIDGE_ENERGY = 7.6e-3 * electron_volt
"""The hollow to bridge barrier of lithium on Cu(111).

The DFT (PBE) climbing image nudged elastic band barrier of G. Lai et al.,
Small 19, 2205416 (2022) (thesis key Lai_2022), arXiv:2208.04089 Fig. S10a.
The figure is a raster image: the peak was found to +- 1 meV by fitting a
parabola to the centre of the plotted line, with the axis ticks giving
1.87 meV per pixel. The two hollow sites are degenerate within this precision.
"""

LITHIUM_COPPER_TOP_ENERGY = 200e-3 * electron_volt
"""The hollow to top energy of lithium on Cu(111).

Read from the neural network potential energy map of Lai et al., Fig. 3c, to
about +- 20 meV. With the bridge barrier it gives a potential whose variance,
3.49e-3 eV^2, is close to the 3.0e-3 eV^2 Lai et al. quote for the map.
"""

LITHIUM_COPPER_GAMMA = 1.2e12
"""The friction of lithium on Cu(111)."""

LITHIUM_COPPER_SPACING = 2.556e-10
"""The nearest-neighbor Cu-Cu spacing of the slab of Lai et al."""

LITHIUM_COPPER_SYSTEM_2D = build_2d_fcc(
    gamma=LITHIUM_COPPER_GAMMA,
    temperature=140,
    barrier_energy=LITHIUM_COPPER_BRIDGE_ENERGY,
    top_energy=LITHIUM_COPPER_TOP_ENERGY,
    delta_x=LITHIUM_COPPER_SPACING,
    m=LITHIUM_MASS,
)
"""Lithium on Cu(111), with the Cu-Cu distance of the slab of Lai et al."""

LITHIUM_COPPER_SYSTEM_1D = CosineSystem1D(
    gamma=LITHIUM_COPPER_GAMMA,
    temperature=140,
    barrier_energy=LITHIUM_COPPER_BRIDGE_ENERGY,
    delta_x=(1 / np.sqrt(3)) * LITHIUM_COPPER_SPACING,
    m=LITHIUM_MASS,
)
"""The 1D analogue of LITHIUM_COPPER_SYSTEM_2D, with the same hollow to bridge barrier."""

LITHIUM_COPPER_100_BRIDGE_ENERGY = 117.5e-3 * electron_volt
"""The hollow to bridge barrier of lithium on Cu(100).

Read from the same figure in the same way as LITHIUM_COPPER_BRIDGE_ENERGY, to +- 1 meV.
"""

LITHIUM_COPPER_100_SYSTEM_2D = build_2d_square(
    gamma=LITHIUM_COPPER_GAMMA,
    temperature=140,
    barrier_energy=LITHIUM_COPPER_100_BRIDGE_ENERGY,
    delta_x=LITHIUM_COPPER_SPACING,
    m=LITHIUM_MASS,
)
"""Lithium on Cu(100), with a single Fourier component.

Lai et al. give no DFT top-site energy, so the top sites are at twice the
bridge, 235 meV. Their neural network map (Fig. 3a), read by matching each
pixel to the colour bar (7.9 meV per bar pixel), puts the top sites at about
253 meV, so a second parameter is not needed. The potential's variance,
3.45e-3 eV^2, is close to the 2.9e-3 eV^2 Lai et al. quote for that map.
"""

LITHIUM_COPPER_100_SYSTEM_1D = CosineSystem1D(
    gamma=LITHIUM_COPPER_GAMMA,
    temperature=140,
    barrier_energy=LITHIUM_COPPER_100_BRIDGE_ENERGY,
    delta_x=LITHIUM_COPPER_SPACING,
    m=LITHIUM_MASS,
)
"""The cut through LITHIUM_COPPER_100_SYSTEM_2D from hollow to hollow over a bridge."""

LITHIUM_COPPER_110_TROUGH_ENERGY = 82.4e-3 * electron_volt
"""The barrier of lithium hopping along a trough of Cu(110), over the long bridge.

Path b of Lai et al., Fig. S10a, a hop of 2.556 Angstrom, read in the same way to +- 1 meV.
"""

LITHIUM_COPPER_110_ACROSS_ENERGY = 452.7e-3 * electron_volt
"""The barrier of lithium hopping across a row of Cu(110), over the short bridge.

Path a of Lai et al., Fig. S10a, a hop of 3.615 Angstrom, read in the same way to +- 1 meV.
"""

LITHIUM_COPPER_110_ROW_SPACING = 3.615e-10
"""The spacing of the close-packed rows of Cu(110), the cubic lattice constant."""

LITHIUM_COPPER_110_SYSTEM_2D = build_2d_rectangular(
    gamma=LITHIUM_COPPER_GAMMA,
    temperature=140,
    barrier_energy_x=LITHIUM_COPPER_110_TROUGH_ENERGY,
    barrier_energy_y=LITHIUM_COPPER_110_ACROSS_ENERGY,
    delta_x=LITHIUM_COPPER_SPACING,
    delta_y=LITHIUM_COPPER_110_ROW_SPACING,
    m=LITHIUM_MASS,
)
"""Lithium on Cu(110), with x along the troughs and y across the rows.

The top sites are at the sum of the two barriers, 535 meV. Lai et al.'s neural
network map (Fig. 3b) cannot fix them: its highest point, about 363 meV above
the hollow, is below their DFT barrier for crossing a row, so the map and the
NEB disagree on this surface. Its variance, 1.1e-2 eV^2, is also below the
2.65e-2 eV^2 of this potential; adding a cross term
C (1 - cos k_x x) (1 - cos k_y y) to lower the top sites cannot reach it while
both DFT barriers remain saddle points.
"""

LITHIUM_COPPER_110_SYSTEM_1D = CosineSystem1D(
    gamma=LITHIUM_COPPER_GAMMA,
    temperature=140,
    barrier_energy=LITHIUM_COPPER_110_TROUGH_ENERGY,
    delta_x=LITHIUM_COPPER_SPACING,
    m=LITHIUM_MASS,
)
"""The cut through LITHIUM_COPPER_110_SYSTEM_2D along a trough, the easy direction."""

LITHIUM_COPPER_110_ACROSS_SYSTEM_1D = CosineSystem1D(
    gamma=LITHIUM_COPPER_GAMMA,
    temperature=140,
    barrier_energy=LITHIUM_COPPER_110_ACROSS_ENERGY,
    delta_x=LITHIUM_COPPER_110_ROW_SPACING,
    m=LITHIUM_MASS,
)
"""The cut through LITHIUM_COPPER_110_SYSTEM_2D across the rows, the hard direction."""

HYDROGEN_MASS = 1.008 * atomic_mass

HYDROGEN_COPPER_BRIDGE_ENERGY = 136e-3 * electron_volt
"""The fcc to hcp hollow barrier of hydrogen on Cu(111).

The highest image of the DFT (PBE) nudged elastic band path of W. Fang et al.,
Phys. Rev. Lett. 119, 126001 (2017), Fig. 1a, read from the vector graphics of
the figure. The hcp site lies 14 meV above the fcc site (Table S3), which is
neglected here.
"""

HYDROGEN_COPPER_GAMMA = 1.5e13
"""The friction of hydrogen on Cu(111), about 15 ps^-1.

An upper bound from the HREELS linewidth of the parallel vibration, P. S. M.
Townsend, PhD thesis, University of Cambridge (2017), Section 6.5.6.
"""

HYDROGEN_COPPER_SPACING = 2.558e-10
"""The nearest-neighbor Cu-Cu spacing of Cu(111) used for hydrogen."""

HYDROGEN_COPPER_SYSTEM_2D = build_2d_fcc(
    gamma=HYDROGEN_COPPER_GAMMA,
    temperature=150,
    barrier_energy=HYDROGEN_COPPER_BRIDGE_ENERGY,
    delta_x=HYDROGEN_COPPER_SPACING,
    m=HYDROGEN_MASS,
)
"""Hydrogen on Cu(111). Fang et al. give no top site energy, so it is 9 times the bridge."""

HYDROGEN_COPPER_SYSTEM_1D = CosineSystem1D(
    gamma=HYDROGEN_COPPER_GAMMA,
    temperature=150,
    barrier_energy=HYDROGEN_COPPER_BRIDGE_ENERGY,
    delta_x=(1 / np.sqrt(3)) * HYDROGEN_COPPER_SPACING,
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

XENON_PLATINUM_SPACING = 2.775e-10
"""The nearest-neighbor Pt-Pt spacing of Pt(111)."""

XENON_PLATINUM_SYSTEM_2D = build_2d_triangular(
    gamma=5e10,
    temperature=121,
    barrier_energy=XENON_PLATINUM_BRIDGE_ENERGY,
    delta_x=XENON_PLATINUM_SPACING,
    m=131.293 * atomic_mass,
)

XENON_PLATINUM_SYSTEM_1D = CosineSystem1D(
    gamma=XENON_PLATINUM_SYSTEM_2D.gamma,
    temperature=XENON_PLATINUM_SYSTEM_2D.temperature,
    barrier_energy=XENON_PLATINUM_BRIDGE_ENERGY,
    delta_x=XENON_PLATINUM_SPACING,
    m=XENON_PLATINUM_SYSTEM_2D.m,
)
"""The cut through XENON_PLATINUM_SYSTEM_2D along a close-packed row."""
