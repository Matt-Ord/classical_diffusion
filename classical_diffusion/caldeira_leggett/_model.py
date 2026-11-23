# cspell: ignore eigvalsh ndimage triu
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.ndimage
import scipy.optimize

from classical_diffusion.system import unit_with_trajectory_scale

if TYPE_CHECKING:
    from classical_diffusion.caldeira_leggett._config import CaldeiraLeggettConfig
    from classical_diffusion.langevin._system import PeriodicSystem
    from classical_diffusion.system import UnitSystem

type IntVector = tuple[int, ...]
type FloatArray = np.ndarray[Any, np.dtype[np.float64]]
type IntArray = np.ndarray[Any, np.dtype[np.int64]]
type ComplexArray = np.ndarray[Any, np.dtype[np.complex128]]
type Terms = tuple[tuple[IntVector, float], ...]
"""Integer vectors, each with a weight: the Fourier components of a potential, or the directions of the couplings."""


@dataclass(frozen=True, kw_only=True, eq=False)
class Model:
    """A periodic Caldeira-Leggett model, in natural units.

    The units are those in which hbar = m = k_B T = 1. The unit of length is
    then the coherence length of a free particle, hbar / sqrt(m k_B T): its
    thermal state has the coherence <x + s| rho |x> ~ e^{-s^2 / 2}.

    The simulation cell holds n_periods periods of the potential along each
    lattice vector, so its crystal momenta are the integer vectors kappa of
    the reciprocal lattice of the cell, modulo n_periods. Every crystal
    momentum has the plane waves k = kappa @ cell_reciprocal + G @ reciprocal,
    for the same reciprocal lattice vectors G of the potential, in the same
    order.
    """

    units: UnitSystem
    """The natural units, in which hbar = m = k_B T = 1."""
    cell_reciprocal: FloatArray
    """The primitive reciprocal lattice vectors of the simulation cell, as rows."""
    n_periods: int
    potential: Terms
    """The Fourier components V_G of the potential, for G = sum_i c_i g_i."""
    friction: float
    couplings: Terms
    """The direction c and weight w of each pair of Lindblad operators e^{+-i q x}, q = c @ cell_reciprocal."""
    reciprocal_vectors: IntArray
    """The reciprocal lattice vectors G of the potential in the plane wave basis, (n_plane_waves, n_dim)."""
    saddle_energy: float
    """The energy at which the classically allowed region first connects a site to its periodic images (`get_saddle_energy`)."""
    cutoff: float
    """Every band whose lowest energy is within this (k_B T) of the saddle, or the ground state if it is higher, is kept (`get_bands`)."""
    plane_wave_factor: float
    """The plane waves reach this times the kinetic energy that a particle at the cutoff can have, its height above the minimum of the potential."""

    @property
    def n_dim(self) -> int:
        """The number of spatial dimensions."""
        return self.cell_reciprocal.shape[0]

    @property
    def n_kappa(self) -> int:
        """The number of crystal momenta of the cell."""
        return self.n_periods**self.n_dim

    @property
    def reciprocal(self) -> FloatArray:
        """The primitive reciprocal lattice vectors of the potential, as rows."""
        return self.cell_reciprocal * self.n_periods

    @property
    def kappa(self) -> IntArray:
        """The crystal momenta, (n_kappa, n_dim) (`get_kappa`)."""
        return get_kappa(self.n_periods, self.n_dim)


def get_kappa(n_periods: int, n_dim: int) -> IntArray:
    """Get the crystal momenta of a cell of n_periods along each axis, (n_kappa, n_dim), each in the range closest to zero.

    They are in the order of their flat index, the C order of kappa mod n_periods.
    """
    index = np.unravel_index(np.arange(n_periods**n_dim), (n_periods,) * n_dim)
    return np.mod(np.stack(index, axis=-1) + n_periods // 2, n_periods) - n_periods // 2


MAX_PLANE_WAVES = 5000
"""The most plane waves at one crystal momentum.

The Hamiltonian in them is dense, and building and diagonalizing it takes
about 50 n^2 bytes: 1.2 GB at the limit. A system which needs more is
classical, and has thousands of bands: xenon on Pt(111) in 2D at 121 K
needs 10,411 already in the first guess of the cutoff, which took over
3 GB in 7 s, before the first band was found.
"""


def get_reciprocal_vectors(reciprocal: FloatArray, radius: float) -> IntArray:
    """Get the integer vectors G of every reciprocal lattice vector with |G @ reciprocal| up to radius.

    Raises
    ------
    ValueError
        If there are more than MAX_PLANE_WAVES of them.
    """
    n_dim = reciprocal.shape[0]
    # |k| >= s_min |G|, where s_min is the smallest singular value
    s_min = np.linalg.svd(reciprocal, compute_uv=False).min()
    n_max = int(np.ceil(radius / s_min))
    grid = np.array(list(itertools.product(range(-n_max, n_max + 1), repeat=n_dim)))
    vectors = grid[np.linalg.norm(grid @ reciprocal, axis=-1) <= radius]
    if len(vectors) > MAX_PLANE_WAVES:
        msg = (
            f"The bands need {len(vectors)} plane waves, more than {MAX_PLANE_WAVES}: "
            "the system is too heavy or too hot for the band basis, which is "
            "then classical (use Langevin dynamics)"
        )
        raise ValueError(msg)
    return vectors


def get_potential(reciprocal_vectors: IntArray, potential: Terms) -> ComplexArray:
    """Get the potential in the plane waves, (n_plane_waves, n_plane_waves).

    It couples plane waves which differ by a reciprocal lattice vector, so
    it is the same at every crystal momentum.
    """
    difference = reciprocal_vectors[:, None, :] - reciprocal_vectors[None, :, :]
    coupling = np.zeros(difference.shape[:2], dtype=np.complex128)
    for g, v_g in potential:
        coupling[np.all(difference == np.array(g), axis=-1)] += v_g
    return coupling


def _get_candidate_directions(reciprocal: FloatArray, radius: float) -> IntArray:
    """Get the integer vectors c, up to sign, with |c @ reciprocal| <= radius."""
    n_dim = reciprocal.shape[0]
    # |c @ reciprocal| >= s_min |c|, where s_min is the smallest singular value
    s_min = np.linalg.svd(reciprocal, compute_uv=False).min()
    n_max = int(np.ceil(radius / s_min))
    grid = np.array(list(itertools.product(range(-n_max, n_max + 1), repeat=n_dim)))
    grid = grid[np.linalg.norm(grid @ reciprocal, axis=1) <= radius]
    # Keep one of each pair c, -c: the first non-zero component is positive
    first = np.array([row[np.flatnonzero(row)[0]] if row.any() else 0 for row in grid])
    return grid[first > 0]


def get_couplings(reciprocal: FloatArray) -> Terms:
    """Get the direction c and weight w of each pair of Lindblad operators.

    The operators e^{+-i q x} of the direction q = c @ reciprocal damp only
    the momentum along q, and for small displacements the friction acts
    through W = sum_j w_j n_j n_j^T, with n_j = q_j / |q_j|. It is isotropic
    when W is the identity. Each direction also adds an error in the friction
    which grows as |q_j|^2 (`get_friction_error`), so among the weights
    w_j >= 0 with W = 1 those of least sum_j w_j |q_j|^2 are chosen, a linear
    program over the reciprocal lattice vectors up to a radius, which is
    increased until the program has a solution. This gives the
    two axes, each with weight 1, of a square or rectangular lattice, and the
    three shortest directions, each with weight 2 / 3, of a hexagonal one, in
    any number of dimensions.

    Raises
    ------
    ValueError
        If no reciprocal lattice vectors up to 1.5^8 times the longest basis
        vector give isotropic friction.
    """
    n_dim = reciprocal.shape[0]
    upper = np.triu_indices(n_dim)
    radius = 1.01 * float(np.linalg.norm(reciprocal, axis=1).max())
    # Every direction is close to some lattice vector, so a large enough
    # radius always admits isotropic weights
    for _ in range(8):
        candidates = _get_candidate_directions(reciprocal, radius)
        q = candidates @ reciprocal
        lengths = np.linalg.norm(q, axis=1)
        unit = q / lengths[:, None]
        projectors = np.array([np.outer(n, n)[upper] for n in unit]).T
        result = scipy.optimize.linprog(
            lengths**2,
            A_eq=projectors,
            b_eq=np.eye(n_dim)[upper],
            bounds=(0, None),
            method="highs",
        )
        if result.success:
            break
        radius *= 1.5
    else:
        msg = "No short reciprocal lattice vectors give isotropic friction"
        raise ValueError(msg)
    weights = result.x
    keep = weights > 1e-9 * weights.max()
    return tuple(
        (tuple(int(i) for i in c), float(w))
        for c, w in zip(candidates[keep], weights[keep], strict=True)
    )


def get_friction_error(q: float) -> float:
    """Get the relative error in the friction from a Lindblad operator e^{i q x}.

    The operators e^{+-i q x} / q are a periodic stand-in for the coupling to
    x. They destroy the coherence between positions separated by s at a rate
    proportional to 2 (1 - cos(q s)) / q^2, rather than s^2. The error is taken
    at s = 1, the coherence length of a free particle, where it is close to
    q^2 / 12.
    """
    return float(1 - 2 * (1 - np.cos(q)) / q**2)


def get_period_count(
    reciprocal: FloatArray,
    couplings: Terms,
    friction_tol: float,
) -> int:
    """Get the fewest periods of the potential for which the friction error is below friction_tol.

    With n periods in the cell, the Lindblad operators have the wavevectors
    q = c @ reciprocal / n, and the error is that of the longest
    (`get_friction_error`).
    """
    q_max = max(float(np.linalg.norm(c @ reciprocal)) for c, _ in couplings)
    n_periods = 1
    # Below the first zero of the error at q = 2 pi it grows with q
    while q_max / n_periods > np.pi or (
        get_friction_error(q_max / n_periods) > friction_tol
    ):
        n_periods += 1
    return n_periods


def _get_potential_on_grid(potential: Terms, n_dim: int) -> FloatArray:
    """Get V(u) = sum_G V_G e^{2 pi i G . u} on a grid of the fractional coordinates u of one unit cell."""
    largest = max((max(abs(c) for c in g) for g, _ in potential), default=1)
    n_grid = max(64 if n_dim < 3 else 24, 8 * largest)  # ruff:ignore[magic-value-comparison]
    u = np.stack(
        np.meshgrid(*[np.arange(n_grid) / n_grid] * n_dim, indexing="ij"), axis=-1
    )
    out = np.zeros(u.shape[:-1], dtype=np.complex128)
    for g, v_g in potential:
        out += v_g * np.exp(2j * np.pi * (u @ np.array(g, dtype=float)))
    return out.real


def _is_percolating(allowed: np.ndarray) -> bool:
    """Get whether the region connects a point of the unit cell to its periodic image along some lattice vector.

    The region is tiled over 3 cells along each axis and labelled, and a
    point in the central cell is connected to its image when both have the
    same label.
    """
    n_dim, n_grid = allowed.ndim, allowed.shape[0]
    labels, _ = scipy.ndimage.label(np.tile(allowed, (3,) * n_dim))
    centre = tuple(slice(n_grid, 2 * n_grid) for _ in range(n_dim))
    for axis in range(n_dim):
        image = tuple(
            slice(2 * n_grid, 3 * n_grid) if i == axis else slice(n_grid, 2 * n_grid)
            for i in range(n_dim)
        )
        if ((labels[centre] == labels[image]) & allowed).any():
            return True
    return False


def get_saddle_energy(potential: Terms, n_dim: int) -> tuple[float, float]:
    """Get the lowest energy E at which the region V < E connects a site to its periodic image along some lattice vector, and the minimum of the potential.

    This is the energy of the saddle a particle must cross to diffuse along
    its easiest direction, found by bisection between the minimum and maximum
    of the potential on a grid of one unit cell, in any number of dimensions.
    Diffusion in a harder direction, such as across the troughs of an fcc(110)
    surface, is exponentially slower.
    """
    values = _get_potential_on_grid(potential, n_dim)
    low, high = float(values.min()), float(values.max())
    for _ in range(40):
        middle = 0.5 * (low + high)
        if _is_percolating(values <= middle):
            high = middle
        else:
            low = middle
    return high, float(values.min())


def build_model(
    system: PeriodicSystem,
    config: CaldeiraLeggettConfig,
    *,
    cutoff: float,
    plane_wave_factor: float,
) -> Model:
    """Build the model of a periodic system, in natural units, for bands up to the cutoff above the saddle (`Model.cutoff`).

    A state of energy E has a kinetic energy of at most E - V_min where it is
    classically allowed, and its plane wave components fall off beyond that,
    so the plane waves reach plane_wave_factor times the height of the
    cutoff above the minimum of the potential, measured from the saddle
    (the ground state, from which the cutoff is measured if it is higher, is
    not known until the bands are). How high the potential rises beyond that
    does not matter: a band below the top of the potential never reaches it.
    """
    kbt = system.kbt
    time = system.units.hbar / kbt
    length = system.units.hbar / np.sqrt(system.m * kbt)
    units = unit_with_trajectory_scale(
        system.units, length=length, time=time, momentum=system.m * length / time
    )

    # The rows g_i satisfy a_i . g_j = 2 pi delta_ij
    reciprocal = 2 * np.pi * np.linalg.inv(system.lattice_vectors).T * length
    potential = list[tuple[IntVector, float]]()
    for term in system.terms:
        # a cos(g . x) = (a / 2) (e^{i g . x} + e^{-i g . x})
        amplitude = term.amplitude / (2 * kbt)
        potential.extend(
            [(term.miller, amplitude), (tuple(-n for n in term.miller), amplitude)]
        )
    couplings = get_couplings(reciprocal)
    n_periods = get_period_count(reciprocal, couplings, config.friction_tol)
    saddle_energy, minimum = get_saddle_energy(
        tuple(potential), len(system.lattice_vectors)
    )
    plane_wave_energy = plane_wave_factor * (saddle_energy + cutoff - minimum)
    # Every crystal momentum shares the vectors G, so they reach the kinetic
    # energy cutoff around the crystal momentum furthest from zero. A shift
    # across the edge of the zone of the cell maps G to G + dG, with dG one of
    # the primitive vectors, so one more is added for every plane wave within
    # the cutoff to stay in the basis after a shift
    kappa = get_kappa(n_periods, len(reciprocal)) @ (reciprocal / n_periods)
    radius = (
        np.sqrt(2 * plane_wave_energy)
        + np.linalg.norm(kappa, axis=-1).max()
        + np.linalg.norm(reciprocal, axis=-1).max()
    )
    return Model(
        units=units,
        cell_reciprocal=reciprocal / n_periods,
        n_periods=n_periods,
        potential=tuple(potential),
        friction=system.units.frequency_into(system.gamma, units),
        couplings=couplings,
        reciprocal_vectors=get_reciprocal_vectors(reciprocal, radius),
        saddle_energy=saddle_energy,
        cutoff=cutoff,
        plane_wave_factor=plane_wave_factor,
    )
