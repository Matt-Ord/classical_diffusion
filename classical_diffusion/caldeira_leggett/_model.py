import hashlib
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.ndimage
import scipy.optimize

if TYPE_CHECKING:
    from classical_diffusion.langevin._system import PeriodicSystem

type IntVector = tuple[int, ...]
type FloatArray = np.ndarray[Any, np.dtype[np.float64]]
type IntArray = np.ndarray[Any, np.dtype[np.int64]]
type ComplexArray = np.ndarray[Any, np.dtype[np.complex128]]


@dataclass(frozen=True, kw_only=True, eq=False)
class BlochModel:
    """A periodic Caldeira-Leggett model, in natural units.

    The units are those in which hbar = m = k_B T = 1. The unit of length is
    then the coherence length of a free particle, hbar / sqrt(m k_B T): its
    thermal state has the coherence <x + s| rho |x> ~ e^{-s^2 / 2}.

    The simulation cell holds n_periods periods of the potential along each
    lattice vector. Wavevectors are stored as integer vectors n on the
    reciprocal lattice of the cell, k = n @ cell_reciprocal. The crystal
    momentum of n is n mod n_periods, and n // n_periods is the reciprocal
    lattice vector G of the potential.
    """

    reciprocal: FloatArray
    """The primitive reciprocal lattice vectors g_i of the potential, as rows."""
    n_periods: int
    potential: tuple[tuple[IntVector, float], ...]
    """The Fourier components V_G of the potential, for G = sum_i c_i g_i."""
    friction: float
    couplings: tuple[tuple[IntVector, float], ...]
    """The direction c and weight w of each pair of Lindblad operators e^{+-i q x}, q = c @ cell_reciprocal."""
    plane_wave_energy: float
    """The kinetic energy cutoff of the plane wave basis."""
    max_energy: float
    """Every band whose lowest energy is within this of the ground state is kept."""
    potential_minimum: float
    """The minimum of the potential, which has no constant term."""
    saddle_energy: float
    """The energy at which the classically allowed region first connects a site to its periodic images (`get_saddle_energy`)."""
    length: float
    """The natural unit of length, in the units of the system."""
    time: float
    """The natural unit of time, in the units of the system."""

    def get_hash(self) -> str:
        """Get a hash of the model, which is the same in every process and unit system."""

        def rounded(values: Any) -> list[Any]:  # ruff:ignore[any-type]
            return [float(f"{v:.12e}") for v in np.ravel(values).tolist()]

        key = (
            rounded(self.reciprocal),
            self.n_periods,
            [(c, *rounded(v)) for c, v in self.potential],
            rounded(self.friction),
            [(c, *rounded(w)) for c, w in self.couplings],
            rounded([self.plane_wave_energy, self.max_energy]),
        )
        return hashlib.sha256(repr(key).encode()).hexdigest()[:24]

    @property
    def n_dim(self) -> int:
        """The number of spatial dimensions."""
        return self.reciprocal.shape[0]

    @property
    def n_kappa(self) -> int:
        """The number of crystal momenta of the cell."""
        return self.n_periods**self.n_dim

    @property
    def cell_reciprocal(self) -> FloatArray:
        """The reciprocal lattice vectors of the simulation cell, as rows."""
        return self.reciprocal / self.n_periods

    def wavevector(self, n: IntArray) -> FloatArray:
        """Get the cartesian wavevector of the integer vectors n."""
        return n @ self.cell_reciprocal

    def plane_waves(self) -> IntArray:
        """Get the integer vectors n of every plane wave in the basis."""
        k_max = np.sqrt(2 * self.plane_wave_energy)
        # |k| >= s_min |n|, where s_min is the smallest singular value
        s_min = np.linalg.svd(self.cell_reciprocal, compute_uv=False).min()
        n_max = int(np.ceil(k_max / s_min))
        grid = np.array(
            list(itertools.product(range(-n_max, n_max + 1), repeat=self.n_dim))
        )
        k = self.wavevector(grid)
        return grid[0.5 * np.sum(k**2, axis=-1) <= self.plane_wave_energy]


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


def get_couplings(reciprocal: FloatArray) -> tuple[tuple[IntVector, float], ...]:
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
    couplings: tuple[tuple[IntVector, float], ...],
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


def _get_potential_on_grid(
    potential: tuple[tuple[IntVector, complex], ...], n_dim: int
) -> FloatArray:
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


def get_saddle_energy(
    potential: tuple[tuple[IntVector, complex], ...], n_dim: int
) -> float:
    """Get the lowest energy E at which the region V < E connects a site to its periodic image along some lattice vector.

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
    return high


def build_model(
    system: PeriodicSystem,
    *,
    friction_tol: float,
    max_energy: float,
    plane_wave_energy: float,
) -> BlochModel:
    """Build the model of a periodic system, in natural units."""
    kbt = system.kbt
    time = system.units.hbar / kbt
    length = system.units.hbar / np.sqrt(system.m * kbt)

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
    return BlochModel(
        reciprocal=reciprocal,
        n_periods=get_period_count(reciprocal, couplings, friction_tol),
        potential=tuple(potential),
        friction=system.gamma * time,
        couplings=couplings,
        plane_wave_energy=plane_wave_energy,
        max_energy=max_energy,
        # The constant of the system puts the minimum of its potential at zero
        potential_minimum=-system.params[-1] / kbt,
        saddle_energy=get_saddle_energy(tuple(potential), len(system.lattice_vectors)),
        length=float(length),
        time=float(time),
    )
