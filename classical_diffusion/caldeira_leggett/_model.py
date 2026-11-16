import hashlib
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from classical_diffusion.langevin._system import PeriodicSystem

type IntVector = tuple[int, ...]
type FloatArray = np.ndarray[Any, np.dtype[np.float64]]
type IntArray = np.ndarray[Any, np.dtype[np.int64]]
type ComplexArray = np.ndarray[Any, np.dtype[np.complex128]]

_MAX_DIM = 2


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


def _get_coupling_directions(reciprocal: FloatArray) -> tuple[IntVector, ...]:
    """Get the directions of the Lindblad operators, on the reciprocal lattice.

    In 2D these are the two primitive vectors and the shorter of their sum and
    difference, which are the three shortest vectors of a reduced lattice.
    """
    if reciprocal.shape[0] == 1:
        return ((1,),)
    diagonal = min(
        ((1, 1), (1, -1)), key=lambda c: float(np.linalg.norm(c @ reciprocal))
    )
    return ((1, 0), (0, 1), diagonal)


def get_couplings(reciprocal: FloatArray) -> tuple[tuple[IntVector, float], ...]:
    """Get the direction c and weight w of each pair of Lindblad operators.

    The operators e^{+-i q x} of the direction q damp only the momentum along
    q. The weights are those for which sum_j w_j q_j q_j^T / |q_j|^2 is the
    identity, so that the friction is isotropic: 2 / 3 for each of the three
    directions of a hexagonal lattice, and 1 for each axis of a rectangular one.

    Raises
    ------
    NotImplementedError
        If the system has more than two dimensions.
    ValueError
        If no positive weights make the friction isotropic, which happens when
        the lattice vectors are far from the shortest of the lattice.
    """
    n_dim = reciprocal.shape[0]
    if n_dim > _MAX_DIM:
        msg = "Only 1D and 2D systems are supported"
        raise NotImplementedError(msg)
    directions = _get_coupling_directions(reciprocal)
    unit = np.array([c @ reciprocal for c in directions], dtype=np.float64)
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    upper = np.triu_indices(n_dim)
    projectors = np.array([np.outer(u, u)[upper] for u in unit]).T
    weights = np.linalg.solve(projectors, np.eye(n_dim)[upper])
    if np.any(weights < -1e-9):  # ruff:ignore[magic-value-comparison]
        msg = "The friction cannot be made isotropic on these lattice vectors"
        raise ValueError(msg)
    return tuple(
        (c, float(w))
        for c, w in zip(directions, weights, strict=True)
        if w > 1e-9  # ruff:ignore[magic-value-comparison]
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
        length=float(length),
        time=float(time),
    )
