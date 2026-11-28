"""A periodic system in the natural units of its classical phase space, with the plane waves of one unit cell."""

# cspell: ignore ifftn fftn
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from classical_diffusion._periodic import (
    IntVector,
    get_potential_on_grid,
    get_potential_terms,
    get_reciprocal,
    get_vectors_within,
)
from classical_diffusion.system import unit_with_trajectory_scale

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion._periodic import Terms
    from classical_diffusion._results import ComplexArray, FloatArray, IntArray
    from classical_diffusion.langevin._system import PeriodicSystem
    from classical_diffusion.system import UnitSystem


@dataclass(frozen=True, kw_only=True, eq=False)
class Model:
    """A periodic system in natural units, m = k_B T = 1, with the length unit a / 2 pi of the first lattice vector a.

    The thermal velocity is 1. The classical equilibrium is the Gibbs state
    e^{-V(x)} e^{-p^2 / 2}, and every quantity here is found in the
    symmetrised frame, where a density is written e^{-V / 2} times its
    amplitude. A distribution e^{-i K x} h(x, p), with h periodic, keeps its
    wavevector K under the dynamics (Bloch's theorem), so one unit cell holds
    it at any K.
    """

    units: UnitSystem
    reciprocal: FloatArray
    """The primitive reciprocal lattice vectors, as rows, (n_dim, n_dim)."""
    potential: Terms
    """The Fourier components V_G of the potential in units of k_B T, for G = c @ reciprocal, with V_{-G} = V_G."""
    friction: float

    @property
    def n_dim(self) -> int:
        """The number of spatial dimensions."""
        return self.reciprocal.shape[0]


def build_model(system: PeriodicSystem) -> Model:
    """Build the model of a periodic system, in natural units."""
    kbt = system.kbt
    length = float(np.linalg.norm(system.lattice_vectors[0])) / (2 * np.pi)
    velocity = np.sqrt(kbt / system.m)
    units = unit_with_trajectory_scale(
        system.units,
        length=length,
        time=length / velocity,
        momentum=system.m * velocity,
    )
    reciprocal = get_reciprocal(system, length)
    potential = list(get_potential_terms(system))
    # The constant of the system puts the minimum of its potential at zero,
    # which keeps e^{-V / 2} of order one in the wells
    potential.append(((0,) * len(reciprocal), complex(system.params[-1] / kbt)))
    return Model(
        units=units,
        reciprocal=reciprocal,
        potential=_merge(potential),
        friction=float(system.units.frequency_into(system.gamma, units)),
    )


def _merge(terms: list[tuple[IntVector, complex]]) -> Terms:
    out = dict[IntVector, complex]()
    for g, v in terms:
        out[g] = out.get(g, 0.0) + v
    return tuple((g, v) for g, v in out.items() if v != 0)


def get_rows(array: IntArray) -> list[tuple[int, ...]]:
    """Get the rows of an integer array, as tuples of ints."""
    return [tuple(row) for row in array.tolist()]


def get_box_vectors(n_max: int, n_dim: int) -> IntArray:
    """Get every integer vector with components in [-n_max, n_max], (n, n_dim)."""
    return np.array(
        list(itertools.product(range(-n_max, n_max + 1), repeat=n_dim)), dtype=np.int64
    ).reshape(-1, n_dim)


def get_grid_function(
    model: Model, n_grid: int, f: Callable[[FloatArray], FloatArray]
) -> ComplexArray:
    """Get the Fourier coefficients c_G, G in [-n_grid / 2, n_grid / 2)^d, of f(V(u)) for u on a grid of the unit cell, by FFT."""
    values = get_potential_on_grid(model.potential, model.n_dim, n_grid)
    return np.fft.fftn(f(values)) / n_grid**model.n_dim


def get_fourier_coefficients(
    model: Model,
    vectors: IntArray,
    f: Callable[[FloatArray], FloatArray],
    *,
    n_grid: int,
) -> ComplexArray:
    """Get the Fourier coefficients of f(V(x)) at the integer vectors (in the order given), from a grid of n_grid points along each axis."""
    coefficients = get_grid_function(model, n_grid, f)
    return coefficients[tuple(np.mod(vectors, n_grid).T)]


def _get_amplitude_spectrum(model: Model, n_max: int) -> tuple[IntArray, FloatArray]:
    """Get the integer vectors of the box [-n_max, n_max]^d and |w_G| / |w| of the equilibrium amplitude w = e^{-V / 2} there."""
    n_grid = 4 * n_max + 4
    vectors = get_box_vectors(n_max, model.n_dim)
    full = get_grid_function(model, n_grid, lambda v: np.exp(-v / 2))
    norm = np.sqrt(np.sum(np.abs(full) ** 2))
    return vectors, np.abs(full[tuple(np.mod(vectors, n_grid).T)]) / norm


def get_plane_waves(model: Model, tol: float) -> IntArray:
    """Get the plane waves of the unit cell which hold the equilibrium amplitude e^{-V / 2} to tol, as integer vectors (n, n_dim), shortest first.

    The plane waves are those with |G| up to a radius, the smallest at which
    the norm of the components of w = e^{-V / 2} beyond it is below tol
    times its norm. The modes which carry the ISF are as smooth as w, the
    slow ones smoother.
    """
    n_max = 4
    while True:
        vectors, spectrum = _get_amplitude_spectrum(model, n_max)
        lengths = np.linalg.norm(vectors @ model.reciprocal, axis=-1)
        order = np.argsort(lengths, kind="stable")
        # The norm of what lies beyond each radius
        tail = np.sqrt(np.cumsum((spectrum[order] ** 2)[::-1])[::-1])
        radii = lengths[order]
        ok = np.flatnonzero(tail < tol)
        # The radius must be well inside the box, where every vector is kept
        inner = (
            np.min(np.linalg.norm(model.reciprocal, axis=-1))
            * n_max
            / np.sqrt(model.n_dim)
        )
        if len(ok) > 0 and radii[ok[0]] < 0.7 * inner:
            radius = radii[ok[0] - 1] if ok[0] > 0 else 0.0
            keep = lengths <= radius + 1e-12
            kept = vectors[keep]
            return kept[np.argsort(lengths[keep], kind="stable")]
        n_max *= 2


def get_plane_waves_within(model: Model, radius: float) -> IntArray:
    """Get the plane waves |G| <= radius, as integer vectors, shortest first."""
    vectors = get_vectors_within(model.reciprocal, radius + 1e-12)
    lengths = np.linalg.norm(vectors @ model.reciprocal, axis=-1)
    return vectors[np.argsort(lengths, kind="stable")]
