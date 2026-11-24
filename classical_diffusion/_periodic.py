"""The geometry and potential of a periodic system, shared by the quantum (`caldeira_leggett`) and classical (`kramers`) solvers.

Each solver keeps its own natural units (the length unit differs), so these
take the length unit as an argument.
"""

# cspell: ignore ndimage
import itertools
from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage

if TYPE_CHECKING:
    from classical_diffusion._results import FloatArray, IntArray
    from classical_diffusion.langevin._system import PeriodicSystem

type IntVector = tuple[int, ...]
type Terms = tuple[tuple[IntVector, float], ...]
"""Integer vectors, each with a weight: the Fourier components of a potential, or the directions of the couplings."""


def get_reciprocal(system: PeriodicSystem, length: float) -> FloatArray:
    """Get the primitive reciprocal lattice vectors as rows, in units of 1 / length: a_i . g_j = 2 pi delta_ij."""
    return 2 * np.pi * np.linalg.inv(system.lattice_vectors).T * length


def get_potential_terms(system: PeriodicSystem) -> Terms:
    """Get the Fourier components V_G of the potential in units of k_B T, a cos(g . x) = (a / 2) (e^{i g . x} + e^{-i g . x}), without its constant."""
    terms = list[tuple[IntVector, float]]()
    for term in system.terms:
        half = term.amplitude / (2 * system.kbt)
        terms.extend([(term.miller, half), (tuple(-n for n in term.miller), half)])
    return tuple(terms)


def get_potential_on_grid(potential: Terms, n_dim: int, n_grid: int) -> FloatArray:
    """Get V(u) = sum_G V_G e^{2 pi i G . u} on a grid of n_grid points along each fractional coordinate u of one unit cell."""
    u = np.stack(
        np.meshgrid(*[np.arange(n_grid) / n_grid] * n_dim, indexing="ij"), axis=-1
    )
    out = np.zeros(u.shape[:-1], dtype=np.complex128)
    for g, v_g in potential:
        out += v_g * np.exp(2j * np.pi * (u @ np.array(g, dtype=float)))
    return out.real


def get_vectors_within(reciprocal: FloatArray, radius: float) -> IntArray:
    """Get every integer vector c with |c @ reciprocal| up to radius, (n, n_dim), in no particular order."""
    n_dim = reciprocal.shape[0]
    # |c @ reciprocal| >= s_min |c|, where s_min is the smallest singular value
    s_min = np.linalg.svd(reciprocal, compute_uv=False).min()
    n_max = int(np.ceil(radius / s_min))
    grid = np.array(
        list(itertools.product(range(-n_max, n_max + 1), repeat=n_dim)), dtype=np.int64
    ).reshape(-1, n_dim)
    return grid[np.linalg.norm(grid @ reciprocal, axis=-1) <= radius]


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
    largest = max((max(abs(c) for c in g) for g, _ in potential), default=1)
    n_grid = max(64 if n_dim < 3 else 24, 8 * largest)  # ruff:ignore[magic-value-comparison]
    values = get_potential_on_grid(potential, n_dim, n_grid)
    low, high = float(values.min()), float(values.max())
    for _ in range(40):
        middle = 0.5 * (low + high)
        if _is_percolating(values <= middle):
            high = middle
        else:
            low = middle
    return high, float(values.min())
