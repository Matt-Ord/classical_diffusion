# cspell: ignore eigh eigvalsh
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from classical_diffusion.caldeira_leggett._model import (
    get_hamiltonian,
)

if TYPE_CHECKING:
    from classical_diffusion._periodic import IntVector
    from classical_diffusion.caldeira_leggett._model import (
        ComplexArray,
        FloatArray,
        IntArray,
        Model,
    )


@dataclass(frozen=True, kw_only=True)
class Bands:
    """The Bloch eigenstates of a model, at each crystal momentum kappa of the cell.

    The plane waves of every crystal momentum are k = kappa + G for the same
    reciprocal lattice vectors G (`Model.reciprocal_vectors`), in the same
    order, so every crystal momentum is handled at once.
    """

    energies: FloatArray
    """The band energies, (n_kappa, n_bands)."""
    vectors: ComplexArray
    """The plane wave coefficients of each band, (n_kappa, n_plane_waves, n_bands)."""


def get_kappa_index(model: Model, n: IntArray) -> IntArray:
    """Get the flat index of the crystal momentum of each integer vector n."""
    shape = (model.n_periods,) * model.n_dim
    return np.ravel_multi_index(tuple(np.mod(n, model.n_periods).T), shape)


def get_shifted_kappa(model: Model, shift: IntVector) -> IntArray:
    """Get the flat index of kappa + shift, for each kappa."""
    return get_kappa_index(model, model.kappa + np.asarray(shift))


def _get_wavevectors(model: Model) -> FloatArray:
    """Get the wavevectors k = kappa + G of every plane wave, (n_kappa, n_plane_waves, n_dim)."""
    return (model.kappa @ model.cell_reciprocal)[
        :, None, :
    ] + model.reciprocal_vectors @ model.reciprocal


def _get_hamiltonian(model: Model, i: int) -> ComplexArray:
    """Get the plane wave Hamiltonian at the crystal momentum of index i."""
    return get_hamiltonian(
        (model.kappa @ model.cell_reciprocal)[i : i + 1],
        model.reciprocal_vectors,
        model.reciprocal,
        model.potential,
    )[0]


def get_energies(model: Model) -> FloatArray:
    """Get every energy of the plane wave Hamiltonian at each crystal momentum, (n_kappa, n_plane_waves)."""
    return np.array(
        [
            scipy.linalg.eigvalsh(_get_hamiltonian(model, i))
            for i in range(model.n_kappa)
        ]
    )


def get_band_count(model: Model, energies: FloatArray, cutoff: float) -> int:
    """Get the number of bands whose lowest energy, over the crystal momenta, is within the cutoff of the higher of the ground state and the saddle energy.

    Diffusion is carried by the states near the saddle, which a particle
    crosses to hop between sites, so the cutoff is measured from it. The
    lowest energy of band b is below a cutoff exactly when some crystal
    momentum has more than b energies below it, so the number of bands is the
    most energies below the cutoff at any crystal momentum.
    """
    top = max(energies.min(), model.saddle_energy) + cutoff
    return max(int(np.count_nonzero(energies < top, axis=-1).max()), 1)


def get_bands(model: Model, energies: FloatArray | None = None) -> Bands:
    """Diagonalize the Hamiltonian at each crystal momentum of the cell, and keep the bands within model.cutoff (`get_band_count`).

    The cutoff is chosen from the weight the steady state and each sector
    leave in the highest bands (`_caldeira_leggett._get_basis`). The plane
    wave Hamiltonian is diagonalized one crystal momentum at a time, first
    for its energies (unless given, `get_energies`), then for the vectors of
    the bands kept only, which is cheaper than all of its vectors and holds
    one Hamiltonian at once.
    """
    if energies is None:
        energies = get_energies(model)
    n_bands = get_band_count(model, energies, model.cutoff)
    kept = [
        scipy.linalg.eigh(_get_hamiltonian(model, i), subset_by_index=(0, n_bands - 1))
        for i in range(model.n_kappa)
    ]
    return Bands(
        energies=np.array([e for e, _ in kept]), vectors=np.array([v for _, v in kept])
    )


def get_next_cutoff(model: Model, energies: FloatArray, cutoff: float) -> float:
    """Get the cutoff raised, if it keeps no more bands than model.cutoff, to just keep one more."""
    n_bands = get_band_count(model, energies, model.cutoff)
    if (
        get_band_count(model, energies, cutoff) > n_bands
        or n_bands >= energies.shape[1]
    ):
        return cutoff
    bottom = float(np.sort(energies, axis=-1)[:, n_bands].min())
    return bottom - max(energies.min(), model.saddle_energy) + 1e-6


def get_edge_weight(model: Model, bands: Bands) -> FloatArray:
    """Get the weight of each band on the outermost plane waves, those within the longest primitive reciprocal lattice vector of the largest |k|, (n_kappa, n_bands).

    A band which the plane waves resolve has none there.
    """
    norms = np.linalg.norm(_get_wavevectors(model), axis=-1)
    outer = norms > norms.max() - np.linalg.norm(model.reciprocal, axis=-1).max()
    return np.sum(np.abs(bands.vectors) ** 2 * outer[..., None], axis=1)


def get_basis_map(
    old: Model, old_bands: Bands, new: Model, new_bands: Bands
) -> ComplexArray:
    """Get <new, m|old, n> at each crystal momentum, (n_kappa, n_new, n_old), for plane waves of old which are all in new.

    Both models share the cell, so a state X in the old bands is
    W X W^dagger in the new ones, its projection onto their span.
    """
    index = {tuple(g): i for i, g in enumerate(new.reciprocal_vectors.tolist())}
    rows = np.array([index[tuple(g)] for g in old.reciprocal_vectors.tolist()])
    return new_bands.vectors[:, rows, :].conj().swapaxes(-1, -2) @ old_bands.vectors


def get_shift_operator(model: Model, bands: Bands, shift: IntVector) -> ComplexArray:
    """Get the band matrix elements of the operator e^{i q x}, for q = shift @ cell_reciprocal.

    e^{i q x} maps the plane wave kappa + G to kappa + shift + G, which is the
    plane wave kappa' + G + dG of the crystal momentum kappa' in its range,
    so it permutes the reciprocal lattice vectors. The returned array D has
    shape (n_kappa, n_bands, n_bands), where
    D[kappa] = <kappa + shift, m| e^{i q x} |kappa, n>. Plane waves shifted
    outside the basis are dropped.
    """
    target = get_shifted_kappa(model, shift)
    vectors = model.reciprocal_vectors
    n_plane_waves = len(vectors)
    shift_g = (model.kappa + np.asarray(shift) - model.kappa[target]) // model.n_periods
    # A table of the index of each G, over a box which holds every G + dG
    radius = int(np.abs(vectors).max()) + int(np.abs(shift_g).max())
    box = (2 * radius + 1,) * model.n_dim

    def flat(g: IntArray) -> IntArray:
        return np.ravel_multi_index(tuple(np.moveaxis(g + radius, -1, 0)), box)

    table = np.full(np.prod(box), n_plane_waves)
    table[flat(vectors)] = np.arange(n_plane_waves)
    index = table[flat(vectors[None, :, :] + shift_g[:, None, :])]
    # The plane wave of index n_plane_waves, outside the basis, is zero
    padded = np.concatenate(
        [bands.vectors, np.zeros_like(bands.vectors[:, :1])], axis=1
    )
    # <kappa', m| e^{i q x} |kappa, n> = sum_G U_kappa'[G + dG, m]^* U_kappa[G, n]
    shifted = padded[target[:, None], index]
    return shifted.conj().swapaxes(-1, -2) @ bands.vectors


def get_velocity(model: Model, bands: Bands, direction: FloatArray) -> ComplexArray:
    """Get the band matrix elements of the velocity k . direction, (n_kappa, n_bands, n_bands).

    Shifting every wavevector by theta adds theta . k + theta^2 / 2 to the
    Hamiltonian, so this is the derivative of the Hamiltonian with respect
    to an offset of the grid along the direction, within the span of the
    bands (k . p perturbation theory).
    """
    velocity = _get_wavevectors(model) @ direction
    return bands.vectors.conj().swapaxes(-1, -2) @ (velocity[..., None] * bands.vectors)
