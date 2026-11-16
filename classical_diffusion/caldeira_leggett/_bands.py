# cspell: ignore eigh
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from classical_diffusion.caldeira_leggett._model import (
    get_hamiltonian,
)

if TYPE_CHECKING:
    from classical_diffusion.caldeira_leggett._model import (
        ComplexArray,
        FloatArray,
        IntArray,
        IntVector,
        Model,
    )


class Bands(NamedTuple):
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


def get_bands(model: Model) -> Bands:
    """Diagonalize the Hamiltonian at each crystal momentum of the cell.

    Every band whose lowest energy, over the crystal momenta, is within
    model.max_energy of the higher of the ground state and the saddle energy
    is kept. Diffusion is carried by the states near the saddle, which a
    particle crosses to hop between sites, so the cutoff is measured from it:
    for hydrogen on Cu(111) in 1D at 10 times its mass, bands to 12 k_B T
    above the ground state, only 2 k_B T above the saddle, give D 30% low.
    The lowest energy of band b is below a cutoff exactly when some crystal
    momentum has more than b energies below it, so the number of bands is the
    most energies below the cutoff at any crystal momentum.
    """
    energies, vectors = np.linalg.eigh(
        get_hamiltonian(
            model.kappa @ model.cell_reciprocal,
            model.reciprocal_vectors,
            model.reciprocal,
            model.potential,
        )
    )
    cutoff = max(energies.min(), model.saddle_energy) + model.max_energy
    n_bands = int(np.count_nonzero(energies < cutoff, axis=-1).max())
    return Bands(energies=energies[:, :n_bands], vectors=vectors[:, :, :n_bands])


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
