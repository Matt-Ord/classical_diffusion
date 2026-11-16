# cspell: ignore eigh
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy.linalg

from classical_diffusion.caldeira_leggett._model import (
    get_hamiltonian,
    get_plane_waves,
)

if TYPE_CHECKING:
    from classical_diffusion.caldeira_leggett._model import (
        BoolArray,
        ComplexArray,
        FloatArray,
        IntArray,
        IntVector,
        Model,
    )


class Bands(NamedTuple):
    """The Bloch eigenstates of a model, at each crystal momentum kappa of the cell.

    The plane waves of a crystal momentum are those n with n mod n_periods =
    kappa, and their number differs between crystal momenta. The arrays are
    padded to the largest number, with zero coefficients, so that every
    crystal momentum is handled at once.
    """

    energies: FloatArray
    """The band energies, (n_kappa, n_bands)."""
    vectors: ComplexArray
    """The plane wave coefficients of each band, (n_kappa, n_plane_waves, n_bands)."""
    plane_waves: IntArray
    """The integer vectors n of the plane waves, (n_kappa, n_plane_waves, n_dim)."""
    is_plane_wave: BoolArray
    """Whether each plane wave is in the basis, rather than padding, (n_kappa, n_plane_waves)."""


def get_kappa_index(model: Model, n: IntArray) -> IntArray:
    """Get the flat index of the crystal momentum of each integer vector n."""
    shape = (model.n_periods,) * model.n_dim
    return np.ravel_multi_index(tuple(np.mod(n, model.n_periods).T), shape)


def get_shifted_kappa(model: Model, shift: IntVector) -> IntArray:
    """Get the flat index of kappa + shift, for each kappa."""
    shape = (model.n_periods,) * model.n_dim
    kappa = np.stack(np.unravel_index(np.arange(model.n_kappa), shape), axis=-1)
    return get_kappa_index(model, kappa + np.asarray(shift))


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
    most energies below the cutoff at any crystal momentum. Each crystal
    momentum has the same number of bands, so it is also limited by the
    fewest plane waves of any.
    """
    all_plane_waves = get_plane_waves(model.cell_reciprocal, model.plane_wave_energy)
    kappa = get_kappa_index(model, all_plane_waves)
    plane_waves = [all_plane_waves[kappa == i] for i in range(model.n_kappa)]
    # The number of plane waves differs between crystal momenta, so each is
    # diagonalized on its own
    eigenstates = [
        scipy.linalg.eigh(
            get_hamiltonian(pw, model.cell_reciprocal, model.n_periods, model.potential)
        )
        for pw in plane_waves
    ]
    ground = min(e[0] for e, _ in eigenstates)
    cutoff = max(ground, model.saddle_energy) + model.max_energy
    n_bands = min(
        max(int(np.count_nonzero(e < cutoff)) for e, _ in eigenstates),
        *(e.size for e, _ in eigenstates),
    )

    n_plane_waves = max(len(pw) for pw in plane_waves)
    vectors = np.zeros((model.n_kappa, n_plane_waves, n_bands), dtype=np.complex128)
    padded = np.zeros((model.n_kappa, n_plane_waves, model.n_dim), dtype=np.int64)
    is_plane_wave = np.zeros((model.n_kappa, n_plane_waves), dtype=np.bool_)
    for i, (pw, (_, u)) in enumerate(zip(plane_waves, eigenstates, strict=True)):
        vectors[i, : len(pw)] = u[:, :n_bands]
        padded[i, : len(pw)] = pw
        is_plane_wave[i, : len(pw)] = True
    return Bands(
        energies=np.array([e[:n_bands] for e, _ in eigenstates]),
        vectors=vectors,
        plane_waves=padded,
        is_plane_wave=is_plane_wave,
    )


def get_shift_operator(bands: Bands, shift: IntVector) -> ComplexArray:
    """Get the band matrix elements of the operator e^{i q x}, for q = shift @ cell_reciprocal.

    e^{i q x} maps the plane wave n to the plane wave n + shift. The
    returned array D has shape (n_kappa, n_bands, n_bands), where
    D[kappa] = <kappa + shift, m| e^{i q x} |kappa, n>.
    Plane waves shifted outside the basis are dropped.
    """
    n_kappa, n_plane_waves, n_dim = bands.plane_waves.shape
    dropped = n_kappa * n_plane_waves
    # A table of the flat index kappa * n_plane_waves + p of each plane wave n,
    # over a box of n which holds every n + shift
    radius = int(np.abs(bands.plane_waves).max()) + int(np.abs(shift).max())
    box = (2 * radius + 1,) * n_dim

    def flat(n: IntArray) -> IntArray:
        return np.ravel_multi_index(tuple(np.moveaxis(n + radius, -1, 0)), box)

    table = np.full(np.prod(box), dropped)
    table[flat(bands.plane_waves)[bands.is_plane_wave]] = np.flatnonzero(
        bands.is_plane_wave
    )
    target = np.where(
        bands.is_plane_wave,
        table[flat(bands.plane_waves + np.asarray(shift))],
        dropped,
    )
    # The row of index dropped is zero
    rows = np.concatenate(
        [
            bands.vectors.reshape(dropped, -1),
            np.zeros((1, bands.vectors.shape[-1]), dtype=np.complex128),
        ]
    )
    # <kappa + shift, m| e^{i q x} |kappa, n> = sum_p U_{kappa + shift}[n_p + shift, m]^* U_kappa[n_p, n]
    return rows[target].conj().swapaxes(-1, -2) @ bands.vectors


def get_velocity(bands: Bands, model: Model, direction: FloatArray) -> ComplexArray:
    """Get the band matrix elements of the velocity k . direction, (n_kappa, n_bands, n_bands).

    Shifting every wavevector by theta adds theta . k + theta^2 / 2 to the
    Hamiltonian, so this is the derivative of the Hamiltonian with respect
    to an offset of the grid along the direction, within the span of the
    bands (k . p perturbation theory).
    """
    velocity = (bands.plane_waves @ model.cell_reciprocal) @ direction
    return bands.vectors.conj().swapaxes(-1, -2) @ (velocity[..., None] * bands.vectors)
