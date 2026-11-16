import dataclasses
import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

if TYPE_CHECKING:
    from classical_diffusion.caldeira_leggett._model import (
        BlochModel,
        ComplexArray,
        FloatArray,
        IntArray,
        IntVector,
    )


def _get_kappa_index(n: IntArray, n_periods: int) -> IntArray:
    """Get the flat index of the crystal momentum of each integer vector n."""
    r = np.mod(n, n_periods)
    return np.ravel_multi_index(tuple(r.T), (n_periods,) * n.shape[-1])


@dataclass(frozen=True, kw_only=True)
class BlochBands:
    """The Bloch eigenstates of a model, at each crystal momentum of the cell."""

    model: BlochModel
    energies: FloatArray
    """The band energies, shape (n_kappa, n_bands)."""
    vectors: tuple[ComplexArray, ...]
    """The plane wave coefficients of each band, shape (n_plane_waves, n_bands)."""
    plane_waves: tuple[IntArray, ...]
    """The integer vectors n of the plane waves at each crystal momentum."""
    offset: FloatArray
    """The offset theta of the grid of crystal momenta, so the plane wave n has wavevector n @ cell_reciprocal + theta."""

    def wavevector(self, n: IntArray) -> FloatArray:
        """Get the cartesian wavevector of the integer vectors n."""
        return self.model.wavevector(n) + self.offset

    @property
    def n_kappa(self) -> int:
        """The number of crystal momenta."""
        return len(self.vectors)

    @property
    def n_bands(self) -> int:
        """The number of bands at each crystal momentum."""
        return self.energies.shape[1]

    @cached_property
    def lookup(self) -> tuple[dict[IntVector, int], ...]:
        """The index of each plane wave n, at each crystal momentum."""
        return tuple(
            {tuple(n): p for p, n in enumerate(pw.tolist())} for pw in self.plane_waves
        )

    def shift_index(self, shift: IntVector) -> IntArray:
        """Get the index of kappa + shift, for each kappa."""
        kappas = np.array(
            list(
                itertools.product(range(self.model.n_periods), repeat=self.model.n_dim)
            )
        )
        return _get_kappa_index(kappas + np.array(shift), self.model.n_periods)

    def shift_operator(self, shift: IntVector) -> ComplexArray:
        """Get the band matrix elements of the operator e^{i q x}, for q = shift @ cell_reciprocal.

        e^{i q x} maps the plane wave n to the plane wave n + shift. The
        returned array D has shape (n_kappa, n_bands, n_bands), where
        D[kappa] = <kappa + shift, m| e^{i q x} |kappa, n>.
        Plane waves shifted outside the basis are dropped.
        """
        target = self.shift_index(shift)
        out = np.zeros((self.n_kappa, self.n_bands, self.n_bands), dtype=np.complex128)
        for i in range(self.n_kappa):
            j = target[i]
            shifted = self.plane_waves[i] + np.array(shift)
            source, dest = [], []
            for p, n in enumerate(shifted.tolist()):
                q = self.lookup[j].get(tuple(n))
                if q is not None:
                    source.append(p)
                    dest.append(q)
            # <j, m| e^{i q x} |i, n> = sum_p U_j[dest_p, m]^* U_i[source_p, n]
            out[i] = self.vectors[j][dest].conj().T @ self.vectors[i][source]
        return out

    def velocity(self, direction: FloatArray) -> ComplexArray:
        """Get the band matrix elements of the velocity k . direction, shape (n_kappa, n_bands, n_bands).

        Shifting every wavevector by theta adds theta . k + theta^2 / 2 to the
        Hamiltonian, so this is the derivative of the Hamiltonian with respect
        to an offset of the grid along the direction, within the span of the
        bands (k . p perturbation theory).
        """
        return np.array(
            [
                u.conj().T @ ((self.wavevector(pw) @ direction)[:, None] * u)
                for u, pw in zip(self.vectors, self.plane_waves, strict=True)
            ]
        )

    def on_offset_grid(self, offset: FloatArray) -> BlochBands:
        """Get the Bloch states on the grid offset by theta = offset, within the span of these bands.

        Shifting every wavevector by theta adds theta . k + theta^2 / 2 to
        the Hamiltonian, which is diagonalized in the basis of these bands.
        The states on the offset grid therefore span exactly the same space,
        so no weight leaks out of the basis. Diagonalizing the Hamiltonian of
        the offset grid in the full plane wave basis instead gives a slightly
        different truncated space, and where the truncation splits a degenerate
        set of bands an arbitrarily different one.
        """
        offset = np.asarray(offset, float)
        energies, vectors = [], []
        for e, u, pw in zip(self.energies, self.vectors, self.plane_waves, strict=True):
            k = self.wavevector(pw)
            shift = k @ offset + 0.5 * offset @ offset
            hamiltonian = np.diag(e) + u.conj().T @ (shift[:, None] * u)
            e_offset, w = np.linalg.eigh(hamiltonian)
            energies.append(e_offset)
            vectors.append(u @ w)
        return BlochBands(
            model=self.model,
            energies=np.array(energies),
            vectors=tuple(vectors),
            plane_waves=self.plane_waves,
            offset=self.offset + offset,
        )


def _get_hamiltonian(model: BlochModel, plane_waves: IntArray) -> ComplexArray:
    """Get the Hamiltonian in the plane waves of one crystal momentum."""
    k = model.wavevector(plane_waves)
    hamiltonian = np.diag(0.5 * np.sum(k**2, axis=-1)).astype(np.complex128)
    # The difference between two plane waves at the same kappa is n_periods * G
    g_difference = (
        plane_waves[:, None, :] - plane_waves[None, :, :]
    ) // model.n_periods
    for g, v_g in model.potential:
        hamiltonian[np.all(g_difference == np.array(g), axis=-1)] += v_g
    return hamiltonian


_PLANE_WAVE_FACTOR = 2.5
_PLANE_WAVE_TOL = 1e-3


def get_plane_wave_energy(model: BlochModel) -> float:
    """Get the kinetic energy cutoff of the plane waves which resolves the bands up to model.max_energy.

    A state of energy E has a kinetic energy of at most E - V_min where it is
    classically allowed, and its plane wave components fall off beyond that.
    The highest band kept is max_energy above the higher of the ground state
    and the saddle energy (`get_bloch_bands`), so the cutoff is a multiple of
    its height above V_min. How high the potential rises beyond that does not
    matter: a band below the top of the potential never reaches it.

    With a factor of 2.5 the diffusion constant of lithium on Cu(111) at
    140 K is converged to 1e-8 and that of hydrogen at 150 K to 0.2%, where
    a cutoff of max_energy + 2 k_B T is wrong by 3e-4 and by a factor of 11.

    The zero point energy is that of the lowest band at zero crystal momentum,
    found in the plane waves of a single period of the potential, with a cutoff
    which is raised until it is consistent.
    """
    primitive = dataclasses.replace(model, n_periods=1)
    cutoff = _PLANE_WAVE_FACTOR * model.max_energy
    for _ in range(20):
        primitive = dataclasses.replace(primitive, plane_wave_energy=cutoff)
        ground = scipy.linalg.eigvalsh(
            _get_hamiltonian(primitive, primitive.plane_waves()),
            subset_by_index=(0, 0),
        )[0]
        previous, cutoff = (
            cutoff,
            _PLANE_WAVE_FACTOR
            * (
                max(float(ground), model.saddle_energy)
                + model.max_energy
                - model.potential_minimum
            ),
        )
        if abs(cutoff - previous) < _PLANE_WAVE_TOL * cutoff:
            break
    # Rounded, so that the hash of the model does not depend on the last digits
    # of the zero point energy, which can change with the number of threads
    return round(cutoff, 6)


def get_bloch_bands(model: BlochModel) -> BlochBands:
    """Diagonalize the Hamiltonian at each crystal momentum of the cell.

    Every band whose lowest energy, over the crystal momenta, is within
    model.max_energy of the higher of the ground state and the saddle energy
    is kept. Diffusion is carried by the states near the saddle, which a
    particle crosses to hop between sites, so the cutoff is measured from it:
    for hydrogen on Cu(111) in 1D at 10 times its mass, bands to 12 k_B T
    above the ground state, only 2 k_B T above the saddle, give D 30% low.
    The lowest energy of band b
    is below a cutoff exactly when some crystal momentum has more than b
    energies below it, so the number of bands is the most energies below the
    cutoff at any crystal momentum. Each crystal momentum has the same number
    of bands, so it is also limited by the fewest plane waves of any.
    """
    all_plane_waves = model.plane_waves()
    kappa = _get_kappa_index(all_plane_waves, model.n_periods)
    plane_waves = [all_plane_waves[kappa == i] for i in range(model.n_kappa)]
    eigenstates = [scipy.linalg.eigh(_get_hamiltonian(model, pw)) for pw in plane_waves]

    ground = min(e[0] for e, _ in eigenstates)
    cutoff = max(ground, model.saddle_energy) + model.max_energy
    n_bands = min(
        max(int(np.count_nonzero(e < cutoff)) for e, _ in eigenstates),
        *(e.size for e, _ in eigenstates),
    )
    return BlochBands(
        model=model,
        energies=np.array([e[:n_bands] for e, _ in eigenstates]),
        vectors=tuple(u[:, :n_bands] for _, u in eigenstates),
        plane_waves=tuple(plane_waves),
        offset=np.zeros(model.n_dim),
    )
