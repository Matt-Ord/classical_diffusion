# cspell: ignore Daleckii Krein Davidovic digamma jkab jkac
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import scipy.special

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from classical_diffusion.caldeira_leggett._bands import BlochBands
    from classical_diffusion.caldeira_leggett._model import (
        ComplexArray,
        FloatArray,
        IntArray,
        IntVector,
    )

_DEGENERATE_GAP = 1e-7
"""Bands closer in energy than this (k_B T) are treated as degenerate."""
_DERIVATIVE_STEP = 1e-5


def _bath_amplitude(omega: FloatArray) -> FloatArray:
    """Get the bath amplitude g(omega) = sqrt(omega / (e^omega - 1)) of a transition of energy omega, with g(0) = 1.

    This is the spectral function of an Ohmic bath, as in the universal
    Lindblad equation (Nathan and Rudner, PRB 102, 115109). The rates of
    omega and -omega are in the ratio e^-omega, so detailed balance holds.
    """
    ratio = np.divide(omega, np.expm1(omega), out=np.ones_like(omega), where=omega != 0)
    return np.sqrt(ratio)


def _symmetric_spectrum(omega: FloatArray) -> FloatArray:
    """Get the symmetric part F(omega) = (omega / 2) coth(omega / 2) of the Ohmic bath spectrum, with F(0) = 1."""
    return np.divide(
        omega / 2, np.tanh(omega / 2), out=np.ones_like(omega), where=omega != 0
    )


def _symmetric_spectrum_hilbert(omega: FloatArray) -> FloatArray:
    """Get Sigma(omega) = (omega / 2 pi) Re psi(1 + i omega / 2 pi), the finite part of the Hilbert transform of F.

    The Hilbert transform (1 / 2 pi) P int F(x) / (omega - x) dx diverges
    with the cutoff of the bath, but only through a term linear in omega,
    whose contribution summed over the pair e^{+-i q x} is a constant.
    """
    return (
        omega
        / (2 * np.pi)
        * np.real(scipy.special.digamma(1 + 1j * omega / (2 * np.pi)))
    )


def _lamb_amplitude(omega: FloatArray) -> ComplexArray:
    """Get A(-omega), where A(nu) = F(nu) / 4i + Sigma(nu) / 2 for the energy nu given to the bath.

    With the one-sided bath correlation int_0^inf C(tau) e^{-i nu tau} d tau
    = F(nu) / 2 - i Sigma(nu), the Lamb shift (1 / 2i) int_0^inf C(tau)
    [B^dagger, B(-tau)] d tau of a coupling B is
    sum_l conj(B_lm) B_ln [A(E_n - E_l) + conj(A(E_m - E_l))] = M + M^dagger,
    with M = B^dagger (B o A) in the eigenbasis of H.
    """
    return np.asarray(
        _symmetric_spectrum(-omega) / 4j + _symmetric_spectrum_hilbert(-omega) / 2,
        dtype=np.complex128,
    )


@dataclass(frozen=True, kw_only=True)
class BandOperators:
    """The Lindblad operators and effective hamiltonian of a set of Bloch bands."""

    bands: BlochBands
    jumps: ComplexArray
    """The band matrix elements of each Lindblad operator, (n_ops, n_kappa, n_bands, n_bands)."""
    jump_targets: IntArray
    """The index of kappa + q for each Lindblad operator, (n_ops, n_kappa)."""
    effective_hamiltonian: ComplexArray
    """H - i/2 sum_j L_j^dagger L_j at each kappa, (n_kappa, n_bands, n_bands)."""


def _get_couplings(bands: BlochBands) -> Iterator[tuple[IntArray, ComplexArray]]:
    """Get the index of kappa + q and the coupling sqrt(4 lambda w) e^{i q x} / (2 |q|), for each Lindblad operator."""
    model = bands.model
    for direction, weight in model.couplings:
        for sign in (1, -1):
            c = tuple(sign * i for i in direction)
            q = model.wavevector(np.array(c))
            prefactor = np.sqrt(4 * model.friction * weight) / (2 * np.linalg.norm(q))
            yield bands.shift_index(c), prefactor * bands.shift_operator(c)


def _get_effective_hamiltonian(
    bands: BlochBands, jumps: ComplexArray, lamb: ComplexArray
) -> ComplexArray:
    gamma = np.einsum("jkab,jkac->kbc", jumps.conj(), jumps)
    return (
        np.einsum("ka,ab->kab", bands.energies, np.eye(bands.n_bands))
        + lamb
        - 0.5j * gamma
    )


def get_band_operators(bands: BlochBands) -> BandOperators:
    """Build the Lindblad operators of the model in the basis of the bands.

    The periodic Caldeira-Leggett operators couple to e^{+-i q x} in place of
    x, and have the band matrix elements

        <b| L_q |a> = sqrt(4 lambda w) / (2 |q|) <b| e^{i q x} |a> g(E_b - E_a)

    where g is the spectral function of an Ohmic bath (`_bath_amplitude`).
    Expanding g to first order in the energy gives the usual temperature
    correction of the Caldeira-Leggett equation; the full function has the
    same classical limit, but satisfies detailed balance (and so has a Gibbs
    steady state) when hbar omega > k_B T.

    The Hamiltonian H_U of the Bloch-Redfield equation is also kept, as in
    the universal Lindblad equation and the geometric-arithmetic master
    equation (Davidovic, Quantum 4, 326 (2020)). Its imaginary-time part
    is a constant for e^{+-i q x}, but the part from the symmetric spectrum is
    not, and is added to H at each kappa (`_lamb_amplitude`).
    """
    jumps, targets = [], []
    lamb = np.zeros((bands.n_kappa, bands.n_bands, bands.n_bands), dtype=np.complex128)
    for target, coupling in _get_couplings(bands):
        omega = bands.energies[target][:, :, None] - bands.energies[:, None, :]
        jumps.append(coupling * _bath_amplitude(omega))
        targets.append(target)
        m = coupling.conj().swapaxes(-1, -2) @ (coupling * _lamb_amplitude(omega))
        lamb += m + m.conj().swapaxes(-1, -2)
    jumps_arr = np.array(jumps)
    return BandOperators(
        bands=bands,
        jumps=jumps_arr,
        jump_targets=np.array(targets),
        effective_hamiltonian=_get_effective_hamiltonian(bands, jumps_arr, lamb),
    )


def _get_perturbed_function(  # ruff:ignore[too-many-arguments]
    function: Callable[[FloatArray], FloatArray | ComplexArray],
    coupling: ComplexArray,
    *,
    target_energies: FloatArray,
    energies: FloatArray,
    target_velocity: ComplexArray,
    velocity: ComplexArray,
) -> ComplexArray:
    """Get the derivative of B o f(E_b - E_a) when H gains theta v at each crystal momentum, in the basis at theta = 0.

    Here (B o f)_ba = B_ba f(E_b - E_a) in the eigenbasis of H, with b at the
    target crystal momentum and a at the source. Perturbing H rotates the
    eigenbasis and moves the energies, and for a fixed B the derivative is
    given by the divided differences of f (Daleckii and Krein):

        sum_c v_bc B_ca [f(E_b - E_a) - f(E_c - E_a)] / (E_b - E_c)
        + sum_c B_bc v_ca [f(E_b - E_a) - f(E_b - E_c)] / (E_a - E_c)

    Each sum separates into two matrix products, so the cost is that of
    building the operators. Between degenerate bands the divided difference
    is the derivative of f, found by a central difference.
    """
    omega = target_energies[:, :, None] - energies[:, None, :]
    f = function(omega)
    df = (function(omega + _DERIVATIVE_STEP) - function(omega - _DERIVATIVE_STEP)) / (
        2 * _DERIVATIVE_STEP
    )

    def split(v: ComplexArray, gap: FloatArray) -> tuple[ComplexArray, ComplexArray]:
        degenerate = np.abs(gap) < _DEGENERATE_GAP
        return (
            np.where(degenerate, 0, v / np.where(degenerate, 1, gap)),
            np.where(degenerate, v, 0),
        )

    # The gap E_b - E_c of the bands (b, c) at the target
    m, v_degenerate = split(
        target_velocity, target_energies[:, :, None] - target_energies[:, None, :]
    )
    out = f * (m @ coupling) - m @ (coupling * f) + df * (v_degenerate @ coupling)
    # The gap E_a - E_c of the bands (c, a) at the source
    m, v_degenerate = split(velocity, energies[:, None, :] - energies[:, :, None])
    return (
        out + f * (coupling @ m) - (coupling * f) @ m - df * (coupling @ v_degenerate)
    )


def get_twist_derivative(
    operators: BandOperators, direction: FloatArray
) -> tuple[ComplexArray, ComplexArray]:
    """Get the derivative of the Lindblad operators and effective hamiltonian with the offset of the grid along a direction.

    These are the derivatives at theta = 0 of e^{-i theta x} O e^{i theta x},
    the operators of `get_offset_grid_operators`, found analytically by k . p
    perturbation theory. In the basis of the bands e^{i q x} does not depend on
    the offset, and H gains theta k . direction, so each Lindblad operator
    and the Lamb shift changes only through the spectral function
    (`_get_perturbed_function`).
    """
    bands = operators.bands
    velocity = bands.velocity(direction)
    jumps = []
    hamiltonian = velocity.copy()
    for target, coupling in _get_couplings(bands):

        def perturbed(
            function: Callable[[FloatArray], FloatArray | ComplexArray],
            target: IntArray = target,
            coupling: ComplexArray = coupling,
        ) -> ComplexArray:
            return _get_perturbed_function(
                function,
                coupling,
                target_energies=bands.energies[target],
                energies=bands.energies,
                target_velocity=velocity[target],
                velocity=velocity,
            )

        jumps.append(perturbed(_bath_amplitude))
        m = coupling.conj().swapaxes(-1, -2) @ perturbed(_lamb_amplitude)
        hamiltonian += m + m.conj().swapaxes(-1, -2)
    jumps_arr = np.array(jumps)
    gamma = np.einsum("jkab,jkac->kbc", jumps_arr.conj(), operators.jumps)
    hamiltonian -= 0.5j * (gamma + gamma.conj().swapaxes(-1, -2))
    return jumps_arr, hamiltonian


@dataclass(frozen=True, kw_only=True)
class SectorLiouvillian:
    """The Liouvillian of the model, restricted to the sector rho_{kappa + s, kappa}.

    A state in the sector is stored as an array X of shape (n_kappa, n_bands, n_bands),
    where X[kappa] is the block <kappa + s, m| rho |kappa, n>. The ket and bra
    can use Bloch states on differently offset grids, which gives the sectors of
    scattering vectors that are not on the reciprocal lattice of the cell.
    """

    ket: BandOperators
    bra: BandOperators
    shift: IntVector
    ket_index: IntArray = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ket_index", self.bra.bands.shift_index(self.shift))

    @property
    def bands(self) -> BlochBands:
        """The bands of the bra."""
        return self.bra.bands

    @property
    def jump_targets(self) -> IntArray:
        """The index of kappa + q for each Lindblad operator, (n_ops, n_kappa)."""
        return self.bra.jump_targets

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of a state in this sector."""
        return (self.bands.n_kappa, self.bands.n_bands, self.bands.n_bands)


def get_sector_liouvillian(bands: BlochBands) -> SectorLiouvillian:
    """Build the Liouvillian of the sector of the steady state, rho_{kappa, kappa}."""
    operators = get_band_operators(bands)
    return SectorLiouvillian(
        ket=operators, bra=operators, shift=(0,) * bands.model.n_dim
    )


def get_offset_grid_operators(bands: BlochBands, offset: FloatArray) -> BandOperators:
    """Get the band operators e^{-i theta x} O e^{i theta x}, in the basis of the bands.

    Conjugating by e^{i theta x} shifts every wavevector by theta = offset,
    so these are the operators of the Bloch states on the offset grid
    (`BlochBands.on_offset_grid`), rotated back into the basis of the bands.
    Since both span the same space, the operators are a smooth function of
    the offset in a fixed basis.
    """
    offset_bands = bands.on_offset_grid(offset)
    operators = get_band_operators(offset_bands)
    # The band m on the offset grid is sum_n w[kappa, n, m] |kappa, n>
    w = np.array(
        [
            u.conj().T @ v
            for u, v in zip(bands.vectors, offset_bands.vectors, strict=True)
        ]
    )
    w_dagger = w.conj().swapaxes(-1, -2)
    return BandOperators(
        bands=bands,
        jumps=w[operators.jump_targets] @ operators.jumps @ w_dagger,
        jump_targets=operators.jump_targets,
        effective_hamiltonian=w @ operators.effective_hamiltonian @ w_dagger,
    )


def get_scattering_sector(
    liouvillian: SectorLiouvillian, delta_k: FloatArray
) -> SectorLiouvillian:
    """Get the scattering sector of dK, which contains rho e^{-i dK x}.

    dK is split into the nearest point n @ cell_reciprocal and an offset theta.
    Every operator shifts k by a multiple of cell_reciprocal, so k mod
    cell_reciprocal is conserved, and the sector is that of crystal momentum
    transfer -n. The offset is removed by the gauge transformation
    X -> X e^{i theta x}, after which the bra evolves under the operators of
    the Bloch states on the offset grid, e^{-i theta x} O e^{i theta x}
    (`get_offset_grid_operators`). The ISF is then Tr[e^{i (dK - theta) x} X(t)],
    in the basis of the bands, so any dK is reached with the same cell.
    """
    model = liouvillian.bands.model
    n = np.rint(np.linalg.solve(model.cell_reciprocal.T, delta_k)).astype(int)
    shift = tuple(-int(i) for i in n)
    offset = delta_k - model.wavevector(n)
    bra = (
        get_offset_grid_operators(liouvillian.bands, offset)
        if np.any(offset)
        else liouvillian.bra
    )
    return SectorLiouvillian(ket=liouvillian.ket, bra=bra, shift=shift)
