# cspell: ignore Daleckii Krein Davidovic digamma jkab jkac eigh
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

from classical_diffusion.caldeira_leggett._bands import (
    get_shift_operator,
    get_shifted_kappa,
    get_velocity,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.caldeira_leggett._bands import Bands
    from classical_diffusion.caldeira_leggett._model import (
        ComplexArray,
        FloatArray,
        IntArray,
        Model,
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
class Couplings:
    """The operators sqrt(4 lambda w) e^{i q x} / (2 |q|) of each Lindblad operator, in the basis of the bands."""

    matrices: ComplexArray
    """<kappa + q, m| . |kappa, n>, (n_ops, n_kappa, n_bands, n_bands)."""
    targets: IntArray
    """The index of kappa + q, (n_ops, n_kappa)."""


@dataclass(frozen=True, kw_only=True)
class Operators:
    """The Lindblad operators and effective Hamiltonian, in the basis of the bands."""

    jumps: ComplexArray
    """<kappa + q, m| L_q |kappa, n>, (n_ops, n_kappa, n_bands, n_bands)."""
    targets: IntArray
    """The index of kappa + q, (n_ops, n_kappa)."""
    hamiltonian: ComplexArray
    """H - i/2 sum_j L_j^dagger L_j at each kappa, (n_kappa, n_bands, n_bands)."""


def _dagger(x: ComplexArray) -> ComplexArray:
    return x.conj().swapaxes(-1, -2)


def _diagonal(energies: FloatArray) -> FloatArray:
    return energies[..., None] * np.eye(energies.shape[-1])


def get_couplings(model: Model, bands: Bands) -> Couplings:
    """Get the couplings of the pair of Lindblad operators e^{+-i q x} of each direction of the model."""
    matrices, targets = [], []
    for direction, weight in model.couplings:
        for sign in (1, -1):
            c = tuple(sign * i for i in direction)
            q = np.array(c) @ model.cell_reciprocal
            prefactor = np.sqrt(4 * model.friction * weight) / (2 * np.linalg.norm(q))
            matrices.append(prefactor * get_shift_operator(model, bands, c))
            targets.append(get_shifted_kappa(model, c))
    return Couplings(matrices=np.array(matrices), targets=np.array(targets))


def get_operators(energies: FloatArray, couplings: Couplings) -> Operators:
    """Build the Lindblad operators and effective Hamiltonian of bands of the given energies.

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
    omega = _get_omega(energies, couplings)
    return _build_operators(
        energies, couplings, _bath_amplitude(omega), _lamb_amplitude(omega)
    )


def _get_omega(energies: FloatArray, couplings: Couplings) -> FloatArray:
    """Get E_b - E_a for each coupling, b at its target and a at its source."""
    return energies[couplings.targets][..., :, None] - energies[:, None, :]


def _build_operators(
    energies: FloatArray,
    couplings: Couplings,
    bath: FloatArray,
    lamb_amplitude: ComplexArray,
) -> Operators:
    """Get the operators from the bath and Lamb amplitudes of each coupling (`get_operators`)."""
    coupling = couplings.matrices
    jumps = coupling * bath
    m = _dagger(coupling) @ (coupling * lamb_amplitude)
    lamb = np.sum(m + _dagger(m), axis=0)
    gamma = np.sum(_dagger(jumps) @ jumps, axis=0)
    hamiltonian = _diagonal(energies) + lamb - 0.5j * gamma
    return Operators(jumps=jumps, targets=couplings.targets, hamiltonian=hamiltonian)


def get_offset_operators(
    model: Model, bands: Bands, couplings: Couplings, offset: FloatArray
) -> Operators:
    """Get the operators e^{-i theta x} O e^{i theta x}, in the basis of the bands.

    Conjugating by e^{i theta x} shifts every wavevector by theta, which adds
    the perturbation theta . k + theta^2 / 2 to H. The operators are built in
    the eigenbasis of H plus the perturbation within the span of the bands,
    the Bloch states on the grid offset by theta, and rotated back. Since both
    span the same space, the operators are a smooth function of the offset in
    a fixed basis. Diagonalizing the Hamiltonian of the offset grid in the
    full plane wave basis instead gives a slightly different truncated space,
    and where the truncation splits a degenerate set of bands an arbitrarily
    different one.
    """
    perturbation = get_velocity(model, bands, offset) + 0.5 * offset @ offset * np.eye(
        bands.energies.shape[1]
    )
    offset_energies, w = np.linalg.eigh(_diagonal(bands.energies) + perturbation)
    # The band m on the offset grid is sum_n w[kappa, n, m] |kappa, n>
    w_target = w[couplings.targets]
    operators = get_operators(
        offset_energies,
        Couplings(
            matrices=_dagger(w_target) @ couplings.matrices @ w,
            targets=couplings.targets,
        ),
    )
    return Operators(
        jumps=w_target @ operators.jumps @ _dagger(w),
        targets=couplings.targets,
        hamiltonian=w @ operators.hamiltonian @ _dagger(w),
    )


def _get_perturbed_function(
    function: Callable[[FloatArray], FloatArray | ComplexArray],
    f: FloatArray | ComplexArray,
    couplings: Couplings,
    energies: FloatArray,
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
    is the derivative of f, found by a central difference. f is the function
    at E_b - E_a (`_get_omega`).
    """
    coupling = couplings.matrices
    target_energies = energies[couplings.targets]
    omega = _get_omega(energies, couplings)
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
        velocity[couplings.targets],
        target_energies[..., :, None] - target_energies[..., None, :],
    )
    out = f * (m @ coupling) - m @ (coupling * f) + df * (v_degenerate @ coupling)
    # The gap E_a - E_c of the bands (c, a) at the source
    m, v_degenerate = split(velocity, energies[:, None, :] - energies[:, :, None])
    return (
        out + f * (coupling @ m) - (coupling * f) @ m - df * (coupling @ v_degenerate)
    )


def get_twist_derivative(
    energies: FloatArray,
    couplings: Couplings,
    velocity: ComplexArray,
) -> tuple[Operators, Operators]:
    """Get the operators (`get_operators`) and their derivative with the offset of the grid along a direction, of velocity v.

    These are the derivatives at theta = 0 of e^{-i theta x} O e^{i theta x},
    the operators of `get_offset_operators`, found analytically by k . p
    perturbation theory. In the basis of the bands e^{i q x} does not depend on
    the offset, and H gains theta v, so each Lindblad operator and the Lamb
    shift changes only through the spectral function
    (`_get_perturbed_function`). The spectral functions at the energies of the
    bands are found once for both.
    """
    omega = _get_omega(energies, couplings)
    bath, lamb_amplitude = _bath_amplitude(omega), _lamb_amplitude(omega)
    operators = _build_operators(energies, couplings, bath, lamb_amplitude)
    jumps = _get_perturbed_function(
        _bath_amplitude, bath, couplings, energies, velocity
    )
    m = _dagger(couplings.matrices) @ _get_perturbed_function(
        _lamb_amplitude, lamb_amplitude, couplings, energies, velocity
    )
    gamma = np.sum(_dagger(jumps) @ operators.jumps, axis=0)
    hamiltonian = (
        velocity + np.sum(m + _dagger(m), axis=0) - 0.5j * (gamma + _dagger(gamma))
    )
    return operators, Operators(
        jumps=jumps, targets=couplings.targets, hamiltonian=hamiltonian
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class Liouvillian:
    """The Liouvillian of a sector rho_{kappa + s, kappa}, as a pytree of jax arrays.

    A state X has shape (n_kappa, n_bands, n_bands), where X[kappa] is the
    block <kappa + s, m| rho |kappa, n>, and

        L(X) = -i (K_ket X - X K_bra^dagger) + sum_j L_j,ket X L_j,bra^dagger

    where the jump j maps the block at kappa to kappa + q_j. The ket and bra
    can use Bloch states on differently offset grids, which gives the sectors
    of scattering vectors that are not on the reciprocal lattice of the cell.
    """

    hamiltonian_ket: jax.Array
    hamiltonian_bra_dagger: jax.Array
    jumps_ket: jax.Array
    """(n_ops, n_kappa, n_bands, n_bands), indexed by the kappa of the bra."""
    jumps_bra_dagger: jax.Array
    sources: jax.Array
    """(n_ops, n_kappa), the index of kappa - q_j for each jump and kappa, from which the jump reaches kappa."""


def get_liouvillian(
    ket: Operators, bra: Operators, shifted_kappa: IntArray | None = None
) -> Liouvillian:
    """Get the Liouvillian of the sector whose ket is at shifted_kappa, the index of kappa + s, by default kappa."""
    index = np.arange(len(bra.hamiltonian)) if shifted_kappa is None else shifted_kappa
    return Liouvillian(
        hamiltonian_ket=jnp.asarray(ket.hamiltonian[index]),
        hamiltonian_bra_dagger=jnp.asarray(_dagger(bra.hamiltonian)),
        jumps_ket=jnp.asarray(ket.jumps[:, index]),
        jumps_bra_dagger=jnp.asarray(_dagger(bra.jumps)),
        # Each jump permutes the crystal momenta, so its sources are the inverse
        sources=jnp.asarray(np.argsort(bra.targets, axis=-1)),
    )


@jax.jit
def apply_liouvillian(liouvillian: Liouvillian, x: jax.Array) -> jax.Array:
    """Apply the Liouvillian to the state x."""
    return add_jumps(
        -1j
        * (liouvillian.hamiltonian_ket @ x - x @ liouvillian.hamiltonian_bra_dagger),
        x,
        (liouvillian.jumps_ket, liouvillian.jumps_bra_dagger, liouvillian.sources),
    )


def add_jumps(
    out: jax.Array,
    x: jax.Array,
    jumps: tuple[jax.Array, jax.Array, jax.Array],
    *,
    gather: bool = False,
) -> jax.Array:
    """Add the jumps J X J'^dagger of a state to out, for the jumps (J, J'^dagger, sources) of a Liouvillian.

    The jumps are added one at a time, which holds a few states at once
    rather than one for each jump, at no cost in speed. If gather, each
    jump takes X from its index (J X[index] J'^dagger at each kappa) rather
    than putting the product there, as the jumps of the adjoint do.
    """

    def add_jump(
        out: jax.Array, jump: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, None]:
        ket, bra_dagger, index = jump
        if gather:
            return out + ket @ x[index] @ bra_dagger, None
        return out + (ket @ x @ bra_dagger)[index], None

    return jax.lax.scan(add_jump, out, jumps)[0]
