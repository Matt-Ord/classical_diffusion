# cspell: ignore Arnoldi matvec maxiter jkab jkac
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse.linalg

from classical_diffusion.caldeira_leggett._liouvillian import (
    SectorLiouvillian,
    get_twist_derivative,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.caldeira_leggett._liouvillian import BandOperators
    from classical_diffusion.caldeira_leggett._model import ComplexArray, FloatArray

logger = logging.getLogger(__name__)


def _get_rk4_stable_step(rate: complex) -> float:
    """Get the largest h for which RK4 is stable for every step up to h, |R(h' rate)| <= 1 for h' <= h.

    R(z) = 1 + z + z^2 / 2 + z^3 / 6 + z^4 / 24, whose stability region
    reaches |z| = 2.78 on the negative real axis and 2.83 on the imaginary axis.
    """
    steps = np.linspace(0, 3 / abs(rate), 3001)[1:]
    z = steps * rate
    amplification = np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)
    unstable = np.flatnonzero(amplification > 1 + 1e-12)
    return float(steps[unstable[0] - 1] if unstable.size > 0 else steps[-1])


def _linear_operator(
    size: int, matvec: Callable[[ComplexArray], ComplexArray]
) -> scipy.sparse.linalg.LinearOperator:
    """Get a square complex LinearOperator from its matvec."""
    # LinearOperator.__new__ dispatches to a subclass which accepts matvec
    return scipy.sparse.linalg.LinearOperator(  # ty: ignore[missing-argument]
        (size, size),
        matvec=matvec,  # ty: ignore[unknown-argument]
        dtype=np.complex128,  # ty: ignore[parameter-already-assigned]
    )


@dataclass(frozen=True, kw_only=True)
class _BandFrame:
    """The eigenbasis K = V diag(mu) V^-1 of one set of band operators."""

    mu: ComplexArray
    vectors: ComplexArray
    inverse: ComplexArray
    jumps: ComplexArray
    """V^-1 L_j V, (n_ops, n_kappa, n_bands, n_bands)."""
    decay: ComplexArray | None

    @staticmethod
    def from_operators(operators: BandOperators, *, include_decay: bool) -> _BandFrame:
        effective_hamiltonian = operators.effective_hamiltonian
        decay = None
        if not include_decay:
            hermitian = 0.5 * (
                effective_hamiltonian + effective_hamiltonian.conj().swapaxes(-1, -2)
            )
            decay = effective_hamiltonian - hermitian
            effective_hamiltonian = hermitian
        mu, vectors = np.linalg.eig(effective_hamiltonian)
        inverse = np.linalg.inv(vectors)
        jumps = np.array(
            [
                inverse[operators.jump_targets[j]] @ operators.jumps[j] @ vectors
                for j in range(operators.jump_targets.shape[0])
            ]
        )
        return _BandFrame(
            mu=mu,
            vectors=vectors,
            inverse=inverse,
            jumps=jumps,
            decay=None if decay is None else inverse @ decay @ vectors,
        )


@dataclass(frozen=True, kw_only=True)
class JumpFrame:
    """A sector Liouvillian, in the eigenbasis of the effective hamiltonian.

    At each kappa, K = H - i/2 sum_j L_j^dagger L_j = V diag(mu) V^-1. Writing
    X[kappa] = V[kappa + s] Y[kappa] V[kappa]^dagger, the Liouvillian becomes

        L(Y) = Z * Y + sum_j D_j[kappa + s] Y D_j[kappa]^dagger

    where Z = -i (mu[kappa + s, a] - mu[kappa, b]^*) is elementwise and
    D_j = V^-1 L_j V. The no-jump evolution is therefore diagonal, which gives
    an exact preconditioner for the linear solves.
    """

    liouvillian: SectorLiouvillian
    ket: _BandFrame
    bra: _BandFrame
    rates: ComplexArray
    jumps_ket: ComplexArray
    jumps_bra: ComplexArray

    @staticmethod
    def from_liouvillian(
        liouvillian: SectorLiouvillian, *, include_decay: bool = True
    ) -> JumpFrame:
        """Build the frame of a sector Liouvillian.

        If include_decay is False, the frame is the eigenbasis of H, and the
        decay -1/2 {sum_j L_j^dagger L_j, rho} is applied together with the
        jumps, so that time propagation keeps the cancellation between the
        decay and jump terms exactly.
        """
        bra = _BandFrame.from_operators(liouvillian.bra, include_decay=include_decay)
        ket = (
            bra
            if liouvillian.ket is liouvillian.bra
            else _BandFrame.from_operators(liouvillian.ket, include_decay=include_decay)
        )
        index = liouvillian.ket_index
        return JumpFrame(
            liouvillian=liouvillian,
            ket=ket,
            bra=bra,
            rates=-1j * (ket.mu[index][:, :, None] - bra.mu.conj()[:, None, :]),
            jumps_ket=np.ascontiguousarray(ket.jumps[:, index]),
            jumps_bra=np.ascontiguousarray(bra.jumps.conj().swapaxes(-1, -2)),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of a state in this sector."""
        return self.liouvillian.shape

    def to_frame(self, x: ComplexArray) -> ComplexArray:
        """Convert a state X into the frame."""
        index = self.liouvillian.ket_index
        return self.ket.inverse[index] @ x @ self.bra.inverse.conj().swapaxes(-1, -2)

    def from_frame(self, y: ComplexArray) -> ComplexArray:
        """Convert a state Y out of the frame."""
        index = self.liouvillian.ket_index
        return self.ket.vectors[index] @ y @ self.bra.vectors.conj().swapaxes(-1, -2)

    def observable_to_frame(self, a: ComplexArray) -> ComplexArray:
        """Get A' such that sum(A^* X) = sum(A'^* Y)."""
        index = self.liouvillian.ket_index
        return self.ket.vectors[index].conj().swapaxes(-1, -2) @ a @ self.bra.vectors

    def apply(self, y: ComplexArray) -> ComplexArray:
        """Apply the Liouvillian in the frame."""
        out = self.rates * y
        if self.bra.decay is not None and self.ket.decay is not None:
            index = self.liouvillian.ket_index
            out -= 1j * (
                self.ket.decay[index] @ y - y @ self.bra.decay.conj().swapaxes(-1, -2)
            )
        targets = self.liouvillian.jump_targets
        for j in range(targets.shape[0]):
            out[targets[j]] += self.jumps_ket[j] @ y @ self.jumps_bra[j]
        return out

    def _operator(self) -> scipy.sparse.linalg.LinearOperator:
        return _linear_operator(
            int(np.prod(self.shape)),
            lambda v: self.apply(v.reshape(self.shape)).reshape(-1),
        )

    def solve(self, rhs: ComplexArray, *, rtol: float = 1e-10) -> ComplexArray:
        """Solve L Y = rhs with GMRES, preconditioned by the inverse of the no-jump part of L."""
        inverse_rates = (1 / self.rates).reshape(-1)
        preconditioner = _linear_operator(
            int(np.prod(self.shape)), lambda v: inverse_rates * v
        )
        out, info = scipy.sparse.linalg.gmres(
            self._operator(),
            rhs.reshape(-1),
            M=preconditioner,
            rtol=rtol,
            atol=0,
            restart=200,
            maxiter=50,
        )
        if info != 0:
            logger.warning("GMRES did not converge (info=%s)", info)
        return out.reshape(self.shape)

    @cached_property
    def time_step(self) -> float:
        """The RK4 time step: half the largest stable step for the eigenvalues of L of largest magnitude.

        RK4 propagates with a polynomial in L, so it keeps the eigenvectors of
        L and only shifts each rate lambda by O(h^4 lambda^5): the slow rates
        are exact at any stable step, and the error is in the fast coherences
        at short times. The largest eigenvalues (the coherences of the highest
        bands) are found to 1e-3 by Arnoldi iteration.
        """
        extreme = scipy.sparse.linalg.eigs(
            self._operator(), k=6, which="LM", tol=1e-3, return_eigenvectors=False
        )
        return 0.5 * min(_get_rk4_stable_step(complex(z)) for z in extreme)

    def evolve(self, y: ComplexArray, h: float, n_steps: int) -> ComplexArray:
        """Evolve Y by n_steps steps of h, using the RK4 method."""
        y = y.copy()
        for _ in range(n_steps):
            a = self.apply(y)
            b = self.apply(y + 0.5 * h * a)
            c = self.apply(y + 0.5 * h * b)
            d = self.apply(y + h * c)
            y += (h / 6) * (a + 2 * (b + c) + d)
        return y

    def propagate(
        self, y: ComplexArray, times: FloatArray, observable: ComplexArray
    ) -> ComplexArray:
        """Get sum(observable^* Y(t)) at each of the increasing times from 0, using the RK4 method.

        In each interval the step is reduced from the stable step so that it
        divides the interval.
        """
        observable = observable.conj()
        out = np.zeros(times.size, dtype=np.complex128)
        out[0] = np.sum(observable * y)
        for i, interval in enumerate(np.diff(times).tolist(), start=1):
            n_sub = max(1, int(np.ceil(interval / self.time_step - 1e-9)))
            y = self.evolve(y, interval / n_sub, n_sub)
            out[i] = np.sum(observable * y)
        return out


def get_steady_state(liouvillian: SectorLiouvillian) -> ComplexArray:
    """Get the steady state in the basis of the bands, using preconditioned GMRES starting from a Gibbs state."""
    frame = JumpFrame.from_liouvillian(liouvillian)
    energies = liouvillian.bands.energies
    gibbs = np.zeros(liouvillian.shape, dtype=np.complex128)
    diagonal = np.arange(liouvillian.shape[1])
    gibbs[:, diagonal, diagonal] = np.exp(-(energies - energies.min()))
    gibbs /= np.sum(gibbs[:, diagonal, diagonal])

    y_gibbs = frame.to_frame(gibbs)
    # L is singular, but -L rho_gibbs is in its range. The null component of
    # the correction only changes the normalization, which is restored below
    correction = frame.solve(-frame.apply(y_gibbs))
    rho = frame.from_frame(y_gibbs + correction)
    rho /= np.sum(rho[:, diagonal, diagonal])
    rho = 0.5 * (rho + rho.conj().swapaxes(-1, -2))
    residual = np.linalg.norm(liouvillian.apply(rho)) / np.linalg.norm(rho)
    logger.info("Steady state residual %.2e", residual)
    return rho


def get_isf(
    sector: SectorLiouvillian, steady_state: ComplexArray, times: FloatArray
) -> ComplexArray:
    """Get I(dK, t) = Tr[e^{i dK x} V(t) (rho_ss e^{-i dK x})] in the scattering sector of dK, by time propagation.

    The scattering operator is that of the point of the reciprocal lattice of
    the cell of the sector, n = -shift; the rest of dK is in the offset of the
    grid of the bra.
    """
    frame = JumpFrame.from_liouvillian(sector, include_decay=False)
    # S[kappa] = <kappa - dK, m| e^{-i dK x} |kappa, n>
    scattering = sector.bands.shift_operator(sector.shift)
    # (rho_ss e^{-i dK x}) at the block (kappa - dK, kappa) is rho[kappa - dK] S[kappa]
    initial = frame.to_frame(steady_state[sector.ket_index] @ scattering)
    return frame.propagate(initial, times, frame.observable_to_frame(scattering))


def get_diffusion_coefficient(
    liouvillian: SectorLiouvillian, steady_state: ComplexArray, direction: FloatArray
) -> float:
    """Get the diffusion coefficient D along a unit direction, from the curvature of the dephasing rate at dK = 0.

    After the gauge transformation of `get_scattering_sector`, the ISF at a
    small dK = theta n is I(t) = Tr[exp(L(theta) t) rho_ss], where L(theta) is
    the Liouvillian with a bra on the grid offset by theta n. Its slowest
    eigenvalue is -D theta^2 + O(theta^4), and second order perturbation
    theory around the steady state, with L(theta) = L_0 + theta L_1 +
    theta^2 L_2 + ..., gives

        -D = Tr[L_2 rho_ss] - Tr[L_1 L_0^+ L_1 rho_ss].

    This is a Green-Kubo formula for the master equation, which needs a single
    linear solve rather than an ISF which decays only slowly at small dK.

    With the derivatives O' of the operators of the bra (`get_twist_derivative`),

        L_1 X = i X K'^dagger + sum_j L_j X L_j'^dagger.

    L_2 needs no second derivative: the Liouvillian with both the ket and the
    bra on the offset grid preserves the trace for every theta, so the second
    derivative of its trace vanishes, which gives
    Re Tr[L_2 rho_ss] = -1/2 sum_j Tr[L_j' rho_ss L_j'^dagger].
    """
    operators = liouvillian.bra
    jumps, hamiltonian = get_twist_derivative(operators, direction)
    jumps_dagger = jumps.conj().swapaxes(-1, -2)
    hamiltonian_dagger = hamiltonian.conj().swapaxes(-1, -2)

    def first_derivative(x: ComplexArray) -> ComplexArray:
        out = 1j * x @ hamiltonian_dagger
        for j in range(operators.jump_targets.shape[0]):
            out[operators.jump_targets[j]] += operators.jumps[j] @ x @ jumps_dagger[j]
        return out

    def trace(x: ComplexArray) -> complex:
        return complex(np.trace(x, axis1=-2, axis2=-1).sum())

    second_order = -0.5 * np.einsum(
        "jkab,kbc,jkac->", jumps, steady_state, jumps.conj()
    )
    drift = first_derivative(steady_state)
    # Tr[L_1 rho_ss] is the first order rate, which vanishes without a net drift
    first_order = trace(drift)
    frame = JumpFrame.from_liouvillian(liouvillian)
    response = frame.from_frame(
        frame.solve(frame.to_frame(drift - first_order * steady_state))
    )
    rate = second_order - trace(first_derivative(response))
    logger.info(
        "First order rate %.2e, second order rate %.6e", abs(first_order), rate.real
    )
    return -rate.real
