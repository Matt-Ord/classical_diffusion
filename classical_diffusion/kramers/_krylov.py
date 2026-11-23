# cspell: ignore Gragg orthogonalizing REORTHOGONALIZE orthogonalize vdot Golub Kahan bidiagonalization bidiagonal
"""The Krylov methods of the classical solvers: a correlation [u, f(A) u] as a modal sum, the spaces of the time steps, and Ritz pairs.

The Kramers operator in its real form is J-symmetric, L^T = J L J for J the
parity of the momentum, and so is (s - L)^-1: Arnoldi. The generator of the
overdamped limit is -L = D_0 A^dag A (J = 1): Golub-Kahan bidiagonalization
of A, which keeps the slow rates to round off relative to themselves.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from classical_diffusion._modal import ModalSum

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion._results import ComplexArray

_AGREEMENTS = 2
"""The successive blocks over which a modal sum must agree: a slow mode of small weight can leave one block unchanged before the method finds it."""
_INVARIANT = 1e-13
"""The norm of the next Krylov vector, relative to the projected operator, below which the Krylov space is invariant and the modes exact."""
_REORTHOGONALIZE = 0.7
"""The fraction of |w| left after Gram-Schmidt below which it is done a second time (Daniel, Gragg, Kaufman and Stewart 1976)."""


@dataclass(kw_only=True)
class _Arnoldi:
    """An orthonormal basis V of the Krylov space of u, and A in it, A V = V H + h v e_m^T."""

    apply: Callable[[np.ndarray], np.ndarray]
    vectors: np.ndarray
    """The basis as rows, with spare rows beyond the first `steps` + 1."""
    hessenberg: np.ndarray
    """H, of the dtype of the vectors, with spare rows and columns."""
    steps: int = 0
    invariant: bool = False

    def extend(self, steps: int) -> None:
        """Take steps more, orthogonalizing in full, twice where the first pass cancelled most of w."""
        for _ in range(steps):
            k = self.steps
            if k + 1 >= len(self.vectors):
                grown = np.zeros(
                    (2 * (k + 1), self.vectors.shape[1]), self.vectors.dtype
                )
                grown[: len(self.vectors)] = self.vectors
                self.vectors = grown
            if k + 1 >= self.hessenberg.shape[0] or k >= self.hessenberg.shape[1]:
                grown = np.zeros((2 * k + 2, 2 * k + 1), self.hessenberg.dtype)
                rows, cols = self.hessenberg.shape
                grown[:rows, :cols] = self.hessenberg
                self.hessenberg = grown
            basis = self.vectors[: k + 1]
            w = self.apply(basis[k])
            before = float(np.linalg.norm(w))
            overlap = basis.conj() @ w
            w -= overlap @ basis
            if np.linalg.norm(w) < _REORTHOGONALIZE * before:
                again = basis.conj() @ w
                w -= again @ basis
                overlap += again
            self.hessenberg[: k + 1, k] = overlap
            norm = float(np.linalg.norm(w))
            self.hessenberg[k + 1, k] = norm
            self.steps += 1
            if norm <= _INVARIANT * max(float(np.abs(overlap).max()), 1e-300):
                self.invariant = True
                return
            self.vectors[k + 1] = w / norm

    def poles(self, measurement: np.ndarray) -> tuple[ComplexArray, ComplexArray]:
        """Get the eigenvalues mu_j of H and the residues r_j, with <measurement, f(A) v_0> = sum_j r_j f(mu_j) for f a polynomial of degree below m."""
        m = self.steps
        mu, vectors = scipy.linalg.eig(self.hessenberg[:m, :m])
        first = np.zeros(m, complex)
        first[0] = 1
        seen = self.vectors[:m] @ measurement.conj()
        return mu, (seen @ vectors) * np.linalg.solve(vectors, first)


@dataclass(kw_only=True)
class _GolubKahan:
    """Orthonormal bases V and U with A V = U B, B upper bidiagonal: the Lanczos process of A^dag A in the Krylov space of v_0, with V^dag A^dag A V = B^dag B (Golub and Kahan 1965).

    The eigenvalues of A^dag A in the space are the squared singular values
    of B, each found to an absolute error of round off times |A| rather
    than |A^dag A|: a rate 1e-15 of the largest keeps a relative error of
    1e-8 (a cosine of 30 k_B T in the overdamped limit), where it would be
    lost to round off from A^dag A itself.
    """

    apply: Callable[[np.ndarray], np.ndarray]
    adjoint: Callable[[np.ndarray], np.ndarray]
    vectors: list[np.ndarray]
    """v_j."""
    images: list[np.ndarray]
    """u_j."""
    diagonal: list[float]
    """alpha_j = B_{jj}."""
    upper: list[float]
    """beta_j = B_{j, j+1}."""
    steps: int = 0
    invariant: bool = False

    @staticmethod
    def _orthogonalize(w: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
        """Remove the components of w along the basis, twice where the first pass cancelled most of it."""
        stacked = np.array(basis)
        before = float(np.linalg.norm(w))
        w -= (stacked.conj() @ w) @ stacked
        if np.linalg.norm(w) < _REORTHOGONALIZE * before:
            w -= (stacked.conj() @ w) @ stacked
        return w

    def extend(self, steps: int) -> None:
        """Take steps more, orthogonalizing each new vector in full against its basis."""
        for _ in range(steps):
            k = self.steps
            if k > 0:
                r = self._orthogonalize(self.adjoint(self.images[-1]), self.vectors)
                beta = float(np.linalg.norm(r))
                if beta <= _INVARIANT * max(self.diagonal[-1], 1e-300):
                    self.invariant = True
                    return
                self.upper.append(beta)
                self.vectors.append(r / beta)
            p = self.apply(self.vectors[-1])
            p = self._orthogonalize(p, self.images) if self.images else p
            alpha = float(np.linalg.norm(p))
            self.steps += 1
            scale = max(self.upper[-1] if self.upper else 0.0, alpha, 1e-300)
            self.diagonal.append(alpha)
            if alpha <= _INVARIANT * scale:
                self.invariant = True
                return
            self.images.append(p / alpha)

    def poles(self, measurement: np.ndarray) -> tuple[ComplexArray, ComplexArray]:
        """Get the eigenvalues sigma_j^2 of B^dag B and the residues r_j, with <measurement, f(A^dag A) v_0> = sum_j r_j f(sigma_j^2) for f a polynomial of degree below m."""
        m = self.steps
        bidiagonal = np.diag(self.diagonal[:m]) + np.diag(self.upper[: m - 1], 1)
        _, singular, right = scipy.linalg.svd(bidiagonal)
        vectors = right.conj().T
        seen = np.array(self.vectors[:m]) @ measurement.conj()
        return (singular**2).astype(complex), (seen @ vectors) * vectors[0].conj()


@dataclass(frozen=True, kw_only=True)
class KrylovProblem:
    """A correlation C = [u, f(A) u] = <J u, f(A) u>, as a modal sum of the rates rates_of(mu) of the eigenvalues mu of A in its Krylov space."""

    apply: Callable[[np.ndarray], np.ndarray]
    parity: np.ndarray | None
    """J, or None for J = 1."""
    rates_of: Callable[[ComplexArray], ComplexArray]
    """The rate of each eigenvalue mu of A: -mu for A = -L, shift - 1 / mu for A = (shift - L)^-1."""
    span: float
    """The longest time of the correlation, over which a pole may grow by the tolerance (`ModalSum.of`)."""
    error: Callable[[ModalSum, ModalSum], float]
    """The change between two modal sums, relative to the correlation (at t = 0, or of what is found from it)."""
    tolerance: float
    """The relative change, and the weight of the poles dropped relative to |C(0)|, below which the modal sum has converged."""
    block: int = 5
    max_steps: int = 5000
    adjoint: Callable[[np.ndarray], np.ndarray] | None = None
    """A^dag, where the operator whose modes are sought is A^dag A: then the modes are found by Golub-Kahan bidiagonalization of A (`_GolubKahan`), and rates_of maps the eigenvalues of A^dag A."""


def get_modal_sum(problem: KrylovProblem, start: np.ndarray) -> ModalSum:
    """Get C = [u, f(A) u] as a modal sum, by the Arnoldi method, extended a block at a time until _AGREEMENTS blocks in a row change it by less than the tolerance.

    The basis is orthonormal, so the method cannot break down (it ends at
    an invariant space, where the modes are exact) and no mode is found
    twice. The J-symmetry would allow Lanczos in the indefinite form
    x^T J y, which matches 2m moments in m steps; but at low friction its
    J-normalized vectors grow without bound (|w| = 6e4 at [w, w] = 1.5e-3
    for the mean square displacement of a cosine of 2.2 k_B T at 0.01 times
    the friction of lithium on Cu(111)) until [w, w] is round off, also
    from split starts, where Arnoldi converges in 1,800 solves. Arnoldi
    needs 1.2-3.5 times the solves of J-Lanczos where both converge.

    Raises
    ------
    RuntimeError
        If it does not converge within max_steps.
    """
    measurement = start if problem.parity is None else problem.parity * start
    size = float(np.linalg.norm(start))
    if size == 0:
        empty = np.zeros(0, complex)
        return ModalSum(rates=empty, residues=empty, stable=np.zeros(0, bool))
    scale = abs(float(np.vdot(measurement, start).real))
    arnoldi: _Arnoldi | _GolubKahan
    if problem.adjoint is None:
        vectors = np.zeros((min(16, start.size + 1), start.size), start.dtype)
        vectors[0] = start / size
        arnoldi = _Arnoldi(
            apply=problem.apply,
            vectors=vectors,
            hessenberg=np.zeros((17, 16), start.dtype),
        )
    else:
        arnoldi = _GolubKahan(
            apply=problem.apply,
            adjoint=problem.adjoint,
            vectors=[start / size],
            images=[],
            diagonal=[],
            upper=[],
        )
    previous: ModalSum | None = None
    agreed = 0
    while arnoldi.steps < problem.max_steps:
        # Blocks grow with the basis, so the O(m^3) poles are found a few times
        steps = max(problem.block, arnoldi.steps // 16)
        arnoldi.extend(min(steps, start.size - arnoldi.steps))
        mu, residues = arnoldi.poles(measurement)
        # A zero eigenvalue of (shift - L)^-1 is a mode of infinite rate, with no weight
        current = ModalSum.of(
            problem.rates_of(np.where(mu == 0, 1e-300, mu)),
            residues * size,
            problem.span,
            problem.tolerance * scale,
        )
        if arnoldi.invariant or arnoldi.steps >= start.size:
            return current
        agreed = (
            agreed + 1
            if previous is not None
            and max(problem.error(previous, current), current.dropped / scale)
            < problem.tolerance
            else 0
        )
        if agreed >= _AGREEMENTS:
            return current
        previous = current
    msg = f"The Arnoldi method did not converge within {problem.max_steps} steps"
    raise RuntimeError(msg)


def get_arnoldi_space(
    apply: Callable[[np.ndarray], np.ndarray], u: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Get an orthonormal basis V of the Krylov space of m vectors of u, as rows, L in it, the Hessenberg H with one more row, and |u|, with u = |u| V e_0.

    The space ends early, with a square H, where it is invariant.
    """
    size = float(np.linalg.norm(u))
    vectors = np.zeros((m + 1, u.size), u.dtype)
    vectors[0] = u / size
    arnoldi = _Arnoldi(
        apply=apply, vectors=vectors, hessenberg=np.zeros((m + 1, m), u.dtype)
    )
    arnoldi.extend(m)
    k = arnoldi.steps
    if arnoldi.invariant:
        return arnoldi.vectors[:k], arnoldi.hessenberg[:k, :k], size
    return arnoldi.vectors[: k + 1], arnoldi.hessenberg[: k + 1, :k], size


def get_slow_pairs(  # ruff:ignore[too-many-arguments]
    apply: Callable[[np.ndarray], np.ndarray],
    propagate: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    *,
    duration: float,
    rate: float,
    count: int,
    residual: float,
) -> tuple[ComplexArray, ComplexArray]:
    """Get the eigenvalues and vectors (lambda, r) of A that start holds with Re lambda > -rate, and at least the count slowest, as Ritz pairs of E = e^{A tau} (`propagate`).

    E maps the slow modes near 1 and the fast ones near 0, so its Arnoldi
    space of start finds the slow modes in a few vectors, where that of A
    itself needs hundreds: A spreads its spectrum over |lambda| ~ 100 times
    the slow rates. The space grows a vector at a time until each wanted
    pair has |A r - lambda r| <= residual |r|, with lambda the Rayleigh
    quotient of A. The vectors are rows.

    Raises
    ------
    RuntimeError
        If the pairs do not converge within _MAX_SLOW_STEPS vectors.
    """
    size = float(np.linalg.norm(start))
    vectors = np.zeros((count + 8, start.size))
    vectors[0] = start / size
    arnoldi = _Arnoldi(
        apply=propagate, vectors=vectors, hessenberg=np.zeros((count + 8, count + 7))
    )
    while arnoldi.steps < _MAX_SLOW_STEPS:
        arnoldi.extend(1)
        m = arnoldi.steps
        mu, ritz = scipy.linalg.eig(arnoldi.hessenberg[:m, :m])
        order = np.argsort(-np.abs(mu))
        wanted = order[
            : max(count, int(np.count_nonzero(np.abs(mu) > np.exp(-rate * duration))))
        ]
        pairs = ritz[:, wanted].T @ arnoldi.vectors[:m]
        images = np.array([apply(r.real) + 1j * apply(r.imag) for r in pairs])
        norms = np.sum(np.abs(pairs) ** 2, axis=1)
        values = np.sum(pairs.conj() * images, axis=1) / norms
        errors = np.linalg.norm(images - values[:, None] * pairs, axis=1)
        converged = len(wanted) >= count and np.all(errors <= residual * np.sqrt(norms))
        if arnoldi.invariant or converged:
            return values, pairs
    msg = f"The slow modes did not converge within {_MAX_SLOW_STEPS} vectors"
    raise RuntimeError(msg)


_MAX_SLOW_STEPS = 60
"""The most vectors of the Arnoldi space of e^{A tau} for the slow modes: those of lithium on Cu(111) at 10 times its friction converge in 10."""
