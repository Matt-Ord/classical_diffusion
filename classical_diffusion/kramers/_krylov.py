# cspell: ignore Gragg orthogonalizing REORTHOGONALIZE vdot
"""The Krylov method of the classical solvers, Arnoldi: a correlation [u, f(A) u] as a modal sum, the spaces of the time steps, and Ritz pairs.

The Kramers operator in its real form is J-symmetric, L^T = J L J for J the
parity of the momentum, and so is (s - L)^-1; the generator of the
overdamped limit is Hermitian (J = 1).
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
    vectors = np.zeros((min(16, start.size + 1), start.size), start.dtype)
    vectors[0] = start / size
    arnoldi = _Arnoldi(
        apply=problem.apply,
        vectors=vectors,
        hessenberg=np.zeros((17, 16), start.dtype),
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


def get_ritz_pairs(
    apply: Callable[[np.ndarray], np.ndarray], start: np.ndarray, m: int
) -> tuple[ComplexArray, ComplexArray, np.ndarray]:
    """Get the Ritz values, vectors (as rows) and residual norms |A r - lambda r| of A in the Arnoldi space of m vectors of start."""
    basis, hessenberg, _ = get_arnoldi_space(apply, start, m)
    k = hessenberg.shape[1]
    values, ritz = scipy.linalg.eig(hessenberg[:k, :k])
    ritz /= np.linalg.norm(ritz, axis=0)
    last = hessenberg[k, k - 1] if hessenberg.shape[0] > k else 0.0
    return values, ritz.T @ basis[:k], np.abs(last * ritz[-1])
