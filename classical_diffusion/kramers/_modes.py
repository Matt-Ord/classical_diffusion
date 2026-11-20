"""A correlation <u| e^{L t} |u> of a J-symmetric operator as a sum of modes, by shift-invert Lanczos in the indefinite form x^T J y."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.kramers._model import ComplexArray, FloatArray


@dataclass(frozen=True, kw_only=True)
class Modes:
    """A correlation C(t) = sum_j c_j e^{lambda_j t}, the rates and weights in complex conjugate pairs, so C is real."""

    rates: ComplexArray
    """lambda_j."""
    weights: ComplexArray
    """c_j, with sum_j c_j = C(0)."""
    n_solves: int

    def _decaying(self) -> np.ndarray:
        # A Ritz value of a mode the start barely holds can land just right
        # of the axis, with no weight
        return self.rates.real < 0

    def __call__(self, times: FloatArray) -> FloatArray:
        """C(t) at the times."""
        keep = self.rates.real <= 0
        return (np.exp(np.outer(times, self.rates[keep])) @ self.weights[keep]).real

    def integral(self) -> float:
        """int_0^inf C(t) dt."""
        keep = self._decaying()
        return float(np.sum(self.weights[keep] / -self.rates[keep]).real)

    def double_integral(self, times: FloatArray) -> FloatArray:
        """int_0^t (t - u) C(u) du at the times."""
        keep = self._decaying()
        rates, weights = self.rates[keep], self.weights[keep]
        t = np.asarray(times)[:, None]
        return np.sum(
            weights * (t / -rates - (-np.expm1(rates * t)) / rates**2), axis=1
        ).real


def _get_modes(
    alpha: list[float], beta: list[float], signs: list[float], shift: float, norm: float
) -> Modes:
    """Get the modes from the tridiagonal T of (shift - L)^{-1} in the Lanczos basis.

    The basis V has V^T J V = diag(signs), and (shift - L)^{-1} V = V T, with
    T_{k, k} = alpha_k, T_{k+1, k} = beta_k and
    T_{k, k+1} = beta_k signs_{k+1} / signs_k. Each eigenvalue mu of T is a
    rate shift - 1 / mu, and its weight is (S)_{0 j} (S^{-1})_{j 0} times C(0).
    """
    m = len(alpha)
    upper = [beta[k] * signs[k + 1] / signs[k] for k in range(m - 1)]
    tridiagonal = np.diag(alpha) + np.diag(beta[: m - 1], -1) + np.diag(upper, 1)
    mu, vectors = scipy.linalg.eig(tridiagonal)
    left = scipy.linalg.solve(vectors, np.eye(m)[:, 0])
    # A zero eigenvalue of T is a mode at infinite rate, which has no weight
    mu = np.where(mu == 0, 1e-300, mu)
    return Modes(
        rates=shift - 1 / mu,
        weights=norm * signs[0] * vectors[0] * left,
        n_solves=m,
    )


def get_modes(  # ruff:ignore[too-many-arguments]
    solve: Callable[[FloatArray], FloatArray],
    start: FloatArray,
    parity: FloatArray,
    shift: float,
    *,
    converged: Callable[[Modes, Modes], bool],
    every: int = 5,
    max_solves: int = 5000,
) -> Modes:
    """Get the modes of C(t) = u^T J e^{L t} u from Lanczos on (shift - L)^{-1}, in the indefinite form x^T J y.

    L is J-symmetric, L^T = J L J, so it is self-adjoint in the form
    [x, y] = x^T J y, and so is (shift - L)^{-1}. The Lanczos method in this
    form builds a basis in which it is tridiagonal, with a three term
    recurrence, so it holds three vectors whatever the number of steps. The
    start u must have one parity, J u = +-u, so that
    C(t) = [u, e^{L t} u] = +-u^T e^{L t} u.

    The shift maps the slow modes, close to the origin, to the largest
    eigenvalues of the inverse, which the Lanczos method finds first, so a
    few tens of solves give the ISF at every time. It is extended `every`
    solves at a time, until `converged(previous, current)`.

    If the form of a new vector vanishes, the Lanczos method breaks down,
    and the modes are found again by the Arnoldi method
    (`get_modes_by_arnoldi`), which cannot.

    Raises
    ------
    RuntimeError
        If the modes do not converge within max_solves.
    """
    form = float(start @ (parity * start))
    v = start / np.sqrt(abs(form))
    sign = float(np.sign(form))
    norm = abs(form)
    previous_v = np.zeros_like(v)
    alpha, beta, signs = list[float](), list[float](), [sign]
    previous: Modes | None = None
    for k in range(max_solves):
        w = solve(v)
        a = float(v @ (parity * w)) / signs[k]
        alpha.append(a)
        w = w - a * v
        if k > 0:
            w = w - beta[k - 1] * signs[k] / signs[k - 1] * previous_v
        form = float(w @ (parity * w))
        b = np.sqrt(abs(form))
        if b <= 1e-13 * max(abs(a), float(np.linalg.norm(w)), 1e-300) or b == 0:
            # The Krylov space is invariant: the modes are exact
            return _get_modes(alpha, beta, signs, shift, norm)
        if abs(form) < 1e-12 * float(w @ w):
            # A breakdown of the indefinite form: the Arnoldi method has none
            return get_modes_by_arnoldi(
                solve,
                start,
                parity * start,
                shift,
                converged=converged,
                every=every,
            )
        beta.append(b)
        signs.append(float(np.sign(form)))
        previous_v, v = v, w / b
        if (k + 1) % every == 0:
            current = _get_modes(alpha, beta, signs, shift, norm)
            if previous is not None and converged(previous, current):
                return current
            previous = current
    msg = f"The modes did not converge within {max_solves} solves"
    raise RuntimeError(msg)


def get_modes_by_arnoldi(  # ruff:ignore[too-many-arguments, too-many-locals]
    solve: Callable[[FloatArray], FloatArray],
    start: FloatArray,
    observable: FloatArray,
    shift: float,
    *,
    converged: Callable[[Modes, Modes], bool],
    every: int = 5,
    max_solves: int = 2000,
) -> Modes:
    """Get the modes of C(t) = o^T e^{L t} u from the Arnoldi method on (shift - L)^{-1}.

    The fallback of `get_modes` where its Lanczos method breaks down: the
    Arnoldi basis V is orthonormal, so it cannot, at the cost of holding
    every vector. With (shift - L)^{-1} V = V H, L is shift - H^{-1} in the
    basis, and with H = Y diag(mu) Y^{-1} the mode of rate shift - 1 / mu_j
    has weight |u| (o^T V Y)_j (Y^{-1} e_0)_j.
    """
    norm = float(np.linalg.norm(start))
    basis = np.zeros((min(64, max_solves) + 1, start.size))
    basis[0] = start / norm
    projected = [float(observable @ basis[0])]
    hessenberg = np.zeros((max_solves + 1, max_solves))
    previous: Modes | None = None

    def modes(m: int) -> Modes:
        mu, vectors = scipy.linalg.eig(hessenberg[:m, :m])
        left = scipy.linalg.solve(vectors, np.eye(m)[:, 0])
        mu = np.where(mu == 0, 1e-300, mu)
        return Modes(
            rates=shift - 1 / mu,
            weights=norm * (np.array(projected[:m]) @ vectors) * left,
            n_solves=m,
        )

    for k in range(max_solves):
        if k + 2 > len(basis):
            grown = np.zeros((min(2 * len(basis), max_solves + 1), start.size))
            grown[: len(basis)] = basis
            basis = grown
        w = solve(basis[k])
        built = basis[: k + 1]
        # Gram-Schmidt twice keeps the basis orthogonal to round off
        for _ in range(2):
            overlap = built @ w
            w -= overlap @ built
            hessenberg[: k + 1, k] += overlap
        h = float(np.linalg.norm(w))
        hessenberg[k + 1, k] = h
        if h <= 1e-14 * norm:
            return modes(k + 1)
        basis[k + 1] = w / h
        projected.append(float(observable @ basis[k + 1]))
        if (k + 1) % every == 0:
            current = modes(k + 1)
            if previous is not None and converged(previous, current):
                return current
            previous = current
    msg = f"The modes did not converge within {max_solves} solves"
    raise RuntimeError(msg)
