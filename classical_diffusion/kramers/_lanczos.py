"""A correlation [u, f(A) u] as a sum of modes, by the Lanczos method in the form [x, y] = x^H J y, for an operator self-adjoint in it.

The Kramers operator in its real form is J-symmetric, L^T = J L J for J the
parity of the momentum, and so is (s - L)^-1; the generator of the
overdamped limit is Hermitian, J = 1. One recurrence serves both.
"""

# cspell: ignore reorthogonalized reorthogonalizing vdot
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from classical_diffusion._modal import ModalSum, get_poles

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion._results import ComplexArray

_AGREEMENTS = 2
"""The successive blocks over which a modal sum must agree: a slow mode of small weight can leave one block unchanged before the Lanczos method finds it."""
_INVARIANT = 1e-13
"""The norm of the next Lanczos vector, relative to the tridiagonal, below which the Krylov space is invariant and the modes exact."""
_BREAKDOWN = 1e-12
"""The form |[w, w]| / |w|^2 of the next vector below which the recurrence breaks down (J indefinite only)."""
_SPLIT_SEED = 2026
"""The seed of the vector that splits a start on a breakdown."""


class _BreakdownError(Exception):
    """A breakdown of the recurrence: [w, w] = 0 for nonzero w, possible only where J is indefinite."""


@dataclass(kw_only=True)
class _Recurrence:
    """The Lanczos basis V, with V^H J V = diag(signs), and A V = V T + w e_m^T, T tridiagonal with alpha on the diagonal, beta below and beta signs_{k+1} / signs_k above."""

    apply: Callable[[np.ndarray], np.ndarray]
    parity: np.ndarray | None
    vectors: np.ndarray
    """The basis as rows, with spare rows beyond the first len(signs)."""
    scale: float
    """|[u, u]|."""
    signs: list[float]
    alpha: list[float] = field(default_factory=list)
    beta: list[float] = field(default_factory=list)
    invariant: bool = False

    def form(self, x: np.ndarray) -> np.ndarray:
        """Get J x."""
        return x if self.parity is None else self.parity * x

    def extend(self, steps: int) -> None:
        """Take steps more, reorthogonalizing in full in the form, so that no mode is found twice.

        Raises
        ------
        _BreakdownError
            If [w, w] vanishes for a nonzero w.
        """
        for _ in range(steps):
            k = len(self.signs)
            if k == len(self.vectors):
                grown = np.zeros((2 * k, self.vectors.shape[1]), self.vectors.dtype)
                grown[:k] = self.vectors
                self.vectors = grown
            basis, signs = self.vectors[:k], np.array(self.signs)
            v = basis[-1]
            w = self.apply(v)
            a = float(np.vdot(v, self.form(w)).real) / signs[-1]
            self.alpha.append(a)
            before = float(np.linalg.norm(w))
            # Gram-Schmidt in the form, a second time only where the first
            # cancelled most of w (Daniel, Gragg, Kaufman and Stewart 1976),
            # keeps V^H J V diagonal to round off
            w -= ((basis.conj() @ self.form(w)) / signs) @ basis
            norm = float(np.linalg.norm(w))
            if norm < 0.7 * before:
                w -= ((basis.conj() @ self.form(w)) / signs) @ basis
                norm = float(np.linalg.norm(w))
            size = max([abs(a), *(abs(b) for b in self.beta[-1:])])
            if norm <= _INVARIANT * size:
                self.invariant = True
                return
            form = float(np.vdot(w, self.form(w)).real)
            if abs(form) < _BREAKDOWN * norm**2:
                raise _BreakdownError
            b = np.sqrt(abs(form))
            self.beta.append(b)
            self.signs.append(float(np.sign(form)))
            self.vectors[k] = w / b

    def poles(self) -> tuple[ComplexArray, ComplexArray]:
        """Get the eigenvalues mu_j of T and the residues r_j, with [u, f(A) u] = sum_j r_j f(mu_j) for f a polynomial of degree below 2m."""
        m = len(self.alpha)
        signs, beta = np.array(self.signs[:m]), np.array(self.beta[: m - 1])
        upper = beta * signs[1:] / signs[:-1]
        return get_poles(np.array(self.alpha), upper, beta, self.scale * signs[0])


@dataclass(frozen=True, kw_only=True)
class LanczosProblem:
    """A correlation C = [u, f(A) u], as a modal sum of the rates rates_of(mu) of the eigenvalues mu of A in its Krylov space."""

    apply: Callable[[np.ndarray], np.ndarray]
    parity: np.ndarray | None
    """J, or None for J = 1 (A Hermitian)."""
    rates_of: Callable[[ComplexArray], ComplexArray]
    """The rate of each eigenvalue mu of A: -mu for A = -L, shift - 1 / mu for A = (shift - L)^-1."""
    span: float
    """The longest time of the correlation, over which a pole may grow by the tolerance (`ModalSum.of`)."""
    error: Callable[[ModalSum, ModalSum], float]
    """The change between two modal sums, relative to the correlation (at t = 0, or of what is found from it)."""
    tolerance: float
    """The relative change, and the weight of the poles dropped relative to C(0), below which the modal sum has converged."""
    block: int = 5
    max_steps: int = 5000


def _run(problem: LanczosProblem, start: np.ndarray, tolerance: float) -> ModalSum:
    """Extend the recurrence a block at a time, until _AGREEMENTS blocks in a row change the modal sum by less than the tolerance.

    Raises
    ------
    _BreakdownError
        On a breakdown of the recurrence.
    RuntimeError
        If it does not converge within max_steps.
    """
    form = float(
        np.vdot(start, start if problem.parity is None else problem.parity * start).real
    )
    size = float(np.vdot(start, start).real)
    if size == 0:
        empty = np.zeros(0, complex)
        return ModalSum(rates=empty, residues=empty, stable=np.zeros(0, bool))
    if abs(form) < _BREAKDOWN * size:
        raise _BreakdownError
    vectors = np.zeros((min(16, start.size + 1), start.size), start.dtype)
    vectors[0] = start / np.sqrt(abs(form))
    recurrence = _Recurrence(
        apply=problem.apply,
        parity=problem.parity,
        vectors=vectors,
        scale=abs(form),
        signs=[float(np.sign(form))],
    )
    previous: ModalSum | None = None
    agreed = 0
    while len(recurrence.alpha) < problem.max_steps:
        # Blocks grow with the basis, so the O(m^3) poles are found a few times
        steps = max(problem.block, len(recurrence.alpha) // 16)
        recurrence.extend(min(steps, start.size - len(recurrence.alpha)))
        mu, residues = recurrence.poles()
        # A zero eigenvalue of (shift - L)^-1 is a mode of infinite rate, with no weight
        current = ModalSum.of(
            problem.rates_of(np.where(mu == 0, 1e-300, mu)),
            residues,
            problem.span,
            tolerance * recurrence.scale,
        )
        if recurrence.invariant or len(recurrence.alpha) >= start.size:
            return current
        agreed = (
            agreed + 1
            if previous is not None
            and max(
                problem.error(previous, current),
                current.dropped / recurrence.scale,
            )
            < tolerance
            else 0
        )
        if agreed >= _AGREEMENTS:
            return current
        previous = current
    msg = f"The Lanczos method did not converge within {problem.max_steps} steps"
    raise RuntimeError(msg)


def get_modal_sum(problem: LanczosProblem, start: np.ndarray) -> ModalSum:
    """Get C = [u, f(A) u] as a modal sum, by the Lanczos method in the form [x, y] = x^H J y.

    The basis is reorthogonalized in full in the form, so no mode is found
    twice; it is m vectors of memory. In an indefinite form the recurrence
    can break down, [w, w] = 0 for nonzero w. The breakdown belongs to the
    start u, so then C is found from three other starts by polarization,
    [u, f u] = ([u + h, f (u + h)] + [u - h, f (u - h)]) / 2 - [h, f h] for
    a fixed random h, each by the same recurrence.

    Raises
    ------
    RuntimeError
        If a split start also breaks down, or the recurrence does not
        converge within max_steps.
    """
    try:
        return _run(problem, start, problem.tolerance)
    except _BreakdownError:
        pass
    rng = np.random.default_rng(_SPLIT_SEED)
    split = rng.standard_normal(start.shape).astype(start.dtype)
    split *= np.linalg.norm(start) / np.linalg.norm(split)
    try:
        sums = [
            _run(problem, start + split, problem.tolerance / 4),
            _run(problem, start - split, problem.tolerance / 4),
            _run(problem, split, problem.tolerance / 4),
        ]
    except _BreakdownError as error:
        msg = "The Lanczos recurrence broke down, also with a split start"
        raise RuntimeError(msg) from error
    weights = (0.5, 0.5, -1.0)
    return ModalSum(
        rates=np.concatenate([s.rates for s in sums]),
        residues=np.concatenate(
            [w * s.residues for s, w in zip(sums, weights, strict=True)]
        ),
        stable=np.concatenate([s.stable for s in sums]),
    )
