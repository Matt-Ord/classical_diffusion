# cspell: ignore Gragg orthogonalizing REORTHOGONALIZE orthogonalize vdot Golub Kahan bidiagonalization bidiagonal jitted argnums fori
"""The Krylov methods of the classical solvers: a correlation [u, f(A) u] as a modal sum, the spaces of the time steps, and Ritz pairs.

The Kramers operator in its real form is J-symmetric, L^T = J L J for J the
parity of the momentum, and so is (s - L)^-1: Arnoldi, with the sparse LU
of s - L (`_lu`) and the basis on the device, a block of steps per jitted
call. The generator of the overdamped limit is -L = D_0 A^dag A (J = 1):
Golub-Kahan bidiagonalization of A, which keeps the slow rates to round off
relative to themselves, on the host (milliseconds per result).
"""

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from classical_diffusion._modal import ModalSum

# A runtime import: jax.jit reads the annotations of _extend
from classical_diffusion.kramers._lu import SparseFactor

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion._results import ComplexArray

_AGREEMENTS = 2
"""The successive blocks over which a modal sum must agree: a slow mode of small weight can leave one block unchanged before the method finds it."""
INVARIANT = 1e-13
"""The norm of the next Krylov vector, relative to the projected operator, below which the Krylov space is invariant and the modes exact."""
REORTHOGONALIZE = 0.7
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
            if np.linalg.norm(w) < REORTHOGONALIZE * before:
                again = basis.conj() @ w
                w -= again @ basis
                overlap += again
            self.hessenberg[: k + 1, k] = overlap
            norm = float(np.linalg.norm(w))
            self.hessenberg[k + 1, k] = norm
            self.steps += 1
            if norm <= INVARIANT * max(float(np.abs(overlap).max()), 1e-300):
                self.invariant = True
                return
            self.vectors[k + 1] = w / norm


def orthogonalize(
    project: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    w: jax.Array,
    size: int,
) -> tuple[jax.Array, jax.Array]:
    """Remove from w its components along a basis, `project` returning w less them and the overlaps (size of them), twice where the first pass cancelled most of w.

    The passes are a loop rather than a branch, so `project` is compiled once.
    """

    def again(c: tuple) -> jax.Array:
        w, _, before, passes = c
        return (passes == 0) | (
            (passes == 1) & (jnp.linalg.norm(w) < REORTHOGONALIZE * before)
        )

    def body(c: tuple) -> tuple:
        w, overlap, _, passes = c
        projected, more = project(w)
        return projected, overlap + more, jnp.linalg.norm(w), passes + 1

    w, overlap, _, _ = jax.lax.while_loop(again, body, (w, jnp.zeros(size), 0.0, 0))
    return w, overlap


BLOCK = 4
"""The rows of the basis read at a time by Gram-Schmidt (block modified Gram-Schmidt): only the blocks filled so far are read."""


def remove_block(w: jax.Array, rows: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Remove from w its components along the rows of a block, with their overlaps."""
    h = rows @ w
    return w - h @ rows, h


@functools.partial(jax.jit, donate_argnums=(1, 2))
def _extend(  # ruff:ignore[too-many-arguments]
    factor: SparseFactor,
    vectors: jax.Array,
    records: jax.Array,
    measurement: jax.Array,
    *,
    k: int,
    end: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Take Arnoldi steps of A = factor^{-1} from step k to end, or until the space is invariant, in one call.

    Gram-Schmidt goes a block of rows at a time over the rows filled so far
    (the rows beyond are zero), twice where the first pass cancelled most
    of w. XLA copies each block, a dynamic slice, but a few rows of a 1D
    basis stay in the cache: a step at 7,076 unknowns takes 0.9 ms (0.37 ms
    the solve), where SuperLU and classical Gram-Schmidt in numpy take 1.3
    ms. Step k records the overlap of the new vector with the measurement
    and then column k of H, so the basis never leaves the device.
    """
    capacity = len(vectors)

    def project(vectors: jax.Array, k: jax.Array, w: jax.Array) -> tuple:
        def remove(j: jax.Array, c: tuple) -> tuple:
            w, h = c
            rows = jax.lax.dynamic_slice_in_dim(vectors, j * BLOCK, BLOCK)
            w, more = remove_block(w, rows)
            return w, jax.lax.dynamic_update_slice_in_dim(h, more, j * BLOCK, 0)

        return jax.lax.fori_loop(0, k // BLOCK + 1, remove, (w, jnp.zeros(capacity)))

    def going(c: tuple) -> jax.Array:
        _, _, k, invariant = c
        return (k < end) & ~invariant

    def step(c: tuple) -> tuple:
        vectors, records, k, _ = c
        w, overlap = orthogonalize(
            functools.partial(project, vectors, k), factor.solve(vectors[k]), capacity
        )
        norm = jnp.linalg.norm(w)
        invariant = norm <= INVARIANT * jnp.maximum(jnp.abs(overlap).max(), 1e-300)
        # At an invariant space v is not used (nor its overlap, the last recorded)
        v = w / norm
        record = jnp.concatenate([(v @ measurement)[None], overlap.at[k + 1].set(norm)])
        return vectors.at[k + 1].set(v), records.at[k].set(record), k + 1, invariant

    start = (vectors, records, jnp.asarray(k), jnp.zeros((), bool))
    return jax.lax.while_loop(going, step, start)


@functools.partial(jax.jit, static_argnames="capacity")
def _grow(
    vectors: jax.Array, records: jax.Array, capacity: int
) -> tuple[jax.Array, jax.Array]:
    more = capacity - len(vectors)
    return jnp.pad(vectors, ((0, more), (0, 0))), jnp.pad(
        records, ((0, more), (0, more))
    )


_CAPACITY = 512
"""The rows of the basis first held on the device; it doubles as needed (a new compile)."""


class _DeviceArnoldi:
    """The Arnoldi method of A = factor^{-1} on the device (`_extend`), with H and the overlaps of the basis with the measurement brought to the host."""

    def __init__(
        self, factor: SparseFactor, start: np.ndarray, measurement: np.ndarray
    ) -> None:
        self.factor = factor
        u = np.zeros(factor.size)
        u[: start.size] = start / np.linalg.norm(start)
        padded = np.zeros(factor.size)
        padded[: start.size] = measurement
        self._first = float(u @ padded)
        vectors = np.zeros((_CAPACITY, factor.size))
        vectors[0] = u
        with jax.enable_x64(new_val=True):
            self.measurement = jax.device_put(padded)
            self.vectors = jax.device_put(vectors)
            self.records = jax.device_put(np.zeros((_CAPACITY, _CAPACITY + 1)))
        self._host = np.zeros((1, 1))
        self.steps = 0
        self.invariant = False

    def extend(self, steps: int) -> None:
        end = self.steps + steps
        with jax.enable_x64(new_val=True):
            capacity = len(self.vectors)
            while end >= capacity:
                capacity *= 2
            if capacity > len(self.vectors):
                self.vectors, self.records = _grow(self.vectors, self.records, capacity)
            self.vectors, self.records, k, invariant = _extend(
                self.factor,
                self.vectors,
                self.records,
                self.measurement,
                k=self.steps,
                end=end,
            )
            self.steps = int(k)
            self.invariant = bool(invariant)
            m = self.steps
            self._host = np.array(np.asarray(self.records)[:m, : m + 2])

    def poles(self, _: np.ndarray) -> tuple[ComplexArray, ComplexArray]:
        """Get the eigenvalues mu_j of H and the residues r_j, with <measurement, f(A) v_0> = sum_j r_j f(mu_j) for f a polynomial of degree below m."""
        m = self.steps
        mu, vectors = scipy.linalg.eig(self._host[:m, 1 : m + 1].T)
        seen = np.concatenate([[self._first], self._host[: m - 1, 0]])
        first = np.zeros(m, complex)
        first[0] = 1
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
        if np.linalg.norm(w) < REORTHOGONALIZE * before:
            w -= (stacked.conj() @ w) @ stacked
        return w

    def extend(self, steps: int) -> None:
        """Take steps more, orthogonalizing each new vector in full against its basis."""
        for _ in range(steps):
            k = self.steps
            if k > 0:
                r = self._orthogonalize(self.adjoint(self.images[-1]), self.vectors)
                beta = float(np.linalg.norm(r))
                if beta <= INVARIANT * max(self.diagonal[-1], 1e-300):
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
            if alpha <= INVARIANT * scale:
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

    operator: SparseFactor | tuple[Callable[[np.ndarray], np.ndarray], ...]
    """The factor of A^{-1}, whose Krylov space is built on the device (`_DeviceArnoldi`), or G and G^dag, where A = G^dag G: then the modes are found by Golub-Kahan bidiagonalization of G on the host (`_GolubKahan`)."""
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
    arnoldi: _DeviceArnoldi | _GolubKahan
    if isinstance(problem.operator, SparseFactor):
        arnoldi = _DeviceArnoldi(problem.operator, start, measurement)
    else:
        apply, adjoint = problem.operator
        arnoldi = _GolubKahan(
            apply=apply,
            adjoint=adjoint,
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
