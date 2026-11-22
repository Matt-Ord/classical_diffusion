"""The slowest modes of a measurement of exp(L t) X_0, by Jacobi-Davidson with a partial Schur form, in the eigenvectors of K."""

# cspell: ignore maxiter bicgstab vdot eigenpair finfo
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from classical_diffusion.caldeira_leggett._linear import (
    INNER_RTOL,
    MAXITER,
    PASSES,
    EigenLiouvillian,
    apply_eigen,
    bicgstab,
    check,
    gmres,
    refine,
)

if TYPE_CHECKING:
    from classical_diffusion.caldeira_leggett._config import EvolutionConfig
    from classical_diffusion.caldeira_leggett._model import ComplexArray


@dataclass(frozen=True, kw_only=True)
class Mode:
    """An eigenvalue lambda of L and its weight in a measurement of exp(L t) X_0."""

    rate: complex
    """lambda."""
    amplitude: complex
    """a = measure(r) c for the eigenvector r and its weight c in X_0, the weight of exp(lambda t) in the measurement."""


@dataclass(frozen=True, kw_only=True)
class ModeCount:
    """Exactly n_modes modes."""

    n_modes: int


@dataclass(frozen=True, kw_only=True)
class RateCut:
    """Every mode with Re lambda > -rate, and at least one."""

    rate: float


type Selection = ModeCount | RateCut

_AMPLITUDE_TOL = 1e-8
"""The smallest amplitude of a mode, relative to the measurement at t = 0, below which it is not looked for: modes the measurement cannot see, such as the steady state at dK = 0."""
_INNER_RTOL = 0.1
"""The residual reduction of each solve of the correction equation."""
_INNER_MAXITER = 25
"""The most GMRES iterations, each one application of L, of each solve of the correction equation.

GMRES rather than BiCGSTAB (two applications per iteration, at most 25
iterations): over the cases of _RESTART, 7.5% fewer applications of L in
all, and Jacobi-Davidson needs 1,106 rather than 1,605 applications to find
the 4 modes of the ISF of hydrogen on Cu(111) in 2D at 0.3 |G|. At most 35
or 50 GMRES iterations need 1,137 and 1,205, and a reduction of 0.01
rather than 0.1 needs 1,312.
"""
_MAX_OUTER = 60
"""The largest number of iterations to find one mode."""
_MAX_MODES = 50
_ROUGH = 0.1
"""The relative residual of a Ritz pair below which a selection is decided on it, as its value and weight are then close."""
_RESTART = 20
"""The most Ritz vectors kept when the search space restarts after a mode is found.

The restart needs no application of L (`_Search.lock`), and keeping 20
rather than 5 keeps more of the partners of a conjugate or degenerate
group: over H, Li and Na on Cu(111) in 1D and H in 2D, 2.7% fewer
applications of L.
"""
_INDEPENDENT = 1e-8
"""The norm of a restart vector, once orthogonalized, below which it is dropped."""
_STAGNATION = 1e-14
"""The norm of a new search vector, once orthogonalized, below which the search has stagnated."""


def _dot(a: jax.Array, b: jax.Array) -> jax.Array:
    """<a_i, b_j> for stacks of states a (k, ...) and b (m, ...)."""
    axes = tuple(range(1, a.ndim))
    return jnp.tensordot(a.conj(), b, axes=(axes, axes))


_PAD = 8


def _padded(q: jax.Array) -> jax.Array:
    """Pad a stack of states with zeros to a multiple of _PAD, at least one, so that few sizes are compiled."""
    size = _PAD * max(-(-len(q) // _PAD), 1)
    return jnp.concatenate([q, jnp.zeros((size - len(q), *q.shape[1:]), q.dtype)])


def _project_out(q: jax.Array, x: jax.Array) -> jax.Array:
    """(1 - Q Q^dagger) x for an orthonormal stack Q."""
    return x - jnp.tensordot(_dot(q, x[None])[:, 0], q, axes=1)


@partial(jax.jit, static_argnames=("maxiter", "method"))
def _projected_solve(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    locked: jax.Array,
    rhs: jax.Array,
    theta: jax.Array,
    rtol: jax.Array,
    *,
    maxiter: int,
    method: Literal["gmres", "bicgstab"],
) -> tuple[jax.Array, jax.Array]:
    """Solve P (L - theta) P t = rhs for t orthogonal to the orthonormal stack Q, P = 1 - Q Q^dagger.

    GMRES or BiCGSTAB, preconditioned on the right by (N - theta)^-1
    projected so that it keeps t orthogonal to Q,
    M_p z = M z - M Q (Q^dagger M Q)^-1 Q^dagger M z, with M exact for the
    no-jump part N. Also get the number of applications of L.
    """
    rates = eigen.rates - theta
    m_q = locked / rates
    # Rows of zeros pad Q, so that few sizes are compiled: they are made
    # inert by a one on the diagonal of Q^dagger M Q
    padding = jnp.all(locked == 0, axis=tuple(range(1, locked.ndim)))
    small = _dot(locked, m_q) + jnp.diag(padding.astype(m_q.dtype))

    def precondition(z: jax.Array) -> jax.Array:
        m_z = z / rates
        weights = jnp.linalg.solve(small, _dot(locked, m_z[None])[:, 0])
        return m_z - jnp.tensordot(weights, m_q, axes=1)

    def operator(z: jax.Array) -> jax.Array:
        t = precondition(z)
        out = apply_eigen(eigen, t, theta)
        return _project_out(locked, out)

    if method == "gmres":
        z, iterations = gmres(operator, rhs, rtol, maxiter)
        return precondition(z), iterations
    z, iterations = bicgstab(operator, rhs, rtol, maxiter)
    return precondition(z), 2 * iterations


def _get_amplitude(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    rate: complex,
    vector: jax.Array,
    state: jax.Array,
    *,
    measurement: jax.Array,
    rtol: float,
) -> tuple[complex, complex, int]:
    """Get the amplitude a and coefficient c of a mode in a state, from a bordered solve, and the applications of L.

    The system (L - lambda) y + c r = X, r^dagger y = 0 has
    c = <l|X> / <l|r>, for l the left eigenvector, without needing l: with
    P = 1 - r r^dagger, P (L - lambda) P y = P X is well conditioned, as
    lambda is projected out, and then c = r^dagger (X - (L - lambda) y). It
    is solved to rtol by passes of BiCGSTAB (`refine`).
    """
    amplitude, coefficient, applications, reached = jax.device_get(
        _amplitude(eigen, jnp.asarray(rate), vector, state, measurement, rtol)
    )
    check(reached, rtol)
    return complex(amplitude), complex(coefficient), int(applications)


@jax.jit
def _amplitude(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    eigen: EigenLiouvillian,
    shift: jax.Array,
    vector: jax.Array,
    state: jax.Array,
    measurement: jax.Array,
    rtol: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Get the amplitude, coefficient, applications of L and residual reached of `_get_amplitude`, on the device."""

    def apply(y: jax.Array) -> jax.Array:
        out = apply_eigen(eigen, y, shift)
        return out - jnp.vdot(vector, out) * vector

    rhs = state - jnp.vdot(vector, state) * vector

    def solved(_: None) -> tuple[jax.Array, jax.Array, jax.Array]:
        y, applications, reached = refine(
            apply,
            lambda residual, inner_rtol: _projected_solve(
                eigen,
                vector[None],
                residual,
                shift,
                inner_rtol,
                maxiter=MAXITER,
                method="bicgstab",
            ),
            rhs,
            rtol=jnp.asarray(rtol),
            inner_rtol=INNER_RTOL,
            passes=PASSES,
        )
        coefficient = jnp.vdot(vector, state - apply_eigen(eigen, y, shift))
        return coefficient, applications + 1, reached

    def direct(_: None) -> tuple[jax.Array, jax.Array, jax.Array]:
        return jnp.vdot(vector, state), jnp.asarray(0), jnp.asarray(0.0)

    # Less than rtol of the state off the mode, as for the steady state at
    # dK = 0, means y = 0 solves it to rtol: refining that rhs, of round
    # off, to rtol of itself took 2,002 applications of L for hydrogen on
    # Cu(111) in 1D
    coefficient, applications, reached = jax.lax.cond(
        jnp.linalg.norm(rhs) <= rtol * jnp.linalg.norm(state), direct, solved, None
    )
    return (
        jnp.vdot(measurement, vector) * coefficient,
        coefficient,
        applications,
        reached,
    )


@dataclass(frozen=True, kw_only=True)
class _Ritz:
    """The Ritz pairs of the search space the measurement sees, slowest first."""

    values: ComplexArray
    right: ComplexArray
    """The Ritz vectors in the search space, (n, n), columns of unit norm."""
    weights: ComplexArray
    """c of each pair, relative to |X_0|."""
    order: list[int]


@partial(jax.jit, donate_argnames=("basis", "images"))
def _expand(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    eigen: EigenLiouvillian,
    locked: jax.Array,
    basis: jax.Array,
    images: jax.Array,
    size: jax.Array,
    x: jax.Array,
    vectors: tuple[jax.Array, jax.Array],
    threshold: jax.Array,
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]:
    """Add x, orthogonalized twice against the locked vectors and the basis, at row size of the buffers, if more than threshold of it is new (`_Search.expand`).

    Get the buffers and the norm of the new part of x, the new column and row
    of V^dagger W, its diagonal, <x, X_0> and <measurement, x>, all small.
    Rows of zeros pad the buffers and the locked vectors.
    """
    start, measurement = vectors
    for _ in range(2):
        x -= jnp.tensordot(_dot(locked, x[None])[:, 0], locked, axes=1)
        x -= jnp.tensordot(_dot(basis, x[None])[:, 0], basis, axes=1)
    norm = jnp.linalg.norm(x)
    accepted = norm > threshold
    x /= jnp.where(accepted, norm, 1)
    image = _project_out(locked, apply_eigen(eigen, x, jnp.asarray(0j)))
    small = (
        norm,
        _dot(basis, image[None])[:, 0],
        _dot(x[None], images)[0],
        jnp.vdot(x, image),
        jnp.vdot(x, start),
        jnp.vdot(measurement, x),
    )
    return (
        basis.at[size].set(jnp.where(accepted, x, 0)),
        images.at[size].set(jnp.where(accepted, image, 0)),
        small,
    )


@jax.jit
def _pair(
    basis: jax.Array, images: jax.Array, s: jax.Array, theta: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Get the Ritz vector u = V s, its residual r = W s - theta u, and |r|."""
    u = jnp.tensordot(s, basis, axes=1)
    r = jnp.tensordot(s, images, axes=1) - theta * u
    return u, r, jnp.linalg.norm(r)


@dataclass(kw_only=True)
class _Search:
    """The search of Jacobi-Davidson: the locked Schur vectors Q with T = Q^dagger L Q, and the search space V, with W = (1 - Q Q^dagger) L V.

    V and W are held in buffers on the device whose rows beyond `size` are
    zero, grown _PAD rows at a time, so that each iteration compiles nothing
    new, copies no state, and reads back only small arrays (`_expand`).
    """

    eigen: EigenLiouvillian
    start: jax.Array
    measurement: jax.Array
    locked: list[jax.Array] = field(default_factory=list)
    schur: ComplexArray = field(default_factory=lambda: np.zeros((0, 0), complex))
    basis: jax.Array = field(init=False)
    """V, orthonormal, in its first `size` rows."""
    images: jax.Array = field(init=False)
    """W."""
    size: int = 0
    projected: ComplexArray = field(default_factory=lambda: np.zeros((0, 0), complex))
    """V^dagger W, kept as V grows, rather than k^2 products of states each iteration."""
    overlaps: ComplexArray = field(default_factory=lambda: np.zeros(0, complex))
    """V^dagger X_0."""
    measured: ComplexArray = field(default_factory=lambda: np.zeros(0, complex))
    """<measurement, V>."""
    applications: int = 0
    """The applications of L to find the modes."""
    inner: list[jax.Array] = field(default_factory=list)
    """The applications of L of the correction solves, on the device until the end."""
    coefficient_applications: int = 0
    """The applications of L to find their coefficients."""
    reference: float = field(init=False)
    """|<measurement, X_0>|."""
    norm: float = field(init=False)
    """|X_0|."""

    def __post_init__(self) -> None:
        self.basis, self.images = (_padded(self.start[None][:0]) for _ in range(2))
        self.reference, self.norm = (
            float(x)
            for x in jax.device_get(
                (
                    jnp.abs(jnp.vdot(self.measurement, self.start)),
                    jnp.linalg.norm(self.start),
                )
            )
        )

    def _set(self, basis: jax.Array, images: jax.Array) -> None:
        """Set V and W to stacks of states, padded into buffers."""
        self.size = len(basis)
        self.basis, self.images = _padded(basis), _padded(images)

    def expand(self, x: jax.Array, *, scale: float = 1.0) -> bool:
        """Add x, orthogonalized, to the search space, unless less than _STAGNATION scale of it is new.

        A correction already in the space means the search has stagnated. A
        restart vector is measured against its own norm.
        """
        if self.size == len(self.basis):
            self.basis, self.images = (
                _padded(jnp.concatenate([x, jnp.zeros_like(x[:1])]))
                for x in (self.basis, self.images)
            )
        self.basis, self.images, small = _expand(
            self.eigen,
            _padded(jnp.stack(self.locked) if self.locked else self.start[None][:0]),
            self.basis,
            self.images,
            jnp.asarray(self.size),
            x,
            (self.start, self.measurement),
            jnp.asarray(_STAGNATION * scale),
        )
        norm, column, row, diagonal, overlap, measured = jax.device_get(small)
        if norm <= _STAGNATION * scale:
            return False
        self.applications += 1
        k = self.size
        grown = np.zeros((k + 1, k + 1), complex)
        grown[:k, :k] = self.projected
        grown[:k, k] = column[:k]
        grown[k, :k] = row[:k]
        grown[k, k] = diagonal
        self.projected = grown
        self.overlaps = np.append(self.overlaps, overlap)
        self.measured = np.append(self.measured, measured)
        self.size += 1
        return True

    def correct(self, u: jax.Array, r: jax.Array, theta: complex) -> bool:
        """Expand the search space by an approximate solution of the correction equation, or get False if it has stagnated."""
        correction, inner = _projected_solve(
            self.eigen,
            _padded(jnp.stack([*self.locked, u])),
            -r,
            jnp.asarray(theta),
            jnp.asarray(_INNER_RTOL),
            maxiter=_INNER_MAXITER,
            method="gmres",
        )
        self.inner.append(inner)
        return self.expand(correction)

    @property
    def total(self) -> int:
        """The applications of L to find the modes, with the correction solves."""
        return self.applications + int(sum(int(n) for n in jax.device_get(self.inner)))

    def ritz(self) -> _Ritz:
        values, left, right = scipy.linalg.eig(self.projected, left=True, right=True)
        right /= np.linalg.norm(right, axis=0)
        # c = (z^dagger V^dagger X_0) / (z^dagger s) for the left and right Ritz vectors z and s
        weights = (left.conj().T @ self.overlaps) / (
            np.einsum("ij,ij->j", left.conj(), right)
        )
        estimates = (self.measured @ right) * weights
        order = [int(j) for j in np.argsort(-values.real)]
        seen = [
            j for j in order if abs(estimates[j]) >= _AMPLITUDE_TOL * self.reference
        ]
        return _Ritz(
            values=values,
            right=right,
            weights=weights / self.norm,
            order=seen or order,
        )

    def pair(self, ritz: _Ritz, j: int) -> tuple[jax.Array, jax.Array, float]:
        """Get the Ritz vector u, its residual r = (1 - Q Q^dagger) L u - theta u, and |r|."""
        s = np.zeros(len(self.basis), complex)
        s[: self.size] = ritz.right[:, j]
        u, r, norm = _pair(
            self.basis, self.images, jnp.asarray(s), jnp.asarray(ritz.values[j])
        )
        return u, r, float(norm)

    def extend(self, u: jax.Array) -> None:
        """Extend the Schur form with u, of unit norm and orthogonal to the locked vectors."""
        image = apply_eigen(self.eigen, u, jnp.asarray(0j))
        self.applications += 1
        column = np.asarray(_dot(jnp.stack([*self.locked, u]), image[None])[:, 0])
        n = len(self.locked)
        grown = np.zeros((n + 1, n + 1), complex)
        grown[:n, :n] = self.schur
        grown[:, n] = column
        self.schur = grown
        self.locked.append(u)

    def lock(self, u: jax.Array, ritz: _Ritz) -> None:
        """Extend the Schur form with u, the Ritz vector V s of ritz.order[0], and restart the search space from the next Ritz vectors.

        The restart vectors V y, for y the next Ritz vectors made orthonormal
        to s and each other, are orthogonal to u, and their images
        (1 - u u^dagger) W y follow from those of V, with no application of L.
        """
        self.extend(u)
        s_0 = ritz.right[:, ritz.order[0]]
        rest = ritz.right[:, ritz.order[1 : 1 + _RESTART]]
        q, r = np.linalg.qr(rest - np.outer(s_0, s_0.conj() @ rest))
        q = jnp.asarray(q[:, np.abs(np.diag(r)) > _INDEPENDENT])
        basis = jnp.tensordot(q.T, self.basis[: self.size], axes=1)
        images = jnp.tensordot(q.T, self.images[: self.size], axes=1)
        images -= jnp.tensordot(_dot(u[None], images)[0], u[None], axes=0)[:, 0]
        self._set(basis, images)
        q = np.asarray(q)
        # The images lose only their part along u, to which V q is orthogonal
        self.projected = q.conj().T @ self.projected @ q
        self.overlaps = q.conj().T @ self.overlaps
        self.measured = np.matmul(self.measured, q)
        if not self.size:
            # No other Ritz vector is seen, as at dK = 0, where X_0 is the
            # steady state: search on in what is left of X_0, if anything
            self.expand(self.start, scale=self.norm)

    def modes(self, rtol: float) -> list[Mode]:
        """Get the modes from the Schur form, slowest first, with their amplitudes from coefficients to rtol of |X_0|.

        Each eigenvector is only accurate to its residual, |r| ~ mode_rtol |lambda|,
        which leaks a little of it onto the slower modes. The coefficients are
        taken on what remains of X_0 as each mode is removed, fastest first, so
        that the slowest are removed exactly. A failed coefficient solve
        raises a RuntimeError (`refine`).
        """
        if not self.locked:
            return []
        rates, vectors = np.linalg.eig(self.schur)
        q = jnp.stack(self.locked)
        modes, remainder = [], self.start
        for j in np.argsort(rates.real):
            vector = jnp.tensordot(jnp.asarray(vectors[:, j]), q, axes=1)
            vector /= jnp.linalg.norm(vector)
            amplitude, coefficient, applications = _get_amplitude(
                self.eigen,
                complex(rates[j]),
                vector,
                remainder,
                measurement=self.measurement,
                rtol=rtol,
            )
            self.coefficient_applications += applications
            remainder -= coefficient * vector
            modes.append(Mode(rate=complex(rates[j]), amplitude=amplitude))
        return modes[::-1]


def find_slow_modes(
    eigen: EigenLiouvillian,
    start: jax.Array,
    measurement: jax.Array,
    selection: Selection,
    *,
    config: EvolutionConfig,
) -> tuple[list[Mode], dict[str, int]]:
    """Get the slowest modes of <measurement, exp(L t) start>, slowest first, by Jacobi-Davidson (`_search`), and the applications of L to find them and their coefficients.

    The coefficients are one bordered solve per mode (`_get_amplitude`),
    counted apart. It needs no time steps, and no solve near a slow
    eigenvalue, which is badly conditioned: the slowest mode of the ISF of lithium on Cu(110) in
    2D costs 380-780 applications of L, where a shift-invert Arnoldi method
    needs 5,000-12,000.
    """
    search = _search(eigen, start, measurement, selection, config=config)
    # The coefficients to well below the error allowed in an evolution
    return search.modes(0.1 * config.stop), {
        "modes": search.total,
        "coefficients": search.coefficient_applications,
    }


def _search(
    eigen: EigenLiouvillian,
    start: jax.Array,
    measurement: jax.Array,
    selection: Selection,
    *,
    config: EvolutionConfig,
) -> _Search:
    """Find the slowest modes of <measurement, exp(L t) start> as a partial Schur form, by Jacobi-Davidson.

    Jacobi-Davidson targets the Ritz value of largest real part in a search
    space V, and expands V by an approximate solution of the correction
    equation (1 - u u^dagger)(L - theta)(1 - u u^dagger) t = -r: a few
    GMRES iterations, preconditioned by the exact inverse of the no-jump
    part (`_projected_solve`). The mode itself is projected out, so the
    equation stays well conditioned as theta converges. A mode is locked once
    |r| <= config.mode_rtol |theta|, or round off, into a partial Schur form,
    and the search continues outside it. Modes the measurement does not see
    are skipped, and the selection decides when to stop.

    Raises
    ------
    RuntimeError
        If a mode does not converge in _MAX_OUTER iterations, or its search
        stagnates.
    """
    search = _Search(eigen=eigen, start=start, measurement=measurement)
    floor = 1e3 * np.finfo(float).eps * float(jnp.abs(eigen.rates).max())
    target = selection.n_modes if isinstance(selection, ModeCount) else _MAX_MODES
    search.expand(start)
    iterations = 0
    # The search space is empty once X_0 is in the span of the modes locked
    while len(search.locked) < target and search.size:
        ritz = search.ritz()
        j = ritz.order[0]
        theta = complex(ritz.values[j])
        u, r, norm = search.pair(ritz, j)
        if (
            isinstance(selection, RateCut)
            and search.locked
            and norm <= _ROUGH * abs(theta)
            and theta.real < -selection.rate
        ):
            break
        if norm <= max(config.mode_rtol * abs(theta), floor):
            search.lock(u, ritz)
            iterations = 0
            continue
        iterations += 1
        if iterations > _MAX_OUTER or not search.correct(u, r, theta):
            msg = f"Jacobi-Davidson did not converge: |r| {norm:.1e} at {theta:.4e}"
            raise RuntimeError(msg)
    return search
