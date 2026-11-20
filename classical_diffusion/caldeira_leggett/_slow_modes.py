"""The slowest modes of a measurement of exp(L t) X_0, by Jacobi-Davidson with a partial Schur form, in the eigenvectors of K."""

# cspell: ignore maxiter bicgstab vdot eigenpair finfo
import math
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
    gmres,
    refine,
)

if TYPE_CHECKING:
    from classical_diffusion.caldeira_leggett._config import EvolutionConfig
    from classical_diffusion.caldeira_leggett._model import ComplexArray


@dataclass(frozen=True, kw_only=True)
class Mode:
    """An eigenpair L r = lambda r, |r| = 1, and its weight in a measurement of exp(L t) X_0."""

    rate: complex
    """lambda."""
    vector: jax.Array
    """r, in the eigenvectors of K."""
    coefficient: complex
    """c, the weight of r in X_0 (or in what remains of it once the faster modes are removed)."""
    amplitude: complex
    """a = measure(r) c, the weight of exp(lambda t) in the measurement."""


@dataclass(frozen=True, kw_only=True)
class ModeCount:
    """Exactly n_modes modes."""

    n_modes: int


@dataclass(frozen=True, kw_only=True)
class RateCut:
    """Every mode with Re lambda > -rate, and at least one."""

    rate: float


@dataclass(frozen=True, kw_only=True)
class CostCut:
    """Every mode which costs fewer applications of L to find than to step through.

    The modes are taken in groups of nearly equal Re lambda, conjugate or
    degenerate partners, which only shorten the stepping together. The
    stepping stops when the slowest mode left, of weight c, has decayed to
    stop, at ln(|c| / stop) / |Re lambda|, and not before start_time, which is
    stepped already. A group is found if its estimated cost, cost_factor
    times its size times the applications per mode so far (none for the
    first), is below step_rate times the time it saves. A group which costs
    more than that is dropped, so a group is never worse than twice its
    saving.
    """

    step_rate: float
    """The applications of L per unit time of stepping."""
    start_time: float
    stop: float
    cost_factor: float
    """The margin on the estimated cost of a mode, which varies from mode to mode (`EvolutionConfig.cost_factor`)."""


type Selection = ModeCount | RateCut | CostCut

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
_PARTNER = 0.05
"""The relative difference of Re lambda within which modes are a group."""
_RESTART = 20
"""The most Ritz vectors kept when the search space restarts after a mode is found.

The restart needs no application of L (`_Search.lock`), and keeping 20
rather than 5 keeps more of the partners of a group: over H, Li and Na on
Cu(111) in 1D and H in 2D (ISF at 0.3 and 0.05 |G| and MSD, 2 seeds), 2.7%
fewer applications of L.
"""
_WEIGHT = 0.1
"""The weight of a mode in an evolution, relative to |X_0|, below which it is found to a looser residual (`_get_mode_rtol`)."""
_LOOSEST = 1e-3
"""The loosest relative residual of a mode."""
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
    shift = jnp.asarray(rate)

    def apply(y: jax.Array) -> jax.Array:
        out = apply_eigen(eigen, y, shift)
        return out - jnp.vdot(vector, out) * vector

    rhs = state - jnp.vdot(vector, state) * vector
    if float(jnp.linalg.norm(rhs)) <= rtol * float(jnp.linalg.norm(state)):
        # Less than rtol of the state is off the mode, as for the steady
        # state at dK = 0, so y = 0 solves it to rtol: refining that rhs, of
        # round off, to rtol of itself took 2,002 applications of L for
        # hydrogen on Cu(111) in 1D
        coefficient, applications = complex(jnp.vdot(vector, state)), 0
    else:
        y, applications = refine(
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
            rtol=rtol,
            inner_rtol=INNER_RTOL,
            passes=PASSES,
        )
        coefficient = complex(jnp.vdot(vector, state - apply_eigen(eigen, y, shift)))
        applications += 1
    amplitude = complex(jnp.vdot(measurement, vector)) * coefficient
    return amplitude, coefficient, applications


@dataclass(frozen=True, kw_only=True)
class _Ritz:
    """The Ritz pairs of the search space the measurement sees, slowest first."""

    values: ComplexArray
    right: ComplexArray
    """The Ritz vectors in the search space, (n, n), columns of unit norm."""
    weights: ComplexArray
    """c of each pair, relative to |X_0|."""
    order: list[int]


@dataclass(kw_only=True)
class _Search:
    """The search of Jacobi-Davidson: the locked Schur vectors Q with T = Q^dagger L Q, and the search space V, with W = (1 - Q Q^dagger) L V."""

    eigen: EigenLiouvillian
    start: jax.Array
    measurement: jax.Array
    blind: bool = False
    """Whether to look for every mode, rather than those the measurement sees."""
    locked: list[jax.Array] = field(default_factory=list)
    schur: ComplexArray = field(default_factory=lambda: np.zeros((0, 0), complex))
    basis: jax.Array = field(init=False)
    """V, (k, ...), orthonormal."""
    images: jax.Array = field(init=False)
    """W."""
    projected: ComplexArray = field(default_factory=lambda: np.zeros((0, 0), complex))
    """V^dagger W, kept as V grows, rather than k^2 products of states each iteration."""
    overlaps: ComplexArray = field(default_factory=lambda: np.zeros(0, complex))
    """V^dagger X_0."""
    measured: ComplexArray = field(default_factory=lambda: np.zeros(0, complex))
    """<measurement, V>."""
    applications: int = 0
    """The applications of L to find the modes."""
    coefficient_applications: int = 0
    """The applications of L to find their coefficients."""

    def __post_init__(self) -> None:
        self.basis = self.images = jnp.zeros((0, *self.start.shape), self.start.dtype)

    def expand(self, x: jax.Array, *, scale: float = 1.0) -> bool:
        """Add x, orthogonalized, to the search space, unless less than _STAGNATION scale of it is new.

        A correction already in the space means the search has stagnated. A
        restart vector is measured against its own norm.
        """
        known = (
            jnp.concatenate([jnp.stack(self.locked), self.basis])
            if self.locked
            else self.basis
        )
        if len(known):
            for _ in range(2):
                x = _project_out(known, x)
        size = float(jnp.linalg.norm(x))
        if size <= _STAGNATION * scale:
            return False
        x /= size
        image = apply_eigen(self.eigen, x, jnp.asarray(0j))
        self.applications += 1
        if self.locked:
            image = _project_out(jnp.stack(self.locked), image)
        k = len(self.basis)
        grown = np.zeros((k + 1, k + 1), complex)
        grown[:k, :k] = self.projected
        if k:
            grown[:k, k] = np.asarray(_dot(self.basis, image[None])[:, 0])
            grown[k, :k] = np.asarray(_dot(x[None], self.images)[0])
        grown[k, k] = complex(jnp.vdot(x, image))
        self.projected = grown
        self.overlaps = np.append(self.overlaps, complex(jnp.vdot(x, self.start)))
        self.measured = np.append(self.measured, complex(jnp.vdot(self.measurement, x)))
        self.basis = jnp.concatenate([self.basis, x[None]])
        self.images = jnp.concatenate([self.images, image[None]])
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
        self.applications += int(inner)
        return self.expand(correction)

    def ritz(self) -> _Ritz:
        values, left, right = scipy.linalg.eig(self.projected, left=True, right=True)
        right /= np.linalg.norm(right, axis=0)
        # c = (z^dagger V^dagger X_0) / (z^dagger s) for the left and right Ritz vectors z and s
        weights = (left.conj().T @ self.overlaps) / (
            np.einsum("ij,ij->j", left.conj(), right)
        )
        estimates = (self.measured @ right) * weights
        reference = abs(complex(jnp.vdot(self.measurement, self.start)))
        # A blind search looks for the mode with most weight in the start (the
        # remainder of a stepping, which a mode not found decays slowly in)
        order = [
            int(j) for j in np.argsort(-np.abs(weights) if self.blind else -values.real)
        ]
        seen = [
            j
            for j in order
            if self.blind or abs(estimates[j]) >= _AMPLITUDE_TOL * reference
        ]
        return _Ritz(
            values=values,
            right=right,
            weights=weights / float(jnp.linalg.norm(self.start)),
            order=seen or order,
        )

    def pair(self, ritz: _Ritz, j: int) -> tuple[jax.Array, jax.Array]:
        """Get the Ritz vector u and its residual r = (1 - Q Q^dagger) L u - theta u."""
        s = jnp.asarray(ritz.right[:, j])
        u = jnp.tensordot(s, self.basis, axes=1)
        r = jnp.tensordot(s, self.images, axes=1) - ritz.values[j] * u
        return u, r

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
        basis = jnp.tensordot(q.T, self.basis, axes=1)
        images = jnp.tensordot(q.T, self.images, axes=1)
        images -= jnp.tensordot(_dot(u[None], images)[0], u[None], axes=0)[:, 0]
        self.basis, self.images = basis, images
        q = np.asarray(q)
        # The images lose only their part along u, to which V q is orthogonal
        self.projected = q.conj().T @ self.projected @ q
        self.overlaps = q.conj().T @ self.overlaps
        self.measured = np.matmul(self.measured, q)
        if not len(self.basis):
            # No other Ritz vector is seen, as at dK = 0, where X_0 is the
            # steady state: search on in what is left of X_0, if anything
            self.expand(self.start, scale=float(jnp.linalg.norm(self.start)))

    def truncate(self, n: int) -> None:
        self.locked, self.schur = self.locked[:n], self.schur[:n, :n]

    def modes(self, rtol: float, *, optional: bool) -> list[Mode]:
        """Get the modes from the Schur form, slowest first, with their coefficients to rtol of |X_0|.

        Each eigenvector is only accurate to its residual, |r| ~ mode_rtol |lambda|,
        which leaks a little of it onto the slower modes. The coefficients are
        taken on what remains of X_0 as each mode is removed, fastest first, so
        the slowest, which would take longest to step through, are removed
        exactly. If the modes are optional, a mode whose coefficient solve
        fails is left in the remainder, which the stepper takes: the
        coefficient of each other mode is that of its left eigenvector, which
        the mode left in does not change.

        Raises
        ------
        RuntimeError
            If a coefficient solve fails, and the modes are not optional.
        """
        if not self.locked:
            return []
        rates, vectors = np.linalg.eig(self.schur)
        q = jnp.stack(self.locked)
        modes, remainder = [], self.start
        for j in np.argsort(rates.real):
            vector = jnp.tensordot(jnp.asarray(vectors[:, j]), q, axes=1)
            vector /= jnp.linalg.norm(vector)
            try:
                amplitude, coefficient, applications = _get_amplitude(
                    self.eigen,
                    complex(rates[j]),
                    vector,
                    remainder,
                    measurement=self.measurement,
                    rtol=rtol,
                )
            except RuntimeError:
                if not optional:
                    raise
                continue
            self.coefficient_applications += applications
            remainder -= coefficient * vector
            modes.append(
                Mode(
                    rate=complex(rates[j]),
                    vector=vector,
                    coefficient=coefficient,
                    amplitude=amplitude,
                )
            )
        return modes[::-1]


@dataclass(kw_only=True)
class _CostState:
    """The progress of a `CostCut`, in applications of L."""

    rule: CostCut
    since: int = 0
    """The count at the last mode found."""
    per_mode: float = 0.0
    best_saving: float = 0.0
    group: tuple[float, int, int, float] | None = None
    """Re lambda of the group being found, the count it began at, the modes found before it, and its budget."""

    def saving(self, ritz: _Ritz, j: int) -> tuple[float, int]:
        """Get the applications saved by finding the group of Ritz pair j, and its size."""
        re = ritz.values.real
        members = [i for i in ritz.order if abs(re[i] - re[j]) <= _PARTNER * abs(re[j])]
        beyond = [i for i in ritz.order if re[i] < re[j] - _PARTNER * abs(re[j])]
        stop = self.rule.stop

        def end(i: int) -> float:
            return math.log(max(abs(ritz.weights[i]), stop) / stop) / max(
                abs(re[i]), 1e-300
            )

        without = max([self.rule.start_time, *(end(i) for i in members + beyond)])
        with_group = max([self.rule.start_time, *(end(i) for i in beyond)])
        return self.rule.step_rate * (without - with_group), len(members)

    def done(
        self, search: _Search, ritz: _Ritz, j: int, count: int, *, rough: bool
    ) -> bool:
        """Decide whether to stop before finding Ritz pair j, dropping any group over its budget."""
        saving, size = self.saving(ritz, j)
        re = ritz.values.real[j]
        if self.group is not None and abs(re - self.group[0]) <= _PARTNER * abs(
            self.group[0]
        ):
            if count - self.group[1] > self.group[3]:
                search.truncate(self.group[2])
                return True
            return False
        self.best_saving = max(self.best_saving, saving)
        if count - self.since > self.best_saving:
            return True
        if rough:
            if self.rule.cost_factor * size * self.per_mode > saving:
                return True
            self.group = (re, self.since, len(search.locked), saving)
        return False

    def found(self, count: int, n_locked: int) -> None:
        self.since, self.best_saving = count, 0.0
        self.per_mode = count / n_locked


@dataclass(frozen=True, kw_only=True)
class Schur:
    """A partial Schur form L Q = Q T of the slow modes, Q an orthonormal stack of states and T upper triangular."""

    vectors: jax.Array
    """Q, (k, ...), in the eigenvectors of K."""
    matrix: ComplexArray
    """T, (k, k)."""


def find_slow_schur(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    start: jax.Array,
    measurement: jax.Array,
    selection: Selection,
    *,
    config: EvolutionConfig,
    seed: jax.Array | None = None,
    prior: Schur | None = None,
    blind: bool = False,
    scale: float = 1.0,
) -> tuple[Schur, int]:
    """Get a partial Schur form of the slowest modes of <measurement, exp(L t) start> (`find_slow_modes`) without their coefficients, and the applications of L.

    It extends a prior Schur form, if any, and looks for every mode rather
    than those the measurement sees if blind. The weights of the modes are
    scale times those in start, relative to |X_0| (`_get_mode_rtol`).
    """
    search = _search(
        eigen,
        start,
        measurement,
        selection,
        config=config,
        seed=seed,
        prior=prior,
        blind=blind,
        scale=scale,
    )
    vectors = (
        jnp.stack(search.locked)
        if search.locked
        else jnp.zeros((0, *start.shape), start.dtype)
    )
    return Schur(vectors=vectors, matrix=search.schur), search.applications


def get_schur_of(eigen: EigenLiouvillian, vector: jax.Array) -> Schur:
    """Get the Schur form of one known eigenvector, such as the steady state."""
    vector /= jnp.linalg.norm(vector)
    image = apply_eigen(eigen, vector, jnp.asarray(0j))
    return Schur(
        vectors=vector[None],
        matrix=np.array([[complex(jnp.vdot(vector, image))]]),
    )


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
    # The coefficients to well below the stop of the stepping
    return search.modes(0.1 * config.stop, optional=isinstance(selection, CostCut)), {
        "modes": search.applications,
        "coefficients": search.coefficient_applications,
    }


def _get_mode_rtol(mode_rtol: float, ritz: _Ritz, j: int, scale: float | None) -> float:
    """Get the residual of Ritz pair j, relative to |theta|, at which to lock it.

    The modes of an evolution are evolved exactly in their span, and a mode
    of weight c (relative to |X_0|, scale |X_0| / |start|) and relative
    residual e leaves an error of about c e over its lifetime. A mode of
    weight below _WEIGHT is found to (_WEIGHT / c) mode_rtol, at most
    _LOOSEST, and the others, such as the hop, to mode_rtol: over the cases
    of _RESTART, 6.5% fewer applications of L, with the same largest error
    against the reference ISF (9.4e-8). Without a scale (`get_slow_modes`),
    every mode is found to mode_rtol.
    """
    if scale is None:
        return mode_rtol * abs(ritz.values[j])
    weight = abs(ritz.weights[j]) * scale
    factor = min(_WEIGHT / max(weight, 1e-300), _LOOSEST / mode_rtol)
    return mode_rtol * max(factor, 1.0) * abs(ritz.values[j])


def _search(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    start: jax.Array,
    measurement: jax.Array,
    selection: Selection,
    *,
    config: EvolutionConfig,
    seed: jax.Array | None = None,
    prior: Schur | None = None,
    blind: bool = False,
    scale: float | None = None,
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
    are skipped, and the selection decides when to stop. The search begins
    from a seed if given, rather than X_0, and extends a prior Schur form. A
    blind search looks for every mode, the one with most weight in X_0
    first, rather than for those the measurement sees.

    Raises
    ------
    RuntimeError
        If a mode does not converge in _MAX_OUTER iterations, or its search
        stagnates, unless the selection is a `CostCut`: its modes only save
        time steps, so it keeps the modes locked before, each exact, and the
        stepper takes the rest.
    """
    search = _Search(eigen=eigen, start=start, measurement=measurement, blind=blind)
    if prior is not None:
        search.locked = list(prior.vectors)
        search.schur = np.array(prior.matrix)
    known_count = len(search.locked)
    floor = 1e3 * np.finfo(float).eps * float(jnp.abs(eigen.rates).max())
    target = selection.n_modes if isinstance(selection, ModeCount) else _MAX_MODES
    cost = _CostState(rule=selection) if isinstance(selection, CostCut) else None
    # From a seed rather than X_0, such as X_0 stepped through the probe, in
    # which the fastest modes have decayed: over the cases of _RESTART, 4.7%
    # fewer applications of L
    search.expand(start if seed is None else seed)
    iterations = 0
    # The search space is empty once X_0 is in the span of the modes locked
    while len(search.locked) < target and len(search.basis):
        ritz = search.ritz()
        j = ritz.order[0]
        theta = complex(ritz.values[j])
        u, r = search.pair(ritz, j)
        norm = float(jnp.linalg.norm(r))
        rough = norm <= _ROUGH * abs(theta)
        if (
            isinstance(selection, RateCut)
            and search.locked
            and rough
            and theta.real < -selection.rate
        ):
            break
        if cost is not None and cost.done(
            search, ritz, j, search.applications, rough=rough
        ):
            break
        if norm <= max(_get_mode_rtol(config.mode_rtol, ritz, j, scale), floor):
            search.lock(u, ritz)
            if cost is not None:
                cost.found(search.applications, len(search.locked) - known_count)
            iterations = 0
            continue
        iterations += 1
        if iterations > _MAX_OUTER or not search.correct(u, r, theta):
            if cost is not None:
                # The modes only save time steps, and each one locked is
                # exact: the stepper takes the rest
                break
            msg = f"Jacobi-Davidson did not converge: |r| {norm:.1e} at {theta:.4e}"
            raise RuntimeError(msg)
    return search
