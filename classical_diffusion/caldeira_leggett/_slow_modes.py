"""The slowest modes of a measurement of exp(L t) X_0, by Jacobi-Davidson with a partial Schur form, in the eigenvectors of K."""

# cspell: ignore maxiter bicgstab vdot eigenpair finfo
import math
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from classical_diffusion.caldeira_leggett._linear import (
    EigenLiouvillian,
    apply_eigen,
    bicgstab,
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
    """The cost of a mode over that of its Jacobi-Davidson iterations, which leave out its amplitude: 1.3 to 2.5 for H, Li and Na on Cu(111), so 3 bounds it."""


type Selection = ModeCount | RateCut | CostCut

_AMPLITUDE_TOL = 1e-8
"""The smallest amplitude of a mode, relative to the measurement at t = 0, below which it is not looked for: modes the measurement cannot see, such as the steady state at dK = 0."""
_INNER_RTOL = 0.1
"""The residual reduction of each solve of the correction equation."""
_INNER_MAXITER = 25
_MAX_OUTER = 60
"""The largest number of iterations to find one mode."""
_MAX_MODES = 50
_ROUGH = 0.1
"""The relative residual of a Ritz pair below which a selection is decided on it, as its value and weight are then close."""
_PARTNER = 0.05
"""The relative difference of Re lambda within which modes are a group."""
_RESTART = 5
"""The number of Ritz vectors kept when the search space restarts after a mode is found."""
_STAGNATION = 1e-14
"""The norm of a new search vector, once orthogonalized, below which the search has stagnated."""


def _dot(a: jax.Array, b: jax.Array) -> jax.Array:
    """<a_i, b_j> for stacks of states a (k, ...) and b (m, ...)."""
    axes = tuple(range(1, a.ndim))
    return jnp.tensordot(a.conj(), b, axes=(axes, axes))


def _project_out(q: jax.Array, x: jax.Array) -> jax.Array:
    """(1 - Q Q^dagger) x for an orthonormal stack Q."""
    return x - jnp.tensordot(_dot(q, x[None])[:, 0], q, axes=1)


@partial(jax.jit, static_argnames=("maxiter",))
def _projected_solve(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    locked: jax.Array,
    rhs: jax.Array,
    theta: jax.Array,
    rtol: jax.Array,
    *,
    maxiter: int,
) -> tuple[jax.Array, jax.Array]:
    """Solve P (L - theta) P t = rhs for t orthogonal to the orthonormal stack Q, P = 1 - Q Q^dagger.

    BiCGSTAB, preconditioned on the right by (N - theta)^-1 projected so that
    it keeps t orthogonal to Q, M_p z = M z - M Q (Q^dagger M Q)^-1 Q^dagger M z,
    with M exact for the no-jump part N. Also get the number of iterations.
    """
    rates = eigen.rates - theta
    m_q = locked / rates
    small = _dot(locked, m_q)

    def precondition(z: jax.Array) -> jax.Array:
        m_z = z / rates
        weights = jnp.linalg.solve(small, _dot(locked, m_z[None])[:, 0])
        return m_z - jnp.tensordot(weights, m_q, axes=1)

    def operator(z: jax.Array) -> jax.Array:
        t = precondition(z)
        out = apply_eigen(eigen, t, theta)
        return _project_out(locked, out)

    z, iterations = bicgstab(operator, rhs, rtol, maxiter)
    return precondition(z), iterations


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

    y, applications = refine(
        apply,
        lambda residual, inner_rtol: _projected_solve(
            eigen, vector[None], residual, shift, inner_rtol, maxiter=1000
        ),
        state - jnp.vdot(vector, state) * vector,
        rtol=rtol,
        inner_rtol=1e-4,
        passes=10,
    )
    coefficient = complex(jnp.vdot(vector, state - apply_eigen(eigen, y, shift)))
    amplitude = complex(jnp.vdot(measurement, vector)) * coefficient
    return amplitude, coefficient, applications + 1


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
    locked: list[jax.Array] = field(default_factory=list)
    schur: ComplexArray = field(default_factory=lambda: np.zeros((0, 0), complex))
    basis: list[jax.Array] = field(default_factory=list)
    images: list[jax.Array] = field(default_factory=list)
    applications: int = 0
    """The applications of L to find the modes."""
    coefficient_applications: int = 0
    """The applications of L to find their coefficients."""

    def expand(self, x: jax.Array) -> None:
        for _ in range(2):
            stack = self.locked + self.basis
            if stack:
                x = _project_out(jnp.stack(stack), x)
        size = float(jnp.linalg.norm(x))
        if size < _STAGNATION:
            msg = "Jacobi-Davidson stagnated"
            raise RuntimeError(msg)
        x /= size
        image = apply_eigen(self.eigen, x, jnp.asarray(0j))
        self.applications += 1
        if self.locked:
            image = _project_out(jnp.stack(self.locked), image)
        self.basis.append(x)
        self.images.append(image)

    def ritz(self) -> _Ritz:
        v, w = jnp.stack(self.basis), jnp.stack(self.images)
        values, left, right = scipy.linalg.eig(
            np.asarray(_dot(v, w)), left=True, right=True
        )
        right /= np.linalg.norm(right, axis=0)
        # c = (z^dagger V^dagger X_0) / (z^dagger s) for the left and right Ritz vectors z and s
        weights = (left.conj().T @ np.asarray(_dot(v, self.start[None])[:, 0])) / (
            np.einsum("ij,ij->j", left.conj(), right)
        )
        estimates = (np.asarray(_dot(self.measurement[None], v)[0]) @ right) * weights
        reference = abs(complex(jnp.vdot(self.measurement, self.start)))
        order = [int(j) for j in np.argsort(-values.real)]
        seen = [j for j in order if abs(estimates[j]) >= _AMPLITUDE_TOL * reference]
        return _Ritz(
            values=values,
            right=right,
            weights=weights / float(jnp.linalg.norm(self.start)),
            order=seen or order,
        )

    def pair(self, ritz: _Ritz, j: int) -> tuple[jax.Array, jax.Array]:
        """Get the Ritz vector u and its residual r = (1 - Q Q^dagger) L u - theta u."""
        s = jnp.asarray(ritz.right[:, j])
        u = jnp.tensordot(s, jnp.stack(self.basis), axes=1)
        r = jnp.tensordot(s, jnp.stack(self.images), axes=1) - ritz.values[j] * u
        return u, r

    def lock(self, u: jax.Array, ritz: _Ritz) -> None:
        """Extend the Schur form with u, and restart the search space from the next Ritz vectors."""
        image = apply_eigen(self.eigen, u, jnp.asarray(0j))
        self.applications += 1
        column = np.asarray(_dot(jnp.stack([*self.locked, u]), image[None])[:, 0])
        n = len(self.locked)
        grown = np.zeros((n + 1, n + 1), complex)
        grown[:n, :n] = self.schur
        grown[:, n] = column
        self.schur = grown
        self.locked.append(u)
        restart = [
            jnp.tensordot(jnp.asarray(ritz.right[:, j]), jnp.stack(self.basis), axes=1)
            for j in ritz.order[1 : 1 + _RESTART]
        ]
        self.basis, self.images = [], []
        for x in restart:
            self.expand(x)

    def truncate(self, n: int) -> None:
        self.locked, self.schur = self.locked[:n], self.schur[:n, :n]

    def modes(self, rtol: float) -> list[Mode]:
        """Get the modes from the Schur form, slowest first, with their coefficients to rtol of |X_0|.

        Each eigenvector is only accurate to its residual, |r| ~ mode_rtol |lambda|,
        which leaks a little of it onto the slower modes. The coefficients are
        taken on what remains of X_0 as each mode is removed, fastest first, so
        the slowest, which would take longest to step through, are removed
        exactly.
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


def find_slow_modes(
    eigen: EigenLiouvillian,
    start: jax.Array,
    measurement: jax.Array,
    selection: Selection,
    *,
    config: EvolutionConfig,
) -> tuple[list[Mode], dict[str, int]]:
    """Get the slowest modes of <measurement, exp(L t) start>, slowest first, by Jacobi-Davidson, and the applications of L to find them and their coefficients.

    Jacobi-Davidson targets the Ritz value of largest real part in a search
    space V, and expands V by an approximate solution of the correction
    equation (1 - u u^dagger)(L - theta)(1 - u u^dagger) t = -r: a few
    BiCGSTAB iterations, preconditioned by the exact inverse of the no-jump
    part (`_projected_solve`). The mode itself is projected out, so the
    equation stays well conditioned as theta converges. A mode is locked once
    |r| <= config.mode_rtol |theta|, or round off, into a partial Schur form,
    and the search continues outside it. Modes the measurement does not see
    are skipped, and the selection decides when to stop.

    The coefficients are one bordered solve per mode (`_get_amplitude`),
    counted apart. It needs no time steps, and no solve near a slow
    eigenvalue, which is badly conditioned: the slowest mode of the ISF of lithium on Cu(110) in
    2D costs 380-780 applications of L, where a shift-invert Arnoldi method
    needs 5,000-12,000.

    Raises
    ------
    RuntimeError
        If a mode does not converge in _MAX_OUTER iterations.
    """
    search = _Search(eigen=eigen, start=start, measurement=measurement)
    floor = 1e3 * np.finfo(float).eps * float(jnp.abs(eigen.rates).max())
    target = selection.n_modes if isinstance(selection, ModeCount) else _MAX_MODES
    cost = _CostState(rule=selection) if isinstance(selection, CostCut) else None
    search.expand(start)
    iterations = 0
    while len(search.locked) < target:
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
        if norm <= max(config.mode_rtol * abs(theta), floor):
            search.lock(u, ritz)
            if cost is not None:
                cost.found(search.applications, len(search.locked))
            iterations = 0
            continue
        iterations += 1
        if iterations > _MAX_OUTER:
            msg = f"Jacobi-Davidson did not converge: |r| {norm:.1e} at {theta:.4e}"
            raise RuntimeError(msg)
        correction, inner = _projected_solve(
            eigen,
            jnp.stack([*search.locked, u]),
            -r,
            jnp.asarray(theta),
            jnp.asarray(_INNER_RTOL),
            maxiter=_INNER_MAXITER,
        )
        search.applications += 2 * int(inner)
        search.expand(correction)
    # The coefficients to well below the stop of the stepping
    return search.modes(0.1 * config.stop), {
        "modes": search.applications,
        "coefficients": search.coefficient_applications,
    }
