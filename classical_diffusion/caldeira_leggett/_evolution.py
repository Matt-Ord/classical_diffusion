"""The time evolution of a measurement of exp(L t) X_0, by two-sided Lanczos (Pade via Lanczos)."""

# cspell: ignore vdot fori finfo jitted LAPACK's Strakos Tichy Liouvillians Feldmann Freund rebiorthogonalization
import dataclasses
import warnings
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from classical_diffusion.caldeira_leggett._linear import (
    EigenLiouvillian,
    apply_adjoint_split,
    apply_split,
    from_eigen,
    inner,
    magnitudes,
    multiply,
    split,
    to_eigen,
)
from classical_diffusion.caldeira_leggett._tridiagonal import (
    continued_fraction,
    get_poles,
    get_size,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.caldeira_leggett._config import EvolutionConfig
    from classical_diffusion.caldeira_leggett._model import ComplexArray, FloatArray

_BLOCK = 25
"""The Lanczos steps of each block, after which convergence is checked.

The transform of F_m at points from 1 / span to the width of T is a
continued fraction of the tridiagonal, O(m), checked on the device after
each block (`_advance`); only once it agrees are the poles of the
tridiagonal found, O(m^2) (`get_poles`).
"""
_SPREAD = 16
"""The fraction of the steps so far, 1 / _SPREAD, below which two evaluations of F_m are not apart (`_evolve`).

An evaluation finds the poles of T_m, O(m^2): about 1 s at m = 1,500. Up to
m = 800 (every benchmark case) the evaluations stay a block apart.
"""
_TRANSFER_BLOCK = 10
"""The steps of each block of the solve for the transfer function, which converges in tens of steps."""
_MAX_STEPS = 5000
"""The most Lanczos steps.

Underdamped hydrogen (gamma = 0.01 / ps, 150 K) over 2000 ps needs 1,000 to
2,300 steps, hydrogen at 1000 times its mass on Cu(111) in 1D 3,800, and
lithium at 300 K with gamma = 0.05 / ps at 0.05 |G| about 4,500, although
its sector has only 1,800 states: with full rebiorthogonalization the
recurrence needs all 1,800 steps there, as many slow modes are seen, and
without it copies of converged poles fill the rest.
"""
_BREAKDOWN = 100 * np.finfo(float).eps
"""The cosine |<w, v>| / (|w| |v|) of the next pair of Lanczos vectors below which <w, v> is round off, and the recurrence has broken down.

Small cosines are common for these non-normal Liouvillians and mostly
harmless: over the benchmark (H, Li and Na on Cu(111) in 1D and 2D) and
underdamped hydrogen, the smallest before convergence was 3.8e-9 (median
per case 3e-7 to 4e-5); at 3 |G| for hydrogen in 1D they fall to 1e-13
and the ISF is still within 8e-7 of the reference, but a threshold of
1e-12 stopped it. The error tests decide whether the recurrence has
converged. An exact breakdown (a measurement orthogonal to X_0, F(0) = 0)
is cured by splitting the measurement (`get_reduced_evolution`).
"""
_EPS = float(np.finfo(float).eps)
_INVARIANT = 1e-13
"""The norm of the next Lanczos vector, relative to the tridiagonal, below which the Krylov space is invariant and T_m exact."""
_TRANSFER_TOL = 0.1
"""The accuracy of the transfer function at 1 / span, in units of the tolerance it is compared to."""
_FREQUENCIES = 16
"""The frequencies, from 1 to 1000 times the real part on each side of zero, at which the transform of F_m is compared between blocks before it is evaluated in time."""
_ROUNDOFF = 1e-8
"""The accuracy, relative to |F(0)| (or to the transform), above which convergence that has stalled is not round off.

The two-sided recurrence has a floor in finite precision: over the 15
benchmark cases, 800 steps changed F_m by at least 3e-16 (MSD) to 1.7e-9
(H on Cu(111) in 2D at 0.05 |G|) per block. Asked for less than its
floor, the recurrence stops once its error has not halved over _STALL
evaluations below this, with a warning of the accuracy reached.
"""
_STALL = 4
"""The blocks over which an error that has not halved has stalled at its floor."""
_AGREEMENTS = 2
"""The successive blocks over which a result must agree to within its tolerance.

With one, the change of F_m over a block stopped H, Li and Na on Cu(111)
in 1D and 2D (15 cases, stop 1e-4, 1e-6 and 1e-8, replayed from stored
coefficients) with an error of up to 1.4 stop (sodium in 1D at 0.3 |G|,
1e-8); with two, of up to 0.35 stop, for one block more.
"""
_SPLIT_SEED = 2026
"""The seed of the vector that splits a measurement on a breakdown."""


_FALSE = jnp.zeros((), bool)
_TRUE = jnp.ones((), bool)
_LINES = 24
"""The most lines Re s' = a of the screen of the transform of F_m, a decade apart from 1 / span (`_screen`): T of width up to 1e23 / span."""
_CAPACITY = get_size(_MAX_STEPS + _BLOCK)
"""The length of the coefficient buffers on the device, at least the padded size of the most steps (`get_poles`)."""
_ERRORS = _MAX_STEPS // _TRANSFER_BLOCK + 2
"""The length of the buffer of the errors of the blocks (`_Progress`): at most one per block."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class _Progress:
    """The errors of successive blocks of a recurrence, on the device, to tell a floor set by round off from slow convergence."""

    tolerance: jax.Array
    """The tolerance asked for, absolute."""
    floor: jax.Array
    """The error, absolute, below which a stall is round off (_ROUNDOFF)."""
    precision: jax.Array
    """The error, absolute, that round off of the recurrence leaves (`_Recurrence.roundoff`)."""
    errors: jax.Array
    """The errors of the blocks so far, (_ERRORS,), the first `count` of them."""
    count: jax.Array

    @staticmethod
    def start(tolerance: float, floor: float) -> _Progress:
        return _Progress(
            tolerance=jnp.asarray(tolerance),
            floor=jnp.asarray(floor),
            precision=jnp.asarray(0.0),
            errors=jnp.zeros(_ERRORS),
            count=jnp.asarray(0),
        )

    @property
    def reached(self) -> jax.Array:
        """The tolerance that can be reached: the one asked for, or the precision of the recurrence, or once stalled at its floor, twice the floor."""
        # The estimate is of the order of the error: twice it, so that an
        # error at the estimate does not stop each agreement
        reached = jnp.maximum(self.tolerance, 2 * self.precision)
        index = jnp.arange(_ERRORS)
        recent = jnp.where(
            (index >= self.count - _STALL) & (index < self.count), self.errors, jnp.inf
        ).min()
        before = jnp.where(index < self.count - _STALL, self.errors, jnp.inf).min()
        stalled = (
            (self.count > _STALL) & (recent <= self.floor) & (recent >= 0.5 * before)
        )
        return jnp.where(stalled, jnp.maximum(reached, 2 * recent), reached)

    def append(self, error: jax.Array, *, where: jax.Array = _TRUE) -> _Progress:
        """Append an error, if where."""
        return dataclasses.replace(
            self,
            errors=self.errors.at[self.count].set(
                jnp.where(where, error, self.errors[self.count])
            ),
            count=self.count + jnp.where(where, 1, 0),
        )


def _warn(what: str, progress: _Progress) -> None:
    reached, tolerance = (float(x) for x in (progress.reached, progress.tolerance))
    if reached > tolerance:
        warnings.warn(
            f"{what} is accurate to about {reached:.1e} (absolute) from round"
            f" off, above the {tolerance:.1e} asked for",
            stacklevel=4,
        )


class _BreakdownError(Exception):
    """A serious breakdown of the two-sided Lanczos recurrence: <w, v> = 0 for nonzero v and w."""


def _operators(
    eigen: EigenLiouvillian, shift: jax.Array, *, preconditioned: bool
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    """Get A and A^dagger on split states: L - shift, or (L - shift) M, preconditioned on the right by M = (N - shift)^-1 for N the no-jump part."""
    if not preconditioned:
        return (
            lambda v: apply_split(eigen, v, shift),
            lambda w: apply_adjoint_split(eigen, w, shift),
        )
    inverse = 1 / (eigen.rates - shift)
    return (
        lambda v: apply_split(eigen, multiply(inverse, v), shift),
        lambda w: multiply(inverse.conj(), apply_adjoint_split(eigen, w, shift)),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class _Recurrence:
    """The two-sided Lanczos recurrence of A from (start, measurement), on the device: the last two pairs of vectors, and the coefficients of T so far.

    The right vectors v have unit norm and the left w are scaled so that
    <w, v> = 1. Then A V = V T + delta v_{m+1} e_m^T and
    A^dagger W = W T^dagger + beta^* w_{m+1} e_m^T, for T tridiagonal with
    alpha on the diagonal, beta above and delta below. The measurement of
    exp(A t) start is about scale e_1^T exp(t T_m) e_1, which matches the
    first 2m moments <measurement, A^k start>.
    """

    vectors: tuple[jax.Array, ...]
    """v, w, the v and w before them, split into real and imaginary parts (`split`), and the last beta and delta."""
    alpha: jax.Array
    """The diagonal of T, (_CAPACITY,), the first `steps` of it."""
    coupling: jax.Array
    """beta_k delta_k, the products of the off-diagonals of T, (_CAPACITY,), zero beyond the steps."""
    steps: jax.Array
    invariant: jax.Array
    """Whether the Krylov space is invariant, so that T_m is exact."""
    broken: jax.Array
    """Whether the recurrence has broken down: the next <w, v> is round off."""
    roundoff: jax.Array
    """The relative error of the coefficients from round off: the sum over the steps of eps times the cancellation in <w_hat, v_hat> (`_take`)."""

    @staticmethod
    def start(
        start: jax.Array, measurement: jax.Array, scale: jax.Array
    ) -> _Recurrence:
        size = jnp.linalg.norm(start)
        v = start / size
        w = measurement * (size / jnp.conj(scale))
        zero = jnp.zeros_like(split(v))
        return _Recurrence(
            vectors=(split(v), split(w), zero, zero, jnp.asarray(0j), jnp.asarray(0j)),
            alpha=jnp.zeros(_CAPACITY, complex),
            coupling=jnp.zeros(_CAPACITY, complex),
            steps=jnp.asarray(0),
            invariant=_FALSE,
            broken=_FALSE,
            roundoff=jnp.asarray(0.0),
        )

    @property
    def stopped(self) -> jax.Array:
        return self.invariant | self.broken


def _steps(
    eigen: EigenLiouvillian,
    vectors: tuple[jax.Array, ...],
    shift: jax.Array,
    *,
    steps: int,
    preconditioned: bool,
) -> tuple[tuple[jax.Array, ...], jax.Array]:
    """Take steps of the two-sided Lanczos recurrence, and get the new vectors and, for each step, alpha, beta, delta, |w_hat|, |w| and sum |w_hat| |v_hat|.

    Each step is one application of A, one of A^dagger and four reductions,
    with no basis kept. The vectors are split into their real and imaginary
    parts (`split`), as the products of L are real.
    """
    forward, backward = _operators(eigen, shift, preconditioned=preconditioned)

    def step(  # ruff:ignore[too-many-locals]
        i: int, state: tuple[tuple[jax.Array, ...], jax.Array]
    ) -> tuple[tuple[jax.Array, ...], jax.Array]:
        (v, w, v_last, w_last, beta, delta), out = state
        av, aw = forward(v), backward(w)
        alpha = inner(w, av)
        v_hat = av - multiply(alpha, v) - multiply(beta, v_last)
        w_hat = aw - multiply(alpha.conj(), w) - multiply(delta.conj(), w_last)
        v_squares, w_squares = magnitudes(v_hat), magnitudes(w_hat)
        delta_next = jnp.sqrt(v_squares.sum())
        beta_next = inner(w_hat, v_hat) / delta_next
        recorded = out.at[i].set(
            jnp.stack(
                [
                    alpha,
                    beta_next,
                    delta_next.astype(alpha.dtype),
                    jnp.sqrt(w_squares.sum()).astype(alpha.dtype),
                    jnp.sqrt(jnp.sum(w**2)).astype(alpha.dtype),
                    jnp.sum(jnp.sqrt(w_squares * v_squares)).astype(alpha.dtype),
                ]
            )
        )
        vectors = (
            v_hat / delta_next,
            multiply(1 / beta_next.conj(), w_hat),
            v,
            w,
            beta_next,
            delta_next.astype(alpha.dtype),
        )
        return vectors, recorded

    out = jnp.zeros((steps, 6), complex)
    return jax.lax.fori_loop(0, steps, step, (vectors, out))


def _take(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    recurrence: _Recurrence,
    shift: jax.Array,
    norm: jax.Array,
    *,
    steps: int,
    preconditioned: bool,
) -> _Recurrence:
    """Take steps more, or fewer if the space becomes invariant or the recurrence breaks down, keeping the steps before (and the diagonal of that step).

    norm is the scale of A: the largest rate of its no-jump part, or 1
    preconditioned.
    """
    vectors, out = _steps(
        eigen, recurrence.vectors, shift, steps=steps, preconditioned=preconditioned
    )
    alpha, beta, delta, w_hat, w, magnitude = out.T
    # The next right or left vector vanishes: the space is invariant, and
    # T_m is exact
    invariant = ~(jnp.isfinite(alpha) & jnp.isfinite(beta) & jnp.isfinite(delta)) | (
        jnp.minimum(jnp.abs(delta), jnp.abs(w_hat) / jnp.abs(w)) <= _INVARIANT * norm
    )
    broken = ~invariant & (jnp.abs(beta) / jnp.abs(w_hat) < _BREAKDOWN)
    stop = invariant | broken
    first = jnp.where(stop.any(), jnp.argmax(stop), steps)
    kept = jnp.arange(steps) < first
    # <w_hat, v_hat> = beta delta is a sum of terms up to magnitude, so has a
    # relative error of eps magnitude / |beta delta|
    roundoff = jnp.where(kept, _EPS * jnp.abs(magnitude) / jnp.abs(beta * delta), 0)
    m = recurrence.steps
    return _Recurrence(
        vectors=vectors,
        alpha=jax.lax.dynamic_update_slice(
            recurrence.alpha, jnp.where(jnp.arange(steps) <= first, alpha, 0), (m,)
        ),
        coupling=jax.lax.dynamic_update_slice(
            recurrence.coupling, jnp.where(kept, beta * delta, 0), (m,)
        ),
        steps=m + jnp.where(stop.any(), first + 1, steps),
        invariant=stop.any() & invariant[jnp.minimum(first, steps - 1)],
        broken=stop.any() & broken[jnp.minimum(first, steps - 1)],
        roundoff=recurrence.roundoff + roundoff.sum(),
    )


@partial(jax.jit, static_argnames=("max_steps",))
def _transfer_loop(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    recurrence: _Recurrence,
    shift: jax.Array,
    scale: jax.Array,
    progress: _Progress,
    *,
    max_steps: int,
) -> tuple[_Recurrence, _Progress, jax.Array, jax.Array]:
    """Run the recurrence of `get_transfer` on the device until its value agrees over _AGREEMENTS blocks, the space is invariant, it breaks down, or max_steps; get the recurrence, the progress, the value and whether it agreed."""

    def proceed(state: tuple) -> jax.Array:
        recurrence, _progress, _last, agreed, _value = state
        return (
            ~recurrence.stopped
            & (agreed < _AGREEMENTS)
            & (recurrence.steps < max_steps)
        )

    def block(state: tuple) -> tuple:
        recurrence, progress, last, agreed, _value = state
        recurrence = _take(
            eigen,
            recurrence,
            shift,
            jnp.asarray(1.0),
            steps=_TRANSFER_BLOCK,
            preconditioned=True,
        )
        value = (
            scale
            * continued_fraction(
                recurrence.alpha,
                recurrence.coupling,
                recurrence.steps,
                jnp.zeros(1, complex),
            )[0]
        )
        progress = dataclasses.replace(
            progress,
            floor=_ROUNDOFF * jnp.abs(value),
            precision=recurrence.roundoff * jnp.abs(value),
        )
        compared = jnp.isfinite(last) & ~recurrence.invariant
        change = jnp.abs(value - last)
        progress = progress.append(change, where=compared)
        agreed = jnp.where(compared & (change <= progress.reached), agreed + 1, 0)
        return recurrence, progress, value, agreed, value

    recurrence, progress, _, agreed, value = jax.lax.while_loop(
        proceed,
        block,
        (
            recurrence,
            progress,
            jnp.asarray(np.inf + 0j),
            jnp.asarray(0),
            jnp.asarray(0j),
        ),
    )
    return recurrence, progress, value, agreed >= _AGREEMENTS


def get_transfer(
    eigen: EigenLiouvillian,
    start: jax.Array,
    measurement: jax.Array,
    s: float,
    tolerance: float,
) -> tuple[complex, int]:
    """Get <measurement, (s - L)^-1 start> to an absolute tolerance, and the applications of L.

    This is the Laplace transform at s of the measurement of exp(L t) start.
    It is the same two-sided Lanczos recurrence, on A = (L - s) M,
    preconditioned on the right by the exact inverse M of the shifted no-jump
    part: <g, (s - L)^-1 X> = -<M^dagger g, A^-1 X>, about
    -scale e_1^T T_m^-1 e_1, a continued fraction. This is the estimate of BiCG for a bilinear
    form, whose error is quadratic in the residuals (Strakos and Tichy, SIAM
    J. Sci. Comput. 33, 565 (2011)): over H, Li and Na on Cu(111) in 1D and
    H in 2D at s = 0 (the response of D), it reaches 1e-12 relative in
    66-266 applications of L, where BiCGSTAB needs 133-357 for the
    solution. It stops once two successive blocks change it by less than
    the tolerance, all on the device (`_transfer_loop`).

    Raises
    ------
    _BreakdownError
        On a serious breakdown of the recurrence.
    RuntimeError
        If it does not converge in _MAX_STEPS steps.
    """
    measurement /= (eigen.rates - s).conj()
    scale = complex(jnp.vdot(measurement, start))
    if scale == 0:
        raise _BreakdownError
    recurrence, progress, value, agreed = _transfer_loop(
        eigen,
        _Recurrence.start(start, measurement, jnp.asarray(scale)),
        jnp.asarray(complex(s)),
        jnp.asarray(scale),
        _Progress.start(tolerance, 0.0),
        max_steps=_MAX_STEPS,
    )
    steps, invariant, broken, agreed = jax.device_get(
        (recurrence.steps, recurrence.invariant, recurrence.broken, agreed)
    )
    if broken:
        raise _BreakdownError
    if not (invariant or agreed):
        msg = f"The transfer function did not converge in {_MAX_STEPS} Lanczos steps"
        raise RuntimeError(msg)
    if agreed:
        _warn("The transfer function", progress)
    return complex(value), 2 * int(steps)


def get_measurement(
    eigen: EigenLiouvillian,
    measure: Callable[[jax.Array], jax.Array],
    like: jax.Array,
) -> jax.Array:
    """Get g with measure(X) = <g, Y> for a state Y = A^-1 X B^-dagger in the eigenvectors of K."""
    (transposed,) = jax.linear_transpose(lambda y: measure(from_eigen(eigen, y)), like)(
        jnp.asarray(1.0 + 0j)
    )
    return transposed.conj()


@dataclass(frozen=True, kw_only=True)
class Evolution:
    """A measurement of exp(L t) X_0 at the times, and how it was found."""

    values: ComplexArray
    applications: dict[str, int]
    """The applications of L and of L^dagger: for the evolution (lanczos), and for the transfer function at 1 / span (transfer)."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class _Screen:
    """The screen of the evolution between evaluations (`_advance`): a times the transform of F_m at points s' on lines Re s' = a (`_transforms`), and the agreement of the evaluations so far."""

    transforms: jax.Array
    """(_LINES, len(frequencies)), zero on the lines beyond `lines`."""
    lines: jax.Array
    """The number of lines, -1 before the first block."""
    screen: jax.Array
    """The change of the transforms over the last block, and of the transform at 1 / span from the one found directly."""
    reset: jax.Array
    """Whether a block failed the screen since the last evaluation, which then is not compared with the one before."""
    due: jax.Array
    """The steps before which no evaluation is due."""


def _transforms(
    recurrence: _Recurrence,
    scale: jax.Array,
    decades: jax.Array,
    frequencies: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Get a times the transform of F_m at points s' on lines Re s' = a, from decades[0] = 1 / span to the width of T a decade apart, and the number of lines.

    A change of F_m by e at times near 1 / a changes its transform there by
    up to e / a, so until these change by less than the tolerance, F_m has
    not converged: an O(m) test before the O(m^2) one.
    """
    s = decades[0]
    valid = jnp.arange(_CAPACITY) < recurrence.steps
    width = jnp.maximum(jnp.where(valid, jnp.abs(recurrence.alpha), 0).max(), s)
    lines = jnp.minimum(jnp.ceil(jnp.log10(width / s) + 1).astype(int), _LINES)
    points = decades[:, None] * (1 + 1j * frequencies[None, :])
    values = scale * continued_fraction(
        recurrence.alpha, recurrence.coupling, recurrence.steps, points.ravel()
    ).reshape(points.shape)
    return jnp.where(
        jnp.arange(_LINES)[:, None] < lines, values * decades[:, None], 0
    ), lines


@partial(jax.jit, static_argnames=("max_steps",))
def _advance(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    recurrence: _Recurrence,
    screen: _Screen,
    progress: _Progress,
    constants: tuple[jax.Array, ...],
    *,
    max_steps: int,
) -> tuple[_Recurrence, _Screen, _Progress, jax.Array]:
    """Take blocks of the evolution's recurrence on the device until an evaluation is due: the transforms of F_m agree with the last block and at 1 / span with the transfer function, and the steps are at least screen.due; or until the space is invariant, the recurrence breaks down, or max_steps. Also get whether an evaluation is due."""
    scale, norm, decades, frequencies, reference = constants

    def proceed(state: tuple) -> jax.Array:
        recurrence, _screen, _progress, due = state
        return ~due & (recurrence.steps < max_steps)

    def block(state: tuple) -> tuple:
        recurrence, screen, progress, _ = state
        recurrence = _take(
            eigen, recurrence, jnp.asarray(0j), norm, steps=_BLOCK, preconditioned=False
        )
        progress = dataclasses.replace(
            progress, precision=recurrence.roundoff * jnp.abs(scale)
        )
        current, lines = _transforms(recurrence, scale, decades, frequencies)
        compared = lines == screen.lines
        change = jnp.where(
            compared,
            jnp.maximum(
                jnp.abs(current - screen.transforms).max(),
                jnp.abs(current[0, 0] - decades[0] * reference),
            ),
            jnp.inf,
        )
        # The transient is still changing, or a slow part of the
        # measurement is not found yet
        failed = ~recurrence.stopped & (change > progress.reached)
        progress = progress.append(change, where=failed & compared)
        due = recurrence.stopped | (~failed & (recurrence.steps >= screen.due))
        screen = _Screen(
            transforms=current,
            lines=lines,
            screen=change,
            reset=screen.reset | failed,
            due=screen.due,
        )
        return recurrence, screen, progress, due

    return jax.lax.while_loop(proceed, block, (recurrence, screen, progress, _FALSE))


@partial(jax.jit, static_argnames=("chunk",))
def _judge(  # ruff:ignore[too-many-arguments, too-many-positional-arguments, too-many-locals]
    poles: tuple[jax.Array, jax.Array],
    times: jax.Array,
    last: tuple[jax.Array, jax.Array, jax.Array],
    stopped: jax.Array,
    screen: _Screen,
    progress: _Progress,
    constants: tuple[jax.Array, jax.Array, jax.Array],
    *,
    chunk: int,
) -> tuple[jax.Array, _Progress, jax.Array, jax.Array, jax.Array]:
    """Evaluate F_m at the times from its poles, keeping those that decay or grow by less than the tolerance over the span, and judge it.

    Get F_m, the progress with the error of this evaluation, the agreements
    so far and whether they are enough.
    """
    rates, residues = poles
    previous, compared, agreed = last
    span, s, reference = constants
    tolerance = progress.tolerance
    growth = jnp.abs(residues) * jnp.expm1(jnp.maximum(rates.real, 0) * span)
    stable = growth <= tolerance
    # A dropped pole may grow by exp(1e3) over the span: inf times 0
    kept, decaying = jnp.where(stable, residues, 0), jnp.where(stable, rates, 0)
    values = jax.lax.map(
        lambda part: jnp.exp(part[:, None] * decaying[None, :]) @ kept,
        times.reshape(-1, chunk),
    ).ravel()
    compared &= ~screen.reset
    agreed = jnp.where(screen.reset, 0, agreed)
    # T_m is exact once the space is invariant, so only then need F_m not be
    # compared with the last evaluation
    changed = jnp.where(
        stopped,
        0.0,
        jnp.where(
            compared,
            jnp.abs(values[: previous.shape[0]] - previous).max(initial=0.0),
            jnp.inf,
        ),
    )
    error = jnp.maximum(
        jnp.maximum(
            changed,
            jnp.abs(jnp.where(stable, residues / (s - rates), 0).sum() - reference) * s,
        ),
        jnp.where(stable, 0, jnp.abs(residues)).sum(),
    )
    progress = progress.append(jnp.maximum(error, jnp.where(stopped, 0, screen.screen)))
    agreed = jnp.where(error <= progress.reached, agreed + 1, 0)
    done = agreed >= jnp.where(stopped, 1, _AGREEMENTS)
    return values, progress, agreed, done, error


def _evolve(  # ruff:ignore[too-many-arguments, too-many-positional-arguments, too-many-locals]
    eigen: EigenLiouvillian,
    start: jax.Array,
    measurement: jax.Array,
    times: FloatArray,
    tolerance: float,
    stop: float,
) -> Evolution:
    """Get <measurement, exp(L t) start> at the times to an absolute tolerance, which is stop relative (`get_reduced_evolution`).

    The blocks of the recurrence and the screen between evaluations run on
    the device (`_advance`); the host reads the steps and flags once per
    evaluation, which finds the poles of T_m in O(m^2) (`get_poles`), and
    once more for its verdict (`_judge`).

    Raises
    ------
    _BreakdownError
        On a serious breakdown of either recurrence before any evaluation.
    RuntimeError
        If either does not converge in _MAX_STEPS steps.
    """
    span = float(times.max())
    s = 1 / span
    reference, transfer_applications = get_transfer(
        eigen, start, measurement, s, _TRANSFER_TOL * tolerance / s
    )
    scale = complex(jnp.vdot(measurement, start))
    if scale == 0:
        raise _BreakdownError
    recurrence = _Recurrence.start(start, measurement, jnp.asarray(scale))
    frequencies = np.concatenate([[0.0], np.geomspace(1, 10**3, _FREQUENCIES)])
    constants = (
        jnp.asarray(scale),
        jnp.abs(eigen.rates).max(),
        jnp.asarray(s * 10.0 ** np.arange(_LINES)),
        jnp.asarray(np.concatenate([frequencies, -frequencies[1:]])),
        jnp.asarray(reference),
    )
    screen = _Screen(
        transforms=jnp.zeros((_LINES, 2 * _FREQUENCIES + 1), complex),
        lines=jnp.asarray(-1),
        screen=jnp.asarray(np.inf),
        reset=_FALSE,
        due=jnp.asarray(0),
    )
    progress = _Progress.start(tolerance, _ROUNDOFF * tolerance / stop)
    last = (jnp.zeros(len(times), complex), _FALSE, jnp.asarray(0))
    while True:
        recurrence, screen, progress, due = _advance(
            eigen, recurrence, screen, progress, constants, max_steps=_MAX_STEPS
        )
        steps, stopped, due = jax.device_get(
            (recurrence.steps, recurrence.stopped, due)
        )
        if not due:
            msg = f"The evolution did not converge in {_MAX_STEPS} Lanczos steps"
            raise RuntimeError(msg)
        steps = int(steps)
        rates, weights = get_poles(
            recurrence.alpha, recurrence.coupling, steps, span=span
        )
        residues = scale * weights
        chunk = max(1, min(len(times), 2**22 // len(rates)))
        values, progress, agreed, done, error = _judge(
            (rates, residues),
            jnp.asarray(np.concatenate([times, np.zeros(-len(times) % chunk)])),
            last,
            recurrence.stopped,
            screen,
            progress,
            (jnp.asarray(span), jnp.asarray(s), jnp.asarray(reference)),
            chunk=chunk,
        )
        last = (values[: len(times)], _TRUE, agreed)
        # Evaluations O(m^2) at least m / _SPREAD steps apart, so that their
        # cost is at most about 6 times that of the last
        screen = dataclasses.replace(
            screen,
            reset=_FALSE,
            due=jnp.asarray(steps + max(_BLOCK, steps // _SPREAD // _BLOCK * _BLOCK)),
        )
        done, broken = jax.device_get((done, recurrence.broken))
        if done:
            if broken:
                warnings.warn(
                    f"The two-sided Lanczos recurrence broke down after {steps}"
                    f" steps, where F_m agreed with its transform to {float(error):.1e}",
                    stacklevel=3,
                )
            _warn("The evolution", progress)
            return Evolution(
                values=np.asarray(values[: len(times)]),
                applications={
                    "lanczos": 2 * steps,
                    "transfer": transfer_applications,
                },
            )
        if stopped:
            raise _BreakdownError
        if steps >= _MAX_STEPS:
            msg = f"The evolution did not converge in {_MAX_STEPS} Lanczos steps"
            raise RuntimeError(msg)


def get_reduced_evolution(
    eigen: EigenLiouvillian,
    initial: jax.Array,
    measure: Callable[[jax.Array], jax.Array],
    times: FloatArray,
    *,
    config: EvolutionConfig,
) -> Evolution:
    """Get measure(exp(L t) initial) at the times (from zero), by two-sided Lanczos (Pade via Lanczos, PVL).

    L is given in the eigenvectors of K (`get_eigen_liouvillian`), and
    initial in the bands. measure must be linear over the complex numbers.
    The two-sided Lanczos recurrence from X_0 and the measurement g, with
    one application of L and one of L^dagger per step and no basis kept,
    gives a tridiagonal T_m whose poles lambda_j and residues r_j match the
    first 2m moments <g, L^k X_0> (Feldmann and Freund, IEEE Trans. CAD 14,
    639 (1995)), and F_m(t) = sum_j r_j exp(lambda_j t) at any time. The
    poles are those the measurement sees, so its slow modes are among them,
    found without a search.

    The error at every time is controlled by two tests against
    config.stop |F(0)|:

    - F_m at the times changes by less than it over each of two blocks of
      _BLOCK steps. This resolves the transient, which converges first.
    - The Laplace transform of F_m at s = 1 / span agrees with the
      transform found directly (`get_transfer`) to within it / s. A slow
      mode that the recurrence has not found yet leaves F_m unchanged from
      block to block, but not its transform: one of weight c and rate up to
      1 / span changes it by about c span. For hydrogen on Cu(111) in 2D at
      0.3 |G|, the exchange mode (0.010 / ps, weight 1e-3) is found only
      after 350 steps, and the change of F_m alone stops after 175-250
      steps with an error of 1.1e-3 when asked for 1e-4.

    Poles with Re lambda > 0 are spurious (L has no growing mode): they are
    dropped, and their weight must be below the tolerance. A pole that grows
    by less than a factor 1 + stop over the span, such as the steady state at
    dK = 0 (whose rate comes out at +-1e-16), is kept.

    The recurrence has an accuracy floor in finite precision, from 1e-16 to
    2e-9 of |F(0)| on the benchmark: asked for less, it stops once its error
    has stalled below _ROUNDOFF, with a warning of what it reached. The
    floor is estimated as it goes, as eps times the cancellation in the
    inner products <w, v> summed over the steps (`_Lanczos.roundoff`): it
    is 1e-10 to 1e-9 on the benchmark, but at 4 |G| for hydrogen on
    Cu(111) in 1D, where the ISF falls by 4 decades at once, the
    cancellation reaches 1e9 and the estimate 2.0e-5, against an error of
    2.2e-5 from the reference: the tolerance is raised to it, with a
    warning.

    A serious breakdown of the recurrence (<w, v> = 0, as when F(0) = 0)
    belongs to the pair (X_0, g), and is cured by splitting the
    measurement, g = (g + h) - h for a fixed random h.

    Raises
    ------
    RuntimeError
        If the split recurrences also break down, or a recurrence does not
        converge in _MAX_STEPS steps.
    """
    start = to_eigen(eigen, initial)
    measurement = get_measurement(eigen, measure, start)
    times = np.asarray(times, dtype=np.float64)
    first, size = jax.device_get(
        (
            jnp.vdot(measurement, start),
            jnp.linalg.norm(measurement) * jnp.linalg.norm(start),
        )
    )
    first, size = complex(first), float(size)
    if times.max(initial=0.0) <= 0:
        return Evolution(
            values=np.full(len(times), first),
            applications={"lanczos": 0, "transfer": 0},
        )
    # Relative to |F(0)|, or if the measurement does not see X_0 at t = 0, to
    # the largest it could be
    tolerance = config.stop * (abs(first) if abs(first) > 1e-12 * size else size)
    try:
        return _evolve(eigen, start, measurement, times, tolerance, config.stop)
    except _BreakdownError:
        pass
    rng = np.random.default_rng(_SPLIT_SEED)
    split = jnp.asarray(
        rng.standard_normal(start.shape) + 1j * rng.standard_normal(start.shape)
    )
    split *= jnp.linalg.norm(measurement) / jnp.linalg.norm(split)
    try:
        both = _evolve(
            eigen, start, measurement + split, times, tolerance / 2, config.stop
        )
        alone = _evolve(eigen, start, split, times, tolerance / 2, config.stop)
    except _BreakdownError as error:
        msg = (
            "The two-sided Lanczos recurrence broke down, also with a split measurement"
        )
        raise RuntimeError(msg) from error
    return Evolution(
        values=both.values - alone.values,
        applications={
            key: both.applications[key] + alone.applications[key]
            for key in both.applications
        },
    )
