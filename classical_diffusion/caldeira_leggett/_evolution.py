"""The time evolution of a measurement of exp(L t) X_0, by two-sided Lanczos (Pade via Lanczos)."""

# cspell: ignore vdot fori Strakos Tichy Liouvillians Feldmann Freund
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from classical_diffusion.caldeira_leggett._linear import (
    EigenLiouvillian,
    apply_adjoint,
    apply_eigen,
    from_eigen,
    to_eigen,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.caldeira_leggett._config import EvolutionConfig
    from classical_diffusion.caldeira_leggett._model import ComplexArray, FloatArray

_BLOCK = 25
"""The Lanczos steps of each compiled block, after which the host checks for convergence.

The coefficients of a block are one transfer to the host. The transfer
function at 1 / span is a continued fraction of the tridiagonal, O(m), and
only once it agrees is the tridiagonal diagonalized, O(m^3) (`_Poles`).
"""
_SPREAD = 16
"""The fraction of the steps so far, 1 / _SPREAD, below which two evaluations of F_m are not apart (`_evolve`).

An evaluation diagonalizes T_m, O(m^3): 40 s at m = 3,000. Hydrogen at 1000
times its mass on Cu(111) in 1D (141 bands) needs 3,400 steps at 0.3 |G|,
and evaluating after every block of _BLOCK steps took 950 s. Up to
m = 800 (every benchmark case) the evaluations stay a block apart.
"""
_TRANSFER_BLOCK = 10
"""The steps of each block of the solve for the transfer function, which converges in tens of steps."""
_MAX_STEPS = 4000
"""The most Lanczos steps. Underdamped hydrogen (gamma = 0.01 / ps, 150 K) over 2000 ps needs 1,000."""
_BREAKDOWN = 1e-12
"""The cosine |<w, v>| / (|w| |v|) of the next pair of Lanczos vectors below which the recurrence has broken down.

Over the benchmark (H, Li and Na on Cu(111) in 1D and 2D) and underdamped
hydrogen, the smallest cosine before convergence was 3.8e-9 (median per
case 3e-7 to 4e-5), and the results were unaffected: near breakdowns of
the two-sided recurrence are common for these non-normal Liouvillians and
harmless. An exact breakdown (for example a measurement orthogonal to
X_0, F(0) = 0) is cured by splitting the measurement
(`get_reduced_evolution`).
"""
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
_SLOWEST = 8
"""The slowest poles reported with an evolution."""
_AGREEMENTS = 2
"""The successive blocks over which a result must agree to within its tolerance.

With one, the change of F_m over a block stopped H, Li and Na on Cu(111)
in 1D and 2D (15 cases, stop 1e-4, 1e-6 and 1e-8, replayed from stored
coefficients) with an error of up to 1.4 stop (sodium in 1D at 0.3 |G|,
1e-8); with two, of up to 0.35 stop, for one block more.
"""
_SPLIT_SEED = 2026
"""The seed of the vector that splits a measurement on a breakdown."""


@dataclass(kw_only=True)
class _Progress:
    """The errors of successive blocks of a recurrence, to tell a floor set by round off from slow convergence."""

    tolerance: float
    """The tolerance asked for, absolute."""
    floor: float
    """The error, absolute, below which a stall is round off (_ROUNDOFF)."""
    errors: list[float] = field(default_factory=list)

    @property
    def reached(self) -> float:
        """The tolerance reached: the one asked for, or once stalled at its floor, twice the floor."""
        n = len(self.errors)
        if n <= _STALL:
            return self.tolerance
        recent, before = min(self.errors[-_STALL:]), min(self.errors[:-_STALL])
        if recent <= self.floor and recent >= 0.5 * before:
            return max(self.tolerance, 2 * recent)
        return self.tolerance

    def warn(self, what: str) -> None:
        if self.reached > self.tolerance:
            warnings.warn(
                f"{what} stalled at {self.reached:.1e} (absolute) from round off, above the {self.tolerance:.1e} asked for",
                stacklevel=4,
            )


class _BreakdownError(Exception):
    """A serious breakdown of the two-sided Lanczos recurrence: <w, v> = 0 for nonzero v and w."""


def _operators(
    eigen: EigenLiouvillian, shift: jax.Array, *, preconditioned: bool
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    """Get A and A^dagger: L - shift, or (L - shift) M, preconditioned on the right by M = (N - shift)^-1 for N the no-jump part."""
    if not preconditioned:
        return (
            lambda v: apply_eigen(eigen, v, shift),
            lambda w: apply_adjoint(eigen, w, shift),
        )
    rates = eigen.rates - shift
    return (
        lambda v: apply_eigen(eigen, v / rates, shift),
        lambda w: apply_adjoint(eigen, w, shift) / rates.conj(),
    )


@partial(jax.jit, static_argnames=("steps", "preconditioned"))
def _steps(
    eigen: EigenLiouvillian,
    carry: tuple[jax.Array, ...],
    shift: jax.Array,
    *,
    steps: int,
    preconditioned: bool,
) -> tuple[tuple[jax.Array, ...], jax.Array]:
    """Take steps of the two-sided Lanczos recurrence, and get the new carry and, for each step, alpha, beta, delta, |w_hat| and |w|.

    The right vectors v have unit norm and the left w are scaled so that
    <w, v> = 1. Then A V = V T + delta v_{m+1} e_m^T and
    A^dagger W = W T^dagger + beta^* w_{m+1} e_m^T, for T tridiagonal with
    alpha on the diagonal, beta above and delta below. Each step is one
    application of A, one of A^dagger and four reductions, with no basis
    kept.
    """
    forward, backward = _operators(eigen, shift, preconditioned=preconditioned)

    def step(  # ruff:ignore[too-many-locals]
        i: int, state: tuple[tuple[jax.Array, ...], jax.Array]
    ) -> tuple[tuple[jax.Array, ...], jax.Array]:
        (v, w, v_last, w_last, beta, delta), out = state
        av, aw = forward(v), backward(w)
        alpha = jnp.vdot(w, av)
        v_hat = av - alpha * v - beta * v_last
        w_hat = aw - alpha.conj() * w - delta.conj() * w_last
        delta_next = jnp.linalg.norm(v_hat)
        beta_next = jnp.vdot(w_hat, v_hat) / delta_next
        recorded = out.at[i].set(
            jnp.stack(
                [
                    alpha,
                    beta_next,
                    delta_next.astype(alpha.dtype),
                    jnp.linalg.norm(w_hat).astype(alpha.dtype),
                    jnp.linalg.norm(w).astype(alpha.dtype),
                ]
            )
        )
        carry = (
            v_hat / delta_next,
            w_hat / beta_next.conj(),
            v,
            w,
            beta_next,
            delta_next.astype(alpha.dtype),
        )
        return carry, recorded

    out = jnp.zeros((steps, 5), carry[0].dtype)
    return jax.lax.fori_loop(0, steps, step, (carry, out))


@dataclass(kw_only=True)
class _Lanczos:
    """The two-sided Lanczos recurrence of A from (start, measurement), as the coefficients of T so far.

    The measurement of exp(A t) start is about scale e_1^T exp(t T_m) e_1,
    which matches the first 2m moments <measurement, A^k start>.
    """

    eigen: EigenLiouvillian
    start: jax.Array
    measurement: jax.Array
    shift: complex = 0j
    preconditioned: bool = False
    alpha: list[complex] = field(default_factory=list)
    beta: list[complex] = field(default_factory=list)
    delta: list[float] = field(default_factory=list)
    smallest_cosine: float = 1.0
    invariant: bool = False
    """Whether the Krylov space is invariant, so that T_m is exact."""
    scale: complex = field(init=False)
    """<measurement, start>."""
    norm: float = field(init=False)
    """The scale of A: the largest rate of its no-jump part, or 1 preconditioned."""
    carry: tuple[jax.Array, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.scale = complex(jnp.vdot(self.measurement, self.start))
        self.norm = (
            1.0
            if self.preconditioned
            else float(jnp.abs(self.eigen.rates - self.shift).max())
        )
        size = float(jnp.linalg.norm(self.start))
        if self.scale == 0:
            raise _BreakdownError
        v = self.start / size
        w = self.measurement * (size / np.conj(self.scale))
        zero = jnp.zeros_like(v)
        self.carry = (v, w, zero, zero, jnp.asarray(0j), jnp.asarray(0j))

    @property
    def steps(self) -> int:
        return len(self.alpha)

    @property
    def applications(self) -> int:
        """One application of A and one of A^dagger per step."""
        return 2 * self.steps

    def extend(self, steps: int) -> None:
        """Take steps more, or fewer if the space is invariant.

        Raises
        ------
        _BreakdownError
            On a serious breakdown, keeping the steps before it.
        """
        self.carry, out = _steps(
            self.eigen,
            self.carry,
            jnp.asarray(self.shift),
            steps=steps,
            preconditioned=self.preconditioned,
        )
        out = np.asarray(out)
        size = self.norm
        for alpha, beta, delta, w_hat, w in out:
            self.alpha.append(complex(alpha))
            if (
                not np.all(np.isfinite([alpha, beta, delta]))
                or min(abs(delta), abs(w_hat) / abs(w)) <= _INVARIANT * size
            ):
                # The next right or left vector vanishes: the space is
                # invariant, and T_m is exact
                self.invariant = True
                return
            cosine = abs(beta) / abs(w_hat)
            self.smallest_cosine = min(self.smallest_cosine, cosine)
            if cosine < _BREAKDOWN:
                raise _BreakdownError
            self.beta.append(complex(beta))
            self.delta.append(float(delta.real))

    def tridiagonal(self) -> ComplexArray:
        m = self.steps
        return (
            np.diag(self.alpha)
            + np.diag(self.beta[: m - 1], 1)
            + np.diag(self.delta[: m - 1], -1)
        )

    def transfer(self, s: ComplexArray) -> ComplexArray:
        """Get scale e_1^T (s - T_m)^-1 e_1 at each s, as a continued fraction from the last step back."""
        value = np.zeros_like(s, dtype=complex)
        for j in range(self.steps - 1, -1, -1):
            coupling = self.beta[j] * self.delta[j] if j < self.steps - 1 else 0
            value = 1 / (s - self.alpha[j] - coupling * value)
        return self.scale * value


@dataclass(frozen=True, kw_only=True)
class _Poles:
    """The poles and residues of T_m: the measurement of exp(L t) X_0 is about sum_j residues_j exp(rates_j t)."""

    rates: ComplexArray
    residues: ComplexArray
    growth: float
    """The largest Re lambda of a pole that is kept: one of rate zero, such as the steady state at dK = 0 or a multiple of |G|, comes out at +-1e-16."""

    @staticmethod
    def of(lanczos: _Lanczos, growth: float) -> _Poles:
        rates, vectors = np.linalg.eig(lanczos.tridiagonal())
        first = np.zeros(len(rates), complex)
        first[0] = 1
        residues = lanczos.scale * vectors[0] * np.linalg.solve(vectors, first)
        return _Poles(rates=rates, residues=residues, growth=growth)

    @property
    def stable(self) -> np.ndarray:
        """Whether each pole is kept. L has no growing mode, so a pole with Re lambda > growth is spurious, and is dropped."""
        return self.rates.real <= self.growth

    @property
    def dropped(self) -> float:
        """The weight of the dropped poles."""
        return float(np.abs(self.residues[~self.stable]).sum())

    def values(self, times: FloatArray) -> ComplexArray:
        """Get sum_j r_j exp(lambda_j t) over the stable poles, a chunk of times at a time."""
        rates, residues = self.rates[self.stable], self.residues[self.stable]
        chunk = max(1, 2**22 // max(len(rates), 1))
        return np.concatenate(
            [
                np.exp(np.outer(times[i : i + chunk], rates)) @ residues
                for i in range(0, len(times), chunk)
            ]
        )

    def slowest(self, weight: float) -> ComplexArray:
        """Get the slowest kept poles of residue above weight, slowest first."""
        seen = self.stable & (np.abs(self.residues) > weight)
        rates = self.rates[seen]
        return rates[np.argsort(-rates.real)][:_SLOWEST]

    def transfer(self, s: float) -> complex:
        """Get the Laplace transform of `values` at s."""
        stable = self.stable
        return complex(np.sum(self.residues[stable] / (s - self.rates[stable])))


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
    the tolerance. A serious breakdown of the recurrence raises
    _BreakdownError.

    Raises
    ------
    RuntimeError
        If it does not converge in _MAX_STEPS steps.
    """
    lanczos = _Lanczos(
        eigen=eigen,
        start=start,
        measurement=measurement / (eigen.rates - s).conj(),
        shift=complex(s),
        preconditioned=True,
    )
    last, agreed = None, 0
    progress = _Progress(tolerance=tolerance, floor=0.0)
    while lanczos.steps < _MAX_STEPS:
        lanczos.extend(_TRANSFER_BLOCK)
        value = complex(lanczos.transfer(np.zeros(1, complex))[0])
        if lanczos.invariant:
            return value, lanczos.applications
        progress.floor = _ROUNDOFF * abs(value)
        if last is not None:
            progress.errors.append(abs(value - last))
        agreed = (
            agreed + 1
            if last is not None and abs(value - last) <= progress.reached
            else 0
        )
        if agreed >= _AGREEMENTS:
            progress.warn("The transfer function")
            return value, lanczos.applications
        last = value
    msg = f"The transfer function did not converge in {_MAX_STEPS} Lanczos steps"
    raise RuntimeError(msg)


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
    rates: ComplexArray
    """The slowest poles of weight above the tolerance, slowest first, in natural units of 1 / time."""
    applications: dict[str, int]
    """The applications of L and of L^dagger: for the evolution (lanczos), and for the transfer function at 1 / span (transfer)."""


def _transforms(lanczos: _Lanczos, s: float) -> ComplexArray:
    """Get a times the transform of F_m at points s' on lines Re s' = a, from s to the width of T a decade apart.

    A change of F_m by e at times near 1 / a changes its transform there by
    up to e / a, so until these change by less than the tolerance, F_m has
    not converged: an O(m) test (`_Lanczos.transfer`) before the O(m^3) one.
    """
    width = max(float(np.abs(lanczos.alpha).max()), s)
    lines = s * 10.0 ** np.arange(0, np.log10(width / s) + 1)
    frequencies = np.concatenate([[0.0], np.geomspace(1, 10**3, _FREQUENCIES)])
    frequencies = np.concatenate([frequencies, -frequencies[1:]])
    points = (lines[:, None] * (1 + 1j * frequencies[None, :])).ravel()
    return lanczos.transfer(points) * np.repeat(lines, len(frequencies))


def _evolve(  # ruff:ignore[too-many-arguments, too-many-positional-arguments, too-many-locals]
    eigen: EigenLiouvillian,
    start: jax.Array,
    measurement: jax.Array,
    times: FloatArray,
    tolerance: float,
    stop: float,
) -> Evolution:
    """Get <measurement, exp(L t) start> at the times to an absolute tolerance, which is stop relative (`get_reduced_evolution`).

    Raises
    ------
    _BreakdownError
        On a serious breakdown of either recurrence before any evaluation.
    RuntimeError
        If either does not converge in _MAX_STEPS steps.
    """
    s = 1 / float(times.max())
    reference, transfer_applications = get_transfer(
        eigen, start, measurement, s, _TRANSFER_TOL * tolerance / s
    )
    lanczos = _Lanczos(eigen=eigen, start=start, measurement=measurement)
    last, agreed, transforms, due = None, 0, np.zeros(0, complex), 0
    progress = _Progress(tolerance=tolerance, floor=_ROUNDOFF * tolerance / stop)
    while lanczos.steps < _MAX_STEPS:
        try:
            lanczos.extend(_BLOCK)
        except _BreakdownError:
            if last is None:
                raise
            # The steps before the breakdown are all there are
            lanczos.invariant = True
        current = _transforms(lanczos, s)
        compared = len(transforms) == len(current)
        screen = (
            max(
                float(np.abs(current - transforms).max()),
                abs(current[0] - s * reference),
            )
            if compared
            else np.inf
        )
        transforms = current
        if not lanczos.invariant and screen > progress.reached:
            # The transient is still changing, or a slow part of the
            # measurement is not found yet
            if compared:
                progress.errors.append(screen)
            last, agreed = None, 0
            continue
        if not lanczos.invariant and lanczos.steps < due:
            continue
        # Evaluations O(m^3) at least m / _SPREAD steps apart, so that their
        # cost is at most about 6 times that of the last
        due = lanczos.steps + max(_BLOCK, lanczos.steps // _SPREAD // _BLOCK * _BLOCK)
        # A pole that grows by less than 1 + stop over the span is kept
        poles = _Poles.of(lanczos, stop * s)
        values = poles.values(times)
        # T_m is exact once the space is invariant, so only then need F_m
        # not be compared with the last block
        changed = (
            0.0
            if lanczos.invariant
            else np.inf
            if last is None
            else float(np.abs(values - last).max(initial=0.0))
        )
        error = max(
            changed,
            abs(poles.transfer(s) - reference) * s,
            poles.dropped,
        )
        progress.errors.append(max(error, 0.0 if lanczos.invariant else screen))
        agreed = agreed + 1 if error <= progress.reached else 0
        last = values
        if agreed >= (1 if lanczos.invariant else _AGREEMENTS):
            progress.warn("The evolution")
            return Evolution(
                values=values,
                rates=poles.slowest(tolerance),
                applications={
                    "lanczos": lanczos.applications,
                    "transfer": transfer_applications,
                },
            )
        if lanczos.invariant:
            raise _BreakdownError
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
    has stalled below _ROUNDOFF, with a warning of what it reached.

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
    first = complex(jnp.vdot(measurement, start))
    if times.max(initial=0.0) <= 0:
        return Evolution(
            values=np.full(len(times), first),
            rates=np.zeros(0, complex),
            applications={"lanczos": 0, "transfer": 0},
        )
    # Relative to |F(0)|, or if the measurement does not see X_0 at t = 0, to
    # the largest it could be
    size = float(jnp.linalg.norm(measurement) * jnp.linalg.norm(start))
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
        rates=both.rates,
        applications={
            key: both.applications[key] + alone.applications[key]
            for key in both.applications
        },
    )
