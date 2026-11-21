"""The time evolution of a measurement of exp(L t) X_0, as its slow modes exactly plus a Krylov exponential stepper on the rest."""

# cspell: ignore vdot eigvals
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np

from classical_diffusion.caldeira_leggett._krylov import KrylovExponential
from classical_diffusion.caldeira_leggett._linear import (
    EigenLiouvillian,
    apply_eigen,
    from_eigen,
    to_eigen,
)
from classical_diffusion.caldeira_leggett._slow_modes import (
    _MAX_MODES,
    CostCut,
    ModeCount,
    Schur,
    _padded,
    find_slow_schur,
    get_schur_of,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.caldeira_leggett._config import EvolutionConfig
    from classical_diffusion.caldeira_leggett._model import ComplexArray, FloatArray


def _to_real(x: jax.Array) -> jax.Array:
    return jnp.stack([x.real, x.imag])


def _to_complex(y: jax.Array) -> jax.Array:
    return y[0] + 1j * y[1]


def _overlaps(q: jax.Array, y: jax.Array) -> jax.Array:
    """Q^dagger y for a stack of states Q."""
    return jnp.tensordot(
        q.conj(), y, axes=(tuple(range(1, q.ndim)), tuple(range(y.ndim)))
    )


def _outside(q: jax.Array, y: jax.Array) -> jax.Array:
    """(1 - Q Q^dagger) y for an orthonormal stack Q, which rows of zeros may pad."""
    return y - jnp.tensordot(_overlaps(q, y), q, axes=1)


_CHUNK = 250
"""The most steps of one solve. A longer evolution continues in another, from where it stopped, and the probe ends after one."""
_PROBE = 2.0
"""The time stepped before the slow modes are found, in units of 1 / friction (`get_reduced_evolution`).

The search starts from the state there, in which the fastest parts have
decayed, and `CostCut` estimates the stepping each group of modes saves
from it. From 1 / friction rather than 2 the estimates find 8 or 9 modes
for hydrogen on Cu(111) in 2D (ISF at 0.3 |G|: 2,335 rather than 1,756
applications of L), though the probe costs less in 1D (-7% in all).
Without the estimates (the modes of `get_slow_modes` and the measured check
alone, branch robust-ratecut), 1 / friction is best, but lithium in 2D
then needs 25% more at 0.05 |G|.
"""
_CHECK = 7
"""The steps (of the mean length of the last) after which the remainder is checked for a mode left in it (`get_reduced_evolution`).

Every 7 rather than 10 steps, with the cost per mode updated as the check
finds modes: over H, Li and Na on Cu(111) in 1D and H in 2D (3 seeds), from
-11% (hydrogen in 1D, ISF at 0.05 |G|) to +0%, -3.2% in all; lithium in 2D
at 0.3 |G| is unchanged. Without the estimates of `CostCut` (branch
robust-ratecut), checks every 5 steps found more modes than they saved for
hydrogen and lithium in 2D.
"""
_COST_FACTOR = 3.0
"""The margin on the estimated cost of a mode, the mean applications of L per mode found so far, against the stepping that the remainder still needs (`get_reduced_evolution`).

The cost of a mode varies from mode to mode (the hop costs most). Over H,
Li and Na on Cu(111) in 1D and H in 2D, halving it (with _ESTIMATED kept)
changes the applications of L per case by -7% to +9%, and doubling it by
-12% to +16%, and the total by +0.2% and +1.2%.
"""
_ESTIMATED = 2.0
"""The further margin of a decision on the Ritz estimate of the stepping a group of modes saves (`CostCut`) over one on the stepping measured.

A group whose estimated saving is within a factor 2 of its cost is left to
the check of the remainder, which measures it. With one margin for both,
the groups decided closest to it (saving / cost 1.05, 1.15 and 1.17 for the
MSD of hydrogen in 2D and 1D and the ISF of lithium in 1D at 0.3 |G|) went
either way with the round off of the steady state: the MSD of hydrogen in
1D took 786-951 applications of L over 4 seeds, and 773-815 with this
margin, and that in 2D 1,408 rather than 1,366-1,374; the ISF of lithium in
1D takes 4% more (1,716-1,753).
"""
_REMAINDER_STOP = 0.1
"""The stop of the remainder off the slow modes, in units of config.stop.

The part of the remainder in the span of the slow modes is evolved exactly
from where the stepping stops, but some of it belongs to the faster modes
left in the remainder, about as much as the norm of the remainder off the
span. This error lasts as long as the slow modes, so the remainder is
stepped further than config.stop: the largest error over H, Li and Na on
Cu(111) in 1D, 2 seeds, falls from 6.8e-7 (1) and 2.7e-7 (0.3) to 8.9e-8
(0.1), against 2.0e-7 when the coefficients of the modes were solved for to
0.1 config.stop, for 4% more applications of L than 0.3.
"""

_FLOOR = 3.0
"""The norm of a remainder that no longer decays, in units of its stop, below which it is the error of the steps rather than a mode, and the stepping ends.

The steps each leave an error of up to step_atol |X_0|, and over tens of
steps their sum reaches the stop of the remainder (0.1 stop |X_0|). For
lithium on Cu(111) in 2D at 0.05 |G| in 176 bands, the remainder fell to
the stop and then grew (1.9e-8 to 2.3e-8 over 10 time units), the search
for a mode in it failed, and the stepper took it to the end of the span:
13,340 applications of L rather than 300.
"""


@jax.jit
def _step(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    initial: jax.Array,
    measurement: jax.Array,
    times: jax.Array,
    *,
    locked: jax.Array,
    threshold: jax.Array,
    tolerances: tuple[jax.Array, jax.Array],
    dt0: jax.Array,
    horizon: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Step <measurement, exp(L (t - t_0)) initial> from t_0 = times[0] towards times[-1], for at most _CHUNK steps, until the state is below the threshold, or until a step ends past the horizon.

    Get the measurement at the times (non finite after the end), the state
    and time at the end, and the number of applications of L.

    Each complex state is split into its real and imaginary parts, as diffrax
    supports complex states only experimentally.
    """
    rtol, atol = tolerances
    solver = KrylovExponential()
    solution = dfx.diffeqsolve(
        dfx.ODETerm(
            lambda _t, y, _args: _to_real(
                apply_eigen(eigen, _to_complex(y), jnp.asarray(0j))
            )
        ),
        solver,
        t0=times[0],
        t1=times[-1],
        dt0=dt0,
        y0=_to_real(initial),
        saveat=dfx.SaveAt(
            subs=[
                dfx.SubSaveAt(
                    ts=times,
                    fn=lambda _t, y, _args: jnp.vdot(measurement, _to_complex(y)),
                ),
                dfx.SubSaveAt(t1=True),
            ]
        ),
        # The steps need not end on the times, which are found within a step
        # exactly (`KrylovInterpolation`)
        stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
        # A step past the horizon ends the solve rather than the times, which
        # are then found only once, however many solves they span
        event=dfx.Event(
            lambda t, y, _args, **_: (
                (jnp.linalg.norm(_outside(locked, _to_complex(y))) < threshold)
                | (t >= horizon)
            )
        ),
        # Nothing is differentiated
        adjoint=dfx.ForwardMode(),
        max_steps=_CHUNK,
        throw=False,
    )
    measured, end = solution.ys
    applications = solution.stats["num_steps"] * solver.dimension
    return measured, _to_complex(end[0]), solution.ts[1][0], applications


@dataclass(frozen=True, kw_only=True)
class _Stepped:
    """A measurement of exp(L (t - t_0)) X stepped from t_0, by `_step_until`."""

    values: ComplexArray
    """At each time from t_0 to the end, zero after it."""
    state: jax.Array
    """At the end."""
    end: float
    applications: int
    below: bool
    """Whether it ended as the state fell below the threshold."""
    step: float
    """The mean step of the last solve, to start the next."""


def _step_until(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    initial: jax.Array,
    measurement: jax.Array,
    times: FloatArray,
    span: tuple[float, float],
    *,
    threshold: jax.Array,
    tolerances: tuple[jax.Array, jax.Array],
    locked: jax.Array,
    solves: int | None = None,
    dt0: float | None = None,
    until: float | None = None,
) -> _Stepped:
    """Step from span[0] to span[1], until the state off the span of locked is below the threshold, for a number of solves, or until a step ends past until.

    Each solve (`_step`) continues from where the last ended, with its mean
    step (or dt0 for the first), so a long evolution costs the same as one
    solve.
    """
    t_0, t_1 = span
    values = np.zeros(len(times), complex)
    state, start, applications = initial, t_0, 0
    if dt0 is None:
        dt0 = (t_1 - t_0) * 1e-6
    while True:
        measured, state, end, count = _step(
            eigen,
            state,
            measurement,
            jnp.asarray(np.clip(times, start, t_1)),
            locked=locked,
            threshold=threshold,
            tolerances=tolerances,
            # Not weakly typed, as dt0 is from numpy or Python, so _step compiles once
            dt0=jnp.asarray(np.float64(dt0)),
            horizon=jnp.asarray(np.float64(t_1 if until is None else until)),
        )
        end, count = float(end), int(count)
        measured = np.asarray(measured)
        new = (times >= start if start == t_0 else times > start) & (times <= t_1)
        values = np.where(new & np.isfinite(measured), measured, values)
        applications += count
        below = float(jnp.linalg.norm(_outside(locked, state))) < float(threshold)
        if solves is not None:
            solves -= 1
        dt0 = (end - start) * KrylovExponential().dimension / max(count, 1)
        if end >= t_1 or below or solves == 0 or (until is not None and end >= until):
            return _Stepped(
                values=values,
                state=state,
                end=end,
                applications=applications,
                below=below,
                step=dt0,
            )
        start = end


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
    """The slow modes found exactly, in natural units of 1 / time."""
    stopped: float
    """The time after which nothing is left to step."""
    applications: dict[str, int]
    """The applications of L in each stage: probe, modes (Jacobi-Davidson), stepping."""


def get_reduced_evolution(  # ruff:ignore[too-many-arguments, too-many-locals, too-many-statements]
    eigen: EigenLiouvillian,
    initial: jax.Array,
    measure: Callable[[jax.Array], jax.Array],
    times: FloatArray,
    *,
    friction: float,
    config: EvolutionConfig,
    steady_state: jax.Array | None = None,
) -> Evolution:
    """Get measure(exp(L t) initial) at the times, exactly, as its slow modes plus a stepper on the rest.

    L is given in the eigenvectors of K (`get_eigen_liouvillian`), and
    initial in the bands. measure must be linear over the complex numbers.
    The slow modes are found as a partial Schur form L Q = Q T, Q orthonormal
    and T triangular (`find_slow_schur`). Its span is invariant, so the part
    of any state in it, X = Q a, evolves exactly as exp(L t) Q a =
    Q exp(T t) a, at any time. The rest, (1 - Q Q^dagger) X, is stepped by
    the Krylov exponential stepper (`KrylovExponential`) until its norm off
    the span falls below _REMAINDER_STOP config.stop of |X_0|, a few times
    1 / friction rather than the hopping time, and what it has put in the
    span by then is again evolved exactly. The result is exact at every time.

    No coefficient of a mode is needed, where solving for each (as a
    bordered linear solve) cost as many applications of L as finding the
    modes. With the search of `find_slow_schur`, the ISF of hydrogen on
    Cu(111) in 2D takes 1,750 applications rather than 4,123-4,252 at
    0.3 |G|, and 1,328 rather than 3,024-3,083 at 0.05 |G| (4 round-off
    seeds of the steady state, which no longer change the modes found).

    Which modes to find is a question of cost alone, one rule decided
    twice: a mode is found if stepping through it would cost more than
    _COST_FACTOR times the applications of L per mode found so far. X_0 is
    first stepped to _PROBE / friction, which gives the measurement there and
    the applications of L per unit time of stepping. The modes the
    measurement sees are found while the stepping each group of them saves,
    estimated from their Ritz values and weights, is _ESTIMATED times above
    that (`CostCut`). The remainder is then stepped, and checked every
    _CHECK steps: if it decays so slowly that stepping it to the stop would
    cost more than _COST_FACTOR modes, a mode is left in it that the estimate
    missed or left to the measurement, such as the partner of a conjugate
    pair, or a mode the measurement does not see. The mode is found from the
    remainder (blind to the measurement) and added to the Schur form, and the
    stepping goes on.

    At dK = 0 (steady_state, the null vector of L, in the bands) the steady
    state begins the Schur form, so its weight, which the round off of the
    stepping leaves, is evolved exactly rather than stepped.
    """
    start = to_eigen(eigen, initial)
    measurement = get_measurement(eigen, measure, start)
    times = np.asarray(times, dtype=np.float64)
    end = float(times[-1])
    size = float(jnp.linalg.norm(start))
    # The evolution is linear, so the absolute tolerance is relative to |X_0|,
    # as the stop is: with step_atol absolute, an MSD (|X_0| = 5e-3 for
    # hydrogen in 2D) stepped with a Krylov basis of 12 could not step its
    # remainder below _REMAINDER_STOP stop, and found 22 spurious modes in
    # 91,241 applications of L; relative, 1,356
    tolerances = (jnp.asarray(config.step_rtol), jnp.asarray(config.step_atol * size))
    threshold = config.stop * size
    nothing = _padded(jnp.zeros((0, *start.shape), start.dtype))
    early = _step_until(
        eigen,
        start,
        measurement,
        times,
        (0.0, min(_PROBE / friction, end)),
        threshold=jnp.asarray(threshold),
        tolerances=tolerances,
        locked=nothing,
        solves=1,
    )
    probe = early.end
    applications = {"probe": early.applications, "modes": 0, "stepping": 0}
    if probe >= end or early.below:
        # The probe reached the end, or X(t) fell below the stop within it
        return Evolution(
            values=early.values,
            rates=np.zeros(0, complex),
            stopped=probe,
            applications=applications,
        )
    step_rate = early.applications / probe
    prior = (
        None
        if steady_state is None
        else get_schur_of(eigen, to_eigen(eigen, steady_state))
    )
    known = 0 if prior is None else 1
    schur, applications["modes"] = find_slow_schur(
        eigen,
        start,
        measurement,
        CostCut(
            step_rate=step_rate,
            start_time=probe,
            stop=config.stop,
            cost_factor=_ESTIMATED * _COST_FACTOR,
        ),
        config=config,
        seed=early.state,
        prior=prior,
    )

    def per_mode() -> float:
        """Get the applications of L per mode found so far, or before any is found, to step the probe."""
        found = len(schur.vectors) - known
        return applications["modes"] / found if found else early.applications

    slow = np.zeros(len(times), complex)

    def split(x: jax.Array, since: float) -> jax.Array:
        """Add the exact evolution of the part of x in the span of Q after since to slow, and get the rest."""
        nonlocal slow
        q = schur.vectors
        if not len(q):
            return x
        rates, vectors = np.linalg.eig(schur.matrix)
        overlaps = np.asarray(_overlaps(q, x))
        seen = np.asarray(_overlaps(q, measurement)).conj() @ vectors
        weights = np.linalg.solve(vectors, overlaps)
        after = times > since
        growth = np.exp(np.outer(np.where(after, times - since, 0), rates))
        slow += np.where(after, growth @ (seen * weights), 0)
        return x - jnp.tensordot(jnp.asarray(overlaps), q, axes=1)

    remainder, now, dt0 = split(early.state, probe), probe, early.step
    late = np.zeros(len(times), complex)
    norm = float(jnp.linalg.norm(remainder))
    stop = _REMAINDER_STOP * threshold
    searching = True
    while True:
        stepped = _step_until(
            eigen,
            remainder,
            measurement,
            times,
            (now, end),
            threshold=jnp.asarray(stop),
            tolerances=tolerances,
            locked=_padded(schur.vectors),
            dt0=dt0,
            until=now + _CHECK * dt0,
        )
        late = np.where((times > now) & (times <= stepped.end), stepped.values, late)
        applications["stepping"] += stepped.applications
        previous, remainder, now, dt0 = now, stepped.state, stepped.end, stepped.step
        if stepped.below or now >= end:
            break
        last = norm
        norm = float(jnp.linalg.norm(_outside(schur.vectors, remainder)))
        rate = math.log(last / norm) / (now - previous) if norm < last else 0.0
        if rate <= 0 and norm < _FLOOR * stop:
            # What is left no longer decays, at the floor the errors of the
            # steps leave: it is noise, not a mode
            break
        if (
            searching
            and len(schur.vectors) < _MAX_MODES
            and (
                rate <= 0
                or step_rate * math.log(norm / stop) / rate > _COST_FACTOR * per_mode()
            )
        ):
            # The remainder decays too slowly: a mode is left in it
            try:
                schur, cost = _find_left(
                    eigen,
                    schur,
                    remainder,
                    measurement,
                    size=size,
                    config=config,
                )
            except RuntimeError:
                # The stepper takes it
                searching = False
                continue
            applications["modes"] += cost
            remainder = split(remainder, now)
            norm = float(jnp.linalg.norm(remainder))
    split(remainder, now)
    return Evolution(
        values=np.where(times <= probe, early.values, slow + late),
        rates=np.linalg.eigvals(schur.matrix[known:, known:]),
        stopped=now,
        applications=applications,
    )


def _find_left(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    schur: Schur,
    remainder: jax.Array,
    measurement: jax.Array,
    *,
    size: float,
    config: EvolutionConfig,
) -> tuple[Schur, int]:
    """Extend the Schur form by the mode with most weight in what is left of the remainder off its span, whether the measurement sees it or not, and get the applications of L.

    size is |X_0|, against which the weight of the mode is measured.
    """
    q = schur.vectors
    left = _outside(q, remainder) if len(q) else remainder
    return find_slow_schur(
        eigen,
        left,
        measurement,
        ModeCount(n_modes=len(q) + 1),
        config=config,
        seed=left,
        prior=schur if len(q) else None,
        blind=True,
        scale=float(jnp.linalg.norm(left)) / size,
    )
