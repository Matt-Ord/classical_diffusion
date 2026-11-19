"""The time evolution of a measurement of exp(L t) X_0, as its slow modes exactly plus a Krylov exponential stepper on the rest."""
# cspell: ignore vdot

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
from classical_diffusion.caldeira_leggett._slow_modes import CostCut, find_slow_modes

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.caldeira_leggett._config import EvolutionConfig
    from classical_diffusion.caldeira_leggett._model import ComplexArray, FloatArray


def _to_real(x: jax.Array) -> jax.Array:
    return jnp.stack([x.real, x.imag])


def _to_complex(y: jax.Array) -> jax.Array:
    return y[0] + 1j * y[1]


_CHUNK = 250
"""The most steps of one solve. A longer evolution continues in another, from where it stopped, and the probe ends after one."""


@jax.jit
def _step(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    initial: jax.Array,
    measurement: jax.Array,
    times: jax.Array,
    *,
    threshold: jax.Array,
    tolerances: tuple[jax.Array, jax.Array],
    dt0: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Step <measurement, exp(L (t - t_0)) initial> from t_0 = times[0] towards times[-1], for at most _CHUNK steps, or until the state is below the threshold.

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
        event=dfx.Event(lambda _t, y, _args, **_: jnp.linalg.norm(y) < threshold),
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


def _step_until(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    initial: jax.Array,
    measurement: jax.Array,
    times: FloatArray,
    span: tuple[float, float],
    *,
    threshold: jax.Array,
    tolerances: tuple[jax.Array, jax.Array],
    once: bool = False,
) -> _Stepped:
    """Step from span[0] to span[1], until the state is below the threshold, or for one solve only (once).

    Each solve (`_step`) continues from where the last ended, with its mean
    step, so a long evolution costs the same as one solve.
    """
    t_0, t_1 = span
    values = np.zeros(len(times), complex)
    state, start, applications = initial, t_0, 0
    dt0 = (t_1 - t_0) * 1e-6
    while True:
        measured, state, end, count = _step(
            eigen,
            state,
            measurement,
            jnp.asarray(np.clip(times, start, t_1)),
            threshold=threshold,
            tolerances=tolerances,
            dt0=jnp.asarray(dt0),
        )
        end, count = float(end), int(count)
        measured = np.asarray(measured)
        new = (times >= start if start == t_0 else times > start) & (times <= t_1)
        values = np.where(new & np.isfinite(measured), measured, values)
        applications += count
        below = float(jnp.linalg.norm(state)) < float(threshold)
        if end >= t_1 or below or once:
            return _Stepped(
                values=values,
                state=state,
                end=end,
                applications=applications,
                below=below,
            )
        dt0 = (end - start) * KrylovExponential().dimension / max(count, 1)
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
    """The applications of L in each stage: probe, modes (Jacobi-Davidson), coefficients, stepping."""


def get_reduced_evolution(  # ruff:ignore[too-many-arguments, too-many-locals]
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
    With the slow modes (lambda_j, r_j) of L, and the weight c_j of each in
    X_0 (`find_slow_modes`),

        measure(exp(L t) X_0) = sum_j measure(r_j) c_j exp(lambda_j t)
            + measure(exp(L t) (X_0 - sum_j c_j r_j)).

    The first term is exact at any time. What is left has no weight on the
    slow modes, so it decays at the rate of the next mode, and the Krylov
    exponential stepper (`KrylovExponential`) only has to follow it until its
    norm falls below config.stop of |X_0|: a few times 1 / friction, rather
    than the hopping time. The result is exact at every time.

    Which modes to find is a question of cost alone. X_0 is first stepped to
    config.probe / friction, which gives the measurement there and the
    applications of L per unit time of stepping. A mode is found if that
    costs fewer applications than stepping through it saves (`CostCut`): for
    hydrogen and lithium on Cu(111) and Cu(110), within a few percent of the
    best number of modes, or better, as the stepping continues from the
    probe. The hopping modes are always found, in-well vibrations only where
    the stepper is costly. Against the rational Krylov reduced model which
    this replaces, it needs 3.8 to 4.9 times fewer applications of L in 2D.

    At dK = 0 (steady_state, the null vector of L, in the bands) X_0 has no
    trace, but the round off of the modes leaves a little weight on the
    steady state, which never decays. It is removed exactly, and its constant
    measurement added back.
    """
    start = to_eigen(eigen, initial)
    measurement = get_measurement(eigen, measure, start)
    times = np.asarray(times, dtype=np.float64)
    tolerances = (jnp.asarray(config.step_rtol), jnp.asarray(config.step_atol))
    threshold = jnp.asarray(config.stop * float(jnp.linalg.norm(start)))
    early = _step_until(
        eigen,
        start,
        measurement,
        times,
        (0.0, min(config.probe / friction, float(times[-1]))),
        threshold=threshold,
        tolerances=tolerances,
        once=True,
    )
    probe, state = early.end, early.state
    applications = {
        "probe": early.applications,
        "modes": 0,
        "coefficients": 0,
        "stepping": 0,
    }
    if probe >= times[-1] or early.below:
        # The probe reached the end, or X(t) fell below the stop within it
        return Evolution(
            values=early.values,
            rates=np.zeros(0, complex),
            stopped=probe,
            applications=applications,
        )
    modes, counts = find_slow_modes(
        eigen,
        start,
        measurement,
        CostCut(
            step_rate=applications["probe"] / probe,
            start_time=probe,
            stop=config.stop,
            cost_factor=config.cost_factor,
        ),
        config=config,
    )
    applications.update(counts)
    # exp(L t) X_0 = sum_j c_j exp(lambda_j t) r_j + exp(L (t - probe)) R, with
    # R = X(probe) - sum_j c_j exp(lambda_j probe) r_j
    remainder = state - sum(
        (m.coefficient * np.exp(m.rate * probe) * m.vector for m in modes),
        start=jnp.zeros_like(start),
    )
    constant = 0.0j
    if steady_state is not None:
        steady = to_eigen(eigen, steady_state)
        weight = jnp.sum(eigen.trace * remainder) / jnp.sum(eigen.trace * steady)
        remainder -= weight * steady
        constant = complex(weight * jnp.vdot(measurement, steady))
    late = _step_until(
        eigen,
        remainder,
        measurement,
        times,
        (probe, float(times[-1])),
        threshold=threshold,
        tolerances=tolerances,
    )
    slow = sum(
        (m.amplitude * np.exp(m.rate * times) for m in modes),
        start=np.zeros(len(times), complex),
    )
    values = np.where(times <= probe, early.values, slow + late.values + constant)
    applications["stepping"] = late.applications
    return Evolution(
        values=values,
        rates=np.array([m.rate for m in modes]),
        stopped=late.end,
        applications=applications,
    )
