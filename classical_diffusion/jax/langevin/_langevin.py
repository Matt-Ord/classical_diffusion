from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.langevin import CanonicalSystem, System
    from classical_diffusion.simulation import (
        TimeSpan,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Adaptive:
    """Adaptive time steps, chosen by a PID controller.

    This is the default, and works for any system. The noise is sampled from
    a virtual Brownian tree, so that it is consistent when a step is rejected.
    """

    rtol: float = 1e-2
    atol: float = 1e-3


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FixedStep:
    """Fixed time steps of step_size, in the natural units of the simulation.

    The natural unit of time is 1 / nu_0, where nu_0 is the larger of the
    friction and the force rate F_max / sqrt(m k_B T). The step is reduced so
    that it divides the spacing of the output times. Since no step is rejected,
    the noise is sampled independently for each step, which is much faster
    than the Brownian tree needed for adaptive steps. The step must be small
    enough to resolve the fastest motion of the system.
    """

    step_size: float


type StepSize = Adaptive | FixedStep

DEFAULT_STEP_SIZE = Adaptive()
"""The default step size, which works for any system."""


def _get_step_size_config(
    step_size: StepSize,
    times: jnp.ndarray,
    *,
    shape: tuple[int, ...],
    key: jax.Array,
    levy_area: type[
        dfx.BrownianIncrement | dfx.SpaceTimeLevyArea | dfx.SpaceTimeTimeLevyArea
    ],
) -> tuple[dfx.AbstractBrownianPath, dfx.AbstractStepSizeController, Any, Any]:
    """Get the Brownian path, step size controller, adjoint and initial step."""
    spacing = times[1] - times[0]
    if isinstance(step_size, FixedStep):
        dt0 = spacing / jnp.ceil(spacing / step_size.step_size)
        # Steps are never rejected, so the increments need not be consistent
        bm = dfx.UnsafeBrownianPath(shape=shape, key=key, levy_area=levy_area)
        return bm, dfx.ConstantStepSize(), dfx.ForwardMode(), dt0
    bm = dfx.VirtualBrownianTree(
        t0=0, t1=times[-1], tol=1e-4, shape=shape, key=key, levy_area=levy_area
    )
    controller = dfx.ClipStepSizeController(
        dfx.PIDController(rtol=step_size.rtol, atol=step_size.atol), step_ts=times
    )
    return bm, controller, dfx.RecursiveCheckpointAdjoint(), spacing


def _get_force_fn(
    system: System,
) -> Callable[[jnp.ndarray, tuple[float, ...]], jnp.ndarray]:
    """Compute a callable force function, taking and returning an array."""
    raw_fn = sp.lambdify(
        system.lambda_symbols,
        system.force_expr,
        modules=[{"DerivativeSafeMod": jnp.mod}, "jax"],
    )
    return lambda x_array, params: jnp.array(raw_fn(*x_array, *params))


@jax.jit
def _run_deterministic_ensemble_jit(
    system: "CanonicalSystem",  # ruff:ignore[quoted-annotation]
    xs0: jnp.ndarray,
    ps0: jnp.ndarray,
    times: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    force_fn = _get_force_fn(system)

    def vector_field(
        _t: Any,  # ruff: ignore[any-type]
        y: tuple[jnp.ndarray, jnp.ndarray],
        _args: Any,  # ruff: ignore[any-type]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x, p = y
        return (p / system.m, force_fn(x, system.params))

    term = dfx.ODETerm(vector_field)

    forward_times = jnp.maximum(times, 0.0)
    backward_times = jnp.minimum(times[::-1], 0.0)
    is_positive = times >= 0.0
    dt0 = jnp.abs(times[1] - times[0])

    def solve_one(x0: jnp.ndarray, p0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        sol_fwd = dfx.diffeqsolve(
            term,
            solver=dfx.Tsit5(),  # cspell: disable-line
            t0=0.0,
            t1=jnp.maximum(0.0, times[-1]),
            dt0=dt0,
            y0=(x0, p0),
            saveat=dfx.SaveAt(ts=forward_times),
            stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-8),
            max_steps=None,
        )

        sol_bwd = dfx.diffeqsolve(
            term,
            solver=dfx.Tsit5(),  # cspell: disable-line
            t0=0.0,
            t1=jnp.minimum(0.0, times[0]),
            dt0=-dt0,
            y0=(x0, p0),
            saveat=dfx.SaveAt(ts=backward_times),
            stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-8),
            max_steps=None,
        )

        # Broadcast mask across spatial/momentum dimensions
        mask_x = jnp.reshape(is_positive, (-1,) + (1,) * x0.ndim)
        x_out = jnp.where(mask_x, sol_fwd.ys[0], sol_bwd.ys[0][::-1])
        p_out = jnp.where(mask_x, sol_fwd.ys[1], sol_bwd.ys[1][::-1])

        return x_out, p_out

    return jax.vmap(solve_one, in_axes=(0, 0))(xs0, ps0)


@jax.jit
def _run_deterministic_ensemble_jit_forward(
    system: "CanonicalSystem",  # ruff:ignore[quoted-annotation]
    xs0: jnp.ndarray,
    ps0: jnp.ndarray,
    times: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    force_fn = _get_force_fn(system)

    def vector_field(
        _t: Any,  # ruff: ignore[any-type]
        y: tuple[jnp.ndarray, jnp.ndarray],
        _args: Any,  # ruff: ignore[any-type]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x, p = y
        return (p / system.m, force_fn(x, system.params))

    term = dfx.ODETerm(vector_field)

    def solve_one(x0: jnp.ndarray, p0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return dfx.diffeqsolve(
            term,
            solver=dfx.Tsit5(),  # cspell: disable-line
            t0=0.0,
            t1=jnp.maximum(0.0, times[-1]),
            dt0=jnp.abs(times[1] - times[0]),
            y0=(x0, p0),
            saveat=dfx.SaveAt(ts=times),
            stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-8),
            max_steps=None,
        ).ys

    return jax.vmap(solve_one, in_axes=(0, 0))(xs0, ps0)


@jax.jit
def _run_langevin_ensemble_jit(  # ruff: ignore[too-many-arguments]
    system: "CanonicalSystem",  # ruff:ignore[quoted-annotation]
    xs0: jnp.ndarray,
    ps0: jnp.ndarray,
    keys: jax.Array,
    times: jnp.ndarray,
    *,
    step_size: StepSize,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    gamma = jnp.broadcast_to(system.gamma / system.m, (system.n_dim,))
    u = jnp.broadcast_to(system.kbt / system.m, (system.n_dim,))
    force_fn = _get_force_fn(system)

    def grad_f(x: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return -force_fn(x, system.params) / system.kbt

    def solve_one(
        x0: jnp.ndarray, p0: jnp.ndarray, key: jax.Array
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        bm, controller, adjoint, dt0 = _get_step_size_config(
            step_size,
            times,
            shape=(system.n_dim,),
            key=key,
            levy_area=dfx.SpaceTimeTimeLevyArea,
        )

        drift_term = dfx.UnderdampedLangevinDriftTerm(gamma, u, grad_f)
        diffusion_term = dfx.UnderdampedLangevinDiffusionTerm(gamma, u, bm)
        terms = dfx.MultiTerm(drift_term, diffusion_term)

        sol = dfx.diffeqsolve(
            terms,
            solver=dfx.ALIGN(),
            t0=0,
            t1=times[-1],
            dt0=dt0,
            y0=(x0, p0),
            args=None,
            stepsize_controller=controller,
            saveat=dfx.SaveAt(ts=times),
            max_steps=100_000_000,
            adjoint=adjoint,
        )
        return sol.ys

    x_out, v_out = jax.vmap(solve_one, in_axes=(0, 0, 0))(xs0, ps0 / system.m, keys)
    return x_out, v_out * system.m


def solve_many[S: CanonicalSystem](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[jnp.ndarray, jnp.ndarray],
    *,
    step_size: StepSize = DEFAULT_STEP_SIZE,
    _key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve an ensemble of ULD Langevin trajectories in parallel via jax.vmap.

    The step size is only used when the friction is non-zero.
    """
    times = jnp.linspace(
        time_span.t_start, time_span.t_end, time_span.n_steps + 1, endpoint=True
    )

    n_run = initial_conditions[0].shape[0]

    if np.isclose(system.gamma, 0.0):
        if times[0] < 0.0:
            xs_batch, ps_batch = _run_deterministic_ensemble_jit(
                system, initial_conditions[0], initial_conditions[1], times
            )
        else:
            xs_batch, ps_batch = _run_deterministic_ensemble_jit_forward(
                system, initial_conditions[0], initial_conditions[1], times
            )
    else:
        keys = jax.random.split(_key, n_run)

        xs_batch, ps_batch = _run_langevin_ensemble_jit(
            system,
            initial_conditions[0],
            initial_conditions[1],
            keys,
            times,
            step_size=step_size,
        )

    # Diffrax + vmap naturally outputs: (n_run, n_time, n_dim)
    # We transpose axes 1 and 2 to match your target layout: (n_run, n_dim, n_time)
    xs_batch = jnp.transpose(xs_batch, (0, 2, 1))
    ps_batch = jnp.transpose(ps_batch, (0, 2, 1))
    return times, xs_batch, ps_batch


@jax.jit
def _run_many_overdamped_jit(
    system: "CanonicalSystem",  # ruff:ignore[quoted-annotation]
    xs0: jnp.ndarray,
    keys: jax.Array,
    times: jnp.ndarray,
    step_size: StepSize,
) -> jnp.ndarray:
    gamma = jnp.broadcast_to(system.gamma, (system.n_dim,))
    force_fn = _get_force_fn(system)
    diffusion_matrix = jnp.diag(jnp.sqrt(2.0 * system.kbt / gamma))

    def solve_one(x0: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        bm, controller, adjoint, dt0 = _get_step_size_config(
            step_size,
            times,
            shape=(system.n_dim,),
            key=key,
            levy_area=dfx.SpaceTimeLevyArea,
        )

        # dx = (F(x) / gamma) dt + sqrt(2 kB T / gamma) dW
        drift_term = dfx.ODETerm(
            lambda _t, x, _args: force_fn(x, system.params) / gamma
        )
        diffusion_term = dfx.ControlTerm(lambda _t, _x, _args: diffusion_matrix, bm)
        terms = dfx.MultiTerm(drift_term, diffusion_term)

        sol = dfx.diffeqsolve(
            terms,
            solver=dfx.ShARK(),
            t0=0,
            t1=times[-1],
            dt0=dt0,
            y0=x0,
            args=None,
            stepsize_controller=controller,
            saveat=dfx.SaveAt(ts=times),
            max_steps=None,
            adjoint=adjoint,
        )
        return sol.ys

    return jax.vmap(solve_one, in_axes=(0, 0))(xs0, keys)


def solve_many_overdamped[S: CanonicalSystem](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[jnp.ndarray, jnp.ndarray],
    *,
    step_size: StepSize = DEFAULT_STEP_SIZE,
    _key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve an ensemble of overdamped Langevin trajectories in parallel via jax.vmap."""
    n_run = initial_conditions[0].shape[0]

    times = jnp.linspace(
        time_span.t_start, time_span.t_end, time_span.n_steps + 1, endpoint=True
    )

    keys = jax.random.split(_key, n_run)
    xs_batch = _run_many_overdamped_jit(
        system.as_canonical(), initial_conditions[0], keys, times, step_size
    )

    xs_batch = jnp.transpose(xs_batch, (0, 2, 1))

    return times, xs_batch, jnp.zeros_like(xs_batch)
