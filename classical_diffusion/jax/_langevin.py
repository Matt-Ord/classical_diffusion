from typing import TYPE_CHECKING

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import sympy as sp

from classical_diffusion.simulation import (
    TimeSpan,  # ruff: ignore[typing-only-first-party-import]
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.langevin import CanonicalSystem, System


def get_force_fn(
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
def _run_overdamped_ensemble_jit(
    system: "CanonicalSystem",  # ruff:ignore[quoted-annotation]
    xs0: jnp.ndarray,
    keys: jax.Array,
    times: jnp.ndarray,
) -> jnp.ndarray:
    gamma = jnp.broadcast_to(system.gamma, (system.n_dim,))
    force_fn = get_force_fn(system)
    diffusion_matrix = jnp.diag(jnp.sqrt(2.0 * system.kbt / gamma))

    def solve_one(x0: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        bm = dfx.VirtualBrownianTree(
            t0=0,
            t1=times[-1],
            tol=1e-4,
            shape=(system.n_dim,),
            key=key,
            levy_area=dfx.SpaceTimeTimeLevyArea,
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
            dt0=times[1] - times[0],
            y0=x0,
            args=None,
            stepsize_controller=dfx.ClipStepSizeController(
                dfx.PIDController(
                    rtol=1e-2,  # cspell: disable-line
                    atol=1e-3,
                ),
                step_ts=times,
            ),
            saveat=dfx.SaveAt(ts=times),
            max_steps=100_000_000,
        )
        return sol.ys

    return jax.vmap(solve_one, in_axes=(0, 0))(xs0, keys)


@eqx.filter_jit
def solve_overdamped_ensemble_jax[S: CanonicalSystem](
    system: S,
    time_span: TimeSpan,
    initial_conditions: tuple[jnp.ndarray, jnp.ndarray],
    _key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve an ensemble of overdamped Langevin trajectories in parallel via jax.vmap."""
    n_run = initial_conditions[0].shape[0]

    times = jnp.linspace(
        time_span.t_start, time_span.t_end, time_span.n_steps + 1, endpoint=True
    )

    keys = jax.random.split(_key, n_run)

    xs_batch = _run_overdamped_ensemble_jit(system, initial_conditions[0], keys, times)

    xs_batch = jnp.transpose(xs_batch, (0, 2, 1))

    return (times, xs_batch)
