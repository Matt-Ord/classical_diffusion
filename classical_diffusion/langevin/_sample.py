from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import sympy as sp

if TYPE_CHECKING:
    from classical_diffusion.langevin import CanonicalSystem


SAMPLE_REGION = 100


@jax.jit
def _sample_initial_conditions(
    system: CanonicalSystem,
    n_samples: int = 1,
    minimum_energy: float = 0.0,
    *,
    _key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    potential_fn = sp.lambdify(
        (*system.coordinate_symbols, *system.parameter_symbols),
        system.potential_expr,
        "jax",
    )

    # TODO: this is left to later
    v_min = 0
    p_std = jnp.sqrt(system.kbt * system.m)

    def _sample_one(key: jax.Array) -> tuple:
        def _cond_fn(state: tuple) -> bool:
            _, _, _, accepted = state
            return ~accepted

        def _body_fn(state: tuple) -> tuple:
            key, _x, _p, _ = state
            key, kx, ku, kp = jax.random.split(key, 4)

            # Draw position candidate from full continuous space q(x) ~ N(0, SAMPLE_REGION^2 * I)
            x_candidate = jax.random.normal(kx, shape=(system.n_dim,)) * SAMPLE_REGION
            v = potential_fn(*x_candidate, *system.params)

            p_candidate = jax.random.normal(kp, (system.n_dim,)) * p_std
            energy = jnp.sum(p_candidate**2) / (2 * system.m) + v

            x_ok = jax.random.uniform(ku) < jnp.exp(-(v - v_min) / system.kbt)
            accept = x_ok & (energy > minimum_energy)

            return key, x_candidate, p_candidate, accept

        init = (key, jnp.zeros(system.n_dim), jnp.zeros(system.n_dim), jnp.array(False))  # ruff: ignore[boolean-positional-value-in-call]
        _, x_finals, p_finals, _ = jax.lax.while_loop(_cond_fn, _body_fn, init)

        return x_finals, p_finals

    keys = jax.random.split(_key, n_samples)
    x_points, p_points = jax.vmap(_sample_one)(keys)
    return x_points, p_points


def get_initial_conditions(
    system: CanonicalSystem, n_samples: int, *, _key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    x_points, p_points = _sample_initial_conditions(system, n_samples, _key=_key)
    x_points = x_points.reshape(-1, system.n_dim)
    p_points = p_points.reshape(-1, system.n_dim)
    return x_points, p_points


def get_over_barrier_initial_conditions(
    system: CanonicalSystem,
    barrier_energy: float,
    n_samples: int,
    *,
    _key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    x_points, p_points = _sample_initial_conditions(
        system, n_samples, minimum_energy=barrier_energy, _key=_key
    )
    x_points = x_points.reshape(-1, system.n_dim)
    p_points = p_points.reshape(-1, system.n_dim)
    return x_points, p_points
