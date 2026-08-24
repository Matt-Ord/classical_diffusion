from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from classical_diffusion.util import _get_key

if TYPE_CHECKING:
    from classical_diffusion.langevin import CanonicalSystem, System


SAMPLE_REGION = 10


@jax.jit(static_argnames=("n_samples",))
def _sample_initial_conditions(
    system: "CanonicalSystem",  # ruff: ignore[quoted-annotation]
    n_samples: int = 1,
    *,
    minimum_energy: float = 0.0,
    _key: jax.Array,
) -> tuple[jax.Array, jax.Array]:

    potential_fn = sp.lambdify(
        system.lambda_symbols,
        system.potential_expr,
        modules=[{"DerivativeSafeMod": jnp.mod}, "jax"],
    )

    # TODO: the assumption here is that the minimum of the potential  # ruff: ignore[line-contains-todo]
    # is at 0, which is not always true. Finding the minimum of the
    # potential can be done in sympy automatically.
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
    return jax.vmap(_sample_one)(keys)


def get_random_initial_conditions(
    system: System,
    n_samples: int,
    *,
    minimum_energy: float = 0.0,
    _key: jax.Array | None = None,
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
]:
    """Get random initial conditions for a given system."""
    _key = _get_key(_key)
    normalized_system = system.with_normalized_units().as_canonical()
    x_points, p_points = _sample_initial_conditions(
        normalized_system,
        n_samples,
        minimum_energy=system.units.energy_into(
            minimum_energy, normalized_system.units
        ),
        _key=_key,
    )
    x_points = np.array(x_points.reshape(-1, system.n_dim))
    p_points = np.array(p_points.reshape(-1, system.n_dim))
    return (
        normalized_system.units.length_into(x_points, system.units),
        normalized_system.units.momentum_into(p_points, system.units),
    )
