from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.langevin import System


def make_free_point_sampler(system: System, barrier_energy: float) -> Callable:
    potential_fn = sp.lambdify(
        (*system.coordinate_symbols, *system.parameter_symbols),
        system.potential_expr,
        "jax",
    )
    bounds = system.sampling_domain

    lo = jnp.array([b[0] for b in bounds])
    hi = jnp.array([b[1] for b in bounds])
    lattice = jnp.asarray(system.lattice_vectors)

    grids = jnp.meshgrid(*[jnp.linspace(b[0], b[1], 50) for b in bounds], indexing="ij")
    frac_grid = jnp.stack([g.ravel() for g in grids], axis=0)
    cart_grid = lattice @ frac_grid
    v_grid = potential_fn(*cart_grid, *system.params)
    pdf_max = jnp.exp(-jnp.min(v_grid) / system.kbt)

    def _sample_one(key: jax.Array) -> tuple:
        def _cond_fn(state: tuple) -> bool:
            _, _, _, accepted = state
            return ~accepted

        def _body_fn(state: tuple) -> tuple:
            key, _x, _p, _ = state
            key, kx, ku, kp = jax.random.split(key, 4)

            frac_candidates = jax.random.uniform(
                kx, minval=lo, maxval=hi, shape=(system.n_dim,)
            )
            x_candidate = lattice @ frac_candidates
            v = potential_fn(*x_candidate, *system.params)
            x_ok = jax.random.uniform(ku) < jnp.exp(-v / system.kbt) / pdf_max

            p_std = jnp.sqrt(system.kbt * system.m)
            p_candidate = jax.random.normal(kp, (system.n_dim,)) * p_std
            energy = jnp.sum(p_candidate**2) / (2 * system.m) + v

            accept = x_ok & (energy > barrier_energy)
            return key, x_candidate, p_candidate, accept

        init = (key, jnp.zeros(system.n_dim), jnp.zeros(system.n_dim), jnp.array(False))  # ruff:ignore[boolean-positional-value-in-call]
        _, x_final, p_final, _ = jax.lax.while_loop(_cond_fn, _body_fn, init)
        return x_final, p_final

    return jax.jit(jax.vmap(_sample_one))


def make_initial_conditions_sampler(system: System, n_grid: int = 60) -> Callable:
    potential_fn = sp.lambdify(
        (*system.coordinate_symbols, *system.parameter_symbols),
        system.potential_expr,
        "jax",
    )

    bounds = system.sampling_domain

    lo = jnp.array([b[0] for b in bounds])
    hi = jnp.array([b[1] for b in bounds])

    lattice = jnp.asarray(system.lattice_vectors)

    grids = jnp.meshgrid(
        *[jnp.linspace(b[0], b[1], n_grid) for b in bounds], indexing="ij"
    )
    frac_grid = jnp.stack([g.ravel() for g in grids], axis=0)
    cart_grid = lattice @ frac_grid
    v_grid = potential_fn(*cart_grid, *system.params)
    pdf_max = jnp.exp(-jnp.min(v_grid) / system.kbt)

    def _sample_one(key: jax.Array) -> tuple:
        def _cond_fn(state: tuple) -> bool:
            _, _, accepted = state
            return ~accepted

        def _body_fn(state: tuple) -> tuple:
            key, _x, _ = state
            key, kx, ku = jax.random.split(key, 3)
            frac_candidates = jax.random.uniform(
                kx, minval=lo, maxval=hi, shape=(system.n_dim,)
            )
            x_candidate = lattice @ frac_candidates
            v = potential_fn(*x_candidate, *system.params)
            accept = jax.random.uniform(ku) < jnp.exp(-v / system.kbt) / pdf_max

            return key, x_candidate, accept

        key, p_key = jax.random.split(key, 2)
        init = (key, jnp.zeros(system.n_dim), jnp.array(False))
        _, x_finals, _ = jax.lax.while_loop(_cond_fn, _body_fn, init)
        p_std = jnp.sqrt(system.kbt * system.m)
        p_finals = jax.random.normal(p_key, (system.n_dim,)) * p_std

        return x_finals, p_finals

    return jax.jit(jax.vmap(_sample_one))
