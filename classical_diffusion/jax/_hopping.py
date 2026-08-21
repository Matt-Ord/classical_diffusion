from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import Tsit5  # cspell: disable-line

from classical_diffusion.hopping._system import CanonicalLattice, Lattice
from classical_diffusion.simulation import (
    TimeSpan,  # ruff: ignore[typing-only-first-party-import]
)


@jax.jit
def _get_deterministic_probabilities_jit[L: Lattice](
    initial_p: jnp.ndarray,
    times: jnp.ndarray,
    hop_sites: jnp.ndarray,
    hop_rates: jnp.ndarray,
) -> jnp.ndarray:
    """Use deterministic formula to return the ISF, inefficiently."""
    total_outgoing_rates = jnp.sum(hop_rates, axis=-1)

    def vector_field(
        _t: Any,  # ruff:ignore[any-type]
        p: jnp.ndarray,
        _args: Any,  # ruff:ignore[any-type]
    ) -> jnp.ndarray:
        return jnp.sum(hop_rates * p[hop_sites], axis=-1) - p * total_outgoing_rates

    return dfx.diffeqsolve(
        terms=dfx.ODETerm(vector_field),
        solver=Tsit5(),  # cspell: disable-line
        t0=0,
        t1=times[-1],
        dt0=times[1] - times[0],
        y0=initial_p,
        args=None,
        saveat=dfx.SaveAt(ts=times),
        stepsize_controller=dfx.PIDController(
            rtol=1e-6,  # cspell: disable-line
            atol=1e-8,
        ),
        max_steps=100_000_000,
    ).ys


@eqx.filter_jit
def get_deterministic_probabilities_jax[L: CanonicalLattice](
    system: L,
    time_span: TimeSpan,
    max_lattice_index: int = 1000,
) -> jnp.ndarray:
    """Generate the Lattice then use a deterministic PDE to find the probabilities at all times."""
    initial_position = jnp.array(max_lattice_index / 2, int)

    times = jnp.linspace(time_span.t_start, time_span.t_end, time_span.n_steps + 1)

    initial_p = jnp.full(
        max_lattice_index, 0.0, dtype=jnp.float32
    )  # jnp.full(max_lattice_shape, 0.0)
    initial_p = initial_p.at[initial_position].set(1)

    hop_sites, hop_rates = system.get_rates(
        jnp.arange(max_lattice_index)
    )  # max_lattice_index = jnp.prod(max_lattice_shape)

    return _get_deterministic_probabilities_jit(initial_p, times, hop_sites, hop_rates)
