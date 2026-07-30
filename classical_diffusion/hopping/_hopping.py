import os
from functools import cached_property
from typing import TYPE_CHECKING, Any

os.environ["JAX_ENABLE_X64"] = "True"
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm

from classical_diffusion.hopping._system import CanonicalLattice, Lattice
from classical_diffusion.simulation import SimulationResult
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from classical_diffusion.simulation import TimeSpan


class HoppingSimulationResult[L: Lattice](SimulationResult[L]):
    def __init__(
        self,
        *,
        system: L,
        x_indices: np.ndarray[Any, np.dtype[np.int_]],
        times: np.ndarray[Any, np.dtype[np.floating]],
    ) -> None:
        self._system = system
        self._x_indices = x_indices
        self._times = times

    @cached_property
    def x_points(self) -> np.ndarray:
        return self.system.x_points_from_indices(self._x_indices)


@jax.jit
def _run_hopping_simulation_jit(
    system: CanonicalLattice,
    initial_position: jnp.ndarray,
    sample_times: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Run a hopping simulation and return positions directly at sample times."""
    max_sample_time = sample_times[-1]

    # Carry state: (t_prev, site_prev, t_curr, site_curr, rng_key)
    init_state = (
        jnp.array(0.0, dtype=sample_times.dtype),
        initial_position,
        jnp.array(0.0, dtype=sample_times.dtype),
        initial_position,
        key,
    )

    def scan_body(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array],
        target_time: jnp.ndarray,
    ) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array],
        jnp.ndarray,
    ]:
        def inner_condition(
            state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array],
        ) -> jnp.ndarray:
            _, _, current_t, _, _ = state
            return (current_t <= target_time) & (current_t < max_sample_time)

        def inner_body(
            state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array]:
            _, _, current_t, current_site, rng_key = state
            destination_key, dt_key, next_key = jax.random.split(rng_key, 3)

            hop_sites, hop_rates = system.get_rates(current_site)
            total_rate = jnp.sum(hop_rates)
            dt = (
                -jnp.log(jax.random.uniform(dt_key, dtype=sample_times.dtype))
                / total_rate
            )
            next_site = jax.random.choice(
                destination_key, hop_sites, p=hop_rates / total_rate
            )

            return (current_t, current_site, current_t + dt, next_site, next_key)

        # Take as many steps as needed to reach the target time, but stop if we exceed the last requested sample time.
        # Note if we already exceeded the target time, this will return the incoming carry state unchanged.
        final_state = jax.lax.while_loop(inner_condition, inner_body, carry)

        # final_state[1] is previous_site, which is the last site visited before exceeding the target time.
        return final_state, final_state[1]

    # On a gpu, if the number of samples >> number of hops, it will be faster to collect
    # all hops (possibly in a batched manner) and then use search sorted to find all sample
    # positions in parallel. If the number of hops >> number of samples, there will be no
    # difference. Here we use an approach which is optimal on the cpu, and significantly
    # easier to implement.
    _, sample_positions = jax.lax.scan(scan_body, init_state, sample_times)
    return sample_positions


@timed
def solve_ensemble[L: Lattice = Lattice](
    system: L,
    time_span: TimeSpan,
    initial_condition: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    key: jax.Array,
) -> HoppingSimulationResult[L]:
    """Solve the hopping ensemble."""
    keys = jax.random.split(key, initial_condition.shape[0])
    times = jnp.linspace(time_span.t_start, time_span.t_end, time_span.n_steps)

    results = jax.vmap(
        _run_hopping_simulation_jit,
        in_axes=(None, 0, None, 0),
    )(system.as_canonical(), initial_condition, times, keys)

    return HoppingSimulationResult[L](
        system=system,
        times=np.array(times),
        x_indices=np.array(jnp.transpose(results, (0, 2, 1))),
    )


def _get_deterministic_isf_slow[L: Lattice[Any]](
    system: L,
    finite_lattice_shape: tuple,
    time_span: TimeSpan,
    delta_k: float,
    initial_position: jnp.ndarray,
) -> None:  # tuple[jnp.ndarray, jnp.ndarray]:
    """Use deterministic formula to return the ISF, inefficiently."""
    #
    # Rate matrix, M
    # M[a,b] = - rate (b -> a)
    # M[a,a] = sum_i ( rates a -> i)

    max_lattice_index = jnp.prod(jnp.array(finite_lattice_shape))

    times = jnp.linspace(time_span.t_start, time_span.t_end, time_span.n_steps)
    initial_p = jnp.full(max_lattice_index, 0.0)
    initial_p = initial_p.at[
        jnp.ravel_multi_index(tuple(initial_position), finite_lattice_shape)
    ].set(1)

    rate_matrix = jnp.full((max_lattice_index, max_lattice_index), 0.0)

    for site in range(max_lattice_index):
        hop_sites, hop_rates = system.get_rates(
            jnp.unravel_index(site, finite_lattice_shape)
        )
        hop_sites = jnp.clip(
            hop_sites, min=0
        )  # Remove negative indices as these will wrap around when forming the matrix
        rate_row = jnp.full(max_lattice_index, 0.0)

        rate_row = rate_row.at[hop_sites[:, 0]].set(hop_rates)
        rate_row = rate_row.at[site].set(-jnp.sum(hop_rates))
        rate_matrix = rate_matrix.at[site].set(rate_row)

    # Find probabilities at a given time by solving DE: P(t) = exp(Mt) P(0)

    def solve_single_time(time: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(expm(rate_matrix * time), initial_p)

    p_at_each_time = jax.vmap(solve_single_time)(times)
    p_at_each_time = jnp.clip(p_at_each_time, min=0)
    p_at_each_time /= jnp.sum(p_at_each_time, axis=-1, keepdims=True)
