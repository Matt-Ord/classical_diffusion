from functools import cached_property
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from classical_diffusion.hopping._system import Lattice
from classical_diffusion.simulation import SimulationResult
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from classical_diffusion.simulation import TimeSpan


class HoppingSimulationResult[L: Lattice[Any]](SimulationResult[L]):
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
        return self.system.transform_indices_to_coordinates(self._x_indices)


@jax.jit
def _run_hopping_simulation_jit(
    lattice: Lattice,
    initial_position: jnp.ndarray,
    sample_times: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Run a hopping simulation and return positions only at specified sample times."""
    n_times = sample_times.shape[0]
    n_dimensions = initial_position.shape[0]

    # The positions sampled at sample_times
    sample_positions = jnp.zeros((n_times, n_dimensions), dtype=initial_position.dtype)

    # Outer loop state: (sample_idx, current_time, current_position, sample_positions, rng_key)
    init_outer_state = (0, jnp.float64(0.0), initial_position, sample_positions, key)

    def outer_condition(
        state: tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array],
    ) -> bool:
        sample_idx, _, _, _, _ = state
        return sample_idx < n_times

    def outer_body(
        state: tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array],
    ) -> tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array]:
        sample_idx, current_time, current_site, sample_positions, rng_key = state

        # Split key for destination, exponential time step, and next loop state
        destination_key, dt_key, next_key = jax.random.split(rng_key, 3)

        # The total rate out of current position is sum of individual rates
        hop_sites, hop_rates = lattice.get_rates(current_site)
        total_rate = jnp.sum(hop_rates)
        next_time = current_time - jnp.log(jax.random.uniform(dt_key)) / total_rate
        next_site = jax.random.choice(
            destination_key, hop_sites, p=hop_rates / total_rate
        )

        # For each sample time point that has passed in stochastic time till this jump occurred,
        # update the positions array to show the location stayed the same
        def inner_condition(inner_state: tuple[int, jnp.ndarray]) -> jnp.ndarray:
            idx, _ = inner_state
            return (idx < n_times) & (sample_times[idx] < next_time)

        def inner_body(inner_state: tuple[int, jnp.ndarray]) -> tuple[int, jnp.ndarray]:
            idx, positions = inner_state
            updated_positions = positions.at[idx].set(current_site)
            return (idx + 1, updated_positions)

        new_sample_idx, updated_sample_positions = jax.lax.while_loop(
            inner_condition, inner_body, (sample_idx, sample_positions)
        )

        return (
            new_sample_idx,
            next_time,
            next_site,
            updated_sample_positions,
            next_key,
        )

    _, _, _, final_positions, _ = jax.lax.while_loop(
        outer_condition, outer_body, init_outer_state
    )

    return final_positions


@timed
def solve_ensemble[L: Lattice[Any]](
    lattice: L,
    time_span: TimeSpan,
    initial_condition: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    key: jax.Array,
) -> HoppingSimulationResult[L]:
    """Solve the hopping ensemble."""
    keys = jax.random.split(key, initial_condition.shape[0])
    times = jax.numpy.linspace(time_span.t_start, time_span.t_end, time_span.n_steps)

    results = jax.vmap(
        _run_hopping_simulation_jit,
        in_axes=(None, 0, None, 0),
    )(lattice, initial_condition, times, keys)

    x_indices = np.einsum("ijk->ikj", np.array(results))

    return HoppingSimulationResult[L](
        system=lattice,
        times=np.array(times),
        x_indices=x_indices,
    )
