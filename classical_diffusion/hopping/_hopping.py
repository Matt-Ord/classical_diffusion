from dataclasses import dataclass
from functools import cached_property
from typing import Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Tsit5  # cspell: disable-line

from classical_diffusion.hopping._system import CanonicalLattice, Lattice
from classical_diffusion.simulation import SimulationResult, TimeSpan
from classical_diffusion.util import timed


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


@dataclass(frozen=True, kw_only=True)
class DeterministicSolverResult[L: Lattice]:
    system: L
    times: np.ndarray[tuple[int], np.dtype[np.float32]]
    probabilities: np.ndarray[tuple[int, int], np.dtype[np.float32]]


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


@timed
def get_ensemble_probabilities[L: Lattice](
    system: L,
    shape: tuple[int, ...],
    time_span: TimeSpan,
    initial_position: int,
) -> DeterministicSolverResult:
    """Use a deterministic PDE to find the ensemble probabilities at all times."""
    times = jnp.linspace(time_span.t_start, time_span.t_end, time_span.n_steps)

    initial_p = jnp.full(np.prod(shape), 0.0, dtype=jnp.float32)
    initial_p = initial_p.at[initial_position].set(1)

    hop_sites, hop_rates = system.get_rates(np.arange(np.prod(shape)))

    sol = _get_deterministic_probabilities_jit(initial_p, times, hop_sites, hop_rates)

    return DeterministicSolverResult(
        system=system, times=np.array(times), probabilities=np.array(sol)
    )
