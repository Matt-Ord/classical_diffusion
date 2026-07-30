import os
from functools import cached_property
from typing import TYPE_CHECKING, Any

from classical_diffusion.plot import get_figure

os.environ["JAX_ENABLE_X64"] = "True"
import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm

from classical_diffusion.hopping._system import CanonicalLattice, Lattice
from classical_diffusion.simulation import SimulationResult, TimeSpan
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


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


class DeterministicSolverResult[L: Lattice]:
    def __init__(
        self, *, system: L, times: jnp.ndarray, probability_matrix: jnp.ndarray
    ) -> None:
        self._system = system
        self._times = times
        self._probability_matrix = probability_matrix

    @property
    def system(self) -> L:
        return self._system

    @property
    def times(self) -> jnp.ndarray:
        return self._times

    @property
    def probability_matrix(self) -> jnp.ndarray:
        return self._probability_matrix


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


@timed
def get_deterministic_probabilities_slow[L: Lattice[Any]](
    system: L,
    finite_lattice_shape: tuple,
    time_span: TimeSpan,
    initial_position: jnp.ndarray,
) -> DeterministicSolverResult:
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

    prob_matrix = jax.vmap(solve_single_time)(times)
    prob_matrix = jnp.clip(prob_matrix, min=0)
    prob_matrix /= jnp.sum(prob_matrix, axis=-1, keepdims=True)

    return DeterministicSolverResult(
        system=system, times=times, probability_matrix=prob_matrix
    )


@timed
def get_deterministic_probabilities[L: Lattice[Any]](
    system: L,
    finite_lattice_shape: tuple,
    time_span: TimeSpan,
    initial_position: jnp.ndarray,
) -> DeterministicSolverResult:
    """Use deterministic formula to return the ISF, inefficiently."""
    #
    # Rate matrix, M
    # M[a,b] = - rate (b -> a)
    # M[a,a] = sum_i ( rates a -> i)

    max_lattice_index = jnp.prod(jnp.array(finite_lattice_shape))
    all_sites = jnp.arange(0, max_lattice_index)

    times = jnp.linspace(time_span.t_start, time_span.t_end, time_span.n_steps)
    initial_p = jnp.full(max_lattice_index, 0.0)
    initial_p = initial_p.at[
        jnp.ravel_multi_index(tuple(initial_position), finite_lattice_shape)
    ].set(1)

    # Find probabilities at a given time with diffrax
    def vector_field(
        _t: Any,  # ruff:ignore[any-type]
        p: jnp.ndarray,
        _args: Any,  # ruff:ignore[any-type]
    ) -> jnp.ndarray:

        hop_sites, hop_rates = system.get_rates(all_sites)

        # p[hop_site] has shape (N, 4)
        # hop_rates    has shape (N, 4)
        """
        def dot_product(
            singles_hop_sites: jnp.ndarray, hop_rates: jnp.ndarray
        ) -> jnp.ndarray:

            def scan_body(total: float, data: tuple[int, float]) -> float:
                site, rate = data
                return total + rate * p[site]

            return jax.lax.scan(scan_body, 0, (singles_hop_sites, hop_rates))

        return jax.vmap(dot_product, in_axes=(0, 0))(hop_sites, hop_rates)
        """
        return jnp.sum(hop_rates * p[hop_sites], axis=-1)

    term = dfx.ODETerm(vector_field)

    # Core solver for an initial condition
    @jax.jit
    def solve_one(p0: jnp.ndarray) -> jnp.ndarray:
        sol = dfx.diffeqsolve(
            term,
            solver=dfx.Tsit5(),  # cspell: disable-line
            t0=0,
            t1=times[-1],
            dt0=times[1] - times[0],
            y0=p0,
            args=None,
            saveat=dfx.SaveAt(ts=times),
            stepsize_controller=dfx.PIDController(
                rtol=1e-6,  # cspell: disable-line
                atol=1e-8,
            ),  # cspell: disable-line
            max_steps=100_000_000,
        )
        return sol.ys

    prob_matrix = solve_one(initial_p)

    return DeterministicSolverResult(
        system=system, times=times, probability_matrix=prob_matrix
    )


@timed
def _get_deterministic_isf[L: Lattice[Any]](
    system: L,
    prob_matrix: jnp.ndarray,
    delta_k: float,
) -> jnp.ndarray:
    distances = system.x_points_from_indices(np.arange(prob_matrix.shape[1]))
    phase_factors = jnp.exp(1j * delta_k * distances)
    return jnp.abs(jnp.dot(prob_matrix, phase_factors))


@timed
def plot_deterministic_isf[L: Lattice[Any]](
    system: L,
    result: DeterministicSolverResult,
    delta_k: float,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the ensemble-averaged ISF over time, with a shaded ±1 SEM band."""
    fig, ax = get_figure(ax)

    isf = _get_deterministic_isf(system, result.probability_matrix, delta_k)
    (line,) = ax.plot(np.array(result.times), np.array(isf))
    line.set_label("ISF")

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    return fig, ax, line
