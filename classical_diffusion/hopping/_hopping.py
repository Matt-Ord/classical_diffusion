from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from classical_diffusion.analysis import SimulationResult
from classical_diffusion.system._system import (
    Lattice,  # ruff:ignore[typing-only-first-party-import]
)


@dataclass(frozen=True, kw_only=True)
class HoppingSimulationResult(SimulationResult):
    lattice: Lattice[Any]


class Particle:
    position: np.ndarray
    dimensions: int

    def __init__(
        self, initial_position: np.ndarray[Any, np.dtype[np.floating]], dimensions: int
    ) -> None:
        self.positions = np.array([initial_position])
        self.dimensions = dimensions

    def hop(self, p_hop: np.floating, lattice_spacing: np.floating) -> None:
        # Generate a random number to determine if the particle hops
        if np.random.rand() < p_hop:
            # Randomly choose a direction to hop
            if self.dimensions == 1:
                direction = np.random.choice([-1, 1])
            elif self.dimensions == 2:
                direction_index = np.random.choice(4)
                direction = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])[
                    direction_index
                ]
            elif self.dimensions == 3:
                direction_index = np.random.choice(6)
                direction = np.array(
                    [
                        [0, 0, 1],
                        [0, 0, -1],
                        [-1, 0, 0],
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 1, 0],
                    ]
                )[direction_index]

            current_position = self.positions[-1]
            new_position = current_position + np.array(direction) * lattice_spacing

            self.positions = np.append(self.positions, [new_position], axis=0)
        else:
            # If the particle does not hop, it stays in the same position
            self.positions = np.append(self.positions, [self.positions[-1]], axis=0)


def _run_inefficient_hopping_simulation(
    lattice: Lattice,
    key: jax.Array,
    start_position: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a simple hopping simulation."""
    # Initialize the position of the particle
    position = start_position
    positions = [position]

    1 / lattice.diff_time

    for _characteristic_time in range(n_steps):
        # Generate a random number to decide whether to hop or not
        rand_val = jax.random.uniform(key)
        if rand_val < lattice.r_hop:
            # Randomly choose to hop left or right
            hop_direction = jax.random.choice(key, lattice.directions)
            position += hop_direction * lattice.lattice_spacing
        positions.append(position)

    times = np.arange(n_steps) * lattice.diff_time

    print(positions)
    return np.array(positions), times


def _run_hopping_simulation(  # For debugging
    lattice: Lattice,
    initial_position: jnp.ndarray,
    total_time: float,
    max_steps: int,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run a simple hopping simulation."""
    positions = jnp.zeros((max_steps, initial_position.shape[0]))
    times = jnp.zeros(max_steps)

    # Initialise the first step
    positions = positions.at[0].set(initial_position)
    times = times.at[0].set(0.0)

    print("r_hop: ", lattice.r_hop)
    # Pack current simulation state into single tuple
    initial_state = (0, positions, times, key)

    def loop_condition(
        state: tuple[int, jnp.ndarray, jnp.ndarray, jax.Array],
    ) -> bool:
        step, _, times, _ = state
        current_time = jnp.array(times)[step]
        # Loop continues if time remains and maximum not exceed
        return (current_time < total_time) & (step < max_steps - 1)  # ty:ignore[invalid-return-type]

    def loop_body(
        state: tuple[int, jnp.ndarray, jnp.ndarray, jax.Array],
    ) -> tuple[int, jnp.ndarray, jnp.ndarray, jax.Array]:

        step, positions, times, key = state
        print("\n\n")
        print(step)
        print(positions)
        print(times)
        # Split random key
        key_hop, key_dir, key_dt, next_key = jax.random.split(key, 4)

        u = jax.random.uniform(key_hop)  # Random variable to select hopping/not hopping
        hop_direction = jax.random.choice(
            key_dir, lattice.directions
        )  # Random choice of direction to hop in

        position = positions[step]
        print("\nHop attempt in direction, from")
        print(hop_direction, ",", position)
        print("\n")
        print(u)

        if u < lattice.r_hop:
            print("Hop should occur")

        position = jnp.where(
            u < lattice.r_hop,
            position + hop_direction * lattice.lattice_spacing,
            position,
        )

        print(position)

        # Progress time by dt from an exponential distribution
        u_time = jax.random.uniform(key_dt)
        new_time = times[step] - jnp.log(u_time)

        # Update state of simulation
        next_step = step + 1
        positions = positions.at[next_step].set(position)
        times = times.at[next_step].set(new_time)

        return (next_step, positions, times, next_key)

    state = initial_state
    while loop_condition(state):
        state = loop_body(state)

    _, final_pos, final_tim, _ = state
    return (final_pos, final_tim)


print("before jit")


@jax.jit(static_argnames=("total_time", "max_steps"))
def _run_hopping_simulation_jit(  # Rejection free KMC, 2 outcome states
    lattice: Lattice,
    initial_position: jnp.ndarray,
    total_time: float,
    max_steps: int,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run a simple hopping simulation."""
    positions = jnp.zeros((max_steps, initial_position.shape[0]))
    times = jnp.zeros(max_steps)

    # Initialise the first step
    positions = positions.at[0].set(initial_position)
    times = times.at[0].set(0.0)

    # Pack current simulation state into single tuple
    initial_state = (0, positions, times, key)

    def loop_condition(
        state: tuple[int, jnp.ndarray, jnp.ndarray, jax.Array],
    ) -> bool:
        step, _, times, _ = state
        current_time = jnp.array(times)[step]
        # Loop continues if time remains and maximum not exceed
        return (current_time < total_time) & (step < max_steps - 1)  # ty:ignore[invalid-return-type]

    def loop_body(
        state: tuple[int, jnp.ndarray, jnp.ndarray, jax.Array],
    ) -> tuple[int, jnp.ndarray, jnp.ndarray, jax.Array]:

        step, positions, times, key = state

        # Split random key
        key_hop, key_dir, key_dt, next_key = jax.random.split(key, 4)

        u = jax.random.uniform(key_hop)  # Random variable to select hopping/not hopping
        hop_direction = jax.random.choice(
            key_dir, lattice.directions
        )  # Random choice of direction to hop in

        position = positions[step]
        position = jnp.where(
            u < lattice.r_hop,
            position + hop_direction * lattice.lattice_spacing,
            position,
        )

        # Progress time by dt from an exponential distribution
        u_time = jax.random.uniform(key_dt)
        new_time = times[step] - jnp.log(u_time)

        # Update state of simulation
        next_step = step + 1
        positions = positions.at[next_step].set(position)
        times = times.at[next_step].set(new_time)

        return (next_step, positions, times, next_key)

    _final_step, positions, times, _ = jax.lax.while_loop(
        loop_condition, loop_body, initial_state
    )

    # positions = jnp.transpose(positions) if needed to match langevin position format
    return (jnp.array(positions), jnp.array(times))


print("after jit")


def _solve_hopping_ensemble(
    lattice: Lattice[Any],
    total_time: float,
    initial_position: jnp.ndarray,
    n_samples: int,
    key: jax.Array,
) -> HoppingSimulationResult:
    """Solve the hopping ensemble."""
    # Split the key for each sample
    keys = jax.random.split(key, n_samples)
    max_steps = 10000  # (int(total_time / lattice.diff_time) * 10)  # take average number of steps required and set the limit to be 10 times this

    # Run the simulation for each sample
    results = jax.vmap(
        _run_hopping_simulation_jit,
        in_axes=(None, None, None, None, 0),
    )(lattice, initial_position, total_time, max_steps, keys)

    print(results[0].shape)

    return HoppingSimulationResult(
        times=np.array(results[1]), x_points=np.array(results[0]), lattice=lattice
    )
