from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

MIN_SPLIT_POINTS = 4
MIN_VARIANCE = 1e-15


def _identity(u: jax.Array) -> jax.Array:
    return u


@eqx.filter_jit
def get_trajectory_breakpoints(
    x: jnp.ndarray,
    *,
    process_points: "Callable[[jnp.ndarray], jnp.ndarray] | None" = None,  # ruff: ignore[quoted-annotation]
) -> jnp.ndarray:
    """Discretize the signal using the objective Kalafut-Visscher step detection algorithm."""  # cspell: disable-line
    n = x.shape[0]

    process_points = process_points if process_points is not None else _identity

    # Precalculate prefix sums for O(1) segment variance and sum-of-squares evaluation
    p1 = jnp.pad(jnp.cumsum(x), (1, 0))
    p2 = jnp.pad(jnp.cumsum(x**2), (1, 0))

    # Initialize fixed-size execution stack and boundary tracking
    stack_starts = jnp.zeros(n, dtype=jnp.int32).at[0].set(0)
    stack_ends = jnp.zeros(n, dtype=jnp.int32).at[0].set(n)
    stack_ptr = jnp.int32(1)
    breakpoints = jnp.zeros(n + 1, dtype=jnp.bool_).at[0].set(True).at[n].set(True)

    init_state = (stack_starts, stack_ends, stack_ptr, breakpoints)

    def cond_fn(
        state: tuple[jnp.ndarray, jnp.ndarray, jax.Array, jnp.ndarray],
    ) -> jax.Array:
        _, _, ptr, _ = state
        return ptr > 0

    def body_fn(  # ruff: ignore[too-many-locals]
        state: tuple[jnp.ndarray, jnp.ndarray, jax.Array, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array, jnp.ndarray]:
        starts, ends, ptr, breakpoints = state

        # Pop current top segment
        pop_ptr = ptr - 1
        start = starts[pop_ptr]
        end = ends[pop_ptr]
        n_seg = end - start

        # Base segment statistics
        s1 = p1[end] - p1[start]
        s2 = p2[end] - p2[start]
        base_mu = process_points(s1 / jnp.maximum(n_seg, 1))
        base_ssr = s2 - 2 * base_mu * s1 + n_seg * base_mu**2
        base_var = base_ssr / jnp.maximum(n_seg, 1)

        can_split = (n_seg > MIN_SPLIT_POINTS) & (base_var > MIN_VARIANCE)

        # Evaluate all candidate split points k in parallel across length N+1
        k = jnp.arange(n + 1)
        k_left_len = k - start
        k_right_len = end - k

        s1_left = p1[k] - p1[start]
        s1_right = s1 - s1_left

        mu_left = process_points(s1_left / jnp.maximum(k_left_len, 1))
        mu_right = process_points(s1_right / jnp.maximum(k_right_len, 1))

        split_ssr = (
            s2
            - 2 * mu_left * s1_left
            + k_left_len * mu_left**2
            - 2 * mu_right * s1_right
            + k_right_len * mu_right**2
        )

        split_vars = jnp.maximum(split_ssr / jnp.maximum(n_seg, 1), MIN_VARIANCE)
        delta_sic = n_seg * jnp.log(base_var / split_vars) - jnp.log(
            jnp.maximum(n_seg, 1)
        )

        # Mask invalid candidate split points: start + 2 <= k <= end - 2
        valid_k_mask = (k >= start + 2) & (k <= end - 2)
        delta_sic_masked = jnp.where(valid_k_mask, delta_sic, -jnp.inf)

        best_k = jnp.argmax(delta_sic_masked)
        best_delta = delta_sic_masked[best_k]

        should_split = can_split & (best_delta > 0)

        # Update breakpoints and stack buffers
        new_breakpoints = jnp.where(
            should_split, breakpoints.at[best_k].set(True), breakpoints
        )

        new_starts = jnp.where(
            should_split,
            starts.at[pop_ptr].set(start).at[pop_ptr + 1].set(best_k),
            starts,
        )
        new_ends = jnp.where(
            should_split,
            ends.at[pop_ptr].set(best_k).at[pop_ptr + 1].set(end),
            ends,
        )
        new_ptr = jnp.where(should_split, pop_ptr + 2, pop_ptr)

        return (new_starts, new_ends, new_ptr, new_breakpoints)

    # Execute bounded segmentation loop
    _, _, _, final_breakpoints = jax.lax.while_loop(cond_fn, body_fn, init_state)

    return final_breakpoints


@eqx.filter_jit
def partition_trajectory(
    x: jnp.ndarray,
    *,
    process_points: "Callable[[jnp.ndarray], jnp.ndarray] | None" = None,  # ruff: ignore[quoted-annotation]
) -> jnp.ndarray:
    """Discretize the signal using the objective Kalafut-Visscher step detection algorithm."""  # cspell: disable-line
    n = x.shape[0]

    process_points = process_points if process_points is not None else _identity

    final_breakpoints = get_trajectory_breakpoints(x, process_points=process_points)

    # Reconstruct piecewise constant path in parallel
    segment_ids = jnp.cumsum(final_breakpoints[:-1]) - 1
    counts = jnp.bincount(segment_ids, length=n)
    sums = jnp.bincount(segment_ids, weights=x, length=n)

    segment_means = jnp.where(counts > 0, sums / jnp.maximum(counts, 1), 0.0)
    processed_means = process_points(segment_means)

    return processed_means[segment_ids]
