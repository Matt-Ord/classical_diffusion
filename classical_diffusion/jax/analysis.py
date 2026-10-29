# cspell: ignore ifft
import jax
import jax.numpy as jnp

_BATCH_SIZE = 256
"""The number of trajectories correlated at once, which bounds the memory used."""


def _get_pairwise_sum(positions: jnp.ndarray, delta_k: jnp.ndarray) -> jnp.ndarray:
    """Get sum_i^N-j e^(ik.x_i+j) e^(-ik.x_i) for every offset j, along the last axis.

    The autocorrelation is calculated with an FFT, zero padded to at least twice
    the length so that the circular correlation does not wrap around.
    """
    scatter = jnp.exp(1j * jnp.einsum("i,...ij->...j", delta_k, positions))
    size = scatter.shape[-1]
    n_fft = 1 << (2 * size - 1).bit_length()
    transformed = jnp.fft.fft(scatter, n_fft, axis=-1)
    return jnp.fft.ifft(jnp.abs(transformed) ** 2, axis=-1)[..., :size]


def _time_average(
    time_sum: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the time-averaging denominator."""
    size = time_sum.shape[-1]
    return time_sum / jnp.arange(1, size + 1)[::-1]


@jax.jit
def get_pairwise_isf(positions: jnp.ndarray, delta_k: jnp.ndarray) -> jnp.ndarray:
    """Get the ISF of each trajectory, from the pairwise correlation of the positions."""
    samples = positions.reshape(-1, *positions.shape[-2:])
    isf = jax.lax.map(
        lambda x: _time_average(_get_pairwise_sum(x, delta_k)),
        samples,
        batch_size=_BATCH_SIZE,
    )
    return isf.reshape(*positions.shape[:-2], positions.shape[-1])
