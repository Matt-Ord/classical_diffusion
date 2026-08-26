import jax
import jax.numpy as jnp


def _calculate_total_offsset_multiplications_complex(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    # scipy.signal.correlate handles complex numbers and conjugation automatically
    # Note: correlate(a, b) conjugates the first argument by default
    return jax.scipy.signal.correlate(lhs, rhs, mode="full")[: lhs.size][::-1]


def _time_average(
    time_sum: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the time-averaging denominator."""
    size = time_sum.shape[-1]
    return time_sum / jnp.arange(1, size + 1)[::-1]


def get_pairwise_isf(
    positions: jnp.ndarray,
    delta_k: jnp.ndarray,
) -> jnp.ndarray:
    """Get the restored displacement of a wavepacket."""
    scatter = jnp.exp(-1j * jnp.einsum("i,...ij->...j", delta_k, positions))

    # convolution_j = \sum_i^N-j e^(ik.x_i+j) e^(-ik.x_i)
    convolution = jnp.apply_along_axis(
        lambda m: _calculate_total_offsset_multiplications_complex(m, m),
        axis=-1,
        arr=scatter,
    )
    return _time_average(convolution)
