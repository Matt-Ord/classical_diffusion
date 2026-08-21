import jax
import jax.numpy as jnp

from classical_diffusion.hopping import CanonicalLattice


@jax.jit
def get_deterministic_isf_jax[L: CanonicalLattice](
    system: L,
    probabilities: jnp.ndarray,
    delta_k: float,
) -> jnp.ndarray:
    distances = system.x_points_from_indices(jnp.arange(probabilities.shape[1]))
    phase_factors = jnp.exp(1j * delta_k * distances)
    return jnp.abs(jnp.dot(probabilities, phase_factors))
