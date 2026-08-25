from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from classical_diffusion.hopping import CanonicalLattice


def get_deterministic_isf_jax[L: CanonicalLattice](
    system: L,
    probabilities: jnp.ndarray,
    delta_k: tuple[float, ...],
) -> jnp.ndarray:
    distances = system.x_points_from_indices(jnp.arange(probabilities.shape[1]))
    phase_factors = jnp.exp(1j * delta_k[0] * distances)
    return jnp.abs(jnp.dot(probabilities, phase_factors))
