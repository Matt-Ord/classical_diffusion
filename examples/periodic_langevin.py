from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True, kw_only=True)
class PeriodicSimulationResult:
    """Results of a simulation of the periodic Langevin equation."""

    times: jnp.ndarray
    xs: jnp.ndarray
    ps: jnp.ndarray
    omega: float
    gamma: float
    sigma: float
    m: float


def perioidc_potential(x, V0=2.0, k=1.0):
    return -V0 * jnp.cos(k * x)
