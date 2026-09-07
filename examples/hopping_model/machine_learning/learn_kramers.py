import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"

import jax
import jax.numpy as jnp
from pysr import PySRRegressor


def _kramers_rate(
    omega_well: jnp.ndarray,
    omega_barrier: jnp.ndarray,
    barrier_energy: jnp.ndarray,
    mass: jnp.ndarray,
    kbt: jnp.ndarray,
    gamma: jnp.ndarray,
) -> jnp.ndarray:
    return ((omega_well * omega_barrier) / (2 * jnp.pi * gamma)) * jnp.exp(
        -barrier_energy / kbt
    )


if __name__ == "__main__":
    test_params = jax.random.uniform(
        jax.random.key(100), (200, 6), minval=0.5, maxval=5.0
    )

    test_rates = _kramers_rate(
        test_params[:, 0],
        test_params[:, 1],
        test_params[:, 2],
        test_params[:, 3],
        test_params[:, 4],
        test_params[:, 5],
    )

    pysr_model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["exp"],
    )
    pysr_model.fit(test_params, test_rates)

    best_formula = pysr_model.sympy()
    print("Form: ", best_formula)
