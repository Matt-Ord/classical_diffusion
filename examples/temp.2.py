import jax.random as jrandom
import numpy as np

from classical_diffusion.langevin import (
    TimeSpan,
    solve_ensemble,
    solve_single,
)
from classical_diffusion.system import PeriodicSystem1D

if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=0.1, temperature=1.0, m=1.0, delta_x=5.0, barrier_energy=1.5
    )
    time_span = TimeSpan(t0=1, t1=10, dt=0.01)

    dist_result = solve_single.load_or_call_cached(
        system, time_span, (np.array([0]), np.array([0])), _key=key
    )
    dist_result = solve_single.call_cached(
        system, time_span, (np.array([0]), np.array([0])), _key=key
    )
    dist_result = solve_single.call_cached(
        system, time_span, (np.array([0]), np.array([0])), _key=key
    )

    dist_result = solve_ensemble.load_or_call_cached(
        system, time_span, (np.zeros((1, 1)), np.zeros((1, 1))), _key=key
    )
    dist_result = solve_ensemble.call_cached(
        system, time_span, (np.zeros((1, 1)), np.zeros((1, 1))), _key=key
    )
    dist_result = solve_ensemble.call_cached(
        system, time_span, (np.zeros((1, 1)), np.zeros((1, 1))), _key=key
    )
