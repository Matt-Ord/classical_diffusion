"""Simulations of classical diffusion."""

import jax

# Simulation times become large compared to a single time step, which 32 bit
# floats cannot resolve. Long simulations are then inaccurate, and very slow
# as the adaptive step size controllers repeatedly reject steps.
jax.config.update("jax_enable_x64", True)  # ruff: ignore[non-empty-init-module, boolean-positional-value-in-call]
