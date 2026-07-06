import diffrax as dfx
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt


def drift(t, y, args):
    """Drift function for the harmonic Langevin equation."""
    x, v = y
    omega, gamma, _sigma = args
    return jnp.array([v, -(omega**2) * x - gamma * v])


def diffusion(t, y, args):
    """Diffusion function for the harmonic Langevin equation."""
    _x, _v = y
    _omega, _gamma, sigma = args
    return jnp.array([0.0, sigma])


def solve_single_trajectory(key):
    """Solve a single trajectory of the harmonic Langevin equation."""
    # Parameters
    omega, gamma, sigma = 1.0, 0, 1.0
    args = (omega, gamma, sigma)
    y0 = jnp.array([0])

    # Initial conditions
    x0 = 5.0
    v0 = 2.0
    y0 = jnp.array([x0, v0])

    # Time span
    t0 = 0.0
    t1 = 10.0
    dt0 = 0.01
    n = int((t1 - t0) / dt0)

    # Brownian noise
    bm = dfx.VirtualBrownianTree(t0, t1, tol=1e-4, shape=(), key=key)

    # Solver using Euler-Maruyama
    terms = dfx.MultiTerm(dfx.ODETerm(drift), dfx.ControlTerm(diffusion, bm))
    solver = dfx.EulerHeun()
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, n))

    sol = dfx.diffeqsolve(
        terms, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, args=args, saveat=saveat
    )
    return sol.ts, sol.ys


# Solve a single trajectory
master_key = jrandom.PRNGKey(42)
ts, ys = solve_single_trajectory(master_key)
fig, axs = plt.subplots(2, 1, figsize=(10, 6))
axs[0].plot(ts, ys[:, 0], label="1D Position (x)", color="royalblue", lw=1.5)
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Position")
axs[0].set_title("1D Langevin Harmonic Oscillator Trajectory")
axs[0].legend()
axs[1].plot(ts, ys[:, 1], label="1D Velocity (v)", color="darkgreen", lw=1.5)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Velocity")
axs[1].set_title("1D Langevin Harmonic Oscillator Trajectory")
axs[1].legend()
plt.tight_layout()
plt.savefig("/workspaces/classical_diffusion/examples/langevin_plot.png")
print("Plot successfully saved to file")
