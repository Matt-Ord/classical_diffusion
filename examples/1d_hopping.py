import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from classical_diffusion.hopping._analysis import plot_hopping_isf
from classical_diffusion.hopping._hopping import _solve_hopping_ensemble
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import (
    Lattice1D,
)


def _plot_1d_hopping_isf() -> None:

    lattice = Lattice1D(lattice_spacing=2.5, diff_time=10)
    total_time = 100
    results = _solve_hopping_ensemble(
        lattice=lattice,
        total_time=total_time,
        initial_position=jnp.array([0.0]),
        n_samples=100,
        key=jrandom.PRNGKey(100),
    )

    print(results.x_points.shape)
    fig, ax = get_fancy_figure()

    delta_k = np.array([0.5 * 2 * np.pi / lattice.lattice_spacing])
    _, ax, line_0 = plot_hopping_isf(
        result=results,
        delta_k=delta_k,
        ax=ax,
    )
    line_0.set_label("Hopping simulation")

    print("saving figure")
    fig.savefig("./examples/1d_lattice.isf.pdf")


if __name__ == "__main__":
    print("running")
    _plot_1d_hopping_isf()
