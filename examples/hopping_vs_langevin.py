import jax
import numpy as np

from classical_diffusion.analysis import plot_isf
from classical_diffusion.hopping import Lattice1D, solve_ensemble
from classical_diffusion.hopping._system import get_kramers_lattice_cosine
from classical_diffusion.langevin import PeriodicSystem1D
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan


def plot_1d_hopping_isf(
    system: Lattice1D = Lattice1D(lattice_spacing=5, hop_time=15),
    time_span: TimeSpan = TimeSpan(t_end=400, n_steps=4000),
    initial_condition: np.ndarray = np.full((4000, 1), 0.0),
) -> None:

    results = solve_ensemble(
        system=system,
        time_span=time_span,
        initial_condition=initial_condition,
        key=jax.random.PRNGKey(seed=100),
    )

    fig, ax = get_fancy_figure()
    delta_k = (0.5 * 2 * np.pi / results.system.lattice_spacing,)
    _, ax, line_0, _ = plot_isf(result=results, delta_k=delta_k, ax=ax)
    line_0.set_label("Kramers simulation")
    ax.set_xlim(0, right=5)
    ax.set_ylim(0, 1)

    print("saving isf")
    fig.savefig("./examples/1d_karmers.isf.pdf")

    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig("./examples/1d_kramers .isf.log.pdf")


def _kramers_test() -> None:
    system = PeriodicSystem1D(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        delta_x=5,
        barrier_energy=0.5,
    )
    lattice = get_kramers_lattice_cosine(system)
    print(lattice.hop_time)
    print(lattice.lattice_spacing)

    plot_1d_hopping_isf(system=lattice)


if __name__ == "__main__":
    _kramers_test()
