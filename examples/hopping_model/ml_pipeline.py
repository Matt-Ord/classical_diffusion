import operator
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from classical_diffusion.analysis import get_isf
from classical_diffusion.langevin import (
    CanonicalSystem,
    KramersSystem1D,
)
from classical_diffusion.langevin._langevin import solve_overdamped_ensemble_jax
from classical_diffusion.plot import get_fancy_figure, get_figure, get_measured_data
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from classical_diffusion.hopping import KramersParameters


class JaxEnsembleResults(TypedDict):
    parameters: KramersParameters
    initial_conditions: tuple[
        np.ndarray[Any, np.dtype[np.floating[Any]]],
        np.ndarray[Any, np.dtype[np.floating[Any]]],
    ]
    result: tuple[jnp.ndarray, jnp.ndarray]


@jax.jit
def _generate_canonical_kramers_system(
    _key: jax.Array,
) -> tuple[tuple[float, float, float, float, float, float], CanonicalSystem]:

    rand = jax.random.uniform(_key, shape=(), minval=0.5, maxval=3.0)
    omega_well = 1.0
    omega_barrier = 1.0
    barrier_energy = rand
    m = 1.0
    kbt = 0.5
    gamma = 0.1

    params = (omega_well, omega_barrier, barrier_energy, m, kbt, gamma)
    system = KramersSystem1D(
        omega_well=omega_well,
        omega_barrier=omega_barrier,
        barrier_energy=barrier_energy,
        m=m,
        kbt=kbt,
        gamma=gamma,
    ).as_canonical()

    return params, system  # ty: ignore[invalid-return-type]


const_initial_cond = np.full((1, 1), 0.0)


def _inits_constant(
    _key: jax.Array,
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating[Any]]],
    np.ndarray[Any, np.dtype[np.floating[Any]]],
]:
    return (const_initial_cond, const_initial_cond)


@jax.jit
def run_langevin_trajectories(
    keys: jax.Array,
) -> JaxEnsembleResults:
    """Run langevin trajectories."""
    print("Compiling JAX function")

    # Times
    times = TimeSpan(t_start=0, t_end=40, n_steps=200)

    def body(time_span: TimeSpan, _key: jax.Array) -> JaxEnsembleResults:

        param_key, cond_key, sim_key = jax.random.split(_key, 3)

        params, system = _generate_canonical_kramers_system(param_key)

        initial_conditions = _inits_constant(cond_key)

        result = solve_overdamped_ensemble_jax(
            system,
            time_span,
            initial_conditions,
            sim_key,
        )

        return {
            "parameters": params,
            "initial_conditions": initial_conditions,
            "result": result,
        }

    return jax.vmap(body, (None, 0))(times, keys)


@timed
def generate_langevin_trajectories(
    folderpath: str,
    n_samples: int,
) -> None:
    """Run the langevin simulations and save to file."""
    keys = jax.random.split(jax.random.key(100), n_samples)

    batched_trajectories = jax.tree.map(np.asarray, run_langevin_trajectories(keys))
    trajectories = [
        jax.tree.map(operator.itemgetter(i), batched_trajectories)
        for i in range(n_samples)
    ]

    # Define file path
    filepath = folderpath + "/langevin_traj.pkl"
    with Path(filepath).open("wb") as file:
        pickle.dump(trajectories, file)


@timed
def generate_isfs(folderpath: str) -> None:

    # Define local paths to save data to
    trajectories_path = folderpath + "/langevin_traj.pkl"
    isfs_path = folderpath + "/langevin_isfs.pkl"

    delta_k = 0.5

    # Open the trajectories file and load in the trajectories
    with Path(trajectories_path).open("rb") as file:
        trajectory_records = pickle.load(file)

    # Open the isfs file and calculate, then save, the isfs
    with Path(isfs_path).open("wb") as file:
        for trajectory in trajectory_records:
            times, x_points = trajectory.get("result")
            isf = get_isf(x_points, (delta_k,))[0]

            isf_record = {
                "parameters": trajectory.get("parameters"),
                "isf": {"time": times, "isf": isf},
            }
            pickle.dump(isf_record, file)


@timed
def plot_isfs(folderpath: str) -> None:

    isfs_path = folderpath + "/langevin_isfs.pkl"
    isf_records = []
    with Path(isfs_path).open("rb") as file:
        while True:
            try:
                isf_records.append(pickle.load(file))
            except EOFError:
                break

    for index in range(len(isf_records)):
        isf_data = isf_records[index].get("isf")
        isf = get_measured_data(isf_data.get("isf"), "real")

        fig, ax = get_fancy_figure()
        fig, ax = get_figure(ax)
        (line,) = ax.plot(isf_data.get("time"), isf)
        line.set_label("ISF")

        ax.set_xlabel("Time / s")
        ax.set_ylabel("ISF")

        ax.set_xlim(0, right=20)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("Langevin ISFs from pipeline")

        fig.savefig(f"./examples/hopping_model/test_{index}.isf.pdf")


if __name__ == "__main__":
    path = "./examples/data"
    print("running")
    generate_langevin_trajectories(path, 10)
    generate_isfs(path)
    plot_isfs(path)
