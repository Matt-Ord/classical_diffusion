from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from classical_diffusion._isf import get_isf as get_ensemble_isf
from classical_diffusion._isf import with_span_times
from classical_diffusion.hopping._hopping import (
    DeterministicSolverResult,
    get_deterministic_probabilities,
)
from classical_diffusion.jax.hopping import (
    get_deterministic_isf as get_deterministic_isf_jax,
)
from classical_diffusion.simulation import IsfResult
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from classical_diffusion.hopping._system import Lattice
    from classical_diffusion.simulation import SimulationResult, TimeSpan


def get_isf(
    result: SimulationResult | DeterministicSolverResult,
    delta_k: tuple[float, ...],
    *,
    pairwise: bool = True,
) -> IsfResult:
    """Get the ISF of a hopping simulation.

    For an ensemble of trajectories this is the ISF of
    ``classical_diffusion.analysis.get_isf``, with its standard error. The
    probabilities of the deterministic solver give the ISF directly, as the
    sum of e^{i delta_k x} over the sites, with no error.
    """
    if not isinstance(result, DeterministicSolverResult):
        return get_ensemble_isf(result, delta_k, pairwise=pairwise)
    isf = get_deterministic_isf_jax(
        result.system.as_canonical(), jnp.array(result.probabilities), delta_k
    )
    return IsfResult(
        times=result.times - result.times[0],
        isf=np.array(isf),
        delta_k=delta_k,
        units=result.system.units,
    )


@timed
def solve_isf(
    system: Lattice,
    delta_k: tuple[float, ...],
    time_span: TimeSpan,
    *,
    shape: tuple[int, ...],
) -> IsfResult:
    """Solve the rate equation of the hopping model, and get its ISF.

    The probabilities of a lattice of the given shape are found by
    ``get_deterministic_probabilities``, starting from the central site. The
    times of the ISF are those of the time span, measured from its start.
    """
    result = get_deterministic_probabilities(system, shape, time_span)
    return with_span_times(get_isf(result, delta_k), time_span)
