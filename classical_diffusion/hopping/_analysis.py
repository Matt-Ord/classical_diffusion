from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from classical_diffusion.hopping._hopping import get_deterministic_probabilities
from classical_diffusion.jax.hopping import (
    get_deterministic_isf as get_deterministic_isf_jax,
)
from classical_diffusion.simulation import IsfResult
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from classical_diffusion.hopping._system import Lattice
    from classical_diffusion.simulation import TimeSpan


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
    solver is deterministic, so the ISF has no error.
    """
    result = get_deterministic_probabilities(system, shape, time_span)
    isf = get_deterministic_isf_jax(
        system.as_canonical(), jnp.array(result.probabilities), delta_k
    )
    return IsfResult(
        times=result.times - result.times[0],
        isf=np.array(isf),
        delta_k=delta_k,
        units=system.units,
    )
