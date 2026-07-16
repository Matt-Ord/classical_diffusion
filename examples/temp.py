from pathlib import Path

import numpy as np

from classical_diffusion.langevin import (
    FlatParameters,
    InitialConditions,
    PhysicalParameters,
    TimeSpan,
)
from classical_diffusion.util import cached


@cached(Path(__file__).parent / "temp.cache")
def test() -> FlatParameters:
    return FlatParameters(
        physical_parameters=PhysicalParameters(gamma=0.5, temperature=3, m=1),
        time_span=TimeSpan(t0=0, t1=30_000, dt=0.01, burn_in=1),
        initial_conditions=InitialConditions(
            x0=np.array([[0.0, 0.0]]), p0=np.array([[0.0, 0.0]])
        ),
    )


test.call_cached()

test.load_or_call_cached()
