from typing import TYPE_CHECKING

from classical_diffusion._isf import get_isf, with_span_times
from classical_diffusion.jax.langevin import DEFAULT_STEP_CONFIG, StepConfig
from classical_diffusion.langevin._langevin import (
    solve_ensemble,
    solve_ensemble_overdamped,
)

if TYPE_CHECKING:
    import jax

    from classical_diffusion.langevin._system import System
    from classical_diffusion.simulation import IsfResult, TimeSpan


def solve_isf(  # ruff:ignore[too-many-arguments]
    system: System,
    delta_k: tuple[float, ...],
    time_span: TimeSpan,
    n_samples: int,
    *,
    step_config: StepConfig = DEFAULT_STEP_CONFIG,
    pairwise: bool = True,
    _key: jax.Array | None = None,
) -> IsfResult:
    """Solve an ensemble of Langevin trajectories, and get its ISF.

    The trajectories are those of ``solve_ensemble``, which are cached, so the
    ISF at another scattering wavevector reuses them. The times of the ISF are
    those of the time span, measured from its start.
    """
    result = solve_ensemble(
        system, time_span, n_samples, step_config=step_config, _key=_key
    )
    return with_span_times(get_isf(result, delta_k, pairwise=pairwise), time_span)


def solve_overdamped_isf(  # ruff:ignore[too-many-arguments]
    system: System,
    delta_k: tuple[float, ...],
    time_span: TimeSpan,
    n_samples: int,
    *,
    step_config: StepConfig = DEFAULT_STEP_CONFIG,
    pairwise: bool = True,
    _key: jax.Array | None = None,
) -> IsfResult:
    """Solve an ensemble of overdamped Langevin trajectories, and get its ISF.

    The trajectories are those of ``solve_ensemble_overdamped``, which are
    cached, so the ISF at another scattering wavevector reuses them. The times
    of the ISF are those of the time span, measured from its start.
    """
    result = solve_ensemble_overdamped(
        system, time_span, n_samples, step_config=step_config, _key=_key
    )
    return with_span_times(get_isf(result, delta_k, pairwise=pairwise), time_span)
