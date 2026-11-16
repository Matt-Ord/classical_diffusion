from typing import Any

import jax.numpy as jnp
import numpy as np

from classical_diffusion.jax.analysis import get_pairwise_isf as get_pairwise_isf_jax
from classical_diffusion.simulation import IsfResult, SimulationResult, TimeSpan


def _get_origin_isf(
    positions: np.ndarray[Any, np.dtype[np.floating]],
    delta_k: tuple[float, ...],
    *,
    origin_idx: int,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """Get the ISF of each trajectory, with respect to its position at the origin."""
    phase = np.einsum(
        "i,...ij->...j",
        delta_k,
        positions - positions[..., origin_idx].reshape((*positions.shape[:-1], 1)),
    )
    return np.exp(1j * phase)


def _get_pairwise_isf(
    positions: np.ndarray[Any, np.dtype[np.floating]],
    delta_k: tuple[float, ...],
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """Get the ISF of each trajectory, from the pairwise correlation of its positions."""
    return np.array(get_pairwise_isf_jax(jnp.array(positions), jnp.asarray(delta_k)))


def get_isf(
    result: SimulationResult,
    delta_k: tuple[float, ...],
    *,
    pairwise: bool = True,
) -> IsfResult:
    """Get the ISF of an ensemble of trajectories, with its standard error.

    The pairwise ISF averages every pair of times of each trajectory with the
    same separation, which assumes the trajectory is stationary. Otherwise, the
    ISF is taken with respect to the position at the time closest to zero.
    """
    if pairwise:
        isf = _get_pairwise_isf(result.x_points, delta_k)
        times = result.times - result.times[0]
    else:
        origin_idx = np.argmin(np.abs(result.times)).item()
        isf = _get_origin_isf(result.x_points, delta_k, origin_idx=origin_idx)
        times = result.times - result.times[origin_idx]

    return IsfResult(
        times=times,
        isf=np.mean(isf, axis=0),
        error=np.std(isf, axis=0) / np.sqrt(isf.shape[0]),
        delta_k=delta_k,
        units=result.units,
    )


def with_span_times(result: IsfResult, time_span: TimeSpan) -> IsfResult:
    """Return the ISF of a simulation over a time span, at exactly the times of the span since its start.

    A solver returns the times it sampled, after a round trip through its own
    units and the precision of jax, so they differ from the times of the span
    in the last digits (by 1 part in 10^7 in single precision).
    """
    return IsfResult(
        times=np.linspace(
            0, time_span.t_end - time_span.t_start, time_span.n_steps + 1
        ),
        isf=result.isf,
        error=result.error,
        delta_k=result.delta_k,
        units=result.units,
    )
