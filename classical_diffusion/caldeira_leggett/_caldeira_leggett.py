from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from classical_diffusion.caldeira_leggett._bands import (
    Bands,
    get_bands,
    get_shift_operator,
    get_shifted_kappa,
    get_velocity,
)
from classical_diffusion.caldeira_leggett._config import (
    DEFAULT_CONFIG,
    CaldeiraLeggettConfig,
)
from classical_diffusion.caldeira_leggett._liouvillian import (
    get_couplings,
    get_liouvillian,
    get_offset_operators,
    get_operators,
    get_twist_derivative,
)
from classical_diffusion.caldeira_leggett._model import Model, build_model
from classical_diffusion.caldeira_leggett._solver import (
    get_diffusion_coefficient,
    get_isf,
)
from classical_diffusion.caldeira_leggett._solver import (
    get_steady_state as get_band_steady_state,
)
from classical_diffusion.simulation import IsfResult
from classical_diffusion.util import cached, timed

if TYPE_CHECKING:
    from classical_diffusion.caldeira_leggett._model import ComplexArray, FloatArray
    from classical_diffusion.langevin._system import PeriodicSystem


class SteadyState(NamedTuple):
    """The steady state of the master equation of a periodic system, with the bands it is found in.

    The state is diagonal in the crystal momentum. The bands are only defined
    up to a phase, and up to a rotation where they are degenerate, which can
    differ each time they are found, so the state keeps its own.
    """

    model: Model
    bands: Bands
    density: ComplexArray
    """The density matrix in the basis of the bands, (n_kappa, n_bands, n_bands)."""


def _get_steady_state_path(
    system: PeriodicSystem, *, config: CaldeiraLeggettConfig = DEFAULT_CONFIG
) -> Path:
    return Path(f"caldeira_leggett_steady_state_{hash(system)}_{hash(config)}.npz")


@cached(_get_steady_state_path, default_call="load_or_call_uncached")
@timed
def get_steady_state(
    system: PeriodicSystem, *, config: CaldeiraLeggettConfig = DEFAULT_CONFIG
) -> SteadyState:
    """Get the steady state of the master equation of a periodic system.

    Every ISF, and the diffusion constant, starts from the steady state. By
    default they load it from the cache, if it is there, but do not save it.
    Call ``get_steady_state.call_cached`` once, with the same configuration,
    to save it for them.
    """
    model = build_model(system, config)
    bands = get_bands(model)
    operators = get_operators(bands.energies, get_couplings(model, bands))
    density = get_band_steady_state(operators, bands.energies, config)
    return SteadyState(model=model, bands=bands, density=density)


@timed
def solve_isf(
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    times: FloatArray,
    *,
    config: CaldeiraLeggettConfig = DEFAULT_CONFIG,
) -> IsfResult:
    """Solve the Caldeira-Leggett master equation of a periodic system, and get its ISF.

    The ISF I(delta_k, t) = Tr[e^{i delta_k x} V(t) (rho e^{-i delta_k x})]
    of the steady state rho is propagated in time, with adaptive steps, to the
    increasing times from zero. The solver is deterministic, so the ISF has
    no error.

    The Lindblad operators couple to e^{+-i q x} in place of x, which keeps
    the equation periodic, with the spectral function of the universal
    Lindblad equation as the temperature correction, so that the steady state
    is close to the Gibbs state at any temperature.

    Parameters
    ----------
    system
        The periodic system. Its friction is isotropic.
    delta_k
        The scattering wavevector, in the units of the system.
    times
        The times of the ISF, in the units of the system.
    config
        The convergence parameters, see `CaldeiraLeggettConfig`.

    Raises
    ------
    ValueError
        If the times do not increase from zero.
    """
    times = np.asarray(times, dtype=np.float64)
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        msg = "The times must increase from zero"
        raise ValueError(msg)
    model, bands, rho = get_steady_state.load_or_call_uncached(system, config=config)
    couplings = get_couplings(model, bands)
    operators = get_operators(bands.energies, couplings)
    # dK is split into the nearest point n @ cell_reciprocal and an offset
    # theta. Every operator shifts k by a multiple of cell_reciprocal, so k mod
    # cell_reciprocal is conserved, and rho e^{-i dK x} is in the sector of
    # crystal momentum transfer s = -n. The offset is removed by the gauge
    # transformation X -> X e^{i theta x}, after which the bra evolves under
    # the operators of the Bloch states on the offset grid,
    # e^{-i theta x} O e^{i theta x} (`get_offset_operators`). The ISF is then
    # Tr[e^{i (dK - theta) x} X(t)], so any dK is reached with the same cell.
    scaled = np.asarray(delta_k) * model.length
    n = np.rint(np.linalg.solve(model.cell_reciprocal.T, scaled)).astype(int)
    shift = tuple(-int(i) for i in n)
    offset = scaled - n @ model.cell_reciprocal
    bra = (
        get_offset_operators(
            bands.energies,
            couplings,
            get_velocity(bands, model, offset)
            + 0.5 * offset @ offset * np.eye(bands.energies.shape[1]),
        )
        if np.any(offset)
        else operators
    )
    shifted_kappa = get_shifted_kappa(model, shift)
    # S[kappa] = <kappa - dK, m| e^{-i dK x} |kappa, n>, and the block
    # (kappa - dK, kappa) of rho e^{-i dK x} is rho[kappa - dK] S[kappa]
    scattering = get_shift_operator(bands, shift)
    isf = get_isf(
        get_liouvillian(operators, bra, shifted_kappa),
        rho[shifted_kappa] @ scattering,
        scattering,
        times / model.time,
        config,
    )
    return IsfResult(times=times, isf=isf, delta_k=delta_k, units=system.units)


@timed
def get_diffusion_constant(
    system: PeriodicSystem,
    *,
    direction: tuple[float, ...] | None = None,
    config: CaldeiraLeggettConfig = DEFAULT_CONFIG,
) -> float:
    """Get the diffusion constant of a periodic system under the Caldeira-Leggett master equation.

    The diffusion constant D along a direction is the limit of the dephasing
    rate of the ISF over delta_k^2 as delta_k -> 0. It is found exactly from
    perturbation theory in delta_k about the steady state, with one linear
    solve, rather than from an ISF, which decays ever more slowly as
    delta_k -> 0.

    Parameters
    ----------
    system
        The periodic system. Its friction is isotropic.
    direction
        The direction along which to find D, by default the first axis.
    config
        The convergence parameters, see `CaldeiraLeggettConfig`.

    Returns
    -------
    float
        The diffusion constant, in the units of the system.
    """
    model, bands, rho = get_steady_state.load_or_call_uncached(system, config=config)
    couplings = get_couplings(model, bands)
    operators = get_operators(bands.energies, couplings)
    unit = np.eye(model.n_dim)[0] if direction is None else np.asarray(direction, float)
    velocity = get_velocity(bands, model, unit / np.linalg.norm(unit))
    derivative = get_twist_derivative(bands.energies, couplings, operators, velocity)
    diffusion = get_diffusion_coefficient(operators, derivative, rho, config)
    return diffusion * model.length**2 / model.time
