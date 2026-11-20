"""The Kramers equation of a periodic system: its ISF, mean square displacement, diffusion constant and slow modes, as sums of modes."""

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from classical_diffusion.caldeira_leggett import SLOW_RATE, SlowModes
from classical_diffusion.caldeira_leggett._solver import RmsResult
from classical_diffusion.kramers._config import DEFAULT_CONFIG, KramersConfig
from classical_diffusion.kramers._model import (
    Model,
    build_model,
    get_box_vectors,
    get_plane_waves,
)
from classical_diffusion.kramers._modes import Modes, get_modes
from classical_diffusion.kramers._overdamped import (
    _check_times,
    _direction,
    _scaled_delta_k,
)
from classical_diffusion.kramers._real import build_real_operator, get_start
from classical_diffusion.kramers._smoluchowski import get_amplitude, get_grid_size
from classical_diffusion.simulation import IsfResult
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.kramers._model import FloatArray, IntArray
    from classical_diffusion.langevin._system import PeriodicSystem

_SHIFT = 1.0
"""The shift sigma of the shift-invert Lanczos method, in units of the thermal velocity over the length unit."""
_GROW = 1.3
"""The factor by which the Hermite order and the plane wave radius grow while the basis converges."""


@dataclass(frozen=True, kw_only=True)
class Basis:
    """The Hermite functions up to a total order, and the plane waves up to a radius."""

    max_order: int
    radius: float

    def grow_order(self) -> Basis:
        return Basis(max_order=int(np.ceil(self.max_order * _GROW)), radius=self.radius)

    def grow_radius(self) -> Basis:
        return Basis(max_order=self.max_order, radius=self.radius * _GROW)


def get_plane_waves_within(model: Model, radius: float) -> IntArray:
    """Get the plane waves |G| <= radius, as integer vectors, shortest first."""
    s_min = np.linalg.svd(model.reciprocal, compute_uv=False).min()
    box = get_box_vectors(int(np.ceil(radius / s_min)), model.n_dim)
    lengths = np.linalg.norm(box @ model.reciprocal, axis=-1)
    keep = lengths <= radius + 1e-12
    return box[keep][np.argsort(lengths[keep], kind="stable")]


def get_initial_basis(model: Model, delta_k: FloatArray, config: KramersConfig) -> Basis:
    """Get a first guess of the basis which converges the ISF.

    The plane waves resolve e^{-V / 2} to config.basis_tol, with a margin
    that grows as gamma^{-1/3}, and |K| more: as the friction falls, the
    distribution is drawn into ever finer filaments in phase space before
    the friction smooths it. The Hermite order is 12 / sqrt(gamma) (in units
    of the thermal velocity over the length unit), and at least 30: in the
    narrowed Hermite functions (`get_width`) the order which converges the
    ISF and D of hydrogen, lithium and xenon in 1D grows as gamma^{-1/2}.
    """
    vectors = get_plane_waves(model, config.basis_tol)
    radius = float(np.max(np.linalg.norm(vectors @ model.reciprocal, axis=-1)))
    margin = 1 + 0.15 * model.friction ** (-1 / 3)
    return Basis(
        max_order=int(max(30, 12 / np.sqrt(model.friction))),
        radius=max(radius * margin + float(np.linalg.norm(delta_k)), 1.0),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _Problem:
    """The operator of a basis, with its parity and the equilibrium amplitude."""

    operator: scipy.sparse.csr_array
    parity: FloatArray
    indices: IntArray
    width: float
    """The width of the Hermite functions (`get_width`)."""
    amplitude: FloatArray
    """w = e^{-V / 2} on the plane waves."""

    @property
    def norm(self) -> float:
        """<w|w>."""
        return float(self.amplitude @ self.amplitude)

    @cached_property
    def solve(self) -> Callable[[FloatArray], FloatArray]:
        """(shift - L)^{-1}, by sparse LU."""
        shifted = scipy.sparse.identity(self.operator.shape[0], format="csc") * _SHIFT
        return scipy.sparse.linalg.splu((shifted - self.operator).tocsc()).solve

    def start(self, direction: FloatArray | None = None) -> FloatArray:
        """w psi_0(p), or e . p w psi_0(p) for a direction, in the real form (`get_start`)."""
        return get_start(self.indices, self.width, self.amplitude, direction)


def _build_problem(model: Model, basis: Basis, delta_k: FloatArray) -> _Problem:
    vectors = get_plane_waves_within(model, basis.radius)
    operator, parity, indices, width = build_real_operator(
        model, vectors, basis.max_order, delta_k
    )
    return _Problem(
        operator=operator,
        parity=parity,
        indices=indices,
        width=width,
        amplitude=get_amplitude(model, vectors, get_grid_size(vectors)).real,
    )


def _converge_basis[R](
    initial: Basis,
    compute: Callable[[Basis], R],
    change: Callable[[R, R], float],
    tol: float,
    max_rounds: int = 12,
) -> tuple[Basis, R]:
    """Grow the Hermite order and the plane wave radius until neither changes the result by tol / 3."""
    basis = initial
    result = compute(basis)
    for _ in range(max_rounds):
        by_order = compute(basis.grow_order())
        by_radius = compute(basis.grow_radius())
        d_order, d_radius = change(result, by_order), change(result, by_radius)
        if d_order < tol / 3 and d_radius < tol / 3:
            return (
                (basis.grow_order(), by_order)
                if d_order > d_radius
                else (basis.grow_radius(), by_radius)
            )
        basis = Basis(
            max_order=basis.grow_order().max_order
            if d_order >= tol / 3
            else basis.max_order,
            radius=basis.grow_radius().radius if d_radius >= tol / 3 else basis.radius,
        )
        result = compute(basis)
    msg = "The basis did not converge"
    raise RuntimeError(msg)


def _modes_converged_on(
    times: FloatArray, tol: float
) -> Callable[[Modes, Modes], bool]:
    return lambda a, b: bool(np.max(np.abs(a(times) - b(times))) < tol)


def _get_isf_modes(
    model: Model, delta_k: FloatArray, times: FloatArray, config: KramersConfig
) -> Modes:
    def compute(basis: Basis) -> Modes:
        problem = _build_problem(model, basis, delta_k)
        modes = get_modes(
            problem.solve,
            problem.start(),
            problem.parity,
            _SHIFT,
            converged=_modes_converged_on(times, 1e-2 * config.tol),
        )
        return Modes(
            rates=modes.rates,
            weights=modes.weights / problem.norm,
            n_solves=modes.n_solves,
        )

    _, modes = _converge_basis(
        get_initial_basis(model, delta_k, config),
        compute,
        lambda a, b: float(np.max(np.abs(a(times) - b(times)))),
        config.tol,
    )
    return modes


@timed
def solve_isf(
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    times: FloatArray,
    *,
    config: KramersConfig = DEFAULT_CONFIG,
) -> IsfResult:
    """Get the ISF of a periodic system under the Langevin equation, from its Kramers equation.

    The phase space density at K, e^{-i K x} h(x, p) with h periodic, is
    expanded in Hermite functions of the momentum and plane waves of one
    unit cell. In the frame symmetrised by the equilibrium, the Kramers
    operator L is real and J-symmetric, with J the parity of the momentum,
    so the ISF, <w| e^{L t} |w> / <w|w> for w = e^{-V / 2}, is a sum of modes
    found by Lanczos on (1 - L)^{-1} in the form x^T J y (`get_modes`), with
    one sparse LU factorisation. The modes give the ISF at any time at once.
    The basis grows until the ISF changes by less than config.tol / 3.

    Parameters
    ----------
    system
        The periodic system.
    delta_k
        The scattering wavevector, in the units of the system.
    times
        The times of the ISF, in the units of the system, increasing from zero.
    config
        The convergence, see `KramersConfig`.
    """
    times = _check_times(times)
    model = build_model(system)
    natural = system.units.time_into(times, model.units)
    modes = _get_isf_modes(
        model, _scaled_delta_k(system, model, delta_k), natural, config
    )
    return IsfResult(times=times, isf=modes(natural), delta_k=delta_k, units=system.units)


@timed
def get_slow_modes(
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    n_modes: int | None = None,
    *,
    config: KramersConfig = DEFAULT_CONFIG,
) -> SlowModes:
    """Get the slowest modes of the ISF of a periodic system at dK, I(t) = sum_j a_j exp(lambda_j t) once the faster modes have decayed.

    The modes are those of `solve_isf`, converged over the times to 20 /
    gamma, by default every mode which decays more slowly than SLOW_RATE of
    the friction, and at least one.
    """
    model = build_model(system)
    window = np.linspace(0, 20 / model.friction, 200)
    modes = _get_isf_modes(
        model, _scaled_delta_k(system, model, delta_k), window, config
    )
    order = np.argsort(-modes.rates.real)
    rates, weights = modes.rates[order], modes.weights[order]
    if n_modes is None:
        n_modes = max(1, int(np.count_nonzero(rates.real > -SLOW_RATE * model.friction)))
    natural = model.units.time_into(1.0, system.units)
    return SlowModes(
        system=system,
        delta_k=tuple(delta_k),
        rates=rates[:n_modes] / natural,
        amplitudes=weights[:n_modes],
    )


def _get_diffusion(model: Model, basis: Basis, direction: FloatArray) -> float:
    """D = v^T (-L_0)^{-1} v / <w|w>, by one sparse solve, bordered by the null vector u = w psi_0 to fix its multiple."""
    problem = _build_problem(model, basis, np.zeros(model.n_dim))
    u = problem.start()
    v = problem.start(direction)
    column = scipy.sparse.csr_array(u[:, None])
    bordered = scipy.sparse.block_array(
        [[-problem.operator, column], [column.T, None]], format="csc"
    )
    solution = scipy.sparse.linalg.spsolve(bordered, np.append(v, 0.0))
    return float(v @ solution[:-1]) / problem.norm


@timed
def get_diffusion_constant(
    system: PeriodicSystem,
    *,
    direction: tuple[float, ...] | None = None,
    config: KramersConfig = DEFAULT_CONFIG,
) -> float:
    """Get the diffusion constant of a periodic system under the Langevin equation, from its Kramers equation.

    D = int_0^inf <v(t) v(0)> dt along the direction, which in the
    symmetrised frame is v^T (-L_0)^{-1} v / <w|w>, with v the equilibrium
    amplitude w in the Hermite functions of order one: one sparse solve. The
    basis grows until D changes by less than config.tol of itself.

    Returns
    -------
    float
        The diffusion constant, in the units of the system.
    """
    model = build_model(system)
    unit = _direction(model, direction)
    _, diffusion = _converge_basis(
        get_initial_basis(model, np.zeros(model.n_dim), config),
        lambda basis: _get_diffusion(model, basis, unit),
        lambda a, b: abs(a - b) / abs(b),
        config.tol,
    )
    length = model.units.length_into(1.0, system.units)
    return diffusion * length**2 / model.units.time_into(1.0, system.units)


@timed
def solve_rms(
    system: PeriodicSystem,
    times: FloatArray,
    *,
    direction: tuple[float, ...] | None = None,
    config: KramersConfig = DEFAULT_CONFIG,
) -> RmsResult:
    """Get the mean square displacement of a periodic system along a direction under the Langevin equation, from its Kramers equation.

    <(x(t) - x(0))^2> = 2 int_0^t (t - u) <v(u) v(0)> du, with the velocity
    autocorrelation v^T e^{L_0 t} v / <w|w> as a sum of modes (`get_modes`),
    each integrated exactly. The modes and basis converge until the mean
    square displacement changes by less than config.tol of its value.

    Raises
    ------
    ValueError
        If the times do not increase from zero.
    """
    times = _check_times(times)
    model = build_model(system)
    natural = system.units.time_into(times, model.units)
    unit = _direction(model, direction)
    def mean_square(modes: Modes) -> FloatArray:
        return 2 * modes.double_integral(natural)

    def change(a: Modes, b: Modes) -> float:
        # Relative to the mean square displacement itself, which starts as
        # t^2 and can level off in a well, far below 2 k_B T t / (m gamma)
        reference = mean_square(b)
        scale = np.maximum(np.abs(reference), 1e-300)
        return float(np.max(np.abs(mean_square(a) - reference)[1:] / scale[1:]))

    def compute(basis: Basis) -> Modes:
        problem = _build_problem(model, basis, np.zeros(model.n_dim))
        v = problem.start(unit)
        modes = get_modes(
            problem.solve,
            v,
            problem.parity,
            _SHIFT,
            converged=lambda a, b: change(a, b) < 1e-2 * config.tol,
        )
        # J v = -v, so the modes are of -v^T e^{L t} v
        return Modes(
            rates=modes.rates,
            weights=-modes.weights / problem.norm,
            n_solves=modes.n_solves,
        )

    _, modes = _converge_basis(
        get_initial_basis(model, np.zeros(model.n_dim), config),
        compute,
        change,
        config.tol,
    )
    length = model.units.length_into(1.0, system.units)
    return RmsResult(
        times=times,
        mean_square=mean_square(modes) * length**2,
        direction=tuple(unit),
        units=system.units,
    )
