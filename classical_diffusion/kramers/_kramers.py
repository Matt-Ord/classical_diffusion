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
    get_plane_waves,
    get_plane_waves_within,
)
from classical_diffusion.kramers._modes import Modes, get_modes
from classical_diffusion.kramers._overdamped import (
    _check_times,
    _direction,
    _scaled_delta_k,
)
from classical_diffusion.kramers._real import build_real_operator, get_start
from classical_diffusion.kramers._smoluchowski import get_amplitude, get_grid_size
from classical_diffusion.kramers._stepping import (
    MatrixFreeOperator,
    build_matrix_free_operator,
    propagate_form,
    get_ritz_pairs,
    propagate_integrals,
    propagate_state,
    solve_by_propagation,
)
from classical_diffusion.simulation import IsfResult
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.kramers._model import ComplexArray, FloatArray, IntArray
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


def get_initial_basis(
    model: Model, delta_k: FloatArray, config: KramersConfig
) -> Basis:
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


@dataclass(frozen=True, kw_only=True, eq=False)
class _SteppedProblem:
    """The matrix-free operator of a basis, for systems in more than one dimension, whose LU does not fit."""

    operator: MatrixFreeOperator
    amplitude: FloatArray

    @property
    def norm(self) -> float:
        return float(self.amplitude @ self.amplitude)

    @property
    def parity(self) -> FloatArray:
        return np.repeat(self.operator.parity, self.operator.shape[1])

    def start(self, direction: FloatArray | None = None) -> FloatArray:
        return get_start(
            self.operator.indices, self.operator.width, self.amplitude, direction
        )


def _build_stepped_problem(
    model: Model, basis: Basis, delta_k: FloatArray
) -> _SteppedProblem:
    vectors = get_plane_waves_within(model, basis.radius)
    return _SteppedProblem(
        operator=build_matrix_free_operator(model, vectors, basis.max_order, delta_k),
        amplitude=get_amplitude(model, vectors, get_grid_size(vectors)).real,
    )


def _is_stepped(model: Model) -> bool:
    """Whether to step in time rather than factorise: the sparse LU fills in beyond one dimension."""
    return model.n_dim > 1


def _get_isf(
    model: Model, delta_k: FloatArray, times: FloatArray, config: KramersConfig
) -> FloatArray:
    if not _is_stepped(model):
        return _get_isf_modes(model, delta_k, times, config)(times)

    basis = get_initial_stepped_basis(model, delta_k, config)
    for _ in range(_MAX_STEPPED_ROUNDS):
        problem = _build_stepped_problem(model, basis, delta_k)
        tails = _Tails(problem, basis, model)
        result = propagate_form(
            problem.operator.apply,
            problem.start(),
            problem.parity,
            times,
            tol=1e-2 * config.tol,
            monitor=tails.update,
        )
        grow_order = tails.hermite >= 3 * config.tol
        grow_radius = tails.plane_wave**2 >= 2 * config.tol
        if not (grow_order or grow_radius):
            return result.values / problem.norm
        basis = Basis(
            max_order=basis.grow_order().max_order if grow_order else basis.max_order,
            radius=basis.grow_radius().radius if grow_radius else basis.radius,
        )
    msg = "The basis did not converge"
    raise RuntimeError(msg)


_MAX_STEPPED_ROUNDS = 12


class _Tails:
    """The largest share of the state, over the steps, in the outermost Hermite shells and plane waves.

    The amplitude in the top eighth of the Hermite orders (at least two) and
    in the plane waves beyond 0.85 of the radius estimate the error of
    cutting the basis there, from a single run. For hydrogen, lithium and
    xenon on Cu(111) and Pt(111) in 1D, at 0.1 to 10 times their friction,
    the error of the ISF was below 0.2 times the Hermite tail and 0.3 times
    the square of the plane wave tail, so the basis is kept when these are
    below 3 and 2 times the tolerance.
    """

    def __init__(self, problem: _SteppedProblem, basis: Basis, model: Model) -> None:
        order = problem.operator.indices.sum(axis=1)
        top = int(order.max())
        self._top = order >= top - max(2, top // 8)
        vectors = get_plane_waves_within(model, basis.radius)
        lengths = np.linalg.norm(vectors @ model.reciprocal, axis=-1)
        self._outer = lengths >= 0.85 * basis.radius
        self._shape = problem.operator.shape
        self.hermite = 0.0
        self.plane_wave = 0.0

    def update(self, state: FloatArray) -> None:
        x = state.reshape(self._shape)
        total = float(np.sum(x**2))
        if total == 0:
            return
        self.hermite = max(self.hermite, np.sqrt(np.sum(x[self._top] ** 2) / total))
        self.plane_wave = max(
            self.plane_wave, np.sqrt(np.sum(x[:, self._outer] ** 2) / total)
        )


def get_initial_stepped_basis(
    model: Model, delta_k: FloatArray, config: KramersConfig
) -> Basis:
    """Get a first guess of the basis for time steps: the Hermite order of `get_initial_basis`, and the plane waves which hold e^{-V / 2} to config.tol, and |K| more.

    Each run of the time steps checks its own basis (`_Tails`), so the
    guess starts small and grows only where it must.
    """
    vectors = get_plane_waves(model, config.tol)
    radius = float(np.max(np.linalg.norm(vectors @ model.reciprocal, axis=-1)))
    return Basis(
        max_order=get_initial_basis(model, delta_k, config).max_order,
        radius=max(radius + float(np.linalg.norm(delta_k)), 1.0),
    )


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
    isf = _get_isf(model, _scaled_delta_k(system, model, delta_k), natural, config)
    return IsfResult(times=times, isf=isf, delta_k=delta_k, units=system.units)


@timed
def get_slow_modes(
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    n_modes: int | None = None,
    *,
    config: KramersConfig = DEFAULT_CONFIG,
) -> SlowModes:
    """Get the slowest modes of the ISF of a periodic system at dK, I(t) = sum_j a_j exp(lambda_j t) once the faster modes have decayed.

    In 1D the modes are those of `solve_isf`, converged over the times to
    20 / gamma. Beyond 1D, the start is stepped to 4 / gamma, by when the
    fast modes have decayed, and the slow modes are the Ritz pairs of the
    Krylov space of what is left, each with its weight in the ISF from the
    J-symmetry of L, (u^T J r)^2 / (r^T J r) / <w|w>. By default every mode
    which decays more slowly than SLOW_RATE of the friction, and at least
    one.
    """
    model = build_model(system)
    scaled = _scaled_delta_k(system, model, delta_k)
    if _is_stepped(model):
        rates, weights = _get_stepped_slow_modes(model, scaled, config)
    else:
        window = np.linspace(0, 20 / model.friction, 200)
        modes = _get_isf_modes(model, scaled, window, config)
        rates, weights = modes.rates, modes.weights
    # A Ritz value of a mode the start barely holds can land right of the
    # axis, or anywhere, with no weight
    real = (rates.real <= 0) & (np.abs(weights) > 1e-12)
    rates, weights = rates[real], weights[real]
    order = np.argsort(-rates.real)
    rates, weights = rates[order], weights[order]
    if n_modes is None:
        n_modes = max(
            1, int(np.count_nonzero(rates.real > -SLOW_RATE * model.friction))
        )
    natural = model.units.time_into(1.0, system.units)
    return SlowModes(
        system=system,
        delta_k=tuple(delta_k),
        rates=rates[:n_modes] / natural,
        amplitudes=weights[:n_modes],
    )


_RITZ_VECTORS = 80
_RITZ_RTOL = 1e-4
"""The residual of a Ritz pair, in units of the friction, above which it is not a mode: those kept have rates good to about its square."""


def _get_stepped_slow_modes(
    model: Model, delta_k: FloatArray, config: KramersConfig
) -> tuple[ComplexArray, ComplexArray]:
    """Get the slow modes of the ISF by stepping the start past the fast modes, then the Ritz pairs of what is left."""
    basis = get_initial_stepped_basis(model, delta_k, config)
    for _ in range(_MAX_STEPPED_ROUNDS):
        problem = _build_stepped_problem(model, basis, delta_k)
        tails = _Tails(problem, basis, model)
        start = problem.start()
        state, _, _ = propagate_state(
            problem.operator.apply,
            start,
            4 / model.friction,
            tol=1e-2 * config.tol,
            monitor=tails.update,
        )
        grow_order = tails.hermite >= 3 * config.tol
        grow_radius = tails.plane_wave**2 >= 2 * config.tol
        if not (grow_order or grow_radius):
            break
        basis = Basis(
            max_order=basis.grow_order().max_order if grow_order else basis.max_order,
            radius=basis.grow_radius().radius if grow_radius else basis.radius,
        )
    else:
        msg = "The basis did not converge"
        raise RuntimeError(msg)
    values, vectors, residuals = get_ritz_pairs(
        problem.operator.apply, state, _RITZ_VECTORS
    )
    keep = residuals <= _RITZ_RTOL * model.friction
    values, vectors = values[keep], vectors[keep]
    parity = problem.parity
    overlap = vectors @ (parity * start)
    form = np.sum(vectors * parity * vectors, axis=1)
    return values, overlap**2 / form / problem.norm


def _get_stepped_diffusion(
    model: Model, basis: Basis, direction: FloatArray, tol: float
) -> float:
    """D = v^T (-L_0)^{-1} v / <w|w>, solved by propagation over 1 / gamma (`solve_by_propagation`)."""
    problem = _build_stepped_problem(model, basis, np.zeros(model.n_dim))
    v = problem.start(direction)
    solution, _ = solve_by_propagation(
        problem.operator.apply, v, 1 / model.friction, tol=1e-2 * tol
    )
    return float(v @ solution) / problem.norm


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
        (lambda basis: _get_stepped_diffusion(model, basis, unit, config.tol))
        if _is_stepped(model)
        else (lambda basis: _get_diffusion(model, basis, unit)),
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

    def change(a: FloatArray, b: FloatArray) -> float:
        # Relative to the mean square displacement itself, which starts as
        # t^2 and can level off in a well, far below 2 k_B T t / (m gamma)
        scale = np.maximum(np.abs(b), 1e-300)
        return float(np.max(np.abs(a - b)[1:] / scale[1:]))

    def compute(basis: Basis) -> FloatArray:
        if _is_stepped(model):
            stepped = _build_stepped_problem(model, basis, np.zeros(model.n_dim))
            v = stepped.start(unit)
            # The error of the steps is relative to |v|^2 = <w|w> <v^2>, and the
            # mean square displacement grows from t^2, so the tolerance is
            # scaled by the shortest time
            result = propagate_integrals(
                stepped.operator.apply,
                v,
                v,
                natural,
                tol=1e-2 * config.tol * min(1.0, float(natural[1]) ** 2),
            )
            return 2 * result.values[:, 2] / stepped.norm
        problem = _build_problem(model, basis, np.zeros(model.n_dim))
        v = problem.start(unit)
        modes = get_modes(
            problem.solve,
            v,
            problem.parity,
            _SHIFT,
            converged=lambda a, b: (
                change(-2 * a.double_integral(natural), -2 * b.double_integral(natural))
                < 1e-2 * config.tol
            ),
        )
        # J v = -v, so the modes are of -v^T e^{L t} v
        return -2 * modes.double_integral(natural) / problem.norm

    _, mean_square = _converge_basis(
        get_initial_basis(model, np.zeros(model.n_dim), config),
        compute,
        change,
        config.tol,
    )
    length = model.units.length_into(1.0, system.units)
    return RmsResult(
        times=times,
        mean_square=mean_square * length**2,
        direction=tuple(unit),
        units=system.units,
    )
