"""The Kramers equation of a periodic system: its ISF, mean square displacement, diffusion constant and slow modes, as sums of modes."""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse

from classical_diffusion._modal import ModalSum
from classical_diffusion._periodic import get_saddle_energy
from classical_diffusion._results import (
    ModeCount,
    RateCut,
    RmsResult,
    SlowModes,
    fold_times,
    get_selection,
    get_unit_direction,
    select_modes,
)
from classical_diffusion.kramers._config import DEFAULT_CONFIG, KramersConfig
from classical_diffusion.kramers._krylov import (
    KrylovProblem,
    get_modal_sum,
    get_slow_pairs,
)
from classical_diffusion.kramers._lu import SparseFactor, factorize, solve
from classical_diffusion.kramers._model import (
    Model,
    build_model,
    get_plane_waves,
    get_plane_waves_within,
)
from classical_diffusion.kramers._overdamped import (
    WEIGHT_FLOOR,
    get_slow_window_modes,
    scaled_delta_k,
)
from classical_diffusion.kramers._real import build_real_operator, get_start
from classical_diffusion.kramers._smoluchowski import get_amplitude, get_grid_size
from classical_diffusion.kramers._stepping import (
    MatrixFreeOperator,
    build_matrix_free_operator,
    propagate_form,
    propagate_state,
    solve_by_propagation,
)
from classical_diffusion.kramers._zero_friction import build as build_orbits
from classical_diffusion.kramers._zero_friction import series_from_model
from classical_diffusion.simulation import IsfResult
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion._results import Selection
    from classical_diffusion.kramers._model import ComplexArray, FloatArray, IntArray
    from classical_diffusion.langevin._system import PeriodicSystem

_SHIFT = 1.0
"""The shift sigma of the shift-invert Arnoldi method, in units of the thermal velocity over the length unit."""
_GROW = 1.3
"""The factor by which the Hermite order and the plane wave radius grow while the basis converges."""


@dataclass(frozen=True, kw_only=True)
class Basis:
    """The Hermite functions up to a total order, and the plane waves up to a radius."""

    max_order: int
    radius: float

    def grow_order(self) -> Basis:
        return Basis(max_order=int(np.ceil(self.max_order * _GROW)), radius=self.radius)

    def grow_radius(self, model: Model) -> Basis:
        """Grow the radius by 1.3, or to the next plane waves if that adds none."""
        radius = self.radius * _GROW
        lengths = np.linalg.norm(
            get_plane_waves_within(
                model, radius + 2 * np.max(np.linalg.norm(model.reciprocal, axis=-1))
            )
            @ model.reciprocal,
            axis=-1,
        )
        beyond = lengths[lengths > self.radius + 1e-9]
        return Basis(max_order=self.max_order, radius=max(radius, float(beyond.min())))


def get_initial_basis(
    model: Model, delta_k: FloatArray, config: KramersConfig
) -> Basis:
    """Get a first guess of the basis which converges the ISF.

    The plane waves resolve e^{-V / 2} to config.amplitude_tol, with a margin
    that grows as gamma^{-1/3}, and |K| more: as the friction falls, the
    distribution is drawn into ever finer filaments in phase space before
    the friction smooths it. The Hermite order is 12 / sqrt(gamma) (in units
    of the thermal velocity over the length unit), and at least 30: in the
    narrowed Hermite functions (`get_width`) the order which converges the
    ISF and D of hydrogen, lithium and xenon in 1D grows as gamma^{-1/2}.
    """
    vectors = get_plane_waves(model, config.amplitude_tol)
    radius = float(np.max(np.linalg.norm(vectors @ model.reciprocal, axis=-1)))
    margin = 1 + 0.15 * model.friction ** (-1 / 3)
    return Basis(
        max_order=int(max(30, 12 / np.sqrt(model.friction))),
        radius=max(radius * margin + float(np.linalg.norm(delta_k)), 1.0),
    )


def get_initial_diffusion_basis(model: Model, config: KramersConfig) -> Basis:
    """Get a first guess of the basis which converges D to config.tol, for a saddle at E_s above the minimum.

    D is a quadratic form, so a component of e^{-V / 2} of size e left out
    of the plane waves errs by about e^2 D_0, which relative to D,
    about e^{-E_s} D_0 over a barrier, is e^2 e^{E_s}: the plane waves hold
    e^{-V / 2} to sqrt(tol e^{-E_s}). As the friction falls, the particles
    near the saddle energy fill the boundary layer of the separatrix, which
    narrows as gamma^{-1/2}: the radius grows by (0.6 E_s + 0.5) / sqrt(gamma),
    and the Hermite order is (4.6 + 4.3 p_s) / sqrt(gamma) + 4, with
    p_s = sqrt(2 E_s) the momentum that crosses the saddle (`get_width`).
    These are the smallest bases that give D to tol / 4 for cosines of
    0.5 to 16 k_B T and hydrogen, lithium, sodium and xenon on Cu(111) and
    Pt(111) in 1D, at 0.1 to 100 times their friction, with 0 to 40 % to
    spare; the basis is checked on D itself (`_converge_basis`).
    """
    saddle, minimum = get_saddle_energy(tuple(model.potential), model.n_dim)
    energy = saddle - minimum
    root = np.sqrt(model.friction)
    vectors = get_plane_waves(model, np.sqrt(config.tol * np.exp(-energy)))
    radius = float(np.max(np.linalg.norm(vectors @ model.reciprocal, axis=-1)))
    return Basis(
        max_order=int(np.ceil((4.6 + 4.3 * np.sqrt(2 * energy)) / root + 4)),
        radius=max(radius + (0.6 * energy + 0.5) / root, 1.0),
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

    def factorize(self) -> SparseFactor:
        """Get the sparse LU of shift - L, whose inverse is the operator of the Arnoldi method."""
        shift = scipy.sparse.identity(self.operator.shape[0], format="csr") * _SHIFT
        return factorize(shift - self.operator)

    def start(self, direction: FloatArray | None = None) -> FloatArray:
        """Get w psi_0(p), or e . p w psi_0(p) for a direction, in the real form (`get_start`)."""
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


def _converge_basis[R](  # ruff:ignore[too-many-arguments]
    model: Model,
    initial: Basis,
    compute_new: Callable[[Basis], R],
    change: Callable[[R, R], float],
    tol: float,
    *,
    max_rounds: int = 12,
) -> tuple[Basis, R, float]:
    """Get the result in a basis whose error, bounded by twice its change as the Hermite order and the plane wave radius each grow by 1.3, is below tol, with that bound.

    Each grows the basis by at least one Hermite order or plane wave
    shell, and the result converges exponentially in both, by far more
    than a factor of two each time once it is near tol, so twice the sum
    of the two changes bounds the error of the smaller basis, which is
    returned. Where the bound is above tol, the direction whose change is
    above tol / 4 grows.

    Raises
    ------
    RuntimeError
        If the basis does not converge within max_rounds.
    """
    known = dict[Basis, R]()

    def compute(basis: Basis) -> R:
        # A basis grown along one direction only was computed in the last round
        if basis not in known:
            known[basis] = compute_new(basis)
        return known[basis]

    basis = initial
    for _ in range(max_rounds):
        result = compute(basis)
        by_order, by_radius = basis.grow_order(), basis.grow_radius(model)
        d_order = change(result, compute(by_order))
        d_radius = change(result, compute(by_radius))
        error = 2 * (d_order + d_radius)
        if error <= tol:
            return basis, result, error
        basis = Basis(
            max_order=by_order.max_order if d_order >= tol / 4 else basis.max_order,
            radius=by_radius.radius if d_radius >= tol / 4 else basis.radius,
        )
    msg = "The basis did not converge"
    raise RuntimeError(msg)


def _get_modes(
    problem: _Problem,
    start: FloatArray,
    times: FloatArray,
    error: Callable[[ModalSum, ModalSum], float],
    tol: float,
) -> ModalSum:
    """Get the correlation [u, e^{L t} u] / <w|w> = u^T J e^{L t} u / <w|w> as a modal sum, by Arnoldi on (shift - L)^-1 (`get_modal_sum`).

    The shift maps the slow modes, close to the origin, to the largest
    eigenvalues of the inverse, which the Arnoldi method finds first, so a
    few tens of solves give the correlation at every time.
    """
    modal = get_modal_sum(
        KrylovProblem(
            operator=problem.factorize(),
            parity=problem.parity,
            rates_of=lambda mu: _SHIFT - 1 / mu,
            span=float(times[-1]),
            error=error,
            tolerance=tol,
        ),
        start,
    )
    return dataclasses.replace(modal, residues=modal.residues / problem.norm)


def _get_isf_modes(
    model: Model, delta_k: FloatArray, times: FloatArray, config: KramersConfig
) -> ModalSum:
    def change(a: ModalSum, b: ModalSum) -> float:
        return float(np.max(np.abs(a.values(times) - b.values(times)), initial=0.0))

    def compute(basis: Basis) -> ModalSum:
        problem = _build_problem(model, basis, delta_k)
        start = problem.start()
        scale = float(start @ start)
        return _get_modes(
            problem,
            start,
            times,
            lambda a, b: change(a, b) / scale,
            1e-2 * config.tol,
        )

    _, modes, _ = _converge_basis(
        model, get_initial_basis(model, delta_k, config), compute, change, config.tol
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
        return _get_isf_modes(model, delta_k, times, config).values(times).real

    basis = get_initial_stepped_basis(model, delta_k, config)
    for _ in range(_MAX_STEPPED_ROUNDS):
        problem = _build_stepped_problem(model, basis, delta_k)
        tails = _Tails(problem, basis, model)
        result = propagate_form(
            problem.operator,
            problem.start(),
            problem.parity,
            times,
            tol=1e-2 * config.tol,
            monitor=tails.update,
        )
        grow_order, grow_radius = tails.excess(config.tol)
        if grow_order == grow_radius == 1:
            return result.values / problem.norm
        basis = Basis(
            max_order=int(np.ceil(basis.max_order * grow_order)),
            radius=basis.radius * grow_radius,
        )
    msg = "The basis did not converge"
    raise RuntimeError(msg)


_MAX_STEPPED_ROUNDS = 12
_TAIL_RATIOS = {1: (3.0, 2.0), 2: (100.0, 20.0)}
"""The Hermite tail and the square of the plane wave tail, over the tolerance, below which the basis is kept, by dimension (`_Tails`)."""
_FAR = 100.0
_GROW_FAR = 1.6


class _Tails:
    """The largest share of the state, over the steps, in the outermost Hermite shells and plane waves.

    The amplitude in the top eighth of the Hermite orders (at least two) and
    in the plane waves beyond 0.85 of the radius estimate the error of
    cutting the basis there, from a single run. For hydrogen, lithium and
    xenon on Cu(111) and Pt(111) in 1D, at 0.1 to 10 times their friction,
    the error of the ISF was below 0.2 times the Hermite tail and 0.3 times
    the square of the plane wave tail, so the basis is kept when these are
    below 3 and 2 times the tolerance, beyond what the start state holds in
    the top shells. In 2D (lithium on Cu(111) at 10 times its friction, to
    order 24 and radius 11) the error was below 3e-3 times the Hermite tail
    and 0.03 times the square of the plane wave tail, so there the limits
    are 100 and 20 times the tolerance (`_TAIL_RATIOS`).
    """

    def __init__(self, problem: _SteppedProblem, basis: Basis, model: Model) -> None:
        order = problem.operator.indices.sum(axis=1)
        top = int(order.max())
        self._top = order >= top - max(2, top // 8)
        vectors = get_plane_waves_within(model, basis.radius)
        lengths = np.linalg.norm(vectors @ model.reciprocal, axis=-1)
        self._outer = lengths >= 0.85 * basis.radius
        self._shape = problem.operator.shape
        self._hermite_ratio, self._plane_wave_ratio = _TAIL_RATIOS[min(model.n_dim, 2)]
        self.hermite = 0.0
        self.plane_wave = 0.0
        # The equilibrium itself reaches the top shells: the narrowed Hermite
        # functions hold the Maxwellian only to its value at their reach,
        # e^{-reach^2 / 4}, whatever their order. Only what the dynamics
        # adds to that is an error of the cut
        self.update(problem.start())
        self._hermite_floor = self.hermite
        self.hermite = 0.0

    def excess(self, tol: float) -> tuple[float, float]:
        """Get the factors by which to grow the Hermite order and plane wave radius: 1 where the tail is within tolerance, 1.3 near it, and 1.6 more than 100 times beyond it.

        The tails fall at least exponentially with the order and radius once
        they resolve the state, so far beyond the tolerance a larger step
        saves runs of the time steps.
        """

        def factor(tail: float, allowed: float) -> float:
            if tail < allowed:
                return 1.0
            return _GROW if tail < _FAR * allowed else _GROW_FAR

        return (
            factor(self.hermite, self._hermite_ratio * tol + 2 * self._hermite_floor),
            factor(self.plane_wave**2, self._plane_wave_ratio * tol),
        )

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


def _check_zero_friction(model: Model) -> None:
    """Raise unless the zero friction (Hamiltonian) solver applies: in 1D only.

    Raises
    ------
    ValueError
        If the system is not 1D.
    """
    if model.n_dim != 1:
        msg = (
            "At zero friction only 1D systems are solved: in 2D the Hamiltonian"
            " flow is chaotic, and no deterministic method reaches the tolerance"
        )
        raise ValueError(msg)


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
    found by Arnoldi on (1 - L)^{-1} (`get_modal_sum`), with
    one sparse LU factorisation. The modes give the ISF at any time at once.
    The basis grows until the ISF changes by less than config.tol / 3.
    Beyond 1D, the LU fills in, and the state is stepped in time instead
    (`propagate_form`).

    At zero friction the dynamics is Hamiltonian, and in 1D the ISF is a sum
    over orbits, found exactly in their action-angle variables at any time
    (`build_orbits`).

    Parameters
    ----------
    system
        The periodic system.
    delta_k
        The scattering wavevector, in the units of the system.
    times
        The times of the ISF, in the units of the system: any real times, in
        any order. The classical ISF is real and even in t, so it is found
        once, at each |t|.
    config
        The convergence, see `KramersConfig`.

    Raises
    ------
    ValueError
        If the times are not a one dimensional array of finite numbers, or
        at zero friction beyond 1D.
    """  # ruff:ignore[docstring-extraneous-exception]
    folded = fold_times(times)
    model = build_model(system)
    natural = system.units.time_into(folded.magnitudes, model.units)
    scaled = scaled_delta_k(system, model, delta_k)
    if model.friction == 0:
        _check_zero_friction(model)
        orbits = build_orbits(
            series_from_model(model), float(scaled[0]), tol=config.tol
        )
        isf = np.real(orbits.isf(natural, tol=config.tol))
    else:
        isf = _get_isf(model, scaled, natural, config)
    return IsfResult(
        times=folded.times,
        isf=folded.hermitian(isf),
        delta_k=delta_k,
        units=system.units,
    )


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
    20 / gamma, or to 20 times the slowest decay time seen where that is
    longer (`get_slow_window_modes`). Beyond 1D, the start is stepped to 4 / gamma, by when the
    fast modes have decayed, and the slow modes are the Ritz pairs of the
    Krylov space of what is left, each with its weight in the ISF from the
    J-symmetry of L, (u^T J r)^2 / (r^T J r) / <w|w>.

    Parameters
    ----------
    system
        The periodic system.
    delta_k
        The scattering wavevector, in the units of the system.
    n_modes
        The number of modes, by default every mode slower than SLOW_RATE of
        the friction, and at least one.
    config
        The convergence, see `KramersConfig`.

    Raises
    ------
    ValueError
        If n_modes is negative, or at zero friction, where the ISF does not
        decay.
    """
    model = build_model(system)
    selection = get_selection(n_modes, model.friction)
    if model.friction == 0:
        msg = "At zero friction the ISF does not decay, and has no slow modes"
        raise ValueError(msg)
    scaled = scaled_delta_k(system, model, delta_k)
    window = 20 / model.friction
    if _is_stepped(model):
        modes = ModalSum.of(
            *_get_stepped_slow_modes(model, scaled, selection, config),
            window,
            config.tol,
        )
    else:
        modes = get_slow_window_modes(
            lambda span: _get_isf_modes(
                model, scaled, np.linspace(0, span, 200), config
            ),
            window,
        )
    rates, weights = modes.kept
    # A Ritz value of a mode the start barely holds can land anywhere, with
    # no weight
    seen = np.abs(weights) > WEIGHT_FLOOR
    rates, weights = select_modes(rates[seen], weights[seen], selection)
    natural = model.units.time_into(1.0, system.units)
    return SlowModes(
        system=system,
        delta_k=tuple(delta_k),
        rates=rates / natural,
        amplitudes=weights,
    )


def _get_stepped_slow_modes(
    model: Model, delta_k: FloatArray, selection: Selection, config: KramersConfig
) -> tuple[ComplexArray, ComplexArray]:
    """Get the slow modes of the ISF by stepping the start past the fast modes, then the Ritz pairs of e^{L / gamma} in the Krylov space of what is left (`get_slow_pairs`).

    Each pair converges to a residual |L r - lambda r| of tol gamma |r|. The
    pairs are converged up to twice the rate of the selection, so that a
    mode just slower than it is not missed while its Ritz value is still
    below it.

    Raises
    ------
    RuntimeError
        If the basis does not converge.
    """
    basis = get_initial_stepped_basis(model, delta_k, config)
    for _ in range(_MAX_STEPPED_ROUNDS):
        problem = _build_stepped_problem(model, basis, delta_k)
        tails = _Tails(problem, basis, model)
        start = problem.start()
        state, _, _ = propagate_state(
            problem.operator,
            start,
            4 / model.friction,
            tol=1e-2 * config.tol,
            monitor=tails.update,
        )
        grow_order, grow_radius = tails.excess(config.tol)
        if grow_order == grow_radius == 1:
            break
        basis = Basis(
            max_order=int(np.ceil(basis.max_order * grow_order)),
            radius=basis.radius * grow_radius,
        )
    else:
        msg = "The basis did not converge"
        raise RuntimeError(msg)
    duration = 1 / model.friction
    values, vectors = get_slow_pairs(
        problem.operator.apply,
        lambda x: propagate_state(problem.operator, x, duration, tol=1e-2 * config.tol)[
            0
        ],
        state,
        duration=duration,
        rate=2 * selection.rate if isinstance(selection, RateCut) else 0.0,
        count=selection.n_modes if isinstance(selection, ModeCount) else 1,
        residual=config.tol * model.friction,
    )
    parity = problem.parity
    overlap = vectors @ (parity * start)
    form = np.sum(vectors * parity * vectors, axis=1)
    return values, overlap**2 / form / problem.norm


_PANEL_NODES = 16
"""The Gauss-Legendre nodes on each panel of the integral of the velocity autocorrelation."""


def _get_doubled_mean_square(
    problem: _SteppedProblem, v: FloatArray, times: FloatArray, tol: float
) -> FloatArray:
    """Get 2 int_0^t (t - s) C(s) ds at the times, with C(t) = v^T e^{L t} v / <w|w> found by `propagate_form`.

    The velocity v is odd in the momentum, so C(t) = -v^T J e^{L t} v, which
    `propagate_form` gets stepping only to t / 2. C is found at Gauss-Legendre
    nodes on panels between the times, no longer than a thermal transit of
    the length unit, and the mean square displacement is
    2 (t int_0^t C - int_0^t s C), each integral a cumulative sum over the
    panels, exact for C a polynomial of degree 31 on each panel.
    """
    nodes, weights = np.polynomial.legendre.leggauss(_PANEL_NODES)
    edges = [0.0]
    for t in times[1:]:
        count = max(1, int(np.ceil((t - edges[-1]) / _PANEL_LENGTH)))
        edges.extend(np.linspace(edges[-1], t, count + 1)[1:].tolist())
    edges_array = np.array(edges)
    middle = 0.5 * (edges_array[1:] + edges_array[:-1])
    half = 0.5 * np.diff(edges_array)
    points = (middle[:, None] + half[:, None] * nodes[None, :]).ravel()
    correlation = (
        -propagate_form(
            problem.operator,
            v,
            problem.parity,
            np.concatenate([[0.0], points]),
            tol=tol,
        ).values[1:]
        / problem.norm
    )
    panel_c = (correlation.reshape(len(middle), -1) * weights).sum(axis=1) * half
    panel_sc = ((correlation * points).reshape(len(middle), -1) * weights).sum(
        axis=1
    ) * half
    cumulative_c = np.concatenate([[0.0], np.cumsum(panel_c)])
    cumulative_sc = np.concatenate([[0.0], np.cumsum(panel_sc)])
    at = np.searchsorted(edges_array, times)
    return 2 * (times * cumulative_c[at] - cumulative_sc[at])


_PANEL_LENGTH = 1.0
"""The longest panel of the integral of the velocity autocorrelation, in units of the transit time of the length unit at the thermal velocity."""


def _get_stepped_diffusion(
    model: Model, basis: Basis, direction: FloatArray, tol: float
) -> float:
    """D = v^T (-L_0)^{-1} v / <w|w>, solved by propagation over 1 / gamma (`solve_by_propagation`)."""
    problem = _build_stepped_problem(model, basis, np.zeros(model.n_dim))
    v = problem.start(direction)
    solution, _ = solve_by_propagation(
        problem.operator, v, 1 / model.friction, tol=1e-2 * tol
    )
    return float(v @ solution) / problem.norm


def _get_diffusion(model: Model, basis: Basis, direction: FloatArray) -> float:
    """D = v^T (-L_0)^{-1} v / <w|w>, by one sparse solve, bordered by the null vector u = w psi_0 to fix its multiple."""
    problem = _build_problem(model, basis, np.zeros(model.n_dim))
    u = problem.start()
    v = problem.start(direction)
    column = scipy.sparse.csr_array(u[:, None])
    bordered = scipy.sparse.block_array(
        [[-problem.operator, column], [column.T, None]], format="coo"
    )
    solution = solve(bordered, np.append(v, 0.0))
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
    if model.friction == 0:
        _check_zero_friction(model)
        # Running orbits move ballistically forever
        return float("inf")
    unit = get_unit_direction(model.n_dim, direction)
    _, diffusion, _ = _converge_basis(
        model,
        get_initial_diffusion_basis(model, config),
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

    Parameters
    ----------
    system
        The periodic system.
    times
        The times, in the units of the system: any real times, in any order.
        The mean square displacement is even in t.
    direction
        The direction along which x is measured, by default the first axis.
    config
        The convergence, see `KramersConfig`.

    Raises
    ------
    ValueError
        If the times are not a one dimensional array of finite numbers, or
        at zero friction beyond 1D.
    """  # ruff:ignore[docstring-extraneous-exception]
    folded = fold_times(times)
    model = build_model(system)
    natural = system.units.time_into(folded.magnitudes, model.units)
    unit = get_unit_direction(model.n_dim, direction)
    length = model.units.length_into(1.0, system.units)
    if len(natural) == 1:
        # Only t = 0
        return RmsResult(
            times=folded.times,
            mean_square=folded.even(np.zeros(1)),
            direction=tuple(unit),
            units=system.units,
        )
    if model.friction == 0:
        _check_zero_friction(model)
        orbits = build_orbits(series_from_model(model), 0.0, tol=config.tol)
        return RmsResult(
            times=folded.times,
            mean_square=folded.even(orbits.msd(natural, tol=config.tol)) * length**2,
            direction=tuple(unit),
            units=system.units,
        )

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
            return _get_doubled_mean_square(
                stepped,
                v,
                natural,
                tol=1e-2 * config.tol * min(1.0, float(natural[1]) ** 2),
            )
        problem = _build_problem(model, basis, np.zeros(model.n_dim))
        # J v = -v, so the modes are of -v^T e^{L t} v
        modes = _get_modes(
            problem,
            problem.start(unit),
            natural,
            lambda a, b: change(
                a.double_integral(natural).real, b.double_integral(natural).real
            ),
            1e-2 * config.tol,
        )
        return -2 * modes.double_integral(natural).real

    _, mean_square, _ = _converge_basis(
        model,
        get_initial_basis(model, np.zeros(model.n_dim), config),
        compute,
        change,
        config.tol,
    )
    return RmsResult(
        times=folded.times,
        mean_square=folded.even(mean_square) * length**2,
        direction=tuple(unit),
        units=system.units,
    )
