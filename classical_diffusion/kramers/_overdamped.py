# cspell: ignore btol lsqr tocsr unconverged vdot
"""The overdamped limit of the Langevin equation, the Smoluchowski equation, solved by Lanczos on its symmetrised generator."""

import dataclasses
import operator
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from classical_diffusion._results import (
    RmsResult,
    SlowModes,
    fold_times,
    get_selection,
    get_unit_direction,
    select_modes,
)
from classical_diffusion.kramers._config import DEFAULT_CONFIG, KramersConfig
from classical_diffusion.kramers._lanczos import LanczosProblem, get_modal_sum
from classical_diffusion.kramers._model import (
    Model,
    build_model,
    get_plane_waves,
    get_plane_waves_within,
    get_rows,
)
from classical_diffusion.kramers._smoluchowski import (
    get_amplitude,
    get_extended_vectors,
    get_gradient_operator,
    get_grid_size,
)
from classical_diffusion.simulation import IsfResult
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion._modal import ModalSum
    from classical_diffusion.kramers._model import ComplexArray, FloatArray, IntArray
    from classical_diffusion.langevin._system import PeriodicSystem


class _Generator:
    """-L = D_0 A_+^dag A_+ at K, on the plane waves, as a sparse operator, with the equilibrium amplitude w."""

    def __init__(self, model: Model, vectors: IntArray, delta_k: FloatArray) -> None:
        dense = get_gradient_operator(model, vectors, delta_k)
        self.gradient = [scipy.sparse.csr_array(a) for a in dense]
        self.adjoint = [a.conj().T.tocsr() for a in self.gradient]
        self.diffusion = 1 / model.friction
        self.vectors = vectors
        self.model = model
        self.amplitude = get_amplitude(model, vectors, get_grid_size(vectors))
        self.norm = float(np.vdot(self.amplitude, self.amplitude).real)

    def apply(self, v: ComplexArray) -> ComplexArray:
        return self.diffusion * sum(
            (adj @ (a @ v) for a, adj in zip(self.gradient, self.adjoint, strict=True)),
            start=np.zeros_like(v),
        )

    def deflated(self) -> Callable[[ComplexArray], ComplexArray]:
        """-L on the complement of w, its null vector at K = 0."""
        w = self.amplitude / np.sqrt(self.norm)

        def project(v: ComplexArray) -> ComplexArray:
            return v - w * np.vdot(w, v)

        return lambda v: project(self.apply(project(v)))

    def force_amplitude(self, direction: FloatArray) -> ComplexArray:
        """(e . grad V) w = -2 e . grad w, in the plane waves."""
        k = self.vectors @ self.model.reciprocal @ direction
        return -2j * k * self.amplitude


def scaled_delta_k(
    system: PeriodicSystem, model: Model, delta_k: tuple[float, ...]
) -> FloatArray:
    """Get dK in the natural units of the model."""
    return np.asarray(delta_k, dtype=float) / system.units.length_into(1.0, model.units)


_BLOCK = 8
"""The Lanczos steps between evaluations of the modal sum."""


def _get_modes(  # ruff:ignore[too-many-arguments]
    apply: Callable[[ComplexArray], ComplexArray],
    start: ComplexArray,
    norm: float,
    times: FloatArray,
    *,
    error: Callable[[ModalSum, ModalSum], float],
    tol: float,
) -> ModalSum:
    """Get <u| e^{L t} |u> / norm as a modal sum, by Lanczos on the Hermitian -L (`get_modal_sum`)."""
    modal = get_modal_sum(
        LanczosProblem(
            apply=apply,
            parity=None,
            rates_of=operator.neg,
            span=float(times[-1]),
            error=error,
            tolerance=tol,
            block=_BLOCK,
        ),
        start,
    )
    return dataclasses.replace(modal, residues=modal.residues / norm)


_MAX_ROUNDS = 12
_GROW = 1.3


def _converge_radius[R](
    model: Model,
    config: KramersConfig,
    compute: Callable[[IntArray], R],
    change: Callable[[R, R], float],
) -> R:
    """Grow the radius of the plane waves by 1.3, from those which hold e^{-V / 2} to config.amplitude_tol, until the result changes by less than config.tol / 3, and get the result in the larger.

    The check is on the result itself, so whatever the first radius, the
    result is not returned unconverged.

    Raises
    ------
    RuntimeError
        If the result does not converge within _MAX_ROUNDS.
    """
    vectors = get_plane_waves(model, config.amplitude_tol)
    radius = max(
        float(np.max(np.linalg.norm(vectors @ model.reciprocal, axis=-1))), 1.0
    )
    previous = compute(get_plane_waves_within(model, radius))
    for _ in range(_MAX_ROUNDS):
        radius *= _GROW
        current = compute(get_plane_waves_within(model, radius))
        if change(previous, current) < config.tol / 3:
            return current
        previous = current
    msg = (
        "The overdamped limit did not converge in the plane waves: e^{-V / 2}"
        " varies over too many orders of magnitude for them to hold it at the saddle"
    )
    raise RuntimeError(msg)


def _get_isf_modes(
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    times: FloatArray,
    config: KramersConfig,
) -> ModalSum:
    """Get the ISF as a modal sum, converged until it changes by less than config.tol / 10 at the times, in plane waves which change it by less than config.tol / 3 (`_converge_radius`)."""
    model = build_model(system)
    scaled = scaled_delta_k(system, model, delta_k)

    def compute(vectors: IntArray) -> ModalSum:
        generator = _Generator(model, vectors, scaled)
        return _get_modes(
            generator.apply,
            generator.amplitude,
            generator.norm,
            times,
            error=lambda a, b: (
                float(np.max(np.abs(a.values(times) - b.values(times))))
                / generator.norm
            ),
            tol=0.1 * config.tol,
        )

    return _converge_radius(
        model,
        config,
        compute,
        lambda a, b: float(np.max(np.abs(a.values(times) - b.values(times)))),
    )


@timed
def solve_isf_overdamped(
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    times: FloatArray,
    *,
    config: KramersConfig = DEFAULT_CONFIG,
) -> IsfResult:
    """Get the ISF of a periodic system in the overdamped limit of the Langevin equation, the Smoluchowski equation.

    The position diffuses with D_0 = k_B T / (m gamma) in the force of the
    potential. In the symmetrised frame, where a density is e^{-V / 2} times
    its amplitude, the generator at K is -D_0 A^dag A, with
    A = grad - i K + grad V / 2, Hermitian, so the ISF
    I(K, t) = <w| e^{L t} |w> / <w|w>, w = e^{-V / 2}, is a sum of decaying
    exponentials. Their rates and weights are the Gauss quadrature of the
    spectral measure of -L in w, by the Lanczos method (`get_modal_sum`),
    extended until the ISF at the times changes by less than config.tol / 10
    over two blocks of steps.

    Parameters
    ----------
    system
        The periodic system.
    delta_k
        The scattering wavevector, in the units of the system.
    times
        The times of the ISF, in the units of the system: any real times, in
        any order. The ISF is real and even in t.
    config
        The convergence, see `KramersConfig`.

    Raises
    ------
    ValueError
        If the times are not a one dimensional array of finite numbers.
    """  # ruff:ignore[docstring-extraneous-exception]
    folded = fold_times(times)
    model = build_model(system)
    natural = system.units.time_into(folded.magnitudes, model.units)
    isf = _get_isf_modes(system, delta_k, natural, config).values(natural).real
    return IsfResult(
        times=folded.times,
        isf=folded.hermitian(isf),
        delta_k=delta_k,
        units=system.units,
    )


@timed
def get_slow_modes_overdamped(
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    n_modes: int | None = None,
    *,
    config: KramersConfig = DEFAULT_CONFIG,
) -> SlowModes:
    """Get the slowest modes of the overdamped ISF of a periodic system at dK, I(t) = sum_j a_j exp(lambda_j t).

    The modes are those of `solve_isf_overdamped`, converged at the times
    0 to 100 / (D_0 |G|^2) of the shortest reciprocal vector G.

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
        If n_modes is negative.
    """  # ruff:ignore[docstring-extraneous-exception]
    model = build_model(system)
    selection = get_selection(n_modes, model.friction)
    g = float(np.min(np.linalg.norm(model.reciprocal, axis=-1)))
    window = np.linspace(0, 100 * model.friction / g**2, 400)
    rates, amplitudes = select_modes(
        *_get_isf_modes(system, delta_k, window, config).kept, selection
    )
    natural = model.units.time_into(1.0, system.units)
    return SlowModes(
        system=system,
        delta_k=tuple(delta_k),
        rates=rates / natural,
        amplitudes=amplitudes,
    )


@timed
def get_diffusion_constant_overdamped(
    system: PeriodicSystem,
    *,
    direction: tuple[float, ...] | None = None,
    config: KramersConfig = DEFAULT_CONFIG,
) -> float:
    """Get the diffusion constant of a periodic system in the overdamped limit of the Langevin equation.

    D = D_0 - D_0^2 int_0^inf <F(t) F(0)> dt / (k_B T)^2 along the direction e,
    F = e . grad V, found as a least squares residual, a sum of squares, so
    that it is accurate relative to itself however far below D_0
    (`_get_diffusion_ratio`).

    Parameters
    ----------
    system
        The periodic system.
    direction
        The direction along which to find D, by default the first axis.
    config
        The convergence, see `KramersConfig`.

    Returns
    -------
    float
        The diffusion constant, in the units of the system.
    """
    model = build_model(system)
    ratio = _get_diffusion_ratio(
        model, get_unit_direction(model.n_dim, direction), config
    )
    length = model.units.length_into(1.0, system.units)
    return (
        ratio * length**2 / (model.friction * model.units.time_into(1.0, system.units))
    )


_LSQR_TOL = 1e-10
"""The relative residual of the normal equations at which LSQR stops (`_get_diffusion_ratio`): the error of the least squares residual is quadratic in that of the solution."""


def _get_diffusion_ratio(
    model: Model, unit: FloatArray, config: KramersConfig
) -> float:
    """Get D / D_0 = min_u |e w - A u|^2 / <w|w>, with A = grad + grad V / 2 at K = 0, growing the plane waves until it converges.

    D = D_0 - D_0^2 <F w| (-L)^{-1} |F w> / <w|w> with F w = A^dag (e w),
    and -L = D_0 A^dag A, so D / D_0 is the norm of the part of e w outside
    the range of A, over <w|w>: the least squares residual, a sum of
    squares. It is accurate relative to itself even for deep barriers,
    where D is many orders below D_0 and the difference of the two terms
    would be lost to round off: for lithium across the rows of Cu(110),
    D = 5.9e-15 D_0. The least squares problem is solved by LSQR, and the
    residual e w - A u formed explicitly, in plane waves which change D by
    less than config.tol / 3 of itself (`_converge_radius`).

    Raises
    ------
    RuntimeError
        If D does not converge in the plane waves.
    """  # ruff:ignore[docstring-extraneous-exception]

    def compute(vectors: IntArray) -> float:
        gradient = get_gradient_operator(model, vectors, np.zeros(model.n_dim))
        w = get_amplitude(model, vectors, get_grid_size(vectors))
        # e w, on the rows of the gradient: the inputs come first in its
        # output plane waves only up to their order, so place it by index
        extended = get_extended_vectors(model, vectors)
        index = {tuple(int(i) for i in v): k for k, v in enumerate(get_rows(extended))}
        rows = np.array([index[tuple(int(i) for i in v)] for v in get_rows(vectors)])
        field = np.zeros((model.n_dim, len(extended)), dtype=np.complex128)
        field[:, rows] = unit[:, None] * w[None, :]
        field = field.reshape(-1)
        stacked = scipy.sparse.csr_array(gradient.reshape(-1, len(vectors)))
        solution = scipy.sparse.linalg.lsqr(
            stacked,
            field,
            atol=_LSQR_TOL,
            btol=_LSQR_TOL,
            conlim=0,
            iter_lim=20 * len(vectors),
        )[0]
        residual = float(np.sum(np.abs(field - stacked @ solution) ** 2))
        return residual / float(np.vdot(w, w).real)

    return _converge_radius(model, config, compute, lambda a, b: abs(a - b) / abs(b))


def _mean_square(
    modes: ModalSum, d0: float, diffusion: float, times: FloatArray
) -> FloatArray:
    """2 D t + 2 D_0^2 sum_j w_j (1 - e^{-l_j t}) / l_j^2, for the modes of the force correlation, w_j e^{-l_j t}.

    This is 2 D_0 t - 2 D_0^2 sum_j w_j [t / l_j - (1 - e^{-l_j t}) / l_j^2]
    with D = D_0 - D_0^2 sum_j w_j / l_j, written as a sum of positive terms.
    A pole at l_j = 0 is the null vector w of -L, which the start does not
    hold, seen through round off.
    """
    rates, residues = modes.kept
    nodes = -rates.real
    keep = nodes > 0
    nodes, weights = nodes[keep], residues[keep].real
    memory = (-np.expm1(-nodes * times[:, None])) / nodes**2
    return 2 * diffusion * times + 2 * d0**2 * (memory @ weights)


@timed
def solve_rms_overdamped(
    system: PeriodicSystem,
    times: FloatArray,
    *,
    direction: tuple[float, ...] | None = None,
    config: KramersConfig = DEFAULT_CONFIG,
) -> RmsResult:
    """Get the mean square displacement of a periodic system along a direction, in the overdamped limit of the Langevin equation.

    <(x(t) - x(0))^2> = 2 D_0 t - 2 D_0^2 int_0^t (t - u) <F(u) F(0)> du,
    with the force correlation <F w| e^{L t} |F w> / <w|w> at K = 0 as a
    modal sum (`get_modal_sum`, on -L deflated of its null vector w), which
    is 2 D t + 2 D_0^2 sum_j w_j (1 - e^{-l_j t}) / l_j^2 in its rates l_j
    and weights w_j, with D from `get_diffusion_constant_overdamped`: a sum
    of positive terms, accurate however far D is below D_0. The modal sum is
    extended until the mean square displacement at the times changes by less
    than config.tol / 10 of itself over two blocks of steps.

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
        If the times are not a one dimensional array of finite numbers.
    """  # ruff:ignore[docstring-extraneous-exception]
    folded = fold_times(times)
    model = build_model(system)
    natural = system.units.time_into(folded.magnitudes, model.units)
    unit = get_unit_direction(model.n_dim, direction)
    length = model.units.length_into(1.0, system.units)
    d0 = 1 / model.friction
    diffusion = d0 * _get_diffusion_ratio(model, unit, config)
    if len(natural) == 1:
        # Only t = 0
        mean_square = 2 * diffusion * natural
    else:

        def relative(a: FloatArray, b: FloatArray) -> float:
            return float(np.max(np.abs(a - b)[1:] / np.abs(b[1:])))

        def compute(vectors: IntArray) -> FloatArray:
            generator = _Generator(model, vectors, np.zeros(model.n_dim))
            force = generator.force_amplitude(unit)
            w = generator.amplitude / np.sqrt(generator.norm)
            force -= w * np.vdot(w, force)

            def scaled(m: ModalSum) -> ModalSum:
                return dataclasses.replace(m, residues=m.residues / generator.norm)

            modes = _get_modes(
                generator.deflated(),
                force,
                generator.norm,
                natural,
                error=lambda a, b: relative(
                    _mean_square(scaled(a), d0, diffusion, natural),
                    _mean_square(scaled(b), d0, diffusion, natural),
                ),
                tol=0.1 * config.tol,
            )
            return _mean_square(modes, d0, diffusion, natural)

        mean_square = _converge_radius(model, config, compute, relative)
    return RmsResult(
        times=folded.times,
        mean_square=folded.even(mean_square) * length**2,
        direction=tuple(unit),
        units=system.units,
    )
