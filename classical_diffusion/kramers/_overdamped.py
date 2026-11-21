# cspell: ignore btol lsqr tocsr vdot
"""The overdamped limit of the Langevin equation, the Smoluchowski equation, solved by Gauss quadrature of its symmetrised generator."""

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from classical_diffusion._results import RmsResult, SlowModes
from classical_diffusion.kramers._config import DEFAULT_CONFIG, KramersConfig
from classical_diffusion.kramers._lanczos import Quadrature, get_lanczos_quadrature
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


def _get_generator(
    system: PeriodicSystem, delta_k: FloatArray, config: KramersConfig
) -> tuple[Model, _Generator]:
    model = build_model(system)
    vectors = get_plane_waves(model, config.basis_tol)
    return model, _Generator(model, vectors, delta_k)


def _converged_on(
    times: FloatArray, tol: float
) -> Callable[[Quadrature, Quadrature], bool]:
    def isf(q: Quadrature) -> FloatArray:
        return np.exp(-np.outer(times, np.maximum(q.nodes, 0))) @ q.weights

    t_max = float(times[-1])

    def converged(a: Quadrature, b: Quadrature) -> bool:
        # A Ritz value off by its residual r changes its term by up to
        # w r max_t t e^{-lambda t}, so every term must be resolved over the times as well as
        # the sum agree between blocks
        nodes = np.maximum(b.nodes, 1e-300)
        # |d/d lambda e^{-lambda t}| = t e^{-lambda t}, largest at t = 1 / lambda
        sensitivity = np.where(
            nodes * t_max > 1, 1 / (np.e * nodes), t_max * np.exp(-nodes * t_max)
        )
        unresolved = np.max(b.weights * np.minimum(1.0, b.residuals * sensitivity))
        return bool(
            unresolved < 0.1 * tol and np.max(np.abs(isf(a) - isf(b))) < 0.1 * tol
        )

    return converged


def _scaled_delta_k(
    system: PeriodicSystem, model: Model, delta_k: tuple[float, ...]
) -> FloatArray:
    return np.asarray(delta_k, dtype=float) / system.units.length_into(1.0, model.units)


def _get_isf_quadrature(
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    times: FloatArray,
    config: KramersConfig,
) -> tuple[Model, Quadrature]:
    model = build_model(system)
    vectors = get_plane_waves(model, config.basis_tol)
    generator = _Generator(model, vectors, _scaled_delta_k(system, model, delta_k))
    quadrature = get_lanczos_quadrature(
        generator.apply,
        generator.amplitude,
        converged=_converged_on(times, config.tol),
    )
    return model, dataclasses.replace(
        quadrature,
        nodes=np.maximum(quadrature.nodes, 0),
        weights=quadrature.weights / generator.norm,
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
    spectral measure of -L in w (`get_lanczos_quadrature`), extended until
    the ISF at the times changes by less than config.tol / 10.

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

    Raises
    ------
    ValueError
        If the times do not increase from zero.
    """
    times = np.asarray(times, dtype=np.float64)
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        msg = "The times must increase from zero"
        raise ValueError(msg)
    model = build_model(system)
    natural = system.units.time_into(times, model.units)
    _, quadrature = _get_isf_quadrature(system, delta_k, natural, config)
    isf = np.exp(-np.outer(natural, quadrature.nodes)) @ quadrature.weights
    return IsfResult(times=times, isf=isf, delta_k=delta_k, units=system.units)


@timed
def get_slow_modes_overdamped(
    system: PeriodicSystem,
    delta_k: tuple[float, ...],
    n_modes: int | None = None,
    *,
    config: KramersConfig = DEFAULT_CONFIG,
) -> SlowModes:
    """Get the slowest modes of the overdamped ISF of a periodic system at dK, I(t) = sum_j a_j exp(lambda_j t).

    The modes are those of the Gauss quadrature of `solve_isf_overdamped`,
    converged at the times 0 to 100 / (D_0 |G|^2) of the shortest reciprocal
    vector G, every mode by default.
    """
    model = build_model(system)
    g = float(np.min(np.linalg.norm(model.reciprocal, axis=-1)))
    window = np.linspace(0, 100 * model.friction / g**2, 400)
    _, quadrature = _get_isf_quadrature(system, delta_k, window, config)
    count = len(quadrature.nodes) if n_modes is None else n_modes
    natural = model.units.time_into(1.0, system.units)
    return SlowModes(
        system=system,
        delta_k=tuple(delta_k),
        rates=-quadrature.nodes[:count].astype(complex) / natural,
        amplitudes=quadrature.weights[:count].astype(complex),
    )


def _direction(model: Model, direction: tuple[float, ...] | None) -> FloatArray:
    unit = np.eye(model.n_dim)[0] if direction is None else np.asarray(direction, float)
    return unit / np.linalg.norm(unit)


def _get_force_quadrature(
    system: PeriodicSystem,
    direction: tuple[float, ...] | None,
    config: KramersConfig,
    converged: Callable[[Quadrature, Quadrature], bool],
) -> tuple[Model, _Generator, FloatArray, Quadrature]:
    """Get the quadrature of the measure of -L at K = 0 in the force amplitude (e . grad V) w, over <w|w>."""
    model, generator = _get_generator(
        system, np.zeros(build_model(system).n_dim), config
    )
    unit = _direction(model, direction)
    force = generator.force_amplitude(unit)
    w = generator.amplitude / np.sqrt(generator.norm)
    force -= w * np.vdot(w, force)
    if np.linalg.norm(force) == 0:
        # A flat potential: no force
        return (
            model,
            generator,
            unit,
            Quadrature(nodes=np.ones(1), weights=np.zeros(1), residuals=np.zeros(1)),
        )
    quadrature = get_lanczos_quadrature(
        generator.deflated(), force, converged=converged
    )
    return (
        model,
        generator,
        unit,
        dataclasses.replace(quadrature, weights=quadrature.weights / generator.norm),
    )


def _inverse_moment(q: Quadrature) -> float:
    keep = q.nodes > 0
    return float(np.sum(q.weights[keep] / q.nodes[keep]))


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
    (`_get_diffusion_ratio`). In 1D it is the Lifson-Jackson formula
    D_0 / (<e^V> <e^{-V}>), exactly.

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
    unit = _direction(model, direction)
    ratio = _get_diffusion_ratio(model, unit, config, system=system)
    length = model.units.length_into(1.0, system.units)
    return (
        ratio * length**2 / (model.friction * model.units.time_into(1.0, system.units))
    )


_MAX_ROUNDS = 12
_CANCELLATION_LIMIT = 1e-8
"""The D / D_0 below which the difference D_0 - D_0^2 <F w| (-L)^{-1} |F w> / <w|w> loses more than 1e-8 of D to round off, where D is found as a least squares residual instead."""
_DENSE_LIMIT = 4000
"""The most plane waves for which the least squares problem of D is solved by a full QR factorisation."""


def _get_residual_norm(gradient: ComplexArray, field: ComplexArray) -> float:
    """Get min_u |field - A u|^2, as the norm of the part of field outside the range of A, with no difference of large terms.

    A full QR factorisation A = [Q_1 Q_2] [R; 0] gives the residual as
    Q_2^dag field, whose norm is accurate relative to itself, however small,
    where |field|^2 - |Q_1^dag field|^2 would lose it to round off.
    """
    q, _ = scipy.linalg.qr(gradient, mode="full")
    rank = gradient.shape[1]
    return float(np.sum(np.abs(q[:, rank:].conj().T @ field) ** 2))


def _get_diffusion_ratio(
    model: Model,
    unit: FloatArray,
    config: KramersConfig,
    *,
    system: PeriodicSystem,
) -> float:
    """Get D / D_0 = min_u |e w - A u|^2 / <w|w>, with A = grad + grad V / 2 at K = 0.

    D = D_0 - D_0^2 <F w| (-L)^{-1} |F w> / <w|w> with F w = A^dag (e w),
    and -L = D_0 A^dag A, so D / D_0 is the norm of the part of e w outside
    the range of A, over <w|w>: the least squares residual, a sum of
    squares. It is accurate relative to itself even for deep barriers,
    where D is many orders below D_0 and the difference of the two terms
    would be lost to round off. The radius of the plane waves grows by 1.3
    until D changes by less than config.tol of itself.
    """
    if model.n_dim == 1:
        return _get_lifson_jackson_ratio(model, config.tol)
    d0 = 1 / model.friction

    def converged(a: Quadrature, b: Quadrature) -> bool:
        # A Ritz value off by its residual r changes w / lambda by up to
        # w r / lambda^2
        nodes = np.maximum(b.nodes, 1e-300)
        unresolved = d0 * np.max(
            b.weights * np.minimum(1 / nodes, b.residuals / nodes**2)
        )
        change = d0 * abs(_inverse_moment(a) - _inverse_moment(b))
        return bool(max(unresolved, change) < 1e-2 * config.tol)

    _, _, _, quadrature = _get_force_quadrature(system, tuple(unit), config, converged)
    ratio = 1 - d0 * _inverse_moment(quadrature)
    if ratio > _CANCELLATION_LIMIT:
        return ratio
    return _get_residual_ratio(model, unit, config)


def _get_residual_ratio(model: Model, unit: FloatArray, config: KramersConfig) -> float:
    """Get D / D_0 as the least squares residual min_u |e w - A u|^2 / <w|w>, growing the plane waves until it converges.

    Raises
    ------
    RuntimeError
        If D does not converge in the plane waves.
    """
    vectors = get_plane_waves(model, config.basis_tol)
    radius = float(np.max(np.linalg.norm(vectors @ model.reciprocal, axis=-1)))
    previous = None
    for _ in range(_MAX_ROUNDS):
        vectors = get_plane_waves_within(model, radius)
        gradient = get_gradient_operator(model, vectors, np.zeros(model.n_dim))
        w = get_amplitude(model, vectors, get_grid_size(vectors))
        # e w, on the rows of the gradient: the inputs come first in its
        # output plane waves only up to their order, so place it by index
        extended = get_extended_vectors(model, vectors)
        index = {tuple(int(i) for i in v): k for k, v in enumerate(get_rows(extended))}
        rows = np.array([index[tuple(int(i) for i in v)] for v in get_rows(vectors)])
        field = np.zeros((model.n_dim, len(extended)), dtype=np.complex128)
        field[:, rows] = unit[:, None] * w[None, :]
        stacked = gradient.reshape(-1, len(vectors))
        if len(vectors) <= _DENSE_LIMIT:
            residual = _get_residual_norm(stacked, field.reshape(-1))
        else:
            solution = scipy.sparse.linalg.lsqr(
                scipy.sparse.csr_array(stacked),
                field.reshape(-1),
                atol=0,
                btol=0,
                conlim=0,
                iter_lim=20 * len(vectors),
            )[0]
            residual = float(
                np.sum(np.abs(field.reshape(-1) - stacked @ solution) ** 2)
            )
        ratio = residual / float(np.vdot(w, w).real)
        if previous is not None and abs(ratio - previous) <= config.tol * abs(ratio):
            return ratio
        previous = ratio
        radius *= 1.3
    msg = (
        "D did not converge in the plane waves: e^{-V / 2} varies over too many"
        " orders of magnitude for the plane waves to hold it at the saddle"
    )
    raise RuntimeError(msg)


def _get_lifson_jackson_ratio(model: Model, tol: float) -> float:
    """Get D / D_0 = 1 / (<e^V> <e^{-V}>) in 1D (Lifson and Jackson 1962).

    The averages over a period are found by the trapezoid rule, which
    converges exponentially for a smooth periodic integrand, on a grid
    doubled until D changes by less than tol / 100 of itself. Both are
    taken relative to the minimum and maximum of V, so that neither
    overflows for a deep barrier.
    """
    n_grid = 256
    previous = None
    while True:
        u = np.arange(n_grid) / n_grid
        values = np.zeros(n_grid, dtype=np.complex128)
        for g, v_g in model.potential:
            values += v_g * np.exp(2j * np.pi * g[0] * u)
        v = values.real
        low, high = float(v.min()), float(v.max())
        ratio = np.exp(low - high) / (
            np.mean(np.exp(v - high)) * np.mean(np.exp(low - v))
        )
        if previous is not None and abs(ratio - previous) <= 1e-2 * tol * ratio:
            return float(ratio)
        previous = ratio
        n_grid *= 2


def _mean_square(
    q: Quadrature, d0: float, diffusion: float, times: FloatArray
) -> FloatArray:
    """2 D t + 2 D_0^2 sum_j w_j (1 - e^{-l_j t}) / l_j^2.

    This is 2 D_0 t - 2 D_0^2 sum_j w_j [t / l_j - (1 - e^{-l_j t}) / l_j^2]
    with D = D_0 - D_0^2 sum_j w_j / l_j, written as a sum of positive terms.
    """
    keep = q.nodes > 0
    nodes, weights = q.nodes[keep], q.weights[keep]
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
    with the force correlation from the Gauss quadrature at K = 0 in F w,
    which is 2 D t + 2 D_0^2 sum_j w_j (1 - e^{-l_j t}) / l_j^2 in its nodes
    l_j and weights w_j, with D from `get_diffusion_constant_overdamped`: a
    sum of positive terms, accurate however far D is below D_0. The
    quadrature is extended until the mean square displacement at the times
    changes by less than config.tol / 10 of itself.

    Raises
    ------
    ValueError
        If the times do not increase from zero.
    """
    times = np.asarray(times, dtype=np.float64)
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        msg = "The times must increase from zero"
        raise ValueError(msg)
    model = build_model(system)
    natural = system.units.time_into(times, model.units)
    d0 = 1 / model.friction
    diffusion = d0 * _get_diffusion_ratio(
        model, _direction(model, direction), config, system=system
    )
    t = float(natural[-1])

    def converged(a: Quadrature, b: Quadrature) -> bool:
        reference = _mean_square(b, d0, diffusion, natural)[1:]
        change = _mean_square(a, d0, diffusion, natural)[1:] - reference
        # A Ritz value off by its residual r changes its term of the mean
        # square displacement by up to 2 D_0^2 w r min(t^2, 2 / lambda^3)
        nodes = np.maximum(b.nodes, 1e-300)
        unresolved = (
            2 * d0**2 * np.max(b.weights * b.residuals * np.minimum(t**2, 2 / nodes**3))
        )
        scale = np.maximum(np.abs(reference), 1e-300)
        return bool(
            max(unresolved / scale.min(), np.max(np.abs(change) / scale))
            < 0.1 * config.tol
        )

    model, _, unit, quadrature = _get_force_quadrature(
        system, direction, config, converged
    )
    length = model.units.length_into(1.0, system.units)
    return RmsResult(
        times=times,
        mean_square=_mean_square(quadrature, d0, diffusion, natural) * length**2,
        direction=tuple(unit),
        units=system.units,
    )
