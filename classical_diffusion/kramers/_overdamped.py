"""The overdamped limit of the Langevin equation, the Smoluchowski equation, solved by Gauss quadrature of its symmetrised generator."""

from typing import TYPE_CHECKING

import dataclasses

import numpy as np
import scipy.sparse

from classical_diffusion.caldeira_leggett import SlowModes
from classical_diffusion.caldeira_leggett._solver import RmsResult
from classical_diffusion.kramers._config import DEFAULT_CONFIG, KramersConfig
from classical_diffusion.kramers._lanczos import Quadrature, get_lanczos_quadrature
from classical_diffusion.kramers._model import Model, build_model, get_plane_waves
from classical_diffusion.kramers._smoluchowski import (
    get_amplitude,
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


def _check_times(times: FloatArray) -> FloatArray:
    times = np.asarray(times, dtype=np.float64)
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        msg = "The times must increase from zero"
        raise ValueError(msg)
    return times


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
    times = _check_times(times)
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
    F = e . grad V, which in the symmetrised frame is
    D_0 - D_0^2 <F w| (-L)^{-1} |F w> / <w|w>. The inverse is the Gauss
    quadrature of 1 / lambda, extended until it changes by less than
    config.tol / 10 of D_0. In 1D this is the Lifson-Jackson formula
    D_0 / (<e^V> <e^{-V}>). The difference of the two terms limits D to
    about 1e-14 D_0 from round off, below which it is zero to that accuracy.

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
    d0 = 1 / build_model(system).friction

    def converged(a: Quadrature, b: Quadrature) -> bool:
        # A Ritz value off by its residual r changes w / lambda by up to
        # w r / lambda^2
        nodes = np.maximum(b.nodes, 1e-300)
        unresolved = d0 * np.max(
            b.weights * np.minimum(1 / nodes, b.residuals / nodes**2)
        )
        change = d0 * abs(_inverse_moment(a) - _inverse_moment(b))
        return bool(max(unresolved, change) < 0.1 * config.tol)

    model, _, _, quadrature = _get_force_quadrature(
        system, direction, config, converged
    )
    diffusion = d0 - d0**2 * _inverse_moment(quadrature)
    length = model.units.length_into(1.0, system.units)
    return diffusion * length**2 / model.units.time_into(1.0, system.units)


def _mean_square(q: Quadrature, d0: float, times: FloatArray) -> FloatArray:
    """2 D_0 t - 2 D_0^2 sum_j w_j [t / l_j - (1 - e^{-l_j t}) / l_j^2]."""
    keep = q.nodes > 0
    nodes, weights = q.nodes[keep], q.weights[keep]
    t = times[:, None]
    memory = t / nodes - (-np.expm1(-nodes * t)) / nodes**2
    return 2 * d0 * times - 2 * d0**2 * (memory @ weights)


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
    with the force correlation from the Gauss quadrature at K = 0 in F w
    (`get_diffusion_constant_overdamped`), extended until the mean square
    displacement at the times changes by less than config.tol / 10 of
    2 D_0 t. Its slope at long times is 2 D.

    Raises
    ------
    ValueError
        If the times do not increase from zero.
    """
    times = _check_times(times)
    model = build_model(system)
    natural = system.units.time_into(times, model.units)
    d0 = 1 / model.friction
    scale = 2 * d0 * np.maximum(natural, natural[1] if len(natural) > 1 else 1.0)

    t = float(natural[-1])

    def converged(a: Quadrature, b: Quadrature) -> bool:
        change = _mean_square(a, d0, natural) - _mean_square(b, d0, natural)
        # A Ritz value off by its residual r changes its term of the mean
        # square displacement at t by up to 2 D_0^2 w r min(t^3 / 6, 2 t / lambda^2)
        nodes = np.maximum(b.nodes, 1e-300)
        unresolved = (
            2
            * d0**2
            * np.max(b.weights * b.residuals * np.minimum(t**3 / 6, 2 * t / nodes**2))
            / (2 * d0 * t)
        )
        return bool(max(unresolved, np.max(np.abs(change) / scale)) < 0.1 * config.tol)

    model, _, unit, quadrature = _get_force_quadrature(
        system, direction, config, converged
    )
    length = model.units.length_into(1.0, system.units)
    return RmsResult(
        times=times,
        mean_square=_mean_square(quadrature, d0, natural) * length**2,
        direction=tuple(unit),
        units=system.units,
    )
