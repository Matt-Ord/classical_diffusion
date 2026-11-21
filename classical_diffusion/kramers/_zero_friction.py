"""Deterministic ISF and mean square displacement of a 1D periodic system at zero friction.

Hamiltonian dynamics, Boltzmann-distributed initial conditions, natural units of
`classical_diffusion.kramers._model.build_model`: m = k_B T = 1, the period of the
potential is 2 pi (length unit a / 2 pi), the time unit is that length over the
thermal velocity.

Method (action-angle / energy-shell decomposition)
--------------------------------------------------
dx dp = dE d theta, theta the time along the orbit of energy E.  Each orbit is
periodic with period tau(E) and frequency W = 2 pi / tau: trapped orbits close,
running orbits advance by s 2 pi (s = +-1) per period, at mean velocity s W.
With c_n(E) the Fourier coefficients of f(theta) = exp(i K (x(theta) - s W theta))
along the orbit (trapped: s = 0),

    I(K, t) = sum_families int dW  w(W) sum_n |c_n|^2 exp(i (n + s K) W t),
    w(W) = e^{-E} tau |dE/dW| / Z,  Z = sqrt(2 pi) int e^{-V} dx.

W is the integration variable because the phase is linear in it: on each panel
of W the weight w |c_n|^2 is expanded in Legendre polynomials and
int_{-1}^{1} P_k(u) e^{i nu u} du = 2 i^k j_k(nu) gives the integral exactly
for that polynomial at ANY time, so the cost does not grow with t.  The orbits
near the separatrix (W -> 0) carry the weight e^{-E_s} tau |dE/dW| ~ e^{-c / W},
which is flat: the W < W_cut part is dropped once its weight is below the
tolerance.  Each node needs tau(E) (Gauss quadrature after a substitution which
resolves the inverse square root at the turning points and the logarithm near
the separatrix), E(W) by bisection, and the Fourier coefficients along the orbit
(`_get_orbit_coefficients`): the quarter/half orbit is parametrised by
pi - x = u cosh y (trapped) or r sinh y (running), theta(y) is a Chebyshev
cumulative integral and the c_n are Fejer quadratures in y.  No time stepping
(an earlier symplectic sampler was unstable near the separatrix).

MSD(t) = <(x(t) - x(0))^2> = sum_families int dW w(W) [ s^2 W^2 t^2
          + 2 sum_{n != 0} |d_n|^2 (1 - cos(n W t)) ],
d_n the Fourier coefficients of x(theta) - s W theta.  D is infinite (running orbits).

Limitations: one minimum and one maximum of V per unit cell (all CosineSystem1D),
and a frequency W(E) monotonic in each family (true for the cosine).
"""

# cspell: ignore leggauss legvander chebvander chebint chebval legder legval spherical_jn arccosh arcsinh cumprod Fejer matmul einsum lbnd ktn parametrised symplectic vectorised
from dataclasses import dataclass
from functools import cache
from operator import attrgetter
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial import chebyshev, legendre
from scipy.special import spherical_jn

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.kramers._model import ComplexArray, FloatArray, Model

_N_NODES = 24
"""The Gauss-Legendre nodes per panel of the frequency."""
_N_CHEBYSHEV = 512
"""The largest number of Chebyshev nodes along an orbit."""
_N_HARMONICS = 256
"""The largest harmonic of the orbit frequency."""
_MAX_PANELS = 200
"""The most panels of the frequency in one family."""
_MIN_PANEL_WIDTH = 1e-6
_MAX_CLOSURE = 1e-9
"""The largest disagreement of the two period quadratures of a converged orbit."""
_MAX_ENERGY_TAIL = 1e-10
"""The largest relative Legendre tail of E(W) on a converged panel."""
_SINE_TOL = 1e-14
_REAL_TOL = 1e-10

_PANEL_NODES, _PANEL_WEIGHTS = legendre.leggauss(_N_NODES)
_PANEL_VANDER = legendre.legvander(_PANEL_NODES, _N_NODES - 1)
_PANEL_PROJECTION = (_PANEL_VANDER * _PANEL_WEIGHTS[:, None]).T * (
    (2 * np.arange(_N_NODES) + 1) / 2
)[:, None]
"""The Legendre coefficients of a function from its values at the panel nodes."""
_PERIOD_NODES, _PERIOD_WEIGHTS = legendre.leggauss(160)


@dataclass(frozen=True, kw_only=True)
class CosineSeries:
    """A potential offset + sum_k a_k cos(k x), with its minimum at x = 0 and its maximum at pi."""

    wavenumbers: FloatArray
    amplitudes: FloatArray
    offset: float

    def __call__(self, position: FloatArray | float) -> FloatArray:
        """Get the potential at the positions."""
        position = np.asarray(position, float)
        return self.offset + np.sum(
            self.amplitudes * np.cos(np.multiply.outer(position, self.wavenumbers)),
            axis=-1,
        )

    def drop(self, turning_point: FloatArray, distance: FloatArray) -> FloatArray:
        """Get V(X) - V(X - d), for the turning points X and distances d, without cancellation for small d."""
        turning_point = np.asarray(turning_point, float)[..., None]
        distance = np.asarray(distance, float)[..., None]
        return np.sum(
            -2
            * self.amplitudes
            * np.sin(self.wavenumbers * (2 * turning_point - distance) / 2)
            * np.sin(self.wavenumbers * distance / 2),
            axis=-1,
        )

    @property
    def barrier(self) -> float:
        """The potential at its maximum, pi."""
        return float(self(np.pi))

    @property
    def well_frequency(self) -> float:
        """The frequency of small oscillations about the minimum."""
        return float(np.sqrt(-np.sum(self.amplitudes * self.wavenumbers**2)))

    @property
    def barrier_frequency(self) -> float:
        """The imaginary frequency of the motion about the maximum."""
        return float(
            np.sqrt(
                np.sum(
                    self.amplitudes
                    * self.wavenumbers**2
                    * np.cos(self.wavenumbers * np.pi)
                )
            )
        )

    def log_partition(self) -> float:
        """Get log Z, Z = sqrt(2 pi) int_0^{2 pi} e^{-V} dx (trapezoid: spectrally exact)."""
        position = np.arange(4096) * 2 * np.pi / 4096
        potential = self(position)
        return 0.5 * np.log(2 * np.pi) + np.log(np.mean(np.exp(-potential)) * 2 * np.pi)


def series_from_model(model: Model) -> CosineSeries:
    """Get the potential of a 1D model in its natural units, shifted to its minimum at 0.

    Raises
    ------
    ValueError
        If the model is not 1D.
    NotImplementedError
        If the potential has sine terms, or more than one minimum and maximum per unit cell.
    """
    if model.n_dim != 1:
        msg = "Only 1D models have a cosine series"
        raise ValueError(msg)
    coefficients = {n[0]: complex(v) for n, v in model.potential}
    orders = sorted(k for k in coefficients if k > 0)
    amplitudes = np.array([2 * coefficients[k].real for k in orders])
    if any(abs(coefficients[k].imag) > _SINE_TOL for k in orders):
        msg = "Potentials with sine terms are not supported"
        raise NotImplementedError(msg)
    wavenumbers = np.array(orders, float)
    series = CosineSeries(wavenumbers=wavenumbers, amplitudes=amplitudes, offset=0.0)
    if series(np.pi) < series(0.0):  # put the minimum at 0
        amplitudes *= (-1.0) ** wavenumbers
    series = CosineSeries(
        wavenumbers=wavenumbers,
        amplitudes=amplitudes,
        offset=-float(np.sum(amplitudes)),
    )
    if np.any(np.diff(series(np.linspace(0, np.pi, 4001))) <= 0):
        msg = "Only one minimum and one maximum per unit cell are supported"
        raise NotImplementedError(msg)
    return series


def _trapped_period(series: CosineSeries, gap: FloatArray) -> FloatArray:
    """Get the period of the trapped orbits with turning points +-(pi - u), for the gaps u.

    tau = 4 int_0^X dx / sqrt(2 (V(X) - V(x))), with pi - x = u cosh y, which
    resolves the inverse square root at the turning point and the logarithm
    of the separatrix (the hilltop at pi).
    """
    gap = np.asarray(gap, float)[..., None]
    y_max = np.arccosh(np.pi / gap)
    y = 0.5 * y_max * (1 + _PERIOD_NODES)
    weights = 0.5 * y_max * _PERIOD_WEIGHTS
    distance = 2 * gap * np.sinh(y / 2) ** 2
    drop = series.drop((np.pi - gap) * np.ones_like(distance), distance)
    dx_dy = gap * np.sinh(y)
    return 4 * np.sum(weights * dx_dy / np.sqrt(2 * np.maximum(drop, 1e-300)), axis=-1)


def _running_period(series: CosineSeries, excess: FloatArray) -> FloatArray:
    """Get the period of the running orbits at the energies E_s + eps above the barrier.

    tau = 2 int_0^pi dx / sqrt(2 (eps + V(pi) - V(x))), with pi - x = r sinh y,
    r = sqrt(2 eps) / omega_b.
    """
    excess = np.asarray(excess, float)[..., None]
    scale = np.sqrt(2 * excess) / series.barrier_frequency
    y_max = np.arcsinh(np.pi / scale)
    y = 0.5 * y_max * (1 + _PERIOD_NODES)
    weights = 0.5 * y_max * _PERIOD_WEIGHTS
    distance = scale * np.sinh(y)
    drop = series.drop(np.pi * np.ones_like(distance), distance)
    return 2 * np.sum(
        weights * scale * np.cosh(y) / np.sqrt(2 * (excess + np.maximum(drop, 0.0))),
        axis=-1,
    )


def _bisect(
    function: Callable[[FloatArray], FloatArray],
    lower: float,
    upper: float,
    target: FloatArray,
) -> FloatArray:
    """Get the solution of function(z) = target on [lower, upper], for an increasing function, by vectorised bisection."""
    lo = np.full_like(target, lower, dtype=float)
    hi = np.full_like(target, upper, dtype=float)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        above = function(mid) > target
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return 0.5 * (lo + hi)


def _trapped_turning_point(series: CosineSeries, frequency: FloatArray) -> FloatArray:
    """Get the turning point X of the trapped orbits with the frequencies W < omega_0."""
    frequency = np.asarray(frequency, float)

    # z = log(pi - X): as z decreases, X -> pi and W -> 0, so W increases with z
    def frequency_at(log_gap: FloatArray) -> FloatArray:
        return 2 * np.pi / _trapped_period(series, np.exp(log_gap))

    log_gap = _bisect(frequency_at, -40.0, np.log(np.pi), frequency)
    return np.pi - np.exp(log_gap)


def _running_excess(series: CosineSeries, frequency: FloatArray) -> FloatArray:
    """Get the energy E - E_s above the barrier of the running orbits with the frequencies."""
    frequency = np.asarray(frequency, float)

    def frequency_at(log_excess: FloatArray) -> FloatArray:
        return 2 * np.pi / _running_period(series, np.exp(log_excess))

    return np.exp(_bisect(frequency_at, -80.0, 6.0, frequency))


@dataclass(frozen=True, kw_only=True)
class _Chebyshev:
    """Chebyshev (first kind) nodes on [-1, 1], the cumulative integral matrix from -1 and the Fejer weights."""

    nodes: FloatArray
    cumulative: FloatArray
    weights: FloatArray


@cache
def _get_chebyshev(n_nodes: int) -> _Chebyshev:
    """Get the Chebyshev nodes, cumulative integral and Fejer weights of the given order."""
    nodes = np.cos(np.pi * (np.arange(n_nodes) + 0.5) / n_nodes)[::-1]
    inverse_vander = np.linalg.inv(chebyshev.chebvander(nodes, n_nodes - 1))
    # cumulative integral of each basis polynomial, lower limit -1, evaluated at the nodes and at +1
    integrals = np.zeros((n_nodes + 1, n_nodes))
    for k in range(n_nodes):
        basis = np.zeros(n_nodes)
        basis[k] = 1
        integrals[:, k] = chebyshev.chebval(
            np.append(nodes, 1.0), chebyshev.chebint(basis, lbnd=-1)
        )
    full = integrals @ inverse_vander
    return _Chebyshev(nodes=nodes, cumulative=full[:-1], weights=full[-1])


@dataclass(frozen=True, kw_only=True)
class _OrbitPath:
    """A quarter (trapped) or half (running) of each orbit, at the Chebyshev nodes in y, from its far end to x = 0."""

    position: FloatArray
    """x, (n_orbit, n_chebyshev)."""
    kinetic: FloatArray
    """p^2 / 2 = E - V(x)."""
    dx_dy: FloatArray
    y_max: FloatArray
    """The end of the range of y, (n_orbit,)."""
    energy: FloatArray
    n_pieces: int
    """The pieces in one period, 4 or 2."""


def _trapped_path(
    series: CosineSeries, gap: FloatArray, nodes: FloatArray
) -> _OrbitPath:
    """Get the quarter of the trapped orbits with turning points pi - u, for the gaps u, with pi - x = u cosh y."""
    y_max = np.arccosh(np.pi / gap)
    y = 0.5 * y_max[:, None] * (1 + nodes)
    distance = 2 * gap[:, None] * np.sinh(y / 2) ** 2
    return _OrbitPath(
        position=np.pi - gap[:, None] * np.cosh(y),
        kinetic=series.drop((np.pi - gap)[:, None] * np.ones_like(distance), distance),
        dx_dy=gap[:, None] * np.sinh(y),
        y_max=y_max,
        energy=series(np.pi - gap),
        n_pieces=4,
    )


def _running_path(
    series: CosineSeries, excess: FloatArray, nodes: FloatArray
) -> _OrbitPath:
    """Get the half of the running orbits at E_s + eps, with pi - x = r sinh y."""
    scale = np.sqrt(2 * excess) / series.barrier_frequency
    y_max = np.arcsinh(np.pi / scale)
    y = 0.5 * y_max[:, None] * (1 + nodes)
    return _OrbitPath(
        position=np.pi - scale[:, None] * np.sinh(y),
        kinetic=excess[:, None]
        + series.drop(np.pi * np.ones_like(y), scale[:, None] * np.sinh(y)),
        dx_dy=scale[:, None] * np.cosh(y),
        y_max=y_max,
        energy=series.barrier + excess,
        n_pieces=2,
    )


def _fourier_pair(
    amplitude: ComplexArray, powers: ComplexArray
) -> tuple[ComplexArray, ComplexArray]:
    """Get (a . e^{-i n phi}, a . e^{+i n phi}) for n = -H..H, from a of shape (N, q) and the powers e^{-i k phi}, k = 0..H."""
    lower = np.matmul(amplitude[:, None, :], powers)[:, 0, :]
    upper = np.conj(np.matmul(np.conj(amplitude)[:, None, :], powers)[:, 0, :])
    return (
        np.concatenate([upper[:, :0:-1], lower], axis=1),
        np.concatenate([lower[:, :0:-1], upper], axis=1),
    )


@dataclass(frozen=True, kw_only=True)
class _Family:
    """The problem of one family of orbits (direction 0 trapped, 1 running right) at a wavevector."""

    series: CosineSeries
    direction: int
    wavevector: float
    tol: float
    log_partition: float

    def parameter(self, frequency: FloatArray) -> FloatArray:
        """Get the parameter of the orbits with the frequencies: u = pi - X (trapped) or eps = E - E_s (running)."""
        if self.direction == 0:
            return np.pi - _trapped_turning_point(self.series, frequency)
        return _running_excess(self.series, frequency)

    def path(self, parameter: FloatArray, nodes: FloatArray) -> _OrbitPath:
        """Get the quarter (trapped) or half (running) orbits at the parameters."""
        if self.direction == 0:
            return _trapped_path(self.series, parameter, nodes)
        return _running_path(self.series, parameter, nodes)

    def period(self, parameter: FloatArray) -> FloatArray:
        """Get the period of the orbits at the parameters, by Gauss quadrature."""
        if self.direction == 0:
            return _trapped_period(self.series, parameter)
        return _running_period(self.series, parameter)

    def energy(self, frequency: FloatArray) -> FloatArray:
        """Get the energy of the orbits with the frequencies."""
        if self.direction == 0:
            return self.series(_trapped_turning_point(self.series, frequency))
        return self.series.barrier + _running_excess(self.series, frequency)

    @property
    def multiplicity(self) -> float:
        """The copies of the family: the left movers are the complex conjugate of the right movers."""
        return 2.0 if self.direction != 0 else 1.0


@dataclass(frozen=True, kw_only=True)
class _Phase:
    """The orbit period, and the phase phi = W theta and quadrature weights d phi / (2 pi) at the Chebyshev nodes."""

    period: FloatArray
    phase: FloatArray
    weights: FloatArray


def _get_phase(path: _OrbitPath, grid: _Chebyshev) -> _Phase:
    """Get the phase along the orbits, theta measured from x = 0 (y = y_max), theta(y) = int_y^{y_max}."""
    # d theta / d(Chebyshev variable)
    theta_rate = (
        path.dx_dy
        / np.sqrt(2 * np.maximum(path.kinetic, 1e-300))
        * (0.5 * path.y_max[:, None])
    )
    total = theta_rate @ grid.weights
    theta = total[:, None] - theta_rate @ grid.cumulative.T
    period = path.n_pieces * total
    frequency = 2 * np.pi / period
    return _Phase(
        period=period,
        # in [0, pi/2] (trapped) or [0, pi] (running)
        phase=frequency[:, None] * theta,
        weights=(grid.weights[None, :] * theta_rate) * frequency[:, None] / (2 * np.pi),
    )


def _trapped_coefficients(
    weights: FloatArray, position: FloatArray, wavevector: float, powers: ComplexArray
) -> tuple[ComplexArray, ComplexArray]:
    """Get (c_n, d_n) of trapped orbits from their quarter, the others (pi - phi, x), (pi + phi, -x), (2 pi - phi, -x) by symmetry."""
    sign = (-1.0) ** np.arange(-(powers.shape[-1] - 1), powers.shape[-1])
    plane_wave = np.exp(1j * wavevector * position)
    p1, p1c = _fourier_pair(weights * plane_wave, powers)
    p2, p2c = _fourier_pair(weights * np.conj(plane_wave), powers)
    q1, q1c = _fourier_pair((weights * position).astype(complex), powers)
    return (
        p1 + sign * p1c + sign * p2 + p2c,
        q1 + sign * q1c - sign * q1 - q1c,
    )


def _running_coefficients(
    weights: FloatArray, periodic: FloatArray, wavevector: float, powers: ComplexArray
) -> tuple[ComplexArray, ComplexArray]:
    """Get (c_n, d_n) of running orbits from their half (phi, x), the other (-phi, -x) by symmetry, for the periodic part x - phi."""
    plane_wave = np.exp(1j * wavevector * periodic)
    p1, _ = _fourier_pair(weights * plane_wave, powers)
    _, p2c = _fourier_pair(weights * np.conj(plane_wave), powers)
    q1, q1c = _fourier_pair((weights * periodic).astype(complex), powers)
    return p1 + p2c, q1 - q1c


@dataclass(frozen=True, kw_only=True)
class _OrbitCoefficients:
    """The energy, period and Fourier coefficients of a set of orbits."""

    energy: FloatArray
    period: FloatArray
    isf: ComplexArray
    """c_n, the coefficients of exp(i K x~), n = -H..H."""
    msd: ComplexArray
    """d_n, the coefficients of x~."""
    closure: float
    """The largest disagreement of the period with its Gauss quadrature."""


def _get_orbit_coefficients(
    family: _Family, parameter: FloatArray, n_chebyshev: int, n_harmonics: int
) -> _OrbitCoefficients:
    """Get the orbit quantities at the parameters of the family.

    The quarter (trapped) or half (running) orbit is parametrised by y, which
    resolves the turning point and the hilltop. theta(y) is the Chebyshev
    cumulative integral of d theta / dy = (dx/dy) / p, and the Fourier
    coefficients along the orbit are Fejer quadratures in y, using the
    symmetry V(-x) = V(x) for the rest of the orbit.
    """
    parameter = np.asarray(parameter, float)
    grid = _get_chebyshev(n_chebyshev)
    path = family.path(parameter, grid.nodes)
    phase = _get_phase(path, grid)
    powers = np.empty((*phase.phase.shape, n_harmonics + 1), complex)
    powers[..., 0] = 1.0
    powers[..., 1:] = np.exp(-1j * phase.phase)[..., None]
    powers = np.cumprod(powers, axis=-1)

    if family.direction == 0:
        isf, msd = _trapped_coefficients(
            phase.weights, path.position, family.wavevector, powers
        )
    else:
        # periodic part x - phi
        isf, msd = _running_coefficients(
            phase.weights, path.position - phase.phase, family.wavevector, powers
        )
    return _OrbitCoefficients(
        energy=path.energy,
        period=phase.period,
        isf=isf,
        msd=msd,
        closure=float(np.max(np.abs(phase.period / family.period(parameter) - 1))),
    )


def _energy_derivative(energy: FloatArray, lo: float, hi: float) -> FloatArray:
    """Get dE/dW at the nodes of the panel [lo, hi] from the Legendre interpolant of E(W)."""
    coefficients = np.linalg.solve(_PANEL_VANDER, energy)
    derivative = legendre.legval(_PANEL_NODES, legendre.legder(coefficients))
    return np.asarray(derivative, dtype=np.float64) * 2 / (hi - lo)


@dataclass(frozen=True, kw_only=True)
class _Panel:
    """The orbits at the Gauss nodes of one panel [lo, hi] of the frequency."""

    lo: float
    hi: float
    frequency: FloatArray
    energy: FloatArray
    weight: FloatArray
    """w(W) at the nodes, without the Gauss weights."""
    isf_harmonics: FloatArray
    """|c_n|^2, (n_nodes, 2 H + 1)."""
    msd_harmonics: FloatArray
    """|d_n|^2."""


def _get_panel(family: _Family, lo: float, hi: float) -> _Panel:
    """Get the orbits of a panel, doubling the Chebyshev nodes and harmonics until the highest harmonics are negligible."""
    frequency = 0.5 * (lo + hi) + 0.5 * (hi - lo) * _PANEL_NODES
    n_chebyshev, n_harmonics = _N_CHEBYSHEV // 4, _N_HARMONICS // 4
    while True:
        orbits = _get_orbit_coefficients(
            family, family.parameter(frequency), n_chebyshev, n_harmonics
        )
        isf_harmonics = np.abs(orbits.isf) ** 2
        msd_harmonics = np.abs(orbits.msd) ** 2
        edge = max(
            isf_harmonics[:, :3].max(),
            isf_harmonics[:, -3:].max(),
            (msd_harmonics[:, :3].max() + msd_harmonics[:, -3:].max()) * 1e-2,
        )
        if (
            edge * (hi - lo) < 1e-6 * family.tol and orbits.closure < _MAX_CLOSURE
        ) or n_chebyshev >= _N_CHEBYSHEV:
            break
        n_chebyshev, n_harmonics = 2 * n_chebyshev, 2 * n_harmonics
    # pad the harmonics to the common size
    pad = _N_HARMONICS - n_harmonics
    return _Panel(
        lo=lo,
        hi=hi,
        frequency=frequency,
        energy=orbits.energy,
        weight=np.exp(-orbits.energy - family.log_partition)
        * orbits.period
        * np.abs(_energy_derivative(orbits.energy, lo, hi)),
        isf_harmonics=np.pad(isf_harmonics, ((0, 0), (pad, pad))),
        msd_harmonics=np.pad(msd_harmonics, ((0, 0), (pad, pad))),
    )


def _needs_split(panel: _Panel, tol: float) -> bool:
    """Get whether the Legendre tails of the weighted harmonics, or of E(W), are too large on a panel."""
    weighted = panel.weight[:, None] * np.concatenate(
        [
            panel.isf_harmonics,
            panel.msd_harmonics / (1 + panel.msd_harmonics.max()),
        ],
        axis=1,
    )
    coefficients = np.linalg.solve(_PANEL_VANDER, weighted)
    error = np.abs(coefficients[-3:]).sum(axis=0).sum() * (panel.hi - panel.lo)
    energy = np.linalg.solve(_PANEL_VANDER, panel.energy)
    relative_energy = np.abs(energy[-3:]).sum() / max(1e-300, np.abs(energy[1:]).sum())
    return bool(error > 1e-2 * tol or relative_energy > _MAX_ENERGY_TAIL)


def _separatrix_weight(family: _Family, frequency: float) -> float:
    """Get w(W) by a central difference of E(W), for the separatrix cut."""
    frequencies = np.array([frequency * (1 - 1e-5), frequency * (1 + 1e-5)])
    energy = family.energy(frequencies)
    derivative = (energy[1] - energy[0]) / (frequencies[1] - frequencies[0])
    return float(
        np.exp(-energy.mean() - family.log_partition)
        * 2
        * np.pi
        / frequency
        * abs(derivative)
    )


def _get_panels(family: _Family, top: float) -> list[_Panel]:
    """Get adaptive panels of W below `top`, cut below where the weight near the separatrix is negligible, in increasing W."""
    cut = 0.5 * top
    # the weight below cut is at most w(cut) cut (w increases with W near the separatrix)
    while not (_separatrix_weight(family, cut) * cut < 1e-3 * family.tol):
        cut *= 0.8
    edges = list(np.linspace(cut, top, 9))
    done = list[_Panel]()
    todo = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    while todo:
        lo, hi = todo.pop()
        panel = _get_panel(family, lo, hi)
        if (
            _needs_split(panel, family.tol)
            and len(done) + len(todo) < _MAX_PANELS
            and (hi - lo) > _MIN_PANEL_WIDTH
        ):
            mid = 0.5 * (lo + hi)
            todo += [(lo, mid), (mid, hi)]
        else:
            done.append(panel)
    done.sort(key=attrgetter("lo"))
    return done


def _get_oscillating_integral(
    weighted: FloatArray,
    rates: FloatArray,
    times: FloatArray,
    panel: _Panel,
    tol: float,
) -> ComplexArray:
    """Get sum_n int g_n(W) e^{i r_n W t} dW over the panel at the times, exactly for the Legendre interpolant of g at the panel nodes.

    Harmonics whose g is below the tolerance are dropped.
    """
    centre, half_width = 0.5 * (panel.lo + panel.hi), 0.5 * (panel.hi - panel.lo)
    keep = np.abs(weighted).max(axis=0) * 2 * half_width > 1e-4 * tol
    if not np.any(keep):
        return np.zeros(times.shape, complex)
    coefficients = _PANEL_PROJECTION @ weighted[:, keep]  # (n_nodes, n_keep)
    phase_rate = np.multiply.outer(times, rates[keep])  # (n_times, n_keep)
    orders = np.arange(_N_NODES)[:, None, None]
    bessel = spherical_jn(orders, np.abs(phase_rate * half_width)[None])
    sign = np.where(phase_rate < 0, (-1.0) ** orders, 1.0)
    value = np.einsum(
        "k,ktn,kn->tn",
        2 * (1j) ** np.arange(_N_NODES),
        bessel * sign,
        coefficients,
    )
    return (half_width * np.exp(1j * phase_rate * centre) * value).sum(axis=1)


def _harmonics(panel: _Panel) -> FloatArray:
    """Get the harmonic numbers n = -H..H of a panel."""
    n_harmonics = (panel.isf_harmonics.shape[-1] - 1) // 2
    return np.arange(-n_harmonics, n_harmonics + 1).astype(np.float64)


@dataclass(frozen=True, kw_only=True)
class ZeroFrictionOrbits:
    """The trapped and running orbits of a 1D periodic potential at zero friction, on panels of their frequency."""

    families: tuple[tuple[_Family, tuple[_Panel, ...]], ...]

    def isf(self, times: FloatArray, tol: float = 1e-6) -> FloatArray | ComplexArray:
        """Get the ISF at the wavevector of the orbits, at the times."""
        times = np.asarray(times, float)
        out = np.zeros(times.shape, complex)
        for family, panels in self.families:
            for panel in panels:
                rates = _harmonics(panel) + family.direction * family.wavevector
                weighted = panel.weight[:, None] * panel.isf_harmonics
                contribution = _get_oscillating_integral(
                    weighted, rates, times, panel, tol
                )
                out += (
                    contribution.real * family.multiplicity
                    if family.direction != 0
                    else contribution
                )
        return out.real if np.all(np.abs(out.imag) < _REAL_TOL) else out

    def msd(self, times: FloatArray, tol: float = 1e-6) -> FloatArray:
        """Get the mean square displacement at the times."""
        times = np.asarray(times, float)
        out = np.zeros(times.shape)
        for family, panels in self.families:
            for panel in panels:
                quadrature = (
                    panel.weight * _PANEL_WEIGHTS * (0.5 * (panel.hi - panel.lo))
                )
                if family.direction != 0:  # ballistic part
                    out += (
                        family.multiplicity
                        * np.sum(quadrature * panel.frequency**2)
                        * times**2
                    )
                harmonics = _harmonics(panel)
                nonzero = harmonics != 0
                msd_harmonics = panel.msd_harmonics[:, nonzero]
                out += family.multiplicity * np.sum(
                    quadrature[:, None] * 2 * msd_harmonics
                )  # constant part
                weighted = 2 * panel.weight[:, None] * msd_harmonics
                out -= (
                    family.multiplicity
                    * _get_oscillating_integral(
                        weighted, harmonics[nonzero], times, panel, tol
                    ).real
                )
        return out


def build(
    series: CosineSeries, wavevector: float, tol: float = 1e-6
) -> ZeroFrictionOrbits:
    """Get the trapped and the right-running orbits on adaptive panels of their frequency, to the tolerance."""
    log_partition = series.log_partition()
    trapped, running = (
        _Family(
            series=series,
            direction=direction,
            wavevector=wavevector,
            tol=tol,
            log_partition=log_partition,
        )
        for direction in (0, 1)
    )
    top_energy = series.barrier + max(40.0, -np.log(tol) + 30)
    top_running = float(np.sqrt(2 * (top_energy - series.barrier)))
    return ZeroFrictionOrbits(
        families=(
            (trapped, tuple(_get_panels(trapped, series.well_frequency))),
            (running, tuple(_get_panels(running, top_running))),
        )
    )
