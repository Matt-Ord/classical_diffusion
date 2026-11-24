# cspell: ignore jkab jkac
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from classical_diffusion.caldeira_leggett._evolution import get_reduced_evolution
from classical_diffusion.caldeira_leggett._linear import (
    EigenLiouvillian,
    get_eigen_liouvillian,
    solve,
)
from classical_diffusion.caldeira_leggett._liouvillian import (
    Liouvillian,
    Operators,
    apply_liouvillian,
    get_liouvillian,
)
from classical_diffusion.simulation import IsfResult

if TYPE_CHECKING:
    from classical_diffusion.caldeira_leggett._config import (
        CaldeiraLeggettConfig,
        EvolutionConfig,
    )
    from classical_diffusion.caldeira_leggett._model import ComplexArray, FloatArray
    from classical_diffusion.system import UnitSystem


def _trace(x: jax.Array) -> jax.Array:
    return jnp.trace(x, axis1=-2, axis2=-1).sum()


def get_steady_state(
    operators: Operators, energies: FloatArray, config: CaldeiraLeggettConfig
) -> ComplexArray:
    """Get the steady state in the basis of the bands, by a preconditioned linear solve starting from a Gibbs state.

    L is singular, but -L rho_gibbs is in its range. The null component of the
    correction only changes the normalization, which is restored after.
    """
    liouvillian = get_liouvillian(operators, operators)
    weights = np.exp(-(energies - energies.min()))
    gibbs = (weights / weights.sum())[..., None] * np.eye(energies.shape[1])
    rhs = -apply_liouvillian(liouvillian, jnp.asarray(gibbs))
    rho = gibbs + np.asarray(solve(get_eigen_liouvillian(liouvillian), rhs, config))
    rho /= np.einsum("kaa->", rho)
    return 0.5 * (rho + rho.conj().swapaxes(-1, -2))


def get_isf(  # ruff:ignore[too-many-arguments]
    liouvillian: Liouvillian,
    initial: ComplexArray,
    observable: ComplexArray,
    times: FloatArray,
    *,
    delta_k: tuple[float, ...],
    units: UnitSystem,
    isf_config: EvolutionConfig,
    steady_state: ComplexArray | None = None,
) -> IsfResult:
    """Get the ISF Tr[O^dagger X(t)], where X(t) evolves from the initial state, at the times and scattering vector in the units.

    The evolution is by two-sided Lanczos (`get_reduced_evolution`), less
    the steady state, if the Liouvillian preserves the trace.
    """
    evolution = get_reduced_evolution(
        get_eigen_liouvillian(liouvillian),
        jnp.asarray(initial),
        jnp.asarray(observable),
        times,
        config=isf_config,
        steady_state=None if steady_state is None else jnp.asarray(steady_state),
    )
    return IsfResult(times=times, isf=evolution.values, delta_k=delta_k, units=units)


@dataclass(frozen=True, kw_only=True)
class _Twist:
    """The expansion L(theta) = L_0 + theta L_1 + theta^2 L_2 + ... of the Liouvillian of a bra offset by theta along a direction, about the steady state."""

    liouvillian: Liouvillian
    """L_0."""
    first_derivative: Liouvillian
    """L_1."""
    second_derivative_trace: float
    """Re Tr[L_2 rho_ss]."""
    drift: jax.Array
    """L_1 rho_ss - Tr[L_1 rho_ss] rho_ss."""


def _get_twist(
    operators: Operators, derivative: Operators, steady_state: ComplexArray
) -> _Twist:
    """Expand the Liouvillian of a bra offset by theta along a direction to second order in theta.

    After the gauge transformation X -> X e^{i theta x}, the ISF at a small
    dK = theta n is I(t) = Tr[exp(L(theta) t) rho_ss], where L(theta) is the
    Liouvillian with a bra on the grid offset by theta n. With the derivatives
    O' of the operators of the bra along n (`get_twist_derivative`),

        L_1 X = i X K'^dagger + sum_j L_j X L_j'^dagger,

    the Liouvillian of a ket with jumps L_j and no Hamiltonian, and a bra of
    the derivatives. L_2 needs no second derivative: the Liouvillian with both
    the ket and the bra on the offset grid preserves the trace for every
    theta, so the second derivative of its trace vanishes, which gives
    Re Tr[L_2 rho_ss] = -1/2 sum_j Tr[L_j' rho_ss L_j'^dagger].
    Tr[L_1 rho_ss] is the first order rate, which vanishes without a net
    drift, and is projected out of the drift.
    """
    first_derivative = get_liouvillian(
        dataclasses.replace(
            operators, hamiltonian=np.zeros_like(derivative.hamiltonian)
        ),
        derivative,
    )
    rho = jnp.asarray(steady_state)
    drift = apply_liouvillian(first_derivative, rho)
    return _Twist(
        liouvillian=get_liouvillian(operators, operators),
        first_derivative=first_derivative,
        second_derivative_trace=float(
            -0.5
            * np.einsum(
                "jkab,kbc,jkac->",
                derivative.jumps,
                steady_state,
                derivative.jumps.conj(),
            ).real
        ),
        drift=drift - _trace(drift) * rho,
    )


def _get_response(
    eigen: EigenLiouvillian,
    source: jax.Array,
    steady_state: jax.Array,
    config: CaldeiraLeggettConfig,
) -> jax.Array:
    """Solve L_0 X = source (eigen is L_0), for a source with no trace, with no component along the steady state (X has no trace)."""
    response = solve(eigen, source, config)
    return response - _trace(response) * steady_state


def get_diffusion_coefficient(
    operators: Operators,
    derivative: Operators,
    steady_state: ComplexArray,
    config: CaldeiraLeggettConfig,
) -> tuple[float, ComplexArray]:
    """Get the diffusion coefficient D along a direction, from the curvature of the dephasing rate at dK = 0.

    The slowest eigenvalue of L(theta) (`_get_twist`) is -D theta^2 +
    O(theta^4), and second order perturbation theory around the steady state
    gives

        -D = Tr[L_2 rho_ss] - Tr[L_1 L_0^+ L_1 rho_ss].

    This is a Green-Kubo formula for the master equation, which needs a single
    linear solve rather than an ISF which decays only slowly at small dK. It
    is the long time slope of the mean square displacement over 2
    (`get_rms`). The response X_1 = L_0^+ L_1 rho_ss is returned with it,
    as the mean square displacement needs it too.
    """
    twist = _get_twist(operators, derivative, steady_state)
    response = _get_response(
        get_eigen_liouvillian(twist.liouvillian),
        twist.drift,
        jnp.asarray(steady_state),
        config,
    )
    return _get_diffusion_from_response(twist, response), np.asarray(response)


def _get_diffusion_from_response(twist: _Twist, response: jax.Array) -> float:
    rate = twist.second_derivative_trace - complex(
        _trace(apply_liouvillian(twist.first_derivative, response))
    )
    return -rate.real


@jax.jit
def _get_trace_observable(liouvillian: Liouvillian, like: jax.Array) -> jax.Array:
    """Get O with Tr[L X] = sum(conj(O) * X)."""
    (transposed,) = jax.linear_transpose(
        lambda x: _trace(apply_liouvillian(liouvillian, x)), like
    )(jnp.asarray(1.0 + 0j))
    return transposed.conj()


@dataclass(frozen=True, kw_only=True)
class RmsResult:
    """The mean square displacement <(x(t) - x(0))^2> along a direction."""

    times: FloatArray
    mean_square: FloatArray
    direction: tuple[float, ...]
    """The unit vector along which x is measured."""
    units: UnitSystem

    @property
    def rms(self) -> FloatArray:
        """The root mean square displacement."""
        return np.sqrt(self.mean_square)


def get_rms(  # ruff:ignore[too-many-arguments]
    operators: Operators,
    derivative: Operators,
    steady_state: ComplexArray,
    times: FloatArray,
    *,
    direction: tuple[float, ...],
    units: UnitSystem,
    config: CaldeiraLeggettConfig,
    isf_config: EvolutionConfig,
    response: ComplexArray | None = None,
) -> RmsResult:
    """Get the mean square displacement <(x(t) - x(0))^2> along a direction, at the times from zero.

    The ISF at dK = theta n is 1 - theta^2 <(x(t) - x(0))^2> / 2 + ..., with
    no first order term without a net drift, so the mean square displacement
    is -Re d^2/d theta^2 Tr[exp(L(theta) t) rho_ss] at theta = 0
    (`_get_twist`). This is the limit dK -> 0 of the ISF, so it is that of
    the infinite surface, with no approximation in dK. As exp(L_0 t)
    preserves the trace, expanding the exponential to second order gives

        <(x(t) - x(0))^2> = -2 t Re Tr[L_2 rho_ss]
            - 2 Re int_0^t (t - tau) Tr[L_1 exp(L_0 tau) L_1 rho_ss] d tau.

    With X_1 = L_0^+ L_1 rho_ss, the response which gives D
    (`get_diffusion_coefficient`), and X_2 = L_0^+ X_1, the integral is done
    exactly, which leaves

        <(x(t) - x(0))^2> = 2 D t + 2 Re Tr[L_1 X_2]
            - 2 Re Tr[L_1 exp(L_0 t) X_2].

    Only the last term evolves, from a state which decays, as for an ISF
    (`get_reduced_evolution`), rather than the growing integral. It costs two
    linear solves and one evolution, or one solve if X_1 (response) is
    given.
    """
    twist = _get_twist(operators, derivative, steady_state)
    eigen = get_eigen_liouvillian(twist.liouvillian)
    rho = jnp.asarray(steady_state)
    first = (
        _get_response(eigen, twist.drift, rho, config)
        if response is None
        else jnp.asarray(response)
    )
    second = _get_response(eigen, first, rho, config)
    diffusion = _get_diffusion_from_response(twist, first)
    decay = get_reduced_evolution(
        eigen,
        second,
        _get_trace_observable(twist.first_derivative, rho),
        times,
        config=isf_config,
    ).values.real
    mean_square = 2 * diffusion * times + 2 * (decay[0] - decay)
    return RmsResult(
        times=times,
        mean_square=mean_square,
        direction=direction,
        units=units,
    )
