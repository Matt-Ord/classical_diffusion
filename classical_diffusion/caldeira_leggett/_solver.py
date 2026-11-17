# cspell: ignore jkab jkac maxiter bicgstab
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
import numpy as np

from classical_diffusion.caldeira_leggett._krylov import KrylovExponential
from classical_diffusion.caldeira_leggett._liouvillian import (
    Liouvillian,
    Operators,
    apply_liouvillian,
    get_liouvillian,
)

# jax.jit reads the annotations of _propagate, which needs Adaptive at run time
from classical_diffusion.jax.langevin import Adaptive  # ruff: ignore[typing-only-first-party-import]
from classical_diffusion.simulation import IsfResult

if TYPE_CHECKING:
    from classical_diffusion.caldeira_leggett._config import CaldeiraLeggettConfig
    from classical_diffusion.caldeira_leggett._model import ComplexArray, FloatArray
    from classical_diffusion.system import UnitSystem


class _NoJumpInverse(NamedTuple):
    """The inverse of the no-jump part -i (K X - X K^dagger) of the Liouvillian of the steady state, from the eigenvectors of K = V diag(mu) V^-1.

    Writing X = V Y V^dagger, the no-jump part is elementwise,
    Y -> Z * Y with Z = -i (mu[a] - mu[b]^*). It preconditions the linear solves.
    """

    vectors: jax.Array
    inverse: jax.Array
    vectors_dagger: jax.Array
    inverse_dagger: jax.Array
    inverse_rates: jax.Array


def _get_no_jump_inverse(operators: Operators) -> _NoJumpInverse:
    mu, v = np.linalg.eig(operators.hamiltonian)
    rates = -1j * (mu[:, :, None] - mu.conj()[:, None, :])
    inverse = np.linalg.inv(v)
    return _NoJumpInverse(
        vectors=jnp.asarray(v),
        inverse=jnp.asarray(inverse),
        vectors_dagger=jnp.asarray(v.conj().swapaxes(-1, -2)),
        inverse_dagger=jnp.asarray(inverse.conj().swapaxes(-1, -2)),
        inverse_rates=jnp.asarray(1 / rates),
    )


def _apply_no_jump_inverse(inverse: _NoJumpInverse, x: jax.Array) -> jax.Array:
    y = inverse.inverse @ x @ inverse.inverse_dagger
    return inverse.vectors @ (inverse.inverse_rates * y) @ inverse.vectors_dagger


@partial(jax.jit, static_argnames=("maxiter",))
def _bicgstab(
    liouvillian: Liouvillian,
    inverse: _NoJumpInverse,
    residual: jax.Array,
    rtol: jax.Array,
    *,
    maxiter: int,
) -> jax.Array:
    # Preconditioned on the right, L M y = residual with x = M y, so that the
    # tolerance applies to the true residual rather than to M (b - L x)
    y, _ = jax.scipy.sparse.linalg.bicgstab(
        lambda y: apply_liouvillian(liouvillian, _apply_no_jump_inverse(inverse, y)),
        residual,
        tol=rtol,
        atol=0.0,
        maxiter=maxiter,
    )
    return _apply_no_jump_inverse(inverse, y)


def _solve(
    liouvillian: Liouvillian,
    inverse: _NoJumpInverse,
    rhs: jax.Array,
    config: CaldeiraLeggettConfig,
) -> jax.Array:
    """Solve L X = rhs with BiCGSTAB, preconditioned on the right by the inverse of the no-jump part of L.

    BiCGSTAB keeps a fixed handful of vectors, where GMRES keeps a basis
    which grows until it restarts, and a short restart can stall: for
    hydrogen on Cu(111) in 2D, GMRES(30) takes 18 times longer to find D, in 4
    times the memory. Its residual is not monotonic, so each pass of BiCGSTAB
    only reduces the true residual by linear_inner_rtol, and the next pass
    starts afresh from that residual (iterative refinement), which also
    recovers from a breakdown. The passes stop once the residual is below
    linear_rtol of |rhs|, or once it no longer halves, when round off has set
    its floor.

    Raises
    ------
    RuntimeError
        If the residual is still falling after linear_refinements passes.
    """
    norm = float(jnp.linalg.norm(rhs))
    solution, residual, best = jnp.zeros_like(rhs), rhs, 1.0
    for _ in range(config.linear_refinements):
        update = solution + _bicgstab(
            liouvillian,
            inverse,
            residual,
            jnp.asarray(max(config.linear_inner_rtol, config.linear_rtol / best)),
            maxiter=config.linear_maxiter,
        )
        update_residual = rhs - apply_liouvillian(liouvillian, update)
        relative = float(jnp.linalg.norm(update_residual)) / norm
        if relative > best / 2:
            return solution
        solution, residual, best = update, update_residual, relative
        if best <= config.linear_rtol:
            return solution
    msg = (
        f"BiCGSTAB reached a residual of {best:.1e}, above {config.linear_rtol:.1e}, "
        f"after {config.linear_refinements} passes"
    )
    raise RuntimeError(msg)


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
    rho = gibbs + np.asarray(
        _solve(liouvillian, _get_no_jump_inverse(operators), rhs, config)
    )
    rho /= np.einsum("kaa->", rho)
    return 0.5 * (rho + rho.conj().swapaxes(-1, -2))


def _complex_to_real(x: jax.Array) -> jax.Array:
    return jnp.stack([x.real, x.imag])


def _real_to_complex(y: jax.Array) -> jax.Array:
    return y[0] + 1j * y[1]


def _solve_linear[S](
    vector_field: Any,  # ruff:ignore[any-type]
    initial: S,
    measure: Any,  # ruff:ignore[any-type]
    times: jax.Array,
    step_config: Adaptive,
) -> jax.Array:
    """Propagate the linear equation dy/dt = vector_field(y) of complex arrays from y(0) = initial, and measure it at the times.

    Each complex array is split into its real and imaginary parts, as
    diffrax supports complex states only experimentally.
    """
    return dfx.diffeqsolve(
        dfx.ODETerm(
            lambda _t, y, _args: jax.tree.map(
                _complex_to_real, vector_field(jax.tree.map(_real_to_complex, y))
            )
        ),
        KrylovExponential(),
        t0=times[0],
        t1=times[-1],
        dt0=step_config.initial_step(times),
        y0=jax.tree.map(_complex_to_real, initial),
        saveat=dfx.SaveAt(
            ts=times,
            fn=lambda _t, y, _args: measure(jax.tree.map(_real_to_complex, y)),
        ),
        # The steps need not end on the times, which are found within a step
        # exactly (`KrylovInterpolation`)
        stepsize_controller=dfx.PIDController(
            rtol=step_config.rtol, atol=step_config.atol
        ),
        # Nothing is differentiated, so the number of steps need not be bounded
        adjoint=dfx.ForwardMode(),
        max_steps=None,
    ).ys


@partial(jax.jit, static_argnames=("step_config",))
def _propagate(
    liouvillian: Liouvillian,
    initial: jax.Array,
    observable: jax.Array,
    times: jax.Array,
    *,
    step_config: Adaptive,
) -> jax.Array:
    return _solve_linear(
        lambda x: apply_liouvillian(liouvillian, x),
        initial,
        lambda x: jnp.sum(observable.conj() * x),
        times,
        step_config,
    )


def get_isf(  # ruff:ignore[too-many-arguments]
    liouvillian: Liouvillian,
    initial: ComplexArray,
    observable: ComplexArray,
    times: FloatArray,
    *,
    delta_k: tuple[float, ...],
    units: UnitSystem,
    step_config: Adaptive,
) -> IsfResult:
    """Get the ISF Tr[O^dagger X(t)], where X(t) is propagated from the initial state, at the times and scattering vector in the units.

    Each step is an exponential exp(h L) in a Krylov basis
    (`KrylovExponential`), whose length is set by its error alone, to the
    tolerances of step_config, and not by the spacing of the times.
    """
    isf = _propagate(
        liouvillian,
        jnp.asarray(initial),
        jnp.asarray(observable),
        jnp.asarray(times),
        step_config=step_config,
    )
    return IsfResult(times=times, isf=np.asarray(isf), delta_k=delta_k, units=units)


class _Twist(NamedTuple):
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
        operators._replace(hamiltonian=np.zeros_like(derivative.hamiltonian)),
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
    twist: _Twist,
    inverse: _NoJumpInverse,
    source: jax.Array,
    steady_state: jax.Array,
    config: CaldeiraLeggettConfig,
) -> jax.Array:
    """Solve L_0 X = source, for a source with no trace, with no component along the steady state (X has no trace)."""
    response = _solve(twist.liouvillian, inverse, source, config)
    return response - _trace(response) * steady_state


def get_diffusion_coefficient(
    operators: Operators,
    derivative: Operators,
    steady_state: ComplexArray,
    config: CaldeiraLeggettConfig,
) -> float:
    """Get the diffusion coefficient D along a direction, from the curvature of the dephasing rate at dK = 0.

    The slowest eigenvalue of L(theta) (`_get_twist`) is -D theta^2 +
    O(theta^4), and second order perturbation theory around the steady state
    gives

        -D = Tr[L_2 rho_ss] - Tr[L_1 L_0^+ L_1 rho_ss].

    This is a Green-Kubo formula for the master equation, which needs a single
    linear solve rather than an ISF which decays only slowly at small dK. It
    is the long time slope of the mean square displacement over 2
    (`get_rms`).
    """
    twist = _get_twist(operators, derivative, steady_state)
    response = _get_response(
        twist,
        _get_no_jump_inverse(operators),
        twist.drift,
        jnp.asarray(steady_state),
        config,
    )
    return _get_diffusion_from_response(twist, response)


def _get_diffusion_from_response(twist: _Twist, response: jax.Array) -> float:
    rate = twist.second_derivative_trace - complex(
        _trace(apply_liouvillian(twist.first_derivative, response))
    )
    return -rate.real


@partial(jax.jit, static_argnames=("step_config",))
def _propagate_first_derivative(
    twist: _Twist, initial: jax.Array, times: jax.Array, *, step_config: Adaptive
) -> jax.Array:
    """Re Tr[L_1 exp(L_0 t) initial] at the times."""
    return _solve_linear(
        lambda x: apply_liouvillian(twist.liouvillian, x),
        initial,
        lambda x: _trace(apply_liouvillian(twist.first_derivative, x)).real,
        times,
        step_config,
    )


class RmsResult(NamedTuple):
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
    step_config: Adaptive,
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

    Only the last term is propagated, from a state which decays, so the time
    steps grow as it does, as for an ISF, rather than following the growing
    integral. It costs two linear solves and one propagation.
    """
    twist = _get_twist(operators, derivative, steady_state)
    inverse = _get_no_jump_inverse(operators)
    rho = jnp.asarray(steady_state)
    first = _get_response(twist, inverse, twist.drift, rho, config)
    second = _get_response(twist, inverse, first, rho, config)
    diffusion = _get_diffusion_from_response(twist, first)
    decay = _propagate_first_derivative(
        twist, second, jnp.asarray(times), step_config=step_config
    )
    mean_square = 2 * diffusion * times + 2 * (np.asarray(decay)[0] - np.asarray(decay))
    return RmsResult(
        times=times,
        mean_square=mean_square,
        direction=direction,
        units=units,
    )
