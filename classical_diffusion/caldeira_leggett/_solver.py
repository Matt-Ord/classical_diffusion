# cspell: ignore jkab jkac maxiter
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
import numpy as np

from classical_diffusion.caldeira_leggett._liouvillian import get_twist_derivative

if TYPE_CHECKING:
    from classical_diffusion.caldeira_leggett._config import CaldeiraLeggettConfig
    from classical_diffusion.caldeira_leggett._liouvillian import (
        BandOperators,
        SectorLiouvillian,
    )
    from classical_diffusion.caldeira_leggett._model import ComplexArray, FloatArray


class _Sector(NamedTuple):
    """The arrays of a sector Liouvillian, as a pytree for jax.

    A state X has shape (n_kappa, n_bands, n_bands), where X[kappa] is the
    block <kappa + s, m| rho |kappa, n>, and

        L(X) = -i (K_ket X - X K_bra^dagger) + sum_j L_j,ket X L_j,bra^dagger

    where the jump j maps the block at kappa to kappa + q_j.
    """

    k_ket: jax.Array
    k_bra_dagger: jax.Array
    jumps_ket: jax.Array
    """(n_ops, n_kappa, n_bands, n_bands), indexed by the kappa of the bra."""
    jumps_bra_dagger: jax.Array
    targets: jax.Array
    """(n_ops * n_kappa,), the index of kappa + q_j for each jump and kappa."""


class _NoJumpInverse(NamedTuple):
    """The inverse of the no-jump part -i (K_ket X - X K_bra^dagger), from the eigenvectors of K = V diag(mu) V^-1."""

    vectors_ket: jax.Array
    inverse_ket: jax.Array
    vectors_bra_dagger: jax.Array
    inverse_bra_dagger: jax.Array
    inverse_rates: jax.Array


def _get_sector(sector: SectorLiouvillian) -> _Sector:
    index = sector.ket_index
    return _Sector(
        k_ket=jnp.asarray(sector.ket.effective_hamiltonian[index]),
        k_bra_dagger=jnp.asarray(
            sector.bra.effective_hamiltonian.conj().swapaxes(-1, -2)
        ),
        jumps_ket=jnp.asarray(sector.ket.jumps[:, index]),
        jumps_bra_dagger=jnp.asarray(sector.bra.jumps.conj().swapaxes(-1, -2)),
        targets=jnp.asarray(sector.jump_targets.reshape(-1)),
    )


def _get_no_jump_inverse(sector: SectorLiouvillian) -> _NoJumpInverse:
    """Get the exact inverse of the no-jump part of the Liouvillian, which preconditions the linear solves.

    Writing X = V_ket Y V_bra^dagger, the no-jump part is elementwise,
    Y -> Z * Y with Z = -i (mu_ket[a] - mu_bra[b]^*).
    """
    mu_bra, v_bra = np.linalg.eig(sector.bra.effective_hamiltonian)
    if sector.ket is sector.bra:
        mu_ket, v_ket = mu_bra, v_bra
    else:
        mu_ket, v_ket = np.linalg.eig(sector.ket.effective_hamiltonian)
    index = sector.ket_index
    mu_ket, v_ket = mu_ket[index], v_ket[index]
    rates = -1j * (mu_ket[:, :, None] - mu_bra.conj()[:, None, :])
    inverse_bra = np.linalg.inv(v_bra)
    return _NoJumpInverse(
        vectors_ket=jnp.asarray(v_ket),
        inverse_ket=jnp.asarray(np.linalg.inv(v_ket)),
        vectors_bra_dagger=jnp.asarray(v_bra.conj().swapaxes(-1, -2)),
        inverse_bra_dagger=jnp.asarray(inverse_bra.conj().swapaxes(-1, -2)),
        inverse_rates=jnp.asarray(1 / rates),
    )


def _apply(sector: _Sector, x: jax.Array) -> jax.Array:
    """Apply the Liouvillian of the sector to the state x."""
    out = -1j * (sector.k_ket @ x - x @ sector.k_bra_dagger)
    jumped = sector.jumps_ket @ x[None] @ sector.jumps_bra_dagger
    return out.at[sector.targets].add(jumped.reshape(-1, *x.shape[1:]))


def _apply_no_jump_inverse(inverse: _NoJumpInverse, x: jax.Array) -> jax.Array:
    y = inverse.inverse_ket @ x @ inverse.inverse_bra_dagger
    return (
        inverse.vectors_ket @ (inverse.inverse_rates * y) @ inverse.vectors_bra_dagger
    )


@partial(jax.jit, static_argnames=("restart", "maxiter"))
def _gmres(  # ruff:ignore[too-many-arguments]
    sector: _Sector,
    inverse: _NoJumpInverse,
    residual: jax.Array,
    rtol: jax.Array,
    *,
    restart: int,
    maxiter: int,
) -> jax.Array:
    # Preconditioned on the right, L M y = residual with x = M y, so that the
    # tolerance applies to the true residual rather than to M (b - L x)
    y, _ = jax.scipy.sparse.linalg.gmres(
        lambda y: _apply(sector, _apply_no_jump_inverse(inverse, y)),
        residual,
        tol=rtol,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
    )
    return _apply_no_jump_inverse(inverse, y)


def _solve(
    sector: SectorLiouvillian, rhs: ComplexArray, config: CaldeiraLeggettConfig
) -> ComplexArray:
    """Solve L X = rhs with GMRES, preconditioned on the right by the inverse of the no-jump part of L.

    Restarted GMRES can stall: with a short restart, the residual can stop
    falling however many restarts follow. Each pass of GMRES therefore only
    reduces the true residual by linear_inner_rtol, and the next pass starts
    afresh from that residual (iterative refinement). The passes stop once the
    residual is below linear_rtol of |rhs|, or once it no longer halves, when
    round off has set its floor.

    Raises
    ------
    RuntimeError
        If the residual is still falling after linear_refinements passes.
    """
    arrays, inverse = _get_sector(sector), _get_no_jump_inverse(sector)
    target = jnp.asarray(rhs)
    norm = float(jnp.linalg.norm(target))
    solution, best = jnp.zeros_like(target), 1.0
    for _ in range(config.linear_refinements):
        correction = _gmres(
            arrays,
            inverse,
            target - _apply(arrays, solution),
            jnp.asarray(max(config.linear_inner_rtol, config.linear_rtol / best)),
            restart=config.linear_restart,
            maxiter=config.linear_maxiter,
        )
        relative = (
            float(jnp.linalg.norm(target - _apply(arrays, solution + correction)))
            / norm
        )
        if relative > best / 2:
            return np.asarray(solution)
        solution, best = solution + correction, relative
        if best <= config.linear_rtol:
            return np.asarray(solution)
    msg = (
        f"GMRES reached a residual of {best:.1e}, above {config.linear_rtol:.1e}, "
        f"after {config.linear_refinements} passes"
    )
    raise RuntimeError(msg)


def _trace(x: ComplexArray) -> complex:
    return complex(np.trace(x, axis1=-2, axis2=-1).sum())


def get_steady_state(
    liouvillian: SectorLiouvillian, config: CaldeiraLeggettConfig
) -> ComplexArray:
    """Get the steady state in the basis of the bands, by a preconditioned linear solve starting from a Gibbs state.

    L is singular, but -L rho_gibbs is in its range. The null component of the
    correction only changes the normalization, which is restored after.
    """
    energies = liouvillian.bands.energies
    gibbs = np.zeros(liouvillian.shape, dtype=np.complex128)
    diagonal = np.arange(liouvillian.shape[1])
    gibbs[:, diagonal, diagonal] = np.exp(-(energies - energies.min()))
    gibbs /= np.sum(gibbs[:, diagonal, diagonal])
    rhs = -np.asarray(_apply(_get_sector(liouvillian), jnp.asarray(gibbs)))
    rho = gibbs + _solve(liouvillian, rhs, config)
    rho /= np.sum(rho[:, diagonal, diagonal])
    return 0.5 * (rho + rho.conj().swapaxes(-1, -2))


@jax.jit
def _get_spectral_radius(sector: _Sector, x: jax.Array) -> jax.Array:
    """Get the largest |eigenvalue| of L, to about 10%, by 30 steps of power iteration from x."""

    def step(x: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        y = _apply(sector, x)
        norm = jnp.linalg.norm(y)
        return y / norm, norm

    _, growth = jax.lax.scan(step, x / jnp.linalg.norm(x), None, length=30)
    return growth[-1]


@partial(jax.jit, static_argnames=("rtol", "atol"))
def _propagate(  # ruff:ignore[too-many-arguments]
    sector: _Sector,
    initial: jax.Array,
    observable: jax.Array,
    times: jax.Array,
    dt0: jax.Array,
    *,
    rtol: float,
    atol: float,
) -> jax.Array:
    def vector_field(
        _t: Any,  # ruff:ignore[any-type]
        y: jax.Array,
        _args: Any,  # ruff:ignore[any-type]
    ) -> jax.Array:
        dx = _apply(sector, y[0] + 1j * y[1])
        return jnp.stack([dx.real, dx.imag])

    def measure(
        _t: Any,  # ruff:ignore[any-type]
        y: jax.Array,
        _args: Any,  # ruff:ignore[any-type]
    ) -> jax.Array:
        return jnp.sum(observable.conj() * (y[0] + 1j * y[1]))

    return dfx.diffeqsolve(
        dfx.ODETerm(vector_field),
        dfx.Tsit5(),  # cspell: disable-line
        t0=times[0],
        t1=times[-1],
        dt0=dt0,
        y0=jnp.stack([initial.real, initial.imag]),
        saveat=dfx.SaveAt(ts=times, fn=measure),
        stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
        adjoint=dfx.ForwardMode(),
        max_steps=None,
    ).ys


def get_isf(
    sector: SectorLiouvillian,
    steady_state: ComplexArray,
    times: FloatArray,
    config: CaldeiraLeggettConfig,
) -> ComplexArray:
    """Get I(dK, t) = Tr[e^{i dK x} V(t) (rho_ss e^{-i dK x})] in the scattering sector of dK, by adaptive time propagation.

    The scattering operator is that of the point of the reciprocal lattice of
    the cell of the sector, n = -shift; the rest of dK is in the offset of the
    grid of the bra. The steps are chosen by a PID controller. The fastest
    coherences, between the highest and lowest bands, limit the stable step of
    the explicit Runge-Kutta method, so the first step is the inverse of the
    largest |eigenvalue| of L.
    """
    # S[kappa] = <kappa - dK, m| e^{-i dK x} |kappa, n>
    scattering = sector.bands.shift_operator(sector.shift)
    # (rho_ss e^{-i dK x}) at the block (kappa - dK, kappa) is rho[kappa - dK] S[kappa]
    initial = jnp.asarray(steady_state[sector.ket_index] @ scattering)
    arrays = _get_sector(sector)
    radius = _get_spectral_radius(
        arrays, jax.random.normal(jax.random.key(0), initial.shape) + 0j
    )
    isf = _propagate(
        arrays,
        initial,
        jnp.asarray(scattering),
        jnp.asarray(times),
        jnp.minimum(1 / radius, times[-1]),
        rtol=config.ode_rtol,
        atol=config.ode_atol,
    )
    return np.asarray(isf)


def _get_first_derivative(
    operators: BandOperators, jumps: ComplexArray, hamiltonian: ComplexArray
) -> _Sector:
    """Get L_1 X = i X K'^dagger + sum_j L_j X L_j'^dagger, as a sector whose ket has no Hamiltonian."""
    return _Sector(
        k_ket=jnp.zeros_like(jnp.asarray(hamiltonian)),
        k_bra_dagger=jnp.asarray(hamiltonian.conj().swapaxes(-1, -2)),
        jumps_ket=jnp.asarray(operators.jumps),
        jumps_bra_dagger=jnp.asarray(jumps.conj().swapaxes(-1, -2)),
        targets=jnp.asarray(operators.jump_targets.reshape(-1)),
    )


def get_diffusion_coefficient(
    liouvillian: SectorLiouvillian,
    steady_state: ComplexArray,
    direction: FloatArray,
    config: CaldeiraLeggettConfig,
) -> float:
    """Get the diffusion coefficient D along a unit direction, from the curvature of the dephasing rate at dK = 0.

    After the gauge transformation of `get_scattering_sector`, the ISF at a
    small dK = theta n is I(t) = Tr[exp(L(theta) t) rho_ss], where L(theta) is
    the Liouvillian with a bra on the grid offset by theta n. Its slowest
    eigenvalue is -D theta^2 + O(theta^4), and second order perturbation
    theory around the steady state, with L(theta) = L_0 + theta L_1 +
    theta^2 L_2 + ..., gives

        -D = Tr[L_2 rho_ss] - Tr[L_1 L_0^+ L_1 rho_ss].

    This is a Green-Kubo formula for the master equation, which needs a single
    linear solve rather than an ISF which decays only slowly at small dK.

    With the derivatives O' of the operators of the bra (`get_twist_derivative`),

        L_1 X = i X K'^dagger + sum_j L_j X L_j'^dagger.

    L_2 needs no second derivative: the Liouvillian with both the ket and the
    bra on the offset grid preserves the trace for every theta, so the second
    derivative of its trace vanishes, which gives
    Re Tr[L_2 rho_ss] = -1/2 sum_j Tr[L_j' rho_ss L_j'^dagger].
    """
    operators = liouvillian.bra
    jumps, hamiltonian = get_twist_derivative(operators, direction)
    first = _get_first_derivative(operators, jumps, hamiltonian)
    second_order = -0.5 * np.einsum(
        "jkab,kbc,jkac->", jumps, steady_state, jumps.conj()
    )
    drift = np.asarray(_apply(first, jnp.asarray(steady_state)))
    # Tr[L_1 rho_ss] is the first order rate, which vanishes without a net drift
    response = _solve(liouvillian, drift - _trace(drift) * steady_state, config)
    rate = second_order - _trace(np.asarray(_apply(first, jnp.asarray(response))))
    return -rate.real
