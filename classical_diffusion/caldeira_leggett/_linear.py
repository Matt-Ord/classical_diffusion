"""Linear solves of a Liouvillian, (L - s) X = rhs, in the eigenvectors of its effective Hamiltonians, preconditioned by the exact inverse of its no-jump part."""

# cspell: ignore maxiter bicgstab Hamiltonians vdot
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from classical_diffusion.caldeira_leggett._liouvillian import add_jumps

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.caldeira_leggett._config import CaldeiraLeggettConfig
    from classical_diffusion.caldeira_leggett._liouvillian import Liouvillian


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class EigenLiouvillian:
    """A Liouvillian in the eigenvectors of K_ket = A diag(mu) A^-1 and K_bra = B diag(nu) B^-1, where its no-jump part is elementwise.

    A state X = A Y B^dagger at each kappa is held as Y = A^-1 X B^-dagger, in
    which N X = -i (K_ket X - X K_bra^dagger) is Y -> Z * Y with
    Z = -i (mu[a] - nu[b]^*), so that (N - s)^-1 is exact for any shift s, and
    the jumps J X J'^dagger from kappa to kappa' = t(kappa) are

        (A_{kappa'}^-1 J A_kappa) Y (B_kappa^dagger J'^dagger B_{kappa'}^-dagger),

    stored transformed. One application of L is then two products for each
    jump, rather than two more for the Hamiltonian and four for each
    application of the preconditioner: for hydrogen on Cu(111) in 2D, each
    iteration of a solve is 1.36 times faster.
    """

    rates: jax.Array
    """Z, (n_kappa, n_bands, n_bands)."""
    jumps_ket: jax.Array
    jumps_bra_dagger: jax.Array
    sources: jax.Array
    """(n_ops, n_kappa), the index of the kappa from which each jump reaches kappa."""
    ket_vectors: jax.Array
    """A."""
    ket_inverse: jax.Array
    bra_vectors_dagger: jax.Array
    """B^dagger."""
    bra_inverse_dagger: jax.Array
    trace: jax.Array
    """W = (B^dagger A)^T, so that Tr[X] = sum(W * Y)."""


def get_eigen_liouvillian(liouvillian: Liouvillian) -> EigenLiouvillian:
    mu, a = np.linalg.eig(np.asarray(liouvillian.hamiltonian_ket))
    nu, b = np.linalg.eig(
        np.asarray(liouvillian.hamiltonian_bra_dagger).conj().swapaxes(-1, -2)
    )
    a_inverse = np.linalg.inv(a)
    b_dagger = b.conj().swapaxes(-1, -2)
    b_inverse_dagger = np.linalg.inv(b).conj().swapaxes(-1, -2)
    sources = np.asarray(liouvillian.sources)
    # The jump j takes kappa to targets[j, kappa], as sources[j, targets[j, kappa]] = kappa
    targets = np.argsort(sources, axis=-1)
    jumps_ket = np.asarray(liouvillian.jumps_ket)
    jumps_bra_dagger = np.asarray(liouvillian.jumps_bra_dagger)
    return EigenLiouvillian(
        rates=jnp.asarray(-1j * (mu[:, :, None] - nu.conj()[:, None, :])),
        jumps_ket=jnp.asarray(a_inverse[targets] @ jumps_ket @ a),
        jumps_bra_dagger=jnp.asarray(
            b_dagger @ jumps_bra_dagger @ b_inverse_dagger[targets]
        ),
        sources=jnp.asarray(sources),
        ket_vectors=jnp.asarray(a),
        ket_inverse=jnp.asarray(a_inverse),
        bra_vectors_dagger=jnp.asarray(b_dagger),
        bra_inverse_dagger=jnp.asarray(b_inverse_dagger),
        trace=jnp.asarray((b_dagger @ a).swapaxes(-1, -2)),
    )


def to_eigen(eigen: EigenLiouvillian, x: jax.Array) -> jax.Array:
    """Get Y = A^-1 X B^-dagger."""
    return eigen.ket_inverse @ x @ eigen.bra_inverse_dagger


def from_eigen(eigen: EigenLiouvillian, y: jax.Array) -> jax.Array:
    """Get X = A Y B^dagger."""
    return eigen.ket_vectors @ y @ eigen.bra_vectors_dagger


@jax.jit
def apply_eigen(eigen: EigenLiouvillian, y: jax.Array, shift: jax.Array) -> jax.Array:
    """Apply L - shift to a state in the eigenvectors, adding the jumps one at a time."""
    return add_jumps(
        (eigen.rates - shift) * y,
        y,
        (eigen.jumps_ket, eigen.jumps_bra_dagger, eigen.sources),
    )


def bicgstab(
    operator: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    rtol: jax.Array,
    maxiter: int,
) -> tuple[jax.Array, jax.Array]:
    """Solve A x = rhs from x = 0 by BiCGSTAB, and get the number of iterations, each two applications of A.

    It stops once |rhs - A x| <= rtol |rhs|, after maxiter iterations, or on
    a breakdown, which the next pass of `refine` recovers from.
    """
    target = rtol * jnp.linalg.norm(rhs)
    zero = jnp.zeros_like(rhs)
    one = jnp.asarray(1.0 + 0j)

    def proceed(carry: tuple) -> jax.Array:
        _x, r, _p, _v, rho, _alpha, omega, k = carry
        return (k < maxiter) & (jnp.linalg.norm(r) > target) & (rho != 0) & (omega != 0)

    def iterate(carry: tuple) -> tuple:
        x, r, p, v, rho, alpha, omega, k = carry
        rho_new = jnp.vdot(rhs, r)
        p = r + (rho_new / rho) * (alpha / omega) * (p - omega * v)
        v = operator(p)
        alpha = rho_new / jnp.vdot(rhs, v)
        s = r - alpha * v
        t = operator(s)
        tt = jnp.vdot(t, t)
        # s = 0 once the half step has converged, and then so is t
        omega = jnp.where(tt != 0, jnp.vdot(t, s) / jnp.where(tt != 0, tt, 1), 0)
        return (
            x + alpha * p + omega * s,
            s - omega * t,
            p,
            v,
            rho_new,
            alpha,
            omega,
            k + 1,
        )

    x, *_, k = jax.lax.while_loop(
        proceed, iterate, (zero, rhs, zero, zero, one, one, one, jnp.asarray(0))
    )
    return x, k


def refine(  # ruff:ignore[too-many-arguments]
    apply: Callable[[jax.Array], jax.Array],
    solve_pass: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    rhs: jax.Array,
    *,
    rtol: float,
    inner_rtol: float,
    passes: int,
) -> tuple[jax.Array, int]:
    """Solve A x = rhs by passes of an inner solver, each from the true residual (iterative refinement), and get the applications of A.

    solve_pass(residual, rtol) gives an approximate solution and its BiCGSTAB
    iterations. Its residual is not monotonic, so each pass only reduces the
    true residual by inner_rtol (or to rtol), and the next starts afresh from
    that residual, which also recovers from a breakdown. The passes stop once
    the residual is below rtol of |rhs|, or once it no longer halves, when
    round off has set its floor.

    Raises
    ------
    RuntimeError
        If the residual is still falling after the passes.
    """
    norm = max(float(jnp.linalg.norm(rhs)), 1e-300)
    solution, residual, best = jnp.zeros_like(rhs), rhs, 1.0
    applications = 0
    for _ in range(passes):
        step, iterations = solve_pass(
            residual, jnp.asarray(max(inner_rtol, rtol / best))
        )
        update = solution + step
        update_residual = rhs - apply(update)
        applications += 2 * int(iterations) + 1
        relative = float(jnp.linalg.norm(update_residual)) / norm
        if relative > best / 2:
            return solution, applications
        solution, residual, best = update, update_residual, relative
        if best <= rtol:
            return solution, applications
    msg = f"BiCGSTAB reached a residual of {best:.1e}, above {rtol:.1e}, after {passes} passes"
    raise RuntimeError(msg)


@partial(jax.jit, static_argnames=("maxiter",))
def _solve_pass(
    eigen: EigenLiouvillian, residual: jax.Array, rtol: jax.Array, *, maxiter: int
) -> tuple[jax.Array, jax.Array]:
    # Preconditioned on the right, L M y = residual with x = M y and M = N^-1,
    # so that the tolerance applies to the true residual rather than to
    # M (b - L x)
    y, iterations = bicgstab(
        lambda y: apply_eigen(eigen, y / eigen.rates, jnp.asarray(0j)),
        residual,
        rtol,
        maxiter,
    )
    return y / eigen.rates, iterations


def solve(
    eigen: EigenLiouvillian, rhs: jax.Array, config: CaldeiraLeggettConfig
) -> jax.Array:
    """Solve L X = rhs for a state X in the basis of the bands, for a rhs in the range of L.

    BiCGSTAB in the eigenvectors of K, preconditioned on the right by N^-1, N
    the no-jump part of L, with iterative refinement (`refine`). BiCGSTAB
    keeps a fixed handful of vectors, where GMRES keeps a basis which grows
    until it restarts, and a short restart can stall: for hydrogen on
    Cu(111) in 2D, GMRES(30) takes 18 times longer to find D, in 4 times the
    memory. It raises a RuntimeError if the residual is still falling after
    linear_refinements passes.

    L is singular, and the round off of the trace of L X is larger in the
    eigenvectors of K, which are not orthogonal, than in the bands. Removing
    it in each iteration changes neither the applications of L (within 3%)
    nor D (within 5e-10) for H, Li and Na on Cu(111), as the refinement stops
    once the residual no longer halves.
    """
    y, _ = refine(
        lambda y: apply_eigen(eigen, y, jnp.asarray(0j)),
        lambda residual, rtol: _solve_pass(
            eigen, residual, rtol, maxiter=config.linear_maxiter
        ),
        to_eigen(eigen, rhs),
        rtol=config.linear_rtol,
        inner_rtol=config.linear_inner_rtol,
        passes=config.linear_refinements,
    )
    return from_eigen(eigen, y)
