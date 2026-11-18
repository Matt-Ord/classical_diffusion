"""Linear solves of a Liouvillian, (L - s) X = rhs, in the eigenvectors of its effective Hamiltonians, preconditioned by the exact inverse of its no-jump part."""

# cspell: ignore maxiter bicgstab
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
import numpy as np

if TYPE_CHECKING:
    from classical_diffusion.caldeira_leggett._config import CaldeiraLeggettConfig
    from classical_diffusion.caldeira_leggett._liouvillian import Liouvillian


class EigenLiouvillian(NamedTuple):
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

    def add_jump(
        out: jax.Array, jump: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, None]:
        ket, bra_dagger, sources = jump
        return out + (ket @ y @ bra_dagger)[sources], None

    out, _ = jax.lax.scan(
        add_jump,
        (eigen.rates - shift) * y,
        (eigen.jumps_ket, eigen.jumps_bra_dagger, eigen.sources),
    )
    return out


def _remove_trace(eigen: EigenLiouvillian, y: jax.Array) -> jax.Array:
    """Remove the trace of a state Y along W^*, which keeps it in the range of L."""
    w = eigen.trace.conj()
    return y - w * (jnp.sum(eigen.trace * y) / jnp.sum(eigen.trace * w))


@partial(jax.jit, static_argnames=("maxiter", "singular"))
def _bicgstab(  # ruff:ignore[too-many-arguments]
    eigen: EigenLiouvillian,
    residual: jax.Array,
    rtol: jax.Array,
    shift: jax.Array,
    *,
    maxiter: int,
    singular: bool,
) -> jax.Array:
    # Preconditioned on the right, (L - s) M y = residual with x = M y and
    # M = (N - s)^-1, so that the tolerance applies to the true residual rather
    # than to M (b - L x)
    rates = eigen.rates - shift

    def operator(y: jax.Array) -> jax.Array:
        out = apply_eigen(eigen, y / rates, shift)
        # L conserves the trace, so L X has none, but the eigenvectors of K
        # are not orthogonal and the round off of the trace of L X is then
        # larger than in the bands. At s = 0, where L is singular, BiCGSTAB
        # amplifies it once the residual nears it, so it is removed
        return _remove_trace(eigen, out) if singular else out

    y, _ = jax.scipy.sparse.linalg.bicgstab(
        operator,
        _remove_trace(eigen, residual) if singular else residual,
        tol=rtol,
        atol=0.0,
        maxiter=maxiter,
    )
    return y / rates


def solve_eigen(
    eigen: EigenLiouvillian,
    rhs: jax.Array,
    config: CaldeiraLeggettConfig,
    *,
    shift: complex = 0.0,
    guess: jax.Array | None = None,
) -> jax.Array:
    """Solve (L - shift) Y = rhs in the eigenvectors with BiCGSTAB, preconditioned on the right by (N - shift)^-1, N the no-jump part of L.

    The solve starts from the guess, if there is one and its residual is
    below |rhs|, and from zero otherwise.

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
    s = jnp.asarray(shift, dtype=complex)
    norm = float(jnp.linalg.norm(rhs))
    solution, residual, best = jnp.zeros_like(rhs), rhs, 1.0
    if guess is not None:
        guess_residual = rhs - apply_eigen(eigen, guess, s)
        relative = float(jnp.linalg.norm(guess_residual)) / norm
        if relative < best:
            solution, residual, best = guess, guess_residual, relative
    for _ in range(config.linear_refinements):
        update = solution + _bicgstab(
            eigen,
            residual,
            jnp.asarray(max(config.linear_inner_rtol, config.linear_rtol / best)),
            s,
            maxiter=config.linear_maxiter,
            singular=shift == 0,
        )
        update_residual = rhs - apply_eigen(eigen, update, s)
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


def solve(
    eigen: EigenLiouvillian, rhs: jax.Array, config: CaldeiraLeggettConfig
) -> jax.Array:
    """Solve L X = rhs for a state X in the basis of the bands (`solve_eigen`)."""
    return from_eigen(eigen, solve_eigen(eigen, to_eigen(eigen, rhs), config))
