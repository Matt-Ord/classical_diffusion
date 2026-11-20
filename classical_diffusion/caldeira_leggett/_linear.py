"""Linear solves of a Liouvillian, (L - s) X = rhs, in the eigenvectors of its effective Hamiltonians, preconditioned by the exact inverse of its no-jump part."""

# cspell: ignore maxiter bicgstab Hamiltonians vdot BLAS orthogonalize fori
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
    """The transformed jumps, (n_ops, n_kappa, n_bands, n_bands), or with at least _GAUSS_BANDS bands (n_ops, 3, n_kappa, n_bands, n_bands) as their real part, imaginary part and the sum of both (`_product`)."""
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
        jumps_ket=_parts(a_inverse[targets] @ jumps_ket @ a),
        jumps_bra_dagger=_parts(
            b_dagger @ jumps_bra_dagger @ b_inverse_dagger[targets]
        ),
        sources=jnp.asarray(sources),
        ket_vectors=jnp.asarray(a),
        ket_inverse=jnp.asarray(a_inverse),
        bra_vectors_dagger=jnp.asarray(b_dagger),
        bra_inverse_dagger=jnp.asarray(b_inverse_dagger),
        trace=jnp.asarray((b_dagger @ a).swapaxes(-1, -2)),
    )


_GAUSS_BANDS = 64
"""The number of bands from which the jumps are applied as real products (`_product`).

Per application of L, complex against real products: 2D Li (139 bands)
122 -> 79 ms, 2D H (23 bands) 5.2 -> 6.7 ms, 1D Li (17 bands) 0.06 ->
0.10 ms, as XLA spends longer on each small product than on its
arithmetic.
"""


def _parts(x: np.ndarray) -> jax.Array:
    """Get the jumps as stored: (n, ...), or with at least _GAUSS_BANDS bands the real part, imaginary part and their sum of each matrix, (n, 3, ...)."""
    if x.shape[-1] < _GAUSS_BANDS:
        return jnp.asarray(x)
    return jnp.asarray(np.stack([x.real, x.imag, x.real + x.imag], axis=1))


def _product(
    a: tuple[jax.Array, jax.Array, jax.Array], b: tuple[jax.Array, jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """Get the real and imaginary parts of a product of complex matrices, from three real products (Gauss).

    a and b are each (real, imaginary, real + imaginary). XLA multiplies
    complex matrices on the CPU at about half the flop rate of real ones
    (OpenBLAS does not): (9, 139, 139) by (9, 139, 139) takes 10.2 ms
    complex and 1.45 ms real (OpenBLAS 5.9 and 1.45 ms), so three real
    products take 4.4 ms.
    """
    p_1, p_2, p_3 = a[0] @ b[0], a[1] @ b[1], a[2] @ b[2]
    return p_1 - p_2, p_3 - p_1 - p_2


def to_eigen(eigen: EigenLiouvillian, x: jax.Array) -> jax.Array:
    """Get Y = A^-1 X B^-dagger."""
    return eigen.ket_inverse @ x @ eigen.bra_inverse_dagger


def from_eigen(eigen: EigenLiouvillian, y: jax.Array) -> jax.Array:
    """Get X = A Y B^dagger."""
    return eigen.ket_vectors @ y @ eigen.bra_vectors_dagger


@jax.jit
def apply_eigen(eigen: EigenLiouvillian, y: jax.Array, shift: jax.Array) -> jax.Array:
    """Apply L - shift to a state in the eigenvectors, adding the jumps one at a time, with at least _GAUSS_BANDS bands as real products."""
    if eigen.jumps_ket.ndim == y.ndim + 1:
        return add_jumps(
            (eigen.rates - shift) * y,
            y,
            (eigen.jumps_ket, eigen.jumps_bra_dagger, eigen.sources),
        )
    # The barriers keep XLA from fusing the parts into the products, which
    # would then no longer be calls to its matrix product
    barrier = jax.lax.optimization_barrier
    parts = barrier((y.real, y.imag, y.real + y.imag))
    real, imaginary = jnp.zeros(y.shape), jnp.zeros(y.shape)
    for ket, bra_dagger, sources in zip(
        eigen.jumps_ket, eigen.jumps_bra_dagger, eigen.sources, strict=True
    ):
        left = _product(tuple(ket), parts)
        left = barrier((*left, left[0] + left[1]))
        out = _product(left, tuple(bra_dagger))
        real += out[0][sources]
        imaginary += out[1][sources]
    return (eigen.rates - shift) * y + (real + 1j * imaginary)


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

    solve_pass(residual, rtol) gives an approximate solution and its
    applications of A. Its residual is not monotonic, so each pass only reduces the
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
        step, count = solve_pass(residual, jnp.asarray(max(inner_rtol, rtol / best)))
        update = solution + step
        update_residual = rhs - apply(update)
        applications += int(count) + 1
        relative = float(jnp.linalg.norm(update_residual)) / norm
        if relative > best / 2:
            return solution, applications
        solution, residual, best = update, update_residual, relative
        if best <= rtol:
            return solution, applications
    msg = f"BiCGSTAB reached a residual of {best:.1e}, above {rtol:.1e}, after {passes} passes"
    raise RuntimeError(msg)


MAXITER = 1000
"""The most BiCGSTAB iterations in each pass of a linear solve (`refine`)."""
INNER_RTOL = 1e-4
"""The factor by which each pass of BiCGSTAB reduces the residual."""
PASSES = 20
"""The most passes of BiCGSTAB, each from the true residual.

These were settings of `CaldeiraLeggettConfig` that nothing varied, and the
coefficient solves of `get_slow_modes` had their own 10. For H, Li and Na on
Cu(111) in 1D and H in 2D, every singular solve (steady state, D, the MSD
responses) takes 3 passes and every coefficient solve 2.
"""


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
    return y / eigen.rates, 2 * iterations


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
    PASSES passes.

    L is singular, and the round off of the trace of L X is larger in the
    eigenvectors of K, which are not orthogonal, than in the bands. Removing
    it in each iteration changes neither the applications of L (within 3%)
    nor D (within 5e-10) for H, Li and Na on Cu(111), as the refinement stops
    once the residual no longer halves.
    """
    y, _ = refine(
        lambda y: apply_eigen(eigen, y, jnp.asarray(0j)),
        lambda residual, rtol: _solve_pass(eigen, residual, rtol, maxiter=MAXITER),
        to_eigen(eigen, rhs),
        rtol=config.linear_rtol,
        inner_rtol=INNER_RTOL,
        passes=PASSES,
    )
    return from_eigen(eigen, y)


def gmres(
    operator: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    rtol: jax.Array,
    maxiter: int,
) -> tuple[jax.Array, jax.Array]:
    """Solve A x = rhs from x = 0 by GMRES without restarts, and get the number of iterations, each one application of A.

    It stops once |rhs - A x| <= rtol |rhs|, or after maxiter iterations.
    """
    beta = jnp.linalg.norm(rhs)
    basis = jnp.zeros((maxiter + 1, *rhs.shape), rhs.dtype).at[0].set(rhs / beta)
    hessenberg = jnp.zeros((maxiter + 1, maxiter), rhs.dtype)
    cosines = jnp.zeros(maxiter, rhs.dtype)
    sines = jnp.zeros(maxiter, rhs.dtype)
    g = jnp.zeros(maxiter + 1, rhs.dtype).at[0].set(beta)

    def proceed(carry: tuple) -> jax.Array:
        *_, g, k = carry
        return (k < maxiter) & (jnp.abs(g[k]) > rtol * beta)

    def iterate(carry: tuple) -> tuple:
        basis, hessenberg, cosines, sines, g, k = carry
        w = operator(basis[k])

        # Modified Gram-Schmidt over the k + 1 vectors built so far only: the
        # basis is large next to an application of L when L is cheap
        def orthogonalize(
            i: jax.Array, carry: tuple[jax.Array, jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            basis, w, h = carry
            overlap = jnp.vdot(basis[i], w)
            return basis, w - overlap * basis[i], h.at[i].set(overlap)

        basis, w, h = jax.lax.fori_loop(
            0, k + 1, orthogonalize, (basis, w, jnp.zeros(maxiter + 1, rhs.dtype))
        )
        norm = jnp.linalg.norm(w)
        basis = basis.at[k + 1].set(w / jnp.where(norm > 0, norm, 1))
        h = h.at[k + 1].set(norm)

        # Apply the previous Givens rotations to the new column, then a new one
        def rotate(i: jax.Array, h: jax.Array) -> jax.Array:
            a, b = h[i], h[i + 1]
            return (
                h.at[i]
                .set(cosines[i].conj() * a + sines[i].conj() * b)
                .at[i + 1]
                .set(-sines[i] * a + cosines[i] * b)
            )

        h = jax.lax.fori_loop(0, k, rotate, h)
        a, b = h[k], h[k + 1]
        r = jnp.sqrt(jnp.abs(a) ** 2 + jnp.abs(b) ** 2)
        r = jnp.where(r > 0, r, 1)
        c, s = a / r, b / r
        h = h.at[k].set(c.conj() * a + s.conj() * b).at[k + 1].set(0)
        g = g.at[k + 1].set(-s * g[k]).at[k].set(c.conj() * g[k])
        return (
            basis,
            hessenberg.at[:, k].set(h),
            cosines.at[k].set(c),
            sines.at[k].set(s),
            g,
            k + 1,
        )

    basis, hessenberg, _, _, g, k = jax.lax.while_loop(
        proceed, iterate, (basis, hessenberg, cosines, sines, g, jnp.asarray(0))
    )
    # Back substitution on the leading k x k of the triangular factor
    active = jnp.arange(maxiter) < k
    square = hessenberg[:maxiter]
    square = jnp.where(active[:, None] & active[None, :], square, jnp.eye(maxiter))
    y = jax.scipy.linalg.solve_triangular(square, jnp.where(active, g[:maxiter], 0))
    return jnp.tensordot(y, basis[:maxiter], axes=1), k
