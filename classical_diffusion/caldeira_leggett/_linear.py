"""Linear solves of a Liouvillian, (L - s) X = rhs, in the eigenvectors of its effective Hamiltonians, preconditioned by the exact inverse of its no-jump part."""

# cspell: ignore maxiter bicgstab Hamiltonians vdot BLAS orthogonalize fori
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

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

    The jumps K_j, B_j of every j are stacked side by side, so that an
    application is two products at each kappa whatever the number of jumps
    (`_jumps`), and L^dagger is the same products of the conjugate
    transposes.
    The products are of real matrices (`split`): from _GAUSS_BANDS bands on
    the CPU, each complex product is three real products (Gauss, `_gauss`),
    of the real part, imaginary part and their sum of each matrix, stored
    (3, ...); otherwise one real product with a block matrix
    [[Re, Im], [-Im, Re]] or [Re; Im] (`apply_split`).
    """

    rates: jax.Array
    """Z, (n_kappa, n_bands, n_bands)."""
    forward_ket: jax.Array
    """K_j[s_j(kappa)] side by side over j at each kappa, where s_j(kappa) is the source from which the jump j reaches kappa, (n_kappa, n_bands, n_ops n_bands): as the blocks [Re; Im], or as Gauss parts (3, ...)."""
    forward_bra: jax.Array
    """B_j[s] side by side over j at each source s, (n_kappa, n_bands, n_ops n_bands): as the blocks [[Re, Im], [-Im, Re]], or as Gauss parts."""
    adjoint_ket: jax.Array
    """K_j[s]^dagger side by side at each s, the jumps of L^dagger, as forward_ket."""
    adjoint_bra: jax.Array
    """B_j[s_j(t)]^dagger side by side at each t, as forward_bra."""
    sources: jax.Array
    """(n_kappa, n_ops), the source s_j(kappa) of each kappa and jump."""
    targets: jax.Array
    """(n_kappa, n_ops), the kappa each jump reaches from kappa, the inverse of sources."""
    ket_vectors: jax.Array
    """A."""
    ket_inverse: jax.Array
    bra_vectors_dagger: jax.Array
    """B^dagger."""
    bra_inverse_dagger: jax.Array


_GAUSS_BANDS = 100
"""The number of bands from which, on the CPU, each complex product of an application is three real products (Gauss) rather than one real product of block matrices (`EigenLiouvillian`).

The block products have 4/3 of the flops, but no sums of parts before and
after each product. Per application of L in a Lanczos step: 2D H (52
bands) 20.8 ms Gauss and 18.4 ms blocks, 2D Li (176 bands) 84 and 94 ms,
1D Na (22 bands) 0.067 and 0.058 ms. On a GPU the blocks are plain
products of real matrices.
"""


def _dagger(x: np.ndarray) -> np.ndarray:
    return x.conj().swapaxes(-1, -2)


def _row(x: np.ndarray) -> np.ndarray:
    """Put the matrices (n_ops, n_kappa, n, n) side by side over the jumps at each kappa, (n_kappa, n, n_ops n)."""
    n_ops, n_kappa, n, _ = x.shape
    return x.transpose(1, 2, 0, 3).reshape(n_kappa, n, n_ops * n)


def _left(x: np.ndarray, *, gauss: bool) -> jax.Array:
    """Get matrices as left factors: their Gauss parts, or the blocks [Re; Im]."""
    if gauss:
        return jnp.asarray(np.stack([x.real, x.imag, x.real + x.imag]))
    return jnp.asarray(np.concatenate([x.real, x.imag], axis=-2))


def _right(x: np.ndarray, *, gauss: bool) -> jax.Array:
    """Get matrices as right factors: their Gauss parts, or the blocks [[Re, Im], [-Im, Re]]."""
    if gauss:
        return jnp.asarray(np.stack([x.real, x.imag, x.real + x.imag]))
    return jnp.asarray(np.block([[x.real, x.imag], [-x.imag, x.real]]))


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
    # K_j[s] and B_j[s], (n_ops, n_kappa, n_bands, n_bands), for the jump from s
    ket = a_inverse[targets] @ np.asarray(liouvillian.jumps_ket) @ a
    bra = (
        b_dagger @ np.asarray(liouvillian.jumps_bra_dagger) @ b_inverse_dagger[targets]
    )
    # K_j[s_j(kappa)] and B_j[s_j(kappa)] of the jumps which reach each kappa
    ket_in, bra_in = (x[np.arange(len(x))[:, None], sources] for x in (ket, bra))
    gauss = ket.shape[-1] >= _GAUSS_BANDS and jax.default_backend() == "cpu"
    return EigenLiouvillian(
        rates=jnp.asarray(-1j * (mu[:, :, None] - nu.conj()[:, None, :])),
        forward_ket=_left(_row(ket_in), gauss=gauss),
        forward_bra=_right(_row(bra), gauss=gauss),
        adjoint_ket=_left(_row(_dagger(ket)), gauss=gauss),
        adjoint_bra=_right(_row(_dagger(bra_in)), gauss=gauss),
        sources=jnp.asarray(sources.T),
        targets=jnp.asarray(targets.T),
        ket_vectors=jnp.asarray(a),
        ket_inverse=jnp.asarray(a_inverse),
        bra_vectors_dagger=jnp.asarray(b_dagger),
        bra_inverse_dagger=jnp.asarray(b_inverse_dagger),
    )


def split(x: jax.Array) -> jax.Array:
    """Get complex matrices as their real and imaginary parts side by side, (..., n, 2n)."""
    return jnp.concatenate([x.real, x.imag], axis=-1)


def _halves(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    n = x.shape[-1] // 2
    return x[..., :n], x[..., n:]


def join(x: jax.Array) -> jax.Array:
    """Get the complex matrices of their split parts (`split`)."""
    real, imaginary = _halves(x)
    return real + 1j * imaginary


def multiply(c: jax.Array, x: jax.Array) -> jax.Array:
    """Get c x, for x split (`split`) and c complex, elementwise or a scalar."""
    c_real, c_imaginary = jnp.real(c), jnp.imag(c)
    real, imaginary = _halves(x)
    return jnp.concatenate(
        [
            c_real * real - c_imaginary * imaginary,
            c_real * imaginary + c_imaginary * real,
        ],
        axis=-1,
    )


def inner(a: jax.Array, b: jax.Array) -> jax.Array:
    """Get <a, b> = sum(conj(a) * b) of split arrays."""
    (a_real, a_imaginary), (b_real, b_imaginary) = _halves(a), _halves(b)
    return jnp.sum(a * b) + 1j * jnp.sum(a_real * b_imaginary - a_imaginary * b_real)


def magnitudes(x: jax.Array) -> jax.Array:
    """Get |x|^2 of each complex entry of a split array."""
    real, imaginary = _halves(x)
    return real**2 + imaginary**2


def _gauss(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Get the real and imaginary parts of products of complex matrices, from three real products (Gauss).

    a and b are each (real, imaginary, real + imaginary), (3, ...). XLA
    multiplies complex matrices on the CPU at about half the flop rate of
    real ones (OpenBLAS does not): (9, 139, 139) by (9, 139, 139) takes 10.2
    ms complex and 1.45 ms real (OpenBLAS 5.9 and 1.45 ms), so three real
    products take 4.4 ms. The parts are behind a barrier, which keeps XLA
    from fusing them into the products, which would then no longer be calls
    to its matrix product.
    """
    p = jnp.matmul(*jax.lax.optimization_barrier((a, b)))
    return p[0] - p[1], p[2] - p[0] - p[1]


def _parts(real: jax.Array, imaginary: jax.Array) -> jax.Array:
    """Get (real, imaginary, real + imaginary), (3, ...)."""
    return jnp.stack([real, imaginary, real + imaginary])


def _jumps(
    ket: jax.Array, bra: jax.Array, gather: jax.Array, y: jax.Array
) -> jax.Array:
    """Get sum_j P_j[kappa] Y[g_j(kappa)] Q_j[g_j(kappa)] for a split state Y, the jumps of `apply_split` (P = K, Q = B, g = s) or of `apply_adjoint_split`.

    Y[t] Q_j[t] of every j is one product at each t, with Q side by side,
    and the sum over j at each kappa one product of the row of P_j[kappa]
    with the column of those at g_j(kappa). With blocks,
    [Re Y, Im Y] [[Re Q, Im Q], [-Im Q, Re Q]] = [Re YQ, Im YQ], and
    [Re P; Im P] [Re R, Im R] holds the four real products of P R.
    """
    n_kappa, n_ops = gather.shape
    n = y.shape[-2]
    pick = (gather, jnp.arange(n_ops))
    if ket.ndim == y.ndim + 1:
        real, imaginary = _gauss(_parts(*_halves(y)), bra)
        right = jnp.stack([real, imaginary]).reshape(2, n_kappa, n, n_ops, n)
        right = right.swapaxes(2, 3)[:, *pick].reshape(2, n_kappa, n_ops * n, n)
        return jnp.concatenate(_gauss(ket, _parts(*right)), axis=-1)
    right = (y @ bra).reshape(n_kappa, n, 2, n_ops, n)
    right = right.transpose(0, 3, 1, 2, 4)[pick].reshape(n_kappa, n_ops * n, 2 * n)
    out = (ket @ right).reshape(n_kappa, 2, n, 2, n)
    return jnp.concatenate(
        [out[:, 0, :, 0] - out[:, 1, :, 1], out[:, 0, :, 1] + out[:, 1, :, 0]],
        axis=-1,
    )


def apply_split(eigen: EigenLiouvillian, y: jax.Array, shift: jax.Array) -> jax.Array:
    """Apply L - shift to a split state in the eigenvectors (`EigenLiouvillian`): L Y[kappa] = Z Y[kappa] + sum_j K_j[s] Y[s] B_j[s] with s = s_j(kappa)."""
    jumps = _jumps(eigen.forward_ket, eigen.forward_bra, eigen.sources, y)
    return multiply(eigen.rates - shift, y) + jumps


def apply_adjoint_split(
    eigen: EigenLiouvillian, w: jax.Array, shift: jax.Array
) -> jax.Array:
    """Apply (L - shift)^dagger to a split state, the adjoint of `apply_split` in <w, y> = sum(conj(w) * y).

    L^dagger W[s] = conj(Z) W[s] + sum_j K_j[s]^dagger W[t] B_j[s_j(t)]^dagger
    with t = t_j(s) the target of s.
    """
    jumps = _jumps(eigen.adjoint_ket, eigen.adjoint_bra, eigen.targets, w)
    return multiply((eigen.rates - shift).conj(), w) + jumps


def to_eigen(eigen: EigenLiouvillian, x: jax.Array) -> jax.Array:
    """Get Y = A^-1 X B^-dagger."""
    return eigen.ket_inverse @ x @ eigen.bra_inverse_dagger


def from_eigen(eigen: EigenLiouvillian, y: jax.Array) -> jax.Array:
    """Get X = A Y B^dagger."""
    return eigen.ket_vectors @ y @ eigen.bra_vectors_dagger


@jax.jit
def apply_eigen(eigen: EigenLiouvillian, y: jax.Array, shift: jax.Array) -> jax.Array:
    """Apply L - shift to a complex state in the eigenvectors (`apply_split`)."""
    return join(apply_split(eigen, split(y), shift))


_STAGNATION = 100
"""The BiCGSTAB iterations over which a residual that has not halved has reached the floor of round off, which ends the pass (`bicgstab`)."""


def bicgstab(
    operator: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    rtol: jax.Array,
    maxiter: int,
) -> tuple[jax.Array, jax.Array]:
    """Solve A x = rhs from x = 0 by BiCGSTAB, and get the number of iterations, each two applications of A.

    It stops once |rhs - A x| <= rtol |rhs|, after maxiter iterations, once
    the residual has not halved over _STAGNATION iterations, or on a
    breakdown, which the next pass of `refine` recovers from.
    """
    target = rtol * jnp.linalg.norm(rhs)
    zero = jnp.zeros_like(rhs)
    one = jnp.asarray(1.0 + 0j)

    def proceed(carry: tuple) -> jax.Array:
        _x, r, _p, _v, rho, _alpha, omega, k, mark = carry
        norm = jnp.linalg.norm(r)
        stagnant = (k > 0) & (k % _STAGNATION == 0) & (norm > mark / 2)
        return (k < maxiter) & (norm > target) & (rho != 0) & (omega != 0) & ~stagnant

    def iterate(carry: tuple) -> tuple:
        x, r, p, v, rho, alpha, omega, k, mark = carry
        # The residual at the start of each _STAGNATION iterations
        mark = jnp.where(k % _STAGNATION == 0, jnp.linalg.norm(r), mark)
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
            mark,
        )

    x, *_, k, _ = jax.lax.while_loop(
        proceed,
        iterate,
        (zero, rhs, zero, zero, one, one, one, jnp.asarray(0), jnp.linalg.norm(rhs)),
    )
    return x, k


def refine(  # ruff:ignore[too-many-arguments]
    apply: Callable[[jax.Array], jax.Array],
    solve_pass: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    rhs: jax.Array,
    *,
    rtol: jax.Array,
    inner_rtol: float,
    passes: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve A x = rhs by passes of an inner solver, each from the true residual (iterative refinement), and get the applications of A and the residual reached, relative to |rhs|, or nan if it was still falling after the passes.

    solve_pass(residual, rtol) gives an approximate solution and its
    applications of A. Its residual is not monotonic, so each pass only reduces the
    true residual by inner_rtol (or to rtol), and the next starts afresh from
    that residual, which also recovers from a breakdown. The passes stop once
    the residual is below rtol of |rhs|, or once it no longer halves, when
    round off has set its floor. It is all one loop on the device: the
    caller raises on a nan (`check`).
    """
    norm = jnp.maximum(jnp.linalg.norm(rhs), 1e-300)

    def proceed(state: tuple) -> jax.Array:
        *_, done, k = state
        return ~done & (k < passes)

    def iterate(state: tuple) -> tuple:
        solution, residual, best, applications, _, k = state
        step, count = solve_pass(residual, jnp.maximum(inner_rtol, rtol / best))
        update = solution + step
        update_residual = rhs - apply(update)
        relative = jnp.linalg.norm(update_residual) / norm
        worse = relative > best / 2
        return (
            jnp.where(worse, solution, update),
            jnp.where(worse, residual, update_residual),
            jnp.where(worse, best, relative),
            applications + count + 1,
            worse | (relative <= rtol),
            k + 1,
        )

    solution, _, best, applications, done, _ = jax.lax.while_loop(
        proceed,
        iterate,
        (
            jnp.zeros_like(rhs),
            rhs,
            jnp.asarray(1.0),
            jnp.asarray(0),
            jnp.zeros((), bool),
            jnp.asarray(0),
        ),
    )
    return solution, applications, jnp.where(done, best, jnp.nan)


def check(reached: jax.Array, rtol: float) -> None:
    """Raise if a refinement (`refine`) did not settle within its passes.

    Raises
    ------
    RuntimeError
        If the residual was still falling after PASSES passes.
    """
    if np.isnan(float(reached)):
        msg = f"BiCGSTAB did not reach a residual of {rtol:.1e} in {PASSES} passes"
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


@jax.jit
def _solve(
    eigen: EigenLiouvillian, rhs: jax.Array, rtol: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve L Y = rhs in the eigenvectors (`solve`), and get the applications of L and the residual reached (`refine`)."""
    return refine(
        lambda y: apply_eigen(eigen, y, jnp.asarray(0j)),
        lambda residual, rtol: _solve_pass(eigen, residual, rtol, maxiter=MAXITER),
        rhs,
        rtol=jnp.asarray(rtol),
        inner_rtol=INNER_RTOL,
        passes=PASSES,
    )


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
    y, _, reached = _solve(eigen, to_eigen(eigen, rhs), config.linear_rtol)
    check(reached, config.linear_rtol)
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
