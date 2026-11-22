"""The eigenvalues of a nonsymmetric tridiagonal matrix and the weights of its first row and column, in O(m^2), by the Ehrlich-Aberth iteration."""

# cspell: ignore Ehrlich Aberth Bini Gemignani Tisseur csgraph finfo stebz fori Appl unconverged Dhillon rebiorthogonalization oint lstsq jitted LAPACK's argnums pinv
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

_EPS = float(np.finfo(float).eps)
_SWEEPS = 100
"""The most Aberth sweeps at each level of the divide and conquer; 9 to 18 were needed on every tridiagonal tried."""
_NARROWEST = 128
"""The fewest roots of a block that the Aberth sweeps take (`_aberth`)."""
_LEAF = 48
"""The most rows of the blocks at the bottom of the divide and conquer, whose roots start from their diagonals."""
_CLUSTER = 1e-3
"""The distance between roots, times the time t their poles last (`_merge`), below which they are a cluster whose weights only their moments define."""
_LASTING = 20.0
"""The decay, exp(-_LASTING), after which a pole no longer counts (`_merge`)."""
_MOMENTS = 4
"""The moments of a cluster of roots that its weights match (`_merge`)."""
_CONTOUR = 16
"""The points of the contour integral for the moments of a cluster of roots (`_merge`)."""
_STALLED = 1e-9
"""The correction of a root, relative to the scale of T, below which a correction that has not halved is round off, and the root has converged."""
_CHUNK = 128
"""The roots taken together in the steps whose memory is m times their number."""


def get_size(m: int) -> int:
    """Get the padded size, 2^k or 3 2^(k-1), so that compiled shapes are few and halve evenly."""
    size = 2
    while True:
        for candidate in (size, 3 * size // 2):
            if candidate >= m:
                return candidate
        size *= 2


def _reciprocal(z: jax.Array) -> jax.Array:
    """Get 1 / z without the scaling of complex division: |z| stays within 1e-20 and 1e20 here."""
    return jnp.conj(z) / (z.real**2 + z.imag**2)


def _newton(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    roots: jax.Array,
    first: jax.Array,
    alpha: jax.Array,
    above: jax.Array,
    block: jax.Array,
    pivot: jax.Array,
) -> jax.Array:
    """Get p(lambda) / p'(lambda) for each root, p the characteristic polynomial of its block of rows from first.

    The pivots g_k = lambda - alpha_k - c_k / g_{k-1} of the LU factors of
    lambda - T are ratios p_k / p_{k-1} of the leading minors, so p' / p is
    the sum of g_k' / g_k, with g_k' = 1 + c_k g_{k-1}' / g_{k-1}^2: O(m) for
    each root. A pivot below eps times the scale of T is set to it (as the
    bisection of LAPACK's stebz does), a backward error of round off.
    above[k] is beta delta between rows k - 1 and k.
    """

    def step(
        k: jax.Array, carry: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        inverse, derivative, total = carry
        ratio = jnp.where(k > 0, above[first + k], 0) * inverse
        derivative = 1 + ratio * derivative * inverse
        pivot_k = roots - alpha[first + k] - ratio
        inverse = _reciprocal(jnp.where(jnp.abs(pivot_k) < pivot, pivot, pivot_k))
        return inverse, derivative, total + derivative * inverse

    zero = jnp.zeros_like(roots)
    return _reciprocal(jax.lax.fori_loop(0, block, step, (zero, zero, zero))[2])


def _repulsion(
    roots: jax.Array,
    index: jax.Array,
    first: jax.Array,
    every: jax.Array,
    block: jax.Array,
) -> jax.Array:
    """Get sum over the other roots i of the block of 1 / (lambda_j - lambda_i), _CHUNK at a time; index is that of each root in every."""
    chunk = min(_CHUNK, every.shape[0])

    def add(i: jax.Array, total: jax.Array) -> jax.Array:
        offset = i * chunk + jnp.arange(chunk)
        others = first[:, None] + offset[None, :]
        skip = (offset[None, :] >= block) | (others == index[:, None])
        difference = roots[:, None] - every[jnp.minimum(others, every.shape[0] - 1)]
        return total + jnp.where(
            skip, 0, _reciprocal(jnp.where(skip, 1, difference))
        ).sum(axis=1)

    return jax.lax.fori_loop(
        0, (block + chunk - 1) // chunk, add, jnp.zeros_like(roots)
    )


@partial(jax.jit, donate_argnums=0)
def _aberth(
    roots: jax.Array, alpha: jax.Array, above: jax.Array, block: jax.Array
) -> jax.Array:
    """Refine the roots of each block of rows by Aberth sweeps until each has converged, or _SWEEPS.

    Each sweep is the Newton correction N of each root, deflated by the
    others: lambda_j -= N_j / (1 - N_j sum_i 1 / (lambda_j - lambda_i)),
    which converges cubically to simple roots from any distinct guesses
    (Bini, Gemignani and Tisseur, SIAM J. Matrix Anal. Appl. 27, 153 (2005)).
    A root has converged once its correction is below eps of its size, or
    once it has stopped halving below _STALLED of the scale of T (round off).
    The roots converge at different rates (sodium on Cu(111) in 1D at
    m = 575: 551, 458, 366, 325, 272, 190, 117, 69, 32, 16, 5 and 0 left
    after each sweep), so a sweep takes only the `width` unconverged roots
    first, the width a quarter at a time down to _NARROWEST. The block size
    is not static, so that every level of `_roots` is one compilation.
    """
    size = roots.shape[0]
    scale = jnp.abs(alpha).max() + jnp.sqrt(jnp.abs(above).max())
    blocks = jnp.arange(size) // block * block

    def sweep(state: tuple[jax.Array, ...], width: int) -> tuple[jax.Array, ...]:
        roots, active, last, sweeps = state
        index = jnp.argsort(~active, stable=True)[:width]
        part, moving, first = roots[index], active[index], blocks[index]
        newton = _newton(part, first, alpha, above, block, _EPS * scale)
        correction = newton * _reciprocal(
            1 - newton * _repulsion(part, index, first, roots, block)
        )
        finite = jnp.isfinite(correction)
        correction = jnp.where(finite & moving, correction, 0)
        size = jnp.abs(correction)
        converged = finite & (
            (size <= 2 * _EPS * jnp.abs(part - correction))
            | ((size >= 0.5 * last[index]) & (size < _STALLED * scale))
        )
        return (
            roots.at[index].add(-correction),
            active.at[index].set(moving & ~converged),
            last.at[index].set(jnp.where(moving, size, last[index])),
            sweeps + 1,
        )

    state = (roots, jnp.ones(size, bool), jnp.full(size, jnp.inf), jnp.asarray(0))
    width = size
    while True:
        limit = width // 4 if width > _NARROWEST and width % 4 == 0 else 0
        state = jax.lax.while_loop(
            lambda state, limit=limit: (state[1].sum() > limit) & (state[3] < _SWEEPS),
            partial(sweep, width=width),
            state,
        )
        if limit == 0:
            return state[0]
        width //= 4


def _roots(alpha: jax.Array, coupling: jax.Array) -> jax.Array:
    """Get the eigenvalues of the padded tridiagonal by divide and conquer.

    The blocks of at most _LEAF rows start from their diagonals; then each
    level merges pairs of blocks, from the roots of both halves (the
    tridiagonal is the two blocks plus a rank two coupling), up to the
    whole. Starting the last level from the roots of the last evaluation
    instead (and the diagonal of the new rows) was 3 times faster, but on
    underdamped hydrogen (gamma 0.01 / ps) at m = 1,600-2,400, where copies
    of poles sit 1e-13 apart, it left the weights 1e-6 to 1e-5 wrong, and
    the evolution then needed 2.6 times the steps or did not converge.
    """
    size = alpha.shape[0]
    scale = jnp.abs(alpha).max() + jnp.sqrt(jnp.abs(coupling).max())
    above = jnp.concatenate([jnp.zeros(1, coupling.dtype), coupling[:-1]])
    # Distinct guesses, as Aberth needs
    jitter = 1e-8 * scale * jnp.exp(2.4j * jnp.arange(size))
    block = size
    while block > _LEAF:
        block //= 2
    roots = alpha + jitter
    while True:
        roots = _aberth(roots, alpha, above, jnp.asarray(block))
        if block == size:
            return roots
        block *= 2


def _chunk_weights(
    roots: jax.Array, alpha: jax.Array, coupling: jax.Array, pivot: jax.Array
) -> jax.Array:
    """Get x_1 y_1 / y^T x for the right and left eigenvectors x, y of each root, from its twisted factorization.

    With the pivots g_k from the top and h_k from the bottom, the twist t
    is where |gamma_k| = |g_k + h_k - (lambda - alpha_k)| is least, and
    p_k = x_k y_k is c_k / g_k^2 p_{k+1} above it and c_{k-1} / h_k^2
    p_{k-1} below, from p_t = 1: the stable way to the eigenvectors of a
    tridiagonal (Parlett and Dhillon, Linear Algebra Appl. 309, 121 (2000)),
    where the resolvent itself loses all accuracy at the root. The sums of
    p are rescaled as they go, and kept with the logarithm of their scale.
    """
    above = jnp.concatenate([jnp.zeros(1, coupling.dtype), coupling[:-1]])

    def pivots(
        last: jax.Array, row: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        diagonal, product = row
        pivot_k = roots - diagonal - product * last
        inverse = _reciprocal(jnp.where(jnp.abs(pivot_k) < pivot, pivot, pivot_k))
        return inverse, inverse

    zero = jnp.zeros_like(roots)
    _, top = jax.lax.scan(pivots, zero, (alpha, above))
    _, bottom = jax.lax.scan(pivots, zero, (alpha, coupling), reverse=True)
    gamma = _reciprocal(top) + _reciprocal(bottom) - (roots[None, :] - alpha[:, None])
    twist = jnp.argmin(jnp.abs(gamma), axis=0)

    def accumulate(
        state: tuple[jax.Array, jax.Array, jax.Array],
        row: tuple[jax.Array, jax.Array, jax.Array],
        *,
        side: int,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        p, total, scale = state
        k, product, inverse = row
        p = jnp.where(
            k == twist,
            1,
            jnp.where(side * (k - twist) > 0, p * product * inverse**2, 0),
        )
        total += jnp.where(side * (k - twist) > 0, p, 0)
        size = jnp.maximum(jnp.abs(p), 1)
        return (p / size, total / size, scale + jnp.log(size)), None

    rows = jnp.arange(alpha.shape[0])
    # p_t = 1 counts in the sum above the twist
    (first, upper, upper_scale), _ = jax.lax.scan(
        partial(accumulate, side=-1),
        (zero, jnp.ones_like(roots), jnp.zeros(roots.shape)),
        (rows, coupling, top),
        reverse=True,
    )
    (_, lower, lower_scale), _ = jax.lax.scan(
        partial(accumulate, side=1),
        (zero, zero, jnp.zeros(roots.shape)),
        (rows, above, bottom),
    )
    return first / (upper + lower * jnp.exp(lower_scale - upper_scale))


@jax.jit
def _weights(roots: jax.Array, alpha: jax.Array, coupling: jax.Array) -> jax.Array:
    """Get the weights of the roots (`_chunk_weights`), _CHUNK roots at a time."""
    size = roots.shape[0]
    pivot = _EPS * (jnp.abs(alpha).max() + jnp.sqrt(jnp.abs(coupling).max()))
    chunks = size // _CHUNK if size % _CHUNK == 0 else 1
    return jax.lax.map(
        lambda part: _chunk_weights(part, alpha, coupling, pivot),
        roots.reshape(chunks, -1),
    ).ravel()


def continued_fraction(
    alpha: jax.Array, coupling: jax.Array, steps: jax.Array, points: jax.Array
) -> jax.Array:
    """Get e_1^T (s - T_m)^-1 e_1 at each point s for the first `steps` rows of alpha and coupling, as a continued fraction from the last row back, O(m) for each s.

    The coupling of the last row (to the next, not yet in T_m) is not used.
    """

    def row(i: jax.Array, value: jax.Array) -> jax.Array:
        k = steps - 1 - i
        return 1 / (points - alpha[k] - coupling[k] * value)

    return jax.lax.fori_loop(0, steps, row, jnp.zeros_like(points))


@partial(jax.jit, static_argnames="size")
def _padded(
    alpha: jax.Array, coupling: jax.Array, steps: jax.Array, *, size: int
) -> tuple[jax.Array, jax.Array]:
    """Get the first `steps` rows of alpha and coupling padded to size (`get_size`) by uncoupled diagonal entries left of the spectrum (|lambda| <= max |alpha| + 2 max |c|^1/2, the bound of the complex symmetric form), coupling with a zero last."""
    index = jnp.arange(size)
    alpha, coupling = alpha[:size], coupling[:size]
    diagonal, coupled = index < steps, index < steps - 1
    radius = jnp.where(diagonal, jnp.abs(alpha), 0).max() + 2 * jnp.sqrt(
        jnp.where(coupled, jnp.abs(coupling), 0).max()
    )
    padding = -2 * (radius + 1) * (1 + (index - steps) / size)
    return jnp.where(diagonal, alpha, padding), jnp.where(coupled, coupling, 0)


def _pairwise(
    roots: jax.Array,
    others: jax.Array,
    reduce: Callable[[jax.Array, jax.Array, int], jax.Array],
    initial: jax.Array,
) -> jax.Array:
    """Get reduce over the others, _CHUNK at a time, of the distances |roots_j - others_i| (j along the rows)."""
    size = others.shape[0]
    chunk = _CHUNK if size % _CHUNK == 0 else size

    def add(i: int, total: jax.Array) -> jax.Array:
        part = jax.lax.dynamic_slice_in_dim(others, i * chunk, chunk)
        return reduce(total, jnp.abs(roots[:, None] - part[None, :]), i * chunk)

    return jax.lax.fori_loop(0, size // chunk, add, initial)


def _clusters(roots: jax.Array, width: jax.Array) -> jax.Array:
    """Get the smallest index in the cluster of each root: the roots linked by distances below the smaller of their widths."""
    index = jnp.arange(roots.shape[0])

    def spread(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        labels, _ = state

        def least(total: jax.Array, distance: jax.Array, offset: int) -> jax.Array:
            part = jax.lax.dynamic_slice_in_dim(labels, offset, distance.shape[1])
            near = jax.lax.dynamic_slice_in_dim(width, offset, distance.shape[1])
            linked = jnp.where(
                distance < jnp.minimum(width[:, None], near[None, :]),
                part[None, :],
                index.shape[0],
            )
            return jnp.minimum(total, linked.min(axis=1))

        new = _pairwise(roots, roots, least, labels)
        return new, jnp.any(new != labels)

    def changed(state: tuple[jax.Array, jax.Array]) -> jax.Array:
        return jnp.any(state[1])

    return jax.lax.while_loop(changed, spread, (index, jnp.ones((), bool)))[0]


@jax.jit
def _merge(  # ruff:ignore[too-many-locals]
    roots: jax.Array,
    weights: jax.Array,
    alpha: jax.Array,
    coupling: jax.Array,
    span: jax.Array,
) -> jax.Array:
    """Get the weights with those of each cluster of close roots replaced by the least-norm ones with its first _MOMENTS moments, from a contour integral.

    Two-sided Lanczos without rebiorthogonalization makes copies of its
    converged poles, a few within 1e-13 to 1e-7 of each other (hydrogen on
    Cu(111) in 1D at 0.3 |G| and 1500 steps, ||T|| 2e3). There the weight
    of each copy is ill conditioned, the twisted factorizations of the
    copies do not split it consistently, and their sum was 1e-4 wrong
    (eig, backward stable, splits it consistently). Only the moments
    M_n = sum w_i (lambda_i - c)^n about the center c matter, while
    |lambda_i - c| t is small over the times t that exp(lambda t) lasts,
    t_j = min(span, _LASTING / |Re lambda_j|): two roots are linked if
    closer than _CLUSTER / t_j for both. Then
    M_n = (2 pi i)^-1 oint (z - c)^n f(z) dz on a circle about c, of radius
    the geometric mean of the size of the cluster and its distance to the
    nearest other root, by the trapezoid rule on _CONTOUR points, with f
    evaluated as a continued fraction, accurate away from the roots (to
    eps ||T|| / radius relative: a radius set by the time t_j rather than
    the span keeps it large enough; with the span, underdamped hydrogen at
    |G| over 2000 ps linked copies 5e-11 apart but not their twins 4e-7
    away, and the radius 5e-9 left the weights 5e-5 wrong). A cluster has
    at least two roots, so at most half the roots lead one.
    """
    size = roots.shape[0]
    lasting = jnp.minimum(span, _LASTING / jnp.maximum(-roots.real, _LASTING / span))
    labels = _clusters(roots, _CLUSTER / lasting)
    members = jax.ops.segment_sum(jnp.ones(size), labels, size)
    centers = jax.ops.segment_sum(roots, labels, size) / jnp.maximum(members, 1)
    offsets = roots - centers[labels]
    extent = jax.ops.segment_max(jnp.abs(offsets), labels, size)
    # The clusters of two or more roots first, at most size / 2 of them
    leaders = jnp.argsort(members <= 1, stable=True)[: size // 2]
    center = centers[leaders]

    def nearest(total: jax.Array, distance: jax.Array, offset: int) -> jax.Array:
        part = jax.lax.dynamic_slice_in_dim(labels, offset, distance.shape[1])
        outside = jnp.where(part[None, :] == leaders[:, None], jnp.inf, distance)
        return jnp.minimum(total, outside.min(axis=1))

    distance = _pairwise(center, roots, nearest, jnp.full(center.shape, jnp.inf))
    radius = jnp.sqrt(jnp.maximum(extent[leaders], 1e-9 * distance) * distance)
    unit = jnp.exp(2j * jnp.pi * jnp.arange(_CONTOUR) / _CONTOUR)
    points = (center[:, None] + radius[:, None] * unit).ravel()

    def fraction(
        value: jax.Array, row: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, None]:
        diagonal, product = row
        return _reciprocal(points - diagonal - product * value), None

    values, _ = jax.lax.scan(
        fraction, jnp.zeros_like(points), (alpha, coupling), reverse=True
    )
    values = radius[:, None] * values.reshape(-1, _CONTOUR)
    powers = jnp.arange(_MOMENTS)
    # The moments in units of the radius, M_n / radius^n
    moments = (
        unit[None, None, :] ** (powers[None, :, None] + 1) * values[:, None, :]
    ).mean(axis=2)
    scaled = jnp.zeros(size, roots.real.dtype).at[leaders].set(radius)[labels]
    vandermonde = (offsets / jnp.where(scaled > 0, scaled, 1))[:, None] ** powers[
        None, :
    ]
    gram = jax.ops.segment_sum(
        vandermonde[:, :, None] * vandermonde.conj()[:, None, :], labels, size
    )[leaders]
    coefficients = jnp.einsum("ckn,cn->ck", jnp.linalg.pinv(gram, rtol=1e-12), moments)
    full = (
        jnp.zeros((size, _MOMENTS), roots.dtype).at[leaders].set(coefficients)[labels]
    )
    merged = jnp.sum(vandermonde.conj() * full, axis=1)
    return jnp.where(members[labels] > 1, merged, weights)


def _poles(
    alpha: jax.Array, coupling: jax.Array, span: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Get the roots and weights by jitted parts, each level of the divide and conquer compiled once for its shape (with the block size traced), with no transfer to the host between them."""
    roots = _roots(alpha, coupling)
    weights = _weights(roots, alpha, coupling)
    return roots, _merge(roots, weights, alpha, coupling, span)


def get_poles(
    alpha: jax.Array, coupling: jax.Array, steps: int, *, span: float
) -> tuple[jax.Array, jax.Array]:
    """Get the eigenvalues lambda_j of T and their weights w_j, e_1^T (s - T)^-1 e_1 = sum_j w_j / (s - lambda_j), in O(m^2), on the device.

    T is tridiagonal with the first `steps` of alpha on its diagonal and the
    products of its off-diagonals beta_k delta_k = coupling_k, nonzero
    (unreduced), each at least get_size(steps) long. Its
    eigenvalues are the roots of its characteristic polynomial, found by
    the Ehrlich-Aberth iteration (`_aberth`) on the recurrence of its
    leading minors, from those of its two halves, recursively (`_roots`);
    the weights from the twisted factorization at each root
    (`_chunk_weights`), except for clusters of close roots, whose weights
    come from their moments over the span of times (`_merge`). All of it is jitted
    and stays on the device (`_poles`), padded (`_padded`): the weights of
    the padding are zero.
    """
    alpha, coupling = _padded(alpha, coupling, jnp.asarray(steps), size=get_size(steps))
    return _poles(alpha, coupling, jnp.asarray(span))
