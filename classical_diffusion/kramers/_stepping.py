# cspell: ignore argnums orthogonalize REORTHOGONALIZE Gragg ascontiguousarray indptr matvec tocsr unconverged
"""The Kramers operator applied without being stored, and exponential time steps in its Krylov spaces, for systems too large to factorise."""

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from classical_diffusion.kramers._real import (
    _get_ladder_matrices,
    get_force_shifts,
    get_hermite_indices,
    get_momentum_reach,
    get_width,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from classical_diffusion.kramers._model import (
        FloatArray,
        IntArray,
        Model,
    )


def _x64[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Run a function with 64 bit floats in jax, which the public API does not ask the caller to switch on."""

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with jax.enable_x64(new_val=True):
            return function(*args, **kwargs)

    return wrapped


_PAD = 2
"""The largest Hermite ladder offset: the friction in narrowed Hermite functions couples n to n +- 2."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class _Kernel:
    """The real form of the Kramers operator as a stencil on the (Hermite box) x (plane wave box) array.

    L X = sum_j [k_j (P_j X) - D_j (F_j X)] - gamma sum_j R_j X, where P_j,
    D_j and R_j shift the order n_j by a few steps with a coefficient of the
    target order (`bands`), and F_j shifts the plane waves by the reciprocal
    vectors of the Fourier terms of the potential. In the box padded once,
    each term is a static slice times a coefficient, so the whole operator
    is one streaming loop, with no index arrays, on any device. The compact
    state of the basis is expanded into the box by one gather and read back
    at its positions by another.
    """

    momentum: tuple[jax.Array, ...]
    derivative: tuple[jax.Array, ...]
    friction: tuple[jax.Array, ...]
    wavevector: jax.Array
    """(G - K)_j on the plane wave box, (n_dim, *box)."""
    expand: jax.Array
    """The compact index of each point of the box, or the number of unknowns (a zero) outside the basis."""
    positions: jax.Array
    """The point of the box of each unknown."""
    offsets: tuple[tuple[int, ...], ...] = field(metadata={"static": True})
    """The offsets of the bands of P, D and R."""
    forces: tuple[tuple[tuple[int, ...], tuple[float, ...]], ...] = field(
        metadata={"static": True}
    )
    """Each shift c of F and its coefficient for each j."""
    box: tuple[int, ...] = field(metadata={"static": True})
    margins: tuple[int, ...] = field(metadata={"static": True})


@jax.jit
def _apply(kernel: _Kernel, x: jax.Array) -> jax.Array:
    n_dim = len(kernel.box) // 2
    padded = jnp.take(jnp.append(x, 0.0), kernel.expand).reshape(kernel.box)
    padded = jnp.pad(
        jax.lax.optimization_barrier(padded), [(m, m) for m in kernel.margins]
    )

    def window(j: int, o: int, shift: tuple[int, ...] = (0,) * 3) -> jax.Array:
        start = [m + (o if a == j else 0) for a, m in enumerate(kernel.margins)]
        for a in range(n_dim):
            start[n_dim + a] -= shift[a]
        return jax.lax.slice(
            padded, start, [s + b for s, b in zip(start, kernel.box, strict=False)]
        )

    def along(c: jax.Array, j: int) -> jax.Array:
        return c.reshape([-1 if a == j else 1 for a in range(len(kernel.box))])

    p_offsets, d_offsets, r_offsets = kernel.offsets
    out = jnp.zeros(kernel.box)
    for j in range(n_dim):
        for o, c in zip(r_offsets, kernel.friction, strict=True):
            out -= along(c, j) * window(j, o)
        for o, c in zip(p_offsets, kernel.momentum, strict=True):
            out += along(c, j) * kernel.wavevector[j] * window(j, o)
        for o, c in zip(d_offsets, kernel.derivative, strict=True):
            for shift, coefficient in kernel.forces:
                if coefficient[j]:
                    out -= coefficient[j] * along(c, j) * window(j, o, shift)
    return jnp.take(out.reshape(-1), kernel.positions)


def _bands(matrix: FloatArray) -> tuple[tuple[int, ...], tuple[FloatArray, ...]]:
    """Get the offsets o and coefficients c_o[n] = M[n, n + o] of the nonzero bands of a square matrix."""
    size = matrix.shape[0]
    offsets, coefficients = [], []
    for o in range(-size + 1, size):
        c = np.zeros(size)
        rows = np.arange(max(0, -o), min(size, size - o))
        c[rows] = matrix[rows, rows + o]
        if np.any(c):
            offsets.append(o)
            coefficients.append(c)
    return tuple(offsets), tuple(coefficients)


@dataclass(frozen=True, kw_only=True, eq=False)
class MatrixFreeOperator:
    """The real form of the Kramers operator (`build_real_operator`) on states of shape (n_hermite, n_plane_waves), applied as a stencil (`_Kernel`)."""

    kernel: _Kernel
    shape: tuple[int, int]
    parity: FloatArray
    """(-1)^{|n|} of each Hermite multi-index."""
    indices: IntArray
    width: float

    def apply(self, x: jax.Array) -> jax.Array:
        """Apply L to a flat state."""
        with jax.enable_x64(new_val=True):
            return _apply(self.kernel, jnp.asarray(x))


def build_matrix_free_operator(  # ruff:ignore[too-many-locals]
    model: Model, vectors: IntArray, max_order: int, delta_k: FloatArray
) -> MatrixFreeOperator:
    """Build the matrix-free real form of the Kramers operator at K."""
    n_dim = model.n_dim
    width = get_width(max_order, get_momentum_reach(model))
    indices = get_hermite_indices(n_dim, max_order)
    p, dp, friction = _get_ladder_matrices(max_order, width)
    # p and d/dp are odd, so in the real form (amplitudes times i^n) they
    # are i times a real matrix; the friction is even and real
    phase = np.diag(1j ** np.arange(max_order + 1))
    real_form = phase.conj() @ np.array([p, dp, friction]) @ phase
    bands = [
        _bands(real_form[0].imag),
        _bands(real_form[1].imag),
        _bands(model.friction * real_form[2].real),
    ]
    shifts = [
        (tuple(int(i) for i in c), tuple(float(f) for f in np.imag(coefficient)))
        for c, coefficient in get_force_shifts(model)
    ]
    low = vectors.min(axis=0)
    hermite_box = (max_order + 1,) * n_dim
    plane_wave_box = tuple(int(b) for b in vectors.max(axis=0) - low + 1)
    box = hermite_box + plane_wave_box
    positions = (
        np.ravel_multi_index(tuple(indices.T), hermite_box)[:, None]
        * int(np.prod(plane_wave_box))
        + np.ravel_multi_index(tuple((vectors - low).T), plane_wave_box)[None, :]
    )
    expand = np.full(int(np.prod(box)), positions.size, np.int32)
    expand[positions.ravel()] = np.arange(positions.size)
    grid = np.indices(plane_wave_box).reshape(n_dim, -1).T + low
    wavevector = (grid @ model.reciprocal - delta_k).T.reshape((n_dim, *plane_wave_box))
    reach = np.max(np.abs([c for c, _ in shifts] or [[0] * n_dim]), axis=0)
    with jax.enable_x64(new_val=True):
        kernel = _Kernel(
            momentum=tuple(jnp.asarray(c) for c in bands[0][1]),
            derivative=tuple(jnp.asarray(c) for c in bands[1][1]),
            friction=tuple(jnp.asarray(c) for c in bands[2][1]),
            wavevector=jnp.asarray(wavevector),
            expand=jnp.asarray(expand),
            positions=jnp.asarray(positions.ravel().astype(np.int32)),
            offsets=tuple(b[0] for b in bands),
            forces=tuple(shifts),
            box=box,
            margins=(_PAD,) * n_dim + tuple(int(r) for r in reach),
        )
    return MatrixFreeOperator(
        kernel=kernel,
        shape=(len(indices), len(vectors)),
        parity=np.asarray(1.0 - 2.0 * (indices.sum(axis=1) % 2), dtype=np.float64),
        indices=indices,
        width=width,
    )


_CHUNK = 8
"""The rows of each block of the Krylov basis: Gram-Schmidt works on the blocks a step has filled, each a separate array, so no slice of the basis is copied."""


@dataclass(frozen=True, kw_only=True)
class _Step:
    """The Krylov space of one step: an orthonormal basis V, as blocks of rows on the device, and L in it, H, with one more row (`_arnoldi`)."""

    blocks: tuple[jax.Array, ...]
    hessenberg: FloatArray
    norm: float

    def combine(self, coefficients: FloatArray) -> jax.Array:
        """Get the states c V for coefficients c, (k, rows)."""
        return _combine(self.blocks, _padded(coefficients, self.blocks))

    def forms(self, coefficients: FloatArray, parity: jax.Array) -> FloatArray:
        """Get s^T J s for each state s = c V, by whichever is cheaper: the states, O(k m n) for k of them, or the J-Gram matrix of the basis, O(m^2 n)."""
        k = len(coefficients)
        padded = _padded(coefficients, self.blocks)
        if k < padded.shape[1]:
            return np.asarray(_forms(self.blocks, padded, parity))[:k]
        gram = np.asarray(_gram(self.blocks, parity))
        return np.einsum("ki,ij,kj->k", padded, gram, padded)[:k]


def _padded(coefficients: FloatArray, blocks: tuple[jax.Array, ...]) -> FloatArray:
    """Pad coefficients (k, rows) to whole blocks of rows and to a multiple of _CHUNK states, so that few shapes are compiled."""
    k, rows = coefficients.shape
    out = np.zeros((-(-k // _CHUNK) * _CHUNK, len(blocks) * _CHUNK))
    out[:k, :rows] = coefficients
    return out


@jax.jit
def _combine(blocks: tuple[jax.Array, ...], c: jax.Array) -> jax.Array:
    return functools.reduce(
        jnp.add,
        (c[:, i * _CHUNK : (i + 1) * _CHUNK] @ b for i, b in enumerate(blocks)),
    )


@jax.jit
def _forms(blocks: tuple[jax.Array, ...], c: jax.Array, parity: jax.Array) -> jax.Array:
    states = _combine(blocks, c)
    return jnp.sum(states * parity * states, axis=1)


@jax.jit
def _gram(blocks: tuple[jax.Array, ...], parity: jax.Array) -> jax.Array:
    return jnp.block([[(a * parity) @ b.T for b in blocks] for a in blocks])


@functools.partial(jax.jit, static_argnames=("count", "target"), donate_argnums=0)
def _orthogonalize(
    blocks: tuple[jax.Array, ...], w: jax.Array, row: int, *, count: int, target: int
) -> tuple[tuple[jax.Array, ...], jax.Array, jax.Array, jax.Array]:
    """Remove from w its components on the first count blocks, twice where the first pass cancelled most of it, and store w / |w| as the row of block target (and return it, the next vector to apply L to)."""
    before = jnp.linalg.norm(w)

    def project(w: jax.Array) -> tuple[jax.Array, jax.Array]:
        overlaps = [blocks[i] @ w for i in range(count)]
        w -= sum(h @ blocks[i] for i, h in enumerate(overlaps))
        return w, jnp.concatenate(overlaps)

    def again(args: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        w, overlap = args
        w, more = project(w)
        return w, overlap + more

    w, overlap = project(w)
    w, overlap = jax.lax.cond(
        jnp.linalg.norm(w) < _REORTHOGONALIZE * before,
        again,
        lambda args: args,
        (w, overlap),
    )
    norm = jnp.linalg.norm(w)
    following = jnp.where(norm > 0, w / norm, 0.0)
    out = list(blocks)
    if target < len(blocks):
        out[target] = jax.lax.dynamic_update_slice_in_dim(
            blocks[target], following[None], row, 0
        )
    return tuple(out), overlap, norm, following


def _arnoldi(apply: Callable[[jax.Array], jax.Array], u: jax.Array, m: int) -> _Step:
    """Get an orthonormal basis V of the Krylov space of m vectors of u, L in it, the Hessenberg H with one more row, and |u|, with u = |u| V e_0.

    The steps are queued on the device without waiting for each other; the
    space ends early, with a square H, where it is invariant.
    """
    norm = float(jnp.linalg.norm(u))
    blocks = [jnp.zeros((_CHUNK, u.size)) for _ in range(-(-(m + 1) // _CHUNK))]
    following = u / norm
    blocks[0] = blocks[0].at[0].set(following)
    blocks_tuple = tuple(blocks)
    columns = []
    for k in range(m):
        blocks_tuple, overlap, beta, following = _orthogonalize(
            blocks_tuple,
            apply(following),
            (k + 1) % _CHUNK,
            count=k // _CHUNK + 1,
            target=(k + 1) // _CHUNK,
        )
        columns.append((overlap, beta))
    hessenberg = np.zeros((m + 1, m))
    for k, (overlap, beta) in enumerate(columns):
        hessenberg[: k + 1, k] = np.asarray(overlap)[: k + 1]
        hessenberg[k + 1, k] = float(beta)
        if hessenberg[k + 1, k] <= _INVARIANT * max(
            float(np.abs(hessenberg[: k + 1, k]).max()), 1e-300
        ):
            return _Step(
                blocks=blocks_tuple,
                hessenberg=hessenberg[: k + 1, : k + 1],
                norm=norm,
            )
    return _Step(blocks=blocks_tuple, hessenberg=hessenberg, norm=norm)


_REORTHOGONALIZE = 0.7
"""The fraction of |w| left after Gram-Schmidt below which it is done a second time (Daniel, Gragg, Kaufman and Stewart 1976)."""
_INVARIANT = 1e-13
"""The norm of the next Krylov vector, relative to the projected operator, below which the Krylov space is invariant."""


def _augmented(step: _Step, extra: int) -> FloatArray:
    """[[H, e_0 0...], [0, shift]]: its exponential holds exp(h H) e_0 and its integrals phi_1, phi_2 (Saad 1992, Sidje 1998)."""
    rows, cols = step.hessenberg.shape
    out = np.zeros((rows + extra, rows + extra))
    out[:rows, :cols] = step.hessenberg
    if extra:
        out[0, rows] = 1.0
        for i in range(extra - 1):
            out[rows + i, rows + i + 1] = 1.0
    return out


def _steps(  # ruff:ignore[too-many-arguments]
    apply: Callable[[jax.Array], jax.Array],
    start: FloatArray,
    t_end: float,
    *,
    tol: float,
    m: int,
    extra: int,
) -> Iterator[tuple[float, float, _Step, FloatArray]]:
    """Step u(t) = e^{L t} u to t_end, yielding (t, h, the Krylov space of u(t), the augmented H) of each step.

    Each step builds the Krylov space of m vectors of the state, in which
    L is the Hessenberg matrix H, and e^{h L} u = |u| V exp(h H) e_0. The
    error of a step is the weight it puts on the next vector (Sidje 1998,
    Expokit), held below tol |u(0)| h / t_end, so the error at the end is
    below tol |u(0)|. The same space gives the state at any time within the
    step, and the exponential of H augmented by `extra` rows its integrals.

    Raises
    ------
    RuntimeError
        If a step cannot meet its error below 1e-12 of t_end, rather than
        taking it unconverged.
    """
    scale = float(np.linalg.norm(start))
    u = jnp.asarray(start)
    t, h = 0.0, t_end / 20
    while t < t_end * (1 - 1e-14):
        step = _arnoldi(apply, u, m)
        rows, cols = step.hessenberg.shape
        augmented = _augmented(step, extra)
        while True:
            length = min(h, t_end - t)
            exponential = scipy.linalg.expm(length * augmented)
            # A square H is an invariant space: the step is exact
            error = 0.0 if rows == cols else abs(step.norm * exponential[cols, 0])
            allowed = tol * scale * length / t_end
            if error <= allowed:
                break
            if length < 1e-12 * t_end:
                msg = (
                    f"The time steps stalled at t = {t:.3e} of {t_end:.3e}: the error"
                    f" estimate of a step, {error:.1e}, does not fall to {allowed:.1e}"
                )
                raise RuntimeError(msg)
            h = 0.9 * length * (allowed / error) ** (1 / cols)
        yield t, length, step, augmented
        u = step.combine(step.norm * exponential[None, :rows, 0])[0]
        t += length
        h = min(0.9 * length * (allowed / max(error, 1e-300)) ** (1 / cols), 5 * length)


@dataclass(frozen=True, kw_only=True)
class Propagation:
    """Measurements at each time, with the number of applications of L they took."""

    values: FloatArray
    n_applications: int


@_x64
def propagate_form(  # ruff:ignore[too-many-arguments]
    apply: Callable[[jax.Array], jax.Array],
    start: FloatArray,
    parity: FloatArray,
    times: FloatArray,
    *,
    tol: float,
    m: int = 30,
    monitor: Callable[[FloatArray], None] | None = None,
) -> Propagation:
    """Get u^T J e^{L t} u at the times, stepping only to t / 2.

    `monitor` sees the state at the start of every step.

    L is J-symmetric, L^T = J L J, so e^{L t} = e^{L t / 2} J e^{L^T t / 2} J
    and u^T J e^{L t} u = u(t / 2)^T J u(t / 2), with u(s) = e^{L s} u: the
    state is stepped for half of the time.
    """
    half = np.asarray(times) / 2
    out = np.zeros(len(times))
    device_parity = jnp.asarray(parity)
    count, i = 0, 0
    while i < len(half) and half[i] == 0:
        out[i] = start @ (parity * start)
        i += 1
    if i == len(half):
        return Propagation(values=out, n_applications=0)
    for t, length, step, augmented in _steps(
        apply, start, float(half[-1]), tol=tol / 2, m=m, extra=0
    ):
        count += step.hessenberg.shape[1]
        rows = step.hessenberg.shape[0]
        if monitor is not None:
            monitor(step.norm * np.asarray(step.blocks[0][0]))
        end = int(np.searchsorted(half, t + length * (1 + 1e-12), side="right"))
        coefficients = np.array(
            [
                step.norm * scipy.linalg.expm((h - t) * augmented)[:rows, 0]
                for h in half[i:end]
            ]
        ).reshape(-1, rows)
        if len(coefficients):
            out[i:end] = step.forms(coefficients, device_parity)
        i = end
    return Propagation(values=out, n_applications=count)


@_x64
def propagate_state(  # ruff:ignore[too-many-arguments]
    apply: Callable[[jax.Array], jax.Array],
    start: FloatArray,
    duration: float,
    *,
    tol: float,
    m: int = 30,
    monitor: Callable[[FloatArray], None] | None = None,
) -> tuple[FloatArray, FloatArray, int]:
    """Get e^{L tau} u and int_0^tau e^{L t} u dt, with the number of applications of L.

    `monitor` sees the state at the start of every step.
    """
    integral = jnp.zeros(start.shape)
    state = jnp.asarray(start)
    count = 0
    for _, length, step, augmented in _steps(
        apply, start, duration, tol=tol, m=m, extra=1
    ):
        if monitor is not None:
            monitor(step.norm * np.asarray(step.blocks[0][0]))
        count += step.hessenberg.shape[1]
        rows = step.hessenberg.shape[0]
        e = scipy.linalg.expm(length * augmented)
        integral, state = (
            integral + step.combine(step.norm * e[None, :rows, rows])[0],
            step.combine(step.norm * e[None, :rows, 0])[0],
        )
    return np.array(state), np.array(integral), count


@_x64
def solve_by_propagation(
    apply: Callable[[jax.Array], jax.Array],
    rhs: FloatArray,
    duration: float,
    *,
    tol: float,
    m: int = 30,
) -> tuple[FloatArray, int]:
    """Get x = (-L)^{-1} b, for b orthogonal to the left null vectors of L, without a factorisation.

    (-L) x = b is (1 - e^{L tau}) x = int_0^tau e^{L t} b dt. The operator
    1 - e^{L tau} is close to the identity on every mode which decays within
    tau, and small only on the few slower ones, the hops between sites, so
    GMRES converges in a few iterations, each one propagation over tau
    (a time of the order of 1 / gamma).

    Raises
    ------
    RuntimeError
        If GMRES does not converge.
    """
    _, integral, count = propagate_state(apply, rhs, duration, tol=1e-2 * tol, m=m)
    applications = [count]

    def matvec(x: FloatArray) -> FloatArray:
        state, _, k = propagate_state(apply, x, duration, tol=1e-2 * tol, m=m)
        applications[0] += k
        return x - state

    # The stubs of scipy omit the matvec form of the constructor
    operator = scipy.sparse.linalg.LinearOperator(  # ty: ignore[missing-argument]
        (rhs.size, rhs.size),
        matvec=matvec,  # ty: ignore[unknown-argument]
        dtype=np.float64,  # ty: ignore[parameter-already-assigned]
    )
    solution, info = scipy.sparse.linalg.gmres(
        operator, integral, rtol=tol, atol=0, restart=60, maxiter=4
    )
    if info != 0:
        msg = "GMRES did not converge on (1 - e^{L tau}) x = int_0^tau e^{L t} b dt"
        raise RuntimeError(msg)
    return solution, applications[0]
