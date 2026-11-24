# cspell: ignore fori orthogonalize matvec triu unconverged
"""The Kramers operator applied without being stored, and exponential time steps in its Krylov spaces, for systems too large to factorise."""

import functools
from dataclasses import dataclass, field
from itertools import starmap
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from classical_diffusion.kramers._krylov import (
    BLOCK,
    INVARIANT,
    orthogonalize,
    remove_block,
)
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


_BANDS = (4, 2)
"""The slabs the Hermite and the plane wave axes are cut into (along their first axis), each trimmed to the basis (`_Kernel`)."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class _Kernel:
    """The real form of the Kramers operator as a stencil on blocks of the (Hermite box) x (plane wave box) array.

    L X = sum_j [k_j (P_j X) - D_j (F_j X)] - gamma sum_j R_j X, where P_j,
    D_j and R_j shift the order n_j by a few steps with a coefficient of the
    target order (`bands`), and F_j shifts the plane waves by the reciprocal
    vectors of the Fourier terms of the potential. In the box padded once,
    each term is a static slice times a coefficient. The Hermite triangle
    |n| <= N and the plane wave disc fill only part of the box, so the box
    is cut into slabs along n_1 and along the first plane wave axis, and the
    stencil runs only on each slab's bounding box of the basis: a few
    streaming loops, with no index arrays. The compact state is expanded
    into the box by one gather and read back from the blocks by another.
    """

    momentum: tuple[jax.Array, ...]
    derivative: tuple[jax.Array, ...]
    friction: tuple[jax.Array, ...]
    wavevector: jax.Array
    """(G - K)_j on the plane wave box, (n_dim, *box)."""
    expand: jax.Array
    """The compact index of each point of the box, or the number of unknowns (a zero) outside the basis."""
    positions: jax.Array
    """The point of the blocks (flattened in turn) of each unknown."""
    offsets: tuple[tuple[int, ...], ...] = field(metadata={"static": True})
    """The offsets of the bands of P, D and R."""
    forces: tuple[tuple[tuple[int, ...], tuple[float, ...]], ...] = field(
        metadata={"static": True}
    )
    """Each shift c of F and its coefficient for each j: constants of the program, which makes it 8% faster than reading them."""
    box: tuple[int, ...] = field(metadata={"static": True})
    margins: tuple[int, ...] = field(metadata={"static": True})
    blocks: tuple[tuple[tuple[int, int], ...], ...] = field(metadata={"static": True})
    """The range (first, end) of each block on each axis of the box."""


def _stencil(
    kernel: _Kernel, padded: jax.Array, block: tuple[tuple[int, int], ...]
) -> jax.Array:
    """Get L X on one block of the box, from the padded box."""
    n_dim = len(kernel.box) // 2
    first = [f for f, _ in block]
    shape = [e - f for f, e in block]

    def window(j: int, o: int, shift: tuple[int, ...] = (0,) * 3) -> jax.Array:
        start = [
            f + m + (o if a == j else 0)
            for a, (f, m) in enumerate(zip(first, kernel.margins, strict=True))
        ]
        for a in range(n_dim):
            start[n_dim + a] -= shift[a]
        return jax.lax.slice(
            padded, start, [s + b for s, b in zip(start, shape, strict=True)]
        )

    def along(c: jax.Array, j: int) -> jax.Array:
        return c[block[j][0] : block[j][1]].reshape(
            [-1 if a == j else 1 for a in range(len(shape))]
        )

    wavevector = kernel.wavevector[slice(None), *starmap(slice, block[n_dim:])]
    p_offsets, d_offsets, r_offsets = kernel.offsets
    out = jnp.zeros(shape)
    for j in range(n_dim):
        for o, c in zip(r_offsets, kernel.friction, strict=True):
            out -= along(c, j) * window(j, o)
        for o, c in zip(p_offsets, kernel.momentum, strict=True):
            out += along(c, j) * wavevector[j] * window(j, o)
        for o, c in zip(d_offsets, kernel.derivative, strict=True):
            for shift, coefficient in kernel.forces:
                if coefficient[j]:
                    out -= coefficient[j] * along(c, j) * window(j, o, shift)
    return out


@jax.jit
def _apply(kernel: _Kernel, x: jax.Array) -> jax.Array:
    padded = (
        jnp.append(x, 0.0)
        .at[kernel.expand]
        .get(mode="promise_in_bounds")
        .reshape(kernel.box)
    )
    padded = jnp.pad(
        jax.lax.optimization_barrier(padded), [(m, m) for m in kernel.margins]
    )
    # Each block is a vectorized loop; without the barriers XLA would fuse
    # them into the read back and evaluate the stencil at each position by
    # scattered loads (3x slower)
    out = [
        jax.lax.optimization_barrier(_stencil(kernel, padded, b)).reshape(-1)
        for b in kernel.blocks
    ]
    out = jnp.concatenate([*out, jnp.zeros(1)])
    return out.at[kernel.positions].get(mode="promise_in_bounds")


def _slabs(points: IntArray, count: int) -> list[tuple[int, int]]:
    """Cut the range of points[:, 0] into count slabs of about equal height, as (first, end)."""
    low, high = int(points[:, 0].min()), int(points[:, 0].max()) + 1
    cuts = np.unique(np.linspace(low, high, count + 1).round().astype(int))
    return list(zip(cuts[:-1].tolist(), cuts[1:].tolist(), strict=True))


def _bounding(points: IntArray, slab: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Get the bounding box, (first, end) per axis, of the points in a slab of the first axis."""
    inside = points[(points[:, 0] >= slab[0]) & (points[:, 0] < slab[1])]
    return tuple(
        (int(lo), int(hi) + 1)
        for lo, hi in zip(inside.min(axis=0), inside.max(axis=0), strict=True)
    )


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

    def apply(self, x: FloatArray) -> FloatArray:
        """Apply L to a flat state."""
        with jax.enable_x64(new_val=True):
            return np.asarray(_apply(self.kernel, jnp.asarray(x)))


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
        (tuple(int(i) for i in c), np.imag(f)) for c, f in get_force_shifts(model)
    ]
    low = vectors.min(axis=0)
    hermite_box = (max_order + 1,) * n_dim
    plane_wave_box = tuple(int(b) for b in vectors.max(axis=0) - low + 1)
    box = hermite_box + plane_wave_box
    waves = vectors - low

    def cut(points: IntArray, count: int) -> tuple[list, IntArray, IntArray, IntArray]:
        """Cut points into slabs: each slab's bounding box, and each point's slab, place in its box, and box size."""
        slabs = _slabs(points, count)
        boxes = [_bounding(points, slab) for slab in slabs]
        which = np.searchsorted([f for f, _ in slabs], points[:, 0], side="right") - 1
        first = np.array([[f for f, _ in b] for b in boxes])[which]
        shapes = [[e - f for f, e in b] for b in boxes]
        local = np.array(
            [
                np.ravel_multi_index(tuple(q), shapes[k])
                for q, k in zip(points - first, which, strict=True)
            ]
        )
        return boxes, which, local, np.array([int(np.prod(sh)) for sh in shapes])

    hermite_blocks, h_slab, h_local, h_size = cut(indices, _BANDS[0])
    wave_blocks, w_slab, w_local, w_size = cut(waves, _BANDS[1])
    blocks = [h + w for h in hermite_blocks for w in wave_blocks]
    sizes = np.outer(h_size, w_size).ravel()
    starts = np.concatenate([[0], np.cumsum(sizes)])[:-1].reshape(
        len(h_size), len(w_size)
    )
    # The place of each unknown in the blocks, flattened one after another
    positions = (
        starts[h_slab[:, None], w_slab[None, :]]
        + h_local[:, None] * w_size[w_slab][None, :]
        + w_local[None, :]
    )
    flat_box = (
        np.ravel_multi_index(tuple(indices.T), hermite_box)[:, None]
        * int(np.prod(plane_wave_box))
        + np.ravel_multi_index(tuple(waves.T), plane_wave_box)[None, :]
    )
    expand = np.full(int(np.prod(box)), flat_box.size, np.int32)
    expand[flat_box.ravel()] = np.arange(flat_box.size)
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
            forces=tuple((c, tuple(float(x) for x in f)) for c, f in shifts),
            box=box,
            margins=(_PAD,) * n_dim + tuple(int(r) for r in reach),
            blocks=tuple(blocks),
        )
    return MatrixFreeOperator(
        kernel=kernel,
        shape=(len(indices), len(vectors)),
        parity=np.asarray(1.0 - 2.0 * (indices.sum(axis=1) % 2), dtype=np.float64),
        indices=indices,
        width=width,
    )


@dataclass(frozen=True, kw_only=True)
class _Step:
    """The Krylov space of one step: an orthonormal basis V, as blocks of rows on the device, and L in it, H, with one more row (`_arnoldi`)."""

    start: jax.Array
    blocks: tuple[jax.Array, ...]
    hessenberg: FloatArray
    norm: float


@functools.partial(jax.jit, static_argnames="m")
def _arnoldi_device(
    kernel: _Kernel, u: jax.Array, m: int
) -> tuple[tuple[jax.Array, ...], jax.Array, jax.Array]:
    """Get the blocks of V, the Hessenberg H with one more row and |u|.

    Gram-Schmidt goes a block at a time (block modified Gram-Schmidt), over
    the blocks filled so far, twice where the first pass cancelled most of w
    (`orthogonalize`). The blocks are separate arrays, each read in a branch
    of its own: XLA copies a slice of one array of rows before a product
    (at 253k unknowns the copy doubled the time), and reading every row
    doubles the traffic. The 1D Arnoldi method slices its basis instead
    (`_krylov._extend`): its blocks are a few hundred kB and stay in the cache.
    """
    norm = jnp.linalg.norm(u)
    count = -(-(m + 1) // BLOCK)
    blocks = tuple(
        jnp.split(jnp.zeros((count * BLOCK, u.size)).at[0].set(u / norm), count)
    )

    def project(
        blocks: tuple[jax.Array, ...], k: jax.Array, w: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        overlaps = []
        for i, b in enumerate(blocks):
            w, h = jax.lax.cond(
                i * BLOCK <= k,
                functools.partial(remove_block, rows=b),
                lambda w: (w, jnp.zeros(BLOCK)),
                w,
            )
            overlaps.append(h)
        return w, jnp.concatenate(overlaps)[: m + 1]

    def step(k: jax.Array, carry: tuple) -> tuple:
        blocks, hessenberg, v = carry
        # Without the barrier XLA recomputes L v inside each of its consumers
        w = jax.lax.optimization_barrier(_apply(kernel, v))
        w, overlap = orthogonalize(functools.partial(project, blocks, k), w, m + 1)
        beta = jnp.linalg.norm(w)
        v = jnp.where(beta > 0, w / beta, 0.0)
        # A branch per block: a select over the blocks would rewrite each whole
        blocks = tuple(
            jax.lax.cond(
                (k + 1) // BLOCK == i,
                lambda b: b.at[(k + 1) % BLOCK].set(v),
                lambda b: b,
                b,
            )
            for i, b in enumerate(blocks)
        )
        return blocks, hessenberg.at[:, k].set(overlap.at[k + 1].set(beta)), v

    carry = (blocks, jnp.zeros((m + 1, m)), u / norm)
    blocks, hessenberg, _ = jax.lax.fori_loop(0, m, step, carry)
    return blocks, hessenberg, norm


def _arnoldi(operator: MatrixFreeOperator, u: jax.Array, m: int) -> _Step:
    """Get an orthonormal basis V of the Krylov space of m vectors of u, L in it, the Hessenberg H with one more row, and |u|, with u = |u| V e_0, in one call on the device.

    The space ends early, with a square H, where it is invariant.
    """
    blocks, hessenberg, norm = _arnoldi_device(operator.kernel, u, m)
    h = np.asarray(hessenberg)
    scale = np.maximum(np.abs(np.triu(h[:-1])).max(axis=0), 1e-300)
    invariant = np.flatnonzero(np.diagonal(h, -1) <= INVARIANT * scale)
    if len(invariant):
        h = h[: invariant[0] + 1, : invariant[0] + 1]
    return _Step(start=u, blocks=blocks, hessenberg=h, norm=float(norm))


def _padded(coefficients: FloatArray, step: _Step, k: int) -> FloatArray:
    """Pad coefficients to k states and to every row of the blocks, so that one shape compiles whatever the size of H."""
    out = np.zeros((k, len(step.blocks) * BLOCK))
    out[: len(coefficients), : coefficients.shape[1]] = coefficients
    return out


@jax.jit
def _combine(blocks: tuple[jax.Array, ...], c: jax.Array) -> jax.Array:
    return functools.reduce(
        jnp.add,
        (c[:, i * BLOCK : (i + 1) * BLOCK] @ b for i, b in enumerate(blocks)),
    )


_STATES = 8
"""The states whose J-forms are found from the states themselves (padded to this number, so one shape compiles), rather than from the J-Gram matrix of the basis."""


@jax.jit
def _forms(blocks: tuple[jax.Array, ...], c: jax.Array, parity: jax.Array) -> jax.Array:
    states = _combine(blocks, c)
    return jnp.sum(states * parity * states, axis=1)


_TILE = 2**16
"""The columns of the basis read at a time for its J-Gram matrix (`_gram`)."""


@jax.jit
def _gram(blocks: tuple[jax.Array, ...], parity: jax.Array) -> jax.Array:
    """Get V J V^T, a tile of columns at a time: the blocks joined whole, and times J, took 0.4 GB at 759k unknowns."""
    size = parity.size
    tile = min(_TILE, size)

    def add(i: jax.Array, gram: jax.Array) -> jax.Array:
        # The last tile starts early, over columns already added, with no weight
        start = jnp.minimum(i * tile, size - tile)
        rows = jnp.concatenate(
            [jax.lax.dynamic_slice_in_dim(b, start, tile, axis=1) for b in blocks]
        )
        weight = jnp.where(
            start + jnp.arange(tile) >= i * tile,
            jax.lax.dynamic_slice_in_dim(parity, start, tile),
            0.0,
        )
        return gram + (rows * weight) @ rows.T

    count = len(blocks) * BLOCK
    return jax.lax.fori_loop(0, -(-size // tile), add, jnp.zeros((count, count)))


def _get_forms(step: _Step, coefficients: FloatArray, parity: jax.Array) -> FloatArray:
    """Get s^T J s for each state s = c V, by whichever is cheaper: the states, O(k m n) for k of them, or the J-Gram matrix of the basis, O(m^2 n)."""
    k, rows = coefficients.shape
    if k <= _STATES:
        padded = _padded(coefficients, step, _STATES)
        return np.asarray(_forms(step.blocks, padded, parity))[:k]
    gram = np.asarray(_gram(step.blocks, parity))[:rows, :rows]
    return np.einsum("ki,ij,kj->k", coefficients, gram, coefficients)


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
    operator: MatrixFreeOperator,
    start: FloatArray,
    t_end: float,
    *,
    tol: float,
    m: int,
    extra: int,
) -> Iterator[tuple[float, float, _Step, FloatArray, jax.Array]]:
    """Step u(t) = e^{L t} u to t_end, yielding (t, h, the Krylov space of u(t), the augmented H, and u(t + h) with the `extra` integrals over the step) of each step.

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
        step = _arnoldi(operator, u, m)
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
        columns = [0, *range(rows, rows + extra)]
        states = _combine(
            step.blocks,
            _padded(step.norm * exponential[:rows, columns].T, step, 1 + extra),
        )
        yield t, length, step, augmented, states
        u = states[0]
        t += length
        h = min(0.9 * length * (allowed / max(error, 1e-300)) ** (1 / cols), 5 * length)


@dataclass(frozen=True, kw_only=True)
class Propagation:
    """Measurements at each time, with the number of applications of L they took."""

    values: FloatArray
    n_applications: int


@_x64
def propagate_form(  # ruff:ignore[too-many-arguments]
    operator: MatrixFreeOperator,
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
    for t, length, step, augmented, _ in _steps(
        operator, start, float(half[-1]), tol=tol / 2, m=m, extra=0
    ):
        count += step.hessenberg.shape[1]
        rows = step.hessenberg.shape[0]
        if monitor is not None:
            monitor(np.asarray(step.start))
        end = int(np.searchsorted(half, t + length * (1 + 1e-12), side="right"))
        coefficients = np.array(
            [
                step.norm * scipy.linalg.expm((h - t) * augmented)[:rows, 0]
                for h in half[i:end]
            ]
        ).reshape(-1, rows)
        if len(coefficients):
            out[i:end] = _get_forms(step, coefficients, device_parity)
        i = end
    return Propagation(values=out, n_applications=count)


@_x64
def propagate_state(  # ruff:ignore[too-many-arguments]
    operator: MatrixFreeOperator,
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
    for _, _, step, _, states in _steps(
        operator, start, duration, tol=tol, m=m, extra=1
    ):
        if monitor is not None:
            monitor(np.asarray(step.start))
        count += step.hessenberg.shape[1]
        state = states[0]
        integral += states[1]
    return np.array(state), np.array(integral), count


@_x64
def solve_by_propagation(
    operator: MatrixFreeOperator,
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
    _, integral, count = propagate_state(operator, rhs, duration, tol=1e-2 * tol, m=m)
    applications = [count]

    def matvec(x: FloatArray) -> FloatArray:
        state, _, k = propagate_state(operator, x, duration, tol=1e-2 * tol, m=m)
        applications[0] += k
        return x - state

    # The stubs of scipy omit the matvec form of the constructor
    system = scipy.sparse.linalg.LinearOperator(  # ty: ignore[missing-argument]
        (rhs.size, rhs.size),
        matvec=matvec,  # ty: ignore[unknown-argument]
        dtype=np.float64,  # ty: ignore[parameter-already-assigned]
    )
    solution, info = scipy.sparse.linalg.gmres(
        system, integral, rtol=tol, atol=0, restart=60, maxiter=4
    )
    if info != 0:
        msg = "GMRES did not converge on (1 - e^{L tau}) x = int_0^tau e^{L t} b dt"
        raise RuntimeError(msg)
    return solution, applications[0]
