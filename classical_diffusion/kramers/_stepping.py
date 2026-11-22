# cspell: ignore ascontiguousarray indptr matvec tocsr unconverged
"""The Kramers operator applied without being stored, and exponential time steps in its Krylov spaces, for systems too large to factorise."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from classical_diffusion.kramers._krylov import get_arnoldi_space
from classical_diffusion.kramers._real import (
    _get_ladder_matrices,
    _get_plane_wave_operators,
    _on_indices,
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


@dataclass(frozen=True, kw_only=True, eq=False)
class MatrixFreeOperator:
    """The real form of the Kramers operator (`build_real_operator`) on states of shape (n_hermite, n_plane_waves).

    L X = sum_j [(P_j X) k_j - (D_j X) F_j^T] - gamma sum_j R_j X, with P_j,
    D_j and R_j the real forms of p_j, d/dp_j and the friction on the
    Hermite functions, k_j = (G - K)_j and F_j the real form of d_j V on the
    plane waves. It holds only these sparse factors, a few entries per row
    of either axis, rather than their product.
    """

    momentum: list[scipy.sparse.csr_array]
    derivative: list[scipy.sparse.csr_array]
    friction: scipy.sparse.csr_array
    """gamma sum_j R_j."""
    wavevector: FloatArray
    """(G - K), (n_dim, n_plane_waves)."""
    force: list[scipy.sparse.csr_array]
    """F_j."""
    shape: tuple[int, int]
    parity: FloatArray
    """(-1)^{|n|} of each Hermite multi-index."""
    indices: IntArray
    width: float

    def apply(self, x: FloatArray) -> FloatArray:
        """Apply L to a flat state.

        D_j and F_j act on different axes, so (D_j X) F_j^T = D_j (F_j X^T)^T:
        the plane wave factor is applied to one transposed copy of the state,
        each a sparse product with contiguous columns.
        """
        state = x.reshape(self.shape)
        out = -(self.friction @ state)
        transposed = np.ascontiguousarray(state.T)
        for p, d, k, f in zip(
            self.momentum, self.derivative, self.wavevector, self.force, strict=True
        ):
            out += (p @ state) * k
            out -= d @ np.ascontiguousarray((f @ transposed).T)
        return out.reshape(-1)


def build_matrix_free_operator(
    model: Model, vectors: IntArray, max_order: int, delta_k: FloatArray
) -> MatrixFreeOperator:
    """Build the matrix-free real form of the Kramers operator at K."""
    width = get_width(max_order, get_momentum_reach(model))
    indices = get_hermite_indices(model.n_dim, max_order)
    p, dp, friction = _get_ladder_matrices(max_order, width)
    phase = 1j ** indices.sum(axis=1)
    derivatives, forces = _get_plane_wave_operators(model, vectors, delta_k)

    def real_form(matrix: FloatArray, j: int) -> scipy.sparse.csr_array:
        lifted = _on_indices(matrix, indices, j)
        return (
            scipy.sparse.diags_array(phase.conj())
            @ lifted
            @ scipy.sparse.diags_array(phase)
        ).tocsr()

    def as_real(matrix: scipy.sparse.csr_array, part: str) -> scipy.sparse.csr_array:
        data = matrix.data.imag if part == "imag" else matrix.data.real
        out = scipy.sparse.csr_array(
            (data, matrix.indices, matrix.indptr), shape=matrix.shape
        )
        out.eliminate_zeros()
        return out

    # p and d/dp are odd, so their real forms are i times a real matrix,
    # as are d_j - i K_j = i k_j and d_j V; the friction is even and real
    momentum = [as_real(real_form(p, j), "imag") for j in range(model.n_dim)]
    derivative = [as_real(real_form(dp, j), "imag") for j in range(model.n_dim)]
    total_friction = sum(
        (as_real(real_form(friction, j), "real") for j in range(model.n_dim)),
        start=scipy.sparse.csr_array((len(indices),) * 2),
    )
    return MatrixFreeOperator(
        momentum=momentum,
        derivative=derivative,
        friction=(model.friction * total_friction).tocsr(),
        wavevector=np.array([d.diagonal().imag for d in derivatives]),
        force=[as_real(f, "imag") for f in forces],
        shape=(len(indices), len(vectors)),
        parity=np.asarray(1.0 - 2.0 * (indices.sum(axis=1) % 2), dtype=np.float64),
        indices=indices,
        width=width,
    )


@dataclass(frozen=True, kw_only=True)
class _Step:
    """The Krylov space of one step: an orthonormal basis V and L in it, H, with one more row (`get_arnoldi_space`)."""

    basis: FloatArray
    hessenberg: FloatArray
    norm: float


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


def _arnoldi(apply: Callable[[FloatArray], FloatArray], u: FloatArray, m: int) -> _Step:
    basis, hessenberg, norm = get_arnoldi_space(apply, u, m)
    return _Step(basis=basis, hessenberg=hessenberg, norm=norm)


def _steps(  # ruff:ignore[too-many-arguments]
    apply: Callable[[FloatArray], FloatArray],
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
    u = start
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
        u = step.norm * exponential[:rows, 0] @ step.basis
        t += length
        h = min(0.9 * length * (allowed / max(error, 1e-300)) ** (1 / cols), 5 * length)


@dataclass(frozen=True, kw_only=True)
class Propagation:
    """Measurements at each time, with the number of applications of L they took."""

    values: FloatArray
    n_applications: int


def propagate_form(  # ruff:ignore[too-many-arguments]
    apply: Callable[[FloatArray], FloatArray],
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
        rows = step.basis.shape[0]
        if monitor is not None:
            monitor(step.norm * step.basis[0])
        while i < len(half) and half[i] <= t + length * (1 + 1e-12):
            c = step.norm * scipy.linalg.expm((half[i] - t) * augmented)[:rows, 0]
            state = c @ step.basis
            out[i] = state @ (parity * state)
            i += 1
    return Propagation(values=out, n_applications=count)


def propagate_state(  # ruff:ignore[too-many-arguments]
    apply: Callable[[FloatArray], FloatArray],
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
    integral = np.zeros_like(start)
    state = start
    count = 0
    for _, length, step, augmented in _steps(
        apply, start, duration, tol=tol, m=m, extra=1
    ):
        if monitor is not None:
            monitor(step.norm * step.basis[0])
        count += step.hessenberg.shape[1]
        rows = step.basis.shape[0]
        e = scipy.linalg.expm(length * augmented)
        integral += step.norm * e[:rows, rows] @ step.basis
        state = step.norm * e[:rows, 0] @ step.basis
    return state, integral, count


def solve_by_propagation(
    apply: Callable[[FloatArray], FloatArray],
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
