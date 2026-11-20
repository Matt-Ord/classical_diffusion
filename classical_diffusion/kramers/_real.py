"""The Kramers operator in its real form, in scaled Hermite functions of the momentum and plane waves of the position, as a sparse matrix."""

from typing import TYPE_CHECKING

import functools

import numpy as np
import scipy.sparse

from classical_diffusion.caldeira_leggett._model import get_saddle_energy
from classical_diffusion.kramers._operator import get_hermite_indices

if TYPE_CHECKING:
    from classical_diffusion.kramers._model import (
        ComplexArray,
        FloatArray,
        IntArray,
        Model,
    )

_THERMAL_WIDTH = np.sqrt(2.0)
"""The width s of the Hermite functions h_n(p / s) / sqrt(s) whose first is sqrt of the Maxwellian, e^{-p^2 / 4}."""


def get_momentum_reach(model: Model) -> float:
    """Get the momentum, in thermal momenta, which the Hermite functions must reach: sqrt(2 (E_s + 25)).

    A particle diffuses by crossing the saddle of the potential at energy
    E_s above its minimum (`get_saddle_energy`), so the momenta that carry
    it reach sqrt(2 E_s) at the bottom of a well, and the Boltzmann factor
    e^{-25} beyond 25 k_B T more is negligible at any tolerance. For lithium
    on Cu(111) (E_s = 0.6 k_B T) this is 7.2 thermal momenta: a reach of
    8 sqrt(2) needed Hermite functions to above order 80 for its ISF in 2D,
    a reach of 5 sqrt(2) to order 44, and below that the tail of the
    Maxwellian is cut. For hydrogen (E_s = 10.5 k_B T) it is 8.4: with
    7.1, D converged only algebraically in the order.
    """
    return _get_momentum_reach(tuple(model.potential), model.n_dim)


@functools.cache
def _get_momentum_reach(potential: tuple, n_dim: int) -> float:
    saddle = get_saddle_energy(
        tuple((g, float(np.real(v))) for g, v in potential), n_dim
    )
    minimum = _get_potential_minimum(potential, n_dim)
    return float(np.sqrt(2 * (saddle - minimum + _MARGIN)))


_MARGIN = 25.0
"""The energy above the saddle, in units of k_B T, up to which the Hermite functions resolve the momentum."""


def _get_potential_minimum(potential: tuple, n_dim: int) -> float:
    n_grid = 64 if n_dim < 3 else 24  # ruff:ignore[magic-value-comparison]
    u = np.stack(
        np.meshgrid(*[np.arange(n_grid) / n_grid] * n_dim, indexing="ij"), axis=-1
    )
    values = np.zeros(u.shape[:-1], dtype=np.complex128)
    for g, v in potential:
        values += v * np.exp(2j * np.pi * (u @ np.array(g, dtype=float)))
    return float(values.real.min())


def get_width(max_order: int, reach: float) -> float:
    """Get the width s of the Hermite functions h_n(p / s) / sqrt(s) up to an order, which reach the momentum `reach` (in thermal momenta).

    The Hermite functions up to order N reach |p| ~ s sqrt(2 N + 1), and
    resolve structure in p of size ~ s / sqrt(2 N). Those whose first is the
    root of the Maxwellian (s = sqrt 2) reach ever further into the tail of
    the Maxwellian as N grows, while the structure that needs N, the
    filaments of phase mixing and the boundary layer of the separatrix at
    low friction, lies within the reach (`get_momentum_reach`). So once they
    pass it, the width shrinks to keep the reach, which resolves that
    structure with ~ gamma^{-1/2} functions rather than ~ 1 / gamma: for D
    of lithium on Cu(111) at 1e10 s^-1, 512 rather than 32768.
    """
    return min(_THERMAL_WIDTH, reach / np.sqrt(2 * max_order + 1))


def _get_ladder_matrices(max_order: int, width: float) -> tuple[FloatArray, ...]:
    """Get p, d/dp and the friction -d^2/dp^2 + p^2 / 4 - 1 / 2 in the Hermite functions of the width up to max_order, exactly.

    Each is found in two more functions, so the products are exact, then cut.
    """
    size = max_order + 3
    lowering = np.diag(np.sqrt(np.arange(1, size)), 1)
    q = (lowering + lowering.T) / np.sqrt(2)
    dq = (lowering - lowering.T) / np.sqrt(2)
    p = width * q
    dp = dq / width
    friction = -dp @ dp + p @ p / 4 - np.eye(size) / 2
    keep = slice(0, max_order + 1)
    return p[keep, keep], dp[keep, keep], friction[keep, keep]


def _on_indices(
    matrix: FloatArray, indices: IntArray, j: int
) -> scipy.sparse.csr_array:
    """Lift a matrix on the order of component j to the multi-indices, where every other component is unchanged."""
    lookup = {tuple(int(i) for i in n): k for k, n in enumerate(indices)}
    rows, cols, values = [], [], []
    for k, n in enumerate(indices):
        for m_j in np.flatnonzero(matrix[:, n[j]]):
            m = list(int(i) for i in n)
            m[j] = int(m_j)
            target = lookup.get(tuple(m))
            if target is not None:
                rows.append(target)
                cols.append(k)
                values.append(matrix[m_j, n[j]])
    size = len(indices)
    return scipy.sparse.csr_array((values, (rows, cols)), shape=(size, size))


def get_momentum_overlaps(
    max_order: int, width: float
) -> tuple[FloatArray, FloatArray]:
    """Get the components of psi_0 and p psi_0 on the Hermite functions of the width, psi_0^2 the Maxwellian.

    The integrals are found by the trapezoid rule on a fine grid, exact to
    round off for these Gaussians.
    """
    p = np.linspace(-40, 40, 16001)
    step = p[1] - p[0]
    psi = (2 * np.pi) ** -0.25 * np.exp(-(p**2) / 4)
    q = p / width
    # The orthonormal Hermite functions, by their stable recurrence
    h = np.zeros((max_order + 1, len(p)))
    h[0] = np.pi**-0.25 * np.exp(-(q**2) / 2)
    if max_order > 0:
        h[1] = np.sqrt(2) * q * h[0]
    for n in range(2, max_order + 1):
        h[n] = np.sqrt(2 / n) * q * h[n - 1] - np.sqrt((n - 1) / n) * h[n - 2]
    h /= np.sqrt(width)
    return h @ psi * step, h @ (p * psi) * step


def get_force_shifts(model: Model) -> list[tuple[IntArray, ComplexArray]]:
    """Get the shift c and coefficient i q_c V_c of each term of grad V, q_c = c @ reciprocal."""
    return [
        (
            np.array(c, dtype=np.int64),
            1j * complex(v) * (np.array(c) @ model.reciprocal),
        )
        for c, v in model.potential
        if any(c)
    ]


def _get_plane_wave_operators(
    model: Model, vectors: IntArray, delta_k: FloatArray
) -> tuple[list[scipy.sparse.csr_array], list[scipy.sparse.csr_array]]:
    """Get d_j - i K_j and d_j V on the plane waves, truncated to them."""
    lookup = {tuple(int(i) for i in v): k for k, v in enumerate(vectors)}
    size = len(vectors)
    k = vectors @ model.reciprocal - delta_k
    derivatives, forces = [], []
    shifts = get_force_shifts(model)
    for j in range(model.n_dim):
        derivatives.append(scipy.sparse.diags_array(1j * k[:, j]).tocsr())
        rows, cols, values = [], [], []
        for shift, coefficient in shifts:
            for source, v in enumerate(vectors):
                target = lookup.get(tuple(int(i) for i in v + shift))
                if target is not None:
                    rows.append(target)
                    cols.append(source)
                    values.append(coefficient[j])
        forces.append(
            scipy.sparse.csr_array((values, (rows, cols)), shape=(size, size))
        )
    return derivatives, forces


def build_real_operator(
    model: Model, vectors: IntArray, max_order: int, delta_k: FloatArray
) -> tuple[scipy.sparse.csr_array, FloatArray, IntArray, float]:
    """Get the Kramers operator at K in its real form, with the parity J of each unknown, the Hermite multi-indices and their width.

    In the frame symmetrised by the equilibrium, the Kramers operator is
    L = -p . (grad - i K) + grad V . grad_p - gamma sum_j (-d_j^2 + p_j^2 / 4 - 1 / 2),
    where the first two terms are anti-Hermitian and odd in p and the last
    Hermitian and even, so L^dag = P L P for the parity P of the momentum.
    In Hermite functions of the momentum (`get_width`), which are
    orthonormal and of parity (-1)^{|n|}, and plane waves of the position,
    with the amplitude of each multiplied by i^{|n|}, it is real, and
    J-symmetric, L^T = J L J with J = (-1)^{|n|}. The unknowns are ordered
    with the plane waves fastest.
    """
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
        )

    operator = scipy.sparse.csr_array((len(indices) * len(vectors),) * 2, dtype=complex)
    identity = scipy.sparse.identity(len(vectors), format="csr")
    for j in range(model.n_dim):
        operator = operator - scipy.sparse.kron(real_form(p, j), derivatives[j])
        operator = operator + scipy.sparse.kron(real_form(dp, j), forces[j])
        operator = operator - model.friction * scipy.sparse.kron(
            real_form(friction, j), identity
        )
    operator = scipy.sparse.csr_array(operator)
    operator.sum_duplicates()
    if operator.nnz and np.max(np.abs(operator.data.imag)) > 1e-12 * np.max(
        np.abs(operator.data)
    ):
        msg = "The real form of the Kramers operator is not real"
        raise AssertionError(msg)
    real = scipy.sparse.csr_array(
        (operator.data.real, operator.indices, operator.indptr), shape=operator.shape
    )
    real.eliminate_zeros()
    parity = np.repeat(1.0 - 2.0 * (indices.sum(axis=1) % 2), len(vectors))
    return real, parity, indices, width


def get_start(
    indices: IntArray,
    width: float,
    amplitude: FloatArray,
    direction: FloatArray | None = None,
) -> FloatArray:
    """Get w psi_0(p), or e . p w psi_0(p) for a direction, in the real form (times a phase of i for the velocity, which is odd).

    In Hermite functions of the thermal width this is w in the slot n = 0
    (or e_j); in narrower ones psi_0 spreads over the even orders.
    """
    max_order = int(indices.sum(axis=1).max())
    even, odd = get_momentum_overlaps(max_order, width)
    n_dim = indices.shape[1]
    phase = (1j ** -indices.sum(axis=1)).astype(complex)
    coefficient = np.zeros(len(indices), dtype=complex)
    components = range(n_dim) if direction is not None else [None]
    for j in components:
        term = np.ones(len(indices), dtype=complex)
        for i in range(n_dim):
            term *= (odd if i == j else even)[indices[:, i]]
        coefficient += term * (1.0 if j is None else direction[j])
    coefficient *= phase
    if direction is not None:
        # The odd orders carry a phase of +-i: drop the common factor
        coefficient *= 1j
    return np.kron(coefficient.real, amplitude)
