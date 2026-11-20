"""The Kramers operator of a periodic system at K, in Hermite functions of the momentum and plane waves of the position, applied without being stored."""

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse

if TYPE_CHECKING:
    from classical_diffusion.kramers._model import (
        ComplexArray,
        FloatArray,
        IntArray,
        Model,
    )


def get_hermite_indices(n_dim: int, max_order: int) -> IntArray:
    """Get the multi-indices (n_1, ..., n_d) of total order up to max_order, (n_hermite, n_dim), lowest order first."""
    indices = [
        n
        for total in range(max_order + 1)
        for n in itertools.product(range(total + 1), repeat=n_dim)
        if sum(n) == total
    ]
    return np.array(indices, dtype=np.int64).reshape(-1, n_dim)


def get_lowering(indices: IntArray) -> list[scipy.sparse.csr_array]:
    """Get the lowering operators b_j, b_j |n> = sqrt(n_j) |n - e_j>, on the multi-indices."""
    lookup = {tuple(int(i) for i in n): k for k, n in enumerate(indices)}
    out = list[scipy.sparse.csr_array]()
    for j in range(indices.shape[1]):
        rows, cols, values = [], [], []
        for k, n in enumerate(indices):
            if n[j] == 0:
                continue
            lower = list(int(i) for i in n)
            lower[j] -= 1
            rows.append(lookup[tuple(lower)])
            cols.append(k)
            values.append(np.sqrt(n[j]))
        size = len(indices)
        out.append(scipy.sparse.csr_array((values, (rows, cols)), shape=(size, size)))
    return out


def get_shift_operators(
    model: Model, vectors: IntArray, delta_k: FloatArray
) -> tuple[list[scipy.sparse.csr_array], list[scipy.sparse.csr_array]]:
    """Get A_-j = (d_j - i K_j) - d_j V / 2 and A_+j = (d_j - i K_j) + d_j V / 2 on the plane waves, truncated to them."""
    lookup = {tuple(int(i) for i in v): k for k, v in enumerate(vectors)}
    size = len(vectors)
    k_in = vectors @ model.reciprocal - delta_k
    minus, plus = [], []
    for j in range(model.n_dim):
        rows = list(range(size))
        cols = list(range(size))
        derivative = list(1j * k_in[:, j])
        force_rows, force_cols, force = [], [], []
        for g, v_g in model.potential:
            if not any(g):
                continue
            q = (np.array(g) @ model.reciprocal)[j]
            for k, v in enumerate(vectors):
                target = lookup.get(tuple(int(i) for i in v + np.array(g)))
                if target is not None:
                    force_rows.append(target)
                    force_cols.append(k)
                    # d_j V / 2 = sum_G (i G_j V_G / 2) e^{i G x}
                    force.append(0.5j * q * v_g)
        d = scipy.sparse.csr_array((derivative, (rows, cols)), shape=(size, size))
        f = scipy.sparse.csr_array(
            (force, (force_rows, force_cols)), shape=(size, size)
        )
        minus.append((d - f).tocsr())
        plus.append((d + f).tocsr())
    return minus, plus


@dataclass(frozen=True, kw_only=True, eq=False)
class KramersOperator:
    """L = -sum_j (b_j A_-j + b_j^dag A_+j) - gamma N on states of shape (n_hermite, n_plane_waves).

    The streaming part is anti-Hermitian and odd under the parity P of the
    momentum, P |n> = (-1)^{|n|} |n>, and the friction Hermitian and even, so
    L^dag = P L P.
    """

    indices: IntArray
    """The Hermite multi-indices, (n_hermite, n_dim)."""
    vectors: IntArray
    """The plane waves, as integer vectors, (n_plane_waves, n_dim)."""
    friction: float
    lowering: list[scipy.sparse.csr_array]
    raising: list[scipy.sparse.csr_array]
    minus_t: list[scipy.sparse.csr_array]
    """A_-j^T, applied from the right."""
    plus_t: list[scipy.sparse.csr_array]
    """A_+j^T, applied from the right."""
    order: FloatArray
    """|n| of each multi-index."""

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.indices), len(self.vectors))

    @property
    def parity(self) -> FloatArray:
        """(-1)^{|n|} of each multi-index."""
        return 1.0 - 2.0 * (self.order % 2)

    def apply(self, x: ComplexArray) -> ComplexArray:
        """Apply L to a state of shape (n_hermite, n_plane_waves)."""
        x = np.asarray(x, dtype=np.complex128)
        out = -self.friction * self.order[:, None] * x
        for b, bt, am, ap in zip(
            self.lowering, self.raising, self.minus_t, self.plus_t, strict=True
        ):
            # (b X A_-^T) is b applied to the Hermite axis and A_- to the plane waves
            out -= b @ (x @ am)
            out -= bt @ (x @ ap)
        return out


def build_operator(
    model: Model, vectors: IntArray, max_order: int, delta_k: FloatArray
) -> KramersOperator:
    """Build the Kramers operator at K on the plane waves and the Hermite functions up to total order max_order."""
    indices = get_hermite_indices(model.n_dim, max_order)
    lowering = get_lowering(indices)
    minus, plus = get_shift_operators(model, vectors, delta_k)
    return KramersOperator(
        indices=indices,
        vectors=vectors,
        friction=model.friction,
        lowering=lowering,
        raising=[b.T.tocsr() for b in lowering],
        minus_t=[a.T.tocsr() for a in minus],
        plus_t=[a.T.tocsr() for a in plus],
        order=indices.sum(axis=1).astype(np.float64),
    )
