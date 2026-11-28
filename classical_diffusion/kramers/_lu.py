# cspell: ignore sparsax spineax cuDSS UMFPACK SuiteSparse jitted
"""Sparse LU factors as jax values: `factorize` a matrix once, then `SparseFactor.solve` inside jitted loops.

The factor is KLU's (SuiteSparse), held by sparsax through the XLA FFI,
which runs on the CPU only, so everything that holds a factor runs on the
CPU (`on_cpu`), whatever the default device: with a GPU as the default
the 1D solvers stay on the CPU, where their problems (10^3-10^5 unknowns,
under a second) belong. A GPU would need another backend behind the same
two calls, a factor whose solve traces into jax (spineax, over cuDSS). UMFPACK, the other LU of sparsax, made the shift-
invert Arnoldi method of hydrogen on Cu(111) at 0.1 times its friction
6-15 % faster, but D 1.8 times slower, at up to 1.6 times the memory, and it
ran out of its 32 bit workspace on the bordered matrix of D of lithium on
Cu(111) at 3e9 / s, where KLU took 1.8 GB and 17 s (SuperLU 2.0 GB, 18 s).
"""

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import numpy as np
import scipy.sparse
import sparsax

if TYPE_CHECKING:
    from collections.abc import Callable


def on_cpu[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Run a function, and every jax array it makes, on the CPU, where the factors of sparsax live."""

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with jax.default_device(jax.devices("cpu")[0]):
            return function(*args, **kwargs)

    return wrapped


def get_padded_size(size: int) -> int:
    """Get the size of the arrays of a matrix of this size: the next power of sqrt(2), and at least _SMALLEST.

    The jitted programs are compiled once for each size (0.25 s each for the
    Arnoldi method), so the bases of a basis that grows by 1.3 each round
    share a few of them, and the small bases (milliseconds of work each)
    share one.
    """
    return max(_SMALLEST, int(np.ceil(2 ** (np.ceil(2 * np.log2(size)) / 2))))


_SMALLEST = 4096


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class SparseFactor:
    """The LU factor of a matrix padded by the identity to its padded size (`get_padded_size`)."""

    token: jax.Array
    """The handle of sparsax to the factor."""
    size: int = field(metadata={"static": True})
    """The padded size."""

    def solve(self, b: jax.Array) -> jax.Array:
        """Get A^{-1} b, for b of the padded size: traceable, so it runs inside jitted loops."""
        return sparsax.lu_solve_factor(self.token, b)


@functools.partial(jax.jit, static_argnames="size")
def _factorize(
    rows: jax.Array, cols: jax.Array, values: jax.Array, size: int
) -> jax.Array:
    return sparsax.lu_factor(rows, cols, values, size)


@on_cpu
def factorize(matrix: scipy.sparse.sparray) -> SparseFactor:
    """Factor a square sparse matrix, padded by the identity to its padded size.

    The entries are padded too, by zeros on the diagonal (sparsax sums
    duplicates), to the next power of two, so that one compile serves the
    matrices of similar size.
    """
    coo = scipy.sparse.coo_array(matrix)
    n = coo.shape[0]
    size = get_padded_size(n)
    extra = np.arange(n, size)
    count = len(coo.data) + len(extra)
    padding = 2 ** int(np.ceil(np.log2(count))) - count
    rows = np.concatenate([coo.row, extra, np.zeros(padding, int)])
    cols = np.concatenate([coo.col, extra, np.zeros(padding, int)])
    values = np.concatenate([coo.data, np.ones(len(extra)), np.zeros(padding)])
    with jax.enable_x64(new_val=True):
        token = _factorize(
            jax.device_put(rows.astype(np.int32)),
            jax.device_put(cols.astype(np.int32)),
            jax.device_put(values),
            size,
        )
    return SparseFactor(token=token, size=size)


@jax.jit
def _solve(factor: SparseFactor, b: jax.Array) -> jax.Array:
    return factor.solve(b)


@on_cpu
def solve(matrix: scipy.sparse.sparray, b: np.ndarray) -> np.ndarray:
    """Get A^{-1} b, by one factor (`factorize`)."""
    factor = factorize(matrix)
    padded = np.zeros(factor.size)
    padded[: b.size] = b
    with jax.enable_x64(new_val=True):
        return np.asarray(_solve(factor, jax.device_put(padded)))[: b.size]
