"""Gauss quadrature of the spectral measure of a Hermitian operator in a state, by the Lanczos method."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.kramers._model import ComplexArray, FloatArray


@dataclass(frozen=True, kw_only=True)
class Quadrature:
    """Nodes and weights of a measure, <v| f(H) |v> = sum_j weights_j f(nodes_j), nodes in increasing order."""

    nodes: FloatArray
    weights: FloatArray
    residuals: FloatArray
    """|H y_j - lambda_j y_j| of each Ritz vector y_j, zero where the Krylov space is invariant."""


def get_lanczos_quadrature(
    apply: Callable[[ComplexArray], ComplexArray],
    start: ComplexArray,
    *,
    converged: Callable[[Quadrature, Quadrature], bool],
    block: int = 8,
    max_steps: int | None = None,
) -> Quadrature:
    """Get the Gauss quadrature of the spectral measure of H in the state start, <start| f(H) |start> ~ sum_j w_j f(lambda_j).

    The Lanczos method builds an orthonormal basis of the Krylov space of
    start, in which H is tridiagonal, T. The eigenvalues of T and the squares
    of the first components of its eigenvectors, times |start|^2, are the
    nodes and weights of the Gauss quadrature of the measure: it is exact for
    every polynomial f of degree below 2 m. It is extended a block of steps
    at a time, until `converged(previous, current)` holds for two blocks in
    a row. The basis is
    reorthogonalised in full, so the nodes do not repeat; the basis is m
    states of memory.
    """
    norm = float(np.linalg.norm(start))
    n = start.size
    max_steps = n if max_steps is None else min(max_steps, n)
    basis = np.zeros((min(4 * block, max_steps) + 1, n), dtype=np.complex128)
    basis[0] = start / norm
    alpha = list[float]()
    beta = list[float]()
    previous: Quadrature | None = None
    agreed = 0
    while True:
        # Blocks grow with the basis, so the quadrature is found a few times
        target = min(len(alpha) + max(block, len(alpha) // 4), max_steps)
        if target + 1 > len(basis):
            grown = np.zeros((min(2 * len(basis), max_steps + 1), n), np.complex128)
            grown[: len(basis)] = basis
            basis = grown
        while len(alpha) < target:
            j = len(alpha)
            w = apply(basis[j])
            a = float(np.vdot(basis[j], w).real)
            alpha.append(a)
            built = basis[: j + 1]
            # Gram-Schmidt twice against every vector keeps the basis orthogonal
            for _ in range(2):
                w -= (built @ w.conj()).conj() @ built
            b = float(np.linalg.norm(w))
            if len(alpha) == max_steps or b <= 1e-14 * max(abs(a), 1.0):
                # The Krylov space is invariant: the quadrature is exact
                max_steps = len(alpha)
                break
            beta.append(b)
            basis[j + 1] = w / b
        nodes, vectors = scipy.linalg.eigh_tridiagonal(
            np.array(alpha), np.array(beta[: len(alpha) - 1]), lapack_driver="stev"
        )
        last = beta[len(alpha) - 1] if len(beta) >= len(alpha) else 0.0
        current = Quadrature(
            nodes=nodes,
            weights=norm**2 * np.abs(vectors[0]) ** 2,
            residuals=last * np.abs(vectors[-1]),
        )
        agreed = (
            agreed + 1 if previous is not None and converged(previous, current) else 0
        )
        # Two blocks in a row must agree: a slow mode of small weight can
        # leave one block unchanged before the Krylov space finds it
        if len(alpha) >= max_steps or agreed >= 2:  # ruff:ignore[magic-value-comparison]
            return current
        previous = current
