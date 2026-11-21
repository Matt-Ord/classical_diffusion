"""The generator of the overdamped (Smoluchowski) limit: the gradient operator A_+ on the plane waves, and the equilibrium amplitude."""

from typing import TYPE_CHECKING

import numpy as np

from classical_diffusion.kramers._model import get_fourier_coefficients, get_rows

if TYPE_CHECKING:
    from classical_diffusion.kramers._model import (
        ComplexArray,
        FloatArray,
        IntArray,
        Model,
    )


def _get_index(vectors: IntArray) -> dict[tuple[int, ...], int]:
    return {tuple(int(i) for i in v): j for j, v in enumerate(get_rows(vectors))}


def get_extended_vectors(model: Model, vectors: IntArray) -> IntArray:
    """Get the plane waves and every one shifted by a term of the potential: the outputs of `get_gradient_operator`."""
    shifts = [np.array(g) for g, _ in model.potential if any(g)]
    return np.unique(np.concatenate([vectors, *[vectors + s for s in shifts]]), axis=0)


def get_gradient_operator(
    model: Model, vectors: IntArray, delta_k: FloatArray
) -> ComplexArray:
    """Get A_+ = (grad - i K) + grad V / 2 on the plane waves, (n_dim, n_out, n_in).

    The output plane waves are the inputs and every input shifted by a term
    of the potential, so A_+^dag A_+ is exact on the inputs.
    """
    extended = get_extended_vectors(model, vectors)
    out_index = _get_index(extended)
    n_in = len(vectors)
    operator = np.zeros((model.n_dim, len(extended), n_in), dtype=np.complex128)
    k_in = vectors @ model.reciprocal - delta_k
    rows = np.array([out_index[tuple(int(i) for i in v)] for v in get_rows(vectors)])
    operator[:, rows, np.arange(n_in)] += 1j * k_in.T
    for g, v_g in model.potential:
        if not any(g):
            continue
        q = np.array(g) @ model.reciprocal
        rows = np.array(
            [out_index[tuple(int(i) for i in v)] for v in vectors + np.array(g)]
        )
        # d_j V = sum_G i G_j V_G e^{i G x}, halved
        operator[:, rows, np.arange(n_in)] += 0.5j * q[:, None] * v_g
    return operator


def get_amplitude(model: Model, vectors: IntArray, n_grid: int) -> ComplexArray:
    """Get the plane wave coefficients of the equilibrium amplitude w = e^{-V / 2}."""
    return get_fourier_coefficients(
        model, vectors, lambda v: np.exp(-v / 2), n_grid=n_grid
    )


def get_grid_size(vectors: IntArray) -> int:
    """Get a grid which resolves the plane waves without aliasing their products with the potential."""
    return int(4 * np.abs(vectors).max() + 8)
