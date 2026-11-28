"""The eigenvalues of a nonsymmetric tridiagonal matrix and the weights of its first row and column, by LAPACK."""

# cspell: ignore fori
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg


def get_size(m: int) -> int:
    """Get the padded size, 2^k or 3 2^(k-1), so that compiled shapes are few."""
    size = 2
    while True:
        for candidate in (size, 3 * size // 2):
            if candidate >= m:
                return candidate
        size *= 2


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


def get_poles(
    alpha: jax.Array, coupling: jax.Array, steps: int
) -> tuple[jax.Array, jax.Array]:
    """Get the eigenvalues lambda_j of T and their weights w_j, e_1^T (s - T)^-1 e_1 = sum_j w_j / (s - lambda_j), padded to get_size(steps) by poles of weight zero.

    T is tridiagonal with the first `steps` of alpha on its diagonal and the
    products of its off-diagonals beta_k delta_k = coupling_k. Its
    symmetric form, with sqrt(coupling_k) on both off-diagonals, is similar
    to it by a diagonal matrix whose first entry is one, so it has the same
    eigenvalues and weights w_j = V_{0j} (V^-1)_{j0}, found by LAPACK eig,
    which is backward stable and splits clusters of close eigenvalues (the
    copies of converged poles that two-sided Lanczos makes) consistently.
    An O(m^2) Ehrlich-Aberth iteration on the device took as long for H and
    Li on Cu(111) in 2D (Li: 84 s against 85 s of the evolution), and needed
    the weights of clusters fixed from their moments.
    """
    a = np.asarray(alpha)[:steps]
    off = np.sqrt(np.asarray(coupling)[: steps - 1].astype(complex))
    poles, vectors = scipy.linalg.eig(np.diag(a) + np.diag(off, 1) + np.diag(off, -1))
    first = np.zeros(steps, complex)
    first[0] = 1
    weights = vectors[0] * np.linalg.solve(vectors, first)
    size = get_size(steps)
    # The padding decays, with no weight, so it adds nothing to any measurement
    rates = np.full(size, -1.0 + 0j)
    rates[:steps] = poles
    padded = np.zeros(size, complex)
    padded[:steps] = weights
    return jnp.asarray(rates), jnp.asarray(padded)
