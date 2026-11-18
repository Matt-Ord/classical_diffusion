# cspell: ignore fori
from typing import Any, ClassVar, override

import diffrax as dfx
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg

type _State = Any


def _krylov_coefficients(
    beta: jax.Array, hessenberg: jax.Array, h: jax.Array
) -> jax.Array:
    """Get the coefficients of exp(h L) y on the m + 1 vectors of the basis, from the (m + 1, m + 1) Hessenberg matrix with a column of zeros appended."""
    return beta * jax.scipy.linalg.expm(h * hessenberg)[:, 0]


class KrylovInterpolation(dfx.AbstractLocalInterpolation):
    """The exact solution exp((t - t0) L) y(t0) within a step of `KrylovExponential`.

    The Krylov basis of the step gives the solution at any time in it, to the
    same error as at its end, so the ISF is as accurate between steps as at
    them, however long the steps are.
    """

    # diffrax declares the times as abstract fields (equinox AbstractVar)
    t0: Any  # ty: ignore[invalid-attribute-override]
    t1: Any  # ty: ignore[invalid-attribute-override]
    beta: jax.Array
    hessenberg: jax.Array
    """The Hessenberg matrix of L in the basis, with a column of zeros appended, (m + 1, m + 1)."""
    basis: jax.Array
    """The vectors V of the basis, flattened, (m + 1, n)."""
    y0: _State
    """The state at t0, whose structure the solution takes."""

    @override
    def evaluate(self, t0: Any, t1: Any = None, left: bool = True) -> _State:
        del left
        unravel = jax.flatten_util.ravel_pytree(self.y0)[1]
        c0 = _krylov_coefficients(self.beta, self.hessenberg, t0 - self.t0)
        if t1 is None:
            return unravel(c0 @ self.basis)
        c1 = _krylov_coefficients(self.beta, self.hessenberg, t1 - self.t0)
        return unravel((c1 - c0) @ self.basis)


class KrylovExponential(dfx.AbstractAdaptiveSolver):
    """A step y(t + h) = exp(h L) y(t), for a linear and time independent vector field L, in a Krylov basis.

    The basis of the m vectors y, L y, ..., L^{m-1} y is orthogonalized as
    it is built (the Arnoldi method), which gives L in it as an upper
    Hessenberg matrix H, and exp(h L) y = |y| V exp(h H) e_1. With the row
    h_{m+1,m} of the next vector, and a column of zeros, H is square, and its
    exponential also gives the leading error h h_{m+1,m} [phi_1(h H)]_{m,0}
    on v_{m+1} (Sidje 1998, Expokit), which is added to the step as a
    correction and used by the step size controller as for a method of
    order m. Unlike the term in exp(h H) of Saad (1992), it does not fall
    off on long steps, which would leave their error unchecked. Within a step
    the same basis gives the solution at any time (`KrylovInterpolation`).

    An explicit Runge-Kutta method is limited to steps of about 1 / |lambda|,
    for the largest eigenvalue lambda of L, the coherence between the highest
    and lowest bands. A state has little weight on these fast coherences, so
    a small basis gives exp(h L) y over far longer steps: for lithium on
    Cu(111) the steps are 50 times longer, and the ISF is 17 times faster.
    """

    dimension: int = 20
    """The number of vectors m of the Krylov basis.

    The cost hardly depends on m: the steps grow about as fast as the work in
    each, so from m = 8 to 32 the ISF takes within 10% of the same time, for
    lithium and hydrogen on Cu(111) at 0.2 to 5 times their friction. m = 20
    is never more than 20% slower than the best m, in 21 states of memory.
    """
    term_structure: ClassVar = dfx.AbstractTerm
    interpolation_cls: ClassVar = KrylovInterpolation

    @override
    def order(self, terms: Any) -> int:
        return self.dimension

    @override
    def init(
        self,
        terms: Any,
        t0: Any,
        t1: Any,
        y0: _State,
        args: Any,
    ) -> None:
        return None

    @override
    def func(self, terms: Any, t0: Any, y0: _State, args: Any) -> _State:
        return terms.vf(t0, y0, args)

    @override
    def step(
        self,
        terms: Any,
        t0: Any,
        t1: Any,
        y0: _State,
        args: Any,
        solver_state: None,
        made_jump: Any,
    ) -> tuple[_State, _State, dict[str, _State], None, dfx.RESULTS]:
        m = self.dimension
        flat, unravel = jax.flatten_util.ravel_pytree(y0)

        def apply(v: jax.Array) -> jax.Array:
            return jax.flatten_util.ravel_pytree(terms.vf(t0, unravel(v), args))[0]

        beta = jnp.linalg.norm(flat)
        basis = jnp.zeros((m + 1, flat.size), flat.dtype).at[0].set(flat / beta)
        hessenberg = jnp.zeros((m + 1, m), flat.dtype)

        def arnoldi(
            j: int, carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            basis, hessenberg = carry
            w = apply(basis[j])
            built = jnp.arange(m + 1) <= j
            # Gram-Schmidt twice keeps the basis orthogonal to round off
            for _ in range(2):
                overlap = jnp.where(built, basis.conj() @ w, 0)
                w -= overlap @ basis
                hessenberg = hessenberg.at[:, j].add(overlap)
            norm = jnp.linalg.norm(w)
            # If the basis spans an invariant subspace, exp(h H) e_1 is exact
            next_vector = jnp.where(norm > 0, w / jnp.where(norm > 0, norm, 1), 0)
            return basis.at[j + 1].set(next_vector), hessenberg.at[j + 1, j].set(norm)

        basis, hessenberg = jax.lax.fori_loop(0, m, arnoldi, (basis, hessenberg))
        square = jnp.zeros((m + 1, m + 1), hessenberg.dtype).at[:, :m].set(hessenberg)
        coefficients = _krylov_coefficients(beta, square, t1 - t0)
        y1 = unravel(coefficients @ basis)
        error = unravel(coefficients[m] * basis[m])
        dense = {"beta": beta, "hessenberg": square, "basis": basis, "y0": y0}
        return y1, error, dense, None, dfx.RESULTS.successful
