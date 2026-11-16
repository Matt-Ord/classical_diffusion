# cspell: ignore fori
from typing import Any, ClassVar, override

import diffrax as dfx
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg

type _State = Any


class KrylovExponential(dfx.AbstractAdaptiveSolver):
    """A step y(t + h) = exp(h L) y(t), for a linear and time independent vector field L, in a Krylov basis.

    The basis of the m vectors y, L y, ..., L^{m-1} y is orthogonalized as
    it is built (the Arnoldi method), which gives L in it as an upper
    Hessenberg matrix H, and exp(h L) y = |y| V exp(h H) e_1. Its error is the
    leading term beta h_{m+1,m} [exp(h H)]_{m,0} v_{m+1} (Saad 1992), which
    the step size controller uses as for a method of order m.

    An explicit Runge-Kutta method is limited to steps of about 1 / |lambda|,
    for the largest eigenvalue lambda of L, the coherence between the highest
    and lowest bands. A state has little weight on these fast coherences, so
    a small basis gives exp(h L) y over far longer steps: for lithium on
    Cu(111) the steps are 50 times longer, and the ISF is 17 times faster.
    """

    dimension: int = 12
    """The number of vectors m of the Krylov basis."""
    term_structure: ClassVar = dfx.AbstractTerm
    interpolation_cls: ClassVar = dfx.LocalLinearInterpolation

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
        exponential = jax.scipy.linalg.expm((t1 - t0) * hessenberg[:m])[:, 0]
        y1 = unravel(beta * (exponential @ basis[:m]))
        error = unravel(beta * hessenberg[m, m - 1] * exponential[m - 1] * basis[m])
        return y1, error, {"y0": y0, "y1": y1}, None, dfx.RESULTS.successful
