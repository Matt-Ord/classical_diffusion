"""The time evolution of a measurement of exp(L t) X_0, from a rational Krylov reduced model of L."""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from classical_diffusion.caldeira_leggett._linear import (
    EigenLiouvillian,
    apply_eigen,
    from_eigen,
    get_eigen_liouvillian,
    solve_eigen,
    to_eigen,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.caldeira_leggett._config import CaldeiraLeggettConfig
    from classical_diffusion.caldeira_leggett._liouvillian import (
        Liouvillian,
    )
    from classical_diffusion.caldeira_leggett._model import ComplexArray, FloatArray


@dataclass(frozen=True, kw_only=True)
class RationalKrylovConfig:
    """The convergence of the reduced model of an ISF or mean square displacement (`get_reduced_evolution`)."""

    rtol: float = 1e-9
    """The largest relative residual |X_0 - (s - L) V (s - L_m)^-1 V^dagger X_0| / |X_0| allowed at any candidate shift."""
    max_dimension: int = 160
    """The largest number of basis vectors, each a state."""


DEFAULT_RATIONAL_KRYLOV_CONFIG = RationalKrylovConfig()

_N_FREQUENCIES = 60


def _candidates(slowest: float, fastest: float) -> np.ndarray:
    """Shifts s = sigma + i omega on the line Re s = sigma = slowest, the contour of the inverse Laplace transform.

    The error of the reduced model at a time t is the integral of e^{s t}
    times the error of its Laplace transform along that line, which is
    controlled by the residual there, out to the fastest rate of L. Only
    omega >= 0 is needed, as each shift is taken with its conjugate.
    """
    omega = np.concatenate([[0.0], np.geomspace(slowest, fastest, _N_FREQUENCIES)])
    return slowest + 1j * omega


class _ReducedModel:
    """An orthonormal basis V with X_0 its first vector, L_m = V^dagger L V, and the part R = L V - V L_m of L V outside the basis.

    The residual X_0 - (z - L) V (z - L_m)^-1 V^dagger X_0 at a shift z is exactly
    R c, with c = (z - L_m)^-1 V^dagger X_0, so its norm needs only R^dagger R.
    """

    def __init__(self, eigen: EigenLiouvillian, initial: jax.Array) -> None:
        self._eigen = eigen
        self.norm = float(jnp.linalg.norm(initial))
        self.basis: list[jax.Array] = []
        self._outside: list[jax.Array] = []
        self.reduced = np.zeros((0, 0), complex)
        self._gram = np.zeros((0, 0), complex)
        """R^dagger R."""
        self.add(initial)

    def add(self, x: jax.Array) -> bool:
        """Add the part of x outside the basis, if it is not round off."""
        # Gram-Schmidt, twice for orthogonality to round off
        for _ in range(2):
            for v in self.basis:
                x -= jnp.vdot(v, x) * v
        size = float(jnp.linalg.norm(x))
        if size < 1e-14 * self.norm:
            return False
        new = x / size
        n = len(self.basis)
        grown = np.zeros((n + 1, n + 1), complex)
        grown[:n, :n] = self.reduced
        # The new row, <new| L |v_j> = <new| R_j> = a_j, which then leaves R_j,
        # so that <R_i| R_j> loses a_i^* a_j
        for j, r in enumerate(self._outside):
            grown[n, j] = complex(jnp.vdot(new, r))
            self._outside[j] = r - grown[n, j] * new
        row = grown[n, :n]
        gram = np.zeros((n + 1, n + 1), complex)
        gram[:n, :n] = self._gram - np.outer(row.conj(), row)
        self.basis.append(new)
        image = apply_eigen(self._eigen, new, jnp.asarray(0j))
        for _ in range(2):
            for i, v in enumerate(self.basis):
                overlap = complex(jnp.vdot(v, image))
                grown[i, n] += overlap
                image -= overlap * v
        self._outside.append(image)
        gram[:, n] = [complex(jnp.vdot(r, image)) for r in self._outside]
        gram[n, :] = gram[:, n].conj()
        self._gram = gram
        self.reduced = grown
        return True

    def _start(self) -> ComplexArray:
        start = np.zeros(len(self.basis), complex)
        start[0] = self.norm
        return start

    def guess(self, shift: complex) -> jax.Array:
        """Get V (shift - L_m)^-1 V^dagger X_0, the reduced model's solution of (shift - L) X = X_0."""
        c = np.linalg.solve(
            shift * np.eye(len(self.basis)) - self.reduced, self._start()
        )
        return sum(
            (complex(cj) * v for cj, v in zip(c, self.basis, strict=True)),
            start=jnp.zeros_like(self.basis[0]),
        )

    def residuals(self, shifts: ComplexArray) -> FloatArray:
        """Get the relative residual at each shift."""
        gram = self._gram
        identity = np.eye(len(self.basis))
        out = []
        for z in shifts:
            c = np.linalg.solve(z * identity - self.reduced, self._start())
            out.append(np.sqrt(max((c.conj() @ gram @ c).real, 0.0)) / self.norm)
        return np.array(out)

    def evolve(
        self, measure: Callable[[jax.Array], jax.Array], times: FloatArray
    ) -> ComplexArray:
        """Get measure(V) exp(L_m t) V^dagger X_0 at the times."""
        growing = np.linalg.eigvals(self.reduced).real.max()
        if growing > 0:
            warnings.warn(
                f"The reduced Liouvillian has an eigenvalue with real part {growing:.1e} > 0",
                stacklevel=3,
            )
        compiled = jax.jit(lambda y: measure(from_eigen(self._eigen, y)))
        measured = np.array([complex(compiled(b)) for b in self.basis])
        start = self._start()
        return np.array(
            [measured @ scipy.linalg.expm(self.reduced * t) @ start for t in times]
        )


def get_reduced_evolution(  # ruff:ignore[too-many-arguments]
    liouvillian: Liouvillian,
    initial: jax.Array,
    measure: Callable[[jax.Array], jax.Array],
    times: FloatArray,
    *,
    config: RationalKrylovConfig,
    linear_config: CaldeiraLeggettConfig,
) -> ComplexArray:
    """Get measure(exp(L t) initial) at the times, from a rational Krylov reduced model of L.

    measure must be linear over the complex numbers, as it is applied to each
    basis vector rather than to the state: take any real part of the result.

    Each solve X_j = (s_j - L)^-1 X_0 at a shift s_j is preconditioned by
    (s_j - N)^-1, the exact inverse of the no-jump part (`solve_eigen`), so the
    Hamiltonian, the stiff part of L, costs nothing, and the solve only
    iterates over the jumps. The solutions, with X_0, are orthonormalized
    into V, and L is projected to L_m = V^dagger L V, a matrix of the number of
    solves, so that

        measure(exp(L t) X_0) = measure(V) exp(L_m t) V^dagger X_0

    for every time at once. This matches the Laplace transform
    measure((s - L)^-1 X_0) at each shift, rather than the derivatives at
    t = 0 that a polynomial Krylov basis matches, so it needs no time steps:
    for hydrogen on Cu(111) in 2D, an ISF to 3000 hbar / k_B T needs 8 to 9
    times fewer applications of L than a Krylov exponential integrator to
    the same accuracy. Each solve starts from the reduced model's solution
    at its shift, which saves 15-20% of the applications for lithium in 2D.

    The shifts are chosen greedily, from candidates on the contour of the
    inverse Laplace transform, Re s = 1 / span, out to the largest no-jump
    rate (`_candidates`): the next shift is the one whose residual is
    largest (`_ReducedModel`). L is not real, so each
    complex shift is taken with its conjugate. The basis is complete once
    every residual is below config.rtol of |X_0|, or once the largest is at a
    shift already solved, where it is the round off of that solve.

    Raises
    ------
    RuntimeError
        If the residual is still above config.rtol after config.max_dimension vectors.
    """
    # The basis, the solves and the residuals are all in the eigenvectors of
    # the effective Hamiltonians (`EigenLiouvillian`), and only the measure
    # returns to the bands
    eigen = get_eigen_liouvillian(liouvillian)
    start = to_eigen(eigen, initial)
    model = _ReducedModel(eigen, start)
    candidates = _candidates(1 / times[-1], float(jnp.abs(eigen.rates).max()))
    shift = complex(1 / times[-1])
    solved = set[complex]()
    while True:
        solved.add(shift)
        grew = False
        for s in (shift,) if shift.imag == 0 else (shift, shift.conjugate()):
            # (s - L) X = X_0, from the reduced model's solution, whose
            # residual is already small once the basis is large
            x = solve_eigen(eigen, -start, linear_config, shift=s, guess=model.guess(s))
            grew |= model.add(x)
        residuals = model.residuals(candidates)
        worst = int(np.argmax(residuals))
        # At a shift already solved, the residual is set by the accuracy of
        # that solve, near a slow mode its round off, so solving it again only
        # adds noise to the basis
        if residuals[worst] <= config.rtol or not grew or candidates[worst] in solved:
            return model.evolve(measure, times)
        if len(model.basis) >= config.max_dimension:
            msg = (
                f"The rational Krylov residual is {residuals[worst]:.1e}, above "
                f"{config.rtol:.1e}, with {len(model.basis)} vectors"
            )
            raise RuntimeError(msg)
        shift = complex(candidates[worst])
