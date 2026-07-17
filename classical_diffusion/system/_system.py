from dataclasses import dataclass
from functools import cached_property
from typing import override

import sympy as sp


@dataclass(frozen=True, kw_only=True)
class System:
    """Parameters representing a physical system."""

    gamma: float
    temperature: float
    m: float
    potential: tuple[int, sp.Expr]

    @cached_property
    def kbt(self) -> float:
        """Return the kbt of system."""
        return 1.0 * self.temperature  # k_B set to 1 for now

    @cached_property
    def symbolic_coordinates(self) -> tuple:
        """Return the symbolic coordinates of the system."""
        return sp.symbols(f"x0:{self.n_dim}")

    @property
    def n_dim(self) -> int:
        """Return the number of dimensions of the system."""
        return self.potential[0]

    @property
    def potential_expr(self) -> sp.Expr:
        """Return the potential expression of the system."""
        return self.potential[1]

    @cached_property
    def force_expr(self) -> list[sp.Expr]:
        """Return the symbolic force of the system."""
        return [-sp.diff(self.potential_expr, c) for c in self.symbolic_coordinates]

    def with_gamma(self, gamma: float) -> System:
        return System(gamma=gamma)


class HarmonicSystem(System):
    """Parameters representing a simple harmonic oscillator system."""

    _omega: float

    def __init__(
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        omega: float,
        n_dim: int = 1,
    ) -> None:
        potential = 0.5 * omega**2 * sp.symbols("x0") ** 2
        self._omega = omega

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(n_dim, potential),
        )

    @property
    def omega(self) -> float:
        """Return the angular frequency of the system."""
        return self._omega

    @override
    def with_gamma(self, gamma: float) -> HarmonicSystem:
        return HarmonicSystem(
            gamma=gamma,
            temperature=self.temperature,
            m=self.m,
            omega=self.omega,
            n_dim=self.n_dim,
        )


class PeriodicSystem1D(System):
    """Parameters for a 1D cosine potential system."""

    _delta_x: float
    _barrier_energy: float

    def __init__(  # noqa: PLR0913
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        delta_x: float,
        barrier_energy: float,
        n_dim: int = 1,
    ) -> None:
        potential = 0.5 * barrier_energy * (1 - sp.cos(sp.symbols("x0") / delta_x))
        self._delta_x = delta_x
        self._barrier_energy = barrier_energy

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(n_dim, potential),
        )

    @property
    def delta_x(self) -> float:
        """Return the delta x of the system."""
        return self._delta_x

    @property
    def barrier_energy(self) -> float:
        """Return the barrier energy of the system."""
        return self._barrier_energy


class FlatSystem1D(System):
    """Parameters for a 1D cosine potential system."""

    _delta_x: float
    _barrier_energy: float

    def __init__(
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        n_dim: int = 1,
    ) -> None:
        potential = 0 * sp.symbols("x0")

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(n_dim, potential),
        )


class FlatSystem2D(System):
    """Parameters for a 1D cosine potential system."""

    _delta_x: float
    _barrier_energy: float

    def __init__(
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        n_dim: int = 2,
    ) -> None:
        potential = 0 * sp.symbols("x0")

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(n_dim, potential),
        )


def _get_potential_expr_fcc(barrier_energy: float, delta_x: float) -> sp.Expr:
    """Return the potential energy expression for a 2D FCC lattice."""
    x0, x1 = sp.symbols("x0 x1")
    c = 2.0 * sp.pi / (sp.sqrt(3.0) * delta_x)

    kx0 = c * (-1.0 / sp.sqrt(3.0))
    kx1 = c * 1.0

    ky0 = c * (2.0 / sp.sqrt(3.0))
    ky1 = 0.0

    arg1 = x0 * kx0 + x1 * kx1
    arg2 = x0 * ky0 + x1 * ky1
    arg3 = arg1 + arg2

    cos_sum = sp.cos(arg1) + sp.cos(arg2) + sp.cos(arg3)

    return 2.0 * barrier_energy * cos_sum


class PeriodicSystemFCC(System):
    """Parameters for the face-centered cubic periodic system."""

    _delta_x: float
    _barrier_energy: float

    def __init__(  # noqa: PLR0913
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        delta_x: float,
        barrier_energy: float,
        n_dim: int = 1,
    ) -> None:
        potential = _get_potential_expr_fcc(barrier_energy, delta_x)
        self._delta_x = delta_x
        self._barrier_energy = barrier_energy

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(n_dim, potential),
        )

    @property
    def delta_x(self) -> float:
        """Return the delta x of the system."""
        return self._delta_x

    @property
    def barrier_energy(self) -> float:
        """Return the barrier energy of the system."""
        return self._barrier_energy
