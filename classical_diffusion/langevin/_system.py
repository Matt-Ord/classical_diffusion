import zlib
from dataclasses import dataclass, field
from functools import cached_property
from typing import final, override

import jax
import numpy as np
import sympy as sp

from classical_diffusion.hopping._system import KramersParameters


def _hash_sympy_expr(expr: sp.Expr) -> int:
    stable_string = sp.srepr(expr)
    return zlib.crc32(stable_string.encode("utf-8"))


@dataclass(frozen=True, kw_only=True)
class System:
    """Parameters representing a physical system."""

    gamma: float
    temperature: float
    m: float
    potential: tuple[int, sp.Expr]
    params: tuple[float, ...] = ()

    @property
    def n_dim(self) -> int:
        """The number of dimensions of the system."""
        return self.potential[0]

    @cached_property
    def coordinate_symbols(self) -> tuple[sp.Symbol, ...]:
        """The symbols of each coordinate of the system."""
        return sp.symbols(f"x0:{self.n_dim}")

    @property
    def n_params(self) -> int:
        """The number of parameters of the system."""
        return len(self.params)

    @cached_property
    def parameter_symbols(self) -> tuple[sp.Symbol, ...]:
        """The symbols of each parameter of the system."""
        return sp.symbols(f"s0:{self.n_params}")

    @cached_property
    def lambda_symbols(self) -> tuple[sp.Symbol, ...]:
        """The symbols of the system, including coordinates and parameters."""
        return (*self.coordinate_symbols, *self.parameter_symbols)

    @property
    def potential_expr(self) -> sp.Expr:
        """The potential expression of the system."""
        return self.potential[1]

    @cached_property
    def force_expr(self) -> list[sp.Expr]:
        """The symbolic force of the system."""
        return [
            -sp.simplify(sp.diff(self.potential_expr, c))
            for c in self.coordinate_symbols
        ]

    def with_gamma(self, gamma: float) -> System:
        """Return the system with exchanged gamma."""
        return System(
            gamma=gamma,
            m=self.m,
            temperature=self.temperature,
            potential=self.potential,
        )

    def as_canonical(self) -> CanonicalSystem:
        """Return the canonical form of the system."""
        return CanonicalSystem(
            gamma=self.gamma,
            temperature=self.temperature,
            m=self.m,
            potential=self.potential,
            params=self.params,
            force=self.force_expr,
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.gamma,
                self.temperature,
                self.m,
                self.params,
                (
                    self.potential[0],
                    _hash_sympy_expr(self.potential[1]),
                ),
            )
        )

    @property
    def kbt(self) -> float:
        """Convert to simulation parameters."""
        return self.temperature

    @property
    def sampling_domain(self) -> tuple[float, float]:
        """The domain over which the equilibrium x-density should be sampled."""
        return (-np.inf, np.inf)


@jax.tree_util.register_dataclass
@final
@dataclass(frozen=True, kw_only=True)
class CanonicalSystem(System):
    """Parameters representing a physical system."""

    potential: tuple[int, sp.Expr] = field(metadata={"static": True})

    force: list[sp.Expr] = field(metadata={"static": True})


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
        s0 = sp.Symbol("s0")
        potential = 0.5 * s0**2 * sp.symbols("x0") ** 2

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(n_dim, potential),
            params=(omega,),
        )

    @property
    def omega(self) -> float:
        """The angular frequency of the system."""
        return self.params[0]

    @override
    def with_gamma(self, gamma: float) -> HarmonicSystem:
        return HarmonicSystem(
            gamma=gamma,
            temperature=self.temperature,
            m=self.m,
            omega=self.omega,
            n_dim=self.n_dim,
        )


class KramersSystem1D(System):
    """Parameters representing a periodic double harmonic system."""

    def __init__(
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        kramers_params: KramersParameters,
        n_dim: int = 1,
    ) -> None:
        x0 = sp.symbols("x0")
        s0 = sp.symbols("s0")
        s1 = sp.symbols("s1")
        s2 = sp.symbols("s2")

        # To express a double harmonic potential as a sympy expression, consider moving two harmonics towards each other
        # y = 1/2 omega_well^2 x^2                      represents an upright harmonic centred at the origin
        # y = E_b - 1/2 omega_barrier^2 (x - x_0)^2     represents an inverted harmonic centred at x_0, height E_b
        # At the point where the double harmonic potential is smooth, the two harmonics just touch. So there will only
        # be one solution to the above simultaneous system. Solving these equations gives a quadratic for x: then
        # setting the determinant equal to zero gives an expression for x_0
        # From this, the overlap point, x_meet, can be found and the expression for the periodic potential is as below.

        omegas_ss = s0**2 + s1**2  # Omegas squared sum
        x_0 = sp.sqrt((2 * omegas_ss * s2) / (omegas_ss * s1**2 - s1**4))
        x_meet = (s1**2 / omegas_ss) * x_0

        periodic_x = sp.Mod(x0 + x_meet, 2 * x_0) - x_meet

        potential = sp.Piecewise(
            (0.5 * s0**2 * periodic_x**2, periodic_x <= x_meet),
            (
                s2 - 0.5 * s1**2 * (periodic_x - x_0) ** 2,
                periodic_x >= x_meet,
            ),
            (0, True),
        )

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(n_dim, potential),
            params=(
                kramers_params.omega_well,
                kramers_params.omega_barrier,
                kramers_params.barrier_energy,
            ),
        )

    @property
    def delta_x(self) -> float:
        """The delta x of the system."""
        omega_well, omega_barrier = self.omega_well, self.omega_barrier
        omegas_ss = omega_well**2 + omega_barrier**2  # Omegas squared sum
        return float(
            2
            * np.sqrt(
                (2 * omegas_ss * self.barrier_energy)
                / (omegas_ss * omega_barrier**2 - omega_barrier**4)
            )
        )

    @property
    def kramers_params(self) -> KramersParameters:
        return KramersParameters(self.params[0], self.params[1], self.params[2])

    @property
    def omega_well(self) -> float:
        """The harmonic frequency at the bottom of the well."""
        return self.params[0]

    @property
    def omega_barrier(self) -> float:
        """The harmonic frequency at the top of the barrier."""
        return self.params[1]

    @property
    def barrier_energy(self) -> float:
        """The barrier energy of the system."""
        return self.params[2]

    @override
    def with_gamma(self, gamma: float) -> KramersSystem1D:
        return KramersSystem1D(
            gamma=gamma,
            temperature=self.temperature,
            m=self.m,
            kramers_params=KramersParameters(
                self.omega_well, self.omega_barrier, self.barrier_energy
            ),
            n_dim=self.n_dim,
        )

    @override
    @cached_property
    def force_expr(self) -> list[sp.Expr]:
        """The symbolic force of the system."""
        x0 = sp.symbols("x0")
        s0 = sp.symbols("s0")
        s1 = sp.symbols("s1")
        s2 = sp.symbols("s2")

        omegas_ss = s0**2 + s1**2  # Omegas squared sum
        x_0 = sp.sqrt((2 * omegas_ss * s2) / (omegas_ss * s1**2 - s1**4))
        x_meet = (s1**2 / omegas_ss) * x_0

        periodic_x = sp.Mod(x0 + x_meet, 2 * x_0) - x_meet

        derivative = sp.Piecewise(
            (s0**2 * periodic_x, periodic_x <= x_meet),
            (
                -(s1**2) * (periodic_x - x_0),
                periodic_x >= x_meet,
            ),
            (0, True),
        )

        return [-1 * derivative]


class PeriodicSystem1D(System):
    """Parameters for a 1D cosine potential system."""

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        delta_x: float,
        barrier_energy: float,
        n_dim: int = 1,
    ) -> None:
        s0 = sp.Symbol("s0")
        s1 = sp.Symbol("s1")
        potential = 0.5 * s1 * (1 - sp.cos(2 * sp.pi * sp.symbols("x0") / s0))

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(n_dim, potential),
            params=(delta_x, barrier_energy),
        )

    @property
    def delta_x(self) -> float:
        """The delta x of the system."""
        return self.params[0]

    @property
    def barrier_energy(self) -> float:
        """The barrier energy of the system."""
        return self.params[1]

    @override
    def with_gamma(self, gamma: float) -> PeriodicSystem1D:
        return PeriodicSystem1D(
            gamma=gamma,
            temperature=self.temperature,
            m=self.m,
            delta_x=self.delta_x,
            barrier_energy=self.barrier_energy,
            n_dim=self.n_dim,
        )

    @override
    @property
    def sampling_domain(self) -> tuple[float, float]:
        """The domain over which the equilibrium x-density should be sampled."""
        return (-self.delta_x / 2, self.delta_x / 2)


def _get_potential_expr_fcc() -> sp.Expr:
    """Return the potential energy expression for a 2D FCC lattice."""
    x0, x1 = sp.symbols("x0 x1")
    s0, s1 = sp.symbols("s0 s1")
    c = 2.0 * sp.pi / (sp.sqrt(3.0) * s0)

    kx0 = c * (-1.0 / sp.sqrt(3.0))
    kx1 = c * 1.0

    ky0 = c * (2.0 / sp.sqrt(3.0))
    ky1 = 0.0

    arg1 = x0 * kx0 + x1 * kx1
    arg2 = x0 * ky0 + x1 * ky1
    arg3 = arg1 + arg2

    cos_sum = sp.cos(arg1) + sp.cos(arg2) + sp.cos(arg3)

    return 2.0 * s1 * cos_sum


class PeriodicSystemFCC(System):
    """Parameters for the face-centered cubic periodic system."""

    def __init__(
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        delta_x: float,
        barrier_energy: float,
    ) -> None:
        potential = _get_potential_expr_fcc()

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(2, potential),
            params=(delta_x, barrier_energy),
        )

    @property
    def delta_x(self) -> float:
        """The delta x of the system."""
        return self.params[0]

    @property
    def barrier_energy(self) -> float:
        """The barrier energy of the system."""
        return self.params[1]


def get_diffusion_time(system: System, characteristic_length: float) -> float:
    """Return the average time for a particle to traverse a characteristic length."""
    return np.sqrt(system.m * characteristic_length / system.kbt)


def get_characteristic_periodic_mass(system: PeriodicSystem1D) -> float:
    """Return the characteristic mass for a 1D periodic system."""
    return system.kbt * system.delta_x**2 / system.gamma**2
