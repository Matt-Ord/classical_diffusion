import zlib
from dataclasses import dataclass, field
from functools import cached_property
from typing import final, override

import jax
import numpy as np
import sympy as sp

from classical_diffusion.hopping import KramersParameters
from classical_diffusion.system import CanonicalUnitSystem, UnitSystem


def _hash_sympy_expr(expr: sp.Expr) -> int:
    stable_string = sp.srepr(expr)
    return zlib.crc32(stable_string.encode("utf-8"))


def max_force(system: System) -> sp.Expr:
    """Find max ||F|| analytically."""
    if not system.force_expr or system.potential_expr == 0:
        return sp.Integer(0)

    param_map = dict(zip(system.parameter_symbols, system.params, strict=False))
    force = sp.Matrix(system.force_expr).subs(param_map)
    max_square = sp.simplify(force.dot(force))

    try:
        for sym in system.coordinate_symbols:
            max_square = sp.calculus.util.maximum(max_square, sym)
        return sp.sqrt(max_square)
    except NotImplementedError:
        gradient = sp.Matrix([max_square]).jacobian(system.coordinate_symbols)
        critical_points = sp.solve(gradient, system.coordinate_symbols, dict=True)

        critical_values = [max_square.subs(pt) for pt in critical_points]
        critical_values = [v for v in critical_values if v.is_real]

        return sp.sqrt(sp.Max(*critical_values)) if critical_values else sp.Integer(0)


@dataclass(frozen=True, kw_only=True)
class System:
    """Parameters representing a physical system."""

    gamma: float
    temperature: float
    m: float
    potential: tuple[int, sp.Expr]
    params: tuple[float, ...]
    units: UnitSystem = field(default_factory=UnitSystem)

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

    def as_canonical(self) -> CanonicalSystem:
        """Return the canonical form of the system."""
        return CanonicalSystem(
            gamma=self.gamma,
            temperature=self.temperature,
            m=self.m,
            potential=self.potential,
            params=self.params,
            units=self.units.as_canonical(),
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
        """Kbt."""
        return self.units.boltzmann * self.temperature

    def with_units(self, units: UnitSystem) -> System:
        """Return the system converted to the given units."""
        length_factor = self.units.length_factor / units.length_factor
        energy_factor = self.units.energy_factor / units.energy_factor

        scaled_expr = (
            self.potential_expr.subs(
                {sym: sym * length_factor for sym in self.coordinate_symbols}
            )
            / energy_factor
        )
        return System(
            m=self.units.mass_into(self.m, units),
            gamma=self.units.frequency_into(self.gamma, units),
            units=units,
            temperature=self.temperature,
            potential=(self.n_dim, scaled_expr),
            params=self.params,
        )

    @property
    def normalized_units(self) -> UnitSystem:
        """Units scaled purely via intrinsic physical scales of V(x), T, m, and gamma."""
        # Express rates in uniform units (s^-1)
        rate_force = max_force(self) / np.sqrt(self.m * self.kbt)
        rate_force = 0.0 if rate_force == sp.oo else float(rate_force)
        rate_gamma = self.gamma

        # Select dominant physical rate
        # If gamma and force are both small, then dont scale the units
        # Length is velocity * time, and time is 1 / nu_0
        v_th = np.sqrt(self.kbt / self.m)
        nu_0 = max(rate_force, rate_gamma, v_th / 1.0)
        characteristic_length = v_th / nu_0

        return UnitSystem(
            boltzmann=1 / self.temperature,
            atomic_mass=self.units.atomic_mass / self.m,
            angstrom=self.units.angstrom / characteristic_length,
        )

    def with_normalized_units(self) -> System:
        """Return the parameters of the simulation in normalized units."""
        return self.with_units(self.normalized_units)

    def with_si_units(self) -> System:
        """Return the si parameters of the system."""
        return self.with_units(UnitSystem())


@jax.tree_util.register_dataclass
@final
@dataclass(frozen=True, kw_only=True)
class CanonicalSystem(System):
    """Parameters representing a physical system."""

    potential: tuple[int, sp.Expr] = field(metadata={"static": True})
    units: CanonicalUnitSystem = field(default_factory=CanonicalUnitSystem)

    def with_units(self, units: UnitSystem) -> CanonicalSystem:
        """Return the system converted to the given units."""
        return super().with_units(units).as_canonical()

    def with_normalized_units(self) -> CanonicalSystem:
        """Return the parameters of the simulation in normalized units."""
        return self.with_units(self.normalized_units).as_canonical()


class HarmonicSystem(System):
    """Parameters representing a simple harmonic oscillator system."""

    _omega: float

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        omega: float,
        units: UnitSystem | None = None,
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
            units=units or UnitSystem(),
        )

    @property
    def omega(self) -> float:
        """The angular frequency of the system."""
        return self.params[0]

    @override
    def with_units(self, units: UnitSystem) -> HarmonicSystem:

        return HarmonicSystem(
            gamma=self.units.frequency_into(self.gamma, units),
            temperature=self.temperature,
            m=self.units.mass_into(self.m, units),
            omega=self.units.frequency_into(self.omega, units),
            n_dim=self.n_dim,
            units=units,
        )


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
        units: UnitSystem | None = None,
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
            units=units or UnitSystem(),
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
    def with_units(self, units: UnitSystem) -> PeriodicSystem1D:
        """Return the normalized parameters of the simulation."""
        return PeriodicSystem1D(
            gamma=self.units.frequency_into(self.gamma, units),
            temperature=self.temperature,
            m=self.units.mass_into(self.m, units),
            delta_x=self.units.length_into(self.delta_x, units),
            barrier_energy=self.units.energy_into(self.barrier_energy, units),
            n_dim=self.n_dim,
            units=units,
        )


def _get_potential_expr_fcc() -> sp.Expr:
    """Return the potential energy expression for a 2D FCC lattice."""
    x0, x1 = sp.symbols("x0 x1")
    s0, s1 = sp.symbols("s0 s1")
    c = sp.Integer(2) * sp.pi / (sp.sqrt(3.0) * s0)

    kx0 = c * (sp.Integer(-1) / sp.sqrt(3.0))
    kx1 = c * sp.Integer(1)

    ky0 = c * (sp.Integer(2) / sp.sqrt(3.0))
    ky1 = 0.0

    arg1 = x0 * kx0 + x1 * kx1
    arg2 = x0 * ky0 + x1 * ky1
    arg3 = arg1 + arg2

    cos_sum = sp.cos(arg1) + sp.cos(arg2) + sp.cos(arg3)

    return sp.Integer(2) * s1 * cos_sum + sp.Integer(3) * s1


class PeriodicSystemFCC(System):
    """Parameters for the face-centered cubic periodic system."""

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        delta_x: float,
        barrier_energy: float,
        units: UnitSystem | None = None,
    ) -> None:
        potential = _get_potential_expr_fcc()

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(2, potential),
            params=(delta_x, barrier_energy),
            units=units or UnitSystem(),
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
    def with_units(self, units: UnitSystem) -> PeriodicSystemFCC:
        """Return the parameters of the system in the specified units."""
        return PeriodicSystemFCC(
            gamma=self.units.frequency_into(self.gamma, units),
            temperature=self.temperature,
            m=self.units.mass_into(self.m, units),
            delta_x=self.units.length_into(self.delta_x, units),
            barrier_energy=self.units.energy_into(self.barrier_energy, units),
            units=units,
        )


class DerivativeSafeMod(sp.Mod):
    """A subclass of sympy.Mod that is safe for differentiation.

    This class overrides the _eval_derivative method to ensure that the derivative
    of a Mod expression is computed correctly, even when the modulus depends on the
    variable of differentiation.
    """

    def _eval_derivative(self, s: sp.Symbol) -> sp.Expr:
        p, q = self.args
        if not q.has(s):
            return sp.diff(p, s)
        return sp.Derivative(self, s)


class KramersSystem1D(System):
    """Parameters representing a periodic double harmonic system."""

    def __init__(
        self,
        *,
        m: float,
        params: KramersParameters,
        n_dim: int = 1,
        units: UnitSystem | None = None,
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
        x_0 = sp.sqrt((2 * omegas_ss * s2) / (s0**2 * s1**2))
        x_meet = (s1**2 / omegas_ss) * x_0

        periodic_x = DerivativeSafeMod(x0 + x_meet, 2 * x_0) - x_meet

        potential = sp.Piecewise(
            (0.5 * s0**2 * periodic_x**2, periodic_x <= x_meet),
            (
                s2 - 0.5 * s1**2 * (periodic_x - x_0) ** 2,
                periodic_x >= x_meet,
            ),
            (0, True),
        )

        super().__init__(
            gamma=params.gamma,
            # Note: this currently assumes Boltzmann = 1
            temperature=params.kbt,
            m=m,
            potential=(n_dim, potential),
            params=(
                params.omega_well,
                params.omega_barrier,
                params.barrier_energy,
            ),
            units=units or UnitSystem(),
        )

    @property
    def delta_x(self) -> float:
        """The delta x of the system."""
        return self.kramers_params.delta_x

    @property
    def kramers_params(self) -> KramersParameters:
        return KramersParameters(
            omega_well=self.params[0],
            omega_barrier=self.params[1],
            barrier_energy=self.params[2],
            kbt=self.temperature,
            gamma=self.gamma,
        )

    @property
    def omega_well(self) -> float:
        """The harmonic frequency at the bottom of the well."""
        return self.kramers_params.omega_well

    @property
    def omega_barrier(self) -> float:
        """The harmonic frequency at the top of the barrier."""
        return self.kramers_params.omega_barrier

    @property
    def barrier_energy(self) -> float:
        """The barrier energy of the system."""
        return self.kramers_params.barrier_energy

    @override
    def with_units(self, units: UnitSystem) -> KramersSystem1D:
        """Return the parameters of the system in the specified units."""
        return KramersSystem1D(
            m=self.units.mass_into(self.m, units),
            params=KramersParameters(
                omega_well=self.units.frequency_into(self.omega_well, units),
                omega_barrier=self.units.frequency_into(self.omega_barrier, units),
                barrier_energy=self.units.energy_into(self.barrier_energy, units),
                kbt=self.units.energy_into(self.kbt, units),
                gamma=self.units.frequency_into(self.gamma, units),
            ),
            n_dim=self.n_dim,
            units=units,
        )


def get_diffusion_time(system: System, characteristic_length: float) -> float:
    """Return the average time for a particle to traverse a characteristic length."""
    return np.sqrt(system.m * characteristic_length**2 / system.kbt)


def get_characteristic_periodic_mass(system: PeriodicSystem1D) -> float:
    """Return the characteristic mass for a 1D periodic system."""
    return system.kbt * system.delta_x**2 / system.gamma**2
