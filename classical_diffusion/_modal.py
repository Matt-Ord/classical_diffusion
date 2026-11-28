"""A correlation as a sum of modes, sum_j r_j exp(lambda_j t), from the poles and residues of a tridiagonal (the Lanczos methods of every solver)."""

from dataclasses import dataclass
from math import factorial
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from classical_diffusion._results import ComplexArray, FloatArray

type BoolArray = np.ndarray[Any, np.dtype[np.bool_]]

_CHUNK = 2**22
"""The most entries of exp(lambda_j t) evaluated at once."""


def _phi_2(z: ComplexArray) -> ComplexArray:
    """(e^z - 1 - z) / z^2, by its series where |z| is small, so that a pole of rate zero has its exact limit 1 / 2."""
    small = np.abs(z) < 0.1  # ruff:ignore[magic-value-comparison]
    safe = np.where(small, 1.0, z)
    direct = (np.expm1(safe) - safe) / safe**2
    # Horner's rule for sum_k z^k / (k + 2)!, k < 14
    series = np.zeros_like(z)
    for k in range(13, -1, -1):
        series = series * z + 1 / factorial(k + 2)
    return np.where(small, series, direct)


@dataclass(frozen=True, kw_only=True)
class ModalSum:
    """A correlation C(t) = sum_j r_j exp(lambda_j t) over its kept poles, from the poles and residues of a tridiagonal (`get_poles`)."""

    rates: ComplexArray
    """lambda_j."""
    residues: ComplexArray
    """r_j, with sum_j r_j = C(0)."""
    stable: BoolArray
    """Whether each pole is kept (`of`)."""

    @staticmethod
    def of(
        rates: ComplexArray, residues: ComplexArray, span: float, tolerance: float
    ) -> ModalSum:
        """Keep the poles that decay, or grow by less than the tolerance over the span.

        A generator of a stable dynamics has no growing mode, so a pole with
        Re lambda > 0 is spurious (a Ritz value of a mode the start barely
        holds), or a pole of rate zero (the steady state at dK = 0 or at a
        multiple of |G|) that round off has moved. Kept, a pole of residue r
        errs by at most |r| (exp(Re lambda span) - 1) over the span; the
        rest are dropped (`dropped`), and the caller must count their weight
        in its error.
        """
        # Capped where it would overflow: such a pole is dropped either way
        growth = np.abs(residues) * np.expm1(np.clip(rates.real * span, 0, 700))
        stable = (growth <= tolerance) & (rates.real * span < 700)  # ruff:ignore[magic-value-comparison]
        return ModalSum(rates=rates, residues=residues, stable=stable)

    @property
    def dropped(self) -> float:
        """The weight of the dropped poles."""
        return float(np.abs(self.residues[~self.stable]).sum())

    @property
    def kept(self) -> tuple[ComplexArray, ComplexArray]:
        """The rates and residues of the kept poles."""
        return self.rates[self.stable], self.residues[self.stable]

    def values(self, times: FloatArray) -> ComplexArray:
        """Get C(t) at the times, a chunk of times at a time."""
        rates, residues = self.kept
        chunk = max(1, _CHUNK // max(len(rates), 1))
        return np.concatenate(
            [
                np.exp(np.outer(times[i : i + chunk], rates)) @ residues
                for i in range(0, len(times), chunk)
            ]
            or [np.zeros(0, complex)]
        )

    def transfer(self, s: complex | np.ndarray) -> ComplexArray:
        """Get the Laplace transform of C at each s, sum_j r_j / (s - lambda_j)."""
        rates, residues = self.kept
        s = np.asarray(s, dtype=complex)
        return (residues / (s[..., None] - rates)).sum(axis=-1)

    def double_integral(self, times: FloatArray) -> ComplexArray:
        """Get int_0^t (t - u) C(u) du at the times, exactly: sum_j r_j t^2 phi_2(lambda_j t)."""
        rates, residues = self.kept
        t = np.asarray(times, dtype=float)[:, None]
        return (residues * t**2 * _phi_2(rates * t)).sum(axis=1)
