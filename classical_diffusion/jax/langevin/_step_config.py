import dataclasses
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

import diffrax as dfx
import jax
import jax.numpy as jnp

type LevyArea = type[
    dfx.BrownianIncrement | dfx.SpaceTimeLevyArea | dfx.SpaceTimeTimeLevyArea
]


class StepConfig(ABC):
    """How a stochastic solver chooses its time steps, and samples its noise."""

    @property
    @abstractmethod
    def adjoint(self) -> dfx.AbstractAdjoint:
        """The adjoint of the solve."""

    @abstractmethod
    def brownian_path(
        self,
        times: jnp.ndarray,
        *,
        shape: tuple[int, ...],
        key: jax.Array,
        levy_area: LevyArea,
    ) -> dfx.AbstractBrownianPath:
        """Get the Brownian path driving a solve, up to the last of the output times."""

    @abstractmethod
    def controller(self, times: jnp.ndarray) -> dfx.AbstractStepSizeController:
        """Get the step size controller, for the output times."""

    @abstractmethod
    def initial_step(self, times: jnp.ndarray) -> Any:  # ruff: ignore[any-type]
        """Get the initial step size, for the output times."""

    @override
    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and dataclasses.astuple(
            self  # ty: ignore[invalid-argument-type]
        ) == dataclasses.astuple(other)  # ty: ignore[invalid-argument-type]

    @override
    def __hash__(self) -> int:
        # Stable between runs, so it can be used in cache paths
        return zlib.crc32(repr(self).encode("utf-8"))


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class Adaptive(StepConfig):
    """Adaptive time steps, chosen by a PID controller.

    This is the default, and works for any system. The noise is sampled from
    a virtual Brownian tree, so that it is consistent when a step is rejected.
    """

    rtol: float = 1e-2
    atol: float = 1e-3

    @property
    @override
    def adjoint(self) -> dfx.AbstractAdjoint:
        return dfx.RecursiveCheckpointAdjoint()

    @override
    def brownian_path(
        self,
        times: jnp.ndarray,
        *,
        shape: tuple[int, ...],
        key: jax.Array,
        levy_area: LevyArea,
    ) -> dfx.AbstractBrownianPath:
        return dfx.VirtualBrownianTree(
            t0=0, t1=times[-1], tol=1e-4, shape=shape, key=key, levy_area=levy_area
        )

    @override
    def controller(self, times: jnp.ndarray) -> dfx.AbstractStepSizeController:
        return dfx.ClipStepSizeController(
            dfx.PIDController(rtol=self.rtol, atol=self.atol), step_ts=times
        )

    @override
    def initial_step(self, times: jnp.ndarray) -> Any:
        return times[1] - times[0]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class FixedStep(StepConfig):
    """Fixed time steps of step_size, in the natural units of the simulation.

    The natural unit of time is 1 / nu_0, where nu_0 is the larger of the
    friction and the force rate F_max / sqrt(m k_B T). The step is reduced so
    that it divides the spacing of the output times. Since no step is rejected,
    the noise is sampled independently for each step, which is much faster
    than the Brownian tree needed for adaptive steps. The step must be small
    enough to resolve the fastest motion of the system.
    """

    step_size: float

    @property
    @override
    def adjoint(self) -> dfx.AbstractAdjoint:
        # UnsafeBrownianPath does not support backpropagation
        return dfx.ForwardMode()

    @override
    def brownian_path(
        self,
        times: jnp.ndarray,
        *,
        shape: tuple[int, ...],
        key: jax.Array,
        levy_area: LevyArea,
    ) -> dfx.AbstractBrownianPath:
        # Steps are never rejected, so the increments need not be consistent
        return dfx.UnsafeBrownianPath(shape=shape, key=key, levy_area=levy_area)

    @override
    def controller(self, times: jnp.ndarray) -> dfx.AbstractStepSizeController:
        return dfx.ConstantStepSize()

    @override
    def initial_step(self, times: jnp.ndarray) -> Any:
        spacing = times[1] - times[0]
        return spacing / jnp.ceil(spacing / self.step_size)


DEFAULT_STEP_CONFIG = Adaptive()
"""The default step configuration, which works for any system."""
