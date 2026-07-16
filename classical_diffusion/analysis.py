from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sympy as sp

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.axes import Axes as MPLAxesBase
    from matplotlib.collections import QuadMesh
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


from dataclasses import astuple, dataclass

import matplotlib.font_manager as fm

from .solve import FlatParams, SHOParams, SimulationParams, SimulationResult


@dataclass(frozen=True, kw_only=True)
class CamColor:
    """A class to hold CAM color palettes."""

    light: str
    warm: str
    base: str
    dark: str


CAM_BLUE = CamColor(
    light="#D1F9F1",
    warm="#00BDB6",
    base="#8EE8D8",
    dark="#133844",
)
CAM_CHERRY = CamColor(
    light="#F2CAD8",
    warm="#E18AAC",
    base="#CD3572",
    dark="#911449",
)
CAM_CREST = CamColor(
    light="#FFE2C8",
    warm="#FFC392",
    base="#FD8153",
    dark="#DD3025",
)
CAM_PURPLE = CamColor(
    light="#F2ECF8",
    warm="#D1B7EB",
    base="#A368DF",
    dark="#681FB1",
)
CAM_INDIGO = CamColor(
    light="#EBEDFB",
    warm="#B0B9F1",
    base="#5366E0",
    dark="#29347A",
)
CAM_GREEN = CamColor(
    light="#DFF2EA",
    warm="#AFDFCB",
    base="#4DB78C",
    dark="#13553A",
)


CAM_SLATE_1 = "#ECEEF1"
CAM_SLATE_2 = "#B5BDC8"
CAM_SLATE_3 = "#546072"
CAM_SLATE_4 = "#232830"


def setup_rc_params(*, use_tex: bool = False) -> None:
    """Set up matplotlib rcParams for consistent figure styling."""
    if use_tex:
        fe = fm.FontEntry(
            fname="/workspaces/thesis_calculations/fonts/OpenSans-Regular.ttf",
            name="Open Sans",
        )
        fm.fontManager.ttflist.insert(0, fe)
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": ["Open Sans"],
                "text.latex.preamble": r"\usepackage{fourier}"
                "\n"
                r"\usepackage{amsmath}",  # cspell: disable-line
            },
        )


def get_fig_size() -> tuple[float, float]:
    """Get default figure size in inches based on document text width."""
    total_textwidth_pt = 437.5
    pt_to_inch = 1 / 72.27

    # We want half width
    plot_width_in = (total_textwidth_pt / 2) * pt_to_inch

    # Height using Golden Ratio (Height = Width * 0.618)
    plot_height_in = plot_width_in * 0.85
    return plot_width_in, plot_height_in


def get_fancy_figure(
    *,
    fig_size: tuple[float, float] | None = None,
    use_tex: bool = False,
) -> tuple[Figure, Axes]:
    """Create a figure and axis with fancy styling."""
    setup_rc_params(use_tex=use_tex)
    fig, ax = plt.subplots(
        figsize=fig_size or get_fig_size(),
        layout="constrained",
    )
    ax.set_facecolor(CAM_SLATE_1)
    fig.set_facecolor((0, 0, 0, 0))

    ax.tick_params(
        axis="both", direction="in", top=True, right=True, labelsize=8, which="both"
    )

    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    return fig, ax


def get_figure(ax: MPLAxesBase | None = None) -> tuple[Figure, Axes]:
    """Get the figure of the given axis.

    If no figure exists, a new figure is created
    """  # noqa: DOC501
    if plt is None:
        msg = "Matplotlib is not installed. Please install it with the 'plot' extra."
        raise ImportError(msg)  # noqa: RUF100

    if ax is None:
        return cast("tuple[Figure, Axes]", plt.subplots())  # type: ignore plt.subplots Unknown type

    fig = cast("Figure|None", ax.get_figure())
    if fig is None:
        fig = cast("Figure", plt.figure())  # type: ignore plt.figure Unknown type
        ax.set_figure(fig)
    return fig, ax


Measure = Literal["real", "imag", "abs", "angle"]


def _measure_data[DT: np.dtype[np.number]](
    data: np.ndarray[Any, DT],
    measure: Measure,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    match measure:
        case "real":
            return np.real(data)  # type: ignore[no-any-return]
        case "imag":
            return np.imag(data)  # type: ignore[no-any-return]
        case "abs":
            return np.abs(data)  # type: ignore[no-any-return]
        case "angle":
            return np.unwrap(np.angle(data))  # type: ignore[no-any-return]


def get_measured_data[DT: np.dtype[np.number[Any]]](
    data: np.ndarray[Any, DT],
    measure: Measure,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Transform data with the given measure.

    Raises
    ------
        ValueError: If the data contains NaN values.
    """  # noqa: DOC501
    measured = _measure_data(data, measure)
    if np.any(np.isnan(measured)):
        msg = "The data contains NaN values."
        raise ValueError(msg)
    return measured


def _calculate_total_offsset_multiplications_complex(
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    # scipy.signal.correlate handles complex numbers and conjugation automatically
    # Note: correlate(a, b) conjugates the first argument by default
    return scipy.signal.correlate(lhs, rhs, mode="full")[: lhs.size][::-1]


def _time_average[DT: np.floating](
    time_sum: np.ndarray[Any, np.dtype[DT]],
) -> np.ndarray[Any, np.dtype[DT]]:
    """Apply the time-averaging denominator."""
    size = time_sum.shape[-1]
    return time_sum / np.arange(1, size + 1)[::-1]


def get_isf(
    positions: np.ndarray[Any, np.dtype[np.floating]],
    delta_k: tuple[float, ...],
    *,
    pairwise: bool = True,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """Get the restored displacement of a wavepacket."""
    if not pairwise:
        phase = np.einsum(
            "i,...ij->...j",
            delta_k,
            positions - positions[..., 0].reshape((*positions.shape[:-1], 1)),
        )
        return np.exp(1j * phase)

    scatter = np.exp(-1j * np.einsum("i,...ij->...j", delta_k, positions))

    # convolution_j = \sum_i^N-j e^(ik.x_i+j) e^(-ik.x_i)
    convolution = np.apply_along_axis(
        lambda m: _calculate_total_offsset_multiplications_complex(m, m),
        axis=-1,
        arr=scatter,
    )
    return _time_average(convolution)


@dataclass(frozen=True, kw_only=True)
class IsfConfig:
    """Settings controlling how the ISF is computed from trajectory data."""

    delta_k: tuple[float, ...]
    measure: Measure = "abs"
    pairwise: bool = True


def plot_isf(
    result: SimulationResult,
    config: IsfConfig,
    *,
    ax: Axes | None = None,
    color: str,
    time_scale: float | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the ensemble-averaged ISF over time, with a shaded ±1 SEM band.

    `time_scale` overrides the result's own characteristic time for
    x-axis normalization — pass the same value to multiple calls to
    compare curves on a shared timescale.
    """
    fig, ax = get_figure(ax)

    isf = get_isf(result.xs, delta_k=config.delta_k, pairwise=config.pairwise)

    n_trajectories = isf.shape[0]
    avg_isf = np.mean(isf, axis=0)
    sem_isf = np.std(isf, axis=0) / np.sqrt(n_trajectories)

    avg_data = get_measured_data(avg_isf, config.measure)
    sem_data = get_measured_data(sem_isf, config.measure)

    scale = (
        time_scale
        if time_scale is not None
        else result.params.characteristic_values.time
    )
    scaled_times = result.times / scale
    (line,) = ax.plot(scaled_times, avg_data)
    line.set_label("ISF")
    line.set_color(color)

    ax.fill_between(
        scaled_times,
        avg_data - sem_data,
        avg_data + sem_data,
        alpha=0.3,
        color=color,
        label="SEM",
    )

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time /characteristic time")
    ax.set_ylabel("ISF")

    return fig, ax, line


def plot_exact_isf_sho(
    result: SimulationResult, *, ax: Axes | None = None, color: str
) -> tuple[Figure, Axes, Line2D]:
    """Plot the state occupations of a quantum simulation result."""
    fig, ax = get_figure(ax)

    isf_exact, times = get_exact_isf_sho(result.params)
    (line,) = ax.plot(times, isf_exact)
    line.set_label("ISF")
    line.set_color(color)

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time / characteristic time")
    ax.set_ylabel("ISF")
    ax.legend()

    return fig, ax, line


def plot_x_evolution(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
    idx: int = 0,
    n_trajectories: int = 1,
) -> tuple[Figure, Axes, list[Line2D]]:
    """Plot x against t for the first n_trajectories trajectories.

    Raises
    ------
    ValueError
        If `n_trajectories` exceeds the number of trajectories available in `result`.
    """
    fig, ax = get_figure(ax)

    if n_trajectories > result.xs.shape[0]:
        msg = f"n_trajectories={n_trajectories} exceeds available trajectories ({result.xs.shape[0]})"
        raise ValueError(msg)

    times = result.times - result.times[0]
    times /= result.params.characteristic_values.time

    lines = []
    for traj in range(n_trajectories):
        (line,) = ax.plot(times, result.xs[traj, idx])
        lines.append(line)

    ax.set_xlabel("$t / characteristic time$")
    ax.set_ylabel("$x$")

    return fig, ax, lines


def plot_p_evolution(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
    idx: int = 0,
    n_trajectories: int = 1,
) -> tuple[Figure, Axes, list[Line2D]]:
    """Plot p against t for the first n_trajectories trajectories.

    Raises
    ------
    ValueError
        If `n_trajectories` exceeds the number of trajectories available in `result`.
    """
    fig, ax = get_figure(ax)

    if n_trajectories > result.ps.shape[0]:
        msg = f"n_trajectories={n_trajectories} exceeds available trajectories ({result.ps.shape[0]})"
        raise ValueError(msg)

    times = result.times - result.times[0]
    times /= result.params.characteristic_values.time

    lines = []
    for traj in range(n_trajectories):
        (line,) = ax.plot(times, result.ps[traj, idx])
        lines.append(line)

    ax.set_xlabel("$t / characteristic time$")
    ax.set_ylabel("$p$")

    return fig, ax, lines


def _get_sampled_kinetic_energies[T: SimulationResult](
    result: T,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    return (np.sum(result.ps**2, axis=1)) / (
        2 * result.params.physical.m * result.params.physical.kbt
    )


def _get_all_kinetic_energies[T: SimulationResult](
    result: T | list[T],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    result: list[T] = (
        cast("list[T]", [result]) if isinstance(result, SimulationResult) else result
    )
    return np.concatenate([_get_sampled_kinetic_energies(r) for r in result]).ravel()


def plot_kinetic_probability[T: SimulationResult](  # noqa: PLR0914
    result: T | list[T],
    *,
    ax: Axes | None = None,
    bins: int = 100,
    max_energy: float = 4.0,
) -> tuple[Figure, Axes, tuple[Line2D, Line2D, Line2D, BarContainer]]:
    """Plot the kinetic probabilities for the sample."""
    fig, ax = get_figure(ax)

    kinetic_energy = _get_all_kinetic_energies(result)

    energy_range = (np.min(kinetic_energy), max_energy + np.min(kinetic_energy))
    bin_counts, bin_edges, bars = ax.hist(
        kinetic_energy,
        bins=bins,
        density=True,
        alpha=0.6,
        color="C0",
        label="Simulation Data",
        range=energy_range,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def classical_pdf(
        energies: np.ndarray[tuple[int], np.dtype[np.float64]], mu: float
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return (1.0 / np.sqrt(2 * np.pi * energies * mu)) * np.exp(-energies / (2 * mu))

    energies = np.linspace(0.001, bin_edges[-1], 500)

    (line0,) = ax.plot(energies, classical_pdf(energies, mu=0.5))
    line0.set_color("C3")
    line0.set_linestyle("-")
    line0.set_linewidth(2)
    line0.set_label("Theoretical PDF (Mean = 0.5)")

    optimal_params, _ = scipy.optimize.curve_fit(
        classical_pdf,
        bin_centers,
        bin_counts,
        p0=[0.5],
        bounds=(0, 2),
        sigma=np.asarray(bin_counts) + 1e-5,
    )
    fitted_mean = optimal_params[0]
    (line1,) = ax.plot(energies, classical_pdf(energies, mu=fitted_mean))
    line1.set_color("C4")
    line1.set_linestyle("-.")
    line1.set_linewidth(2)
    line1.set_label(f"Theoretical PDF (Fitted Mean = {fitted_mean:.3f})")

    def general_pdf(
        energies: np.ndarray[tuple[int], np.dtype[np.float64]], mu: float, a: float
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return (
            a
            * (1.0 / np.sqrt(2 * np.pi * energies * mu))
            * np.exp(-energies / (2 * mu))
        )

    optimal_params, _ = scipy.optimize.curve_fit(
        general_pdf,
        bin_centers,
        bin_counts,
        p0=[0.5, 1.0],
        bounds=(0, np.inf),
        sigma=np.asarray(bin_counts) + 1e-6,
    )
    (line2,) = ax.plot(energies, general_pdf(energies, *optimal_params))
    line2.set_linestyle("-.")
    line2.set_linewidth(2)
    line2.set_label(f"Scaled PDF (Fitted Mean = {optimal_params[0]:.3f})")

    ax.set_xlim(0, max_energy)

    exponent = np.floor(np.log10(classical_pdf(np.array([max_energy]), mu=0.5)[0]))
    ax.set_ylim(10 ** (exponent - 1), None)
    ax.set_xlabel(r"Kinetic Energy / $k_B T$")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.set_yscale("log")

    return fig, ax, (line0, line1, line2, cast("BarContainer", bars))


def get_exact_isf_sho(params: SHOParams) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact ISF for simulation."""
    times = np.arange(params.time_span.t0, params.time_span.t1, params.time_span.dt)
    gamma, temp, m = astuple(params.physical)
    f = np.sqrt(params.omega**2 - gamma**2 / 4)
    delta_k = params.delta_k[0]  # unpack from tuple

    return np.exp(
        -(delta_k**2)
        * ((1.0 * temp) / (m * params.omega**2))
        * (
            1
            - np.exp(-gamma * times / 2)
            * (np.cos(f * times) + (gamma / (2 * f)) * np.sin(f * times))
        )
    ), times / params.characteristic_values.time


def split_result(result: SimulationResult) -> tuple[SimulationResult, SimulationResult]:
    """Split a simulation result in half along the time axis, each restarting at t=0."""
    xs1, xs2 = np.split(result.xs, 2, axis=-1)
    ps1, ps2 = np.split(result.ps, 2, axis=-1)
    times1, times2 = np.split(result.times, 2)

    times1 -= times1[0]
    times2 -= times2[0]

    first = SimulationResult(times=times1, xs=xs1, ps=ps1, params=result.params)
    second = SimulationResult(times=times2, xs=xs2, ps=ps2, params=result.params)
    return first, second


def x_exact_pdf(result: SimulationResult, *, n_grid: int = 10_000) -> tuple:
    """Return x boltzman pdf for given potential."""
    potential = sp.lambdify(
        result.params.symbolic_coordinates, result.params.potential, "numpy"
    )
    x_grid = np.linspace(result.xs.min(), result.xs.max(), n_grid)
    v_grid = np.broadcast_to(potential(x_grid), x_grid.shape)

    kbt = result.params.physical.kbt

    v_shifted = v_grid - v_grid.min()
    unnormalised = np.exp(-v_shifted / kbt)

    z = np.trapezoid(unnormalised, x_grid)

    return x_grid, unnormalised / z


def sample_result(result: SimulationResult) -> SimulationResult:
    """Subsample all trajectories at the parameters' stride, along the saved-time axis."""
    stride = result.params.stride
    return SimulationResult(
        times=result.times[::stride],
        xs=result.xs[:, :, ::stride],
        ps=result.ps[:, :, ::stride],
        params=result.params,
    )


def fold_result(result: SimulationResult) -> SimulationResult:
    """Fold x into first BZ zone."""
    return SimulationResult(
        times=result.times,
        xs=result.xs % result.params.lattice_spacing,
        ps=result.ps,
        params=result.params,
    )


def plot_x_histogram(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
    bins: int = 100,
) -> tuple[Figure, Axes, tuple[Line2D, BarContainer]]:
    """Plot a fancy histogram of periodically sampled position or momentum.

    Subsamples the trajectory every `sample_every` steps (to reduce
    autocorrelation between adjacent time points) before histogramming.
    """
    fig, ax = get_figure(ax)

    _bin_counts, _bin_edges, bars = ax.hist(
        result.xs.reshape(-1),
        bins=bins,
        density=True,
        alpha=1.0,
        color=CAM_BLUE.warm,
    )

    x_grid, x_pdf = x_exact_pdf(result)
    ax.plot(x_grid, x_pdf, color=CAM_BLUE.dark, lw=1.5)

    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=False, fontsize=9, labelcolor=CAM_SLATE_4)

    return fig, ax, cast("BarContainer", bars)


def p_exact_pdf(result: SimulationResult, *, n_grid: int = 10_000) -> tuple:
    """Return p boltzman pdf."""
    p_grid = np.linspace(result.ps.min(), result.ps.max(), n_grid)
    m, kbt = result.params.physical.m, result.params.physical.kbt
    pdf_theory = np.sqrt(1 / (2 * np.pi * m * kbt)) * np.exp(
        -(p_grid**2) / (2 * m * kbt)
    )

    return p_grid, pdf_theory


def plot_p_histogram(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
    bins: int = 100,
) -> tuple[Figure, Axes, tuple[Line2D, BarContainer]]:
    """Plot a fancy histogram of periodically sampled position or momentum.

    Subsamples the trajectory every `sample_every` steps (to reduce
    autocorrelation between adjacent time points) before histogramming.
    """
    fig, ax = get_figure(ax)

    _bin_counts, _bin_edges, bars = ax.hist(
        result.ps.reshape(-1),
        bins=bins,
        density=True,
        alpha=1.0,
        color=CAM_BLUE.warm,
    )

    p_grid, p_pdf = p_exact_pdf(result=result)
    ax.plot(p_grid, p_pdf, color=CAM_BLUE.dark, lw=1.5)

    ax.set_xlabel("p")
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=False, fontsize=9, labelcolor=CAM_SLATE_4)

    return fig, ax, cast("BarContainer", bars)


def plot_phase_space_density(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
    bins: int = 100,
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot 2D density map of (x, p) phase space."""
    fig, ax = get_figure(ax)

    _counts, _xedges, _yedges, mesh = ax.hist2d(
        result.xs.reshape(-1),
        result.ps.reshape(-1),
        bins=bins,
        density=True,
        cmap="viridis",
    )

    fig.colorbar(mesh, ax=ax, label="Probability Density")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title("Phase Space Density")

    return fig, ax, mesh


def get_elastic_p(
    result: SimulationResult,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Return the elastic (ballistic straight-line) momentum estimate per trajectory."""
    x0 = result.params.initial_conditions.x0[:, :, None]
    safe_times = np.where(result.times == 0, np.nan, result.times)
    v_elastic = (result.xs - x0) / safe_times
    return v_elastic * result.params.physical.m


def plot_elastic_p(
    result: SimulationResult, *, n_trajectories: int = 5, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """Plot convergence of elastic momenta over all trajectories.

    Raises
    ------
    ValueError
        If `n_trajectories` exceeds the number of trajectories available in `result`.
    """
    if n_trajectories > result.ps.shape[0]:
        msg = f"n_trajectories={n_trajectories} exceeds available trajectories ({result.ps.shape[0]})"
        raise ValueError(msg)

    fig, ax = get_figure(ax)

    result = sample_result(result)
    ts = result.times
    ps = get_elastic_p(result)
    for traj in range(n_trajectories):
        ax.plot(ts, ps[traj, 0], label=f"trajectory {traj}")

    ax.set_xlabel("time")
    ax.set_ylabel("p_elastic")

    return fig, ax


def get_exact_isf_flat(
    params: FlatParams,
    *,
    delta_k: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact ballistic ISF for a 1D flat (potential-free) surface.

    ISF(t) = exp(-delta_k^2 * kbT / (2m) * t^2), the Gaussian decay from
    free-streaming thermal velocities with no confining potential.
    """
    times = np.arange(params.time_span.t0, params.time_span.t1, params.time_span.dt)
    kbt, m = params.physical.kbt, params.physical.m
    k_squared = sum(k_i**2 for k_i in delta_k)

    isf_exact = np.exp(-(k_squared) * kbt / (2 * m) * times**2)

    return isf_exact, times / params.characteristic_values.time


def plot_exact_isf_flat(
    params: FlatParams,
    *,
    delta_k: tuple[float, ...],
    ax: Axes | None = None,
    color: str,
    time_scale: float | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the exact ISF for a 1D flat (potential-free) surface."""
    fig, ax = get_figure(ax)

    isf_exact, times = get_exact_isf_flat(params, delta_k=delta_k)

    if time_scale is not None:
        times = times * params.characteristic_values.time / time_scale

    (line,) = ax.plot(times, isf_exact)
    line.set_label("Exact Flat ISF")
    line.set_color(color)

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time / characteristic time")
    ax.set_ylabel("ISF")
    ax.legend()

    return fig, ax, line


_EXPECTED_NDIM = 2


def plot_2d_trajectory(
    result: SimulationResult, *, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    """Plot x against y for 2d trajectory.

    Raises
    ------
    ValueError
        If n_dimensions is not 2.
    """
    fig, ax = get_figure(ax)

    if result.xs.shape[1] != _EXPECTED_NDIM:
        msg = "incorrect number of system dimensions, must be 2)"
        raise ValueError(msg)

    (line,) = ax.plot(result.xs[0, 0], result.xs[0, 1])

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    return fig, ax, line


DEFAULT_EXTENT_MULTIPLIER = 5.0


def plot_potential(
    params: SimulationParams,
    *,
    ax: Axes | None = None,
    n_grid: int = 200,
    extent: tuple[float, float] | None = None,
) -> tuple[Figure, Axes, Line2D | QuadMesh]:
    """Plot the potential energy surface for a 1D or 2D system.

    For 1D systems, plots V(x) as a line. For 2D systems, plots V(x, y)
    as a filled heatmap.

    Raises
    ------
    NotImplementedError
        If the system has more than 2 spatial dimensions, since a
        potential surface in >2D isn't directly visualizable without
        slicing or projecting onto a lower-dimensional subspace.
    """
    fig, ax = get_figure(ax)

    n_dims = params.n_dimensions
    potential_func = sp.lambdify(params.symbolic_coordinates, params.potential, "numpy")

    lo, hi = (
        extent
        if extent is not None
        else (
            -DEFAULT_EXTENT_MULTIPLIER * params.characteristic_values.length,
            DEFAULT_EXTENT_MULTIPLIER * params.characteristic_values.length,
        )
    )

    if n_dims == 1:
        x_grid = np.linspace(lo, hi, n_grid)
        v_grid = np.broadcast_to(potential_func(x_grid), x_grid.shape)

        (line,) = ax.plot(x_grid, v_grid, color=CAM_BLUE.dark, lw=1.5)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$V(x)$")
        return fig, ax, line

    if n_dims == 2:  # noqa: PLR2004
        x_grid = np.linspace(lo, hi, n_grid)
        y_grid = np.linspace(lo, hi, n_grid)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        v_mesh = np.broadcast_to(potential_func(x_mesh, y_mesh), x_mesh.shape)

        mesh = ax.pcolormesh(x_mesh, y_mesh, v_mesh, cmap="viridis", shading="auto")
        fig.colorbar(mesh, ax=ax, label="$V(x, y)$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title("Potential Energy Surface")
        return fig, ax, mesh

    msg = (
        f"plot_potential only supports 1D and 2D systems; got "
        f"n_dimensions={n_dims}. Slice the potential along a subset of "
        f"coordinates before plotting."
    )
    raise NotImplementedError(msg)
