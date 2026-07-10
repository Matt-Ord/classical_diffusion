from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.axes import Axes as MPLAxesBase
    from matplotlib.figure import Figure

from typing import TYPE_CHECKING

import scipy

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.font_manager as fm

from .solve import SimulationResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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

    # 3. Handle Label Sizes
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


def plot_isf(
    result: SimulationResult,
    *,
    delta_k: tuple[float, ...],
    measure: Measure = "abs",
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the state occupations of a quantum simulation result."""
    fig, ax = get_figure(ax)

    isf = get_isf(result.xs, delta_k=delta_k)
    data = get_measured_data(isf, measure)
    (line,) = ax.plot(result.times, data)
    line.set_label("ISF")

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time /au")
    ax.set_ylabel("ISF")
    ax.legend()

    return fig, ax, line


def plot_exact_isf(
    isf: np.ndarray, times: np.ndarray, *, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    """Plot the state occupations of a quantum simulation result."""
    fig, ax = get_figure(ax)

    (line,) = ax.plot(times, isf)
    line.set_label("ISF")

    ax.set_title("Intermediate Scattering Function Over Time")
    ax.set_xlabel("Time /au")
    ax.set_ylabel("ISF")
    ax.legend()

    return fig, ax, line


def plot_x_evolution(
    result: SimulationResult, *, ax: Axes | None = None, idx: int = 0
) -> tuple[Figure, Axes, Line2D]:
    """Plot x against t."""
    fig, ax = get_figure(ax)

    times = result.times - result.times[0]
    (line,) = ax.plot(times, result.xs[idx])
    line.set_label("$x$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")

    return fig, ax, line


def plot_p_evolution(
    result: SimulationResult, *, ax: Axes | None = None, idx: int = 0
) -> tuple[Figure, Axes, Line2D]:
    """Plot p against t."""
    fig, ax = get_figure(ax)

    times = result.times - result.times[0]
    (line,) = ax.plot(times, result.ps[idx])
    line.set_label("$p$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$p$")

    return fig, ax, line


def _get_sampled_kinetic_energies[T: SimulationResult](
    result: T,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    return (np.sum(result.ps**2, axis=0)) / (
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
