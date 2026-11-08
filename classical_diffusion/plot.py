from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def get_figure(ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Get the figure of the given axis.

    If no figure exists, a new figure is created
    """  # ruff:ignore[docstring-missing-exception]
    if plt is None:
        msg = "Matplotlib is not installed. Please install it with the 'plot' extra."
        raise ImportError(msg)

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
    """  # ruff:ignore[docstring-missing-exception]
    measured = _measure_data(data, measure)
    if np.any(np.isnan(measured)):
        msg = "The data contains NaN values."
        raise ValueError(msg)
    return measured


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


def _register_cmap(
    name: str, nodes: list[tuple[float, str]]
) -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(name, nodes, N=256)
    mpl.colormaps.register(cmap=cmap, force=True)
    return cmap


# The colormaps below are perceptually uniform: they were designed in the
# CAM02-UCS colour space, where the lightness J' rises (or, for the diverging
# map, rises then falls) linearly and the colour difference between
# adjacent entries is constant. The nodes are close enough together that
# matplotlib's interpolation in sRGB keeps the steps within 5% of uniform.
# None reaches the near-white of CAM_SLATE_1 (J' = 95), the background of
# setup_fancy_figure, so every entry stands out against the axes, and the
# chroma stays high through the middle, so no map passes through grey.

CAM_BLUE_CMAP = _register_cmap(
    "CAM_BLUE_CMAP",
    [
        (0.0, CAM_BLUE.dark),
        (0.0625, "#0E414F"),
        (0.125, "#034A59"),
        (0.1875, "#005462"),
        (0.25, "#005E6B"),
        (0.3125, "#006774"),
        (0.375, "#00717C"),
        (0.4375, "#007B85"),
        (0.5, "#00868D"),
        (0.5625, "#009096"),
        (0.625, "#009B9E"),
        (0.6875, "#00A5A6"),
        (0.75, "#00B0AE"),
        (0.8202, CAM_BLUE.warm),
        (0.875, "#13C7BB"),
        (0.9375, "#21D1C0"),
        (1.0, "#2DDCC3"),
    ],
)
"""Sequential CAM blue, from CAM_BLUE.dark (J' = 23) through CAM_BLUE.warm to a light teal (J' = 80)."""

CAM_CHERRY_CMAP = _register_cmap(
    "CAM_CHERRY_CMAP",
    [
        (0.0, "#4B0A30"),
        (0.0625, "#570B34"),
        (0.125, "#640B38"),
        (0.1875, "#710C3C"),
        (0.25, "#7E0E40"),
        (0.337, CAM_CHERRY.dark),
        (0.375, "#99184E"),
        (0.4375, "#A61E56"),
        (0.5, "#B4245F"),
        (0.5625, "#C12C68"),
        (0.6246, CAM_CHERRY.base),
        (0.6875, "#D7427D"),
        (0.75, "#DD5188"),
        (0.8125, "#E06092"),
        (0.875, "#E16F9B"),
        (0.9375, "#E27DA4"),
        (1.0, CAM_CHERRY.warm),
    ],
)
"""Sequential CAM cherry, from a dark wine (J' = 19) through CAM_CHERRY.dark and CAM_CHERRY.base to CAM_CHERRY.warm (J' = 72)."""

CAM_BLUE_CHERRY_DIVERGING_CMAP = _register_cmap(
    "CAM_BLUE_CHERRY_DIVERGING_CMAP",
    [
        (0.0, CAM_BLUE.dark),
        (0.0312, "#084051"),
        (0.0625, "#00495F"),
        (0.0938, "#00526E"),
        (0.125, "#005B7E"),
        (0.1562, "#00638F"),
        (0.1875, "#006AA1"),
        (0.2188, "#0070B5"),
        (0.25, "#1A76C8"),
        (0.2812, "#397CD4"),
        (0.3125, "#5081DE"),
        (0.3438, "#6586E7"),
        (0.375, "#798BED"),
        (0.4062, "#8B90F1"),
        (0.4375, "#9C95F3"),
        (0.4688, "#AB9AF4"),
        (0.5, "#B9A1F3"),
        (0.5312, "#BE97F0"),
        (0.5625, "#C38DEC"),
        (0.5938, "#C882E6"),
        (0.625, "#CC78DD"),
        (0.6562, "#CF6ED3"),
        (0.6875, "#D263C7"),
        (0.7188, "#D45ABA"),
        (0.75, "#D551AB"),
        (0.7812, "#D6499B"),
        (0.8125, "#D5418A"),
        (0.8565, CAM_CHERRY.base),
        (0.875, "#C62F6C"),
        (0.9062, "#BA2562"),
        (0.9375, "#AD1C59"),
        (0.9688, "#9F1651"),
        (1.0, CAM_CHERRY.dark),
    ],
)
"""Diverging CAM blue to cherry with a light lavender centre (J' = 74).

CAM_BLUE.dark (J' = 23) at the low end, through blue, to a lavender between
CAM_INDIGO.warm and CAM_PURPLE.warm at the centre, then through magenta and
CAM_CHERRY.base to CAM_CHERRY.dark (J' = 36) at the high end. The centre keeps
its colour (chroma 24), so it stands out from CAM_SLATE_1 rather than fading
into it. The two ends differ in lightness, so the blue half spans more
lightness than the cherry half.
"""

CAM_SUNSET_CMAP = _register_cmap(
    "CAM_SUNSET_CMAP",
    [
        (0.0, "#2A327D"),
        (0.0625, "#413591"),
        (0.125, "#5835A1"),
        (0.1875, "#6E33AD"),
        (0.25, "#8130AC"),
        (0.3125, "#922FA7"),
        (0.375, "#A22E9D"),
        (0.4375, "#B13091"),
        (0.5, "#BE3381"),
        (0.5625, "#CB3872"),
        (0.625, "#DB4267"),
        (0.6875, "#EA4E5B"),
        (0.75, "#F95B4F"),
        (0.8125, "#FF6E4A"),
        (0.875, "#FF8550"),
        (0.9375, "#FF9A5C"),
        (1.0, "#FFB06E"),
    ],
)
"""Multi-hue sequential map following the CAM hues indigo, purple, cherry and crest (J' from 26 to 82)."""


def get_fig_size() -> tuple[float, float]:
    """Get default figure size in inches based on document text width."""
    total_textwidth_pt = 437.5
    pt_to_inch = 1 / 72.27

    # We want half width
    plot_width_in = 1.5 * (total_textwidth_pt / 2) * pt_to_inch

    # Height using Golden Ratio (Height = Width * 0.618)
    plot_height_in = plot_width_in * 0.618
    return plot_width_in, plot_height_in


CAM_COLOR_CYCLE = [
    CAM_BLUE.dark,
    CAM_BLUE.warm,
    CAM_CHERRY.dark,
    CAM_CHERRY.warm,
    CAM_CREST.dark,
    CAM_CREST.warm,
]


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
    plt.rcParams.update(
        {
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": CAM_SLATE_4,
            "axes.prop_cycle": cycler(color=CAM_COLOR_CYCLE),
        }
    )


def setup_fancy_figure(fig: Figure, ax: list[Axes]) -> None:
    """Set up a figure and axis with fancy styling."""
    fig.set_facecolor((0, 0, 0, 0))

    for a in ax:
        a.set_facecolor(CAM_SLATE_1)
        a.set_prop_cycle(cycler(color=CAM_COLOR_CYCLE))
        a.tick_params(
            axis="both", direction="in", top=True, right=True, labelsize=8, which="both"
        )

        a.xaxis.label.set_fontsize(11)
        a.yaxis.label.set_fontsize(11)


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
    setup_fancy_figure(fig, [ax])
    return fig, ax


def get_two_panel_figure() -> tuple[Figure, list[Axes]]:
    """Create a two-panel figure with fancy styling."""
    setup_rc_params()
    fig, ax = plt.subplots(layout="constrained", ncols=2, figsize=(6, 2.5))
    setup_fancy_figure(fig, ax)
    return fig, ax


def get_three_panel_figure() -> tuple[Figure, list[Axes]]:
    """Create a three-panel figure with fancy styling."""
    setup_rc_params()
    fig, ax = plt.subplots(layout="constrained", ncols=3, figsize=(9, 2.5))
    setup_fancy_figure(fig, ax)
    return fig, ax
