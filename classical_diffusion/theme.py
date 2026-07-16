from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from cycler import cycler

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
    CAM_BLUE.warm,
    CAM_BLUE.dark,
    CAM_CHERRY.warm,
    CAM_CHERRY.dark,
    CAM_CREST.warm,
    CAM_CREST.dark,
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
    ax.set_prop_cycle(cycler(color=CAM_COLOR_CYCLE))
    ax.tick_params(
        axis="both", direction="in", top=True, right=True, labelsize=8, which="both"
    )

    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    return fig, ax
