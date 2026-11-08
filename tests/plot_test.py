import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.colors import Colormap, to_rgb

from classical_diffusion.plot import (
    CAM_BLUE,
    CAM_BLUE_CHERRY_DIVERGING_CMAP,
    CAM_BLUE_SEQUENTIAL_CMAP,
    CAM_CHERRY,
    CAM_CHERRY_SEQUENTIAL_CMAP,
    CAM_SLATE_1,
    CAM_SUNSET_CMAP,
)


def _lightness(rgb: np.ndarray) -> np.ndarray:
    """Lightness L* (CIE 1976) of sRGB colours."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)  # ruff: ignore[magic-value-comparison]
    y = linear @ np.array([0.2126, 0.7152, 0.0722])
    f = np.where(y > (6 / 29) ** 3, y ** (1 / 3), y / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f - 16


def _colors(cmap: Colormap) -> np.ndarray:
    return cmap(np.linspace(0, 1, 256))[:, :3]


SEQUENTIAL = [CAM_BLUE_SEQUENTIAL_CMAP, CAM_CHERRY_SEQUENTIAL_CMAP, CAM_SUNSET_CMAP]
ALL = [*SEQUENTIAL, CAM_BLUE_CHERRY_DIVERGING_CMAP]


@pytest.mark.parametrize("cmap", ALL, ids=lambda c: c.name)
def test_cmap_is_registered(cmap: Colormap) -> None:
    assert mpl.colormaps[cmap.name] == cmap


@pytest.mark.parametrize("cmap", SEQUENTIAL, ids=lambda c: c.name)
def test_sequential_lightness_rises(cmap: Colormap) -> None:
    assert np.all(np.diff(_lightness(_colors(cmap))) > 0)


def test_diverging_lightness_peaks_in_the_centre() -> None:
    lightness = _lightness(_colors(CAM_BLUE_CHERRY_DIVERGING_CMAP))
    centre = np.argmax(lightness)
    assert np.all(np.diff(lightness[: centre + 1]) > 0)
    assert np.all(np.diff(lightness[centre:]) < 0)
    assert 127 <= centre <= 128  # ruff: ignore[magic-value-comparison]


@pytest.mark.parametrize("cmap", ALL, ids=lambda c: c.name)
def test_cmap_contrasts_with_background(cmap: Colormap) -> None:
    background = _lightness(np.array(to_rgb(CAM_SLATE_1)))
    assert np.all(background - _lightness(_colors(cmap)) > 10)  # ruff: ignore[magic-value-comparison]


@pytest.mark.parametrize(
    ("cmap", "colors"),
    [
        (CAM_BLUE_SEQUENTIAL_CMAP, [CAM_BLUE.dark, CAM_BLUE.warm]),
        (
            CAM_CHERRY_SEQUENTIAL_CMAP,
            [CAM_CHERRY.dark, CAM_CHERRY.base, CAM_CHERRY.warm],
        ),
        (
            CAM_BLUE_CHERRY_DIVERGING_CMAP,
            [CAM_BLUE.dark, CAM_CHERRY.base, CAM_CHERRY.dark],
        ),
    ],
    ids=lambda c: getattr(c, "name", ""),
)
def test_cmap_passes_through_cam_colors(cmap: Colormap, colors: list[str]) -> None:
    sampled = cmap(np.linspace(0, 1, 4097))[:, :3]
    for color in colors:
        distance = np.linalg.norm(sampled - np.array(to_rgb(color)), axis=1)
        assert distance.min() < 2 / 255
