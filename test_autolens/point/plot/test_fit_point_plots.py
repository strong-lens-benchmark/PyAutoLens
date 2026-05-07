from pathlib import Path

import pytest

from autolens.point.plot.fit_point_plots import subplot_fit

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_fit_point_plotter_setup():
    return Path(__file__).resolve().parent / "files" / "plots" / "fit_point"


def test__subplot_fit(fit_point_dataset_x2_plane, plot_path, plot_patch):
    subplot_fit(
        fit=fit_point_dataset_x2_plane,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit.png") in plot_patch.paths
