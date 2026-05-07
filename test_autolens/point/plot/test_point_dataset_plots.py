from pathlib import Path

import pytest

from autolens.point.plot.point_dataset_plots import subplot_dataset

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_point_dataset_plotter_setup():
    return directory / "files" / "plots" / "point_dataset"


def test__subplot_dataset(point_dataset, plot_path, plot_patch):
    subplot_dataset(
        dataset=point_dataset,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "dataset_point.png") in plot_patch.paths
