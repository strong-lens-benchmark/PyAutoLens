import shutil
from pathlib import Path

import pytest
import autolens as al
from autolens.analysis import plotter as vis

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plotter_plotter_setup():
    return directory / "files"


def test__tracer(masked_imaging_7x7, tracer_x2_plane_7x7, plot_path, plot_patch):
    if plot_path.exists():
        shutil.rmtree(plot_path)

    plotter = vis.Plotter(image_path=plot_path)

    plotter.tracer(
        tracer=tracer_x2_plane_7x7,
        grid=masked_imaging_7x7.grids.lp,
    )

    assert str(plot_path / "galaxies_images.png") in plot_patch.paths

    image = al.ndarray_via_fits_from(
        file_path=plot_path / "tracer.fits", hdu=0
    )

    assert image.shape == (5, 5)


def test__image_with_positions(image_7x7, positions_x2, plot_path, plot_patch):
    if plot_path.exists():
        shutil.rmtree(plot_path)

    plotter = vis.Plotter(image_path=plot_path)

    plotter.image_with_positions(image=image_7x7, positions=positions_x2)

    assert str(plot_path / "image_with_positions.png") in plot_patch.paths
