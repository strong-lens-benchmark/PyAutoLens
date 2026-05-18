from pathlib import Path

import autoarray as aa
import autolens as al

import pytest

from autolens.weak.plot.fit_weak_plots import (
    plot_data_vs_model,
    plot_residuals,
    plot_chi_squared_map,
    subplot_fit_weak,
)

directory = Path(__file__).resolve().parent


def _isothermal_tracer(einstein_radius=1.6, ell_comps=(0.0, 0.05)):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
        ),
    )
    source = al.Galaxy(redshift=1.0)
    return al.Tracer(galaxies=[lens, source])


@pytest.fixture(name="fit_weak")
def make_fit_weak():
    """Deterministic 4-galaxy FitWeak with non-zero residuals (model einstein_radius is slightly wrong)."""
    grid = aa.Grid2DIrregular(
        values=[(0.7, 0.5), (1.0, 1.0), (-0.3, 0.6), (-1.1, -0.8)]
    )
    truth = _isothermal_tracer(einstein_radius=1.6)
    dataset = al.SimulatorShearYX(noise_sigma=0.0, seed=0).via_tracer_from(
        tracer=truth, grid=grid, name="test"
    )
    dataset.noise_map = aa.ArrayIrregular(values=[0.3, 0.3, 0.3, 0.3])
    model = _isothermal_tracer(einstein_radius=1.5)
    return al.FitWeak(dataset=dataset, tracer=model)


@pytest.fixture(name="plot_path")
def make_plot_path():
    return directory / "files" / "plots" / "fit_weak"


def test__plot_data_vs_model(fit_weak, plot_path, plot_patch):
    plot_data_vs_model(
        fit=fit_weak,
        output_path=plot_path,
        output_filename="data_vs_model",
        output_format="png",
    )
    assert str(plot_path / "data_vs_model.png") in plot_patch.paths


def test__plot_residuals(fit_weak, plot_path, plot_patch):
    plot_residuals(
        fit=fit_weak,
        output_path=plot_path,
        output_filename="residuals",
        output_format="png",
    )
    assert str(plot_path / "residuals.png") in plot_patch.paths


def test__plot_chi_squared_map(fit_weak, plot_path, plot_patch):
    plot_chi_squared_map(
        fit=fit_weak,
        output_path=plot_path,
        output_filename="chi_squared_map",
        output_format="png",
    )
    assert str(plot_path / "chi_squared_map.png") in plot_patch.paths


def test__subplot_fit_weak(fit_weak, plot_path, plot_patch):
    subplot_fit_weak(
        fit=fit_weak,
        output_path=plot_path,
        output_filename="subplot_fit_weak",
        output_format="png",
    )
    assert str(plot_path / "subplot_fit_weak.png") in plot_patch.paths
