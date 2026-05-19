import math

import numpy as np
import pytest

import autoarray as aa
import autolens as al

from autogalaxy.operate.lens_calc import LensCalc

from autolens.weak.fit import FitWeak


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


def _make_dataset(noise_sigma=0.3, seed=0, einstein_radius=1.6):
    grid = aa.Grid2DIrregular(
        values=[(0.7, 0.5), (1.0, 1.0), (-0.3, 0.6), (-1.1, -0.8)]
    )
    tracer = _isothermal_tracer(einstein_radius=einstein_radius)
    simulator = al.SimulatorShearYX(noise_sigma=noise_sigma, seed=seed)
    return simulator.via_tracer_from(tracer=tracer, grid=grid, name="test")


def test__zero_noise_round_trip_zero_residuals():
    """
    The fit's `model_shear` is derived from the exact same `LensCalc` primitive that `SimulatorShearYX` uses,
    so a noise-free round-trip through `FitWeak` with the truth tracer must produce a residual map of zeros.
    """
    truth = _isothermal_tracer()
    dataset = _make_dataset(noise_sigma=0.0)

    fit = FitWeak(dataset=dataset, tracer=truth)

    np.testing.assert_allclose(fit.residual_map, 0.0, atol=1e-12)


def test__chi_squared_zero_for_perfect_fit():
    """
    A noise-free dataset fit by the truth tracer has zero residuals; well-defined ``noise_sigma > 0`` then
    guarantees zero chi-squared (any-finite-σ divided into zero residuals gives zero).
    """
    truth = _isothermal_tracer()
    dataset = al.SimulatorShearYX(noise_sigma=0.0, seed=0).via_tracer_from(
        tracer=truth,
        grid=aa.Grid2DIrregular(values=[(0.7, 0.5), (1.0, 1.0), (-0.3, 0.6)]),
    )
    # Replace the zero noise_map with a finite σ so the chi-squared division is well-defined.
    dataset.noise_map = aa.ArrayIrregular(values=[0.3, 0.3, 0.3])

    fit = FitWeak(dataset=dataset, tracer=truth)

    assert fit.chi_squared == pytest.approx(0.0, abs=1e-12)


def test__log_likelihood_against_hand_computed():
    """
    Verify the log-likelihood formula directly: compute chi-squared and noise normalisation from
    `dataset.shear_yx`, `model_shear` and `noise_map` using plain numpy, and assert the class agrees to 1e-9.
    """
    dataset = _make_dataset(noise_sigma=0.3, seed=42)
    perturbed = _isothermal_tracer(einstein_radius=1.5)

    fit = FitWeak(dataset=dataset, tracer=perturbed)

    data = np.asarray(dataset.shear_yx)
    model = np.asarray(
        LensCalc.from_tracer(perturbed).shear_yx_2d_via_hessian_from(
            grid=dataset.positions
        )
    )
    sigma = np.asarray(dataset.noise_map)

    expected_chi_squared = np.sum(((data - model) / sigma[:, None]) ** 2)
    expected_noise_norm = 2.0 * np.sum(np.log(2.0 * math.pi * sigma**2))
    expected_log_likelihood = -0.5 * (expected_chi_squared + expected_noise_norm)

    assert fit.chi_squared == pytest.approx(expected_chi_squared, abs=1e-9)
    assert fit.noise_normalization == pytest.approx(expected_noise_norm, abs=1e-9)
    assert fit.log_likelihood == pytest.approx(expected_log_likelihood, abs=1e-9)


def test__residual_map_shape_is_n_galaxies_by_2():
    dataset = _make_dataset(noise_sigma=0.3, seed=7)
    fit = FitWeak(dataset=dataset, tracer=_isothermal_tracer(einstein_radius=1.5))

    assert fit.residual_map.shape == (dataset.n_galaxies, 2)
    assert fit.chi_squared_map.shape == (dataset.n_galaxies, 2)


def test__log_likelihood_drops_for_wrong_model():
    """
    A 0.1 perturbation of the lens einstein_radius away from truth should drop the log-likelihood by at
    least 1 nat — sanity-check that wrong models are penalised.
    """
    truth = _isothermal_tracer(einstein_radius=1.6)
    dataset = _make_dataset(noise_sigma=0.05, seed=11, einstein_radius=1.6)

    fit_truth = FitWeak(dataset=dataset, tracer=truth)
    fit_wrong = FitWeak(
        dataset=dataset, tracer=_isothermal_tracer(einstein_radius=1.7)
    )

    assert fit_wrong.log_likelihood < fit_truth.log_likelihood - 1.0


def test__figure_of_merit_equals_log_likelihood():
    dataset = _make_dataset(noise_sigma=0.3, seed=3)
    fit = FitWeak(dataset=dataset, tracer=_isothermal_tracer())

    assert fit.figure_of_merit == fit.log_likelihood
