"""Tests for the lm module using the California Housing dataset."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing

from sampleology.lm import LinearModel, lm


@pytest.fixture
def housing_data():
    """Load California Housing dataset for testing."""
    try:
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df["MedHouseVal"] = housing.target  # Median house value
        return df
    except Exception:
        # Fallback: create synthetic data similar to California Housing
        np.random.seed(42)
        n_samples = 20640

        # Create synthetic features similar to California Housing
        MedInc = np.random.lognormal(3.5, 0.5, n_samples)  # Median income
        HouseAge = np.random.uniform(1, 52, n_samples)  # House age
        AveRooms = np.random.normal(5.4, 1.4, n_samples)  # Average rooms
        AveBedrms = np.random.normal(1.1, 0.5, n_samples)  # Average bedrooms
        Population = np.random.exponential(1425, n_samples)  # Population
        AveOccup = np.random.normal(3.1, 1.0, n_samples)  # Average occupancy
        Latitude = np.random.uniform(32.5, 42.0, n_samples)  # Latitude
        Longitude = np.random.uniform(-124.3, -114.3, n_samples)  # Longitude

        # Create target variable (house prices) with some relationship to features
        MedHouseVal = (
            2.0
            + 0.4 * MedInc
            - 0.01 * HouseAge
            + 0.1 * AveRooms
            - 0.2 * AveBedrms
            - 0.0001 * Population
            + 0.05 * AveOccup
            - 0.01 * Latitude
            + 0.01 * Longitude
            + np.random.normal(0, 0.5, n_samples)
        )

        df = pd.DataFrame(
            {
                "MedInc": MedInc,
                "HouseAge": HouseAge,
                "AveRooms": AveRooms,
                "AveBedrms": AveBedrms,
                "Population": Population,
                "AveOccup": AveOccup,
                "Latitude": Latitude,
                "Longitude": Longitude,
                "MedHouseVal": MedHouseVal,
            }
        )
        return df


def test_linear_model_initialization():
    """Test LinearModel initialization."""
    formula = "y ~ x1 + x2"
    model = LinearModel(formula)

    assert model.formula == formula
    assert not model.fitted
    assert model.coefficients is None
    assert model.intercept is None


def test_linear_model_fit(housing_data):
    """Test fitting a linear model with California Housing data."""
    formula = "MedHouseVal ~ MedInc + AveRooms + HouseAge"
    model = LinearModel(formula)

    # Fit the model
    fitted_model = model.fit(housing_data)

    # Check that model is fitted
    assert fitted_model.fitted
    assert fitted_model.coefficients is not None
    assert fitted_model.intercept is not None
    assert fitted_model.feature_names is not None
    assert len(fitted_model.feature_names) == 4  # Intercept + 3 variables

    # Check that coefficients are reasonable
    assert all(isinstance(coef, (int, float)) for coef in fitted_model.coefficients.values())
    assert isinstance(fitted_model.intercept, (int, float))

    # Check that the three main dataframes are created
    assert fitted_model.results_df is not None
    assert fitted_model.residuals_summary_wide is not None
    assert fitted_model.overall_stats_df is not None


def test_linear_model_predict(housing_data):
    """Test making predictions with fitted model."""
    formula = "MedHouseVal ~ MedInc + AveRooms"
    model = LinearModel(formula)
    model.fit(housing_data)

    # Make predictions on the same data
    predictions = model.predict(housing_data)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(housing_data)
    assert not np.any(np.isnan(predictions))


def test_linear_model_summary(housing_data):
    """Test model summary functionality."""
    formula = "MedHouseVal ~ MedInc + AveRooms + HouseAge"
    model = LinearModel(formula)
    model.fit(housing_data)

    summary = model.summary()

    assert summary["formula"] == formula
    assert summary["intercept"] == model.intercept
    assert summary["coefficients"] == model.coefficients
    assert summary["feature_names"] == model.feature_names
    assert summary["n_features"] == 4  # Intercept + 3 variables
    assert "results_df" in summary
    assert "residuals_summary_wide" in summary
    assert "overall_stats_df" in summary


def test_linear_model_score(housing_data):
    """Test model scoring functionality."""
    formula = "MedHouseVal ~ MedInc + AveRooms + HouseAge"
    model = LinearModel(formula)
    model.fit(housing_data)

    scores = model.score(housing_data)

    assert "r2_score" in scores
    assert "mse" in scores
    assert "rmse" in scores
    assert 0 <= scores["r2_score"] <= 1  # R² should be between 0 and 1
    assert scores["mse"] >= 0  # MSE should be non-negative
    assert scores["rmse"] >= 0  # RMSE should be non-negative


def test_lm_convenience_function(housing_data):
    """Test the convenience lm function."""
    formula = "MedHouseVal ~ MedInc + AveRooms"
    model = lm(formula, housing_data)

    assert model.fitted
    assert model.formula == formula
    assert model.coefficients is not None


def test_complex_formula(housing_data):
    """Test more complex formula with interactions."""
    formula = "MedHouseVal ~ MedInc + AveRooms + MedInc:AveRooms"
    model = LinearModel(formula)
    model.fit(housing_data)

    assert model.fitted
    assert len(model.feature_names) >= 3  # Should include interaction term


def test_formula_with_categorical(housing_data):
    """Test formula with categorical variable."""
    # Create a categorical variable from HouseAge
    housing_data["AgeGroup"] = pd.cut(housing_data["HouseAge"], bins=[0, 20, 40, 60], labels=["Young", "Middle", "Old"])

    formula = "MedHouseVal ~ MedInc + AgeGroup"
    model = LinearModel(formula)
    model.fit(housing_data)

    assert model.fitted
    # Should have more features due to dummy encoding of categorical


def test_error_unfitted_predict():
    """Test that prediction fails on unfitted model."""
    model = LinearModel("y ~ x")

    with pytest.raises(ValueError, match="Model must be fitted"):
        model.predict(pd.DataFrame({"x": [1, 2, 3]}))


def test_error_unfitted_summary():
    """Test that summary fails on unfitted model."""
    model = LinearModel("y ~ x")

    with pytest.raises(ValueError, match="Model must be fitted"):
        model.summary()


def test_error_unfitted_score():
    """Test that scoring fails on unfitted model."""
    model = LinearModel("y ~ x")

    with pytest.raises(ValueError, match="Model must be fitted"):
        model.score(pd.DataFrame({"x": [1, 2, 3]}))
