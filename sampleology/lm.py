"""Linear model functionality using formulaic for formula parsing."""

import polars as pl
import numpy as np
from scipy import linalg as la
from scipy.stats import t as student_t
from scipy.stats import f as f_dist
from formulaic import model_matrix
import tidy_viewer_py as tv
import re
from typing import Dict, Any


def _to_snake(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("**", "_pow_")  # exponent marker first
    s = s.replace(":", "_x_").replace("*", "_x_")
    s = s.replace("/", "_over_").replace("+", "_plus_").replace("-", "_minus_")
    s = re.sub(r"[()\[\]]+", "", s)  # drop parens/brackets
    s = re.sub(r"[^a-z0-9]+", "_", s)  # non-alnum -> _
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def clean_coef_name(name: str) -> str:
    s = name

    # Intercept
    if s.strip().lower() in ("intercept", "(intercept)"):
        return "intercept"

    # C(var)[T.level]  →  var_level
    def repl_cat(m):
        var = _to_snake(m.group("var"))
        lvl = _to_snake(m.group("lvl"))
        return f"{var}_{lvl}"

    s = re.sub(r"C\((?P<var>[^)]+)\)\[T\.(?P<lvl>.+?)\]", repl_cat, s)

    # I(x ** k)  →  x_sq / x_cubed / x_pow_k
    def repl_pow(m):
        var = _to_snake(m.group("var"))
        exp = m.group("exp")
        return f"{var}_sq" if exp == "2" else (f"{var}_cubed" if exp == "3" else f"{var}_pow_{exp}")

    s = re.sub(r"I\(\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*\*\*\s*(?P<exp>\d+)\s*\)", repl_pow, s)

    # Generic I( ... ) fallback: just snake_case the inside
    s = re.sub(r"I\((?P<inner>[^)]+)\)", lambda m: _to_snake(m.group("inner")), s)

    # Final pass (handles ":" → "_x_" and any leftovers)
    return _to_snake(s)


def make_unique(names):
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 1
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")  # suffix if collision
    return out


class LinearModel:
    """A linear model class that uses formulaic for formula parsing."""

    def __init__(self, formula: str):
        """Initialize the linear model with a formula.

        Args:
            formula: A formula string in R-style notation (e.g., "y ~ x1 + x2")
        """
        self.formula = formula
        self.fitted = False
        self.coefficients = None
        self.intercept = None
        self.feature_names = None
        self.residuals = None
        self.fitted_values = None
        self.results_df = None
        self.residuals_summary_wide = None
        self.overall_stats_df = None

    def fit(self, data) -> "LinearModel":
        """Fit the linear model using the provided data.

        Args:
            data: DataFrame (Polars or Pandas) containing the variables in the formula

        Returns:
            self: The fitted model
        """
        # Convert to pandas for formulaic if needed
        if hasattr(data, "to_pandas"):
            df_pd = data.to_pandas()
        else:
            df_pd = data

        # Parse the formula to get design matrices
        y_df, X_df = model_matrix(self.formula, df_pd)

        # Clean coefficient names
        coef_names = list(X_df.columns)
        clean_names = make_unique([clean_coef_name(n) for n in coef_names])

        # Convert to numpy
        y = y_df.to_numpy().ravel()
        X = X_df.to_numpy()

        # OLS via LAPACK
        beta, _, _, _ = la.lstsq(X, y, lapack_driver="gelsd")
        residuals = y - X @ beta

        # Store basic results
        self.coefficients = dict(zip(clean_names, beta))
        self.intercept = beta[0] if "intercept" in clean_names else 0.0
        self.feature_names = clean_names
        self.residuals = residuals
        self.fitted_values = X @ beta
        self.fitted = True

        # Inference pieces
        n, p = X.shape
        df_resid = n - p
        sigma2 = (residuals @ residuals) / df_resid
        XtX_inv = np.linalg.pinv(X.T @ X)  # safe inverse
        var_beta = sigma2 * XtX_inv
        sd = np.sqrt(np.diag(var_beta))
        t_value = beta / sd
        pr_gt_t = 2.0 * (1.0 - student_t.cdf(np.abs(t_value), df=df_resid))

        # Assemble results in polars
        self.results_df = pl.DataFrame(
            {
                "coefficients": clean_names,
                "estimate": beta,
                "std_error": sd,
                "t_value": t_value,
                "pr_gt_t": pr_gt_t,
            }
        )

        # 5-number (and friends) summary of residuals, pivoted wide
        residuals_summary = pl.DataFrame({"residuals": residuals}).describe()

        self.residuals_summary_wide = (
            residuals_summary.unpivot(index="statistic", on="residuals", variable_name="variable", value_name="value")
            .with_columns(pl.col("value").cast(pl.Float64, strict=False))  # unify dtypes
            .pivot(values="value", index="variable", on="statistic", aggregate_function="first")
            .rename(
                {
                    "25%": "residuals_p25",
                    "75%": "residuals_p75",
                    "min": "residuals_min",
                    "max": "residuals_max",
                    "std": "residuals_std",
                }
            )
            .select(
                pl.col(
                    "variable",
                    "residuals_min",
                    "residuals_p25",
                    "mean",
                    "residuals_p75",
                    "residuals_max",
                    "residuals_std",
                )
            )
        )

        # Overall model stats (R^2, adj R^2, F, p, n)
        SST = np.sum((y - y.mean()) ** 2)
        SSE = np.sum((residuals) ** 2)
        SSR = SST - SSE

        # detect intercept to set model df
        has_intercept = any(nm.lower() in ("intercept", "(intercept)") for nm in coef_names) or np.allclose(
            X[:, 0], 1.0
        )
        df_model = p - 1 if has_intercept else p  # numerator df

        r_squared = 1.0 - SSE / SST
        adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - p)

        MSR = SSR / df_model
        MSE = SSE / df_resid
        f_stat = MSR / MSE
        p_val = f_dist.sf(f_stat, df_model, df_resid)

        self.overall_stats_df = pl.DataFrame(
            {
                "multiple_r_squared": [r_squared],
                "adjusted_r_squared": [adj_r_squared],
                "f_statistic": [f_stat],
                "df_model": [df_model],
                "df_resid": [df_resid],
                "p_value": [p_val],
                "n": [n],
            }
        )

        return self

    def predict(self, data) -> np.ndarray:
        """Make predictions using the fitted model.

        Args:
            data: DataFrame (Polars or Pandas) containing the predictor variables

        Returns:
            Predictions as numpy array
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to pandas for formulaic if needed
        if hasattr(data, "to_pandas"):
            df_pd = data.to_pandas()
        else:
            df_pd = data

        y_df, X_df = model_matrix(self.formula, df_pd)
        X = X_df.to_numpy()

        return X @ np.array(list(self.coefficients.values()))

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the fitted model.

        Returns:
            Dictionary containing model summary information
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting summary")

        return {
            "formula": self.formula,
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "residuals": self.residuals,
            "results_df": self.results_df,
            "residuals_summary_wide": self.residuals_summary_wide,
            "overall_stats_df": self.overall_stats_df,
        }

    def score(self, data, y_true=None) -> Dict[str, float]:
        """Calculate model performance metrics.

        Args:
            data: DataFrame (Polars or Pandas) containing the predictor variables
            y_true: True values (if None, will try to extract from data)

        Returns:
            Dictionary containing R² and MSE scores
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before scoring")

        y_pred = self.predict(data)

        if y_true is None:
            # Try to extract response from data using the formula
            if hasattr(data, "to_pandas"):
                df_pd = data.to_pandas()
            else:
                df_pd = data
            y_df, X_df = model_matrix(self.formula, df_pd)
            y_true = y_df.to_numpy().flatten()

        # Calculate R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Calculate MSE and RMSE
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        return {"r2_score": r2, "mse": mse, "rmse": rmse}

    def print_results(self):
        """Print the three main result dataframes."""
        if not self.fitted:
            raise ValueError("Model must be fitted before printing results")

        tv.print_polars_dataframe(self.residuals_summary_wide)
        tv.print_polars_dataframe(self.results_df)
        tv.print_polars_dataframe(self.overall_stats_df)


def lm(formula: str, data) -> LinearModel:
    """Convenience function to create and fit a linear model.

    Args:
        formula: A formula string in R-style notation
        data: DataFrame (Polars or Pandas) containing the variables

    Returns:
        Fitted LinearModel instance
    """
    model = LinearModel(formula)
    return model.fit(data)


# Test the implementation
if __name__ == "__main__":
    # read data (polars) and hand a pandas view to formulaic
    df = pl.read_csv("sampleology/mtcars.csv")

    # Fit the model
    model = lm("mpg ~ C(am) + C(cyl) + hp*wt + I(hp**3)", df)

    # Print results
    model.print_results()

    # Test assertions to match expected output
    def test_model_outputs():
        """Test that the model outputs match expected values."""

        # Test residuals summary structure
        assert model.residuals_summary_wide.shape == (1, 7)
        expected_residuals_cols = [
            "variable",
            "residuals_min",
            "residuals_p25",
            "mean",
            "residuals_p75",
            "residuals_max",
            "residuals_std",
        ]
        assert list(model.residuals_summary_wide.columns) == expected_residuals_cols

        # Test overall stats structure
        assert model.overall_stats_df.shape == (1, 7)
        expected_stats_cols = [
            "multiple_r_squared",
            "adjusted_r_squared",
            "f_statistic",
            "df_model",
            "df_resid",
            "p_value",
            "n",
        ]
        assert list(model.overall_stats_df.columns) == expected_stats_cols

        # Test results structure
        assert model.results_df.shape == (8, 5)
        expected_results_cols = ["coefficients", "estimate", "std_error", "t_value", "pr_gt_t"]
        assert list(model.results_df.columns) == expected_results_cols

        # Test specific coefficient names
        expected_coefficients = ["intercept", "am_1", "cyl_6", "cyl_8", "hp", "wt", "hp_cubed", "hp_x_wt"]
        assert list(model.results_df["coefficients"]) == expected_coefficients

        # Test that R-squared is reasonable
        r_squared = model.overall_stats_df["multiple_r_squared"][0]
        assert 0.8 < r_squared < 0.95  # Should be around 0.89

        # Test that residuals mean is close to zero
        residuals_mean = model.residuals_summary_wide["mean"][0]
        assert abs(residuals_mean) < 1e-9  # Allow for small numerical precision differences

        print("All tests passed!")

    test_model_outputs()
