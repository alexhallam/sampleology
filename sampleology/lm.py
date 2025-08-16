import polars as pl
import numpy as np
from scipy import linalg as la
from scipy.stats import t as student_t
from formulaic import model_matrix
import tidy_viewer_py as tv
from scipy.stats import f as f_dist
import re

# read data (polars) and hand a pandas view to formulaic
df = pl.read_csv("sampleology/mtcars.csv")
df_pd = df.to_pandas()


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


# design matrices
y_df, X_df = model_matrix("mpg ~ C(am) + C(cyl) + hp*wt + I(hp**3)", df_pd)
print(X_df)  # returns pandas objects
coef_names = list(X_df.columns)  # keep names before converting
clean_names = make_unique([clean_coef_name(n) for n in coef_names])

# to numpy
y = y_df.to_numpy().ravel()
X = X_df.to_numpy()

# OLS via LAPACK
beta, _, rank, s = la.lstsq(X, y, lapack_driver="gelsd")
residuals = y - X @ beta

# inference pieces
n, p = X.shape
df_resid = n - p
sigma2 = (residuals @ residuals) / df_resid
XtX_inv = np.linalg.pinv(X.T @ X)  # safe inverse
var_beta = sigma2 * XtX_inv
sd = np.sqrt(np.diag(var_beta))
t_value = beta / sd
pr_gt_t = 2.0 * (1.0 - student_t.cdf(np.abs(t_value), df=df_resid))

###


# assemble results in polars
results_df = pl.DataFrame(
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

residuals_summary_wide = (
    residuals_summary.melt(id_vars="statistic", variable_name="variable", value_name="value")
    .with_columns(pl.col("value").cast(pl.Float64, strict=False))  # unify dtypes
    .pivot(values="value", index="variable", columns="statistic", aggregate_function="first")
    .select(pl.col("variable", "min", "25%", "mean", "75%", "max", "std"))
)

tv.print_polars_dataframe(residuals_summary_wide)

tv.print_polars_dataframe(results_df)


# --- overall model stats (R^2, adj R^2, F, p, n) ---
y_hat = X @ beta
SST = np.sum((y - y.mean()) ** 2)
SSE = np.sum((residuals) ** 2)
SSR = SST - SSE

# detect intercept to set model df
has_intercept = any(nm.lower() in ("intercept", "(intercept)") for nm in coef_names) or np.allclose(X[:, 0], 1.0)
df_model = p - 1 if has_intercept else p  # numerator df
# df_resid already computed above

r_squared = 1.0 - SSE / SST
adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - p)

MSR = SSR / df_model
MSE = SSE / df_resid
f_stat = MSR / MSE
p_val = f_dist.sf(f_stat, df_model, df_resid)

overall_stats_df = pl.DataFrame(
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

tv.print_polars_dataframe(overall_stats_df)
