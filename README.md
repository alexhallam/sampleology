<h1 align="center">Regression</h1>
<p align="center">Sampleology❤️lStats</p>
<p align="center">Modern statistical algorithms that make big problems small and small problems sufficient.</p>

Linear models are super cool! I like them more than I should. In a world where regression data sets are larger than ever, professionals are told to
through more compute at the problem. This regression package is not anti compute, a good CPU and GPU makes a difference. I do hope to side-step super
clusters. It is impressive how well approximations get us to prediction intervals, estimates, and estimate uncertainty. We have outstanding methods
for approximate Bayes. As a tangent, while full Bayes is nice in theory, I find that it is faster to use other tools and usually close enough. I know
this is a hot take. I am a pragmatic practitioner not a theorist.

So, what can you expect from this package?

1. A beautiful interface via formulaic and tidy-viewer-py. I like to look at data so I think it is important to have a nice interface. It is mid 2025
   and it seems that we are approaching an era of AI. Maybe you will never have to use this package because Agents will do your modeling for you. That
   is fine! This is for enjoyment, if not for you then maybe the robots will enjoy it. ;)
2. The same lm() output you would get from Rs lm() for smaller data plus some..
3. There should never be a reason for clusters. Larger than memory .. check. 
4. The cutting edge regression methods as of mid 2025.
5. Most decisions are automatted, but can be reported.
6. I am partial to polars. This will be the dataframe of choice.
7. I am partial to `jax` for cases where autodiff is needed.
8. Data-thinning will be used when possible for cross validation and prediction intervals.
9. I will try to give standard errors for parameters and sampling for predictions. If possible I would like to give distributions for parameters


## Examples

```
import sampleology_linear as sl

model = sl.lm("mpg ~ hp + wt + I(hp**2)", data=mtcars)
model = sl.glm("success ~ dose + age", data=trials, family="logistic") 
model = sl.robust("y ~ x1 + x2:x3", data=df)  # Only when robust is wanted
```

## The magic:

lm() automatically detects if data is big → uses CQRRPT/iBOSS
lm() automatically detects outliers → switches to robust methods
glm() automatically handles numerical instabilities → uses appropriate stabilization
User never thinks about it, but output will be transparent.



## Example Implimentation

```
def lm(formula, data, **kwargs):
    """Linear regression that automatically optimizes itself"""
    
    # Parse with formulaic
    X, y = _parse_formula(formula, data)
    n, p = X.shape
    
    # INVISIBLE DECISION TREE
    if n > 50000:
        # Big data → use CQRRPT
        method = "cqrrpt"
        result = _cqrrpt_regression(X, y)
        
    elif n > 10000 and p > 100:
        # Medium-high dim → use iBOSS  
        method = "iboss"
        result = _iboss_regression(X, y)
        
    elif _detect_outliers(y):
        # Outliers detected → robust regression
        method = "robust"
        result = _huber_regression(X, y)
        
    else:
        # Standard case → fast JAX OLS
        method = "ols"
        result = _jax_ols(X, y)
    
    # Return same interface regardless of method used
    return LinearRegressionResult(
        formula=formula,
        method=method,  # Hidden unless user asks
        **result
    )
```

## Debug For Power Users

```
# Magic by default
model = sl.lm("y ~ x", data=df)

# But transparency available if wanted
model._method  # "cqrrpt"
model._diagnostics["sketch_efficiency"]  # 0.987
model._diagnostics["method_reason"]  # "Large dataset (n=50000)"

# Or enable debug mode
sl.set_debug(True)
model = sl.lm("y ~ x", data=df)
# Now shows: "Using CQRRPT sketching (efficiency: 98.7%)"
```

## API Clean

```
# Core functions - that's it!
sl.lm(formula, data)           # Linear regression
sl.glm(formula, data, family)  # Generalized linear models  
sl.robust(formula, data)       # Explicitly robust (Huber/bisquare)
sl.mixed(formula, data, groups)# Mixed effects (simple cases)

# All return same rich result objects
result.summary()     # Beautiful output
result.plot()        # Diagnostic plots  
result.predict()     # Predictions
result.confint()     # Confidence intervals
```

## API Another Example

```
import sampleology_linear as sl

# Quantile regression - estimates full conditional distribution
model = sl.quantile("mpg ~ hp + wt", data=mtcars)                    # Median by default
model = sl.quantile("mpg ~ hp + wt", data=mtcars, tau=0.9)          # 90th percentile
model = sl.quantile("mpg ~ hp + wt", data=mtcars, tau=[0.1, 0.5, 0.9])  # Multiple quantiles

# Composite quantile regression - uncertainty bands automatically
model = sl.composite_quantile("mpg ~ hp + wt", data=mtcars)         # Auto-selects quantiles
model = sl.composite_quantile("mpg ~ hp + wt", data=mtcars, 
                             quantiles="full")                      # Dense grid of quantiles

# MM-estimators - robust with high efficiency
model = sl.mm("mpg ~ hp + wt", data=mtcars)                         # Auto-selects breakdown point
model = sl.mm("mpg ~ hp + wt", data=mtcars, breakdown=0.5)          # 50% breakdown point

# Double ML - causal inference with ML
model = sl.double_ml("wage ~ treatment", data=df, 
                    controls=["age", "education", "experience"])     # Auto-selects ML methods
model = sl.double_ml("wage ~ treatment", data=df, 
                    controls=["age*education", "I(experience**2)"],
                    ml_method="forest")                              # Specify ML nuisance

# Adaptive LASSO - data chooses regularization
model = sl.adaptive_lasso("y ~ .", data=df)                         # Auto-adapts penalties
model = sl.adaptive_lasso("y ~ x1 + x2 + x3", data=df, 
                         adaptive_weights="ols")                     # OLS-based weights

# Distributional regression - model full distribution
model = sl.distributional("mpg ~ hp + wt", data=mtcars)             # Auto-detects parameters to model
model = sl.distributional("mpg ~ hp + wt", data=mtcars, 
                         parameters=["mu", "sigma"])                 # Explicit parameters
model = sl.distributional("count ~ x1 + x2", data=df, 
                         family="poisson")                          # Non-Gaussian

# Ensemble stacking - optimal combination
model = sl.ensemble("mpg ~ hp + wt", data=mtcars)                   # Auto-selects base methods
model = sl.ensemble("mpg ~ hp + wt", data=mtcars,
                   methods=["lm", "quantile", "robust"])            # Explicit methods
```

## Possible Parameter Uncertainty

```
# Behind the scenes for parameter uncertainty
param_ci = model.confint(method="bootstrap")  # Bootstrap coefficient CIs
param_dist = model.bootstrap_distribution()   # Full parameter distributions
# Standard approach
param_ci = model.confint(method="wald")       # Normal approximation
param_ci = model.confint(method="profile")    # Profile likelihood
# If you want full parameter posteriors
param_dist = model.confint(method="bayes")    # Posterior samples
```

## Approximate Bayes
### Laplace Approximation (Fastest)
### Variational Inference (Very Fast) Mean-field VI:
### Empirical Bayes (Hierarchical Models) For regularized regression: Treat penalties as priors, estimate hyperparameters
### Integrated Nested Laplace Approximation (INLA-style) For hierarchical/mixed models: Fast approximate marginals

```
Method                  Speed       Accuracy
─────────────────────────────────────────────
Laplace approximation   1.1x MLE    ~95% accurate
Variational inference   2-5x MLE    ~90% accurate  
Bootstrap (1000)        100x MLE    Very accurate
Full MCMC              1000x+ MLE   Gold standard
```

```
# User never thinks about the method - it's chosen automatically
model = sl.lm("mpg ~ hp + wt", data=mtcars)

# Parameter uncertainty (ultra-fast approximate Bayes by default)
model.confint()                    # Laplace approximation (fastest)
model.confint(method="exact")      # Bootstrap/asymptotic when precision needed
model.param_distribution()         # Samples from approximate posterior

# Prediction uncertainty (conformal)
model.predict(X_new)               # Conformal intervals

# The magic decision tree behind the scenes:
# - Laplace approximation: Linear/GLM with smooth likelihood
# - Variational inference: Complex models, non-conjugate priors  
# - Bootstrap: When approximate Bayes assumptions fail
# - Conformal: Always for prediction intervals
```

## Users get Bayesian-quality parameter uncertainty at frequentist speeds, completely automatically.

Laplace approximation is perfect because:

Nearly free with JAX autodiff (Hessian comes naturally)
Works great for GLMs, robust regression, most of your methods
Gives proper parameter distributions, not just intervals
Falls back gracefully to bootstrap or other when assumptions break

```
# User does this
model = sl.robust("y ~ x1 + x2", data=df)
model.param_distribution()

# You deliver this
# 1. Fit robust regression (MM-estimator) 
# 2. Compute Hessian of robust likelihood at optimum (JAX autodiff)
# 3. Return multivariate normal approximation to posterior
# 4. Fast, principled parameter uncertainty
```

## Flexmix is cool 

```
import sampleology_linear as sl

# Finite mixture of linear regressions (flexmix equivalent)
model = sl.mixture("mpg ~ hp + wt", data=mtcars)                    # Auto-select components
model = sl.mixture("mpg ~ hp + wt", data=mtcars, k=3)              # 3-component mixture
model = sl.mixture("mpg ~ hp + wt", data=mtcars, k="bic")          # BIC selection

# Mixture of GLMs
model = sl.mixture_glm("success ~ dose + age", data=trials, 
                      family="logistic", k=2)                      # 2-class mixture

# Different model per component (flexmix strength)
model = sl.mixture("mpg ~ hp + wt", data=mtcars,
                  models=["linear", "robust", "quantile"])         # Heterogeneous mixture

# Concomitant variable model (mixing probabilities depend on covariates)
model = sl.mixture("mpg ~ hp + wt", data=mtcars, 
                  concomitant="~ cyl + gear")                      # π_k = f(cyl, gear)
```

## LASSO things

```
sl.relaxed_lasso()
sl.group_lasso("y ~ .", data=df, groups=group_structure)           # Group LASSO
sl.sparse_group_lasso("y ~ .", data=df, groups=groups)             # Sparse + group penalties
sl.fused_lasso("y ~ x1 + x2 + x3", data=df)                       # Fused LASSO (smooth coefficients)
```

## Functional Regression

```
sl.functional("response ~ functional_covariate", data=df)          # Scalar-on-function
sl.concurrent("curve ~ x1 + x2", data=df)                         # Function-on-scalar  
sl.function_on_function("curve1 ~ curve2", data=df)               # Function-on-function
```

## Additive Models

```
sl.gam("y ~ s(x1) + s(x2) + x3", data=df)                        # Generalized additive models
sl.scam("y ~ s(x1, k=10, bs='mpi')", data=df)                    # Shape-constrained additive
sl.varying_coefficient("y ~ x1 * s(time)", data=panel)            # Time-varying coefficients
```

## SEM

```
sl.structural("y1 ~ x1 + x2; y2 ~ y1 + x3", data=df)             # Simultaneous equations
sl.mediation("outcome ~ treatment + mediator", data=df)           # Causal mediation analysis
```

## Modern robust methods:

```
sl.deepest_regression("y ~ x", data=df)                           # Regression based on data depth
sl.robust_ridge("y ~ .", data=df, robust=True)                    # Robust + regularized
sl.trimmed_lasso("y ~ .", data=df, trim=0.1)                      # Trimmed estimation + sparsity
sl.robust_pca_regression("y ~ .", data=df)                        # Robust PCA + regression
```

## Rank Based Methods:

```
sl.theil_sen("y ~ x", data=df)                                    # Theil-Sen estimator
sl.repeated_median("y ~ x", data=df)                              # Siegel repeated median
sl.rank_regression("y ~ x1 + x2", data=df)                       # General rank regression
```

## Interpretable ML:

```
sl.explainable_boosting("y ~ .", data=df)                         # EBM (additive + interactions)
sl.neural_additive("y ~ .", data=df)                              # NAM (neural additive models)
sl.tabnet("y ~ .", data=df)                                       # TabNet (attention-based)
```

## Time Series & Panel

```
sl.tvp_regression("y ~ x1 + x2", data=ts, time_varying=True)      # Time-varying parameters
sl.threshold_regression("y ~ x", data=ts, threshold_var="x")       # Threshold autoregression
sl.regime_switching("y ~ x", data=ts, regimes="auto")             # Markov regime switching
sl.cointegrating("y ~ x1 + x2", data=ts)                          # Cointegrating regression
```

## Panel data methods:
```
sl.interactive_fixed_effects("y ~ x", data=panel, 
                             factors="auto")                      # Interactive FE
sl.matrix_completion("y ~ x", data=panel)                         # Nuclear norm regularization
sl.dynamic_panel("y ~ lag(y) + x", data=panel)                   # System GMM
```

## Distributional & Extreme Value
Beyond location models:

```python
sl.expectile_regression("y ~ x", data=df, expectile=0.9)          # Expectile regression
sl.m_quantile("y ~ x", data=df, tau=0.5)                          # M-quantiles
sl.location_scale("y ~ x1 | x2", data=df)                        # GAMLSS-style
sl.skew_regression("y ~ x", data=df, distribution="skew_normal")   # Skewed distributions
```

## Extreme value regression:

```python
sl.extreme_quantile("y ~ x", data=df, tau=0.99)                   # Extreme quantiles
sl.peaks_over_threshold("y ~ x", data=df)                         # POT regression
sl.gev_regression("extreme_values ~ covariates", data=df)          # GEV with covariates
```

## Computational & Algorithmic

### Modern optimization:

```python
sl.admm_regression("y ~ .", data=df, penalty="elastic_net")       # ADMM algorithm
sl.proximal_gradient("y ~ .", data=df)                            # Proximal methods
sl.coordinate_descent_plus("y ~ .", data=df)                      # Enhanced coordinate descent
```

### Online/streaming:

```python
sl.online_regression("y ~ x", data=stream)                        # Online learning
sl.forgetting_factor("y ~ x", data=stream, lambda_=0.95)         # Exponential forgetting
sl.change_detection("y ~ x", data=stream)                         # Adaptive to concept drift
```

## The Unified Theme
CQRRPT: O(n) sketching instead of O(n³) full regression
iBOSS: Informative subsampling beats random sampling
Data thinning: Statistical theory obviates train/test splits
Conformal > Bootstrap: Exact guarantees with less computation
Empirical Bayes: Borrow strength across problems


------------------


# Tier 1: Perfect Alignment

## Make Big Problems Small

## Sketching & Dimensionality Reduction:

```python
sl.functional("response ~ functional_covariate", data=df)          # Infinite-dim → finite basis
sl.fused_lasso("y ~ x1 + x2 + x3", data=df)                       # Exploit smoothness structure
sl.group_lasso("y ~ .", data=df, groups=group_structure)           # p variables → k groups
sl.matrix_completion("y ~ x", data=panel)                         # Nuclear norm = low-rank structure
```
## Structural Exploitation:
```python
sl.interactive_fixed_effects("y ~ x", data=panel, factors="auto") # n×T panel → r factors (r<<min(n,T))
sl.trend_filtering("y ~ time", data=ts_data, degree=2)             # Exploit piecewise polynomial structure
```

## Make Small Problems Sufficient
Borrowing Strength:

```python
sl.expectile_regression("y ~ x", data=df, expectile=0.9)          # More efficient than quantiles
sl.m_quantile("y ~ x", data=df, tau=0.5)                          # Robust + efficient hybrid
sl.bayesian_neural("y ~ .", data=df)                              # Prior regularization for small n
```

# Tier 2: Strong Alignment
Algorithmic Efficiency:
```python
sl.admm_regression("y ~ .", data=df)                              # Distributed/parallel solving
sl.coordinate_descent_plus("y ~ .", data=df)                      # Exploit sparsity structure
sl.variational_sparse_gp("y ~ .", data=df)                        # O(m) instead of O(n³) GP
```

Smart Structural Assumptions:
```python
sl.threshold_regression("y ~ x", data=ts, threshold_var="x")       # Piecewise structure
sl.change_detection("y ~ x", data=stream)                         # Adapt to structure changes
sl.scad("y ~ .", data=df)                                          # Nearly unbiased with fewer variables
```

# Tier 3: Modest Alignment
These are good methods but don't fit core philosophy:

GAMs, neural networks (make small problems bigger)
Most causal methods (different complexity trade-off)
Standard robust methods (robustness ≠ efficiency)

## The Perfect Core Collection
"algorithmic efficiency" package should focus on:
Dimensionality Reduction via Structure
Functional regression - infinite to finite-dimensional
Group/fused LASSO - exploit coefficient structure
Matrix completion - low-rank panel data
Interactive fixed effects - factor structure in panels

## Efficient Robust Alternatives
Expectile regression - more efficient than quantiles
M-quantiles - robust + efficient hybrid
SCAD/MCP - sparse + unbiased

## Smart Algorithms

ADMM - parallel/distributed optimization
Trend filtering - exploit smoothness
Variational sparse GP - scalable nonparametrics

### Why This Collection is Coherent

Common theme: Each method exploits some structural assumption to make computation tractable:

Functional: Smooth functions live in low-dimensional spaces
Group LASSO: Coefficients have group structure
Matrix completion: Data has low-rank structure
Expectiles: More efficient location/scale modeling than quantiles
ADMM: Problems decompose into parallelizable subproblems

The Magical Implementation
```
model = sl.lm("y ~ many_variables", data=big_df)
```

# Your intelligence:
```
if n_features > 1000:
    # Detect group structure, use group LASSO
elif is_panel_data(data):
    # Use interactive fixed effects  
elif is_functional_data(data):
    # Use functional regression
elif small_n_large_p(data):
    # Use SCAD for nearly unbiased estimation
This gives you a coherent intellectual framework: You're not just building another regression package, you're building the "structural statistics" library - methods that exploit mathematical structure to make hard problems easy.
```
