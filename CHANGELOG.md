# Changelog

## 0.6.0 (2026-11-09)

* `summary()` for every fitted model, printed like R and with the tables as DataFrames:
  * `FormulaEstimator.summary()`: R's `summary.lm` for `LinearRegression` (coefficient table with
    standard errors, t tests and significance stars, residual quantiles, residual standard error,
    R^2, adjusted R^2, F test; `logLik`/`AIC`/`BIC`), refitted with `lm`'s pivoted QR so aliased
    columns are `NA` as in R; weighted fits; one table per response for `cbind()` responses
    (`summary.mlm`). Other estimators get coefficients and the training score.
  * `GamFit.summary()` / `GAMRegressor.summary()`: the full `summary.gam`, now with the
    smooth-term table `s_table` (edf, Ref.df, F, p-value: Wood's tests from mgcv's `testStat`,
    and `reTest` for `re`, `cs`, `ts`, `cc` smooths), `pterms_table`, deviance explained, and
    R's printed layout. Includes a port of mgcv's Davies (1980) algorithm (`psum.chisq`).
  * `PathModel.summary()`: lavaan's `summary(fit, fit.measures = TRUE, rsquare = TRUE)` layout;
    new `PathModel.fit_measures()` (lavaan's `fitMeasures()`: chi-square, baseline model, CFI,
    TLI, log-likelihoods, AIC, BIC, SABIC, RMSEA with confidence interval and tests, SRMR) and
    `PathModel.rsquare()`; `summary(effects=True)` adds indirect/total effects. GAM equations
    show mgcv's smooth-term tests.
  * `SemFit.summary()` (SEM backends) in lavaan's layout; `fit_semopy()` now also returns
    semopy's standard errors, z tests and fit statistics (`SemFit.fit_stats`).
* Documentation: README rewritten around worked examples on real data that ship with
  scikit-learn and semopy (iris, diabetes, breast cancer, Holzinger-Swineford): design matrices,
  `summary(lm())`, scikit-learn pipelines and formula tuning with `GridSearchCV`, classification,
  GAMs, path analysis and CFA, with their actual output. Runnable scripts in `examples/`; the
  test suite runs them and the README code blocks.
* Econometrics and SEM examples (`examples/08`-`13`, README sections 8-13): heteroskedasticity-robust
  inference and WLS (Greene's credit card data), probit/logit (Spector & Mazzeo), the Longley
  NIST benchmark (certified values reproduced to 1e-13), panel fixed effects and Zellner's SUR
  (Grunfeld), instrumental variables with two-part formulas `y ~ x | z`, 2SLS, first-stage,
  Durbin-Wu-Hausman and Sargan tests, FIML of a non-recursive supply/demand system (Kmenta),
  2SLS of Klein's Model I, and Bollen's Political Democracy SEM with equality constraints.
* Documentation site (Sphinx, `docs/`): a technical user guide with the estimators in
  mathematical form (summary.lm, penalized least squares, REML/GCV criteria, `Vp`, Wood's
  smooth-term tests and Davies' algorithm, the path-model likelihood, expected information,
  FIML, delta-method effects, lavaan fit measures), the validation methodology and
  tolerances, an API reference generated from the docstrings, and an examples gallery built
  by running `examples/`. `pip install "statsformula[docs]"`; the `docs` workflow builds it
  with warnings as errors and can publish it to GitHub Pages.
* `sf.load_data()`: bundled textbook datasets not available in other Python packages:
  `kmenta` (Kmenta 1986), `klein` (Klein's Model I, Greene Table F15.1), `grunfeld_greene`
  (Greene Table F13.1), with descriptions and sources in `DataFrame.attrs`.
* Path models: the singularity check of the information matrix is now scale-free, so models
  with variables in very different units (e.g. SUR on Grunfeld's data) get standard errors
  instead of a spurious "not identified" warning.
* Generic `FormulaEstimator.summary()` labels the estimator's own intercept `(Intercept)` when
  the design has none.
* Validated against R 4.5.1 / mgcv 1.9-3 / lavaan 0.7.3: 12 `summary.lm` and 18 `summary.gam`
  fits (numbers and printed output), 7 path models x 2 information types (20 fit measures, R^2,
  printed parameter tables), Davies' algorithm on 60 cases, `format`/`format.pval`/`printCoefmat`.

## 0.5.0 (2026-11-09)

* Path-model inference (`PathModel(se=...)`): `coefficients()` now returns lavaan's
  `parameterEstimates()` columns (`est`, `se`, `z`, `pvalue`, `ci.lower`, `ci.upper`) including
  residual (co)variances (`~~`) in lavaan's row order; `se="expected"` (default) / `"observed"`
  match `lavaan::sem(information = ...)`, `se="lm"` matches `summary(lm())`; `vcov()`; `level=`.
* `effects(kind, se=True)`: direct, indirect and total effects with delta-method standard errors
  (lavaan's `:=`).
* Non-recursive path models (feedback loops) and residual covariances (`y1 ~~ y2`) are fitted by
  full-information maximum likelihood (`estimation="auto"|"ML"|"OLS"`); `predict()` uses the
  reduced form `(I - B)^-1 G x`. Validated against lavaan 0.7.3 on 7 models (two- and
  three-variable loops, correlated disturbances, SUR) for both information types.
* `parse_path_system()`: equations and residual covariances of a path model.
* GAM standard errors: `GamFit.predict(newdata, se_fit=True)` (`predict.gam(se.fit = TRUE)`),
  `GamFit.summary()` (`p.table`, `r.sq`, edf per smooth), `GamFit.residual_df`,
  `GAMRegressor.predict(X, return_std=True)`; validated against mgcv on all 18 GAM fits.
* `gam()` warns when the penalized design is rank deficient (e.g. `y ~ x + s(x)`).
* Behaviour change: non-recursive `PathModel`s were fitted by OLS per equation in 0.4; they are
  now fitted by ML (`estimation="OLS"` restores the old fit).

## 0.4.0 (2026-11-09)

* Path models: `statsformula.sklearn.PathModel(model, estimator)` fits a system of regressions
  written as a multi-line formula (one `response ~ predictors` R formula per line; comments,
  continuation lines, `;`, lavaan-style `y1 + y2 ~ x`, repeated responses merged).
  * Equation-by-equation fit with any scikit-learn estimator (or one per response); default
    `lm`-equivalent, `GAMRegressor` for equations with smooths.
  * `predict()` propagates predictions through the system (reduced form), or uses observed
    predictors (`predict_mode="direct"`); `score()`, `coefficients()`,
    `effects("direct"|"indirect"|"total")`; `clone`/`GridSearchCV`/pickle compatible.
  * Validated against R: `lm()` per equation and propagated predictions on 8 models, and
    `lavaan::sem()` estimates and total effects (uncorrelated disturbances) on 5 of them.
* `parse_path_model()` / `PathEquation`.
* SEM backends (`statsformula.sem`): `sem_data()` rewrites R-expression terms of a lavaan model
  into columns (stateful `transform()`); `fit_semopy()` / `to_semopy()`, `fit_pysem()` /
  `to_pysem()` (twoertwein/pySEM, LISREL translation, L-BFGS fit) and `fit_pysemforge()` /
  `to_pysemforge()` fit lavaan's full model and return lavaan-style estimates with `:=`
  parameters. Validated against lavaan 0.7.3 on 7 models. Works around a semopy 2.3 infinite
  loop for factors without a fixed loading; warns about pysemforge 0.1.0's random refit and
  zeroed fit indices. New extra: `statsformula[sem]` (semopy).

## 0.3.0 (2026-11-09)

* scikit-learn integration (`statsformula.sklearn`, `pip install statsformula[sklearn]`):
  * `FormulaTransformer`: a formula's design matrix as a pipeline step (stateful: factor levels,
    contrasts, `poly`/`ns`/`bs`/`scale`, mgcv smooths learned on `fit`); pandas, numpy or sparse output;
    works with `Pipeline`, `ColumnTransformer` and `set_output`.
  * `FormulaEstimator(formula, estimator)`: builds both `Y` (`y`, `log(y)`, `cbind(y1, y2)`,
    `y1 | y2 ~ ...`, categorical responses) and `X` from the formula and fits any estimator or
    pipeline; compatible with `clone`, `GridSearchCV` (including searching over formulas),
    `cross_val_score`, sample weights and pickling.
  * `design_matrices(formula, data)`: aligned `(Y, X)` with multi-column responses.
* mgcv smoothing penalties: the design matrix of a GAM carries `model_spec.penalties`
  (`GamPenalty`: `S`, column offset, rank and name, as `G$S`/`G$off`/`G$rank` of
  `gam(..., fit = FALSE)`); `penalty_matrix(penalties, p, sp)` builds `sum_j sp_j S_j`;
  `FormulaTransformer.penalties_` / `.penalty_matrix()` and `FormulaEstimator.penalties_` are
  aligned with the output columns.
* `gam(formula, data, method="REML"|"GCV.Cp", sp=, weights=, gamma=)`: Gaussian penalized
  regression with smoothing parameters estimated like `mgcv::gam` (validated on 18 fits:
  sp, coefficients, fitted values, predictions, edf, scale); `GAMRegressor` is its
  scikit-learn estimator (multi-response `cbind`, `sp` tunable by `GridSearchCV`).
* Fix: `gam.side()` now also removes the deleted columns from the smooths' penalties (and
  drops penalties left empty), as mgcv does.
* Linear-model coefficients of `FormulaEstimator(..., LinearRegression(fit_intercept=False))`
  validated against R's `lm()` (including multi-response `mlm` fits).

## 0.2.0 (2026-11-09)

* lavaan model syntax: `lavaan_parse()` (port of `lavParseModelString()`), `lavaanify()`
  (port of `lavaanify()`, with `preset="cfa"|"sem"|"growth"`), `lavaan_partable()`, `lavaan_names()`;
  validated against lavaan 0.7.3 on 132 model strings and 317 option combinations.
* mgcv `gam.side()`: identifiability side constraints for smooths sharing covariates
  (`s(x) + te(x, z)`, `s(x) + s(x, z)`, ...), including prediction on new data.
* mgcv penalty matrices for all supported bases (`smooth_con(...)[i].S`), including the
  `bs` basis derivative penalty; validated against R's `smoothCon()`.
* Removed the "gam.side not implemented" warning.

## 0.1.0 (2026-11-09)

First release.

* Parser and deparser for R formulas (R operator precedence, `**` → `^`, backticks).
* Port of R 4.5 `termsform`: all formula operators, `.` expansion, offsets, specials, keep.order, simplify.
* `model.frame` (subset, weights, offset, na.action, drop.unused.levels, xlev) and `model.matrix`
  (port of `modelmatrix`: contrasts, column names, assign attribute).
* `sparse.model.matrix` (Matrix package conventions), `model.response/offset/weights/extract`.
* `update`, `reformulate`, `drop.terms`, `delete.response`, `all.vars`, `all.names`, `get_all_vars`.
* Formula package: multi-part formulas, `model.part`, `as.Formula`, `update.Formula`.
* R functions inside formulas, including `poly`, `bs`, `ns`, `scale`, `C`, `cut`, `interaction`, `relevel`.
* mgcv `s()`, `te()`, `ti()` with `tp`, `ts`, `cr`, `cs`, `cc`, `ps`, `cp`, `bs`, `re` bases and `by` variables.
* Stateful transforms re-applied to new data (`ModelSpec`, `GamSpec`).
* Validated against R 4.5.1: 367 reference cases, 500 random `terms()` and 400 random `model.matrix()` formulas.
