# Changelog

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
