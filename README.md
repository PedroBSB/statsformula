# statsformula

**R model formulas for Python — the same terms, columns, names and numbers as R.**

`statsformula` re-implements R's formula machinery (`terms`, `model.frame`,
`model.matrix`, `update`, ...), the [Formula](https://cran.r-project.org/package=Formula)
package (multi-part formulas), `Matrix::sparse.model.matrix`, and the model-matrix side of
[mgcv](https://cran.r-project.org/package=mgcv) (`s()`, `te()`, `ti()` smooths). It is a port of
R's own algorithms (`termsform` and `modelmatrix` from `stats/src/model.c`, mgcv's `smoothCon`,
`PredictMat` and thin plate spline code, LINPACK/LAPACK numerics), and it is tested against R.

```python
import pandas as pd
import statsformula as sf

df = pd.read_csv("data.csv")

X = sf.model_matrix("Y ~ C(X1):C(X2) + log(X3)/sqrt(X4) + X5**4", df)
X.columns          # identical to colnames(model.matrix(...)) in R
X.assign           # attr(, "assign")
X.contrasts        # attr(, "contrasts")

sf.formula("Y~C(X1):C(X2)+log(X3)/sqrt(X4)+X5**4")
# <formula: Y ~ C(X1):C(X2) + log(X3)/sqrt(X4) + X5^4>     (R prints exactly this)
```

## Why

`patsy` and `formulaic` are *inspired by* R formulas but differ in many details (operator
semantics, contrast coding of interactions without main effects, column names, `C()`, `poly()`
and spline bases, `.` expansion, `-` / `0` / `1` handling, ...). `statsformula` aims for
**bit-for-bit agreement with R** so that models, coefficients and code can move between R and
Python unchanged. Its API borrows from statsmodels (`dmatrices`) and formulaic (`ModelMatrix`,
`ModelSpec` for applying a fitted design to new data).

## Validation against R

The test-suite compares against results produced by R 4.5.1 (mgcv 1.9-3, Formula 1.2-5,
Matrix 1.7-3); the fixtures are in `tests/fixtures` and can be regenerated with
`python tools/make_fixtures.py`, `tools/make_fuzz.py` and `tools/make_fuzz_mm.py` (needs R).

* 407 hand-written cases: `terms` attributes, `model.matrix` values, column names, `assign` and
  `contrasts`, `sparse.model.matrix`, `model.frame` (NA handling, subset, weights, offset,
  `drop.unused.levels`), `update`, `reformulate`, `drop.terms`, `delete.response`, the Formula
  package, prediction on new data, error messages, 57 mgcv models (16 of them needing
  `gam.side()` side constraints) and 24 mgcv penalty matrices;
* 500 randomly generated formulas compared with R's `terms()` (2,000 more checked during
  development);
* 400 randomly generated formulas compared with R's `model.matrix()`;
* penalty matrices of every mgcv model (`G$S`, `G$off`, `G$rank`, smoothing-parameter names) and
  18 penalized GAM fits (`gam(method = "REML" / "GCV.Cp")`, fixed `sp`, weights, `gamma`):
  smoothing parameters, coefficients, fitted values, predictions, edf and scale;
* 132 lavaan model strings and 353 option/preset combinations compared with lavaan 0.7.3
  (`tools/make_lavaan_fixtures.py`).

Numbers agree to ~1e-12 relative error; names, orderings and attributes are identical.

## Installation

```
pip install statsformula
```

Requires Python ≥ 3.10, numpy, pandas and scipy. R is **not** needed.

## What is supported

### Formula operators (exact R semantics)

| syntax | meaning |
|---|---|
| `a + b` | both terms |
| `a:b` | interaction |
| `a*b` | `a + b + a:b` |
| `a/b` | nesting, `a + a:b` |
| `b %in% a` | `b` nested in `a` |
| `(a + b + c)^2` | all interactions up to order 2 (`**` is accepted and printed as `^`) |
| `- b`, `- 1`, `+ 0`, `0 +` | remove terms / intercept |
| `.` | all other columns of `data` |
| `offset(x)` | offset terms |
| `I(expr)` | arithmetic inside formulas |
| `cbind(y1, y2) ~ x` | matrix responses |
| `y1 | y2 ~ x | z` | multi-part formulas (Formula package) |
| `` `my var` `` | non-syntactic names |

R's corner cases are reproduced too (`y ~ x^1` is an error, `(-a * b)` drops everything,
`y ~ 0/x` drops `x`, the response on the right-hand side, terms deduplication and ordering,
the factor coding heuristic of *Statistical Models in S* for interactions without main effects, ...).

### Functions available inside formulas

* **math:** `log(x, base)`, `log2`, `log10`, `log1p`, `exp`, `expm1`, `sqrt`, `abs`, `sign`,
  `floor`, `ceiling`, `round`, `signif`, `trunc`, trigonometric and hyperbolic functions,
  `gamma`, `lgamma`, `beta`, `choose`, `factorial`, `pmin`, `pmax`, `cumsum`, ...
* **operators:** `+ - * / ^ %% %/%`, comparisons, `& | ! xor`, `%in%`, `ifelse`, `is.na`
* **summaries:** `mean`, `median`, `sd`, `var`, `sum`, `min`, `max`, `quantile`, `length`
* **factors:** `factor`, `as.factor`, `ordered`, `relevel`, `droplevels`, `interaction`, `cut`,
  `C(f, contr, how.many)`
* **contrasts:** `contr.treatment`, `contr.sum`, `contr.helmert`, `contr.poly`, `contr.SAS`
  (via `C()`, the `contrasts=` argument or `sf.options(contrasts=...)`)
* **bases:** `poly(x, degree, raw=)`, multivariate `poly(x, z, degree=)`, `splines::bs`,
  `splines::ns` (exactly as R, including extrapolation), `scale`, `cbind`
* **mgcv smooths:** `s()`, `te()`, `ti()` with bases `tp`, `ts`, `cr`, `cs`, `cc`, `ps`, `cp`,
  `bs`, `re`; numeric, factor and ordered `by=`; `k`, `m`, `bs`, `d`, `knots`
* **conversions:** `as.numeric`, `as.integer`, `as.character`, `as.logical`, `c`, `rep`, `seq`
* **your own Python functions:** any function or object visible where you call statsformula
  (e.g. `center(x)`, `np.log1p(x)`), or registered with `sf.register_function()`.

Data-dependent transformations (`poly`, `bs`, `ns`, `scale`, factor levels, contrasts, mgcv
smooths) are remembered and re-applied to new data, like R's `predict()`.

### R functions

| R | statsformula |
|---|---|
| `formula()`, `as.formula()` | `sf.formula()` → `RFormula` |
| `terms(f, data, keep.order, specials, simplify)` | `sf.terms()` → `Terms` (`.term_labels`, `.factors`, `.order`, `.intercept`, `.response`, `.offset`, `.variables`, `.specials`) |
| `model.frame(f, data, subset, na.action, weights, offset, drop.unused.levels, xlev)` | `sf.model_frame()` |
| `model.matrix(f, data, contrasts.arg, xlev)` | `sf.model_matrix()` |
| `Matrix::sparse.model.matrix()` | `sf.sparse_model_matrix()` (scipy.sparse CSC) |
| `model.response`, `model.offset`, `model.weights`, `model.extract` | same names with `_` |
| `update()` | `sf.update()` (formulas and Formulas) |
| `reformulate`, `drop.terms`, `delete.response`, `all.vars`, `all.names`, `get_all_vars`, `deparse` | same names with `_` |
| `Formula::Formula`, `as.Formula`, `model.part`, `formula(F, lhs, rhs, collapse)` | `sf.Formula`, `sf.as_Formula`, `sf.model_part`, `F.formula(lhs=, rhs=, collapse=)` |
| `mgcv::interpret.gam`, `smoothCon`, `PredictMat`, `model.matrix(gam(...))`, `predict(gam, type = "lpmatrix")` | `sf.interpret_gam`, `sf.smooth_con`, `sf.predict_mat`, `sf.gam_model_matrix`, `X.model_spec.get_model_matrix(new)` |
| `options(contrasts = ...)`, `options(na.action = ...)` | `sf.options(contrasts=..., na_action=...)` |

### Python conveniences (statsmodels / formulaic style)

```python
y, X = sf.dmatrices("y ~ x + C(g, sum)", df)        # pandas frames, NA rows removed jointly
import statsmodels.api as sm
fit = sm.OLS(y, X).fit()

X = sf.model_matrix("y ~ poly(x, 3) + ns(z, df = 4) + g", train)
X_test = X.model_spec.get_model_matrix(test)        # same transformation, levels and contrasts

S = sf.sparse_model_matrix("~ user * item", big_df)  # scipy.sparse
```

### scikit-learn pipelines

`pip install statsformula[sklearn]`

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV
from statsformula.sklearn import FormulaTransformer, FormulaEstimator, design_matrices

# 1. the formula as a transformer step (X only)
pipe = make_pipeline(FormulaTransformer("~ log(x) + ns(z, 3) + C(g, sum)", remove_intercept=True),
                     StandardScaler(), Ridge())
pipe.fit(train, train["y"]).predict(test)

# 2. the formula builds Y *and* X; Y can have several columns
model = FormulaEstimator("cbind(y1, y2) ~ log(x) + poly(z, 2) + g", Ridge())
model.fit(train).predict(test)                  # (n, 2) predictions
FormulaEstimator("cls ~ x + g", LogisticRegression()).fit(train).predict_proba(test)

# hyper-parameters *and* formulas can be tuned
GridSearchCV(model, {"formula": ["cbind(y1, y2) ~ x + g", "cbind(y1, y2) ~ log(x) * g"],
                     "estimator__alpha": [0.1, 1, 10]}, cv=5).fit(train)

Y, X = design_matrices("y1 | y2 ~ log(x) + g", train)   # aligned response and design matrices
```

A scikit-learn `Pipeline` step can only transform `X`, so building the response from the
formula needs the `FormulaEstimator` wrapper. All data-dependent transformations (factor
levels, contrasts, `poly`, `ns`, `bs`, `scale`, mgcv smooths) are learned on `fit` and reused on
`predict`/`transform`. Responses: `y`, transformed `log(y)`, `cbind(y1, y2, ...)`, multi-part
`y1 | y2 ~ ...`; categorical responses as labels (for classifiers), integer codes or one-hot
(`y_encoding=`). Rows with missing values are dropped jointly from `Y`, `X` and `sample_weight`
when fitting; `transform` keeps every row (`na_action="na.pass"`). Custom Python functions used
inside the formula are passed with `env={"name": func}`.

### Smoothing penalties and penalized GAMs (mgcv)

Every mgcv design matrix carries its wiggliness penalties, exactly as `gam(..., fit = FALSE)`
returns them (`G$S`, `G$off`, `G$rank`, names of `G$sp`), so you can regularize the smooths
yourself or let statsformula estimate the smoothing parameters like `mgcv::gam`:

```python
import statsformula as sf
from statsformula.sklearn import FormulaTransformer, GAMRegressor

X = sf.model_matrix("y ~ g + s(x) + te(x, z)", df)
X.model_spec.penalties        # [<GamPenalty s(x): 9x9 at column 2, rank 8>, te(x,z)1, te(x,z)2]
P = sf.penalty_matrix(X.model_spec.penalties, X.shape[1], sp=[0.5, 1.0, 2.0])  # sum_j sp_j S_j
beta = np.linalg.solve(X.T @ X + P, X.T @ y)                                    # generalized ridge

b = sf.gam("y ~ g + s(x) + te(x, z)", df, method="REML")   # or "GCV.Cp"; sp=[...] to fix them
b.sp, b.coefficients, b.edf.sum(), b.scale, b.predict(new)  # same numbers as mgcv

tr = FormulaTransformer("~ g + s(x) + s(z)").fit(df)        # in a pipeline
tr.penalties_, tr.penalty_matrix(sp=[1.0, 3.0])             # aligned with tr.transform() columns

GAMRegressor("cbind(y1, y2) ~ g + s(x) + te(x, z)", method="REML").fit(df).predict(new)
GridSearchCV(GAMRegressor("y ~ s(x) + s(z)"), {"sp": [[0.1, 0.1], [1, 1], [10, 10]]})
```

The penalized fit minimizes `||y - X b||^2_W + sum_j sp_j b' S_j b`. Smoothing parameters are
estimated by REML (scale profiled out) or GCV with exact gradients, and match mgcv's to ~1e-3
on the log scale; when mgcv sends a smoothing parameter to infinity (a smooth reduced to its
null space), both stop at an arbitrary large value and the fitted values agree to ~1e-5.
Only the Gaussian family with identity link is fitted; for other models use `penalties_` with
your own solver.

### lavaan model syntax (structural equation models)

`sf.lavaan_parse()` is a port of `lavaan::lavParseModelString(model, as.data.frame. = TRUE)`
and `sf.lavaanify()` of `lavaan::lavaanify()` (lavaan 0.7.3), validated against R on 132 model
strings (parser output, 317 `lavaanify()` option combinations, 36 `cfa()`/`sem()`/`growth()`
parameter tables).

```python
model = """
  # measurement model
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + a*y2 + b*y3 + c*y4
  # regressions
    dem60 ~ ind60
  # residual correlations and a defined parameter
    y1 ~~ y2
    ab := a*b
"""
flat = sf.lavaan_parse(model)            # lhs, op, rhs, block, label, ..., mod.idx
flat.attrs["constraints"]                # [{'op': ':=', 'lhs': 'ab', 'rhs': 'a*b', 'user': 1}]
pt = sf.lavaanify(model, preset="sem")   # the parameter table lavaan::sem() builds
pt = sf.lavaanify(model, auto_var=True, std_lv=True, meanstructure=True)   # or "std.lv" style keys
sf.lavaan_names(model, "ov.x")           # like lavNames()
```

Supported: all operators (`=~ ~ ~~ ~1 | ~*~ <~ := == < > :~ %`), comments (`#`, `!`), `;`,
line continuation, modifiers (`a*x`, `0.5*x`, `NA*x`, `start()`, `0.5?x`, `label()`,
`equal()`, `lower()`, `upper()`, `c()` vectors for multiple groups, `efa("set")*f` blocks,
combined modifiers), simple bounds on labels (`a > 0` becomes a lower bound, as in lavaan),
interactions `x1:x2`, `group:`/`level:`/`class:`/`block:` blocks, and lavaan's error messages
and warnings (with line/position). `lavaanify()` options: `meanstructure`, `int.ov.free`,
`int.lv.free`, `marker.int.zero`, `orthogonal(.x/.y/.efa)`, `std.lv`, `correlation`,
`composites`, `conditional.x`, `fixed.x`, `parameterization`, `constraints`, `ceq.simple`,
`auto`, `model.type`, `auto.fix.first`, `marker`, `auto.fix.single`, `auto.var`,
`auto.cov.lv.x`, `auto.cov.x`, `auto.cov.y`, `auto.th`, `auto.delta`, `auto.efa`, `ngroups`,
`nthresholds`, `group.equal`, `group.partial`, `group.w.free`. Not supported: `effect.coding`,
the `:~` operator inside `lavaanify()`, `conditional.x` with `level:` blocks, and options that
need data (`varTable`: ordered variables are only detected from `|` thresholds in the syntax;
exogenous covariates are detected from the syntax, as `lavaanify()` does).

## Data types

| pandas / numpy | R equivalent |
|---|---|
| `Categorical` (ordered or not) | factor / ordered factor (categories are the levels) |
| `bool` / nullable `boolean` | logical |
| integer | integer |
| float | double (`NaN` is `NA`) |
| strings / `object` | character (converted to factors by `model.matrix`, as in R) |

String levels are sorted the way R sorts them under ICU collation (macOS/Windows R, and Linux
R built with ICU). Use `sf.set_collation("C")` to mimic `LC_COLLATE=C`. Unlike R, `C()` and
`relevel()` also accept character vectors (R would require a factor).

## Differences from R and limitations

* **Thin plate splines (`tp`, `ts`).** mgcv's basis depends on the eigenvector signs from
  its Lanczos solver, which come from reference LAPACK. statsformula reimplements mgcv's Lanczos
  iteration, LAPACK `DSTEQR`/`DSTEDC` (including divide and conquer) and R's random number
  generator (used for knot sampling above 2,000 unique points), so the basis matches R's
  reference LAPACK build exactly. R linked against a vendor LAPACK (OpenBLAS, MKL, Accelerate)
  can itself produce a different, equally valid sign convention.
* **`gam.side()`** side constraints (mgcv drops columns when smooths share covariates, e.g.
  `s(x) + te(x, z)`) are applied exactly as mgcv does, using the smooths' penalty matrices.
  Penalties match R's `smoothCon(...)$S`, except the `cs` shrinkage penalty: mgcv assigns
  different values to the two null-space eigenvalues, and the null-space basis returned by
  LAPACK's eigen-solver is arbitrary, so that penalty agrees with R only to ~1e-3. Model matrices
  are not affected.
* mgcv bases not yet available: `t2`, `fs`, `gp`, `ds`, `so`, `mrf`, `ad`, `sos`, `sz`, `re`
  inside tensor products, smooths with matrix (summation convention) arguments; `id=` and `sp=`
  inside `s()` are accepted but ignored (pass `sp=` to `gam()`/`GAMRegressor` instead).
* `gam()`/`GAMRegressor` fit Gaussian identity-link models only (no `family=`, `select=`,
  `paraPen=`); offsets are subtracted when fitting but not added by `predict()`.
* R syntax that is not formula-related (`function`, `if`, assignment) is not supported inside
  formulas.
* By default, R's functions take precedence over Python objects of the same name (for example a
  Python `log` in your namespace does not replace R's `log`); use `sf.register_function(...,
  override=True)` to change that.

## License

GPL-2.0-or-later. statsformula contains translations of code from R (GPL-2 | GPL-3), mgcv
(GPL ≥ 2), Formula (GPL-2 | GPL-3) and Matrix (GPL ≥ 2), so it is distributed under the same
terms. The ported LAPACK routines are BSD-licensed.
