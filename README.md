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

* 367 hand-written cases: `terms` attributes, `model.matrix` values, column names, `assign` and
  `contrasts`, `sparse.model.matrix`, `model.frame` (NA handling, subset, weights, offset,
  `drop.unused.levels`), `update`, `reformulate`, `drop.terms`, `delete.response`, the Formula
  package, prediction on new data, error messages and 41 mgcv smooth models;
* 500 randomly generated formulas compared with R's `terms()` (2,000 more checked during
  development);
* 400 randomly generated formulas compared with R's `model.matrix()`.

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
* **`gam.side()`** (side constraints mgcv adds when smooths share covariates, e.g.
  `s(x) + te(x, z)`) is not implemented: a warning is issued and the columns can differ from R.
  Use `ti()` for interaction terms (the mgcv recommendation), which needs no side constraints.
* mgcv bases not yet available: `t2`, `fs`, `gp`, `ds`, `so`, `mrf`, `ad`, `sos`, `sz`, `re`
  inside tensor products, smooths with matrix (summation convention) arguments; `id=`, `sp=` and
  `fx=` only affect penalties, which statsformula does not compute (it builds model matrices).
* R syntax that is not formula-related (`function`, `if`, assignment) is not supported inside
  formulas.
* By default, R's functions take precedence over Python objects of the same name (for example a
  Python `log` in your namespace does not replace R's `log`); use `sf.register_function(...,
  override=True)` to change that.

## License

GPL-2.0-or-later. statsformula contains translations of code from R (GPL-2 | GPL-3), mgcv
(GPL ≥ 2), Formula (GPL-2 | GPL-3) and Matrix (GPL ≥ 2), so it is distributed under the same
terms. The ported LAPACK routines are BSD-licensed.
