# Formulas and design matrices

This chapter describes the formula language, how a formula is turned into terms and design
columns, and how the fitted design is applied to new data. The implementation follows R's C
code (`termsform` and `modelmatrix` in `src/library/stats/src/model.c`) step by step, so the
corner cases behave as in R, not as a re-specification of them would.

## From formula to design matrix

A formula goes through the same four stages as in R.

1. **Parsing.** The formula is parsed into an R expression tree with R's operator
   precedence (`~` < `+ -` < `* /` < `%in%` < `:` < `^`, unary minus, backtick-quoted names).
   `sf.formula()` returns an `RFormula` whose `str()` is R's `deparse()`.
2. **Terms** (`sf.terms`). Each term is a bit set over the *variables* of the formula
   (`attr(, "variables")`). `+` is union, `:` is pairwise union, `*` expands to
   `a + b + a:b`, `/` and `%in%` nest, `^k` forms all products up to order $k$, and `-` removes
   terms. Terms are deduplicated and sorted by order (unless `keep_order=True`). The result
   carries R's attributes: `factors` (the variable × term incidence matrix with codes
   0/1/2), `term_labels`, `order`, `intercept`, `response`, `offset` and `specials`.
3. **Model frame** (`sf.model_frame`). The variables are evaluated in the data (R
   functions, then user functions; see *Functions* below), `subset`, `weights` and
   `offset` are applied, and `na_action` (`na.omit`, `na.exclude`, `na.fail`, `na.pass`)
   drops incomplete rows jointly across the response and all predictors.
4. **Model matrix** (`sf.model_matrix`). Each term contributes the row-wise product of its
   variables' codings. A numeric variable contributes its column(s). A factor contributes
   either contrasts or full dummy (indicator) coding, according to the `factors` code of
   the term. This is the *Statistical Models in S* rule (Chambers & Hastie, 1992, §2.4.1):
   the factor is contrast-coded if the term obtained by deleting it is in the model,
   otherwise it gets indicator coding. Column names are built as in R (`fb`,
   `fb:log(x)`, `poly(z, 2)1`, `C(g, sum)1`), and `assign` maps every column to its term.

```python
X = sf.model_matrix("y ~ g:x + g:z", df)     # no main effects: g gets indicator coding
X.assign                                      # [0, 1, 1, 2, 2]
X.contrasts                                   # {'g': 'contr.treatment'}
```

## Stateful transformations and new data

Data-dependent transformations are fitted once and stored in `X.model_spec`:

* factor levels (`xlev`), contrasts, `drop.unused.levels`;
* `poly()` (the three-term-recurrence coefficients `alpha` and `norm2` of R's `poly`);
* `splines::ns()`/`bs()` (knots and boundary knots, LINPACK QR signs as in R);
* `scale()` (centre and scale);
* mgcv smooths (basis, constraint matrices, knots, `gam.side()` deletions).

`X.model_spec.get_model_matrix(new)` evaluates the design on new data with these stored
values, which is what R's `predict()` does through the `predvars` attribute of `terms`.
Values outside the training range are extrapolated exactly as R does (`ns()` linear beyond
the boundary knots; `bs()` polynomial continuation).

## Operators

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

## Functions available inside formulas

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

## Correspondence with R functions

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

## statsmodels and formulaic style helpers

```python
y, X = sf.dmatrices("y ~ x + C(g, sum)", df)        # pandas frames, NA rows removed jointly
import statsmodels.api as sm
fit = sm.OLS(y, X).fit()

X = sf.model_matrix("y ~ poly(x, 3) + ns(z, df = 4) + g", train)
X_test = X.model_spec.get_model_matrix(test)        # same transformation, levels and contrasts

S = sf.sparse_model_matrix("~ user * item", big_df)  # scipy.sparse
```

## Data types and R semantics

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

## Multi-part formulas

`sf.Formula` implements the *Formula* package (Zeileis & Croissant, 2010): several
responses and several right-hand sides separated by `|`. A common use is instrumental
variables, `y ~ x1 + x2 | z1 + z2 + x2`:

```python
F = sf.Formula("y ~ x1 + x2 | x2 + z1 + z2")
mf = sf.model_frame(F, df)                   # one model frame, NA removed over all parts
y = sf.model_part(F, mf, lhs=1)
X = sf.model_matrix(F, df, rhs=1)            # regressors
Z = sf.model_matrix(F, df, rhs=2)            # instruments
F.formula(rhs=2)                             # y ~ x2 + z1 + z2
```

## Sparse design matrices

`sf.sparse_model_matrix` returns the design as a `scipy.sparse.csc_matrix` wrapper with the
same columns as `Matrix::sparse.model.matrix`. This is useful for high-cardinality factors
and their interactions.
