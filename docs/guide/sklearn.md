# scikit-learn integration

`statsformula.sklearn` provides three estimators that follow the scikit-learn API
(`get_params`/`set_params`, `clone`, pickling, `__sklearn_tags__`). They work with
`Pipeline`, `ColumnTransformer`, `cross_val_score`, `GridSearchCV` and metadata routing.

| class | role |
|---|---|
| {class}`~statsformula.sklearn.FormulaTransformer` | `X` → design matrix of a one-sided formula; a pipeline step |
| {class}`~statsformula.sklearn.FormulaEstimator` | builds `y` *and* `X` from a formula and fits any estimator on them |
| {class}`~statsformula.sklearn.GAMRegressor` | mgcv-equivalent penalized GAM (REML/GCV) |
| {class}`~statsformula.sklearn.PathModel` | a system of regressions written as a multi-line formula |

## Learned state and leakage

`fit` evaluates the formula on the training data and stores the resulting `ModelSpec`. This
holds the factor levels, contrasts, `poly` recurrence coefficients, spline knots, `scale`
centres and smooth bases. `transform`, `predict` and `score` re-apply the stored
specification. Within cross-validation every fold therefore learns its own transformations,
so the design never carries information from the held-out fold. Levels unseen during
training raise an error, as in R's `predict()`.

## Pipelines

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

pipe = make_pipeline(
    FormulaTransformer("~ sex + age + ns(bmi, 3) + bp + log(ltg)", remove_intercept=True),
    StandardScaler(),
    RidgeCV(alphas=np.logspace(-3, 3, 13)),
)
pipe.fit(train, train["progression"])
pipe[0].get_feature_names_out()      # R column names: 'sex2', 'ns(bmi, 3)1', 'log(ltg)', ...
```

`FormulaTransformer(na_action="na.pass")` keeps every row, because a transformer cannot
drop rows inside a pipeline. Use `FormulaEstimator` when rows with missing values should be
removed jointly from `y`, `X` and `sample_weight`.

## The formula as a hyper-parameter

`FormulaEstimator.formula` is an ordinary parameter, so model selection over formulas uses
the standard tools:

```python
search = GridSearchCV(
    FormulaEstimator(estimator=Ridge()),
    {"formula": ["y ~ x + g", "y ~ ns(x, 3) * g", "y ~ (x + z + g)^2"],
     "estimator__alpha": [0.01, 0.1, 1.0]},
    cv=KFold(5, shuffle=True, random_state=0),
).fit(train)
```

## Responses

| left-hand side | `y` passed to the estimator |
|---|---|
| `y`, `log(y)`, `I(y / 1000)` | 1-d numeric array |
| `cbind(y1, y2)`, `y1 + y2 ~` (multi-part: `y1 | y2 ~`) | 2-d array (multi-output) |
| factor / string / boolean | labels (`y_encoding="auto"` or `"labels"`), integer `"codes"`, or `"onehot"` |

If `fit(X, y)` receives an explicit `y`, only the right-hand side of the formula is used. This
is what scikit-learn scorers expect in `cross_val_score(est, df, df["y"], scoring=...)`.

## Intercepts

R's design includes the `(Intercept)` column. Combine it with
`LinearRegression(fit_intercept=False)`, or pass `remove_intercept=True` to the formula
object and let the estimator fit its own intercept. Penalized estimators such as `Ridge`
should not penalize the intercept, so use `remove_intercept=True` with them, or keep the
column and accept that it is penalized.
