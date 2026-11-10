# Quick start

## Formulas and design matrices

```python
import statsformula as sf

f = sf.formula("y ~ C(g, sum) * log(x) + poly(z, 2)")
sf.terms(f).term_labels          # ['C(g, sum)', 'log(x)', 'poly(z, 2)', 'C(g, sum):log(x)']

X = sf.model_matrix(f, df)       # pandas DataFrame subclass (ModelMatrix)
X.columns, X.assign, X.contrasts
X.model_spec.get_model_matrix(new_df)   # the same design on new data (factor levels, poly
                                        # coefficients, spline knots, smooth bases reused)

y, X = sf.dmatrices(f, df)       # response and design, rows with NA removed jointly
S = sf.sparse_model_matrix(f, df)        # scipy.sparse CSC, as Matrix::sparse.model.matrix
```

## Models with R summaries

```python
from sklearn.linear_model import LinearRegression
from statsformula.sklearn import FormulaEstimator, GAMRegressor, PathModel

FormulaEstimator("y ~ x * g", LinearRegression(fit_intercept=False)).fit(df).summary()  # summary.lm
sf.gam("y ~ g + s(x) + te(x, z)", df, method="REML").summary()                          # summary.gam
PathModel("m ~ x\ny ~ m + x").fit(df).summary()                                          # lavaan
```

## scikit-learn

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

search = GridSearchCV(FormulaEstimator(estimator=Ridge()),
                      {"formula": ["y ~ x + g", "y ~ ns(x, 4) * g"],
                       "estimator__alpha": [0.1, 1, 10]}, cv=5).fit(df)
```

Continue with the [user guide](guide/formulas.md) or the [examples](examples/index.md).
