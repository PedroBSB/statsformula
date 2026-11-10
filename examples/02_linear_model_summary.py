"""summary(lm(...)) in Python: a formula, scikit-learn's LinearRegression, and R's summary.

Data: Fisher's iris data (scikit-learn).  The printed table is identical to R's
``summary(lm(Sepal.Length ~ Petal.Length * Species, data = iris))``.
"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

from statsformula.sklearn import FormulaEstimator

raw = load_iris(as_frame=True)
iris = raw.frame.drop(columns="target")
iris.columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
iris["Species"] = pd.Categorical.from_codes(raw.target, ["setosa", "versicolor", "virginica"])

model = FormulaEstimator("Sepal.Length ~ Petal.Length * Species", LinearRegression(fit_intercept=False))
model.fit(iris)
s = model.summary()
print(s)

# the pieces are DataFrames / numbers
print(s.coefficients.round(4).to_string())
print("R2 = %.4f, adj. R2 = %.4f, AIC = %.2f, BIC = %.2f" % (s.r_squared, s.adj_r_squared, s.aic, s.bic))
