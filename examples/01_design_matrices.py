"""R design matrices from a pandas DataFrame: Fisher's iris data (bundled with scikit-learn).

The column names, ``assign`` and ``contrasts`` attributes and values are those of R's
``model.matrix(Sepal.Length ~ Species * Petal.Length + poly(Sepal.Width, 2), iris)``.
"""
import pandas as pd
from sklearn.datasets import load_iris

import statsformula as sf

raw = load_iris(as_frame=True)
iris = raw.frame.drop(columns="target")
iris.columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]   # R's names
iris["Species"] = pd.Categorical.from_codes(raw.target, ["setosa", "versicolor", "virginica"])

f = sf.formula("Sepal.Length ~ Species * Petal.Length + poly(Sepal.Width, 2)")
print(f)
print(sf.terms(f).term_labels)

y, X = sf.dmatrices(f, iris)
print(X.shape)
print(list(X.columns))
print("assign:", X.assign, " contrasts:", X.contrasts)
print(X.head(3).round(4).to_string())

# the fitted design (factor levels, the orthogonal polynomial) is re-applied to new data
new = pd.DataFrame({"Species": ["virginica", "setosa"], "Petal.Length": [5.1, 1.4],
                    "Sepal.Width": [3.0, 3.5]})
print(X.model_spec.get_model_matrix(new).round(4).to_string())

# other contrasts, as in R
print(list(sf.model_matrix("~ C(Species, sum) + Petal.Width", iris).columns))
