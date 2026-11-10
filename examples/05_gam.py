"""Generalized additive models with mgcv's smooths: diabetes progression (scikit-learn).

``sf.gam`` reproduces ``mgcv::gam(method = "REML")`` (smoothing parameters, coefficients,
standard errors, ``summary.gam``); ``GAMRegressor`` is the same model as a scikit-learn
estimator, so it can be cross-validated against a linear model.
"""
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

import statsformula as sf
from statsformula.sklearn import FormulaEstimator, GAMRegressor

df = load_diabetes(scaled=False, as_frame=True).frame
df.columns = ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu", "progression"]
df["sex"] = pd.Categorical(df["sex"].astype(int).astype(str))

b = sf.gam("progression ~ sex + s(age) + s(bmi) + s(bp) + s(ltg) + s(hdl)", df, method="REML")
print(b.summary())

# predictions with standard errors (predict.gam(se.fit = TRUE))
new = pd.DataFrame({"sex": ["1", "2"], "age": [50, 50], "bmi": [22.0, 32.0], "bp": [90, 105],
                    "ltg": [4.4, 5.0], "hdl": [55, 40]})
print(b.predict(new, se_fit=True).round(2).to_string())

# cross-validated comparison with the linear model on the same variables
cv = KFold(5, shuffle=True, random_state=0)
lin = FormulaEstimator("progression ~ sex + age + bmi + bp + ltg + hdl", LinearRegression())
gam = GAMRegressor("progression ~ sex + s(age) + s(bmi) + s(bp) + s(ltg) + s(hdl)")
for name, est in [("linear", lin), ("GAM", gam)]:
    r2 = cross_val_score(est, df, cv=cv)
    print("%-7s CV R2: %.3f" % (name, r2.mean()))
