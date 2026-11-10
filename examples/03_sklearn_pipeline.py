"""R formulas inside scikit-learn pipelines: diabetes progression (Efron et al. 2004, scikit-learn).

1. ``FormulaTransformer`` as the first step of a ``Pipeline`` (splines, logs, factors)
2. ``FormulaEstimator`` builds both y and X from the formula, so formulas can be tuned
   with ``GridSearchCV`` like any other hyper-parameter
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from statsformula.sklearn import FormulaEstimator, FormulaTransformer

df = load_diabetes(scaled=False, as_frame=True).frame
df.columns = ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu", "progression"]
df["sex"] = pd.Categorical(df["sex"].astype(int).astype(str))       # a two-level factor
train, test = train_test_split(df, test_size=0.25, random_state=0)
cv = KFold(5, shuffle=True, random_state=0)

# 1. the formula is the first pipeline step
pipe = make_pipeline(
    FormulaTransformer("~ sex + age + ns(bmi, 3) + bp + log(ltg) + hdl + poly(glu, 2)",
                       remove_intercept=True),
    StandardScaler(),
    RidgeCV(alphas=np.logspace(-3, 3, 13)),
)
r2 = cross_val_score(pipe, train, train["progression"], cv=cv)
print("pipeline CV R2: %.3f (+/- %.3f)" % (r2.mean(), r2.std()))
pipe.fit(train, train["progression"])
print("design columns:", list(pipe[0].get_feature_names_out()))
print("test R2: %.3f" % pipe.score(test, test["progression"]))

# 2. tune the formula and the penalty together
candidates = [
    "progression ~ bmi + bp + ltg",
    "progression ~ sex + age + bmi + bp + ltg + hdl + glu",
    "progression ~ sex + age + ns(bmi, 3) + bp + log(ltg) + hdl + poly(glu, 2)",
    "progression ~ (sex + bmi + bp + ltg)^2 + hdl",
]
search = GridSearchCV(FormulaEstimator(estimator=Ridge()),
                      {"formula": candidates, "estimator__alpha": [0.01, 0.1, 1.0, 10.0]}, cv=cv)
search.fit(train)
res = pd.DataFrame(search.cv_results_)
best = res.loc[res.groupby("param_formula")["mean_test_score"].idxmax(),
               ["param_formula", "param_estimator__alpha", "mean_test_score"]]
print(best.sort_values("mean_test_score", ascending=False).to_string(index=False))
print("best:", search.best_params_)
print("test R2 of the refitted best model: %.3f" % search.score(test))
