"""A path model (mediation) written as a multi-line formula: diabetes progression (scikit-learn).

Does body mass index act on disease progression through serum triglycerides (``ltg``)?
``PathModel`` fits the system like ``lavaan::sem()`` and prints lavaan's summary, with fit
measures, R-square and delta-method standard errors of the indirect and total effects.
"""
from sklearn.datasets import load_diabetes

from statsformula.sklearn import PathModel

df = load_diabetes(scaled=False, as_frame=True).frame
df.columns = ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu", "progression"]
df["sex"] = (df["sex"] == 2).astype(float)

model = """
  ltg         ~ bmi + age + sex          # mediator
  progression ~ ltg + bmi + bp           # outcome
"""
pm = PathModel(model).fit(df)
print(pm.summary(effects=True))

ind = pm.effects("indirect", se=True)
print(ind[(ind.lhs == "progression") & (ind.rhs == "bmi")].round(4).to_string(index=False))
print(pm.fit_measures()[["chisq", "df", "pvalue", "cfi", "rmsea", "srmr"]].round(4).to_string())
print(pm.predict(df.head(3)).round(2).to_string())
