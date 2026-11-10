"""Heteroskedasticity in a cross-section: Greene's credit card expenditure data.

Greene, *Econometric Analysis*, Example "Heteroscedastic regression" (72 cardholders with
positive spending; data bundled with statsmodels).  OLS of average monthly expenditure on age,
home ownership, income and income squared, with
  * R's summary(lm()) of the OLS fit (FormulaEstimator),
  * White's heteroskedasticity-consistent standard errors and the Breusch-Pagan test
    (statsmodels on statsformula's design matrices),
  * weighted least squares with weights 1 / income^2 (summary of the weighted lm).
"""
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_breuschpagan

import statsformula as sf
from statsformula.sklearn import FormulaEstimator

cc = sm.datasets.ccard.load_pandas().data          # AVGEXP AGE INCOME INCOMESQ OWNRENT
formula = "AVGEXP ~ AGE + OWNRENT + INCOME + I(INCOME^2)"

ols = FormulaEstimator(formula, LinearRegression(fit_intercept=False)).fit(cc)
print(ols.summary())

# White (HC0) and HC3 standard errors: statsmodels on the R design matrices
y, X = sf.dmatrices(formula, cc)
fit = sm.OLS(y, X).fit()
table = pd.DataFrame({"OLS se": fit.bse, "White (HC0) se": fit.get_robustcov_results("HC0").bse,
                      "HC3 se": fit.get_robustcov_results("HC3").bse}, index=X.columns)
print(table.round(3).to_string())
lm_stat, lm_p, _, _ = het_breuschpagan(fit.resid, X)
print("Breusch-Pagan LM = %.3f, p = %.4f" % (lm_stat, lm_p))

# weighted least squares, variance proportional to income^2
wls = FormulaEstimator(formula, LinearRegression(fit_intercept=False)).fit(cc, sample_weight=1 / cc["INCOME"] ** 2)
print(wls.summary().coefficients.round(4).to_string())
