"""Instrumental variables and simultaneous equations: Kmenta's food market and Klein's Model I.

An IV model is written as a two-part formula, as in R's ``ivreg(y ~ regressors | instruments)``
(the Formula package syntax): ``sf.model_matrix(F, data, rhs=1)`` is the regressor matrix and
``rhs=2`` the instrument matrix, with every R transformation available in both parts.

1. Kmenta (1986): supply and demand for food, 20 years.  Price and quantity are determined
   jointly, so OLS is inconsistent.  Two-stage least squares (2SLS) for each equation,
   instrument diagnostics, and the whole non-recursive system by full-information maximum
   likelihood with ``PathModel``.
2. Klein's Model I (Greene, *Econometric Analysis*, Table F15.1): 2SLS of the consumption,
   investment and private-wage equations with Greene's list of predetermined variables.

Data: ``sf.load_data("kmenta")`` and ``sf.load_data("klein")``, bundled with statsformula.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.sandbox.regression.gmm import IV2SLS

import statsformula as sf
from statsformula.sklearn import FormulaEstimator, PathModel

kmenta = sf.load_data("kmenta")
klein = sf.load_data("klein").dropna()     # 1921-1941 (the lags are missing in 1920)


def tsls(formula, data):
    """2SLS of a two-part formula ``y ~ regressors | instruments`` (statsmodels' IV2SLS)."""
    F = sf.Formula(formula)
    y = np.asarray(sf.model_part(F, sf.model_frame(F, data), lhs=1), dtype=float).ravel()
    X = sf.model_matrix(F, data, rhs=1)
    Z = sf.model_matrix(F, data, rhs=2)
    fit = IV2SLS(y, X, Z).fit()
    return pd.DataFrame({"Estimate": fit.params, "Std. Error": fit.bse, "t value": fit.tvalues,
                         "Pr(>|t|)": fit.pvalues}, index=X.columns), fit


def ols(formula, data):
    return FormulaEstimator(formula, LinearRegression(fit_intercept=False)).fit(data).summary()


# ---------------------------------------------------------------------------
# 1. Kmenta: demand  consump = a0 + a1 price + a2 income            + u
#            supply  consump = b0 + b1 price + b2 farmPrice + b3 trend + v
# ---------------------------------------------------------------------------
inst = "income + farmPrice + trend"
demand, demand_fit = tsls("consump ~ price + income | " + inst, kmenta)
supply, supply_fit = tsls("consump ~ price + farmPrice + trend | " + inst, kmenta)
print("Demand, 2SLS\n" + demand.round(4).to_string())
print("\nSupply, 2SLS\n" + supply.round(4).to_string())

compare = pd.DataFrame({
    "OLS": [ols("consump ~ price + income", kmenta).coefficients.loc["price", "Estimate"],
            ols("consump ~ price + farmPrice + trend", kmenta).coefficients.loc["price", "Estimate"]],
    "2SLS": [demand.loc["price", "Estimate"], supply.loc["price", "Estimate"]]},
    index=["demand: price slope", "supply: price slope"])
print("\n" + compare.round(4).to_string())

# instrument strength: first stage of the endogenous price on all exogenous variables
first = ols("price ~ income + farmPrice + trend", kmenta)
print("\nfirst stage: F = %.2f on (%d, %d) df, R2 = %.3f" % (*first.fstatistic, first.r_squared))

# endogeneity of price in the demand equation (Durbin-Wu-Hausman, control function)
kmenta["v_price"] = first.residuals.to_numpy()
dwh = ols("consump ~ price + income + v_price", kmenta).coefficients.loc["v_price"]
print("Durbin-Wu-Hausman: t = %.2f, p = %.4f" % (dwh["t value"], dwh["Pr(>|t|)"]))

# the system by full-information ML: price and quantity affect each other.  The supply
# equation is normalized on price; its structural price slope is 1 / (coefficient on consump).
fiml = PathModel("""
  consump ~ price + income                 # demand
  price   ~ consump + farmPrice + trend    # supply, normalized on price
  consump ~~ price
""").fit(kmenta)
co = fiml.coefficients()
print("\nFIML (%s, converged: %s)" % (fiml.estimation_, fiml.converged_))
print(co[co.op == "~"][["lhs", "rhs", "est", "se", "z", "pvalue"]].round(4).to_string(index=False))
b = co[(co.lhs == "price") & (co.rhs == "consump")]["est"].iloc[0]
print("supply price slope implied by FIML: %.4f" % (1 / b))

# ---------------------------------------------------------------------------
# 2. Klein's Model I, 2SLS (Greene)
# ---------------------------------------------------------------------------
pre = "govExp + taxes + govWage + trend + capitalLag + corpProfLag + gnpLag"
equations = {
    "consumption": "consump ~ corpProf + corpProfLag + wages",
    "investment": "invest ~ corpProf + corpProfLag + capitalLag",
    "private wages": "privWage ~ gnp + gnpLag + trend",
}
rows = []
for name, f in equations.items():
    tab, _ = tsls(f + " | " + pre, klein)
    o = ols(f, klein).coefficients
    for term in tab.index:
        rows.append({"equation": name, "term": term, "OLS": o.loc[term, "Estimate"],
                     "2SLS": tab.loc[term, "Estimate"], "2SLS se": tab.loc[term, "Std. Error"]})
print("\nKlein's Model I\n" + pd.DataFrame(rows).round(4).to_string(index=False))
