"""Panel data and seemingly unrelated regressions: Grunfeld's investment data, 1935-1954.

Greene, *Econometric Analysis*, Table F13.1: five US firms observed for 20 years
(``sf.load_data("grunfeld_greene")``, bundled with statsformula).  Investment is explained by
the firm's market value and its capital stock.
  1. pooled OLS vs. fixed effects (least squares dummy variables: one intercept per firm,
     and two-way with year effects -- R's plm "within" estimator), with the F test of
     equal firm intercepts;
  2. Zellner's SUR: one investment equation per firm with contemporaneously correlated
     disturbances, fitted by maximum likelihood as a PathModel (lavaan's sem() on the
     same system).
"""
import pandas as pd
from sklearn.linear_model import LinearRegression

import statsformula as sf
from statsformula.sklearn import FormulaEstimator, PathModel

g = sf.load_data("grunfeld_greene")


def lm(formula, data=g):
    return FormulaEstimator(formula, LinearRegression(fit_intercept=False)).fit(data).summary()


rows = {}
for name, f in [("pooled OLS", "invest ~ value + capital"),
                ("firm fixed effects", "invest ~ value + capital + firm"),
                ("two-way fixed effects", "invest ~ value + capital + firm + factor(year)")]:
    s = lm(f)
    c = s.coefficients
    rows[name] = {"value": c.loc["value", "Estimate"], "se(value)": c.loc["value", "Std. Error"],
                  "capital": c.loc["capital", "Estimate"], "se(capital)": c.loc["capital", "Std. Error"],
                  "sigma": s.sigma, "adj. R2": s.adj_r_squared, "resid. df": s.df[1]}
print(pd.DataFrame(rows).T.round(4).to_string())

# F test of equal firm intercepts (anova(pooled, fixed effects) in R)
p, fe = lm("invest ~ value + capital"), lm("invest ~ value + capital + firm")
rss_p, rss_fe = p.sigma ** 2 * p.df[1], fe.sigma ** 2 * fe.df[1]
F = ((rss_p - rss_fe) / (p.df[1] - fe.df[1])) / (rss_fe / fe.df[1])
print("F test of no firm effects: F(%d, %d) = %.2f" % (p.df[1] - fe.df[1], fe.df[1], F))

# 2. SUR: wide data, one equation per firm, all disturbance covariances free
abbr = {"General Motors": "GM", "Chrysler": "CH", "General Electric": "GE", "Westinghouse": "WE",
        "US Steel": "US"}
wide = g.assign(firm=g.firm.map(abbr)).pivot(index="year", columns="firm", values=["invest", "value", "capital"])
wide.columns = ["%s_%s" % (v, f) for v, f in wide.columns]
firms = list(abbr.values())
equations = ["invest_{0} ~ value_{0} + capital_{0}".format(f) for f in firms]
covariances = ["invest_%s ~~ %s" % (a, " + ".join("invest_" + b for b in firms[i + 1:]))
               for i, a in enumerate(firms[:-1])]
sur = PathModel("\n".join(equations + covariances)).fit(wide)
ols = PathModel("\n".join(equations)).fit(wide)

def est(m):
    return m.coefficients().query("op == '~'").set_index(["lhs", "rhs"])[["est", "se"]]


tab = pd.concat({"OLS": est(ols), "SUR (ML)": est(sur)}, axis=1)
print(tab.round(4).to_string())
print("log-likelihood: OLS (diagonal) %.3f, SUR %.3f;  LR test of a diagonal covariance: %.2f on 10 df"
      % (ols.fit_measures()["logl"], sur.fit_measures()["logl"],
         2 * (sur.fit_measures()["logl"] - ols.fit_measures()["logl"])))
