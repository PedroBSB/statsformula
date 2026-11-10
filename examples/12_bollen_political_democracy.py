"""Bollen's (1989) Political Democracy model: industrialization and democracy in 75 countries.

The structural equation model of Bollen, *Structural Equations with Latent Variables*
(chapter 8), the standard example of lavaan.  Indicators: x1-x3 (industrialization 1960),
y1-y4 (democracy 1960), y5-y8 (democracy 1965); the same indicator measured in 1960 and 1965
has correlated errors.  Data bundled with semopy.
"""
from semopy.examples import political_democracy

from statsformula.sem import fit_semopy

pd_data = political_democracy.get_data()

model = """
  # measurement model
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + y2 + y3 + y4
    dem65 =~ y5 + y6 + y7 + y8
  # regressions
    dem60 ~ ind60
    dem65 ~ ind60 + dem60
  # residual correlations
    y1 ~~ y5
    y2 ~~ y4 + y6
    y3 ~~ y7
    y4 ~~ y8
    y6 ~~ y8
"""
fit = fit_semopy(model, pd_data)
print(fit.summary())

# equal loadings over time (measurement invariance) and the indirect effect of
# industrialization on democracy in 1965, through democracy in 1960
constrained = """
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + a*y2 + b*y3 + c*y4
    dem65 =~ y5 + a*y6 + b*y7 + c*y8
    dem60 ~ g1*ind60
    dem65 ~ ind60 + b2*dem60
    y1 ~~ y5
    y2 ~~ y4 + y6
    y3 ~~ y7
    y4 ~~ y8
    y6 ~~ y8
    indirect := g1*b2
"""
fit2 = fit_semopy(constrained, pd_data)
s1, s2 = fit.fit_stats, fit2.fit_stats
print("free loadings:  chisq = %.3f on %d df" % (s1["chisq"], s1["df"]))
print("equal loadings: chisq = %.3f on %d df" % (s2["chisq"], s2["df"]))
print("difference:     chisq = %.3f on %d df" % (s2["chisq"] - s1["chisq"], s2["df"] - s1["df"]))
print("indirect effect ind60 -> dem60 -> dem65: %.3f" % fit2.defined["indirect"])
