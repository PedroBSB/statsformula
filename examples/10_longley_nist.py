"""Numerical accuracy: the Longley (1967) data and NIST's certified regression results.

Longley's macroeconomic series are nearly collinear and are the NIST StRD benchmark for
linear least squares ("higher difficulty").  statsformula's summary(lm()) uses R's LINPACK QR
(dqrdc2); the estimates are compared with the certified values to 15 significant digits.
"""
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from statsformula.sklearn import FormulaEstimator

lo = sm.datasets.longley.load_pandas().data        # TOTEMP GNPDEFL GNP UNEMP ARMED POP YEAR
s = FormulaEstimator("TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR",
                     LinearRegression(fit_intercept=False)).fit(lo).summary()
print(s)

# https://www.itl.nist.gov/div898/strd/lls/data/LINKS/v-Longley.shtml
nist = pd.DataFrame({
    "certified estimate": [-3482258.63459582, 15.0618722713733, -0.358191792925910e-01,
                           -2.02022980381683, -1.03322686717359, -0.511041056535807e-01,
                           1829.15146461355],
    "certified se": [890420.383607373, 84.9149257747669, 0.334910077722432e-01, 0.488399681651699,
                     0.214274163161675, 0.226073200069370, 455.478499142212]},
    index=s.coefficients.index)
nist["rel. error est"] = (s.coefficients["Estimate"] / nist["certified estimate"] - 1).abs()
nist["rel. error se"] = (s.coefficients["Std. Error"] / nist["certified se"] - 1).abs()
print(nist.to_string(float_format=lambda v: "%.6g" % v))
print("residual SD %.12f (certified 304.854073561965)" % s.sigma)
print("R-squared   %.15f (certified 0.995479004577296)" % s.r_squared)
assert nist[["rel. error est", "rel. error se"]].to_numpy().max() < 1e-8
