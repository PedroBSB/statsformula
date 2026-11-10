"""Binary choice: Spector & Mazzeo's grade data, Greene's probit and logit example.

32 students; GRADE = 1 if the final grade improved, explained by grade point average (GPA),
the score on an economics pre-test (TUCE) and participation in a new teaching method (PSI).
Greene, *Econometric Analysis*, Table "Estimated Probability Models".  statsformula builds the
design matrix, statsmodels fits probit and logit, and the marginal effects of PSI are compared.
"""
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

import statsformula as sf
from statsformula.sklearn import FormulaEstimator

sp = sm.datasets.spector.load_pandas().data        # GPA TUCE PSI GRADE
y, X = sf.dmatrices("GRADE ~ GPA + TUCE + PSI", sp)

probit = sm.Probit(y, X).fit(disp=False)
logit = sm.Logit(y, X).fit(disp=False)
print(pd.DataFrame({"probit": probit.params, "probit se": probit.bse,
                    "logit": logit.params, "logit se": logit.bse}).round(4).to_string())
print("log-likelihood: probit %.4f, logit %.4f" % (probit.llf, logit.llf))
print(probit.get_margeff(at="mean").summary_frame().round(4).to_string())

# the same logit as a scikit-learn classifier (no penalty), e.g. inside cross-validation
clf = FormulaEstimator("factor(GRADE) ~ GPA + TUCE + PSI", LogisticRegression(penalty=None, max_iter=1000),
                       remove_intercept=True).fit(sp)
print(clf.summary().coefficients.round(4).to_string())
