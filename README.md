# statsformula

[![PyPI](https://img.shields.io/pypi/v/statsformula.svg)](https://pypi.org/project/statsformula/)
[![Python](https://img.shields.io/pypi/pyversions/statsformula.svg)](https://pypi.org/project/statsformula/)
[![tests](https://github.com/PedroBSB/statsformula/actions/workflows/ci.yml/badge.svg)](https://github.com/PedroBSB/statsformula/actions/workflows/ci.yml)
[![License: GPL v2+](https://img.shields.io/badge/license-GPL--2.0--or--later-blue.svg)](LICENSE)

**R model formulas for Python: the same terms, columns, names and numbers as R.**

`statsformula` is a port of R's formula machinery (`terms`, `model.frame`, `model.matrix`,
`update`, ...), the [Formula](https://cran.r-project.org/package=Formula) package (multi-part
formulas), `Matrix::sparse.model.matrix`, the [mgcv](https://cran.r-project.org/package=mgcv)
smooths (`s()`, `te()`, `ti()`) and penalized GAM fit, and
[lavaan](https://lavaan.ugent.be)'s model syntax. It translates R's own algorithms
(`termsform` and `modelmatrix` from `stats/src/model.c`, mgcv's `smoothCon`, `PredictMat` and
thin plate spline code, LINPACK/LAPACK numerics) and is tested against R on thousands of cases.
R is not needed at run time.

* **Design matrices** identical to R's `model.matrix()`: column names, `assign`, `contrasts`,
  factor coding, `poly()`, `ns()`, `bs()`, mgcv smooths, re-applied to new data.
* **scikit-learn integration**: formula transformers and estimators that work in `Pipeline`,
  `cross_val_score` and `GridSearchCV` (the formula itself can be tuned).
* **R-style summaries**: `summary.lm`, `summary.gam` (with mgcv's smooth-term tests) and
  lavaan's `summary()`, printed exactly like R.
* **Models**: Gaussian GAMs fitted like `mgcv::gam(method = "REML")`, path models with feedback
  loops by full-information ML (as `lavaan::sem()`), SEM through semopy and pySEM.

**Documentation:** the [user guide](docs/guide/) states the estimators and algorithms
mathematically (REML/GCV, Wood's smooth tests, path-model likelihood and information, fit
measures). The full site, with the API reference and an examples gallery, is built with Sphinx:
`pip install -e ".[docs]" && python -m sphinx -b html docs docs/_build/html`.

**Contents:** [Installation](#installation) · [Examples with real data](#examples-with-real-data) ·
[Why](#why) · [Validation](#validation-against-r) · [Reference](#what-is-supported) ·
[Limitations](#differences-from-r-and-limitations)

## Installation

```
pip install statsformula                 # numpy, pandas, scipy
pip install "statsformula[sklearn]"      # + scikit-learn estimators, pipelines, summaries
pip install "statsformula[sem]"          # + semopy backend for structural equation models
pip install statsmodels                  # optional: used by the econometrics examples
```

Python 3.10 or newer. R is **not** required.

## Examples with real data

Every example uses a dataset that ships with scikit-learn, statsmodels, semopy or statsformula
itself (`sf.load_data()`), so it runs as-is, without downloads. They cover machine learning,
econometrics (Greene's *Econometric Analysis*: robust inference, binary choice, panel data,
SUR, instrumental variables, simultaneous equations) and structural equation models (Bollen). The
complete scripts are in [`examples/`](examples/), and the test suite runs all of them. The
outputs shown are the scripts' actual output. The `lm` and lavaan tables are identical to what R
prints for the same data.

### 1. Design matrices: Fisher's iris data

```python
import pandas as pd
from sklearn.datasets import load_iris
import statsformula as sf

raw = load_iris(as_frame=True)
iris = raw.frame.drop(columns="target")
iris.columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]   # R's names
iris["Species"] = pd.Categorical.from_codes(raw.target, ["setosa", "versicolor", "virginica"])

y, X = sf.dmatrices("Sepal.Length ~ Species * Petal.Length + poly(Sepal.Width, 2)", iris)
list(X.columns)
# ['(Intercept)', 'Speciesversicolor', 'Speciesvirginica', 'Petal.Length',
#  'poly(Sepal.Width, 2)1', 'poly(Sepal.Width, 2)2',
#  'Speciesversicolor:Petal.Length', 'Speciesvirginica:Petal.Length']
X.assign       # [0, 1, 1, 2, 3, 3, 4, 4]                  attr(, "assign") in R
X.contrasts    # {'Species': 'contr.treatment'}

# the fitted design (factor levels, orthogonal polynomial coefficients) is reused on new data
new = pd.DataFrame({"Species": ["virginica", "setosa"], "Petal.Length": [5.1, 1.4],
                    "Sepal.Width": [3.0, 3.5]})
X.model_spec.get_model_matrix(new)
```

### 2. `summary(lm())` in Python

```python
from sklearn.linear_model import LinearRegression
from statsformula.sklearn import FormulaEstimator

model = FormulaEstimator("Sepal.Length ~ Petal.Length * Species", LinearRegression(fit_intercept=False))
print(model.fit(iris).summary())
```

```
Residuals:
     Min       1Q   Median       3Q      Max 
-0.73479 -0.22785 -0.03132  0.24375  0.93608 

Coefficients:
                               Estimate Std. Error t value Pr(>|t|)    
(Intercept)                      4.2132     0.4074  10.341  < 2e-16 ***
Petal.Length                     0.5423     0.2768   1.959  0.05200 .  
Speciesversicolor               -1.8056     0.5984  -3.017  0.00302 ** 
Speciesvirginica                -3.1535     0.6341  -4.973 1.85e-06 ***
Petal.Length:Speciesversicolor   0.2860     0.2951   0.969  0.33405    
Petal.Length:Speciesvirginica    0.4534     0.2901   1.563  0.12029    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.3365 on 144 degrees of freedom
Multiple R-squared:  0.8405,	Adjusted R-squared:  0.8349 
F-statistic: 151.7 on 5 and 144 DF,  p-value: < 2.2e-16
```

The same table is available as data: `s.coefficients` (a DataFrame), `s.sigma`, `s.r_squared`,
`s.adj_r_squared`, `s.fstatistic`, `s.aic` and `s.bic`.

### 3. scikit-learn pipelines: diabetes progression

The diabetes data of Efron et al. (2004): 442 patients, ten baseline measurements, and disease
progression one year later.

```python
import numpy as np, pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsformula.sklearn import FormulaEstimator, FormulaTransformer

df = load_diabetes(scaled=False, as_frame=True).frame
df.columns = ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu", "progression"]
df["sex"] = pd.Categorical(df["sex"].astype(int).astype(str))
train, test = train_test_split(df, test_size=0.25, random_state=0)
cv = KFold(5, shuffle=True, random_state=0)

# (a) the formula as the first step of a pipeline
pipe = make_pipeline(
    FormulaTransformer("~ sex + age + ns(bmi, 3) + bp + log(ltg) + hdl + poly(glu, 2)",
                       remove_intercept=True),
    StandardScaler(),
    RidgeCV(alphas=np.logspace(-3, 3, 13)),
)
cross_val_score(pipe, train, train["progression"], cv=cv).mean()     # 0.526
pipe.fit(train, train["progression"])
pipe[0].get_feature_names_out()
# ['sex2', 'age', 'ns(bmi, 3)1', 'ns(bmi, 3)2', 'ns(bmi, 3)3', 'bp', 'log(ltg)', 'hdl',
#  'poly(glu, 2)1', 'poly(glu, 2)2']

# (b) the formula as a hyper-parameter: FormulaEstimator builds y and X from it
search = GridSearchCV(
    FormulaEstimator(estimator=Ridge()),
    {"formula": ["progression ~ bmi + bp + ltg",
                 "progression ~ sex + age + bmi + bp + ltg + hdl + glu",
                 "progression ~ sex + age + ns(bmi, 3) + bp + log(ltg) + hdl + poly(glu, 2)",
                 "progression ~ (sex + bmi + bp + ltg)^2 + hdl"],
     "estimator__alpha": [0.01, 0.1, 1.0, 10.0]},
    cv=cv,
).fit(train)
search.best_params_
# {'estimator__alpha': 0.01, 'formula': 'progression ~ (sex + bmi + bp + ltg)^2 + hdl'}
```

| formula | best alpha | mean CV R² |
|---|---:|---:|
| `progression ~ (sex + bmi + bp + ltg)^2 + hdl` | 0.01 | 0.533 |
| `progression ~ sex + age + ns(bmi, 3) + bp + log(ltg) + hdl + poly(glu, 2)` | 0.1 | 0.529 |
| `progression ~ sex + age + bmi + bp + ltg + hdl + glu` | 1 | 0.523 |
| `progression ~ bmi + bp + ltg` | 1 | 0.505 |

All data-dependent transformations (factor levels, `ns()` knots, `poly()` coefficients, `log`)
are learned on each training fold and re-applied to the held-out fold, so cross-validation
does not leak information.

### 4. Classification: Wisconsin breast cancer

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

raw = load_breast_cancer(as_frame=True)
bc = raw.frame.drop(columns="target")
bc.columns = [c.replace(" ", "_") for c in bc.columns]
bc["diagnosis"] = pd.Categorical.from_codes(raw.target, list(raw.target_names))

clf = FormulaEstimator(
    "diagnosis ~ log(mean_area) + mean_concave_points * mean_texture + worst_smoothness",
    LogisticRegression(C=10.0, max_iter=5000), remove_intercept=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
cross_val_score(clf, bc, bc["diagnosis"], cv=cv, scoring="roc_auc").mean()   # 0.986
clf.fit(bc).classes_                       # ['benign', 'malignant']
clf.predict_proba(bc.head(3))
```

A factor response is passed to the classifier as its labels. `y_encoding=` selects integer
codes or one-hot columns instead.

### 5. Generalized additive models (mgcv)

```python
b = sf.gam("progression ~ sex + s(age) + s(bmi) + s(bp) + s(ltg) + s(hdl)", df, method="REML")
print(b.summary())
```

```
Family: gaussian 
Link function: identity 

Formula:
progression ~ sex + s(age) + s(bmi) + s(bp) + s(ltg) + s(hdl)

Parametric coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)  163.140      3.709  43.987  < 2e-16 ***
sex2         -23.502      5.788  -4.061 5.82e-05 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Approximate significance of smooth terms:
         edf Ref.df     F  p-value    
s(age) 2.878  3.622  1.98    0.104    
s(bmi) 2.476  3.152 21.81  < 2e-16 ***
s(bp)  1.282  1.514 20.03 3.62e-06 ***
s(ltg) 3.306  4.163 15.16  < 2e-16 ***
s(hdl) 1.000  1.000 18.98 1.70e-05 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

R-sq.(adj) =  0.523   Deviance explained = 53.6%
-REML = 2371.4  Scale est. = 2832.3    n = 442
```

These are mgcv's numbers: smoothing parameters, coefficients, `Vp` standard errors and Wood's
smooth-term tests. The one visible difference from `mgcv::gam` on this model is the edf of
`s(hdl)`: 1.000 here, 1.004 in mgcv. Its REML-optimal smoothing parameter is infinite (the
effect is linear), and mgcv stops slightly before reaching it. Predictions with standard errors,
and the same model as a scikit-learn estimator:

```python
new_patients = pd.DataFrame({"sex": ["1", "2"], "age": [50, 50], "bmi": [22.0, 32.0],
                             "bp": [90, 105], "ltg": [4.4, 5.0], "hdl": [55, 40]})
b.predict(new_patients, se_fit=True)
#       fit  se_fit
# 0  111.84    6.16                     predict.gam(se.fit = TRUE)
# 1  209.80    6.77

from statsformula.sklearn import GAMRegressor
gam = GAMRegressor("progression ~ sex + s(age) + s(bmi) + s(bp) + s(ltg) + s(hdl)")
cross_val_score(gam, df, cv=KFold(5, shuffle=True, random_state=0)).mean()   # 0.497 (linear: 0.488)
```

### 6. Path analysis: does BMI act through triglycerides?

```python
from statsformula.sklearn import PathModel

df["sex"] = (df["sex"] == "2").astype(float)
pm = PathModel("""
  ltg         ~ bmi + age + sex          # mediator: log serum triglycerides
  progression ~ ltg + bmi + bp           # outcome
""").fit(df)
print(pm.summary(effects=True))          # lavaan's summary(fit, fit.measures = TRUE, rsquare = TRUE)
```

```
Regressions:
                   Estimate  Std.Err  z-value  P(>|z|)
  ltg ~                                               
    bmi               0.048    0.005    9.604    0.000
    age               0.007    0.002    4.250    0.000
    sex               0.086    0.044    1.963    0.050
  progression ~                                       
    ltg              49.577    5.673    8.739    0.000
    bmi               6.500    0.709    9.171    0.000
    bp                0.903    0.209    4.325    0.000

Intercepts:
                   Estimate  Std.Err  z-value  P(>|z|)
   .ltg               2.986    0.143   20.906    0.000
   .progression    -334.881   26.946  -12.428    0.000

Variances:
                   Estimate  Std.Err  z-value  P(>|z|)
   .ltg               0.206    0.014   14.866    0.000
   .progression    3083.051  207.388   14.866    0.000

R-Square:
                   Estimate
    ltg               0.242
    progression       0.471

Indirect Effects:
                        Estimate  Std.Err  z-value  P(>|z|)
    progression <- bmi     2.376    0.368    6.464    0.000
    progression <- age     0.358    0.094    3.822    0.000
    progression <- sex     4.286    2.238    1.915    0.055
```

The fit measures above the parameter table (chi-square 24.686 on 3 df, CFI 0.949, TLI 0.848,
RMSEA 0.128, SRMR 0.035, AIC 5381.888) are those of `lavaan::sem()` on the same data, and are
also returned by `pm.fit_measures()`. Models with feedback loops (`y1 ~ y2` and `y2 ~ y1`) and
correlated disturbances (`y1 ~~ y2`) are fitted by full-information maximum likelihood.

### 7. Confirmatory factor analysis: Holzinger & Swineford (1939)

```python
from semopy.examples import holzinger39
from statsformula.sem import fit_semopy

fit = fit_semopy("""
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
""", holzinger39.get_data())
print(fit.summary())
```

```
SEM fitted with semopy (lavaan parameter table)

  Number of model parameters                        21

  Number of observations                           301

Model Test User Model:

  Test statistic                                85.306
  Degrees of freedom                                24
  P-value (Chi-square)                           0.000

Model Test Baseline Model:

  Test statistic                               918.852
  Degrees of freedom                                36

User Model versus Baseline Model:

  Comparative Fit Index (CFI)                    0.931
  Tucker-Lewis Index (TLI)                       0.896
...
Latent Variables:
                   Estimate  Std.Err  z-value  P(>|z|)
  visual =~                                           
    x1                1.000       NA       NA       NA
    x2                0.554    0.100    5.554    0.000
    x3                0.729    0.109    6.685    0.000
  textual =~                                          
    x4                1.000       NA       NA       NA
    x5                1.113    0.065   17.014    0.000
    x6                0.926    0.055   16.703    0.000
  speed =~                                            
    x7                1.000       NA       NA       NA
    x8                1.180    0.165    7.152    0.000
    x9                1.082    0.151    7.155    0.000
...
```

The model is expanded by `lavaanify()` into lavaan's own parameter table and fitted by semopy.
The estimates agree with `lavaan::cfa()` to about 2e-5 (lavaan: chi-square 85.306 on 24 df).

### 8. Econometrics: heteroskedasticity-robust inference (Greene)

Greene's credit card expenditure data (72 cardholders, bundled with statsmodels): OLS with
R's `summary(lm())`, White standard errors and the Breusch-Pagan test from statsmodels on
statsformula's design matrices, and weighted least squares.

```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

cc = sm.datasets.ccard.load_pandas().data
formula = "AVGEXP ~ AGE + OWNRENT + INCOME + I(INCOME^2)"
ols = FormulaEstimator(formula, LinearRegression(fit_intercept=False)).fit(cc)
ols.summary().coefficients          # Greene: -237.15, -3.0818, 27.941, 234.35, -14.997

y, X = sf.dmatrices(formula, cc)
fit = sm.OLS(y, X).fit()
fit.get_robustcov_results("HC0").bse          # White (1980) standard errors
het_breuschpagan(fit.resid, X)[:2]            # LM statistic, p-value

wls = FormulaEstimator(formula, LinearRegression(fit_intercept=False))
wls.fit(cc, sample_weight=1 / cc["INCOME"] ** 2).summary()     # weighted summary(lm())
```

|  | OLS se | White (HC0) se | HC3 se |
|---|---:|---:|---:|
| (Intercept) | 199.352 | 212.991 | 229.574 |
| AGE | 5.515 | 3.302 | 3.605 |
| OWNRENT | 82.922 | 92.188 | 99.314 |
| INCOME | 80.366 | 88.866 | 95.482 |
| I(INCOME^2) | 7.469 | 6.945 | 7.476 |

### 9. Binary choice: probit and logit (Greene, Spector & Mazzeo)

```python
sp = sm.datasets.spector.load_pandas().data            # GPA, TUCE, PSI, GRADE (32 students)
y, X = sf.dmatrices("GRADE ~ GPA + TUCE + PSI", sp)
probit = sm.Probit(y, X).fit(disp=False)
probit.params                     # Greene: -7.452, 1.626, 0.052, 1.426
probit.get_margeff(at="mean").summary_frame()        # marginal effect of PSI: 0.468
```

|  | probit | se | logit | se |
|---|---:|---:|---:|---:|
| (Intercept) | -7.4523 | 2.5425 | -13.0213 | 4.9313 |
| GPA | 1.6258 | 0.6939 | 2.8261 | 1.2629 |
| TUCE | 0.0517 | 0.0839 | 0.0952 | 0.1416 |
| PSI | 1.4263 | 0.5950 | 2.3787 | 1.0646 |

### 10. Numerical accuracy: Longley and NIST's certified results

The Longley (1967) data are nearly collinear, and NIST uses them as a benchmark for linear
least squares. statsformula's `summary(lm())` uses R's LINPACK QR and reproduces the certified
values:

```python
lo = sm.datasets.longley.load_pandas().data
s = FormulaEstimator("TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR",
                     LinearRegression(fit_intercept=False)).fit(lo).summary()
s.sigma, s.r_squared              # 304.854073561964, 0.995479004577296  (NIST: ...565, ...296)
```

The largest relative error against NIST's 15-digit certified estimates and standard errors is
1.1e-13 (see [`examples/10_longley_nist.py`](examples/10_longley_nist.py)).

### 11. Panel data and seemingly unrelated regressions (Greene, Grunfeld)

Greene's five-firm version of Grunfeld's investment data ships with statsformula
(`sf.load_data("grunfeld_greene")`, Table F13.1). Fixed effects are least squares with firm
dummies. Zellner's SUR is a `PathModel` with correlated disturbances, fitted by maximum
likelihood:

```python
g = sf.load_data("grunfeld_greene")
lsdv = lambda f: FormulaEstimator(f, LinearRegression(fit_intercept=False)).fit(g).summary()
lsdv("invest ~ value + capital").coefficients               # pooled OLS
lsdv("invest ~ value + capital + firm").coefficients        # firm fixed effects (plm "within")
lsdv("invest ~ value + capital + firm + factor(year)")      # two-way fixed effects

abbr = {"General Motors": "GM", "Chrysler": "CH", "General Electric": "GE",
        "Westinghouse": "WE", "US Steel": "US"}
wide = g.assign(firm=g.firm.map(abbr)).pivot(index="year", columns="firm",
                                             values=["invest", "value", "capital"])
wide.columns = ["%s_%s" % c for c in wide.columns]
f = list(abbr.values())
sur = PathModel("\n".join(
    ["invest_{0} ~ value_{0} + capital_{0}".format(a) for a in f] +
    ["invest_%s ~~ %s" % (a, " + ".join("invest_" + b for b in f[i + 1:]))
     for i, a in enumerate(f[:-1])])).fit(wide)
sur.fit_measures()["logl"]        # -459.092, as lavaan::sem() on the same system
```

| | pooled OLS | firm fixed effects | two-way fixed effects |
|---|---:|---:|---:|
| value | 0.1051 (0.0114) | 0.1060 (0.0159) | 0.1260 (0.0232) |
| capital | 0.3054 (0.0435) | 0.3467 (0.0242) | 0.3618 (0.0360) |
| residual SE | 127.26 | 69.12 | 70.12 |

The F test of equal firm intercepts is F(4, 93) = 58.96. The likelihood-ratio test of a
diagonal disturbance covariance (SUR vs. equation-by-equation OLS) is 44.76 on 10 df.

### 12. Structural equation model: Bollen's Political Democracy

Bollen (1989, ch. 8): industrialization in 1960 and democracy in 1960 and 1965 in 75
countries, with correlated errors for the same indicator over time. This is the standard
lavaan example.

```python
from semopy.examples import political_democracy

fit = fit_semopy("""
  ind60 =~ x1 + x2 + x3
  dem60 =~ y1 + y2 + y3 + y4
  dem65 =~ y5 + y6 + y7 + y8
  dem60 ~ ind60
  dem65 ~ ind60 + dem60
  y1 ~~ y5
  y2 ~~ y4 + y6
  y3 ~~ y7
  y4 ~~ y8
  y6 ~~ y8
""", political_democracy.get_data())
print(fit.summary())
```

```
Model Test User Model:

  Test statistic                                38.125
  Degrees of freedom                                35
  P-value (Chi-square)                           0.329
...
Regressions:
                   Estimate  Std.Err  z-value  P(>|z|)
  dem60 ~                                             
    ind60             1.483    0.399    3.715    0.000
  dem65 ~                                             
    ind60             0.572    0.221    2.586    0.010
    dem60             0.837    0.098    8.514    0.000
```

The chi-square statistic, degrees of freedom and estimates are lavaan's. With loadings
constrained equal over time (`dem60 =~ y1 + a*y2 + b*y3 + c*y4`, the same labels in `dem65`)
the statistic rises to 40.179 on 38 df, so the difference test is 2.054 on 3 df. The indirect
effect `ind60 -> dem60 -> dem65` (`indirect := g1*b2`) is 1.273.

### 13. Instrumental variables and simultaneous equations (Kmenta; Klein's Model I)

An IV model is a two-part formula, as in R's `ivreg(y ~ regressors | instruments)`.
`sf.model_matrix(F, data, rhs=1)` is the regressor matrix and `rhs=2` the instrument matrix.
Kmenta's food-market data and Klein's Model I ship with statsformula.

```python
from statsmodels.sandbox.regression.gmm import IV2SLS

kmenta = sf.load_data("kmenta")
F = sf.Formula("consump ~ price + income | income + farmPrice + trend")   # demand equation
y = np.asarray(sf.model_part(F, sf.model_frame(F, kmenta), lhs=1)).ravel()
X, Z = sf.model_matrix(F, kmenta, rhs=1), sf.model_matrix(F, kmenta, rhs=2)
iv = IV2SLS(y, X, Z).fit()
iv.params            # 94.6333, -0.2436, 0.3140   (Kmenta 1986; OLS price slope: -0.3163)

# price and quantity determined jointly: the non-recursive system by full-information ML
fiml = PathModel("""
  consump ~ price + income                 # demand
  price   ~ consump + farmPrice + trend    # supply, normalized on price
  consump ~~ price
""").fit(kmenta)
fiml.coefficients().head(2)                # demand price slope -0.2295 (se 0.0900), as lavaan
```

| Klein's Model I, 2SLS (Greene) | | OLS | 2SLS | 2SLS se |
|---|---|---:|---:|---:|
| consumption | profits | 0.1929 | 0.0173 | 0.1312 |
| | lagged profits | 0.0899 | 0.2162 | 0.1192 |
| | wages | 0.7962 | 0.8102 | 0.0447 |
| investment | profits | 0.4796 | 0.1502 | 0.1925 |
| | lagged profits | 0.3330 | 0.6159 | 0.1809 |
| | lagged capital | -0.1118 | -0.1578 | 0.0402 |
| private wages | private output | 0.4395 | 0.4389 | 0.0396 |
| | lagged output | 0.1461 | 0.1467 | 0.0432 |
| | trend | 0.1302 | 0.1304 | 0.0324 |

The full script ([`examples/13_instrumental_variables.py`](examples/13_instrumental_variables.py))
adds the first-stage F test (88.9), the Durbin-Wu-Hausman test of endogeneity (t = -3.38,
p = 0.004) and the supply equation.

### Datasets

Datasets that are not available in another Python package ship with statsformula:

```python
sf.load_data()                     # catalogue: name, title, rows, columns, source
k = sf.load_data("kmenta")         # pandas DataFrame; k.attrs["description"], k.attrs["source"]
```

| name | data | source |
|---|---|---|
| `kmenta` | supply and demand for food, 20 years | Kmenta (1986), *Elements of Econometrics*, Table 13-1 |
| `klein` | Klein's Model I, US 1920-1941 | Greene (2003), *Econometric Analysis*, Table F15.1 |
| `grunfeld_greene` | investment of 5 US firms, 1935-1954 | Greene (2003), *Econometric Analysis*, Table F13.1 |

The other examples use data from scikit-learn (`load_iris`, `load_diabetes`,
`load_breast_cancer`), statsmodels (`ccard`, `spector`, `longley`) and semopy
(`holzinger39`, `political_democracy`).

## Why

`patsy` and `formulaic` are *inspired by* R formulas but differ in many details (operator
semantics, contrast coding of interactions without main effects, column names, `C()`, `poly()`
and spline bases, `.` expansion, `-` / `0` / `1` handling, ...). `statsformula` aims for
**bit-for-bit agreement with R** so that models, coefficients and code can move between R and
Python unchanged. Its API borrows from statsmodels (`dmatrices`) and formulaic (`ModelMatrix`,
`ModelSpec` for applying a fitted design to new data).

## Validation against R

The test-suite compares against results produced by R 4.5.1 (mgcv 1.9-3, Formula 1.2-5,
Matrix 1.7-3); the fixtures are in `tests/fixtures` and can be regenerated with
`python tools/make_fixtures.py`, `tools/make_fuzz.py` and `tools/make_fuzz_mm.py` (needs R).

* 407 hand-written cases: `terms` attributes, `model.matrix` values, column names, `assign` and
  `contrasts`, `sparse.model.matrix`, `model.frame` (NA handling, subset, weights, offset,
  `drop.unused.levels`), `update`, `reformulate`, `drop.terms`, `delete.response`, the Formula
  package, prediction on new data, error messages, 57 mgcv models (16 of them needing
  `gam.side()` side constraints) and 24 mgcv penalty matrices;
* 500 randomly generated formulas compared with R's `terms()` (2,000 more checked during
  development);
* 400 randomly generated formulas compared with R's `model.matrix()`;
* penalty matrices of every mgcv model (`G$S`, `G$off`, `G$rank`, smoothing-parameter names) and
  18 penalized GAM fits (`gam(method = "REML" / "GCV.Cp")`, fixed `sp`, weights, `gamma`):
  smoothing parameters, coefficients, fitted values, predictions, edf, scale, prediction standard
  errors (`predict(se.fit = TRUE)`) and `summary.gam`: `p.table`, `s.table` (Wood's smooth-term
  tests, including the random-effect test of `re`/`cs`/`cc` smooths), `pTerms.table`, `edf1`,
  `r.sq`, deviance explained and the printed output;
* `summary.lm` of 12 linear models (multi-response, weighted, rank deficient, intercept only,
  no intercept, few residual degrees of freedom): coefficient table, sigma, R^2, F test,
  `logLik`/`AIC`/`BIC` and the printed output; mgcv's Davies algorithm (`psum.chisq`) on 60 cases,
  R's `format()`, `format.pval()` and `printCoefmat()` (`tools/make_summary_fixtures.py`);
* 7 SEM models fitted through semopy and pySEM compared with `lavaan::sem()`/`cfa()`
  (`tools/make_sem_fixtures.py`);
* 8 path models: `lm()` per equation, propagated predictions, and `lavaan::sem()` estimates and
  total effects (`tools/make_path_fixtures.py`);
* 7 path models for inference (`tools/make_pathse_fixtures.py`): `lavaan::sem()` standard errors,
  z tests and confidence intervals with `information = "expected"` and `"observed"`, delta-method
  `:=` indirect/total effects, full-information ML of non-recursive models (feedback loops,
  a three-variable cycle) and correlated disturbances, `summary(lm())` t tests, 20
  `fitMeasures()` (chi-square, baseline, CFI, TLI, log-likelihoods, AIC, BIC, SABIC, RMSEA with its
  confidence interval and tests, SRMR), `lavInspect(fit, "r2")` and the printed parameter tables
  of `summary()`;
* 132 lavaan model strings and 353 option/preset combinations compared with lavaan 0.7.3
  (`tools/make_lavaan_fixtures.py`).

Numbers agree to ~1e-12 relative error; names, orderings and attributes are identical.

## What is supported

### Formula operators (exact R semantics)

| syntax | meaning |
|---|---|
| `a + b` | both terms |
| `a:b` | interaction |
| `a*b` | `a + b + a:b` |
| `a/b` | nesting, `a + a:b` |
| `b %in% a` | `b` nested in `a` |
| `(a + b + c)^2` | all interactions up to order 2 (`**` is accepted and printed as `^`) |
| `- b`, `- 1`, `+ 0`, `0 +` | remove terms / intercept |
| `.` | all other columns of `data` |
| `offset(x)` | offset terms |
| `I(expr)` | arithmetic inside formulas |
| `cbind(y1, y2) ~ x` | matrix responses |
| `y1 | y2 ~ x | z` | multi-part formulas (Formula package) |
| `` `my var` `` | non-syntactic names |

R's corner cases are reproduced too (`y ~ x^1` is an error, `(-a * b)` drops everything,
`y ~ 0/x` drops `x`, the response on the right-hand side, terms deduplication and ordering,
the factor coding heuristic of *Statistical Models in S* for interactions without main effects, ...).

### Functions available inside formulas

* **math:** `log(x, base)`, `log2`, `log10`, `log1p`, `exp`, `expm1`, `sqrt`, `abs`, `sign`,
  `floor`, `ceiling`, `round`, `signif`, `trunc`, trigonometric and hyperbolic functions,
  `gamma`, `lgamma`, `beta`, `choose`, `factorial`, `pmin`, `pmax`, `cumsum`, ...
* **operators:** `+ - * / ^ %% %/%`, comparisons, `& | ! xor`, `%in%`, `ifelse`, `is.na`
* **summaries:** `mean`, `median`, `sd`, `var`, `sum`, `min`, `max`, `quantile`, `length`
* **factors:** `factor`, `as.factor`, `ordered`, `relevel`, `droplevels`, `interaction`, `cut`,
  `C(f, contr, how.many)`
* **contrasts:** `contr.treatment`, `contr.sum`, `contr.helmert`, `contr.poly`, `contr.SAS`
  (via `C()`, the `contrasts=` argument or `sf.options(contrasts=...)`)
* **bases:** `poly(x, degree, raw=)`, multivariate `poly(x, z, degree=)`, `splines::bs`,
  `splines::ns` (exactly as R, including extrapolation), `scale`, `cbind`
* **mgcv smooths:** `s()`, `te()`, `ti()` with bases `tp`, `ts`, `cr`, `cs`, `cc`, `ps`, `cp`,
  `bs`, `re`; numeric, factor and ordered `by=`; `k`, `m`, `bs`, `d`, `knots`
* **conversions:** `as.numeric`, `as.integer`, `as.character`, `as.logical`, `c`, `rep`, `seq`
* **your own Python functions:** any function or object visible where you call statsformula
  (e.g. `center(x)`, `np.log1p(x)`), or registered with `sf.register_function()`.

Data-dependent transformations (`poly`, `bs`, `ns`, `scale`, factor levels, contrasts, mgcv
smooths) are remembered and re-applied to new data, like R's `predict()`.

### R functions

| R | statsformula |
|---|---|
| `formula()`, `as.formula()` | `sf.formula()` → `RFormula` |
| `terms(f, data, keep.order, specials, simplify)` | `sf.terms()` → `Terms` (`.term_labels`, `.factors`, `.order`, `.intercept`, `.response`, `.offset`, `.variables`, `.specials`) |
| `model.frame(f, data, subset, na.action, weights, offset, drop.unused.levels, xlev)` | `sf.model_frame()` |
| `model.matrix(f, data, contrasts.arg, xlev)` | `sf.model_matrix()` |
| `Matrix::sparse.model.matrix()` | `sf.sparse_model_matrix()` (scipy.sparse CSC) |
| `model.response`, `model.offset`, `model.weights`, `model.extract` | same names with `_` |
| `update()` | `sf.update()` (formulas and Formulas) |
| `reformulate`, `drop.terms`, `delete.response`, `all.vars`, `all.names`, `get_all_vars`, `deparse` | same names with `_` |
| `Formula::Formula`, `as.Formula`, `model.part`, `formula(F, lhs, rhs, collapse)` | `sf.Formula`, `sf.as_Formula`, `sf.model_part`, `F.formula(lhs=, rhs=, collapse=)` |
| `mgcv::interpret.gam`, `smoothCon`, `PredictMat`, `model.matrix(gam(...))`, `predict(gam, type = "lpmatrix")` | `sf.interpret_gam`, `sf.smooth_con`, `sf.predict_mat`, `sf.gam_model_matrix`, `X.model_spec.get_model_matrix(new)` |
| `options(contrasts = ...)`, `options(na.action = ...)` | `sf.options(contrasts=..., na_action=...)` |

### Python conveniences (statsmodels / formulaic style)

```python
y, X = sf.dmatrices("y ~ x + C(g, sum)", df)        # pandas frames, NA rows removed jointly
import statsmodels.api as sm
fit = sm.OLS(y, X).fit()

X = sf.model_matrix("y ~ poly(x, 3) + ns(z, df = 4) + g", train)
X_test = X.model_spec.get_model_matrix(test)        # same transformation, levels and contrasts

S = sf.sparse_model_matrix("~ user * item", big_df)  # scipy.sparse
```

### scikit-learn pipelines

`pip install statsformula[sklearn]`

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV
from statsformula.sklearn import FormulaTransformer, FormulaEstimator, design_matrices

# 1. the formula as a transformer step (X only)
pipe = make_pipeline(FormulaTransformer("~ log(x) + ns(z, 3) + C(g, sum)", remove_intercept=True),
                     StandardScaler(), Ridge())
pipe.fit(train, train["y"]).predict(test)

# 2. the formula builds Y *and* X; Y can have several columns
model = FormulaEstimator("cbind(y1, y2) ~ log(x) + poly(z, 2) + g", Ridge())
model.fit(train).predict(test)                  # (n, 2) predictions
FormulaEstimator("cls ~ x + g", LogisticRegression()).fit(train).predict_proba(test)

# hyper-parameters *and* formulas can be tuned
GridSearchCV(model, {"formula": ["cbind(y1, y2) ~ x + g", "cbind(y1, y2) ~ log(x) * g"],
                     "estimator__alpha": [0.1, 1, 10]}, cv=5).fit(train)

Y, X = design_matrices("y1 | y2 ~ log(x) + g", train)   # aligned response and design matrices
```

A scikit-learn `Pipeline` step can only transform `X`, so building the response from the
formula needs the `FormulaEstimator` wrapper. All data-dependent transformations (factor
levels, contrasts, `poly`, `ns`, `bs`, `scale`, mgcv smooths) are learned on `fit` and reused on
`predict`/`transform`. Responses: `y`, transformed `log(y)`, `cbind(y1, y2, ...)`, multi-part
`y1 | y2 ~ ...`; categorical responses as labels (for classifiers), integer codes or one-hot
(`y_encoding=`). Rows with missing values are dropped jointly from `Y`, `X` and `sample_weight`
when fitting; `transform` keeps every row (`na_action="na.pass"`). Custom Python functions used
inside the formula are passed with `env={"name": func}`.

### Smoothing penalties and penalized GAMs (mgcv)

Every mgcv design matrix carries its wiggliness penalties, exactly as `gam(..., fit = FALSE)`
returns them (`G$S`, `G$off`, `G$rank`, names of `G$sp`), so you can regularize the smooths
yourself or let statsformula estimate the smoothing parameters like `mgcv::gam`:

```python
import statsformula as sf
from statsformula.sklearn import FormulaTransformer, GAMRegressor

X = sf.model_matrix("y ~ g + s(x) + te(x, z)", df)
X.model_spec.penalties        # [<GamPenalty s(x): 9x9 at column 2, rank 8>, te(x,z)1, te(x,z)2]
P = sf.penalty_matrix(X.model_spec.penalties, X.shape[1], sp=[0.5, 1.0, 2.0])  # sum_j sp_j S_j
beta = np.linalg.solve(X.T @ X + P, X.T @ y)                                    # generalized ridge

b = sf.gam("y ~ g + s(x) + te(x, z)", df, method="REML")   # or "GCV.Cp"; sp=[...] to fix them
b.sp, b.coefficients, b.edf.sum(), b.scale, b.predict(new)  # same numbers as mgcv
b.predict(new, se_fit=True)    # columns fit, se_fit: predict.gam(se.fit = TRUE), from b.Vp
print(b.summary())             # summary.gam, printed like R (p.table, s.table, R-sq., -REML, ...)

tr = FormulaTransformer("~ g + s(x) + s(z)").fit(df)        # in a pipeline
tr.penalties_, tr.penalty_matrix(sp=[1.0, 3.0])             # aligned with tr.transform() columns

GAMRegressor("cbind(y1, y2) ~ g + s(x) + te(x, z)", method="REML").fit(df).predict(new)
GAMRegressor("y ~ s(x)").fit(df).predict(new, return_std=True)   # (prediction, standard error)
GridSearchCV(GAMRegressor("y ~ s(x) + s(z)"), {"sp": [[0.1, 0.1], [1, 1], [10, 10]]})
```

The penalized fit minimizes `||y - X b||^2_W + sum_j sp_j b' S_j b`. Smoothing parameters are
estimated by REML (scale profiled out) or GCV with exact gradients, and match mgcv's to ~1e-3
on the log scale; when mgcv sends a smoothing parameter to infinity (a smooth reduced to its
null space), both stop at an arbitrary large value and the fitted values agree to ~1e-5.
Standard errors use mgcv's Bayesian posterior covariance `Vp`. A rank-deficient model (e.g. `y ~ x + s(x)`) raises a
warning (mgcv sets the unidentifiable coefficients to zero). Only the Gaussian family with
identity link is fitted; for other models use `penalties_` with your own solver.

### Path models (multi-line formulas)

A system of regressions can be written as a multi-line formula, one R formula per line, and
fitted as a single scikit-learn estimator:

```python
from statsformula.sklearn import PathModel

model = """
  # mediation with a transformed covariate
  m  ~ x + f                      # mediator
  y  ~ m + x + log(w) +
       poly(z, 2)                 # continuation line
  y2 + y3 ~ m                     # one equation per response (lavaan style)
"""
pm = PathModel(model).fit(df)                      # each equation: lm on its complete cases
pm.predict(new)        # m, y, y2, y3; downstream equations use the *predicted* m
pm.coefficients()      # lavaan::parameterEstimates(): lhs op rhs est se z pvalue ci.lower ci.upper

pm = PathModel("m ~ x + z\ny ~ m + x").fit(df)
pm.effects("indirect").loc["y", "x"]               # a * b, as lavaan's  ind := a*b
pm.effects("total")                                # (I - B)^-1 B
pm.effects("indirect", se=True)                    # long table with delta-method SEs (lavaan's :=)
pm.vcov()                                          # covariance of the estimates

PathModel(model, estimator=Ridge())                       # same estimator for every equation
PathModel(model, estimator={"y": RandomForestRegressor()})  # per response (others: lm)
PathModel("m ~ x\ny ~ s(m) + s(z)")                       # smooths -> GAMRegressor (REML)
GridSearchCV(PathModel(model, estimator=Ridge()), {"estimator__alpha": [0.1, 1, 10]})
```

**Feedback loops and correlated disturbances.** Non-recursive models and residual covariances
(`y1 ~~ y2`) are fitted by full-information maximum likelihood (Fisher scoring from several
starting values, as `lavaan::sem()` with `fixed.x = TRUE`):

```python
pm = PathModel("""
  y1 ~ y2 + x1          # y1 and y2 affect each other
  y2 ~ y1 + x2 + x3
  y1 ~~ y2              # correlated disturbances
""").fit(df)
pm.estimation_, pm.loglik_      # "ML", log-likelihood
pm.predict(new)                 # reduced form (I - B)^-1 G x
pm.effects("total", se=True)    # includes the feedback through the loop
```

**Inference.** `se="expected"` (default) and `se="observed"` reproduce lavaan's standard errors
with `information = "expected"` / `"observed"` (z tests, normal confidence intervals at
`level`); `se="lm"` gives the per-equation t tests of `summary(lm())`. With linear equations,
recursive models with uncorrelated disturbances are still fitted equation by equation (OLS is
the ML estimate; `estimation="ML"` forces the joint fit, `"OLS"` the per-equation one). The
joint fit and the expected information need every endogenous predictor to enter its equation as
a plain column (`y ~ m + log(x)`, not `y ~ log(m)`) and the equations to share their rows;
otherwise each equation uses its own observed information (`information_` says which was used).
Equations with smooths report mgcv's `Vp` standard errors; other estimators get none.

`predict_mode="propagate"` (default) takes only exogenous variables from the data and feeds each
equation's prediction into the equations downstream (for non-recursive models: the reduced
form); `predict_mode="direct"` uses the observed predictors. Latent variables (`=~`) are not
fitted here; use the SEM backends below or parse those models with `lavaanify()`.

### SEM backends: semopy, pySEM, pysemforge (`statsformula.sem`)

statsformula lets a lavaan model use R expressions and fits it with a Python SEM package.
It rewrites the terms into columns (`log(w)` -> `log_w`, factors -> dummies, `poly()`,
`x:z`, `I(z^2)`, with the transformations reusable on new data). `lavaanify()` then expands
the model into lavaan's full parameter table, so each backend fits exactly the model lavaan
would. Estimates come back in lavaan's format, with `:=` parameters evaluated:

```python
from statsformula.sem import fit_semopy, fit_pysem, sem_data, to_semopy, to_pysem

model = """
  ind =~ x1 + x2 + x3
  m ~ a*ind + log(w) + f + z
  y ~ b*m + ind + I(z^2)
  ab := a*b
"""
fit = fit_semopy(model, df)          # semopy (pip install statsformula[sem])
fit.estimates                        # lhs / op / rhs / label / free / est
fit.defined                          # {'ab': ...}
fit = fit_pysem(model, df)           # twoertwein/pySEM (PyTorch)
fit_semopy(model, df, preset="cfa", std_lv=True)   # lavaanify options

desc, data, sd, partable = to_semopy(model, df)    # inputs for semopy.Model(desc).fit(data)
kwargs, sd, partable, names = to_pysem(model, df)  # pysem.sem.SEM(**kwargs)
sd = sem_data(model, df); sd.model, sd.data, sd.transform(new_df)
```

* **semopy:** the bridge writes the lavaan model out explicitly in semopy syntax and fits it
  with a tight SLSQP tolerance. It also works around a semopy 2.3 hang on factors with no
  fixed loading (`std.lv = TRUE`).
* **pySEM:** pySEM uses LISREL all-y matrices, so observed variables in regressions get
  single-indicator stand-ins (`"eta:<name>"`). It is fitted to convergence with PyTorch
  L-BFGS instead of pySEM's Adam loop, which also uses `np.Inf`, removed in NumPy 2. pySEM has
  no equality constraints.
* **pysemforge:** `to_pysemforge()` gives the plain syntax and data that pysemforge reads.
  Version 0.1.0 refits a random sample drawn from the covariance matrix instead of your data,
  and reports chi-square = 0, df = 0, CFI = 1 because of a missing import.
  `fit_pysemforge()` warns about this; use `fit_semopy()` for correct estimates.

Validated against lavaan 0.7.3 on 7 models (Holzinger-Swineford CFA, `std.lv`, second-order
factor, MIMIC with factor and `log()` covariates, Political Democracy with indirect/total
effects, latent and observed path models with R expressions). semopy matches to about 2e-5
and pySEM to about 5e-5. The exception is one model where two covariates correlate at 0.999,
where pySEM matches only to about 1e-2.

### Model summaries (R and lavaan style)

Every fitted model has a `summary()` that prints like its R counterpart and keeps its tables as
DataFrames:

```python
from sklearn.linear_model import LinearRegression, Ridge

s = FormulaEstimator("y ~ log(x) + f", LinearRegression(fit_intercept=False)).fit(df).summary()
print(s)                     # summary(lm(y ~ log(x) + f, df)): residuals, coefficients with
                             # t tests and stars, residual SE, R-squared, F-statistic
s.coefficients, s.sigma, s.r_squared, s.adj_r_squared, s.fstatistic, s.f_pvalue, s.aic, s.bic
FormulaEstimator("cbind(y1, y2) ~ x", LinearRegression()).fit(df).summary()   # one per response
FormulaEstimator("y ~ x", Ridge()).fit(df).summary()     # other estimators: coefficients, score

g = sf.gam("y ~ f + s(x) + te(x, z)", df).summary()     # or GAMRegressor(...).fit(df).summary()
print(g)                     # summary.gam: Parametric coefficients, Approximate significance of
                             # smooth terms (edf, Ref.df, F, p-value), R-sq.(adj), deviance, -REML
g.p_table, g.s_table, g.pterms_table, g.r_sq, g.dev_expl

pm = PathModel("y1 ~ y2 + x1\ny2 ~ y1 + x2 + x3\ny1 ~~ y2").fit(df)
print(pm.summary())          # lavaan's summary(fit, fit.measures = TRUE, rsquare = TRUE)
pm.fit_measures()            # lavaan's fitMeasures(): chisq, df, pvalue, cfi, tli, logl, aic, bic,
                             # rmsea (+ CI, tests), srmr, ...
pm.rsquare()                 # lavInspect(fit, "r2")
pm.summary(effects=True)     # adds indirect and total effects with delta-method SEs

print(fit_semopy(model, df).summary())   # SEM backends: lavaan layout, semopy's SEs and fit stats
```

`summary.lm` uses `lm`'s pivoted QR, so aliased columns show as `NA` exactly as in R (scikit-learn's
`LinearRegression` returns a minimum-norm solution instead). The smooth-term p-values are Wood's
(2013) tests as computed by mgcv (`testStat`, and `reTest` for smooths without a penalty null
space), with Davies' algorithm for the distribution of the test statistic. Path-model fit
measures need the model as one linear system on shared rows (linear equations); equations with
smooths show mgcv's smooth-term table instead of their basis coefficients.

### lavaan model syntax (structural equation models)

`sf.lavaan_parse()` is a port of `lavaan::lavParseModelString(model, as.data.frame. = TRUE)`
and `sf.lavaanify()` of `lavaan::lavaanify()` (lavaan 0.7.3), validated against R on 132 model
strings (parser output, 317 `lavaanify()` option combinations, 36 `cfa()`/`sem()`/`growth()`
parameter tables).

```python
model = """
  # measurement model
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + a*y2 + b*y3 + c*y4
  # regressions
    dem60 ~ ind60
  # residual correlations and a defined parameter
    y1 ~~ y2
    ab := a*b
"""
flat = sf.lavaan_parse(model)            # lhs, op, rhs, block, label, ..., mod.idx
flat.attrs["constraints"]                # [{'op': ':=', 'lhs': 'ab', 'rhs': 'a*b', 'user': 1}]
pt = sf.lavaanify(model, preset="sem")   # the parameter table lavaan::sem() builds
pt = sf.lavaanify(model, auto_var=True, std_lv=True, meanstructure=True)   # or "std.lv" style keys
sf.lavaan_names(model, "ov.x")           # like lavNames()
```

Supported: all operators (`=~ ~ ~~ ~1 | ~*~ <~ := == < > :~ %`), comments (`#`, `!`), `;`,
line continuation, modifiers (`a*x`, `0.5*x`, `NA*x`, `start()`, `0.5?x`, `label()`,
`equal()`, `lower()`, `upper()`, `c()` vectors for multiple groups, `efa("set")*f` blocks,
combined modifiers), simple bounds on labels (`a > 0` becomes a lower bound, as in lavaan),
interactions `x1:x2`, `group:`/`level:`/`class:`/`block:` blocks, and lavaan's error messages
and warnings (with line/position). `lavaanify()` options: `meanstructure`, `int.ov.free`,
`int.lv.free`, `marker.int.zero`, `orthogonal(.x/.y/.efa)`, `std.lv`, `correlation`,
`composites`, `conditional.x`, `fixed.x`, `parameterization`, `constraints`, `ceq.simple`,
`auto`, `model.type`, `auto.fix.first`, `marker`, `auto.fix.single`, `auto.var`,
`auto.cov.lv.x`, `auto.cov.x`, `auto.cov.y`, `auto.th`, `auto.delta`, `auto.efa`, `ngroups`,
`nthresholds`, `group.equal`, `group.partial`, `group.w.free`. Not supported: `effect.coding`,
the `:~` operator inside `lavaanify()`, `conditional.x` with `level:` blocks, and options that
need data (`varTable`: ordered variables are only detected from `|` thresholds in the syntax;
exogenous covariates are detected from the syntax, as `lavaanify()` does).

## Data types

| pandas / numpy | R equivalent |
|---|---|
| `Categorical` (ordered or not) | factor / ordered factor (categories are the levels) |
| `bool` / nullable `boolean` | logical |
| integer | integer |
| float | double (`NaN` is `NA`) |
| strings / `object` | character (converted to factors by `model.matrix`, as in R) |

String levels are sorted the way R sorts them under ICU collation (macOS/Windows R, and Linux
R built with ICU). Use `sf.set_collation("C")` to mimic `LC_COLLATE=C`. Unlike R, `C()` and
`relevel()` also accept character vectors (R would require a factor).

## Differences from R and limitations

* **Thin plate splines (`tp`, `ts`).** mgcv's basis depends on the eigenvector signs from
  its Lanczos solver, which come from reference LAPACK. statsformula reimplements mgcv's Lanczos
  iteration, LAPACK `DSTEQR`/`DSTEDC` (including divide and conquer) and R's random number
  generator (used for knot sampling above 2,000 unique points), so the basis matches R's
  reference LAPACK build exactly. R linked against a vendor LAPACK (OpenBLAS, MKL, Accelerate)
  can itself produce a different, equally valid sign convention.
* **`gam.side()`** side constraints (mgcv drops columns when smooths share covariates, e.g.
  `s(x) + te(x, z)`) are applied exactly as mgcv does, using the smooths' penalty matrices.
  Penalties match R's `smoothCon(...)$S`, except the `cs` shrinkage penalty: mgcv assigns
  different values to the two null-space eigenvalues, and the null-space basis returned by
  LAPACK's eigen-solver is arbitrary, so that penalty agrees with R only to ~1e-3. Model matrices
  are not affected.
* mgcv bases not yet available: `t2`, `fs`, `gp`, `ds`, `so`, `mrf`, `ad`, `sos`, `sz`, `re`
  inside tensor products, smooths with matrix (summation convention) arguments; `id=` and `sp=`
  inside `s()` are accepted but ignored (pass `sp=` to `gam()`/`GAMRegressor` instead).
* `gam()`/`GAMRegressor` fit Gaussian identity-link models only (no `family=`, `select=`,
  `paraPen=`); offsets are subtracted when fitting but not added by `predict()`.
* R syntax that is not formula-related (`function`, `if`, assignment) is not supported inside
  formulas.
* By default, R's functions take precedence over Python objects of the same name (for example a
  Python `log` in your namespace does not replace R's `log`); use `sf.register_function(...,
  override=True)` to change that.

## License

GPL-2.0-or-later. statsformula contains translations of code from R (GPL-2 | GPL-3), mgcv
(GPL ≥ 2), Formula (GPL-2 | GPL-3) and Matrix (GPL ≥ 2), so it is distributed under the same
terms. The ported LAPACK routines are BSD-licensed.
