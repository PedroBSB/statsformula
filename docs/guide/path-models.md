# Path models

A {class}`~statsformula.sklearn.PathModel` is a system of regression equations written as a
multi-line formula, with one R formula per line:

```python
model = """
  # comments, continuation lines and ';' are allowed
  m  ~ x + f                   # mediator
  y  ~ m + x + log(w) +
       poly(z, 2)
  y2 + y3 ~ m                  # one equation per response (lavaan style)
  y  ~~ y2                     # free covariance of two disturbances
"""
pm = PathModel(model).fit(df)
```

Every right-hand side may use the full formula language (factors, `log()`, `poly()`, splines,
smooths). Variables that appear on a left-hand side are *endogenous*; the remaining
variables are *exogenous*.

## The linear system

When every equation is linear in the endogenous variables (each endogenous predictor enters
as its own design column), the model is the simultaneous system

$$
y_i = B\,y_i + G\,x_i + \zeta_i,\qquad \zeta_i\sim N(0,\Psi),
$$

where $y_i\in\mathbb R^k$ are the endogenous variables, $x_i$ the stacked design columns of
all equations (each equation has its own block, including its intercept), $B$ the
coefficients of endogenous on endogenous variables, $G$ those on exogenous columns, and
$\Psi$ the disturbance covariance: diagonal plus the pairs freed by `~~`. With $A=I-B$ and
$Z=YA^\top-XG^\top$, the log-likelihood conditional on $X$ is

$$
\ell(\theta)=n\log\lvert\det A\rvert-\frac n2\log\lvert\Psi\rvert-\frac12\operatorname{tr}
\big(\Psi^{-1}Z^\top Z\big)-\frac{nk}{2}\log 2\pi .
$$

This is the likelihood `lavaan::sem(fixed.x = TRUE, meanstructure = TRUE)` maximizes for
observed-variable models, and the `logl` lavaan reports.

## Estimation

`estimation="auto"` (the default) chooses between two estimators:

* **Equation by equation.** This applies when the system is *recursive* (no feedback loops:
  the directed graph of $B$ is acyclic) and $\Psi$ is diagonal. Each equation is fitted on its
  own complete cases with a clone of `estimator`. The default is
  `LinearRegression(fit_intercept=False)`, which gives the ML estimate, or `GAMRegressor` for
  equations with smooths. Any scikit-learn estimator may be used, per equation through a
  dict.
* **Full-information maximum likelihood.** This applies when the system has feedback loops
  ($\det A$ depends on $B$) or correlated disturbances, or when `estimation="ML"` is set. The
  rows complete for all equations are used. The maximization is Fisher scoring with
  step-halving,
  $\theta_{t+1}=\theta_t+\alpha_t\,\mathcal I(\theta_t)^{-1}\nabla\ell(\theta_t)$, from three
  starting values: $B=0$ (lavaan's default), two-stage least squares, and OLS. The best
  converged solution is kept. Several starts are needed because the regions
  $\det A>0$ and $\det A<0$ are separated by the singular surface $\det A=0$, so a single start
  can converge to the wrong local maximum.

`estimation="OLS"` forces the per-equation fit, which is inconsistent for non-recursive
models (simultaneity bias), as the [instrumental variables example](../examples/13_instrumental_variables.md)
shows.

## Standard errors

`se="expected"` (the default) uses the structured expected information of lavaan, with the
exogenous moments at their sample values. Writing
$\Omega=A^{-1}\Psi A^{-\top}$ for the reduced-form covariance and
$\mu_i=A^{-1}Gx_i$, the information for parameters $p,q$ is

$$
\mathcal I_{pq}=\sum_i\Big(\frac{\partial\mu_i}{\partial\theta_p}\Big)^{\!\top}\Omega^{-1}
\frac{\partial\mu_i}{\partial\theta_q}+\frac n2\operatorname{tr}\Big(\Omega^{-1}
\frac{\partial\Omega}{\partial\theta_p}\Omega^{-1}\frac{\partial\Omega}{\partial\theta_q}\Big),
$$

with $\partial\mu/\partial G_{ic}=A^{-1}e_ix_c$ and
$\partial\mu/\partial B_{ij}=A^{-1}e_i(A^{-1}Gx)_j$. `se="observed"` uses the observed
information, from central differences of the analytic gradient. `se="lm"` gives the t tests
of `summary(lm())` for each equation; `se="none"` skips inference. When the model is not a
single linear system on shared rows (for example with smooths, a transformed response, or
equations with different missing-value patterns), each equation uses its own observed
information $\hat\sigma^2_{ML}(X^\top X)^{-1}$, or mgcv's $V_p$ for GAM equations.
`information_` records which method was used.

`coefficients()` returns lavaan's `parameterEstimates()` table: `lhs`, `op` (`~`, `~~`, `~1`),
`rhs`, `est`, `se`, `z`, `pvalue`, `ci.lower`, `ci.upper`, in lavaan's row order. `vcov()`
returns the covariance matrix.

## Direct, indirect and total effects

With $D=[B\;\;G]$ restricted to variable columns, the total effects are
$T=(I-B)^{-1}D$ and the indirect effects are $T-D$. For a recursive system these are the
sums over all directed paths; with feedback loops they include the loop multiplier. Standard
errors use the delta method, $\operatorname{Var}(\hat\tau)=J\,\widehat V J^\top$. The
Jacobian is analytic: with $m$ variable columns and
$E=I_m+\begin{pmatrix}T\\0\end{pmatrix}$ (the identity with $T$ added to the rows of the
endogenous variables),

$$
\frac{\partial T}{\partial D_{ic}}=(A^{-1})_{\cdot i}\,E_{c\cdot},
$$

so the effects and their standard errors agree with lavaan's `:=` definitions:

```python
pm.effects("indirect")               # matrix: rows endogenous, columns all variables
pm.effects("total", se=True)         # long table: lhs, rhs, est, se, z, pvalue, ci
```

## Fit measures and summary

For a linear system, `fit_measures()` returns lavaan's `fitMeasures()`. Here $\ell_0$ is the
model log-likelihood, $\ell_1$ that of the saturated model (multivariate regression of $y$ on
all exogenous columns), $\ell_b$ that of the baseline (independent endogenous variables),
$q$ the number of distinct exogenous columns, and $\mathrm{npar}$ the number of free
parameters.

| measure | definition |
|---|---|
| `chisq`, `df` | $\chi^2=2(\ell_1-\ell_0)$; $\mathrm{df}=k(k+1)/2+k(q+1)-\mathrm{npar}$ |
| `baseline.chisq`, `baseline.df` | $2(\ell_1-\ell_b)$; $\mathrm{df}_b=k(k+1)/2+k(q+1)-2k$ |
| `cfi` | $1-\max(\chi^2-\mathrm{df},0)/\max(\chi^2-\mathrm{df},\chi^2_b-\mathrm{df}_b,0)$ |
| `tli` | $(\chi^2_b/\mathrm{df}_b-\chi^2/\mathrm{df})/(\chi^2_b/\mathrm{df}_b-1)$ |
| `logl`, `unrestricted.logl` | $\ell_0$, $\ell_1$ |
| `aic`, `bic`, `bic2` | $-2\ell_0+c\,\mathrm{npar}$ with $c=2$, $\log n$, $\log((n+2)/24)$ |
| `rmsea` | $\sqrt{\max(\chi^2/(n\,\mathrm{df})-1/n,\,0)}$ |
| `rmsea.ci.lower/upper` | $\sqrt{\lambda/(n\,\mathrm{df})}$, where $\lambda$ solves $F_{\chi^2_{\mathrm{df}}(\lambda)}(\chi^2)=0.95$ / $0.05$ |
| `rmsea.pvalue`, `rmsea.notclose.pvalue` | tests of $\mathrm{RMSEA}\le0.05$ and $\ge0.08$ (non-central $\chi^2$) |
| `srmr` | root mean square of the standardized covariance and mean residuals (Bentler) |

`rsquare()` is $1-\psi_{ii}/\sigma_{ii}$ with the model-implied
$\Sigma_{yy}=A^{-1}(GS_{xx}G^\top+\Psi)A^{-\top}$. `summary()` prints all of this in lavaan's
layout. `summary(effects=True)` appends the indirect and total effects. GAM equations are
summarized by mgcv's smooth-term tests instead of their basis coefficients.

## Prediction

`predict_mode="propagate"` (the default) uses only the exogenous variables of the new data.
In a recursive model each equation's prediction feeds the equations downstream; for a
non-recursive model the reduced form $\hat y=(I-B)^{-1}\hat Gx$ is used.
`predict_mode="direct"` uses the observed values of every predictor.
