# Linear models and `summary.lm`

`FormulaEstimator(formula, LinearRegression(fit_intercept=False)).summary()` returns an
{class}`~statsformula.LmSummary`, a port of R's `summary.lm` and `print.summary.lm`.

## Estimation

The summary does not reuse scikit-learn's coefficients. It refits the training design with
R's algorithm, so that rank-deficient designs are handled as in `lm()`. Let $X$ be the
$n\times p$ design and $W=\operatorname{diag}(w)$ the prior weights ($W=I$ when the
estimator was fitted without `sample_weight`).

1. LINPACK `dqrdc2` with R's limited column pivoting (tolerance $10^{-7}$) factorizes
   $W^{1/2}X\Pi = QR$. Its rank $r$ determines the *aliased* columns: those pivoted beyond
   $r$ get coefficient `NA`, exactly as in R.
2. $\hat\beta_{\Pi[1:r]} = R_{11}^{-1}(Q^\top W^{1/2}y)_{1:r}$. Residuals are computed as
   `lm.fit` does, $e = W^{-1/2} Q\,(0_r, (Q^\top W^{1/2}y)_{r+1:n})$, so an exact fit gives
   residuals at the same rounding level as R.

With $\mathrm{RSS}=\sum_i w_i e_i^2$ and residual degrees of freedom $\nu=n-r$:

$$
\hat\sigma^2=\frac{\mathrm{RSS}}{\nu},\qquad
\widehat{\operatorname{Var}}(\hat\beta)=\hat\sigma^2\,(R_{11}^\top R_{11})^{-1},\qquad
t_j=\frac{\hat\beta_j}{\operatorname{se}(\hat\beta_j)}\sim t_\nu .
$$

Let $\mathrm{MSS}=\sum_i w_i(\hat y_i-\bar y_w)^2$, where $\bar y_w$ is the weighted mean of
the fitted values when the model has an intercept ($\mathrm{MSS}=\sum_i w_i\hat y_i^2$ without
one), and let $d=1$ if there is an intercept and $0$ otherwise. Then

$$
R^2=\frac{\mathrm{MSS}}{\mathrm{MSS}+\mathrm{RSS}},\qquad
\bar R^2=1-(1-R^2)\frac{n-d}{\nu},\qquad
F=\frac{\mathrm{MSS}/(r-d)}{\hat\sigma^2}\sim F_{r-d,\,\nu}.
$$

The log-likelihood is that of `logLik.lm`,
$\ell=\tfrac12\big(\sum_i\log w_i-n(\log 2\pi+1-\log n+\log\mathrm{RSS})\big)$, with
$\mathrm{AIC}=-2\ell+2(r+1)$ and $\mathrm{BIC}=-2\ell+(r+1)\log n$.

## Printed output

The printed summary follows `print.summary.lm` character for character. This needed ports of
R's numeric formatting:

* `format()` for doubles (`formatReal` and `scientific` in `src/main/format.c`): the choice
  between fixed and scientific notation, common decimals for a vector, `digits = 4`;
* `format.pval()` (`< 2e-16`, the `eps` rule);
* `printCoefmat()`: estimate and standard-error columns rounded jointly, the test column with
  `dig.tst`, significance stars (`symnum`), and the "Signif. codes" legend;
* `zapsmall()` for the residual quantiles.

```python
s = model.summary()
s.coefficients        # DataFrame: Estimate, Std. Error, t value, Pr(>|t|)
s.aliased             # Series of booleans, R's aliased
s.sigma, s.df         # residual standard error; (rank, residual df, columns)
s.r_squared, s.adj_r_squared, s.fstatistic, s.f_pvalue
s.loglik, s.aic, s.bic
s.cov_unscaled        # (X'WX)^-1 over the non-aliased columns
```

A response `cbind(y1, y2) ~ ...` gives an {class}`~statsformula.MlmSummary`: one
`LmSummary` per response, printed as `summary.mlm` ("Response y1 :").

## Other estimators

For any other estimator (Ridge, Lasso, logistic regression, trees, ...) `summary()` returns an
{class}`~statsformula.EstimatorSummary`. It shows the estimator, the number of observations
and of rows dropped for missing values, the training score (R² or accuracy) and the
coefficients when the estimator has `coef_`. No standard errors are reported: they are not
defined for penalized fits, and scikit-learn does not compute them. For likelihood models
with standard errors, use statsmodels on `sf.dmatrices()` (see the
[econometrics examples](../examples/index.md)).
