# Smooths and generalized additive models

## Smooth constructors

The smooth terms `s()`, `te()` and `ti()` are built exactly as mgcv's
`smooth.construct` methods and `smoothCon(..., absorb.cons = TRUE)` build them, including:

* the bases `tp`, `ts`, `cr`, `cs`, `cc`, `ps`, `cp`, `bs` and `re`; tensor products of any of
  them; numeric, factor and ordered `by=` variables;
* thin plate regression splines with mgcv's Lanczos eigen-decomposition. The eigenvector
  signs follow reference LAPACK (`DSTEQR`/`DSTEDC` are ported), and knot sub-sampling uses
  R's Mersenne-Twister for $n>2000$ unique points;
* identifiability constraints absorbed through a Householder QR of $C=\mathbf 1^\top X$, and
  the `gam.side()` deletions when smooths share covariates;
* the penalty matrices $S_j$ with mgcv's scaling, `rank` and smoothing-parameter names.

`sf.gam_model_matrix(formula, data)` is `model.matrix(gam(formula, data))`, and the spec it
returns evaluates `predict.gam(type = "lpmatrix")` on new data. The penalties are in
`X.model_spec.penalties` as {class}`~statsformula.GamPenalty` objects (`S`, `off`, `rank`,
`name`); `sf.penalty_matrix(penalties, p, sp)` gives $S_\lambda=\sum_j\lambda_jS_j$.

## Penalized least squares

`sf.gam()` and `GAMRegressor` fit the Gaussian identity-link model

$$
\hat\beta=\arg\min_\beta\;\lVert W^{1/2}(y-X\beta)\rVert^2+\beta^\top S_\lambda\beta,
\qquad \hat\beta=(X^\top WX+S_\lambda)^{-1}X^\top Wy,
$$

with influence matrix $F=(X^\top WX+S_\lambda)^{-1}X^\top WX$, effective degrees of freedom
$\operatorname{edf}=\operatorname{diag}(F)$ and $\tau=\operatorname{tr}F$.

**Smoothing-parameter selection.** Optimization is over $\rho=\log\lambda$, with analytic
gradients, by L-BFGS-B from three starting points (mgcv's initial values $\pm 5$). The
criteria are:

* **GCV** (`method="GCV.Cp"`):
  $\mathcal V_g=n\,\lVert W^{1/2}(y-X\hat\beta)\rVert^2/(n-\gamma\tau)^2$.
* **REML** (`method="REML"`), the restricted likelihood with the scale $\phi$ profiled out.
  With penalized deviance $D_p=\lVert W^{1/2}(y-X\hat\beta)\rVert^2+\hat\beta^\top
  S_\lambda\hat\beta$ and $M_p$ the dimension of the penalty null space,

  $$
  \mathcal V_r(\rho)=\frac{D_p}{2\phi}+\frac{n-M_p}{2}\log(2\pi\phi)
  +\tfrac12\log\lvert X^\top WX+S_\lambda\rvert-\tfrac12\log\lvert S_\lambda\rvert_+
  -\tfrac12\textstyle\sum_i\log w_i ,
  \qquad \hat\phi=\frac{D_p}{n-M_p}.
  $$

  $\log\lvert S_\lambda\rvert_+$ is the generalized determinant over the range space,
  computed blockwise per smooth. The value reported as `criterion` is mgcv's `gcv.ubre`
  (printed `-REML`).

Smoothing parameters given in `sp=` are held fixed; negative entries are estimated.

**Uncertainty.** The Bayesian posterior covariance is $V_p=(X^\top WX+S_\lambda)^{-1}\hat\phi$
(mgcv's `Vp`). Prediction standard errors are $\sqrt{\operatorname{diag}(X_pV_pX_p^\top)}$,
as `predict.gam(se.fit = TRUE)`.

## `summary.gam`

`GamFit.summary()` and `GAMRegressor.summary()` return a {class}`~statsformula.GamSummary`
with the tables of `summary.gam`.

**Parametric coefficients** (`p_table`):
$t=\hat\beta_j/\sqrt{(V_p)_{jj}}$ on $n-\sum\operatorname{edf}$ residual degrees of freedom.
`pterms_table` holds the Wald F test of every parametric term, with a rank-revealing
pseudo-inverse for multi-column terms (`anova.gam`).

**Smooth terms** (`s_table`): the tests of Wood (2013), ported from mgcv's `testStat`. For a
smooth with coefficients $\hat\beta_j$ and covariance block $V_j$, let $R$ be the factor of
the QR decomposition of the smooth's columns of $W^{1/2}X$, and let $r=\min(p_j,\operatorname{edf1}_j)$,
with $\operatorname{edf1}=\operatorname{diag}(2F-FF)$. The statistic is

$$
T_r=\hat\beta_j^\top R^\top (R V_j R^\top)^{r-}R\,\hat\beta_j ,
$$

where $(\cdot)^{r-}$ is a rank-$r$ pseudo-inverse. When $r$ is not an integer, the last two
eigen-directions are mixed so that the reference distribution is a weighted sum of
$\chi^2_1$ variables. Because the scale is estimated, the p-value is
$\Pr\big(\sum_i\nu_i\chi^2_{1,i}-\tfrac{T}{k}\chi^2_k>0\big)$ with
$k=\operatorname{round}(n-\sum\operatorname{edf})$. It is evaluated by **Davies' (1980)
algorithm** (`psum.chisq`, translated from mgcv's `davies.c`), or by the F distribution when
$r$ is an integer. `Ref.df` is $r$ and `F` is $T/r$.

Smooths whose penalty has no null space (`null.space.dim == 0` after the constraints are
absorbed: `re`, `cs`, `ts`, `cc`, `cp`, `m = 1`) use mgcv's random-effect test `reTest`:
$T=\lVert R_m\hat\beta_j\rVert^2/\hat\phi$, where $R_m$ is the factor of the penalized
Schur complement of the smooth's block. The eigenvalues of $B^\top R_m^\top R_mB/\hat\phi$
($BB^\top=V_{e,j}$, the frequentist covariance) weight the $\chi^2_1$ terms.

The adjusted $R^2$ and the deviance explained are
$1-\operatorname{var}(\sqrt w\,(y-\hat\mu))(n-1)/\{\operatorname{var}(\sqrt w\,(y-\bar y))\,
\nu\}$ and $1-D/D_0$. The printed layout is `print.summary.gam`.

```python
b = sf.gam("y ~ f + s(x) + te(x, z)", df, method="REML")
s = b.summary()
s.p_table, s.s_table, s.pterms_table, s.edf, s.r_sq, s.dev_expl, s.scale, s.residual_df
b.predict(new, se_fit=True)            # columns fit, se_fit
GAMRegressor("y ~ s(x) + s(z)").fit(df).predict(new, return_std=True)
```
