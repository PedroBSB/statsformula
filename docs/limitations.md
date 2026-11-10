# Differences from R and limitations

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
