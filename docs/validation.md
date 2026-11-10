# Validation against R

statsformula is a reimplementation and claims identity with R, so it is tested against R output
rather than against its own expectations. Reference results are produced by R scripts in
`tools/` and stored as JSON fixtures in `tests/fixtures`. The test suite (about 2,000 tests)
compares every result with them. R is needed only to regenerate the fixtures:

| fixture | generator | content |
|---|---|---|
| `r_reference.json` | `tools/make_fixtures.py` (`r_reference.R`, `r_reference_mgcv.R`) | terms, model frames, model matrices, Formula, mgcv bases, penalties and GAM fits, `summary.lm`, `summary.gam` |
| `r_fuzz.json`, `r_fuzz_mm.json` | `tools/make_fuzz.py`, `tools/make_fuzz_mm.py` | randomly generated formulas: `terms()` and `model.matrix()` |
| `r_lavaan.json` | `tools/make_lavaan_fixtures.py` | lavaan parser and `lavaanify()` parameter tables |
| `r_sem.json` | `tools/make_sem_fixtures.py` | lavaan estimates for the SEM bridges |
| `r_path.json`, `r_pathse.json` | `tools/make_path_fixtures.py`, `tools/make_pathse_fixtures.py` | path models: `lm`, `lavaan::sem()` estimates, standard errors, effects, fit measures, R², printed summaries |
| `r_summary.json` | `tools/make_summary_fixtures.py` | Davies' algorithm (`psum.chisq`), `format`, `format.pval`, `printCoefmat` |

Reference versions: R 4.5.1, mgcv 1.9-3, Formula 1.2-5, Matrix 1.7-3, lavaan 0.7.3.

## What is compared

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

## Tolerances

| quantity | tolerance | reason |
|---|---|---|
| design matrices, terms, names, attributes | exact (values to ~1e-12) | deterministic translation of R's code |
| `summary.lm` | ~1e-10; printed output identical | same QR algorithm and formatting |
| Davies' algorithm, `format`, `printCoefmat` | 1e-12 / identical strings | direct translations |
| GAM smoothing parameters | ~1e-3 on the log scale | both are iterative optimizers with their own stopping rules; the REML surface is flat near the optimum |
| GAM fitted values, `Vp`, `s.table` | 1e-5 to 1e-7 | follows from the smoothing parameters |
| path models fitted by ML | ~1e-7 | lavaan's convergence tolerance |
| path models fitted equation by equation | ~1e-12 | closed form |

Two known exceptions are documented in [Limitations](limitations.md): the `cs` shrinkage
penalty depends on an arbitrary LAPACK eigenbasis, and smoothing parameters that diverge to
infinity stop at different finite values in the two implementations.

## Published benchmarks

The examples also reproduce published results that were not produced by R:

* the NIST StRD certified values for the Longley regression, to $10^{-13}$
  ([example](examples/10_longley_nist.md));
* Kmenta's (1986) 2SLS estimates of the food market, and Greene's 2SLS estimates of Klein's
  Model I ([example](examples/13_instrumental_variables.md));
* Greene's probit estimates for the Spector & Mazzeo data
  ([example](examples/09_greene_probit.md));
* lavaan's results for Bollen's Political Democracy model
  ([example](examples/12_bollen_political_democracy.md)).
