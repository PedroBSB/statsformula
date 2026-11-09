# Changelog

## 0.1.0 (2026-11-09)

First release.

* Parser and deparser for R formulas (R operator precedence, `**` → `^`, backticks).
* Port of R 4.5 `termsform`: all formula operators, `.` expansion, offsets, specials, keep.order, simplify.
* `model.frame` (subset, weights, offset, na.action, drop.unused.levels, xlev) and `model.matrix`
  (port of `modelmatrix`: contrasts, column names, assign attribute).
* `sparse.model.matrix` (Matrix package conventions), `model.response/offset/weights/extract`.
* `update`, `reformulate`, `drop.terms`, `delete.response`, `all.vars`, `all.names`, `get_all_vars`.
* Formula package: multi-part formulas, `model.part`, `as.Formula`, `update.Formula`.
* R functions inside formulas, including `poly`, `bs`, `ns`, `scale`, `C`, `cut`, `interaction`, `relevel`.
* mgcv `s()`, `te()`, `ti()` with `tp`, `ts`, `cr`, `cs`, `cc`, `ps`, `cp`, `bs`, `re` bases and `by` variables.
* Stateful transforms re-applied to new data (`ModelSpec`, `GamSpec`).
* Validated against R 4.5.1: 367 reference cases, 500 random `terms()` and 400 random `model.matrix()` formulas.
