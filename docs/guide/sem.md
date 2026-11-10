# Structural equation models

## lavaan model syntax

`sf.lavaan_parse(model)` is a port of `lavaan::lavParseModelString(as.data.frame. = TRUE)`,
and `sf.lavaanify(model, preset=, **options)` of `lavaan::lavaanify()` (lavaan 0.7.3). They
reproduce the full parameter table that `cfa()`, `sem()` and `growth()` build: identification
constraints, default (co)variances, intercepts, labels, equality constraints, multiple
groups and `:=` definitions.

Supported syntax:
* all operators (`=~ ~ ~~ ~1 | ~*~ <~ := == < > :~ %`);
* modifiers (`a*x`, `0.5*x`, `NA*x`, `start()`, `label()`, `equal()`, `lower()`,
  `upper()`, `c()` vectors over groups, `efa("set")*`);
* interactions, and `group:`/`level:`/`block:` sections.

Supported options: `meanstructure`, `int.ov.free`, `int.lv.free`, `std.lv`, `auto.fix.first`,
`auto.var`, `auto.cov.lv.x`, `auto.cov.y`, `fixed.x`, `conditional.x`, `orthogonal`,
`group.equal`, `group.partial`, `parameterization`, `ngroups` and others. Option names can be
given as `std_lv=True` or `**{"std.lv": True}`.

## R expressions inside SEMs

`statsformula.sem.sem_data(model, data)` rewrites the R expressions of a lavaan model into
data columns, so that a backend that only understands variable names can fit models such as:

```text
ind =~ x1 + x2 + x3
m ~ a*ind + log(w) + f + z          # log() and a factor f
y ~ b*m + ind + I(z^2)
ab := a*b
```

`log(w)` becomes a column `log_w`, factor `f` becomes its contrast columns, and `poly()`,
interactions and `I()` are evaluated. The stored transformations apply to new data
(`sd.transform(new)`).

## Backends

| function | backend | notes |
|---|---|---|
| `fit_semopy` | semopy | ML (`obj="MLW"`), tight SLSQP tolerance; standard errors and fit statistics from semopy |
| `fit_pysem` | pySEM (twoertwein/pySEM) | LISREL all-y translation; L-BFGS fit to convergence (PyTorch) |
| `fit_pysemforge` | pysemforge 0.1 | warns about known backend defects |

Each bridge passes lavaan's full parameter table to the backend, so the backend fits exactly
the model lavaan would, not its own default specification. The estimates come back as a
{class}`~statsformula.sem.SemFit`: `estimates` (lavaan columns), `defined` (`:=` values),
`fit_stats`, and `summary()` in lavaan's layout.

Agreement with lavaan on the validation models is about $2\times10^{-5}$ (semopy) and
$5\times10^{-5}$ (pySEM). See the [Holzinger–Swineford](../examples/07_sem.md) and
[Bollen](../examples/12_bollen_political_democracy.md) examples.
