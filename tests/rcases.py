"""Shared datasets and the list of cases compared against R.

The same datasets are written to CSV for R (``tools/make_fixtures.py``) and
rebuilt in Python by the tests.
"""

import numpy as np
import pandas as pd

SCHEMAS = {}


def make_datasets():
    rng = np.random.default_rng(20261108)
    n = 40
    d1 = pd.DataFrame({
        "y": np.round(rng.normal(10, 2, n), 4),
        "x": np.round(rng.uniform(0.5, 9.5, n), 4),
        "z": np.round(rng.normal(0, 1, n), 4),
        "w": np.round(rng.uniform(1, 3, n), 4),
        "k": rng.integers(0, 6, n).astype(np.int64),
        "f": pd.Categorical(rng.choice(["a", "b", "c"], n), categories=["a", "b", "c"]),
        "g": pd.Categorical(rng.choice(["u", "v"], n), categories=["u", "v"]),
        "h": rng.choice(["Alpha", "beta", "Gamma", "delta"], n).astype(object),
        "o": pd.Categorical(rng.choice(["lo", "mid", "hi"], n), categories=["lo", "mid", "hi"], ordered=True),
        "l": rng.choice([True, False], n),
        "y2": np.round(rng.normal(0, 1, n), 4),
        "cnt": rng.integers(1, 20, n).astype(np.int64),
    })
    d1.loc[[0, 1, 2], "f"] = ["a", "b", "c"]
    d1.loc[[0, 1], "g"] = ["u", "v"]
    d1.loc[[0, 1, 2, 3], "h"] = ["Alpha", "beta", "Gamma", "delta"]
    d1.loc[[0, 1, 2], "o"] = ["lo", "mid", "hi"]
    d1.loc[[0, 1], "l"] = [True, False]

    d2 = d1.copy()
    d2.loc[[3, 7], "x"] = np.nan
    d2.loc[5, "f"] = np.nan
    d2.loc[11, "h"] = None
    d2.loc[13, "y"] = np.nan

    d3 = d1.iloc[:12].copy().reset_index(drop=True)
    d3.loc[[0, 1, 2], "x"] = [0.2, 9.9, 5.0]
    d3["f"] = pd.Categorical(["c", "b", "a", "a", "b", "c", "a", "a", "b", "b", "a", "c"], categories=["a", "b", "c"])
    d3["h"] = ["delta", "beta", "Alpha", "Gamma"] * 3

    # data with unused factor level, non-syntactic names
    d4 = pd.DataFrame({
        "y": np.round(rng.normal(0, 1, 10), 4),
        "my var": np.round(rng.uniform(0, 1, 10), 4),
        "f": pd.Categorical(list("aabbaabbaa"), categories=["a", "b", "zz"]),
        "x.1": np.round(rng.uniform(0, 1, 10), 4),
    })
    dx = pd.DataFrame({
        "Y": np.round(rng.normal(5, 1, 30), 4),
        "X1": pd.Categorical(rng.choice(["p", "q", "r"], 30), categories=["p", "q", "r"]),
        "X2": pd.Categorical(rng.choice(["s", "t"], 30), categories=["s", "t"]),
        "X3": np.round(rng.uniform(1, 5, 30), 4),
        "X4": np.round(rng.uniform(1, 5, 30), 4),
        "X5": np.round(rng.normal(0, 1, 30), 4),
    })
    dx.loc[[0, 1, 2], "X1"] = ["p", "q", "r"]
    dx.loc[[0, 1], "X2"] = ["s", "t"]
    nb = 2500
    d5 = pd.DataFrame({"y": np.round(rng.normal(0, 1, nb), 4), "x": np.round(rng.uniform(0, 10, nb), 5),
                       "z": np.round(rng.normal(0, 1, nb), 5)})
    # smooth signal for penalized GAM fits
    n6 = 150
    x6 = np.round(rng.uniform(0, 10, n6), 4)
    z6 = np.round(rng.normal(0, 1, n6), 4)
    f6 = pd.Categorical(rng.choice(["a", "b", "c"], n6), categories=["a", "b", "c"])
    eff = np.array([0.0, 0.8, -0.5])[f6.codes]
    d6 = pd.DataFrame({"x": x6, "z": z6, "f": f6, "w": np.round(rng.uniform(0.5, 2, n6), 4),
                       "y": np.round(np.sin(x6 / 1.5) + 0.4 * z6 ** 2 + 0.2 * x6 * z6 / 5 + eff
                                     + rng.normal(0, 0.3, n6), 4)})
    d6new = d6.iloc[:10].copy().reset_index(drop=True)
    d6new["x"] = np.round(np.linspace(0.5, 9.5, 10), 4)
    return {"d1": d1, "d2": d2, "d3": d3, "d4": d4, "dx": dx, "d4small": d4, "d5": d5, "d6": d6, "d6new": d6new}


def schema(df):
    out = {}
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            out[c] = {"type": "ordered" if s.cat.ordered else "factor", "levels": list(map(str, s.cat.categories))}
        elif s.dtype == bool:
            out[c] = {"type": "logical"}
        elif s.dtype.kind in "iu":
            out[c] = {"type": "integer"}
        elif s.dtype.kind == "f":
            out[c] = {"type": "numeric"}
        else:
            out[c] = {"type": "character"}
    return out


T = []  # terms cases (no data)
for f in [
    "y ~ x", "y ~ x + z", "y ~ x*z", "y ~ x:z", "y ~ x/z", "y ~ z %in% x", "y ~ (x + z + w)^2",
    "y ~ (x + z + w)^3", "y ~ x*z*w - x:z:w", "y ~ x*z - x", "y ~ x - 1", "y ~ 0 + x", "y ~ x + 0",
    "y ~ -1 + x", "y ~ 1", "y ~ 0", "y ~ -1", "~ x", "~ x - 0", "~ 0 + x + 1", "y ~ x^2", "y ~ I(x^2)",
    "y ~ x**2", "Y~C(X1):C(X2)+log(X3)/sqrt(X4)+X5**4", "y ~ log(x) + exp(z) + sqrt(w)",
    "y ~ poly(x, 2) + z", "y ~ offset(w) + x", "y ~ x + offset(log(w)) + offset(z)",
    "y ~ offset(w) + x + offset(w):x", "log(y) ~ x", "cbind(y, y2) ~ x", "y ~ y + x", "y ~ y:x",
    "y ~ x - z", "y ~ a - b + c", "y ~ b:a + a", "y ~ a:b + a", "y ~ a*b*c - a:b + d %in% a",
    "y ~ a + b %in% a", "y ~ a/(b + c)", "y ~ (a + b)/c", "y ~ a/b/c", "y ~ (a + b)^2 - a:b",
    "y ~ a*(b + c)", "y ~ (a + b) * (c + d)", "y ~ a:b:c", "y ~ (a + b + c + d)^2 - (a + b)^2",
    "y ~ x | z", "y ~ `my var` + x.1", "y ~ I(x + z) + I(x * z) + I(x/z) + I(x - z)", "y ~ x + x",
    "y ~ (x)", "y ~ ((x + z))^2", "y ~ x + TRUE", "y ~ x - TRUE", "y ~ x + z - z", "y ~ log(x, base = 2)",
    "y ~ x + (z + w) - w", "y ~ 1 + x + 1", "y ~ x + 0 + 1", "y ~ x*z - 1", "y ~ (a+b)^2.7",
    "y ~ x^1", "y ~ a %in% b %in% c", "y ~ x:(z + w)", "y ~ -x", "y ~ - (x + z)", "~ -1",
    "y ~ factor(x) + C(z, sum)", "y ~ a + b + a:b + b:a", "y ~ a*b + b*a", "y ~ x + log(x) + I(log(x))",
    "y ~ s(x) + s(z, k = 5)", "y ~ x %in% z", "y ~ 2", "y ~ x + 3",
]:
    T.append({"kind": "terms", "formula": f})
T.append({"kind": "terms", "formula": "y ~ (a + b)^2 + c", "keep_order": True})
T.append({"kind": "terms", "formula": "y ~ a*b + c", "keep_order": True})
T.append({"kind": "terms", "formula": "y ~ x + s(z) + s(w)", "specials": ["s"]})
for f in ["y ~ .", "log(y) ~ .", "~ .", "y ~ . - x", "y ~ .^2", "y ~ x * .", "y ~ log(x) + .",
          "cbind(y, y2) ~ .", "y ~ . - 1", "y ~ (.)"]:
    T.append({"kind": "terms", "formula": f, "data": "d4small"})

MM = []
for f, data, extra in [
    ("y ~ x", "d1", {}), ("y ~ x + z", "d1", {}), ("y ~ f", "d1", {}), ("y ~ f - 1", "d1", {}),
    ("y ~ 0 + f + g", "d1", {}), ("y ~ f + g", "d1", {}), ("y ~ f*g", "d1", {}), ("y ~ f:g", "d1", {}),
    ("y ~ f:g - 1", "d1", {}), ("y ~ f*x", "d1", {}), ("y ~ f:x", "d1", {}), ("y ~ x:f - 1", "d1", {}),
    ("y ~ f/x", "d1", {}), ("y ~ f/x - 1", "d1", {}), ("y ~ x %in% f", "d1", {}), ("y ~ g/f", "d1", {}),
    ("y ~ f*g*o", "d1", {}), ("y ~ (f + g + o)^2", "d1", {}), ("y ~ f:g:o", "d1", {}),
    ("y ~ o", "d1", {}), ("y ~ h", "d1", {}), ("y ~ l", "d1", {}), ("y ~ l:x", "d1", {}),
    ("y ~ l*f", "d1", {}), ("y ~ h*g", "d1", {}), ("y ~ C(f, sum)", "d1", {}),
    ("y ~ C(f, helmert)", "d1", {}), ("y ~ C(f, poly)", "d1", {}), ("y ~ C(f, SAS)", "d1", {}),
    ("y ~ C(f, treatment)", "d1", {}), ("y ~ C(f, contr.sum)", "d1", {}), ("y ~ C(f, \"contr.helmert\")", "d1", {}),
    ("y ~ C(f, sum, 1)", "d1", {}), ("y ~ C(f, treatment, 2)", "d1", {}), ("y ~ C(f, contr.treatment(3, base = 2))", "d1", {}),
    ("y ~ C(h)", "d1", {}), ("y ~ C(k)", "d1", {}), ("y ~ C(o)", "d1", {}), ("y ~ C(g):C(f)", "d1", {}),
    ("Y~C(X1):C(X2)+log(X3)/sqrt(X4)+X5**4", "dx", {}),
    ("y ~ C(f, sum):x", "d1", {}), ("y ~ C(f, sum)*C(g, helmert)", "d1", {}),
    ("y ~ factor(k)", "d1", {}), ("y ~ as.factor(k) + x", "d1", {}), ("y ~ factor(f, levels = c(\"c\", \"b\", \"a\"))", "d1", {}),
    ("y ~ relevel(f, ref = \"b\")", "d1", {}), ("y ~ relevel(f, \"c\") * g", "d1", {}), ("y ~ ordered(k)", "d1", {}),
    ("y ~ factor(h, labels = c(\"A\", \"B\", \"C\", \"D\"))", "d1", {}), ("y ~ interaction(f, g)", "d1", {}),
    ("y ~ interaction(f, g, sep = \"_\", lex.order = TRUE)", "d1", {}), ("y ~ cut(x, 3)", "d1", {}),
    ("y ~ cut(x, breaks = c(0, 2.5, 5, 10))", "d1", {}), ("y ~ cut(x, c(0, 5, 10), right = FALSE)", "d1", {}),
    ("y ~ cut(z, 4, labels = FALSE)", "d1", {}), ("y ~ cut(x, c(0.5, 3, 9.5), include.lowest = TRUE)", "d1", {}),
    ("y ~ log(x) + exp(z) + sqrt(w) + abs(z)", "d1", {}), ("y ~ log(x, base = 2) + log2(x) + log10(x) + log1p(x)", "d1", {}),
    ("y ~ I(x^2) + I(x^3)", "d1", {}), ("y ~ x + I(x^2) + I(x * z) + I(x/z) + I(x - z) + I(-x)", "d1", {}),
    ("y ~ I(x > 5)", "d1", {}), ("y ~ I(x > 5) + I(z <= 0)", "d1", {}), ("y ~ I(f == \"a\")", "d1", {}),
    ("y ~ I(x %% 2) + I(x %/% 2)", "d1", {}), ("y ~ I(k %% 2L) + I(k %/% 3L)", "d1", {}), ("y ~ I(f %in% c(\"a\", \"b\"))", "d1", {}),
    ("y ~ I((x - mean(x))/sd(x))", "d1", {}), ("y ~ scale(x)", "d1", {}), ("y ~ scale(x, scale = FALSE)", "d1", {}),
    ("y ~ scale(x, center = 2, scale = 3)", "d1", {}), ("y ~ ifelse(x > 5, 1, 0) + pmin(x, 5) + pmax(z, 0)", "d1", {}),
    ("y ~ floor(x) + ceiling(z) + round(w, 1) + trunc(z) + sign(z)", "d1", {}), ("y ~ sin(x) + cos(x) + tan(z) + atan(z)", "d1", {}),
    ("y ~ exp(z):f", "d1", {}), ("y ~ log(x):sqrt(w)", "d1", {}), ("y ~ log(x)/sqrt(w)", "d1", {}),
    ("y ~ poly(x, 2)", "d1", {}), ("y ~ poly(x, 3) + f", "d1", {}), ("y ~ poly(x, degree = 3, raw = TRUE)", "d1", {}),
    ("y ~ poly(x, 2):f", "d1", {}), ("y ~ poly(x, z, degree = 2)", "d1", {}), ("y ~ poly(x, z, degree = 2, raw = TRUE)", "d1", {}),
    ("y ~ splines::bs(x, df = 5)", "d1", {}), ("y ~ bs(x, df = 5)", "d1", {}), ("y ~ bs(x, knots = c(3, 6))", "d1", {}),
    ("y ~ bs(x, degree = 2, df = 4, intercept = TRUE) - 1", "d1", {}), ("y ~ ns(x, df = 4)", "d1", {}),
    ("y ~ ns(x, knots = c(2, 5, 8))", "d1", {}), ("y ~ ns(x, 3) * g", "d1", {}), ("y ~ ns(x, df = 3, intercept = TRUE)", "d1", {}),
    ("y ~ ns(x, df = 3, Boundary.knots = c(0, 10))", "d1", {}), ("y ~ bs(x, df = 4, Boundary.knots = c(0, 10))", "d1", {}),
    ("cbind(y, y2) ~ x", "d1", {}), ("log(y) ~ x", "d1", {}), ("y ~ cbind(x, z)", "d1", {}), ("y ~ cbind(x, z):f", "d1", {}),
    ("y ~ cbind(x, log(z + 5))", "d1", {}), ("y ~ x + offset(w)", "d1", {}), ("y ~ x + offset(log(w))", "d1", {}),
    ("y ~ 1", "d1", {}), ("y ~ 0", "d1", {}), ("y ~ -1", "d1", {}), ("~ x + f", "d1", {}), ("~ 1", "d1", {}),
    ("y ~ x + y", "d1", {}), ("y ~ y:x", "d1", {}), ("y ~ x*z*w", "d1", {}), ("y ~ (x + z + w)^2", "d1", {}),
    ("y ~ f + f:g", "d1", {}), ("y ~ g + f:g", "d1", {}), ("y ~ f:g + f", "d1", {}), ("y ~ x + f:g", "d1", {}),
    ("y ~ f:g:x", "d1", {}), ("y ~ f*g - f", "d1", {}), ("y ~ f*g - 1", "d1", {}), ("y ~ x + f - 1", "d1", {}),
    ("y ~ x:z + f", "d1", {}), ("y ~ f %in% g", "d1", {}), ("y ~ o:f", "d1", {}), ("y ~ o*l", "d1", {}),
    ("y ~ .", "d4", {}), ("y ~ . - x.1", "d4", {}), ("y ~ .^2", "d4", {}), ("y ~ `my var` + f", "d4", {}),
    ("y ~ f", "d4", {}), ("y ~ f - 1", "d4", {}), ("y ~ x + f + h", "d2", {}), ("y ~ log(x) * f", "d2", {}),
    ("y ~ f", "d1", {"contrasts": {"f": "contr.sum"}}), ("y ~ f*g", "d1", {"contrasts": {"f": "contr.helmert", "g": "contr.sum"}}),
    ("y ~ o", "d1", {"contrasts": {"o": "contr.treatment"}}), ("y ~ f + h", "d1", {"contrasts": {"h": "contr.SAS"}}),
    ("y ~ f + o", "d1", {"contrasts": {"f": "contr.poly"}}), ("y ~ l", "d1", {"contrasts": {"l": "contr.sum"}}),
    ("y ~ f:x - 1", "d1", {"contrasts": {"f": "contr.sum"}}), ("y ~ x", "d1", {"contrasts": {"q": "contr.sum"}}),
    ("y ~ f", "d1", {"contrasts_matrix": {"f": [[1, 0], [0, 1], [-1, -1]]}}),
    ("y ~ f", "d1", {"contrasts_matrix": {"f": [[-1], [0], [1]]}}),
    ("y ~ k + I(k^2) + I(k/2)", "d1", {}), ("y ~ x + I(x < 3 | x > 7)", "d1", {}), ("y ~ I(x > 3 & z < 0)", "d1", {}),
    ("y ~ I(!l)", "d1", {}), ("y ~ as.numeric(f)", "d1", {}), ("y ~ as.integer(x)", "d1", {}),
    ("y ~ x + z", "d1", {"options_contrasts": ["contr.sum", "contr.poly"]}),
    ("y ~ f*o", "d1", {"options_contrasts": ["contr.helmert", "contr.treatment"]}),
    ("y ~ I(x^-1) + I(2^z) + I(-x^2)", "d1", {}), ("y ~ exp(-z^2/2)", "d1", {}), ("y ~ x + I(1:40)", "d1", {}),
    ("y ~ f + droplevels(f)", "d1", {}), ("y ~ log(cnt) + sqrt(cnt)", "d1", {}),
    ("y ~ tanh(z) + expm1(z/10) + asinh(z)", "d1", {}),
]:
    c = {"kind": "mm", "formula": f, "data": data}
    c.update(extra)
    MM.append(c)

SPARSE = [{"kind": "sparse", "formula": f, "data": "d1"} for f in [
    "~ f", "~ f*g", "~ f*x", "~ f:g - 1", "~ x + z + f:g", "~ poly(x, 2) + h", "~ o + l", "~ f/x", "~ 0 + f:x",
]]

UPD = [{"kind": "update", "old": o, "new": n} for o, n in [
    ("y ~ x", "~ . + z"), ("y ~ x", ". ~ . - x"), ("y ~ x + z", ". ~ .:w"), ("log(y) ~ x", "sqrt(.) ~ .:w"),
    ("y ~ a + b", ". ~ .^2 - 1"), ("y ~ x + offset(o)", ". ~ . + z"), ("y ~ x | z", ". ~ . + w"),
    ("y ~ a*b", ". ~ . - a:b"), ("y ~ a*b", "~ . + c"), ("y ~ x", "log(.) ~ ."), ("y ~ x", "z ~ ."),
    ("~ x", "y ~ ."), ("y ~ x + z", ". ~ . - 1"), ("y ~ x - 1", ". ~ . + 1"), ("y ~ a + b", ". ~ (.)*c"),
    ("y ~ a + b", ". ~ .*c"), ("y ~ a + b", ". ~ .:c"), ("y ~ a + b", ". ~ c/."), ("y ~ a", ". ~ . + b + a"),
    ("y ~ x", "cbind(., z) ~ ."), ("y ~ a + b", ". ~ . %in% c"), ("y ~ (a+b)^2", ". ~ . - a:b + c"),
]]

REFORM = [
    {"kind": "reformulate", "labels": ["a", "b"], "response": None, "intercept": True},
    {"kind": "reformulate", "labels": ["a", "b:c"], "response": "y", "intercept": True},
    {"kind": "reformulate", "labels": ["a", "log(b)"], "response": "log(y)", "intercept": False},
    {"kind": "reformulate", "labels": [], "response": "y", "intercept": True},
    {"kind": "reformulate", "labels": [], "response": "y", "intercept": False},
    {"kind": "drop_terms", "formula": "y ~ a + b + a:b + c", "dropx": [2], "keep_response": True},
    {"kind": "drop_terms", "formula": "y ~ a*b*c", "dropx": [1, 4], "keep_response": False},
    {"kind": "drop_terms", "formula": "y ~ a + b - 1", "dropx": [1], "keep_response": True},
    {"kind": "delete_response", "formula": "y ~ a*b + offset(w)"},
    {"kind": "delete_response", "formula": "log(y) ~ a + s(b)", "specials": ["s"]},
    {"kind": "simplify", "formula": "y ~ a*b - a + offset(w)"},
    {"kind": "simplify", "formula": "y ~ (a+b)^2 - 1"},
    {"kind": "simplify", "formula": "~ a/b"},
    {"kind": "allvars", "formula": "log(y) ~ x + poly(z, 2) + I(w^2) + f:g + offset(o)"},
]

FORM = [{"kind": "Formula", "formula": f, "data": "d1"} for f in [
    "y ~ x | z", "y ~ x + z | f + g", "y | y2 ~ x | f", "y + y2 ~ x", "y ~ x + z | f | g*x",
    "y | y2 ~ x * f | log(w) + g", "y ~ x | 0 + f", "y ~ x | . - x", "y ~ log(x) | poly(z, 2) | f:g",
    "cbind(y, y2) ~ x | f",
]]
FORM_UPDATE = [{"kind": "Formula_update", "old": o, "new": n} for o, n in [
    ("y ~ x | z", ". ~ . + w | ."), ("y ~ x | z", ". ~ . | . + w | v"), ("y | y2 ~ x", ". | log(.) ~ . + z"),
    ("y ~ x + z | f", ". ~ . - z | . - 1"),
]]
FORM_COMBINE = [{"kind": "Formula_combine", "parts": ["y ~ x", "~ z", "w ~ f"]}]

PRED = [{"kind": "predict", "formula": f, "data": "d1", "newdata": "d3"} for f in [
    "y ~ poly(x, 3) + f", "y ~ ns(x, df = 4) + g", "y ~ bs(x, df = 5)*g", "y ~ scale(x) + h", "y ~ C(f, sum) + o",
    "y ~ f*x + l", "y ~ poly(x, z, degree = 2)", "y ~ ns(x, knots = c(3, 6), Boundary.knots = c(1, 9))",
    "y ~ bs(x, knots = c(3, 6), Boundary.knots = c(1, 9))", "y ~ I((x - mean(x))/sd(x))",
]]

MF = [
    {"kind": "mf", "formula": "y ~ x + f", "data": "d2", "na_action": "na.omit"},
    {"kind": "mf", "formula": "y ~ x + f", "data": "d2", "na_action": "na.exclude"},
    {"kind": "mf", "formula": "y ~ x + f", "data": "d2", "na_action": "na.fail"},
    {"kind": "mf", "formula": "y ~ x + offset(log(w))", "data": "d1", "weights": "w", "offset": "z"},
    {"kind": "mf", "formula": "y ~ x + f", "data": "d1", "subset": "x > 3 & g == \"u\""},
    {"kind": "mf", "formula": "cbind(y, y2) ~ x", "data": "d1"},
    {"kind": "mf", "formula": "y ~ f", "data": "d4", "drop_unused_levels": True},
    {"kind": "mf", "formula": "y ~ f + h", "data": "d1", "weights": "cnt"},
    {"kind": "mf", "formula": "log(y) ~ poly(x, 2) + I(z^2) + l + o", "data": "d1"},
]

ERR = [{"kind": "mm", "formula": f, "data": "d1"} for f in [
    "y ~ nosuchvar", "y ~ x^1", "y ~ (x + z)^-1", "y ~ x + 2", "y ~ poly(x, 50)", "y ~ C(z)",
]]

ALL_CASES = T + MM + SPARSE + UPD + REFORM + FORM + FORM_UPDATE + FORM_COMBINE + PRED + MF + ERR

GAM = []
for f, extra in [
    ("y ~ s(x)", {"newdata": "d3"}), ("y ~ s(x, k = 5)", {}), ("y ~ s(x, bs = \"cr\")", {"newdata": "d3"}),
    ("y ~ s(x, bs = \"cs\", k = 6)", {}), ("y ~ s(x, bs = \"cc\")", {"newdata": "d3"}), ("y ~ s(x, bs = \"ps\")", {"newdata": "d3"}),
    ("y ~ s(x, bs = \"cp\")", {"newdata": "d3"}), ("y ~ s(x, bs = \"bs\")", {"newdata": "d3"}), ("y ~ s(x, bs = \"ps\", m = c(3, 1), k = 8)", {}),
    ("y ~ s(x, bs = \"bs\", m = 2, k = 7)", {}), ("y ~ s(x, bs = \"ts\")", {}), ("y ~ s(x, m = 3)", {}),
    ("y ~ f + s(x) + s(z, bs = \"cr\")", {"newdata": "d3"}), ("y ~ s(x, z)", {"newdata": "d3"}), ("y ~ s(x, z, k = 12)", {}),
    ("y ~ s(x, z, w, k = 20)", {}), ("y ~ te(x, z)", {"newdata": "d3"}), ("y ~ te(x, z, k = c(4, 5))", {}),
    ("y ~ te(x, z, bs = c(\"tp\", \"cr\"))", {"newdata": "d3"}), ("y ~ te(x, z, bs = \"ps\")", {}), ("y ~ ti(x, z)", {"newdata": "d3"}),
    ("y ~ s(x) + s(z) + ti(x, z)", {}), ("y ~ te(x, z, w, k = 3)", {}), ("y ~ te(x, z, d = c(2, 1)... ", {}),
    ("y ~ s(x, by = z)", {"newdata": "d3"}), ("y ~ f + s(x, by = f)", {"newdata": "d3"}), ("y ~ o + s(x, by = o)", {}),
    ("y ~ s(x, by = f, bs = \"cr\", k = 5) + f", {}), ("y ~ s(f, bs = \"re\")", {"newdata": "d3"}), ("y ~ s(f, g, bs = \"re\")", {}),
    ("y ~ x + s(z, k = 4, bs = \"cr\") - 1", {}), ("y ~ s(log(x)) + g", {"newdata": "d3"}), ("log(y) ~ s(x, bs = \"cr\") + offset(w)", {}),
    ("y ~ s(x, bs = \"cr\", k = 5)", {"knots": {"x": [0, 2, 4, 6, 10]}, "newdata": "d3"}),
    ("y ~ s(x, bs = \"ps\", k = 6)", {"knots": {"x": [0, 10]}, "newdata": "d3"}), ("y ~ s(x, bs = \"cc\", k = 6)", {"knots": {"x": [0, 10]}}),
    ("y ~ g*x + s(z, by = g)", {}), ("y ~ s(x, k = 40)", {}), ("y ~ s(z, x, bs = \"tp\", k = 15) + h", {}),
]:
    if "..." in f:
        f = "y ~ te(x, z, w, d = c(2, 1), k = c(10, 4))"
    c = {"kind": "gam", "formula": f, "data": "d1"}
    c.update(extra)
    GAM.append(c)
GAM.append({"kind": "gam", "formula": "y ~ s(x)", "data": "d5"})
GAM.append({"kind": "gam", "formula": "y ~ s(x, z, k = 12)", "data": "d5"})
SIDE = [
    ("y ~ s(x) + te(x, z)", {"newdata": "d3"}), ("y ~ s(x) + s(x, z)", {"newdata": "d3"}),
    ("y ~ s(x, bs = \"cr\") + s(z, bs = \"cr\") + te(x, z)", {"newdata": "d3"}),
    ("y ~ s(x) + s(z) + s(x, z, k = 20)", {}), ("y ~ te(x, z) + te(x, z, w, k = 3)", {}),
    ("y ~ s(x, bs = \"ps\") + te(x, z, bs = \"ps\")", {}), ("y ~ s(x, bs = \"cc\") + te(x, z)", {}),
    ("y ~ s(x, k = 5, fx = TRUE) + te(x, z)", {}), ("y ~ s(x, bs = \"cs\") + s(x, z, bs = \"ts\", k = 15)", {}),
    ("y ~ s(x, bs = \"cp\") + te(x, w, bs = c(\"cp\", \"cr\"))", {}), ("y ~ f + s(x) + s(x, by = f)", {}),
    ("y ~ s(x) + s(z) + te(x, z) + s(w)", {"newdata": "d3"}), ("y ~ s(x) + ti(x, z) + te(x, z)", {}),
    ("y ~ x + s(x) + te(x, z)", {}), ("y ~ s(x) + te(x, z) - 1", {}),
    ("y ~ s(x, bs = \"bs\") + te(x, z, bs = \"bs\")", {"newdata": "d3"}),
]
for f, extra in SIDE:
    c = {"kind": "gam", "formula": f, "data": "d1"}
    c.update(extra)
    GAM.append(c)
for sm in ["s(x)", "s(x, k = 6)", "s(x, bs = \"ts\")", "s(x, z)", "s(x, z, k = 12)", "s(x, bs = \"cr\")",
           "s(x, bs = \"cs\")", "s(x, bs = \"cc\")", "s(x, bs = \"ps\")", "s(x, bs = \"ps\", m = c(3, 1), k = 8)",
           "s(x, bs = \"cp\")", "s(f, bs = \"re\")", "te(x, z)", "te(x, z, bs = c(\"tp\", \"cr\"))", "ti(x, z)",
           "te(x, z, bs = \"ps\")", "s(x, by = f)", "s(x, by = z)", "te(x, z, w, k = 3)",
           "s(x, bs = \"bs\")", "s(x, bs = \"bs\", m = c(3, 1), k = 8)", "s(x, bs = \"bs\", m = c(2, 0), k = 7)",
           "s(x, bs = \"bs\", m = c(4, 2))", "te(x, z, bs = \"bs\")"]:
    GAM.append({"kind": "gampen", "smooth": sm, "data": "d1"})
ALL_CASES = ALL_CASES + GAM

LM = [{"kind": "lm", "formula": f, "data": "d2"} for f in [
    "y ~ log(x) + f", "cbind(y, y2) ~ x * g + poly(z, 2)", "cbind(y, y2, w) ~ C(f, sum) + ns(x, 3) + l",
    "log(w) ~ f + g:x + o", "cbind(y, y2) ~ h + bs(x, df = 4) - 1",
]]
ALL_CASES = ALL_CASES + LM

GAMFIT = []
for f, method, extra in [
    ("y ~ s(x)", "REML", {}), ("y ~ s(x)", "GCV.Cp", {}), ("y ~ f + s(x) + s(z)", "REML", {}),
    ("y ~ f + s(x) + s(z)", "GCV.Cp", {}), ("y ~ s(x, bs = \"cr\", k = 12) + s(z, bs = \"ps\")", "REML", {}),
    ("y ~ te(x, z)", "REML", {}), ("y ~ te(x, z, bs = \"cr\", k = c(5, 4))", "GCV.Cp", {}),
    ("y ~ s(x, z, k = 20)", "REML", {}), ("y ~ f + s(x, by = f)", "REML", {}),
    ("y ~ s(x) + s(z) + ti(x, z, k = 4)", "REML", {}), ("y ~ s(x, bs = \"cs\") + s(z, bs = \"ts\")", "REML", {}),
    ("y ~ s(x, bs = \"cc\") + s(f, bs = \"re\")", "REML", {}), ("y ~ s(x) + te(x, z, k = 4)", "REML", {}),
    ("y ~ s(x) + s(z)", "REML", {"weights": "w"}), ("y ~ s(x) + s(z)", "GCV.Cp", {"gamma": 1.4}),
    ("y ~ f + s(x) + s(z)", "REML", {"sp": [0.5, 3.0]}), ("y ~ te(x, z, k = 4)", "REML", {"sp": [0.1, 2.0]}),
    ("y ~ s(x, bs = \"bs\") + z", "REML", {}),
]:
    c = {"kind": "gamfit", "formula": f, "data": "d6", "newdata": "d6new", "method": method}
    c.update(extra)
    GAMFIT.append(c)
ALL_CASES = ALL_CASES + GAMFIT
