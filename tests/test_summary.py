"""summary() methods against R: summary.lm (FormulaEstimator), summary.gam (gam, GAMRegressor),
lavaan's summary/fitMeasures/r2 (PathModel), plus the R formatting and Davies ports they use.

Fixtures: tests/fixtures/r_summary.json (tools/make_summary_fixtures.py), the "lm" and "gamfit"
cases of r_reference.json (tools/make_fixtures.py) and r_pathse.json
(tools/make_pathse_fixtures.py)."""

import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import statsformula as sf
from statsformula._davies import psum_chisq, smooth_test_stat
from statsformula._rprint import format_num, format_pval, print_coefmat

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, "fixtures", name)) as fh:
        return json.load(fh)


RS = load("r_summary.json")
REF = load("r_reference.json")

_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?%?,?$")


def assert_printed_close(got, want, ulp=1.0, rtol=0.0, context=""):
    """Printed output equal line by line; numbers may differ by ``ulp`` units of their last
    printed digit (or by ``rtol``), everything else must be identical."""
    g, w = got.split("\n"), list(want)
    assert len(g) == len(w), "%s\n--- got\n%s\n--- want\n%s" % (context, got, "\n".join(w))
    for lg, lw in zip(g, w):
        if lg == lw:
            continue
        tg, tw = lg.split(), lw.split()
        assert len(tg) == len(tw), "%s\n%r\n%r" % (context, lg, lw)
        for a, b in zip(tg, tw):
            if a == b:
                continue
            assert _NUM.match(a) and _NUM.match(b), "%s\n%r\n%r" % (context, lg, lw)
            x, y = float(a.rstrip("%,")), float(b.rstrip("%,"))
            m = re.match(r"^[-+]?\d*\.?(\d*)", b.rstrip("%,").split("e")[0].split("E")[0])
            dec = len(m.group(1)) if m else 0
            exp = int(b.lower().split("e")[1]) if "e" in b.lower() else 0
            unit = 10.0 ** (exp - dec)
            assert abs(x - y) <= ulp * unit * (1 + 1e-9) or abs(x - y) <= rtol * abs(y), \
                "%s\n%r\n%r" % (context, lg, lw)


# ---------------------------------------------------------------------------
# R formatting and Davies' algorithm
# ---------------------------------------------------------------------------

def test_davies_against_r():
    for c in RS["davies"]:
        p = psum_chisq(c["q"][0], c["lb"], c["df"])
        np.testing.assert_allclose(p, c["p"][0], rtol=0, atol=1e-12, err_msg=str(c))


def test_format_against_r():
    for c in RS["format"]:
        assert format_num(c["x"], 4) == c["format"], c["x"]
        assert format_pval(np.abs(c["x"]), 3) == c["pval"], c["x"]


def test_print_coefmat_against_r():
    for c in RS["coefmat"]:
        m = np.array(c["x"]).reshape(len(c["rn"]), -1)
        kw = {} if c["cs"] is None else {"cs_ind": c["cs"], "has_pvalue": True}
        assert print_coefmat(m, c["rn"], c["cn"], **kw).split("\n") == c["printed"]


def test_test_stat_degenerate():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    V = np.linalg.inv(X.T @ X)
    stat, p, rank = smooth_test_stat(np.zeros(4), X, V, rank=4.0, res_df=40)
    assert stat == 0 and p == 1.0 and rank == 4
    stat, p, rank = smooth_test_stat(np.ones(4), X, V, rank=2.5, res_df=-1)
    assert stat > 0 and 0 <= p <= 1


# ---------------------------------------------------------------------------
# summary.lm
# ---------------------------------------------------------------------------

def sc(v):
    return v[0] if isinstance(v, list) else v


def _cases(kind):
    return [(c, r) for c, r in zip(REF["cases"], REF["results"]) if c["kind"] == kind and "error" not in r]


def _strip_call(lines):
    out, skip = [], False
    for ln in lines:
        if ln == "Call:":
            skip = True
            continue
        if skip:
            skip = ln.strip() != ""
            continue
        out.append(ln)
    return out


@pytest.mark.parametrize("case", _cases("lm"), ids=lambda c: "%s|%s|%s" % (c[0]["formula"], c[0]["data"],
                                                                           c[0].get("weights")))
def test_lm_summary_against_r(case):
    pytest.importorskip("sklearn")
    import rcases
    from sklearn.linear_model import LinearRegression
    from statsformula.sklearn import FormulaEstimator
    cs, r = case
    d = rcases.make_datasets()[cs["data"]]
    kw = {"sample_weight": d[cs["weights"]].to_numpy()} if cs.get("weights") else {}
    est = FormulaEstimator(cs["formula"], LinearRegression(fit_intercept=False)).fit(d, **kw)
    sm = est.summary()
    sms = list(sm) if isinstance(sm, list) else [sm]
    rsum = list(r["summary"].values()) if isinstance(r["summary"], dict) else r["summary"]
    assert len(sms) == len(rsum)
    for s, rs in zip(sms, rsum):
        ct = s.coefficients
        assert list(ct.index) == rs["names"]
        for col, key in (("Estimate", "est"), ("Std. Error", "se"), ("t value", "t"), ("Pr(>|t|)", "p")):
            want = np.array([np.nan if v is None else v for v in rs[key]], dtype=float)
            np.testing.assert_allclose(ct[col].to_numpy(dtype=float), want, rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(s.sigma, sc(rs["sigma"]), rtol=1e-10)
        np.testing.assert_allclose(s.r_squared, sc(rs["r_squared"]), rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(s.adj_r_squared, sc(rs["adj_r_squared"]), rtol=1e-10, atol=1e-14)
        assert list(s.df) == rs["df"]
        if rs["fstatistic"] is None:
            assert s.fstatistic is None
        else:
            np.testing.assert_allclose(s.fstatistic, rs["fstatistic"], rtol=1e-9)
    if r.get("loglik") is not None:
        np.testing.assert_allclose([sms[0].loglik, sms[0].aic, sms[0].bic],
                                   [sc(r["loglik"]), sc(r["aic"]), sc(r["bic"])], rtol=1e-10)
    # printed like print.summary.lm (the Call differs; residuals that are 0 up to
    # rounding print as noise in both, at a different 1e-16 level)
    got = "\n".join(_strip_call(str(sm).split("\n")))
    noise = 1e14 if cs["data"] == "d3" else 1.0  # d3: an exact fit for two rows (residuals ~1e-16)
    assert_printed_close(got, _strip_call(r["printed"]), ulp=noise, context=cs["formula"])


def test_estimator_summary_generic():
    pytest.importorskip("sklearn")
    import rcases
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.tree import DecisionTreeRegressor
    from statsformula.sklearn import FormulaEstimator
    d = rcases.make_datasets()["d2"]
    s = FormulaEstimator("y ~ x + f", Ridge(alpha=2.0)).fit(d).summary()
    assert list(s.coefficients.index) == ["(intercept_)", "(Intercept)", "x", "fb", "fc"]
    assert "Ridge(alpha=2.0)" in str(s) and "R-squared (training data)" in str(s)
    assert s.n_deleted == 4 and s.nobs == 36
    s = FormulaEstimator("g ~ x + z", LogisticRegression()).fit(d).summary()
    assert "Accuracy" in str(s) and s.coefficients.shape == (4, 1)
    s = FormulaEstimator("y ~ x", DecisionTreeRegressor(max_depth=2)).fit(d).summary()
    assert s.coefficients is None and "Coefficients" not in str(s)


# ---------------------------------------------------------------------------
# summary.gam
# ---------------------------------------------------------------------------

def _gam_tol(cs):
    if '"cs"' in cs["formula"] or "ti(" in cs["formula"]:
        return 5e-3     # smoothing parameters near their bounds, see check_gamfit_case
    return 1e-4 if cs["method"] == "GCV.Cp" else 2e-5


@pytest.mark.parametrize("case", _cases("gamfit"), ids=lambda c: "%s|%s" % (c[0]["formula"], c[0]["method"]))
def test_gam_summary_against_r(case):
    import rcases
    cs, r = case
    d = rcases.make_datasets()[cs["data"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = sf.gam(cs["formula"], d, method=cs["method"], sp=cs.get("sp"), weights=cs.get("weights"),
                   gamma=cs.get("gamma", 1.0))
        sm = b.summary()
    tol = _gam_tol(cs)
    st = r["stable"]
    assert list(sm.s_table.index) == st["names"]
    np.testing.assert_allclose(sm.s_table["edf"], st["edf"], rtol=10 * tol)
    np.testing.assert_allclose(sm.s_table["Ref.df"], st["refdf"], rtol=10 * tol)
    np.testing.assert_allclose(sm.s_table["F"], st["stat"], rtol=20 * tol)
    np.testing.assert_allclose(sm.s_table["p-value"], st["p"], rtol=100 * tol, atol=1e-12)
    np.testing.assert_allclose(b._fit.edf1, r["edf1"], rtol=5e-2 if tol > 1e-4 else 2e-3, atol=1e-6)
    np.testing.assert_allclose(sm.dev_expl, sc(r["dev_expl"]), rtol=tol)
    if r.get("pterms"):
        pt = r["pterms"]
        assert list(sm.pterms_table.index) == pt["names"]
        np.testing.assert_allclose(sm.pterms_table["df"], pt["df"])
        np.testing.assert_allclose(sm.pterms_table["F"], pt["stat"], rtol=10 * tol)
        np.testing.assert_allclose(sm.pterms_table["p-value"], pt["p"], rtol=100 * tol, atol=1e-12)
    if '"cs"' in cs["formula"]:
        return  # the REML score itself differs (arbitrary null-space basis, check_gamfit_case)
    assert_printed_close(str(sm), r["printed"], ulp=2.0 if tol <= 1e-4 else 1e3,
                         rtol=1e-4 if tol <= 1e-4 else 5e-3, context=cs["formula"])


def test_gam_summary_layout():
    import rcases
    d = rcases.make_datasets()["d6"]
    b = sf.gam("y ~ f + s(x) + s(z)", d)
    text = str(b.summary())
    for part in ["Family: gaussian", "Link function: identity", "Parametric coefficients:",
                 "Approximate significance of smooth terms:", "R-sq.(adj) =", "Deviance explained =",
                 "-REML =", "Scale est. =", "n = 150", "Signif. codes:"]:
        assert part in text
    assert repr(b.summary()) == text


def test_gamregressor_summary():
    pytest.importorskip("sklearn")
    import rcases
    from statsformula.sklearn import GAMRegressor
    d = rcases.make_datasets()["d6"]
    g = GAMRegressor("y ~ f + s(x) + s(z)").fit(d)
    assert str(g.summary()) == str(sf.gam("y ~ f + s(x) + s(z)", d).summary())
    two = GAMRegressor("cbind(y, x) ~ s(z)").fit(d).summary()
    assert len(two) == 2 and "x ~ s(z)" in str(two[1])
    plain = GAMRegressor("y ~ f + x").fit(d).summary()
    assert len(plain.s_table) == 0 and list(plain.p_table.index) == ["(Intercept)", "fb", "fc", "x"]


# ---------------------------------------------------------------------------
# lavaan summary of path models
# ---------------------------------------------------------------------------

PATHSE = load("r_pathse.json")


def _param_block(lines, rename=()):
    i = lines.index("Regressions:")
    out = []
    for ln in lines[i:]:
        if ln.startswith("Defined Parameters"):
            break
        for a, b in rename:
            ln = ln.replace(a, b)
        out.append(re.sub(r"\((\w+)\)", lambda m: " " * len(m.group(0)), ln))
    while out and out[-1] == "":
        out.pop()
    return out


@pytest.mark.parametrize("i", range(len(PATHSE["cases"])))
@pytest.mark.parametrize("information", ["expected", "observed"])
def test_path_summary_against_lavaan(i, information):
    pytest.importorskip("sklearn")
    import path_cases
    from statsformula.sklearn import PathModel
    cs, r = PATHSE["cases"][i], PATHSE["results"][i][information]
    D, _ = path_cases.make_data()
    data = D if cs["data"] == "d" else path_cases.make_feedback_data()
    pm = PathModel(cs["model"], se=information).fit(data)
    fm = pm.fit_measures()
    for k, v in r["fm"].items():
        if v is None:
            assert np.isnan(fm[k]), k
        else:
            np.testing.assert_allclose(fm[k], v, rtol=1e-6, atol=1e-7, err_msg=k)
    r2 = pm.rsquare()
    for k, v in r["r2"].items():
        np.testing.assert_allclose(r2[k], v, rtol=1e-6, err_msg=k)
    got = _param_block(str(pm.summary()).split("\n"), rename=[("log(w)", "logw  ")])
    assert got == _param_block(r["printed"])


def test_path_summary_other_modes():
    pytest.importorskip("sklearn")
    import path_cases
    from statsformula.sklearn import PathModel
    D, _ = path_cases.make_data()
    pm = PathModel("m ~ x + z\ny ~ m + x", se="lm").fit(D)
    text = str(pm.summary(effects=True))
    assert "t-value  P(>|t|)" in text and "Indirect Effects:" in text and "y <- x" in text
    # GAM equations: no linear system, per-equation summary without fit measures
    g = PathModel("m ~ x + f\ny ~ s(m) + x").fit(D).summary()
    assert g.fit_measures is None and "Model Test User Model" not in str(g)
    assert "Regressions:" in str(g) and "Smooth terms" in str(g) and "s(m).1" not in str(g)
    with pytest.raises(sf.RError, match="one linear system"):
        PathModel("m ~ x + f\ny ~ s(m) + x").fit(D).fit_measures()


def test_semfit_summary():
    pytest.importorskip("semopy")
    from statsformula.sem import fit_semopy
    d = pd.read_csv(os.path.join(HERE, "fixtures", "HolzingerSwineford1939.csv"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = fit_semopy("visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9", d)
    s = f.summary()
    text = str(s)
    assert "Latent Variables:" in text and "  visual =~" in text and "   .x1" in text
    # lavaan: chisq = 85.306 on 24 df, CFI 0.931, TLI 0.896, RMSEA 0.092
    np.testing.assert_allclose(s.fit_stats["chisq"], 85.306, atol=5e-3)
    assert s.fit_stats["df"] == 24
    np.testing.assert_allclose([s.fit_stats["cfi"], s.fit_stats["tli"], s.fit_stats["rmsea"]],
                               [0.931, 0.896, 0.092], atol=1e-3)
    se = f.estimates.set_index(["lhs", "op", "rhs"])["se"]
    np.testing.assert_allclose(se[("visual", "=~", "x2")], 0.100, atol=2e-3)   # lavaan: 0.100
