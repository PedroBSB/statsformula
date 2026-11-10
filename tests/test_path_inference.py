"""Path-model inference against R (tests/fixtures/r_pathse.json): lavaan::sem() standard
errors with expected and observed information, delta-method effects (:=), full-information
ML for non-recursive models and residual covariances, and summary(lm())."""

import json
import os
import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")

import statsformula as sf  # noqa: E402
from statsformula.sklearn import PathModel  # noqa: E402

import path_cases  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "fixtures", "r_pathse.json")) as fh:
    FIX = json.load(fh)
D, _ = path_cases.make_data()
FB = path_cases.make_feedback_data()
DATA = {"d": D, "fb": FB}
CASES = list(zip(FIX["cases"], FIX["results"]))
IDS = [c["model"].replace("\n", " | ") for c, _ in CASES]


def num(v):
    return np.array([np.nan if x is None else x for x in v], dtype=float)


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.parametrize("info", ["expected", "observed"])
def test_lavaan_standard_errors(case, info):
    cs, r = case
    rr = r[info]
    assert rr["converged"]
    pm = PathModel(cs["model"], se=info).fit(DATA[cs["data"]])
    assert pm.converged_
    co = pm.coefficients()
    ren = cs.get("rename", {})
    key = {(a, b, c): i for i, (a, b, c) in enumerate(zip(co.lhs, co.op, co.rhs))}
    n_par = 0
    for j, (lhs, op, rhs) in enumerate(zip(rr["lhs"], rr["op"], rr["rhs"])):
        if op == ":=":
            continue
        n_par += 1
        i = key[(lhs, op, ren.get(rhs, rhs))]
        for col, rtol in [("est", 1e-6), ("se", 1e-5), ("z", 1e-5), ("pvalue", 1e-4), ("ci.lower", 1e-5),
                          ("ci.upper", 1e-5)]:
            np.testing.assert_allclose(co[col].iloc[i], rr[col][j], rtol=rtol, atol=1e-7,
                                       err_msg="%s %s %s %s" % (lhs, op, rhs, col))
    assert n_par == len(co)  # same free parameters as lavaan
    if pm.loglik_ is not None and cs["data"] == "fb":
        assert np.isfinite(pm.loglik_)
    # defined parameters := are effects
    for label, (kind, lhs, rhs) in cs.get("defined", {}).items():
        j = [k for k, (lab, op) in enumerate(zip(rr["lhs"], rr["op"])) if op == ":=" and lab == label][0]
        ef = pm.effects(kind, se=True)
        row = ef[(ef.lhs == lhs) & (ef.rhs == rhs)]
        assert len(row) == 1, (label, kind, lhs, rhs)
        for col in ["est", "se", "z", "pvalue", "ci.lower", "ci.upper"]:
            np.testing.assert_allclose(row[col].iloc[0], rr[col][j], rtol=1e-5, atol=1e-7,
                                       err_msg="%s (%s) %s" % (label, kind, col))


@pytest.mark.parametrize("case", [c for c in CASES if c[0].get("lm")], ids=[i for i, c in zip(IDS, CASES) if c[0].get("lm")])
def test_lm_standard_errors(case):
    cs, r = case
    pm = PathModel(cs["model"], se="lm").fit(DATA[cs["data"]])
    co = pm.coefficients()
    assert "t" in co.columns
    for eq, rl in zip(pm.equations_, r["lm"]):
        sub = co[(co.lhs == eq.response) & (co.op != "~~")]
        names = ["(Intercept)" if op == "~1" else rhs for op, rhs in zip(sub.op, sub.rhs)]
        order = [names.index(n) for n in rl["names"]]
        sub = sub.iloc[order]
        for col, rcol in [("est", "est"), ("se", "se"), ("t", "t"), ("pvalue", "p"), ("ci.lower", "ci.lower"),
                          ("ci.upper", "ci.upper")]:
            np.testing.assert_allclose(sub[col].to_numpy(), num(rl[rcol]), rtol=1e-8, atol=1e-12,
                                       err_msg="%s %s" % (eq.formula, col))


def test_ml_equals_ols_for_recursive_models():
    model = "m1 ~ x\nm2 ~ x + m1\ny ~ m1 + m2 + z"
    a = PathModel(model).fit(D)
    b = PathModel(model, estimation="ML").fit(D)
    assert a.estimation_ == "OLS" and b.estimation_ == "ML"
    np.testing.assert_allclose(b.coefficients().est, a.coefficients().est, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(b.coefficients().se, a.coefficients().se, rtol=1e-8)
    np.testing.assert_allclose(b.loglik_, a.loglik_, rtol=1e-12)


def test_nonrecursive_prediction_is_the_reduced_form():
    pm = PathModel("y1 ~ y2 + x1\ny2 ~ y1 + x2 + x3\ny1 ~~ y2").fit(FB)
    assert pm.order_ is None and pm.estimation_ == "ML"
    co = pm.coefficients().set_index(["lhs", "op", "rhs"]).est
    new = FB[["x1", "x2", "x3"]].head(7)
    A = np.array([[1, -co["y1", "~", "y2"]], [-co["y2", "~", "y1"], 1]])
    c = np.column_stack([co["y1", "~1", ""] + co["y1", "~", "x1"] * new.x1,
                         co["y2", "~1", ""] + co["y2", "~", "x2"] * new.x2 + co["y2", "~", "x3"] * new.x3])
    np.testing.assert_allclose(pm.predict(new).to_numpy(), np.linalg.solve(A, c.T).T, rtol=1e-10)
    # in equilibrium the direct predictions at the reduced form reproduce it
    eq = new.assign(**pm.predict(new))
    np.testing.assert_allclose(PathModel(pm.model, predict_mode="direct").fit(FB).predict(eq).to_numpy(),
                               pm.predict(new).to_numpy(), rtol=1e-8)
    # effects of the loop: total = (I - B)^-1 B
    tot = pm.effects("total")
    np.testing.assert_allclose(tot.loc["y2", "x1"], co["y2", "~", "y1"] * co["y1", "~", "x1"] /
                               (1 - co["y1", "~", "y2"] * co["y2", "~", "y1"]), rtol=1e-10)


def test_vcov_and_options():
    pm = PathModel("m ~ x + z\ny ~ m + x").fit(D)
    V = pm.vcov()
    co = pm.coefficients()
    np.testing.assert_allclose(np.sqrt(np.diag(V)), co.se, rtol=1e-12)
    assert list(V.index) == ["%s%s%s" % t for t in zip(co.lhs, co.op, co.rhs)]
    assert list(co.op) == ["~"] * 4 + ["~~"] * 2 + ["~1"] * 2   # lavaan's order
    wide = PathModel("m ~ x + z\ny ~ m + x", level=0.9).fit(D).coefficients()
    assert (wide["ci.upper"] - wide["ci.lower"] < co["ci.upper"] - co["ci.lower"]).all()
    assert np.isnan(PathModel("m ~ x + z\ny ~ m + x", se="none").fit(D).coefficients().se).all()
    with pytest.raises(sf.RError, match="se='lm'"):
        PathModel("y1 ~ y2 + x1\ny2 ~ y1 + x2", se="lm").fit(FB)
    with pytest.raises(sf.RError, match="estimation='ML' or 'auto'"):
        PathModel("y1 ~ x1\ny2 ~ x2\ny1 ~~ y2", estimation="OLS").fit(FB)
    with pytest.raises(sf.RError, match="plain column"):
        PathModel("y1 ~ log(y2 + 10) + x1\ny2 ~ y1 + x2").fit(FB)
    with pytest.raises(sf.RError, match="between endogenous"):
        PathModel("y1 ~ x1\ny2 ~ x2\nx1 ~~ y2").fit(FB)
    # effect structure: only the effects the model allows
    ind = pm.effects("indirect", se=True)
    assert list(zip(ind.lhs, ind.rhs)) == [("y", "x"), ("y", "z")]


def test_blockwise_inference():
    # different rows per equation and a transformed endogenous predictor: per-equation
    # observed information sigma2_ML (X'X)^-1, GAM equations use Vp
    d = D.copy()
    d.loc[[0, 1, 2], "y"] = np.nan
    pm = PathModel("m ~ x + z\ny ~ m + x").fit(d)
    assert pm.theta_ is None
    co = pm.coefficients()
    fit = sf.model_matrix("y ~ m + x", d.dropna())
    X = fit.to_numpy()
    yv = d.dropna().y.to_numpy()
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    s2 = np.sum((yv - X @ b) ** 2) / len(yv)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    got = co[co.lhs == "y"].set_index(["op", "rhs"]).se
    np.testing.assert_allclose([got["~1", ""], got["~", "m"], got["~", "x"]], se, rtol=1e-10)
    pm2 = PathModel("m ~ x + z\ny ~ s(m) + x").fit(D)
    co2 = pm2.coefficients()
    g = pm2.estimators_["y"].fits_[0]
    np.testing.assert_allclose(co2[(co2.lhs == "y") & (co2.op != "~~")].se.to_numpy(),
                               np.sqrt(np.diag(g.Vp))[np.argsort([c == "(Intercept)" for c in
                                                                  pm2.estimators_["y"].design_columns_],
                                                                 kind="stable")], rtol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.linear_model import Ridge
        pr = PathModel("m ~ x + z\ny ~ m + x", estimator={"y": Ridge()}).fit(D).coefficients()
    assert np.isnan(pr[pr.lhs == "y"].se).all() and np.isfinite(pr[(pr.lhs == "m") & (pr.op != "~~")].se).all()
