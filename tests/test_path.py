"""Path models (statsformula.sklearn.PathModel) against R: lm() per equation, propagated
predictions, and lavaan::sem() estimates and total effects (tests/fixtures/r_path.json)."""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402

import statsformula as sf  # noqa: E402
from statsformula._path import parse_path_model, parse_path_system, path_order, split_statements  # noqa: E402
from statsformula.sklearn import GAMRegressor, PathModel  # noqa: E402

import path_cases  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "fixtures", "r_path.json")) as fh:
    FIX = json.load(fh)
D, ND = path_cases.make_data()
CASES = list(zip(FIX["cases"], FIX["results"]))


def num(v):
    return np.array([np.nan if x is None else x for x in v], dtype=float)


@pytest.mark.parametrize("case", CASES, ids=[c[0]["model"].strip().replace("\n", " | ") for c in CASES])
def test_against_r(case):
    cs, r = case
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = PathModel(cs["model"]).fit(D)
        pred = pm.predict(ND)
    # lm() coefficients per equation
    for resp, rl in r["lm"].items():
        est = pm.estimators_[resp]
        assert est.design_columns_ == rl["names"]
        np.testing.assert_allclose(est.estimator_.coef_, num(rl["coef"]), rtol=1e-8, atol=1e-10)
    # propagated predictions (endogenous variables in the new data are ignored)
    for resp, rp in r["pred"].items():
        np.testing.assert_allclose(pred[resp].to_numpy(), num(rp), rtol=1e-8, atol=1e-10, err_msg=resp)
    if not cs.get("lavaan"):
        return
    # lavaan::sem() regression and intercept estimates
    co = pm.coefficients()
    for lhs, op, rhs, est in zip(r["pe"]["lhs"], r["pe"]["op"], r["pe"]["rhs"], r["pe"]["est"]):
        row = co[(co.lhs == lhs) & (co.op == op) & (co.rhs == rhs)]
        assert len(row) == 1, (lhs, op, rhs)
        np.testing.assert_allclose(row.est.iloc[0], est, rtol=1e-6, atol=1e-8, err_msg="%s %s %s" % (lhs, op, rhs))
    # total effects
    names = r["total"]["names"]
    T = np.column_stack([num(c) for c in r["total"]["m"]])
    tot = pm.effects("total")
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a in tot.index:
                got = tot.loc[a, b] if b in tot.columns else 0.0
                np.testing.assert_allclose(got, T[i, j], rtol=1e-6, atol=1e-8, err_msg="%s <- %s" % (a, b))


def test_split_statements():
    model = """
      # comment line
      m ~ x + z      # trailing comment
      y ~ m +
          x          ! lavaan-style comment
      y2 ~ (m +
            z); q ~ x
      w ~ x
        + z
    """
    assert split_statements(model) == ["m ~ x + z", "y ~ m + x", "y2 ~ (m + z)", "q ~ x", "w ~ x + z"]


def test_parse_path_model():
    eqs = parse_path_model("y1 + y2 ~ x\ncbind(a, b) ~ z\ny1 ~ w\nlog(q) ~ y1")
    assert [e.formula for e in eqs] == ["y1 ~ x + w", "y2 ~ x", "a ~ z", "b ~ z", "log(q) ~ y1"]
    assert [eqs[i].response for i in path_order(eqs)] == ["y1", "y2", "a", "b", "log(q)"]
    assert path_order(parse_path_model("a ~ b\nb ~ a")) is None
    for bad in ["f =~ x1 + x2", "y ~~ x", "ab := a*b", "a > 0", "~ x", ""]:
        with pytest.raises(sf.RError):
            parse_path_model(bad)
    eqs, covs = parse_path_system("y1 ~ x\ny2 ~ x\ny3 ~ y1\ny1 ~~ y2 + y3\ny2 ~~ y2")
    assert [e.response for e in eqs] == ["y1", "y2", "y3"] and covs == [("y1", "y2"), ("y1", "y3")]
    for bad in ["y1 ~ x\ny1 ~~ x", "y1 ~ x\ny1 ~~ log(y2)", "y1 ~ x\ny1 + y2 ~~ y3"]:
        with pytest.raises(sf.RError):
            parse_path_system(bad)


def test_effects_and_modes():
    pm = PathModel("m ~ x + z\ny ~ m + x").fit(D)
    direct, ind, tot = pm.effects("direct"), pm.effects("indirect"), pm.effects("total")
    a, b = direct.loc["m", "x"], direct.loc["y", "m"]
    np.testing.assert_allclose(ind.loc["y", "x"], a * b)
    np.testing.assert_allclose(tot.loc["y", "x"], a * b + direct.loc["y", "x"])
    assert pm.order_ == ["m", "y"] and pm.exogenous_ == ["x", "z"]
    # direct mode uses observed m
    pd_ = PathModel("m ~ x + z\ny ~ m + x", predict_mode="direct").fit(D)
    np.testing.assert_allclose(pd_.predict(D)["y"], pm.estimators_["y"].predict(D))
    assert pd_.score(D) > pm.score(D)
    # non-recursive: full-information ML, propagation through the reduced form
    nr = PathModel("m ~ y + x\ny ~ m + z").fit(D)
    assert nr.order_ is None and nr.estimation_ == "ML" and nr.converged_
    assert np.isfinite(nr.predict(D).to_numpy()).all()
    assert PathModel("m ~ y + x\ny ~ m + z", predict_mode="direct").fit(D).predict(D).shape == (len(D), 2)
    # ... unless an equation is not linear in the endogenous variables
    with pytest.raises(sf.RError, match="feedback loops"):
        PathModel("m ~ exp(y / 5) + x\ny ~ m + z").fit(D)
    nl = PathModel("m ~ exp(y / 5) + x\ny ~ m + z", estimation="OLS").fit(D)
    with pytest.raises(sf.RError, match="non-recursive"):
        nl.predict(D)
    # propagation through a transformed response is refused
    with pytest.raises(sf.RError, match="transformed response"):
        PathModel("log(w) ~ x\ny ~ w + x").fit(D).predict(D)
    with pytest.raises(sf.RError, match="needs every design column"):
        PathModel("m ~ x + f\ny ~ m").fit(D).effects()


def test_estimators_gam_and_sklearn_api():
    model = "m ~ x + f\ny ~ s(m) + s(z) + x"
    pm = PathModel(model).fit(D)
    assert isinstance(pm.estimators_["y"], GAMRegressor)
    b = sf.gam("y ~ s(m) + s(z) + x", D)
    np.testing.assert_allclose(pm.estimators_["y"].coef_, b.coefficients.to_numpy())
    assert pm.predict(ND).shape == (len(ND), 2)
    # per-equation estimators and grid search over shared hyper-parameters
    pr = PathModel("m ~ x + z\ny ~ m + x", estimator={"y": Ridge(alpha=5.0)}).fit(D)
    assert pr.estimators_["y"].estimator_.alpha == 5.0
    grid = GridSearchCV(PathModel("m ~ x + z\ny ~ m + x + z", estimator=Ridge()),
                        {"estimator__alpha": [0.01, 100.0], "predict_mode": ["direct"]}, cv=3).fit(D)
    assert grid.best_params_["estimator__alpha"] == 0.01
    pm2 = pickle.loads(pickle.dumps(clone(pm).fit(D)))
    np.testing.assert_allclose(pm2.predict(ND).to_numpy(), pm.predict(ND).to_numpy())
    # missing values: each equation uses its own complete cases
    d = D.copy()
    d.loc[[0, 1], "y"] = np.nan
    pmn = PathModel("m ~ x + z\ny ~ m + x").fit(d)
    assert np.isfinite(pmn.predict(d).to_numpy()).all()
