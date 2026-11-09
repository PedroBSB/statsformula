"""SEM bridges (statsformula.sem) against lavaan 0.7.3 (tests/fixtures/r_sem.json).

lavaan fits the model after statsformula rewrites R expressions into columns; semopy and pySEM
fit the same rewritten model through the bridges.  Backends are optional (tests skip)."""

import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

import statsformula as sf
from statsformula.sem import sem_data, to_pysem, to_pysemforge, to_semopy

import sem_cases

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "fixtures", "r_sem.json")) as fh:
    FIX = json.load(fh)
D = sem_cases.datasets()
CASES = list(zip(FIX["cases"], FIX["results"]))
IDS = [c[0]["name"] for c in CASES]


def compare(fit, r, rtol):
    R = pd.DataFrame({k: r[k] for k in ("lhs", "op", "rhs", "est")})
    m = fit.estimates.merge(R[R.op != ":="], on=["lhs", "op", "rhs"], how="outer", indicator=True,
                            suffixes=("", "_R"))
    assert (m._merge == "both").all(), m[m._merge != "both"]
    np.testing.assert_allclose(m.est, m.est_R, rtol=rtol, atol=rtol)
    for name, est in zip(R.lhs[R.op == ":="], R.est[R.op == ":="]):
        np.testing.assert_allclose(fit.defined[name], est, rtol=rtol, atol=rtol)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_semopy_matches_lavaan(case):
    pytest.importorskip("semopy")
    from statsformula.sem import fit_semopy
    cs, r = case
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_semopy(cs["model"], D[cs["data"]], preset=cs["preset"], **(cs.get("options") or {}))
    compare(fit, r, 2e-5)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_pysem_matches_lavaan(case):
    pytest.importorskip("torch")
    pytest.importorskip("pysem.sem")
    from statsformula.sem import fit_pysem
    cs, r = case
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_pysem(cs["model"], D[cs["data"]], preset=cs["preset"], **(cs.get("options") or {}))
    # hs_mimic_exprs: ageyr and log(ageyr) correlate 0.999; pySEM's objective is flat there
    compare(fit, r, 2e-2 if cs["name"] == "hs_mimic_exprs" else 5e-5)


def test_sem_data_rewrites_r_expressions():
    model = """
      # comment
      ind =~ x1 + x2 +
             x3
      m ~ a*ind + log(w) + f + z ; y ~ b*m + I(z^2)
      ab := a*b
    """
    sd = sem_data(model, D["syn"])
    assert sd.model.splitlines() == ["ind =~ x1 + x2 + x3", "m ~ a*ind + log_w + fb + fc + z",
                                     "y ~ b*m + I_z_2", "ab := a*b"]
    assert sd.defined == [("ab", "a*b")]
    np.testing.assert_allclose(sd.data["log_w"], np.log(D["syn"].w))
    assert set(sd.data.columns) == {"x1", "x2", "x3", "m", "z", "y", "log_w", "fb", "fc", "I_z_2"}
    new = sd.transform(D["syn"].head(5))
    np.testing.assert_allclose(new.to_numpy(), sd.data.head(5)[new.columns].to_numpy())
    # R crossing x*z is not a label (x is a column); poly() is stateful
    sd2 = sem_data("y ~ x1*z + poly(w, 2)", D["syn"])
    assert sd2.model == "y ~ x1 + z + x1_z + poly_w_2_1 + poly_w_2_2"
    np.testing.assert_allclose(sd2.transform(D["syn"].iloc[:3])["poly_w_2_2"], sd2.data["poly_w_2_2"].iloc[:3])
    with pytest.raises(sf.RError, match="expands to 2 columns"):
        sem_data("y ~ a*f", D["syn"])


def test_semopy_description_is_explicit():
    desc, df, sd, pt = to_semopy("ind =~ x1 + x2 + x3\nm ~ ind + log(w)", D["syn"])
    lines = desc.splitlines()
    assert "ind =~ 1.0*x1" in lines and "m ~ log_w" in lines and "ind ~~ ind" in lines
    assert any(ln.startswith("ind =~ sf_p") and ln.endswith("*x2") for ln in lines)


def test_pysem_translation():
    kw, sd, pt, names = to_pysem("ind =~ x1 + x2 + x3\nm ~ a*ind + z\ny ~ m", D["syn"])
    assert names == {"m": "eta:m", "z": "eta:z", "y": "eta:y"}
    assert kw["latent"] == ["ind", "eta:m", "eta:z", "eta:y"]
    assert kw["lambda_y"][("x1", "ind")] == 1.0 and np.isnan(kw["lambda_y"][("x2", "ind")])
    assert kw["lambda_y"][("m", "eta:m")] == 1.0 and kw["theta"][("m", "m")] == 0.0
    assert np.isnan(kw["beta"][("eta:m", "ind")]) and np.isnan(kw["beta"][("eta:y", "eta:m")])
    with pytest.raises(sf.RError, match="equality"):
        to_pysem("y ~ a*x1 + a*x2", D["syn"])
    with pytest.raises(sf.RError, match="constraints"):
        to_pysem("y ~ a*x1 + b*x2\na == b", D["syn"])


def test_pysemforge_translation():
    syntax, df, sd = to_pysemforge("ind =~ x1 + x2 + x3\nm ~ a*ind + log(w)\nab := a*b", D["syn"])
    assert syntax == "ind =~ x1 + x2 + x3\nm ~ ind + log_w"
    with pytest.raises(sf.RError, match="cannot fix"):
        to_pysemforge("ind =~ x1 + 0.5*x2", D["syn"])


def test_pysemforge_runs_and_warns():
    pytest.importorskip("pysemforge")
    from statsformula.sem import fit_pysemforge
    with pytest.warns(RuntimeWarning, match="random sample"):
        res = fit_pysemforge(sem_cases.HS3, D["hs"])
    assert res.sem_data.model.splitlines()[0] == "visual =~ x1 + x2 + x3"
