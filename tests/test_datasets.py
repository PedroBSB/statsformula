"""Bundled datasets (sf.load_data) and the published results of the econometrics examples."""

import numpy as np
import pandas as pd
import pytest

import statsformula as sf


def test_catalogue_and_loading():
    cat = sf.load_data()
    assert list(cat.name) == ["kmenta", "klein", "grunfeld_greene"]
    assert list(cat.rows) == [20, 22, 100]
    k = sf.load_data("kmenta")
    assert list(k.columns) == ["consump", "price", "income", "farmPrice", "trend"]
    assert "Kmenta" in k.attrs["source"] and k.attrs["title"]
    kl = sf.load_data("klein")
    assert kl.year.tolist() == list(range(1920, 1942)) and kl.loc[0, ["corpProfLag", "gnpLag"]].isna().all()
    np.testing.assert_allclose(kl.wages, kl.privWage + kl.govWage, atol=0.05)
    g = sf.load_data("Grunfeld-Greene")
    assert isinstance(g.firm.dtype, pd.CategoricalDtype)
    assert list(g.firm.cat.categories) == ["General Motors", "Chrysler", "General Electric", "Westinghouse",
                                           "US Steel"]
    assert g.groupby("firm", observed=True).size().tolist() == [20] * 5
    with pytest.raises(KeyError, match="available"):
        sf.load_data("mroz")


def _tsls(formula, data):
    F = sf.Formula(formula)
    y = np.asarray(sf.model_part(F, sf.model_frame(F, data), lhs=1), dtype=float).ravel()
    X = sf.model_matrix(F, data, rhs=1).to_numpy()
    Z = sf.model_matrix(F, data, rhs=2).to_numpy()
    Xh = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    return np.linalg.lstsq(Xh, y, rcond=None)[0]


def test_published_2sls():
    # Kmenta (1986) / systemfit: demand 94.6333, -0.2436, 0.3140; supply 49.5324, 0.2401, 0.2556, 0.2529
    k = sf.load_data("kmenta")
    inst = " | income + farmPrice + trend"
    np.testing.assert_allclose(_tsls("consump ~ price + income" + inst, k), [94.6333, -0.2436, 0.3140], atol=5e-5)
    np.testing.assert_allclose(_tsls("consump ~ price + farmPrice + trend" + inst, k),
                               [49.5324, 0.2401, 0.2556, 0.2529], atol=5e-5)
    # Greene, Econometric Analysis: Klein Model I, 2SLS consumption 16.555, 0.017, 0.216, 0.810
    kl = sf.load_data("klein").dropna()
    pre = " | govExp + taxes + govWage + trend + capitalLag + corpProfLag + gnpLag"
    np.testing.assert_allclose(_tsls("consump ~ corpProf + corpProfLag + wages" + pre, kl),
                               [16.555, 0.0173, 0.2162, 0.8102], atol=5e-4)
    np.testing.assert_allclose(_tsls("invest ~ corpProf + corpProfLag + capitalLag" + pre, kl),
                               [20.278, 0.1502, 0.6159, -0.1578], atol=5e-4)


def test_sur_badly_scaled_variables():
    """SUR of Greene's Grunfeld firms by ML: lavaan::sem() gives logl -459.0922 and these
    standard errors (the information matrix spans ~13 orders of magnitude)."""
    pytest.importorskip("sklearn")
    from statsformula.sklearn import PathModel
    g = sf.load_data("grunfeld_greene")
    abbr = {"General Motors": "GM", "Chrysler": "CH", "General Electric": "GE", "Westinghouse": "WE",
            "US Steel": "US"}
    w = g.assign(firm=g.firm.map(abbr)).pivot(index="year", columns="firm", values=["invest", "value", "capital"])
    w.columns = ["%s_%s" % c for c in w.columns]
    f = list(abbr.values())
    model = "\n".join(["invest_{0} ~ value_{0} + capital_{0}".format(a) for a in f]
                      + ["invest_%s ~~ %s" % (a, " + ".join("invest_" + b for b in f[i + 1:]))
                         for i, a in enumerate(f[:-1])])
    pm = PathModel(model).fit(w)
    np.testing.assert_allclose(pm.fit_measures()["logl"], -459.0922, atol=1e-3)
    co = pm.coefficients().query("op == '~'")
    np.testing.assert_allclose(co.se, [0.020, 0.032, 0.017, 0.026, 0.012, 0.022, 0.010, 0.037, 0.045, 0.118],
                               atol=6e-4)
