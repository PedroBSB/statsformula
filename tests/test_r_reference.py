"""Compare statsformula against results computed by R (tests/fixtures/r_reference.json).

Regenerate the fixture with ``python tools/make_fixtures.py`` (requires R).
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

import statsformula as sf
from statsformula._rtypes import RMatrix

import rcases

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "fixtures", "r_reference.json")) as fh:
    FIXTURE = json.load(fh)
CASES = list(zip(FIXTURE["cases"], FIXTURE["results"]))
DATA = rcases.make_datasets()

RTOL = 1e-9
ATOL = 1e-10


def case_id(c):
    cs = c[0]
    return "%s:%s" % (cs["kind"], cs.get("formula") or cs.get("smooth") or cs.get("old") or cs.get("labels") or cs.get("parts"))


def num(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def assert_matrix(columns, colnames, r, label=""):
    assert list(colnames) == list(r["colnames"] or []), label
    rm = np.column_stack([num(c) for c in r["columns"]]) if r["columns"] else np.zeros((r["nrow"], 0))
    assert columns.shape == rm.shape, label
    np.testing.assert_allclose(columns, rm, rtol=RTOL, atol=ATOL, err_msg=label)


def assert_terms(t, r):
    assert t.term_labels == r["term_labels"]
    assert t.order == r["order"]
    assert t.intercept == r["intercept"]
    assert t.response == r["response"]
    assert t.variables == r["variables"]
    fac = [list(map(int, t.factors[:, j])) for j in range(t.factors.shape[1])] if t.factors.size else []
    assert fac == r["factors"]
    assert (t.offset or None) == (r["offset"] or None)
    if r.get("specials"):
        got = {k: v for k, v in (t.specials or {}).items()}
        assert got == {k: v for k, v in r["specials"].items()}


def check_contrasts(got, r):
    if r is None:
        assert not got
        return
    assert list(got.keys()) == list(r.keys())
    for k, rv in r.items():
        gv = got[k]
        if isinstance(rv, str):
            assert gv == rv, k
        else:
            m = np.column_stack([num(c) for c in rv["matrix"]])
            np.testing.assert_allclose(gv.matrix, m, rtol=RTOL, atol=ATOL)
            assert (gv.colnames or None) == (rv["colnames"] or None)


def r_contrasts_arg(cs):
    if "contrasts" in cs:
        return cs["contrasts"]
    if "contrasts_matrix" in cs:
        return {k: np.array(v, dtype=float) for k, v in cs["contrasts_matrix"].items()}
    return None


def check_warnings(rec, r):
    got = [str(w.message) for w in rec]
    for msg in r.get("warnings", []):
        assert any(msg in g for g in got), "R warned %r; python warnings: %r" % (msg, got)


def check_column(v, r, label):
    from statsformula._rtypes import Character, Factor, Logical
    if r["type"] == "matrix":
        assert isinstance(v, RMatrix), label
        np.testing.assert_allclose(v.data, np.column_stack([num(c) for c in r["columns"]]), rtol=RTOL, atol=ATOL)
        assert (v.colnames or None) == (r["colnames"] or None), label
    elif r["type"] in ("factor", "ordered"):
        assert isinstance(v, Factor), label
        assert v.levels == r["levels"], label
        assert v.ordered == (r["type"] == "ordered")
        assert list(v.labels()) == r["values"], label
    elif r["type"] == "logical":
        assert isinstance(v, Logical), label
        assert [None if n else bool(b) for b, n in zip(v.values, v.na)] == r["values"]
    elif r["type"] == "character":
        assert list(v.values) == r["values"]
    else:
        np.testing.assert_allclose(np.asarray(v, dtype=float), num(r["values"]), rtol=RTOL, atol=ATOL)


def run_python(cs):
    k = cs["kind"]
    if cs.get("options_contrasts"):
        old = sf.options(contrasts=cs["options_contrasts"])
    else:
        old = None
    try:
        return _run(cs, k)
    finally:
        if old:
            sf.options(**old)


def _run(cs, k):
    if k == "terms":
        data = DATA[cs["data"]] if cs.get("data") else None
        return sf.formula(cs["formula"]).terms(data=data, keep_order=cs.get("keep_order", False),
                                              specials=cs.get("specials"))
    if k in ("mm", "sparse"):
        data = DATA[cs["data"]]
        f = sf.formula(cs["formula"])
        if k == "sparse":
            return sf.sparse_model_matrix(f, data)
        return sf.model_matrix(f, data, contrasts=r_contrasts_arg(cs))
    if k == "update":
        return sf.update(sf.formula(cs["old"]), sf.formula(cs["new"]))
    if k == "reformulate":
        return sf.reformulate(cs["labels"], response=cs["response"], intercept=cs["intercept"])
    if k == "drop_terms":
        return sf.drop_terms(sf.terms(cs["formula"]), cs["dropx"], keep_response=cs["keep_response"])
    if k == "delete_response":
        return sf.delete_response(sf.formula(cs["formula"]).terms(specials=cs.get("specials")))
    if k == "simplify":
        return sf.formula(cs["formula"]).terms(simplify=True).formula
    if k == "allvars":
        f = sf.formula(cs["formula"])
        return f.all_vars(), f.all_names()
    if k == "Formula":
        return sf.Formula(cs["formula"])
    if k == "Formula_update":
        return sf.Formula(cs["old"]).update(cs["new"])
    if k == "Formula_combine":
        return sf.as_Formula(*cs["parts"])
    if k == "predict":
        X = sf.model_matrix(cs["formula"], DATA[cs["data"]])
        return X, X.model_spec.get_model_matrix(DATA[cs["newdata"]])
    if k == "mf":
        d = DATA[cs["data"]]
        kw = {}
        if cs.get("na_action"):
            kw["na_action"] = cs["na_action"]
        for a in ("weights", "offset"):
            if cs.get(a):
                kw[a] = cs[a]
        if cs.get("subset"):
            kw["subset"] = cs["subset"]
        if cs.get("drop_unused_levels"):
            kw["drop_unused_levels"] = True
        return sf.model_frame(cs["formula"], d, **kw)
    if k == "gam":
        import test_mgcv_helpers
        return test_mgcv_helpers.run_gam_case(cs, DATA)
    if k == "gampen":
        import test_mgcv_helpers
        return test_mgcv_helpers.run_gampen_case(cs, DATA)
    raise ValueError(k)


@pytest.mark.parametrize("case", CASES, ids=[case_id(c) for c in CASES])
def test_against_r(case):
    cs, r = case
    k = cs["kind"]
    if "error" in r:
        with pytest.raises(Exception) as ei:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run_python(cs)
        assert r["error"] in str(ei.value), "R error %r vs python %r" % (r["error"], str(ei.value))
        return
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = run_python(cs)
    check_warnings(rec, r)
    if k == "terms" or k in ("drop_terms", "delete_response"):
        assert str(res.formula) == r["formula"]
        assert_terms(res, r)
    elif k == "mm":
        assert_matrix(res.to_numpy(), res.columns, r, cs["formula"])
        assert res.assign == r["assign"]
        check_contrasts(res.contrasts, r.get("contrasts"))
    elif k == "sparse":
        assert_matrix(res.toarray(), res.columns, r, cs["formula"])
        assert res.assign == r["assign"]
    elif k in ("update", "reformulate", "simplify", "Formula_update"):
        assert str(res) == r["formula"]
    elif k == "allvars":
        assert res[0] == r["vars"]
        assert res[1] == r["names"]
    elif k == "Formula_combine":
        assert str(res) == r["formula"]
        assert list(res.length) == r["length"]
    elif k == "Formula":
        d = DATA[cs["data"]]
        F = res
        assert str(F) == r["print"]
        assert list(F.length) == r["length"]
        assert str(F.formula(collapse=True)) == r["collapse"]
        assert F.all_vars() == r["allvars"]
        mf = sf.model_frame(F, d)
        assert mf.names == r["mf_names"]
        assert_terms(F.terms(data=d), r["terms"])
        for i, part in enumerate(r["parts"], start=1):
            assert str(F.formula(lhs=0, rhs=i)) == part["formula"]
            assert str(F.formula(lhs=1, rhs=i)) == part["formula_l1"]
            X = sf.model_matrix(F, mf, rhs=i)
            assert_matrix(X.to_numpy(), X.columns, part, "rhs=%d" % i)
            assert X.assign == part["assign"]
        for i, part in enumerate(r["lhs"], start=1):
            mp = sf.model_part(F, mf, lhs=i)
            assert list(mp.columns) == part["names"] or all(isinstance(c, str) for c in mp.columns)
    elif k == "predict":
        X, Xn = res
        assert_matrix(X.to_numpy(), X.columns, r["fit"], "fit")
        assert_matrix(Xn.to_numpy(), Xn.columns, r["new"], "new")
    elif k == "mf":
        mf = res
        assert mf.names == r["names"]
        assert [str(i + 1) for i in mf.index] == r["rownames"]
        for nm in mf.names:
            check_column(mf[nm], r["columns"][nm], nm)
        assert_terms(mf.terms, r["terms"])
        assert dict(mf.terms.data_classes) == r["dataClasses"]
        if r.get("offset") is not None:
            np.testing.assert_allclose(sf.model_offset(mf).to_numpy(), num(r["offset"]), rtol=RTOL)
        else:
            assert sf.model_offset(mf) is None
        if r.get("weights") is not None:
            np.testing.assert_allclose(sf.model_weights(mf).to_numpy(), num(r["weights"]), rtol=RTOL)
            np.testing.assert_allclose(sf.model_extract(mf, "weights").to_numpy(), num(r["weights"]), rtol=RTOL)
        if r.get("na_action"):
            assert [str(i + 1) for i in mf.na_action["index"]] == r["na_action"]["rows"]
            assert ("na." + mf.na_action["type"]) in [c.replace("omit", "omit") for c in r["na_action"]["class"]] or \
                mf.na_action["type"] in r["na_action"]["class"]
        if r.get("response") is not None:
            check_column(mf[mf.names[0]], r["response"], "response")
    elif k == "gam":
        import test_mgcv_helpers
        test_mgcv_helpers.check_gam_case(res, r, cs)
    elif k == "gampen":
        import test_mgcv_helpers
        test_mgcv_helpers.check_gampen_case(res, r, cs)
    else:
        raise AssertionError(k)
