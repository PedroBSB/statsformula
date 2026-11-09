"""Random formulas: statsformula terms() vs R terms() (tests/fixtures/r_fuzz.json)."""
import json
import os
import warnings

import pytest

import statsformula as sf

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "fixtures", "r_fuzz.json")) as fh:
    DATA = json.load(fh)
CASES = list(zip(DATA["formulas"], DATA["results"]))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_fuzz_terms(case):
    f, r = case
    if "error" in r:
        with pytest.raises(Exception) as ei:
            sf.formula(f).terms()
        assert r["error"] in str(ei.value)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = sf.formula(f).terms()
    assert str(sf.formula(f)) == r["formula"]
    assert t.term_labels == r["labels"]
    assert t.intercept == r["intercept"]
    assert t.response == r["response"]
    assert t.order == r["order"]
    assert t.variables == r["variables"]
    assert (t.offset or None) == (r["offset"] or None)
    fac = [list(map(int, t.factors[:, j])) for j in range(t.factors.shape[1])] if t.factors.size else []
    assert fac == r["factors"]
