"""Random model.matrix() formulas: statsformula vs R (tests/fixtures/r_fuzz_mm.json)."""
import json
import os
import warnings

import numpy as np
import pytest

import statsformula as sf
import rcases

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "fixtures", "r_fuzz_mm.json")) as fh:
    DATA = json.load(fh)
CASES = list(zip(DATA["formulas"], DATA["results"]))
D1 = rcases.make_datasets()["d1"]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_fuzz_model_matrix(case):
    f, r = case
    if "error" in r:
        with pytest.raises(Exception) as ei:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sf.model_matrix(sf.formula(f), D1)
        assert r["error"] in str(ei.value)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = sf.model_matrix(sf.formula(f), D1)
    assert list(X.columns) == r["colnames"]
    assert X.assign == r["assign"]
    w = np.arange(1, len(D1) + 1)[:, None]
    A = X.to_numpy()
    np.testing.assert_allclose((A * w).sum(axis=0), r["m1"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose((A ** 2 * w).sum(axis=0), r["m2"], rtol=1e-9, atol=1e-9)
