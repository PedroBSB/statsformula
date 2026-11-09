"""Behavioural tests of the Python API (no R required)."""

import warnings

import numpy as np
import pandas as pd
import pytest

import statsformula as sf


@pytest.fixture
def df():
    rng = np.random.default_rng(1)
    n = 30
    return pd.DataFrame({
        "y": rng.normal(size=n), "x": rng.uniform(1, 5, n), "z": rng.normal(size=n),
        "g": pd.Categorical(rng.choice(list("abc"), n), categories=list("abc")),
        "s": rng.choice(["lo", "hi"], n), "b": rng.choice([True, False], n),
    })


def test_formula_printing_matches_r():
    f = sf.formula("Y~C(X1):C(X2)+log(X3)/sqrt(X4)+X5**4")
    assert str(f) == "Y ~ C(X1):C(X2) + log(X3)/sqrt(X4) + X5^4"
    assert f.terms().term_labels == ["log(X3)", "X5", "C(X1):C(X2)", "log(X3):sqrt(X4)"]


def test_model_matrix_attributes(df):
    X = sf.model_matrix("y ~ g * x + s", df)
    assert list(X.columns) == ["(Intercept)", "gb", "gc", "x", "slo", "gb:x", "gc:x"]
    assert X.assign == [0, 1, 1, 2, 3, 4, 4]
    assert X.contrasts == {"g": "contr.treatment", "s": "contr.treatment"}
    assert isinstance(X, pd.DataFrame)


def test_numpy_output(df):
    X = sf.model_matrix("~ x + z", df, output="numpy")
    assert isinstance(X, np.ndarray) and X.shape == (30, 3)


def test_python_environment_lookup(df):
    k = 2.5

    def center(v):
        return v - v.mean()

    X = sf.model_matrix("~ I(x * k) + center(z)", df)
    np.testing.assert_allclose(X["I(x * k)"], df.x * 2.5)
    np.testing.assert_allclose(X["center(z)"], df.z - df.z.mean())


def test_numpy_namespaced_functions(df):
    X = sf.model_matrix("~ np.log(x)", df)
    np.testing.assert_allclose(X["np.log(x)"], np.log(df.x))


def test_register_function(df):
    sf.register_function("double_it", lambda v: 2 * np.asarray(v))
    X = sf.model_matrix("~ double_it(x)", df)
    np.testing.assert_allclose(X["double_it(x)"], 2 * df.x)
    with pytest.raises(ValueError):
        sf.register_function("log", np.log)


def test_global_contrasts_option(df):
    old = sf.options(contrasts=("contr.sum", "contr.poly"))
    try:
        X = sf.model_matrix("~ g", df)
        assert list(X.columns) == ["(Intercept)", "g1", "g2"]
    finally:
        sf.options(**old)


def test_contrasts_argument_forms(df):
    m = np.array([[1, 0], [0, 1], [-1, -1]])
    for spec in ("contr.sum", sf.contr_sum, m):
        X = sf.model_matrix("~ g", df, contrasts={"g": spec})
        np.testing.assert_allclose(X.iloc[:, 1:].to_numpy()[df.g.cat.codes.to_numpy() == 2], -1)


def test_model_spec_new_data_and_new_levels(df):
    X = sf.model_matrix("y ~ poly(x, 2) + g + ns(z, 3)", df)
    new = df.iloc[:5]
    Xn = X.model_spec.get_model_matrix(new)
    np.testing.assert_allclose(Xn.to_numpy(), X.iloc[:5].to_numpy())
    bad = new.assign(g=pd.Categorical(["a", "b", "zz", "a", "b"]))
    with pytest.raises(sf.RError, match="new level"):
        X.model_spec.get_model_matrix(bad)


def test_dmatrices_statsmodels_style(df):
    y, X = sf.dmatrices("y ~ x + g", df)
    assert y.shape == (30, 1) and X.shape == (30, 4)
    yy, XX = sf.dmatrices("y ~ x + g", df, return_type="matrix")
    assert isinstance(yy, np.ndarray)
    try:
        import statsmodels.api as sm
    except ImportError:
        return
    fit = sm.OLS(y, X).fit()
    assert len(fit.params) == 4


def test_na_handling(df):
    d = df.copy()
    d.loc[[2, 5], "x"] = np.nan
    X = sf.model_matrix("y ~ x", d)
    assert len(X) == 28 and 2 not in X.index
    with pytest.raises(sf.RError):
        sf.model_frame("y ~ x", d, na_action="na.fail")
    mf = sf.model_frame("y ~ x", d, na_action="na.pass")
    assert len(mf) == 30


def test_multipart_formula(df):
    F = sf.Formula("y | z ~ x + g | s")
    assert F.length == (2, 2)
    mf = sf.model_frame(F, df)
    assert list(sf.model_matrix(F, mf, rhs=2).columns) == ["(Intercept)", "slo"]
    assert list(sf.model_part(F, mf, lhs=2).columns) == ["z"]
    assert str(F.update(". ~ . | . + b")) == "y | z ~ x + g | s + b"
    # string with | dispatches to the Formula package semantics
    X = sf.model_matrix("y ~ x | g", df, rhs=2)
    assert list(X.columns) == ["(Intercept)", "gb", "gc"]


def test_sparse_matrix(df):
    S = sf.sparse_model_matrix("~ g * s", df)
    dense = sf.model_matrix("~ g * s", df)
    assert S.columns == list(dense.columns)
    np.testing.assert_allclose(S.toarray(), dense.to_numpy())
    assert S.to_pandas().shape == dense.shape


def test_update_and_helpers():
    assert str(sf.update("y ~ x + z", ". ~ . - z + w")) == "y ~ x + w"
    assert str(sf.reformulate(["a", "b:c"], response="y")) == "y ~ a + b:c"
    t = sf.terms("y ~ a*b + offset(o)")
    assert sf.delete_response(t).variables == ["a", "b", "offset(o)"]
    assert sf.drop_terms(t, [3], keep_response=True).term_labels == ["a", "b"]
    assert sf.all_vars("log(y) ~ x + s(z, by = g)") == ["y", "x", "z", "g"]


def test_model_extract(df):
    mf = sf.model_frame("y ~ x", df, weights="x", offset="log(x)", subset="x > 2")
    assert (sf.model_extract(mf, "weights") > 2).all()
    np.testing.assert_allclose(sf.model_offset(mf), np.log(sf.model_extract(mf, "weights")))
    assert sf.model_response(mf).name == "y"


def test_errors_are_r_like(df):
    with pytest.raises(sf.RError, match="object 'nope' not found"):
        sf.model_matrix("~ nope", df)
    with pytest.raises(sf.RError, match="invalid power in formula"):
        sf.terms("y ~ x^1")
    with pytest.raises(sf.FormulaSyntaxError):
        sf.formula("y ~ (x")
    with pytest.raises(sf.RError, match="no 'data' argument"):
        sf.terms("y ~ .")


def test_gam_dispatch_and_prediction(df):
    X = sf.model_matrix("y ~ g + s(x, bs = 'cr', k = 5) + te(x, z, k = 3)", df)  # gam.side() applied
    assert [c for c in X.columns if c.startswith("s(x)")] == ["s(x).%d" % i for i in range(1, 5)]
    Xn = X.model_spec.get_model_matrix(df.iloc[:4])
    np.testing.assert_allclose(Xn.to_numpy(), X.iloc[:4].to_numpy(), atol=1e-10)


def test_vectorised_large_frame():
    rng = np.random.default_rng(0)
    n = 200_000
    d = pd.DataFrame({"x": rng.normal(size=n), "f": rng.choice(list("abcd"), n), "g": rng.choice(list("uv"), n)})
    X = sf.model_matrix("~ f*g + x", d)
    assert X.shape == (n, 9)
