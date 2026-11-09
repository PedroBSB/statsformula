"""scikit-learn integration (statsformula.sklearn)."""

import pickle

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.base import clone, is_classifier, is_regressor  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import statsformula as sf  # noqa: E402
from statsformula.sklearn import FormulaEstimator, FormulaTransformer, design_matrices  # noqa: E402


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(1, 5, n)
    z = rng.normal(size=n)
    g = rng.choice(list("abc"), n)
    eff = pd.Series(g).map({"a": 0.0, "b": 1.0, "c": -1.0}).to_numpy()
    y1 = 2 * np.log(x) + 0.5 * z ** 2 + eff + rng.normal(scale=0.1, size=n)
    y2 = -np.log(x) + z + rng.normal(scale=0.1, size=n)
    cls = np.where(y2 + rng.normal(scale=0.3, size=n) > 0, "high", "low")
    return pd.DataFrame({"y1": y1, "y2": y2, "x": x, "z": z, "g": g, "cls": cls})


def test_transformer_matches_model_matrix(df):
    tr = FormulaTransformer("~ log(x) + poly(z, 2) + C(g, sum)")
    Xt = tr.fit_transform(df)
    X = sf.model_matrix("~ log(x) + poly(z, 2) + C(g, sum)", df)
    np.testing.assert_allclose(Xt.to_numpy(), X.to_numpy())
    assert list(tr.get_feature_names_out()) == list(X.columns)
    # stateful: transforming a subset reproduces the training rows exactly
    np.testing.assert_allclose(tr.transform(df.iloc[:7]).to_numpy(), X.iloc[:7].to_numpy())


def test_transformer_in_pipeline(df):
    pipe = make_pipeline(FormulaTransformer("~ log(x) + I(z^2) + g", remove_intercept=True), StandardScaler(),
                         LinearRegression())
    pipe.fit(df, df["y1"])
    assert pipe.score(df, df["y1"]) > 0.95
    assert pipe.predict(df.head(3)).shape == (3,)


def test_transformer_outputs(df):
    tr = FormulaTransformer("~ g * x", output="numpy").fit(df)
    assert isinstance(tr.transform(df), np.ndarray)
    sp = FormulaTransformer("~ g * x", output="sparse").fit(df)
    import scipy.sparse as sps
    assert sps.issparse(sp.transform(df))
    np.testing.assert_allclose(sp.transform(df).toarray(), tr.transform(df))


def test_transformer_set_output_and_column_transformer(df):
    ct = ColumnTransformer([("f", FormulaTransformer("~ log(x) + g - 1"), ["x", "g"]),
                            ("s", StandardScaler(), ["z"])])
    out = ct.set_output(transform="pandas").fit_transform(df)
    assert list(out.columns) == ["f__log(x)", "f__ga", "f__gb", "f__gc", "s__z"]


def test_transformer_keeps_rows_with_na(df):
    d = df.copy()
    d.loc[3, "x"] = np.nan
    tr = FormulaTransformer("~ log(x) + g").fit(df)
    out = tr.transform(d)
    assert len(out) == len(d) and np.isnan(out.loc[3, "log(x)"])


def test_new_factor_level_raises(df):
    tr = FormulaTransformer("~ g").fit(df)
    with pytest.raises(sf.RError, match="new level"):
        tr.transform(df.assign(g="zz"))


def test_design_matrices_multi_response(df):
    Y, X = design_matrices("cbind(y1, y2) ~ log(x) + g", df)
    assert list(Y.columns) == ["y1", "y2"] and Y.shape == (200, 2)
    Y2, _ = design_matrices("y1 | y2 ~ log(x) + g", df)
    assert list(Y2.columns) == ["y1", "y2"]
    np.testing.assert_allclose(Y.to_numpy(), Y2.to_numpy())
    Yl, _ = design_matrices("log(y1 + 10) ~ x", df)
    np.testing.assert_allclose(Yl.iloc[:, 0], np.log(df.y1 + 10))
    Yc, _ = design_matrices("cls ~ x", df, y_encoding="onehot")
    assert list(Yc.columns) == ["clshigh", "clslow"]
    Yk, _ = design_matrices("cls ~ x", df, y_encoding="codes")
    assert set(Yk.iloc[:, 0]) == {0, 1}


def test_design_matrices_drop_na_jointly(df):
    d = df.copy()
    d.loc[[1, 5], "y2"] = np.nan
    d.loc[9, "x"] = np.nan
    Y, X = design_matrices("cbind(y1, y2) ~ log(x)", d)
    assert len(Y) == len(X) == 197
    assert list(Y.index) == list(X.index)


def test_formula_estimator_multioutput(df):
    est = FormulaEstimator("cbind(y1, y2) ~ log(x) + I(z^2) + z + g", Ridge(alpha=1e-6)).fit(df)
    assert est.response_names_ == ["y1", "y2"]
    pred = est.predict(df.head(4))
    assert pred.shape == (4, 2)
    assert est.score(df) > 0.98
    assert is_regressor(est)


def test_formula_estimator_classifier(df):
    est = FormulaEstimator("cls ~ log(x) + z", LogisticRegression())
    est.fit(df)
    assert is_classifier(est)
    assert list(est.classes_) == ["high", "low"]
    assert est.predict_proba(df.head(5)).shape == (5, 2)
    assert set(est.predict(df)) <= {"high", "low"}


def test_formula_estimator_with_pipeline_and_gridsearch(df):
    est = FormulaEstimator("y1 ~ log(x) + g", Pipeline([("sc", StandardScaler()), ("r", Ridge())]),
                           remove_intercept=True)
    grid = GridSearchCV(est, {"formula": ["y1 ~ x + g", "y1 ~ log(x) + I(z^2) + g"],
                              "estimator__r__alpha": [0.01, 10.0]}, cv=KFold(3, shuffle=True, random_state=0))
    grid.fit(df)
    assert grid.best_params_["formula"] == "y1 ~ log(x) + I(z^2) + g"
    scores = cross_val_score(clone(est), df, cv=3)
    assert scores.shape == (3,)


def test_formula_estimator_sample_weight_and_na(df):
    d = df.copy()
    d.loc[[2, 4], "x"] = np.nan
    w = np.ones(len(d))
    est = FormulaEstimator("y1 ~ log(x)", LinearRegression()).fit(d, sample_weight=w)
    assert est.estimator_.coef_.shape[-1] == 2


def test_formula_estimator_gam_smooths(df):
    est = FormulaEstimator("y1 ~ g + s(x, k = 5) + s(z, bs = 'cr', k = 6)", LinearRegression(fit_intercept=False))
    est.fit(df)
    assert est.score(df) > 0.97
    assert est.predict(df.head(2)).shape == (2,)


def test_pickle_roundtrip(df):
    est = FormulaEstimator("y2 ~ ns(x, 3) + C(g, helmert)", Ridge()).fit(df)
    est2 = pickle.loads(pickle.dumps(est))
    np.testing.assert_allclose(est.predict(df), est2.predict(df))
    tr = FormulaTransformer("~ poly(x, 3)").fit(df)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(tr)).transform(df), tr.transform(df))


def test_custom_functions_via_env(df):
    tr = FormulaTransformer("~ sq(z)", env={"sq": np.square}).fit(df)
    np.testing.assert_allclose(tr.transform(df)["sq(z)"], df.z ** 2)


# -- mgcv penalties / GAMRegressor ---------------------------------------------

from statsformula.sklearn import GAMRegressor  # noqa: E402


def test_transformer_exposes_penalties(df):
    tr = FormulaTransformer("~ g + s(x) + te(x, z, k = 4)").fit(df)
    names = [p.name for p in tr.penalties_]
    assert names == ["s(x)", "te(x,z)1", "te(x,z)2"]
    X = tr.transform(df)
    P = tr.penalty_matrix([1.0, 2.0, 3.0])
    assert P.shape == (X.shape[1], X.shape[1])
    np.testing.assert_allclose(P, P.T)
    assert np.all(P[:3] == 0) and np.all(P[:, :3] == 0)  # parametric columns are unpenalized
    tr0 = FormulaTransformer("~ g + s(x)", remove_intercept=True).fit(df)
    p0 = tr0.penalties_[0]
    assert list(tr0.get_feature_names_out()[p0.columns]) == ["s(x).%d" % i for i in range(1, 10)]
    assert FormulaTransformer("~ g + x").fit(df).penalties_ == []


def test_penalty_matrix_generalized_ridge_equals_gam(df):
    b = sf.gam("y1 ~ g + s(x) + s(z)", df, method="REML")
    X = b.model_matrix.to_numpy()
    P = b.penalty_matrix()
    beta = np.linalg.solve(X.T @ X + P, X.T @ df["y1"].to_numpy())
    np.testing.assert_allclose(beta, b.coefficients.to_numpy(), rtol=1e-8, atol=1e-10)


def test_gam_regressor(df):
    est = GAMRegressor("y1 ~ g + s(x) + s(z, bs = 'cr')").fit(df)
    b = sf.gam("y1 ~ g + s(x) + s(z, bs = 'cr')", df)
    np.testing.assert_allclose(est.sp_, b.sp.to_numpy())
    np.testing.assert_allclose(est.predict(df.head(5)), b.predict(df.head(5)).to_numpy())
    assert est.score(df) > 0.97 and is_regressor(est)
    assert est.sp_names_ == ["s(x)", "s(z)"]
    # fixed smoothing parameters, tuned by grid search
    grid = GridSearchCV(GAMRegressor("y1 ~ g + s(x) + s(z)"), {"sp": [[1e-3, 1e-3], [10.0, 10.0], [1e4, 1e4]]}, cv=3)
    grid.fit(df)
    assert grid.best_params_["sp"] != [1e4, 1e4]
    # weights and multiple responses
    w = np.linspace(0.5, 2, len(df))
    ew = GAMRegressor("y1 ~ s(x)", method="GCV.Cp").fit(df, sample_weight=w)
    bw = sf.gam("y1 ~ s(x)", df.assign(w=w), method="GCV.Cp", weights="w")
    np.testing.assert_allclose(ew.coef_, bw.coefficients.to_numpy())
    em = GAMRegressor("cbind(y1, y2) ~ g + s(x) + s(z)").fit(df)
    assert em.predict(df.head(3)).shape == (3, 2) and em.sp_.shape == (2, 2)
    est2 = pickle.loads(pickle.dumps(est))
    np.testing.assert_allclose(est2.predict(df), est.predict(df))
