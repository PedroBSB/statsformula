"""scikit-learn integration: R formulas as pipeline steps.

* :class:`FormulaTransformer` - a transformer that turns a DataFrame into the
  design matrix of a formula's right-hand side.  It learns everything that
  depends on the training data on ``fit`` (factor levels, contrasts, ``poly``
  / ``ns`` / ``bs`` / ``scale`` parameters, mgcv smooth bases) and re-applies
  it on ``transform``, exactly like R's ``predict()``.
* :class:`FormulaEstimator` - a meta-estimator that builds **both** the
  response matrix ``Y`` (left-hand side: ``y``, ``log(y)``, ``cbind(y1, y2)``,
  ``y1 | y2 ~ ...``, factors) and the design matrix ``X`` from the formula
  and fits any scikit-learn estimator or pipeline on them.  A scikit-learn
  ``Pipeline`` step can only transform ``X``, so building ``Y`` from the
  formula requires wrapping the model; this class does that while staying
  compatible with ``clone``, ``GridSearchCV``, ``cross_val_score`` and pickling.
* :func:`design_matrices` - ``(Y, X)`` for a formula and a DataFrame.

Requires ``scikit-learn`` (``pip install statsformula[sklearn]``).
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, TransformerMixin, clone
    from sklearn.utils.validation import check_is_fitted
except ImportError as exc:  # pragma: no cover
    raise ImportError("statsformula.sklearn requires scikit-learn: pip install statsformula[sklearn]") from exc

from ._api import ModelMatrix, SparseModelMatrix, model_frame, model_matrix, model_part, sparse_model_matrix
from ._formula import Formula, RFormula, coerce_formula
from ._model import ModelFrame
from ._rtypes import Character, Factor, Logical, RError, RMatrix, as_factor

__all__ = ["FormulaTransformer", "FormulaEstimator", "GAMRegressor", "PathModel", "design_matrices", "response_matrix"]

_Y_ENCODINGS = ("auto", "numeric", "onehot", "codes", "labels")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, Mapping):
        return pd.DataFrame(dict(X))
    raise TypeError("statsformula estimators need a pandas DataFrame (or a dict of columns) whose "
                    "column names match the variables used in the formula; got %s" % type(X).__name__)


def _response_block(name: str, v, y_encoding: str) -> pd.DataFrame:
    """Encode one response variable as columns of the response matrix."""
    if isinstance(v, RMatrix):
        if v.colnames is not None and all(v.colnames):
            cols = list(v.colnames) if name.startswith("cbind(") else [name + c for c in v.colnames]
        else:
            cols = [name + str(j + 1) for j in range(v.ncol)] if v.ncol > 1 else [name]
        return pd.DataFrame(v.data, columns=cols)
    categorical = isinstance(v, (Factor, Character, Logical))
    if categorical:
        if isinstance(v, Logical) and y_encoding in ("auto", "numeric"):
            return pd.DataFrame({name: v.to_float()})
        f = v if isinstance(v, Factor) else as_factor(v, levels=["FALSE", "TRUE"] if isinstance(v, Logical) else None)
        if y_encoding in ("auto", "labels"):
            return pd.DataFrame({name: pd.Categorical.from_codes(f.codes, f.levels, ordered=f.ordered)})
        if y_encoding == "codes":
            return pd.DataFrame({name: f.codes.astype(np.int64)})
        if y_encoding == "onehot":
            oh = np.zeros((len(f), f.nlevels))
            ok = f.codes >= 0
            oh[np.nonzero(ok)[0], f.codes[ok]] = 1.0
            oh[~ok] = np.nan
            return pd.DataFrame(oh, columns=[name + lev for lev in f.levels])
        if y_encoding == "numeric":
            raise RError("response '%s' is categorical; use y_encoding='labels', 'codes' or 'onehot'" % name)
    return pd.DataFrame({name: np.asarray(v, dtype=float)})


def response_matrix(mf: ModelFrame, formula=None, y_encoding: str = "auto") -> Optional[pd.DataFrame]:
    """The response ("Y design matrix") of a model frame as a DataFrame.

    Single responses give one column; ``cbind(y1, y2) ~ ...`` gives one column
    per ``cbind`` argument; multi-part ``y1 | y2 ~ ...`` formulas give the
    variables of every left-hand side part.  Categorical responses are kept as
    labels (``y_encoding="auto"``/``"labels"``), integer codes (``"codes"``) or
    one indicator column per level (``"onehot"``).
    """
    if y_encoding not in _Y_ENCODINGS:
        raise ValueError("y_encoding must be one of %s" % (_Y_ENCODINGS,))
    if isinstance(formula, Formula) and formula.length[0] > 0:
        lhs_terms = formula.terms(lhs=None, rhs=0)
        names = lhs_terms.variables
    elif mf.terms.response:
        names = [list(mf.variables)[0]]
    else:
        return None
    blocks = [_response_block(nm, mf[nm], y_encoding) for nm in names]
    Y = pd.concat(blocks, axis=1)
    Y.index = mf.index
    return Y


def _response_model_frame(f, data, na_action, env):
    """Model frame holding the response (mgcv formulas use their 'fake' formula)."""
    if isinstance(f, RFormula):
        from ._mgcv import has_smooths, interpret_gam
        if has_smooths(f):
            from ._model import model_frame_from_terms
            ft = interpret_gam(f, env).fake_formula.terms(data=data)
            return model_frame_from_terms(ft, data, env=env, na_action=na_action, drop_unused_levels=True)
    return model_frame(f, data, na_action=na_action, env=env)


def design_matrices(formula, data, y_encoding: str = "auto", contrasts=None, na_action: str = "na.omit",
                    remove_intercept: bool = False, env=None, rhs=1):
    """``(Y, X)``: response matrix and design matrix of a formula.

    Rows with missing values in any variable (response or predictors) are
    removed jointly (``na.omit``), so ``Y`` and ``X`` stay aligned.  ``X``
    keeps its ``model_spec`` for transforming new data.
    """
    env = {} if env is None else env
    data = _as_frame(data)
    f = coerce_formula(formula, env)
    if isinstance(f, Formula):
        mf = model_frame(f, data, na_action=na_action, env=env)
        X = model_matrix(f, mf, contrasts=contrasts, rhs=rhs, env=env)
    else:
        from ._mgcv import has_smooths
        if has_smooths(f):
            X = model_matrix(f, data, contrasts=contrasts, na_action=na_action, env=env)
            mf = _response_model_frame(f, data, na_action, env)
        else:
            mf = model_frame(f, data, na_action=na_action, env=env)
            X = model_matrix(mf, contrasts=contrasts, env=env)
    Y = response_matrix(mf, f, y_encoding)
    if Y is None:
        raise RError("the formula has no left-hand side (response)")
    if remove_intercept and "(Intercept)" in X.columns:
        X = _drop_intercept(X)
    return Y, X


def _drop_intercept(X):
    keep = [c != "(Intercept)" for c in X.columns]
    spec = getattr(X, "model_spec", None)
    out = X.loc[:, keep]
    if isinstance(out, ModelMatrix):
        object.__setattr__(out, "assign", [a for a, k in zip(X.assign, keep) if k])
        object.__setattr__(out, "contrasts", X.contrasts)
        object.__setattr__(out, "model_spec", spec)
    return out


def _penalties_for(spec, columns, remove_intercept):
    """mgcv penalties of a fitted spec, with offsets shifted if the intercept is dropped."""
    from ._mgcv import GamPenalty
    pens = list(getattr(spec, "penalties", None) or [])
    if remove_intercept and "(Intercept)" in columns:
        i0 = list(columns).index("(Intercept)")
        pens = [GamPenalty(p.name, p.smooth, p.S, p.off - (p.off > i0), p.rank) for p in pens]
    return pens


def _spec_transform(spec, data, na_action):
    # R's predict() warns "contrasts dropped from factor C(...)" because model.frame(xlev=)
    # re-creates the factor; the stored contrasts are re-applied, so it is noise here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="contrasts dropped from factor")
        try:
            return spec.get_model_matrix(data, na_action=na_action)
        except TypeError:  # GamSpec: rows are always kept (na.pass)
            return spec.get_model_matrix(data)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class FormulaTransformer(TransformerMixin, BaseEstimator):
    """Design matrix of a formula's right-hand side as a scikit-learn transformer.

    Parameters
    ----------
    formula : str
        An R formula, e.g. ``"~ log(x) + poly(z, 2) + C(g, sum)"``; a
        left-hand side is allowed and ignored (use :class:`FormulaEstimator`
        to build the response too).  Multi-part formulas select the part with
        ``rhs``.  mgcv smooths (``s()``, ``te()``, ``ti()``) are supported.
    contrasts : dict, optional
        R's ``contrasts.arg``, e.g. ``{"g": "contr.sum"}``.
    remove_intercept : bool, default False
        Drop the ``(Intercept)`` column after building the matrix (contrast
        coding is unchanged), for estimators that fit their own intercept.
    output : {"pandas", "numpy", "sparse"}
        Return type of ``transform``.  ``"sparse"`` gives a
        ``scipy.sparse.csc_matrix`` built like ``Matrix::sparse.model.matrix``.
    na_action : str, default "na.pass"
        Missing values handling in ``transform``.  ``"na.pass"`` keeps every
        row (rows with missing values contain NaN) so the output stays aligned
        with ``y``; ``"na.fail"`` raises.
    env : dict, optional
        Extra names (functions, constants) usable inside the formula.
    rhs : int, default 1
        Right-hand side part for multi-part ``Formula`` objects.
    """

    def __init__(self, formula: str = "~ .", contrasts: Optional[dict] = None, remove_intercept: bool = False,
                 output: str = "pandas", na_action: str = "na.pass", env: Optional[dict] = None, rhs: int = 1):
        self.formula = formula
        self.contrasts = contrasts
        self.remove_intercept = remove_intercept
        self.output = output
        self.na_action = na_action
        self.env = env
        self.rhs = rhs

    def fit(self, X, y=None):
        data = _as_frame(X)
        if self.output not in ("pandas", "numpy", "sparse"):
            raise ValueError("output must be 'pandas', 'numpy' or 'sparse'")
        env = {} if self.env is None else self.env
        if self.output == "sparse":
            M = sparse_model_matrix(self.formula, data, contrasts=self.contrasts, rhs=self.rhs, env=env)
            cols = list(M.columns)
        else:
            M = model_matrix(self.formula, data, contrasts=self.contrasts, rhs=self.rhs, env=env)
            cols = list(M.columns)
        self.model_spec_ = M.model_spec
        self.columns_ = cols
        self.feature_names_in_ = np.asarray([str(c) for c in data.columns], dtype=object)
        self.n_features_in_ = data.shape[1]
        self.feature_names_out_ = np.asarray([c for c in cols if not (self.remove_intercept and c == "(Intercept)")],
                                             dtype=object)
        self.penalties_ = _penalties_for(self.model_spec_, cols, self.remove_intercept)
        return self

    def penalty_matrix(self, sp=None):
        """``sum_j sp[j] * S_j`` over the output columns: mgcv's smoothing penalties
        (``s()``, ``te()``, ``ti()`` terms; zero elsewhere).  Use it to fit a
        generalized ridge ``||y - X b||^2 + b' P b``; see also :class:`GAMRegressor`."""
        from ._mgcv import penalty_matrix
        check_is_fitted(self, "penalties_")
        return penalty_matrix(self.penalties_, len(self.feature_names_out_), sp)

    def transform(self, X):
        check_is_fitted(self, "model_spec_")
        data = _as_frame(X)
        M = _spec_transform(self.model_spec_, data, self.na_action)
        if isinstance(M, SparseModelMatrix):
            mat = M.matrix
            if self.remove_intercept and "(Intercept)" in M.columns:
                keep = [i for i, c in enumerate(M.columns) if c != "(Intercept)"]
                mat = mat[:, keep]
            return mat
        df = M if isinstance(M, pd.DataFrame) else pd.DataFrame(M, columns=self.columns_)
        if list(df.columns) != self.columns_:
            df.columns = self.columns_
        if self.remove_intercept and "(Intercept)" in df.columns:
            df = df.drop(columns="(Intercept)")
        if self.output == "numpy":
            return df.to_numpy()
        return pd.DataFrame(df.to_numpy(), index=df.index, columns=list(df.columns))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = True
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        tags.requires_fit = True
        return tags


# ---------------------------------------------------------------------------
# Meta-estimator
# ---------------------------------------------------------------------------

class FormulaEstimator(MetaEstimatorMixin, BaseEstimator):
    """Fit any scikit-learn estimator (or pipeline) on the ``(Y, X)`` matrices of a formula.

    >>> from sklearn.linear_model import Ridge
    >>> model = FormulaEstimator("cbind(y1, y2) ~ log(x) + C(g, sum) + ns(z, 3)", Ridge(alpha=1.0))
    >>> model.fit(train_df).predict(test_df)            # doctest: +SKIP

    ``fit(X, y=None)``: ``X`` is the DataFrame holding all variables; the
    response is taken from the formula's left-hand side unless ``y`` is
    given.  Rows with missing values are removed jointly from ``Y`` and ``X``
    (``na_action="na.omit"``), and any ``sample_weight``-like fit parameter is
    subset accordingly.  ``predict``/``score`` build ``X`` with the stored
    transformations (factor levels, contrasts, spline knots, ...).

    Parameters
    ----------
    formula : str
        R formula with a left-hand side.  Multiple responses:
        ``cbind(y1, y2) ~ ...`` or the multi-part ``y1 | y2 ~ ...``.
    estimator : estimator
        Any scikit-learn estimator or ``Pipeline``; its hyper-parameters are
        reachable as ``estimator__<name>`` for ``GridSearchCV``.
    y_encoding : {"auto", "numeric", "labels", "codes", "onehot"}
        Encoding of categorical responses (``"auto"`` keeps labels, which
        classifiers expect).
    remove_intercept : bool, default False
        Drop the ``(Intercept)`` column (for estimators with their own intercept).
    contrasts, env : see :class:`FormulaTransformer`.
    na_action : str, default "na.omit"
        Missing-value handling when fitting.
    """

    def __init__(self, formula: str = "y ~ .", estimator=None, y_encoding: str = "auto",
                 remove_intercept: bool = False, contrasts: Optional[dict] = None,
                 na_action: str = "na.omit", env: Optional[dict] = None):
        self.formula = formula
        self.estimator = estimator
        self.y_encoding = y_encoding
        self.remove_intercept = remove_intercept
        self.contrasts = contrasts
        self.na_action = na_action
        self.env = env

    # -- sklearn plumbing ---------------------------------------------------
    @property
    def _estimator_type(self):
        return getattr(self.estimator, "_estimator_type", None)

    def __sklearn_tags__(self):
        from sklearn.utils import get_tags
        tags = super().__sklearn_tags__()
        if self.estimator is not None:
            inner = get_tags(self.estimator)
            tags.estimator_type = inner.estimator_type
            tags.target_tags = inner.target_tags
            if inner.classifier_tags is not None:
                tags.classifier_tags = inner.classifier_tags
            if inner.regressor_tags is not None:
                tags.regressor_tags = inner.regressor_tags
        tags.target_tags.required = False
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        return tags

    def _design_X(self, X):
        check_is_fitted(self, "model_spec_")
        M = _spec_transform(self.model_spec_, _as_frame(X), "na.pass")
        df = M if isinstance(M, pd.DataFrame) else pd.DataFrame(M)
        if self.remove_intercept and "(Intercept)" in df.columns:
            df = df.drop(columns="(Intercept)")
        return pd.DataFrame(df.to_numpy(), columns=list(df.columns))

    @staticmethod
    def _y_values(Y: pd.DataFrame):
        if Y.shape[1] == 1:
            col = Y.iloc[:, 0]
            if isinstance(col.dtype, pd.CategoricalDtype):
                return np.asarray(col.astype(object))
            return col.to_numpy()
        return Y.to_numpy()

    # -- fitting ------------------------------------------------------------
    def fit(self, X, y=None, **fit_params):
        if self.estimator is None:
            raise ValueError("FormulaEstimator needs an estimator, e.g. FormulaEstimator('y ~ x', Ridge())")
        data = _as_frame(X).reset_index(drop=True)
        n = len(data)
        env = {} if self.env is None else self.env
        if y is not None:
            f = coerce_formula(self.formula, env)
            rhs_formula = f.delete_response() if isinstance(f, RFormula) else f
            Xm = model_matrix(rhs_formula, data, contrasts=self.contrasts, na_action=self.na_action, env=env)
            yv = np.asarray(y)
            Y = pd.DataFrame(yv.reshape(n, -1) if yv.ndim > 1 else yv, index=data.index)
            Y = Y.loc[Xm.index]
        else:
            Y, Xm = design_matrices(self.formula, data, y_encoding=self.y_encoding, contrasts=self.contrasts,
                                    na_action=self.na_action, env=env)
        rows = np.asarray(Xm.index)
        self.model_spec_ = Xm.model_spec
        self.response_names_ = list(map(str, Y.columns))
        self.n_features_in_ = data.shape[1]
        self.feature_names_in_ = np.asarray([str(c) for c in data.columns], dtype=object)
        Xd = Xm.drop(columns="(Intercept)") if (self.remove_intercept and "(Intercept)" in Xm.columns) else Xm
        self.design_columns_ = list(Xd.columns)
        self.penalties_ = _penalties_for(self.model_spec_, list(Xm.columns), self.remove_intercept)
        Xd = pd.DataFrame(Xd.to_numpy(), columns=self.design_columns_)
        params = {}
        for k, v in fit_params.items():
            if v is not None and hasattr(v, "__len__") and not isinstance(v, (str, dict)) and len(v) == n:
                params[k] = np.asarray(v)[rows]
            else:
                params[k] = v
        self.estimator_ = clone(self.estimator).fit(Xd, self._y_values(Y), **params)
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    # -- delegation -----------------------------------------------------------
    def predict(self, X, **params):
        return self.estimator_.predict(self._design_X(X), **params)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self._design_X(X))

    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(self._design_X(X))

    def decision_function(self, X):
        return self.estimator_.decision_function(self._design_X(X))

    def transform(self, X):
        """The design matrix (``X``) for new data."""
        return self._design_X(X)

    def score(self, X, y=None, sample_weight=None):
        data = _as_frame(X).reset_index(drop=True)
        if y is None:
            env = {} if self.env is None else self.env
            f = coerce_formula(self.formula, env)
            mf = _response_model_frame(f, data, self.na_action, env)
            Y = response_matrix(mf, f, self.y_encoding)
            rows = np.asarray(mf.index)
            Xd = self._design_X(data.iloc[rows])
            yv = self._y_values(Y)
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight)[rows]
        else:
            Xd = self._design_X(data)
            yv = y
        if sample_weight is None:
            return self.estimator_.score(Xd, yv)
        return self.estimator_.score(Xd, yv, sample_weight=sample_weight)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "design_columns_")
        return np.asarray(self.design_columns_, dtype=object)


# ---------------------------------------------------------------------------
# Penalized GAM regressor
# ---------------------------------------------------------------------------

class GAMRegressor(RegressorMixin, BaseEstimator):
    """Gaussian GAM with mgcv's smoothing penalties (``mgcv::gam(formula, method=)``).

    The design matrix and the wiggliness penalties ``S_j`` are mgcv's; the
    coefficients minimise ``||y - X b||^2_W + sum_j sp_j b' S_j b`` with the
    smoothing parameters ``sp`` estimated by REML or GCV (matching ``mgcv``'s
    ``sp``, coefficients and fitted values) or fixed by the user.

    >>> GAMRegressor("y ~ f + s(x) + te(x, z)").fit(df).predict(new)   # doctest: +SKIP

    Parameters
    ----------
    formula : str
        Formula with ``s()``, ``te()``, ``ti()`` smooths.  ``cbind(y1, y2) ~ ...``
        fits one GAM per response (each with its own smoothing parameters).
    method : {"REML", "GCV.Cp"}
    sp : array-like, optional
        Fixed smoothing parameters (one per penalty, in ``penalties_`` order);
        negative entries are estimated.  A tunable hyper-parameter for ``GridSearchCV``.
    gamma : float, default 1
        GCV inflation of the effective degrees of freedom (``mgcv``'s ``gamma``).
    knots : dict, optional
        mgcv ``knots`` argument.
    na_action, env : see :class:`FormulaEstimator`.

    Attributes
    ----------
    coef_ : ndarray (n_features,) or (n_responses, n_features)
    sp_ : ndarray of smoothing parameters (per response for ``cbind``)
    edf_ : total effective degrees of freedom; ``scale_`` : residual variance
    penalties_ : list of :class:`statsformula.GamPenalty`
    design_columns_, response_names_, fits_ (the per-response fits)
    """

    def __init__(self, formula: str = "y ~ s(x)", method: str = "REML", sp=None, gamma: float = 1.0,
                 knots: Optional[dict] = None, na_action: str = "na.omit", env: Optional[dict] = None):
        self.formula = formula
        self.method = method
        self.sp = sp
        self.gamma = gamma
        self.knots = knots
        self.na_action = na_action
        self.env = env

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X, y=None, sample_weight=None):
        from ._gamfit import fit_penalized
        from ._mgcv import gam_model_matrix, has_smooths
        data = _as_frame(X).reset_index(drop=True)
        env = {} if self.env is None else self.env
        f = coerce_formula(self.formula, env)
        if y is not None:
            rhs_f = f.delete_response()
            Xm = (gam_model_matrix(rhs_f, data, knots=self.knots, na_action=self.na_action, env=env)
                  if has_smooths(rhs_f) else model_matrix(rhs_f, data, na_action=self.na_action, env=env))
            yv = np.asarray(y, dtype=float)
            Y = pd.DataFrame(yv.reshape(len(data), -1), index=data.index).loc[Xm.index]
            Y.columns = ["y"] if Y.shape[1] == 1 else ["y%d" % (i + 1) for i in range(Y.shape[1])]
        else:
            Xm = (gam_model_matrix(f, data, knots=self.knots, na_action=self.na_action, env=env)
                  if has_smooths(f) else model_matrix(f, data, na_action=self.na_action, env=env))
            mf = _response_model_frame(f, data, self.na_action, env)
            Y = response_matrix(mf, f, "numeric")
            if Y is None:
                raise RError("the formula has no left-hand side (response)")
            Y = Y.loc[Xm.index]
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[np.asarray(Xm.index)]
        self.model_spec_ = Xm.model_spec
        self.design_columns_ = list(Xm.columns)
        self.response_names_ = list(map(str, Y.columns))
        self.penalties_ = list(getattr(Xm.model_spec, "penalties", None) or [])
        self.n_features_in_ = data.shape[1]
        self.feature_names_in_ = np.asarray([str(c) for c in data.columns], dtype=object)
        A = Xm.to_numpy()
        self.fits_ = [fit_penalized(A, Y.iloc[:, j].to_numpy(dtype=float), self.penalties_, method=self.method,
                                    sp=self.sp, weights=w, gamma=self.gamma) for j in range(Y.shape[1])]
        one = len(self.fits_) == 1
        stack = (lambda a: a[0]) if one else np.vstack
        self.coef_ = stack([ft.coef for ft in self.fits_])
        self.sp_ = stack([ft.sp for ft in self.fits_])
        self.edf_ = self.fits_[0].edf.sum() if one else np.array([ft.edf.sum() for ft in self.fits_])
        self.scale_ = self.fits_[0].scale if one else np.array([ft.scale for ft in self.fits_])
        self.sp_names_ = [p.name for p in self.penalties_]
        return self

    def _design(self, X):
        check_is_fitted(self, "model_spec_")
        M = _spec_transform(self.model_spec_, _as_frame(X), "na.pass")
        return M.to_numpy() if isinstance(M, pd.DataFrame) else np.asarray(M)

    def predict(self, X, return_std: bool = False):
        """Predictions; ``return_std=True`` also returns their standard errors
        (``predict.gam(se.fit = TRUE)``, from the Bayesian covariance ``Vp``)."""
        A = self._design(X)
        pred = A @ np.asarray(self.coef_).T
        if not return_std:
            return pred
        se = np.column_stack([np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", A, ft.Vp, A), 0.0))
                              for ft in self.fits_])
        return pred, (se[:, 0] if np.ndim(pred) == 1 else se)

    def transform(self, X):
        """The (lpmatrix) design matrix for new data."""
        return pd.DataFrame(self._design(X), columns=self.design_columns_)

    def penalty_matrix(self, sp=None):
        """``sum_j sp_j S_j`` (defaults to the fitted smoothing parameters of the first response)."""
        from ._mgcv import penalty_matrix
        check_is_fitted(self, "penalties_")
        if sp is None:
            sp = self.fits_[0].sp
        return penalty_matrix(self.penalties_, len(self.design_columns_), sp)

    def score(self, X, y=None, sample_weight=None):
        from sklearn.metrics import r2_score
        data = _as_frame(X).reset_index(drop=True)
        if y is None:
            env = {} if self.env is None else self.env
            f = coerce_formula(self.formula, env)
            mf = _response_model_frame(f, data, self.na_action, env)
            Y = response_matrix(mf, f, "numeric")
            rows = np.asarray(mf.index)
            data = data.iloc[rows]
            y = Y.to_numpy(dtype=float)
            if Y.shape[1] == 1:
                y = y[:, 0]
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight)[rows]
        return r2_score(y, self.predict(data), sample_weight=sample_weight)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "design_columns_")
        return np.asarray(self.design_columns_, dtype=object)


# ---------------------------------------------------------------------------
# Path models (systems of regression equations)
# ---------------------------------------------------------------------------

class PathModel(BaseEstimator):
    """A system of regressions written as a multi-line formula (a path model).

    >>> model = '''
    ...   m ~ x + z            # mediator
    ...   y ~ m + x + log(w)   # outcome
    ... '''
    >>> pm = PathModel(model).fit(df)                        # doctest: +SKIP
    >>> pm.coefficients()          # lavaan's parameterEstimates(): est, se, z, pvalue, CI  # doctest: +SKIP
    >>> pm.predict(new)            # m and y; y uses the *predicted* m   # doctest: +SKIP
    >>> pm.effects("indirect", se=True)   # x -> m -> y with delta-method SEs  # doctest: +SKIP

    Each line is an R formula (see :mod:`statsformula._path` for the syntax:
    comments, continuation lines, ``y1 + y2 ~ x`` and ``cbind(y1, y2) ~ x``
    give one equation per response, repeated responses are merged); lines
    ``y1 ~~ y2`` free the covariance of two disturbances.

    **Estimation.**  A recursive model with uncorrelated disturbances is fitted
    equation by equation (each on its own complete cases) with a clone of
    ``estimator``; with the default linear estimator this is the maximum
    likelihood fit (``lavaan::sem()`` with observed variables).  Models with
    feedback loops (``y1 ~ y2`` and ``y2 ~ y1``: non-recursive) or residual
    covariances are fitted by full-information maximum likelihood on the rows
    complete for every equation (needs linear equations in which every
    endogenous predictor enters as its own column, see below).

    **Inference** (``se``): ``"expected"`` (default) and ``"observed"`` give the
    standard errors of ``lavaan::sem(information = "expected"/"observed")`` with
    z tests; ``"lm"`` the per-equation t tests of ``summary(lm())`` (equation by
    equation estimation only); ``"none"`` skips them.  Expected-information
    standard errors use the whole system, so they need every equation to be
    linear in the endogenous variables (``y ~ m + log(x)`` is, ``y ~ log(m)`` is
    not) and the equations to share their rows; otherwise each equation uses its
    observed information ``sigma2_ML (X'X)^-1``.  Equations with smooths use
    mgcv's Bayesian covariance ``Vp``; other estimators (``Ridge`` ...) get no
    standard errors.

    Parameters
    ----------
    model : str
        Multi-line model: one ``response ~ predictors`` formula per line, and
        optional ``y1 ~~ y2`` residual covariances.
    estimator : estimator or dict, optional
        Estimator cloned for every equation, or ``{response: estimator}``.
        Default: ``LinearRegression(fit_intercept=False)`` on the formula's design
        matrix (which has the intercept), or :class:`GAMRegressor` for equations
        with ``s()``/``te()``/``ti()`` smooths.
    estimation : {"auto", "ML", "OLS"}
        ``"auto"``: equation by equation when that is ML, full-information ML
        otherwise.  ``"ML"``: always the joint fit.  ``"OLS"``: always equation by
        equation (inconsistent for non-recursive models).
    se : {"expected", "observed", "lm", "none"}
    level : float
        Confidence level of the intervals (``ci.lower``/``ci.upper``).
    predict_mode : {"propagate", "direct"}
        ``"propagate"`` (default): only exogenous variables are taken from the
        data; endogenous predictors are replaced by their predictions (the
        reduced form ``(I - B)^-1 G x``, also for non-recursive models).
        ``"direct"``: every equation uses the observed values of its predictors.
    gam_method : {"REML", "GCV.Cp"}
        Smoothing-parameter method of the default GAM equations.
    contrasts, na_action, env : see :class:`FormulaEstimator`.

    Attributes
    ----------
    equations_ : list of :class:`PathEquation`
    covariances_ : list of ``(a, b)`` residual covariances
    estimators_ : dict response -> fitted per-equation estimator
    endogenous_, exogenous_ : variable names
    order_ : responses in causal (topological) order, or None if non-recursive
    estimation_ : ``"OLS"`` (equation by equation) or ``"ML"`` (joint)
    loglik_ : log-likelihood of the linear system (when it is one), else None
    nobs_ : number of rows used (joint fit) or the row indices of every equation (dict)
    information_ : how the standard errors were computed: ``"expected"``,
        ``"observed"``, ``"observed (per equation)"`` (the model is not one linear
        system on shared rows; see *Inference*), ``"lm"`` or ``"none"``
    """

    def __init__(self, model: str = "y ~ x", estimator=None, estimation: str = "auto", se: str = "expected",
                 level: float = 0.95, predict_mode: str = "propagate", gam_method: str = "REML",
                 contrasts: Optional[dict] = None, na_action: str = "na.omit", env: Optional[dict] = None):
        self.model = model
        self.estimator = estimator
        self.estimation = estimation
        self.se = se
        self.level = level
        self.predict_mode = predict_mode
        self.gam_method = gam_method
        self.contrasts = contrasts
        self.na_action = na_action
        self.env = env

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.target_tags.required = False
        tags.target_tags.multi_output = True
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        return tags

    def _estimator_for(self, eq):
        from sklearn.linear_model import LinearRegression
        from ._mgcv import has_smooths
        env = {} if self.env is None else self.env
        est = self.estimator.get(eq.response) if isinstance(self.estimator, dict) else self.estimator
        if est is None and has_smooths(coerce_formula(eq.formula, env)):
            return GAMRegressor(eq.formula, method=self.gam_method, na_action=self.na_action, env=self.env)
        if est is None:
            est = LinearRegression(fit_intercept=False)
        return FormulaEstimator(eq.formula, clone(est), y_encoding="auto", contrasts=self.contrasts,
                                na_action=self.na_action, env=self.env)

    # -- fitting -------------------------------------------------------------
    def fit(self, X, y=None, **fit_params):
        from ._parser import all_vars, parse
        from ._path import parse_path_system, path_order
        if self.predict_mode not in ("propagate", "direct"):
            raise ValueError("predict_mode must be 'propagate' or 'direct'")
        if self.estimation not in ("auto", "ML", "OLS"):
            raise ValueError("estimation must be 'auto', 'ML' or 'OLS'")
        if self.se not in ("expected", "observed", "lm", "none"):
            raise ValueError("se must be 'expected', 'observed', 'lm' or 'none'")
        data = _as_frame(X).reset_index(drop=True)
        self._fit_data = data
        eqs, covs = parse_path_system(self.model)
        self.equations_ = eqs
        self.covariances_ = covs
        self.estimators_ = {}
        for eq in eqs:
            self.estimators_[eq.response] = self._estimator_for(eq).fit(data, **fit_params)
        self.endogenous_ = [eq.response for eq in eqs]
        lhs_vars = {v for eq in eqs for v in all_vars(parse(eq.lhs))}
        seen = []
        for eq in eqs:
            for v in eq.predictors:
                if v not in lhs_vars and v not in seen:
                    seen.append(v)
        self.exogenous_ = seen
        order = path_order(eqs)
        self.order_ = None if order is None else [eqs[i].response for i in order]
        self.n_features_in_ = data.shape[1]
        self.feature_names_in_ = np.asarray([str(c) for c in data.columns], dtype=object)
        self._weighted = bool(fit_params.get("sample_weight") is not None)

        self.system_, why, rows = self._build_system(data)
        same_rows = self.system_ is not None and all(len(r) == len(rows[1]) for r in rows[0].values())
        needs_ml = bool(covs) or self.order_ is None
        if self.estimation == "ML" or (self.estimation == "auto" and needs_ml):
            if self.system_ is None:
                raise RError("this path model needs the joint maximum likelihood fit (%s), which needs linear "
                             "equations: %s" % ("residual covariances" if covs else "feedback loops"
                                                if self.order_ is None else "estimation='ML'", why))
            theta, ll, conv, _ = self.system_.fit()
            if not conv:
                warnings.warn("the maximum likelihood fit of the path model did not converge", RuntimeWarning)
            self.theta_ = theta
            self.loglik_ = float(ll)
            self.estimation_ = "ML"
            self.converged_ = bool(conv)
            self._set_coefficients(theta)
            self.nobs_ = self.system_.n
        else:
            if covs:
                raise RError("residual covariances (~~) need estimation='ML' or 'auto'")
            self.estimation_ = "OLS"
            self.converged_ = True
            self.nobs_ = rows[0]
            if self.system_ is not None and same_rows:
                self.theta_ = self._theta_from_estimators()
                self.loglik_ = float(self.system_.loglik(self.theta_, grad=False))
            else:
                self.theta_ = None
                self.loglik_ = None
        self._inference()
        del self._fit_data
        return self

    def _ols_equation(self, eq):
        from sklearn.linear_model import LinearRegression
        est = self.estimators_[eq.response]
        return (isinstance(est, FormulaEstimator) and type(est.estimator_) is LinearRegression
                and not est.estimator_.fit_intercept)

    def _equation_rows(self, eq, data):
        env = {} if self.env is None else self.env
        Ye, Xe = design_matrices(eq.formula, data, contrasts=self.contrasts, na_action=self.na_action, env=env)
        return np.asarray(Xe.index)

    def _build_system(self, data):
        """The model as one linear system on the listwise rows: (system, reason, (rows, common))."""
        from ._parser import Sym, parse
        from ._pathfit import LinearSystem
        rows = {}
        why = None
        for eq in self.equations_:
            rows[eq.response] = self._equation_rows(eq, data)
            if why is None and not self._ols_equation(eq):
                why = "equation %s is not fitted by least squares" % eq.formula
            if why is None and not isinstance(parse(eq.lhs), Sym):
                why = "the response of %s is transformed" % eq.formula
        if why is None and self._weighted:
            why = "the fit is weighted"
        common = None
        for r in rows.values():
            common = r if common is None else np.intersect1d(common, r)
        if common is None:
            common = np.arange(len(data))
        if why is not None:
            return None, why, (rows, common)
        endo = self.endogenous_
        sub = data.iloc[common].reset_index(drop=True)
        try:
            Y = sub[endo].to_numpy(dtype=float)
        except (ValueError, TypeError):
            return None, "an endogenous variable is not numeric", (rows, common)
        cols, Xs = [], []
        for eq in self.equations_:
            est = self.estimators_[eq.response]
            Xi = est.transform(sub).to_numpy(dtype=float)
            names = list(est.design_columns_)
            for v in eq.predictors:
                if v not in endo:
                    continue
                pert = sub.copy()
                pert[v] = pert[v].astype(float) + 1.0
                dX = est.transform(pert).to_numpy(dtype=float) - Xi
                expect = np.zeros_like(dX)
                if v in names:
                    expect[:, names.index(v)] = 1.0
                if v not in names or not np.allclose(dX, expect, atol=1e-8):
                    return None, "%s does not enter %s as a plain column" % (v, eq.formula), (rows, common)
            cols.append(names)
            Xs.append(Xi)
        return LinearSystem(endo, Y, cols, Xs, self.covariances_), None, (rows, common)

    def _theta_from_estimators(self):
        s = self.system_
        theta = np.empty(s.npar)
        for i, eq in enumerate(self.equations_):
            sel = np.where((s.pi == i) & (s.kind != "P"))[0]
            theta[sel] = np.ravel(self.estimators_[eq.response].estimator_.coef_)
        B, G, _ = s.unpack(np.where(s.kind == "P", 0.0, theta))
        Z = s.Y @ (np.eye(s.k) - B).T - s.X @ G.T
        P = np.diag(np.mean(Z ** 2, axis=0))
        sel = s.kind == "P"
        theta[sel] = P[s.pi[sel], s.pj[sel]]
        return theta

    def _set_coefficients(self, theta):
        s = self.system_
        for i, eq in enumerate(self.equations_):
            sel = np.where((s.pi == i) & (s.kind != "P"))[0]
            inner = self.estimators_[eq.response].estimator_
            inner.coef_ = np.asarray(theta[sel], dtype=float).reshape(np.shape(inner.coef_))

    # -- inference -------------------------------------------------------------
    def _inference(self):
        """The parameter table (lhs, op, rhs, est), its covariance matrix and test kind."""
        rows, blocks, dfs = [], [], []
        self._test_ = "t" if self.se == "lm" else "z"
        if self.se == "lm" and self.estimation_ == "ML":
            raise RError("se='lm' is for equation-by-equation estimation; use 'expected' or 'observed'")
        if self.theta_ is not None and self.se in ("expected", "observed"):
            s = self.system_
            V = s.vcov(self.theta_, self.se)
            self._params_ = pd.DataFrame(s.names(), columns=["lhs", "op", "rhs"])
            self._params_["est"] = self.theta_
            self._vcov_ = V
            self._df_ = np.full(s.npar, np.inf)
            self.information_ = self.se
            return
        self.information_ = {"lm": "lm", "none": "none"}.get(self.se, "observed (per equation)")
        # equation by equation
        psi = []
        for eq in self.equations_:
            est = self.estimators_[eq.response]
            names, coef, V, df, sig2 = self._equation_inference(eq, est)
            for c in names:
                rows.append((eq.response, "~1", "") if c == "(Intercept)" else (eq.response, "~", c))
            blocks.append((coef, V))
            dfs.extend([df] * len(names))
            psi.append(sig2)
        for eq, (sig2, n) in zip(self.equations_, psi):
            rows.append((eq.response, "~~", eq.response))
            se2 = 2.0 * sig2 ** 2 / n if (n and self.se in ("expected", "observed")) else np.nan
            blocks.append((np.array([sig2]), np.array([[se2]])))
            dfs.append(np.inf)
        est = np.concatenate([b[0] for b in blocks]) if blocks else np.zeros(0)
        m = len(est)
        V = np.zeros((m, m))
        o = 0
        for b, Vb in blocks:
            V[o:o + len(b), o:o + len(b)] = Vb
            o += len(b)
        self._params_ = pd.DataFrame(rows, columns=["lhs", "op", "rhs"])
        self._params_["est"] = est
        self._vcov_ = V if self.se != "none" else np.full((m, m), np.nan)
        self._df_ = np.asarray(dfs, dtype=float)

    def _equation_inference(self, eq, est):
        """names, coef, covariance, df, (sigma2_ML, n) of one equation fitted on its own."""
        if isinstance(est, GAMRegressor):
            names = list(est.design_columns_)
            ft = est.fits_[0]
            return names, np.asarray(ft.coef, dtype=float), ft.Vp, np.inf, (float(ft.scale), 0)
        inner = est.estimator_
        names = list(est.design_columns_)
        coef = np.ravel(getattr(inner, "coef_", np.full(len(names), np.nan))).astype(float)
        if len(coef) != len(names):
            coef = np.full(len(names), np.nan)
        if not self._ols_equation(eq) or self._weighted:
            return names, coef, np.full((len(names), len(names)), np.nan), np.inf, (np.nan, 0)
        env = {} if self.env is None else self.env
        data = self._fit_data
        Ye, Xe = design_matrices(eq.formula, data, contrasts=self.contrasts, na_action=self.na_action, env=env)
        Xm = Xe.to_numpy(dtype=float)
        yv = Ye.iloc[:, 0].to_numpy(dtype=float)
        n, p = Xm.shape
        rss = float(np.sum((yv - Xm @ coef) ** 2))
        XtXi = np.linalg.pinv(Xm.T @ Xm)
        if self.se == "lm":
            return names, coef, rss / (n - p) * XtXi, float(n - p), (rss / n, n)
        return names, coef, rss / n * XtXi, np.inf, (rss / n, n)

    def _table(self, est, se, df):
        from scipy import stats
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = est / se
        stat = np.where(se > 0, stat, np.nan)
        dist_sf = np.where(np.isinf(df), stats.norm.sf(np.abs(stat)), stats.t.sf(np.abs(stat), np.where(np.isinf(df), 1, df)))
        a = (1 - self.level) / 2
        q = np.where(np.isinf(df), stats.norm.ppf(1 - a), stats.t.ppf(1 - a, np.where(np.isinf(df), 1, df)))
        return {"se": se, self._test_: stat, "pvalue": 2 * dist_sf, "ci.lower": est - q * se, "ci.upper": est + q * se}

    def coefficients(self) -> pd.DataFrame:
        """Parameter estimates like ``lavaan::parameterEstimates()``: ``lhs``, ``op``
        (``"~"`` regression, ``"~1"`` intercept, ``"~~"`` residual (co)variance),
        ``rhs``, ``est``, ``se``, ``z`` (``t`` with ``se="lm"``), ``pvalue``,
        ``ci.lower``, ``ci.upper``."""
        check_is_fitted(self, "estimators_")
        out = self._params_.copy()
        se = np.sqrt(np.maximum(np.diag(self._vcov_), 0.0)) if len(out) else np.zeros(0)
        se = np.where(np.isnan(np.diag(self._vcov_)), np.nan, se) if len(out) else se
        for k, v in self._table(out["est"].to_numpy(dtype=float), se, self._df_).items():
            out[k] = v
        return out.iloc[self._lavaan_order()].reset_index(drop=True)

    def _lavaan_order(self):
        rank = self._params_["op"].map({"~": 0, "~~": 1, "~1": 2}).to_numpy()
        return np.argsort(rank, kind="stable")

    def vcov(self) -> pd.DataFrame:
        """Covariance matrix of the estimates (rows/columns as in :meth:`coefficients`)."""
        check_is_fitted(self, "estimators_")
        o = self._lavaan_order()
        P = self._params_.iloc[o]
        lab = ["%s%s%s" % (a, b, c) for a, b, c in P[["lhs", "op", "rhs"]].itertuples(index=False)]
        return pd.DataFrame(self._vcov_[np.ix_(o, o)], index=lab, columns=lab)

    # -- prediction ---------------------------------------------------------------
    def predict(self, X) -> pd.DataFrame:
        """Predictions of every endogenous variable (one column per equation)."""
        from ._parser import Sym, all_vars, parse
        check_is_fitted(self, "estimators_")
        data = _as_frame(X).reset_index(drop=True)
        out = {}
        if self.predict_mode == "direct":
            for eq in self.equations_:
                out[eq.response] = np.asarray(self.estimators_[eq.response].predict(data)).ravel()
        elif self.order_ is None:
            if self.system_ is None:
                raise RError("the path model is non-recursive (it has feedback loops) and not linear; "
                             "use predict_mode='direct'")
            # reduced form: y = (I - B)^-1 (G x)
            zero = data.copy()
            for v in self.endogenous_:
                zero[v] = 0.0
            C = np.column_stack([np.asarray(self.estimators_[r].predict(zero), dtype=float).ravel()
                                 for r in self.endogenous_])
            B = self._linear_B()
            Yp = np.linalg.solve(np.eye(len(self.endogenous_)) - B, C.T).T
            out = {r: Yp[:, i] for i, r in enumerate(self.endogenous_)}
        else:
            data = data.copy()
            plain = {eq.response: isinstance(parse(eq.lhs), Sym) for eq in self.equations_}
            transformed = {v: eq.lhs for eq in self.equations_ if not plain[eq.response]
                           for v in all_vars(parse(eq.lhs))}
            by_resp = {eq.response: eq for eq in self.equations_}
            for resp in self.order_:
                eq = by_resp[resp]
                bad = [v for v in eq.predictors if v in transformed]
                if bad:
                    raise RError("cannot propagate through the transformed response %s (used by %s); "
                                 "use a plain response or predict_mode='direct'" % (transformed[bad[0]], eq.formula))
                pred = np.asarray(self.estimators_[resp].predict(data)).ravel()
                out[resp] = pred
                if plain[resp]:
                    data[resp] = pred
        return pd.DataFrame({eq.response: out[eq.response] for eq in self.equations_})

    def score(self, X, y=None, sample_weight=None):
        """Mean R^2 over the equations (predictions follow ``predict_mode``)."""
        from sklearn.metrics import r2_score
        data = _as_frame(X).reset_index(drop=True)
        P = self.predict(data)
        scores = []
        env = {} if self.env is None else self.env
        for eq in self.equations_:
            f = coerce_formula(eq.formula, env)
            mf = _response_model_frame(f, data, "na.pass", env)
            yv = response_matrix(mf, f, "numeric").iloc[:, 0].to_numpy(dtype=float)
            pv = P[eq.response].to_numpy(dtype=float)
            ok = ~(np.isnan(yv) | np.isnan(pv))
            w = None if sample_weight is None else np.asarray(sample_weight)[ok]
            scores.append(r2_score(yv[ok], pv[ok], sample_weight=w))
        return float(np.mean(scores))

    # -- linear path coefficients and effects ----------------------------------
    def _linear_coefs(self):
        coefs = {}
        for eq in self.equations_:
            est = self.estimators_[eq.response]
            if isinstance(est, GAMRegressor):
                cols, coef, icpt = est.design_columns_, np.asarray(est.coef_), 0.0
            else:
                inner = est.estimator_
                if not hasattr(inner, "coef_"):
                    raise RError("effects need linear estimators (equation %s)" % eq.formula)
                cols, coef = est.design_columns_, np.ravel(inner.coef_)
                icpt = float(np.sum(np.ravel(getattr(inner, "intercept_", 0.0))))
            d = {}
            for c, b in zip(cols, coef):
                d[c] = d.get(c, 0.0) + float(b)
            if icpt:
                d["(Intercept)"] = d.get("(Intercept)", 0.0) + icpt
            coefs[eq.response] = d
        return coefs

    def _linear_B(self):
        endo = self.endogenous_
        B = np.zeros((len(endo), len(endo)))
        for i, (resp, d) in enumerate(self._linear_coefs().items()):
            for c, b in d.items():
                if c in endo:
                    B[i, endo.index(c)] += b
        return B

    def _effect_columns(self):
        endo = self.endogenous_
        return list(endo) + [v for v in self.exogenous_ if v not in endo]

    def effects(self, kind: str = "total", se: bool = False) -> pd.DataFrame:
        """Direct, indirect or total effects.

        Needs linear equations whose design columns are the variables themselves
        (``y ~ m + x``); then with ``B`` the direct effects, total effects are
        ``(I - B_yy)^-1 B`` and indirect = total - direct, as in lavaan.  Returns
        a matrix (rows: endogenous, columns: all variables), or with ``se=True``
        a long table (``lhs`` effect on, ``rhs`` cause, ``est``, ``se``, ``z``,
        ``pvalue``, ``ci.lower``, ``ci.upper``) of the effects the model allows,
        with delta-method standard errors (lavaan's ``:=``)."""
        check_is_fitted(self, "estimators_")
        if kind not in ("total", "direct", "indirect"):
            raise ValueError("kind must be 'total', 'direct' or 'indirect'")
        coefs = self._linear_coefs()
        endo = self.endogenous_
        cols = self._effect_columns()
        ne = len(endo)
        D = np.zeros((ne, len(cols)))
        mask = np.zeros_like(D, dtype=bool)
        for i, resp in enumerate(endo):
            for c, b in coefs[resp].items():
                if c == "(Intercept)":
                    continue
                if c not in cols:
                    raise RError("effects() needs every design column to be a variable of the model; "
                                 "%r (equation for %s) is not" % (c, resp))
                D[i, cols.index(c)] = b
                mask[i, cols.index(c)] = True
        Ainv = np.linalg.inv(np.eye(ne) - D[:, :ne])
        T = Ainv @ D
        M = {"direct": D, "total": T, "indirect": T - D}[kind]
        if not se:
            return pd.DataFrame(M, index=endo, columns=cols)
        # structure: which effects exist
        Bm = mask[:, :ne].astype(int)
        reach = Bm.copy()  # reach[r, i]: a path i -> ... -> r of length >= 1
        for _ in range(ne):
            reach = ((reach + reach @ Bm) > 0).astype(int)
        indirect_mask = (reach @ mask.astype(int)) > 0
        smask = {"direct": mask, "indirect": indirect_mask, "total": mask | indirect_mask}[kind]
        # delta method over the regression coefficients (rows "~" of the parameter table)
        P = self._params_
        reg = np.where(P["op"].to_numpy() == "~")[0]
        Ext = np.eye(len(cols))
        Ext[:ne] += T
        J = np.zeros((ne, len(cols), len(reg)))
        for q, pidx in enumerate(reg):
            i, k = endo.index(P["lhs"].iloc[pidx]), cols.index(P["rhs"].iloc[pidx])
            dT = np.outer(Ainv[:, i], Ext[k])
            if kind == "direct":
                dT = np.zeros_like(dT)
                dT[i, k] = 1.0
            elif kind == "indirect":
                dT[i, k] -= 1.0
            J[:, :, q] = dT
        V = self._vcov_[np.ix_(reg, reg)]
        r_idx, c_idx = np.where(smask)
        Jr = J[r_idx, c_idx]
        var = np.einsum("ij,jk,ik->i", Jr, V, Jr)
        est = M[r_idx, c_idx]
        sev = np.sqrt(np.maximum(var, 0.0))
        out = pd.DataFrame({"lhs": [endo[r] for r in r_idx], "rhs": [cols[c] for c in c_idx], "est": est})
        test, self._test_ = self._test_, "z"
        try:
            for k, v in self._table(est, sev, np.full(len(est), np.inf)).items():
                out[k] = v
        finally:
            self._test_ = test
        return out
