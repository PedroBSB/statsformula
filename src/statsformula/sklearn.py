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

    def predict(self, X):
        return self._design(X) @ np.asarray(self.coef_).T

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
    """A system of regressions written as a multi-line formula, fitted equation by equation.

    >>> model = '''
    ...   m ~ x + z            # mediator
    ...   y ~ m + x + log(w)   # outcome
    ... '''
    >>> pm = PathModel(model).fit(df)                        # doctest: +SKIP
    >>> pm.predict(new)            # m and y; y uses the *predicted* m   # doctest: +SKIP
    >>> pm.effects("indirect")     # x -> m -> y, as lavaan's a*b        # doctest: +SKIP

    Each line is an R formula (see :mod:`statsformula._path` for the syntax:
    comments, continuation lines, ``y1 + y2 ~ x`` and ``cbind(y1, y2) ~ x``
    give one equation per response, repeated responses are merged).  Each
    equation is fitted on its own complete cases with a clone of
    ``estimator``.  With linear estimators this is the maximum likelihood fit of
    a recursive path model with uncorrelated disturbances (``lavaan::sem()``
    with observed variables).

    Parameters
    ----------
    model : str
        Multi-line model: one ``response ~ predictors`` formula per line.
    estimator : estimator or dict, optional
        Estimator cloned for every equation, or ``{response: estimator}``
        (responses missing from the dict use ``LinearRegression``).  Default:
        ``LinearRegression(fit_intercept=False)`` on the formula's design matrix
        (which has the intercept), or :class:`GAMRegressor` for equations with
        ``s()``/``te()``/``ti()`` smooths.
    predict_mode : {"propagate", "direct"}
        ``"propagate"`` (default): only exogenous variables are taken from the
        data; endogenous predictors are replaced by their predictions, upstream
        first (the reduced form; needs a recursive model).  ``"direct"``: every
        equation uses the observed values of its predictors.
    gam_method : {"REML", "GCV.Cp"}
        Smoothing-parameter method of the default GAM equations.
    contrasts, na_action, env : see :class:`FormulaEstimator`.

    Attributes
    ----------
    equations_ : list of :class:`PathEquation`
    estimators_ : dict response -> fitted per-equation estimator
    endogenous_, exogenous_ : variable names
    order_ : responses in causal (topological) order, or None if non-recursive
    """

    def __init__(self, model: str = "y ~ x", estimator=None, predict_mode: str = "propagate",
                 gam_method: str = "REML", contrasts: Optional[dict] = None, na_action: str = "na.omit",
                 env: Optional[dict] = None):
        self.model = model
        self.estimator = estimator
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

    def fit(self, X, y=None, **fit_params):
        from ._parser import all_vars, parse
        from ._path import parse_path_model, path_order
        if self.predict_mode not in ("propagate", "direct"):
            raise ValueError("predict_mode must be 'propagate' or 'direct'")
        data = _as_frame(X).reset_index(drop=True)
        eqs = parse_path_model(self.model)
        self.equations_ = eqs
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
        return self

    def predict(self, X) -> pd.DataFrame:
        """Predictions of every endogenous variable (one column per equation)."""
        from ._parser import Sym, all_vars, parse
        check_is_fitted(self, "estimators_")
        data = _as_frame(X).reset_index(drop=True)
        out = {}
        if self.predict_mode == "direct":
            for eq in self.equations_:
                out[eq.response] = np.asarray(self.estimators_[eq.response].predict(data)).ravel()
        else:
            if self.order_ is None:
                raise RError("the path model is non-recursive (it has feedback loops); "
                             "use predict_mode='direct'")
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

    def coefficients(self) -> pd.DataFrame:
        """Coefficients in long form: ``lhs``, ``op`` (``"~"`` or ``"~1"``), ``rhs``, ``est``
        (like the regression rows of ``lavaan::parameterEstimates()``)."""
        check_is_fitted(self, "estimators_")
        rows = []
        for resp, d in self._linear_coefs().items():
            for c, b in d.items():
                rows.append((resp, "~1", "", b) if c == "(Intercept)" else (resp, "~", c, b))
        return pd.DataFrame(rows, columns=["lhs", "op", "rhs", "est"])

    def effects(self, kind: str = "total") -> pd.DataFrame:
        """Direct, indirect or total effects (rows: endogenous, columns: all predictors).

        Needs linear equations whose design columns are the variables themselves
        (``y ~ m + x``); then with ``B`` the direct effects, total effects are
        ``(I - B_yy)^-1 B`` and indirect = total - direct, as in lavaan."""
        check_is_fitted(self, "estimators_")
        if kind not in ("total", "direct", "indirect"):
            raise ValueError("kind must be 'total', 'direct' or 'indirect'")
        coefs = self._linear_coefs()
        endo = self.endogenous_
        cols = list(endo) + [v for v in self.exogenous_ if v not in endo]
        D = np.zeros((len(endo), len(cols)))
        for i, resp in enumerate(endo):
            for c, b in coefs[resp].items():
                if c == "(Intercept)":
                    continue
                if c not in cols:
                    raise RError("effects() needs every design column to be a variable of the model; "
                                 "%r (equation for %s) is not" % (c, resp))
                D[i, cols.index(c)] = b
        if kind == "direct":
            M = D
        else:
            Byy = D[:, :len(endo)]
            T = np.linalg.solve(np.eye(len(endo)) - Byy, D)
            M = T if kind == "total" else T - D
        return pd.DataFrame(M, index=endo, columns=cols)
