"""Public functions mirroring R's modelling API."""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._bases import ContrastMatrix, contrast_from_spec
from ._formula import (Formula, RFormula, _resolve_env, capture_env, coerce_formula,
                       delete_response as _delete_response, drop_terms as _drop_terms,
                       reformulate as _reformulate, update_formula)
from ._model import (ModelFrame, _data_names, build_model_matrix, model_extract as _model_extract,
                     model_frame_from_terms, model_offset as _model_offset,
                     model_response as _model_response)
from ._parser import deparse as _deparse, parse
from ._rtypes import Factor, Logical, RError
from ._terms import Terms

FormulaLike = Union[str, RFormula, Formula, Terms]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class ModelMatrix(pd.DataFrame):
    """A design matrix (``pandas.DataFrame``) with R's attributes.

    * ``assign``: for each column, the 0-based term index (0 = intercept)
    * ``contrasts``: contrasts used for each factor variable
    * ``model_spec``: a :class:`ModelSpec` to build the same design on new data
    """

    _metadata = ["assign", "contrasts", "model_spec"]

    @property
    def _constructor(self):
        return ModelMatrix

    def __finalize__(self, other, method=None, **kwargs):
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self


class SparseModelMatrix:
    """Sparse design matrix (``scipy.sparse.csc_matrix``), like ``Matrix::sparse.model.matrix``."""

    def __init__(self, matrix, columns, index, assign, contrasts, model_spec=None):
        self.matrix = matrix
        self.columns = list(columns)
        self.index = index
        self.assign = list(assign)
        self.contrasts = contrasts
        self.model_spec = model_spec

    @property
    def shape(self):
        return self.matrix.shape

    def toarray(self) -> np.ndarray:
        return self.matrix.toarray()

    def to_pandas(self, sparse: bool = True) -> pd.DataFrame:
        if sparse:
            df = pd.DataFrame.sparse.from_spmatrix(self.matrix, index=self.index, columns=self.columns)
        else:
            df = pd.DataFrame(self.matrix.toarray(), index=self.index, columns=self.columns)
        return df

    def __repr__(self):
        return "<SparseModelMatrix %dx%d, %d stored values; columns=%r>" % (
            self.shape[0], self.shape[1], self.matrix.nnz, self.columns)


# ---------------------------------------------------------------------------
# Model specification (stateful transforms for new data)
# ---------------------------------------------------------------------------

class ModelSpec:
    """Everything needed to rebuild a design matrix on new data.

    It plays the role of R's ``terms`` object with ``predvars``, ``xlevels``
    and ``contrasts`` (what ``predict()`` uses), or formulaic's ``ModelSpec``.
    """

    def __init__(self, terms: Terms, state: dict, xlevels: Dict[str, List[str]],
                 contrasts: Dict[str, Any], na_action: str, sparse: bool, env):
        self.terms = terms
        self.state = state
        self.xlevels = xlevels
        self.contrasts = contrasts
        self.na_action = na_action
        self.sparse = sparse
        self.env = env

    @property
    def column_names(self):
        return None

    def get_model_frame(self, data, na_action: Optional[str] = None) -> ModelFrame:
        t = _delete_response(self.terms)
        return model_frame_from_terms(t, data, env=self.env, na_action=na_action or self.na_action,
                                      xlev=self.xlevels, state=copy.deepcopy(self.state), fitting=False)

    def get_model_matrix(self, data, na_action: Optional[str] = None, output: str = "pandas"):
        mf = self.get_model_frame(data, na_action)
        t = mf.terms
        X, cols, assign, contr = build_model_matrix(t, mf, sparse=self.sparse,
                                                    contrasts_override=self.contrasts)
        return _wrap_matrix(X, cols, mf.index, assign, contr, self, self.sparse, output)

    def __repr__(self):
        return "<ModelSpec: %s>" % (self.terms.formula,)


def _resolved_contrasts(mf: ModelFrame, contrasts_out) -> Dict[str, ContrastMatrix]:
    res = {}
    for nm, spec in contrasts_out.items():
        v = mf.variables.get(nm)
        if isinstance(v, Logical):
            levels = ["FALSE", "TRUE"]
            ordered = False
        elif isinstance(v, Factor):
            levels = v.levels
            ordered = v.ordered
        else:
            from ._rtypes import as_factor
            f = as_factor(v)
            levels, ordered = f.levels, f.ordered
        if spec is None:
            from ._eval import OPTIONS
            spec = OPTIONS["contrasts"][1 if ordered else 0]
        if isinstance(spec, ContrastMatrix):
            res[nm] = spec
        else:
            try:
                res[nm] = contrast_from_spec(spec, levels)
            except RError:
                res[nm] = spec
    return res


def _wrap_matrix(X, cols, index, assign, contrasts, spec, sparse, output):
    if sparse:
        return SparseModelMatrix(X, cols, index, assign, contrasts, spec)
    if output == "numpy":
        return X
    mm = ModelMatrix(X, index=index, columns=pd.Index(cols, dtype=object))
    object.__setattr__(mm, "assign", list(assign))
    object.__setattr__(mm, "contrasts", contrasts)
    object.__setattr__(mm, "model_spec", spec)
    return mm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_env(env, formula, depth):
    if env is not None:
        return env if not isinstance(env, int) else _resolve_env(env, depth + 1)
    fenv = getattr(formula, "env", None)
    caller = capture_env(depth + 1)
    if fenv:
        from collections import ChainMap
        return ChainMap(fenv, caller) if not isinstance(fenv, ChainMap) else ChainMap(*fenv.maps, *caller.maps)
    return caller


def formula(x, env=None) -> RFormula:
    """Create a base R formula (R's ``formula()`` / ``as.formula()``)."""
    if isinstance(x, RFormula):
        return x
    if isinstance(x, Formula):
        return x.formula()
    return RFormula(x, env if env is not None else capture_env(1))


as_formula = formula


def Formula_(x, *more, env=None) -> Formula:
    return Formula(x, *more, env=env if env is not None else capture_env(1))


def as_Formula(x, *more, env=None) -> Formula:
    """R's ``as.Formula(...)``: combine formulas into one multi-part Formula."""
    return Formula(x, *more, env=env if env is not None else capture_env(1))


def terms(x: FormulaLike, data=None, keep_order: bool = False, specials=None, simplify: bool = False,
          allow_dot_as_name: bool = False, lhs=None, rhs=None, dot: str = "separate") -> Terms:
    """R's ``terms()`` for formulas and Formulas."""
    if isinstance(x, Terms):
        return x
    if isinstance(x, ModelFrame):
        return x.terms
    f = coerce_formula(x)
    if isinstance(f, Formula):
        return f.terms(lhs=lhs, rhs=rhs, data=data, dot=dot, keep_order=keep_order, specials=specials)
    return f.terms(data=data, keep_order=keep_order, specials=specials, simplify=simplify,
                   allow_dot_as_name=allow_dot_as_name)


def model_frame(formula: FormulaLike, data=None, subset=None, na_action: Optional[str] = None,
                drop_unused_levels: bool = False, xlev=None, weights=None, offset=None,
                lhs=None, rhs=None, dot: str = "separate", env=None, **extra) -> ModelFrame:
    """R's ``model.frame()``.

    ``subset``, ``weights`` and ``offset`` may be expressions (strings
    evaluated in ``data``) or arrays; extra keyword arguments become
    ``(name)`` columns, retrievable with :func:`model_extract`.
    """
    env = _get_env(env, formula, 1)
    if isinstance(formula, ModelFrame):
        return formula
    t = terms(formula, data=data, lhs=lhs, rhs=rhs, dot=dot) if not isinstance(formula, Terms) else formula
    return model_frame_from_terms(t, data, env=env, subset=subset, na_action=na_action,
                                  drop_unused_levels=drop_unused_levels, xlev=xlev, weights=weights,
                                  offset=offset, extra_args=extra or None)


def _model_matrix_impl(x, data, contrasts, xlev, lhs, rhs, sparse, output, na_action,
                       drop_unused_levels, env, dot):
    if isinstance(x, (str, RFormula)) and not isinstance(data, ModelFrame):
        from ._mgcv import gam_model_matrix, has_smooths
        try:
            smooth = has_smooths(x)
        except Exception:
            smooth = False
        if smooth:
            if sparse:
                raise RError("sparse model matrices are not available for mgcv smooths")
            return gam_model_matrix(x, data, na_action=na_action, env=env, output=output)
    if isinstance(x, ModelFrame) and data is None:
        mf = x
        t = mf.terms
    else:
        f = x
        if not isinstance(f, (Terms, ModelFrame)):
            f = coerce_formula(f)
        if isinstance(f, Formula):
            if dot == "separate":
                form = f.formula(lhs=lhs, rhs=rhs, collapse=(False, True))
                t = _delete_response(form.terms(data=data))
            else:
                t = _delete_response(f.terms(lhs=lhs, rhs=rhs, data=data, dot=dot))
        elif isinstance(f, Terms):
            t = f
        elif isinstance(f, ModelFrame):
            t = f.terms
        else:
            t = f.terms(data=data if not isinstance(data, ModelFrame) else None)
        if isinstance(data, ModelFrame):
            missing = [v for v in t.variables if v not in data.variables]
            if missing:
                raise RError("model frame and formula mismatch in model.matrix()")
            vars_ = OrderedDict((v, data.variables[v]) for v in t.variables)
            mf = ModelFrame(vars_, t, data.index, data.na_action, data.extras, data.state)
        else:
            mf = model_frame_from_terms(t, data, env=env, na_action=na_action, xlev=xlev,
                                        drop_unused_levels=drop_unused_levels)
            t = mf.terms
    X, cols, assign, contr = build_model_matrix(t, mf, contrasts_arg=contrasts, sparse=sparse)
    spec = ModelSpec(t, copy.deepcopy(mf.state), mf.xlevels(), _resolved_contrasts(mf, contr),
                     na_action or "na.omit", sparse, env)
    return _wrap_matrix(X, cols, mf.index, assign, contr, spec, sparse, output)


def model_matrix(formula: FormulaLike, data=None, contrasts: Optional[Mapping[str, Any]] = None,
                 xlev=None, lhs=None, rhs=1, output: str = "pandas", na_action: Optional[str] = None,
                 drop_unused_levels: bool = False, dot: str = "separate", env=None):
    """R's ``model.matrix()``.

    ``formula`` may be a string, :class:`RFormula`, :class:`Formula`,
    :class:`Terms` or :class:`ModelFrame`.  ``contrasts`` maps variable names
    to a contrast name (``"contr.sum"``), a matrix or a callable, as R's
    ``contrasts.arg``.  For multi-part formulas, ``rhs`` selects the part
    (1-based, as in the Formula package).
    """
    env = _get_env(env, formula, 1)
    return _model_matrix_impl(formula, data, contrasts, xlev, lhs, rhs, False, output, na_action,
                              drop_unused_levels, env, dot)


def sparse_model_matrix(formula: FormulaLike, data=None, contrasts: Optional[Mapping[str, Any]] = None,
                        xlev=None, lhs=None, rhs=1, na_action: Optional[str] = None,
                        drop_unused_levels: bool = False, dot: str = "separate", env=None) -> SparseModelMatrix:
    """``Matrix::sparse.model.matrix()``: same columns as :func:`model_matrix`, stored sparsely."""
    env = _get_env(env, formula, 1)
    return _model_matrix_impl(formula, data, contrasts, xlev, lhs, rhs, True, "pandas", na_action,
                              drop_unused_levels, env, dot)


def model_response(mf: ModelFrame, type: str = "any"):
    """R's ``model.response``."""
    return _model_response(mf, type)


def model_offset(mf: ModelFrame):
    """R's ``model.offset``."""
    return _model_offset(mf)


def model_weights(mf: ModelFrame):
    """R's ``model.weights``."""
    return _model_extract(mf, "weights")


def model_extract(mf: ModelFrame, component: str):
    """R's ``model.extract(frame, component)`` (``"response"``, ``"offset"``, ``"weights"``, ...)."""
    return _model_extract(mf, component)


def model_part(formula: Formula, data: ModelFrame, lhs=0, rhs=0, drop: bool = False, dot=None):
    """``Formula::model.part``: the columns of a model frame belonging to some parts."""
    F = formula if isinstance(formula, Formula) else Formula(formula)
    if lhs is None:
        lhs = list(range(1, len(F.lhs_parts) + 1))
    if rhs is None:
        rhs = list(range(1, len(F.rhs_parts) + 1))
    lz = all(i == 0 for i in (lhs if isinstance(lhs, (list, tuple)) else [lhs]))
    rz = all(i == 0 for i in (rhs if isinstance(rhs, (list, tuple)) else [rhs]))
    if lz and rz:
        raise RError("Either some 'lhs' or 'rhs' has to be selected.")
    if not isinstance(data, ModelFrame):
        raise RError("'data' does not seem to be an appropriate 'model.frame'")
    mt = F.terms(lhs=lhs, rhs=rhs)
    ix = mt.variables
    missing = [v for v in ix if v not in data]
    if missing:
        raise RError("'data' does not seem to be an appropriate 'model.frame': %s not found" %
                     ", ".join("'%s'" % m for m in missing))
    df = data.to_pandas()
    cols = []
    for v in ix:
        val = data[v]
        from ._rtypes import RMatrix
        if isinstance(val, RMatrix):
            cn = val.colnames or [str(j + 1) for j in range(val.ncol)]
            cols.extend("%s.%s" % (v, c) for c in cn)
        else:
            cols.append(v)
    out = df[cols]
    if drop and out.shape[1] == 1:
        return out.iloc[:, 0]
    return out


def update(old, new):
    """R's ``update()`` for formulas (``update.formula``) and Formulas (``update.Formula``)."""
    if isinstance(old, Formula) or isinstance(new, Formula):
        o = old if isinstance(old, Formula) else Formula(old)
        return o.update(new)
    o = coerce_formula(old)
    if isinstance(o, Formula):
        return o.update(new)
    n = coerce_formula(new)
    if isinstance(n, Formula):
        return Formula(o).update(n)
    return update_formula(o, n)


def all_vars(x, functions: bool = False) -> List[str]:
    """R's ``all.vars``."""
    if isinstance(x, Terms):
        x = x.formula
    f = coerce_formula(x) if isinstance(x, str) else x
    if isinstance(f, Formula):
        return f.all_vars()
    return f.all_vars()


def all_names(x, functions: bool = True, unique: bool = False) -> List[str]:
    """R's ``all.names``."""
    from ._parser import all_names as an
    if isinstance(x, (RFormula,)):
        return an(x.node, functions, unique)
    if isinstance(x, Formula):
        return an(x.formula().node, functions, unique)
    return an(parse(x), functions, unique)


def reformulate(termlabels: Sequence[str], response=None, intercept: bool = True) -> RFormula:
    """R's ``reformulate``."""
    return _reformulate(termlabels, response, intercept, env=capture_env(1))


def drop_terms(termobj: Terms, dropx=None, keep_response: bool = False) -> Terms:
    """R's ``drop.terms`` (1-based ``dropx``)."""
    return _drop_terms(termobj, dropx, keep_response)


def delete_response(termobj: Terms) -> Terms:
    """R's ``delete.response``."""
    return _delete_response(termobj)


def get_all_vars(formula: FormulaLike, data) -> pd.DataFrame:
    """R's ``get_all_vars``: the raw data columns used by a formula."""
    vars_ = all_vars(formula)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    cols = [v for v in vars_ if v in data.columns]
    missing = [v for v in vars_ if v not in data.columns and v != "."]
    if missing:
        raise RError("object '%s' not found" % missing[0])
    if "." in vars_:
        cols = list(data.columns)
    return data[cols]


def deparse(x) -> str:
    """R's ``deparse`` for formulas and expressions."""
    if isinstance(x, (RFormula, Formula)):
        return str(x)
    if isinstance(x, str):
        return _deparse(parse(x))
    return _deparse(x)


# ---------------------------------------------------------------------------
# statsmodels / patsy style helpers
# ---------------------------------------------------------------------------

def dmatrix(formula: FormulaLike, data=None, return_type: str = "dataframe", contrasts=None,
            na_action: Optional[str] = None, env=None, rhs=1):
    """Design matrix for the right hand side (patsy/statsmodels style)."""
    env = _get_env(env, formula, 1)
    X = _model_matrix_impl(formula, data, contrasts, None, None, rhs, False, "pandas", na_action, False, env,
                           "separate")
    return X if return_type == "dataframe" else X.to_numpy()


def dmatrices(formula: FormulaLike, data=None, return_type: str = "dataframe", contrasts=None,
              na_action: Optional[str] = None, env=None, rhs=1):
    """``(y, X)`` as used by ``statsmodels`` (rows with NA in either are removed together)."""
    env = _get_env(env, formula, 1)
    f = coerce_formula(formula)
    if isinstance(f, Formula):
        mf = model_frame(f, data, na_action=na_action, env=env)
        X = _model_matrix_impl(f, mf, contrasts, None, None, rhs, False, "pandas", na_action, False, env,
                               "separate")
        y = model_part(f, mf, lhs=1)
    else:
        from ._mgcv import has_smooths
        if has_smooths(f):
            X = _model_matrix_impl(f, data, contrasts, None, None, rhs, False, "pandas", na_action, False, env,
                                   "separate")
            mf = model_frame(f.terms(data=data) if False else _gam_fake_terms(f, data), data, na_action=na_action, env=env)
            y = model_response(mf).to_frame() if mf.terms.response else None
            if y is None:
                raise RError("model is missing a response (left hand side)")
            y = y.loc[X.index]
            if return_type == "dataframe":
                return y, X
            return y.to_numpy(), X.to_numpy()
        mf = model_frame(f, data, na_action=na_action, env=env)
        if not mf.terms.response:
            raise RError("model is missing a response (left hand side)")
        X = _model_matrix_impl(mf, None, contrasts, None, None, rhs, False, "pandas", na_action, False, env,
                               "separate")
        y = model_response(mf)
        if isinstance(y, pd.Series):
            y = y.to_frame()
    if return_type == "dataframe":
        return y, X
    return y.to_numpy(), X.to_numpy()


def _gam_fake_terms(f, data):
    from ._mgcv import interpret_gam
    return interpret_gam(f).fake_formula.terms(data=data)
