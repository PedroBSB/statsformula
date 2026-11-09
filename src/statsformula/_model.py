"""``model.frame``, ``model.matrix``, ``sparse.model.matrix`` and friends.

``model_matrix`` is a port of R's C routine ``modelmatrix`` (stats/src/model.c):
column ordering, naming (``fb:gv``, ``poly(x, 2)1``, ``C(f, sum)1``),
the ``assign`` attribute, the contrasts heuristic (factor code 1 = contrasts,
2 = full dummy coding) and the no-intercept adjustment are all reproduced.
Column blocks are built with vectorised numpy (``einsum``) or scipy.sparse
row-wise Khatri-Rao products.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ._bases import ContrastMatrix, contrast_from_spec
from ._eval import OPTIONS, EvalContext, set_contrasts, _as_logical
from ._parser import Node, deparse, parse
from ._rtypes import (Character, Factor, Logical, RError, RMatrix, as_double, as_factor,
                      class_of, isna, rlen, take, to_r)
from ._terms import Terms


# ---------------------------------------------------------------------------
# Model frame
# ---------------------------------------------------------------------------

class ModelFrame:
    """Equivalent of an R model frame (a data.frame with a ``terms`` attribute).

    ``variables`` maps the deparsed variable names (``"log(x)"``) to R values;
    ``extras`` holds ``(weights)``, ``(offset)``, ``(subset)``-style columns.
    """

    def __init__(self, variables: "OrderedDict[str, Any]", terms: Terms, index, na_action=None,
                 extras: Optional["OrderedDict[str, Any]"] = None, state=None, nrow_original=None):
        self.variables = variables
        self.terms = terms
        self.index = index
        self.na_action = na_action
        self.extras = extras if extras is not None else OrderedDict()
        self.state = state if state is not None else {}
        self.nrow_original = nrow_original

    @property
    def names(self) -> List[str]:
        return list(self.variables) + list(self.extras)

    def __len__(self):
        return len(self.index)

    @property
    def shape(self):
        return (len(self.index), len(self.names))

    def __getitem__(self, name):
        if name in self.variables:
            return self.variables[name]
        return self.extras[name]

    def __contains__(self, name):
        return name in self.variables or name in self.extras

    def xlevels(self) -> Dict[str, List[str]]:
        """R's ``.getXlevels``: levels of factor/character predictors."""
        out = OrderedDict()
        names = list(self.variables)
        resp = self.terms.response
        for i, nm in enumerate(names):
            if resp and i == resp - 1:
                continue
            v = self.variables[nm]
            if isinstance(v, Factor):
                out[nm] = list(v.levels)
            elif isinstance(v, Character):
                out[nm] = list(as_factor(v).levels)
        return out

    def to_pandas(self):
        import pandas as pd
        cols = OrderedDict()
        for nm, v in list(self.variables.items()) + list(self.extras.items()):
            if isinstance(v, RMatrix):
                cn = v.colnames or [str(j + 1) for j in range(v.ncol)]
                for j in range(v.ncol):
                    cols["%s.%s" % (nm, cn[j])] = v.data[:, j]
            elif isinstance(v, Factor):
                cols[nm] = pd.Categorical.from_codes(v.codes, v.levels, ordered=v.ordered)
            elif isinstance(v, Logical):
                cols[nm] = pd.array([None if n else bool(b) for b, n in zip(v.values, v.na)], dtype="boolean") if v.na.any() else v.values
            elif isinstance(v, Character):
                cols[nm] = v.values
            else:
                cols[nm] = np.asarray(v)
        return pd.DataFrame(cols, index=self.index)

    def __repr__(self):
        try:
            return repr(self.to_pandas())
        except Exception:  # pragma: no cover
            return "ModelFrame(%r)" % self.names


def _data_names(data) -> Optional[List[str]]:
    if data is None:
        return None
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return [str(c) for c in data.columns]
    except Exception:  # pragma: no cover
        pass
    if isinstance(data, ModelFrame):
        return data.names
    if isinstance(data, Mapping):
        return [str(k) for k in data.keys()]
    raise TypeError("data must be a pandas DataFrame or a mapping of column name -> values")


def _data_mapping(data):
    if data is None:
        return {}, None
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            cols = {str(c): data[c] for c in data.columns}
            return cols, data.index
    except Exception:  # pragma: no cover
        pass
    if isinstance(data, ModelFrame):
        m = dict(data.variables)
        m.update(data.extras)
        return m, data.index
    n = None
    for v in data.values():
        try:
            n = len(v)
            break
        except TypeError:
            continue
    import pandas as pd
    return dict(data), pd.RangeIndex(n if n is not None else 0)


def _row_count(values, index):
    if index is not None:
        return len(index)
    for v in values:
        return rlen(v)
    return 0


def model_frame_from_terms(terms: Terms, data=None, env=None, subset=None, na_action=None,
                           drop_unused_levels: bool = False, xlev=None, weights=None, offset=None,
                           state=None, fitting=True, extra_args: Optional[Dict[str, Any]] = None) -> ModelFrame:
    import pandas as pd
    if isinstance(data, ModelFrame) and fitting and state is None:
        state = data.state
    mapping, index = _data_mapping(data)
    state = {} if state is None else state
    ctx = EvalContext(mapping, env or {}, state=state, fitting=fitting)
    variables = OrderedDict()
    for node, name in zip(terms.variable_nodes, terms.variables):
        if name in variables:
            continue
        val = ctx.eval(node)
        if val is None:
            raise RError("invalid type (NULL) for variable '%s'" % name)
        variables[name] = val
    extras = OrderedDict()
    for nm, val in (("(weights)", weights), ("(offset)", offset)):
        if val is None:
            continue
        if isinstance(val, str):
            val = ctx.eval(parse(val))
        else:
            val = to_r(val)
        extras[nm] = val
    for nm, val in (extra_args or {}).items():
        if val is None:
            continue
        extras["(%s)" % nm] = ctx.eval(parse(val)) if isinstance(val, str) else to_r(val)
    n = None
    if index is not None and len(mapping):
        n = len(index)
    allv = list(variables.items()) + list(extras.items())
    if n is None:
        n = rlen(allv[0][1]) if allv else 0
        index = pd.RangeIndex(n)
    for nm, v in allv:
        if rlen(v) != n:
            if rlen(v) == 1 and not len(mapping):
                continue
            raise RError("variable lengths differ (found for '%s')" % nm)
    # subset
    keep = np.arange(n)
    if subset is not None:
        if isinstance(subset, str):
            sv = ctx.eval(parse(subset))
        else:
            sv = to_r(subset)
        if isinstance(sv, Logical):
            mask = sv.values & ~sv.na
            keep = np.nonzero(np.resize(mask, n))[0]
        else:
            keep = (as_double(sv).astype(np.int64))
            keep = keep[keep >= 0]
    sub = lambda v: take(v, keep) if len(keep) != n or not np.array_equal(keep, np.arange(n)) else v
    variables = OrderedDict((k, sub(v)) for k, v in variables.items())
    extras = OrderedDict((k, sub(v)) for k, v in extras.items())
    index = index[keep]
    # na.action
    na_action = na_action or OPTIONS["na.action"]
    na_action = na_action.replace("_", ".")
    omitted = None
    m = len(index)
    if na_action in ("na.omit", "na.exclude", "na.fail"):
        bad = np.zeros(m, dtype=bool)
        for v in list(variables.values()) + list(extras.values()):
            bad |= isna(v)
        if bad.any():
            if na_action == "na.fail":
                raise RError("missing values in object")
            good = np.nonzero(~bad)[0]
            omitted = {"type": "omit" if na_action == "na.omit" else "exclude",
                       "index": list(index[bad])}
            variables = OrderedDict((k, take(v, good)) for k, v in variables.items())
            extras = OrderedDict((k, take(v, good)) for k, v in extras.items())
            index = index[good]
    elif na_action != "na.pass":
        raise ValueError("na_action must be one of 'na.omit', 'na.exclude', 'na.fail', 'na.pass'")
    # xlev / drop.unused.levels
    if xlev:
        for nm, xl in xlev.items():
            if nm not in variables or xl is None:
                continue
            xi = variables[nm]
            if isinstance(xi, Character):
                xi = as_factor(xi)
            if not isinstance(xi, Factor):
                warnings.warn("variable '%s' is not a factor" % nm)
                continue
            ctr = xi.contrasts
            dropped = xi.drop_unused()
            new = [l for l in dropped.levels if l not in xl]
            if new:
                raise RError("factor %s has new level%s %s" % (nm, "s" if len(new) > 1 else "", ", ".join(new)))
            nf = as_factor(Character(list(dropped.labels())), levels=list(xl), ordered=xi.ordered)
            variables[nm] = nf
            if ctr is not None:
                warnings.warn("contrasts dropped from factor %s" % nm)
    elif drop_unused_levels:
        for nm, x in list(variables.items()):
            if isinstance(x, Factor):
                d = x.drop_unused()
                if d.nlevels < x.nlevels:
                    if x.contrasts is not None:
                        warnings.warn("contrasts dropped from factor %s due to missing levels" % nm)
                    variables[nm] = Factor(d.codes, d.levels, x.ordered, None)
    t = terms.copy()
    t.data_classes = OrderedDict((k, class_of(v)) for k, v in list(variables.items()) + list(extras.items()))
    t.predvars = state
    return ModelFrame(variables, t, index, omitted, extras, state, n)


# ---------------------------------------------------------------------------
# helpers on model frames
# ---------------------------------------------------------------------------

def model_response(mf: ModelFrame, type: str = "any"):
    """R's ``model.response``."""
    if not mf.terms.response:
        return None
    nm = list(mf.variables)[0]
    v = mf.variables[nm]
    return _to_output(v, mf.index, nm, type)


def _to_output(v, index, name, type="any"):
    import pandas as pd
    if isinstance(v, RMatrix):
        if v.ncol == 1:
            return pd.Series(v.data[:, 0], index=index, name=name)
        cols = v.colnames if v.colnames is not None else [str(j + 1) for j in range(v.ncol)]
        return pd.DataFrame(v.data, index=index, columns=cols)
    if isinstance(v, Factor):
        if type == "numeric":
            warnings.warn("using type = \"numeric\" with a factor response will be ignored")
        return pd.Series(pd.Categorical.from_codes(v.codes, v.levels, ordered=v.ordered), index=index, name=name)
    if isinstance(v, Logical):
        if type in ("numeric", "double"):
            return pd.Series(v.to_float(), index=index, name=name)
        return pd.Series(v.values if not v.na.any() else pd.array([None if n else bool(b) for b, n in zip(v.values, v.na)], dtype="boolean"), index=index, name=name)
    if isinstance(v, Character):
        return pd.Series(v.values, index=index, name=name)
    arr = np.asarray(v)
    if type in ("numeric", "double"):
        arr = arr.astype(float)
    return pd.Series(arr, index=index, name=name)


def model_offset(mf: ModelFrame):
    """R's ``model.offset``: the sum of all ``offset()`` terms and ``(offset)``."""
    import pandas as pd
    offsets = mf.terms.offset or []
    total = None
    names = list(mf.variables)
    for i in offsets:
        v = as_double(mf.variables[names[i - 1]])
        total = v if total is None else total + v
    if "(offset)" in mf.extras:
        v = as_double(mf.extras["(offset)"])
        total = v if total is None else total + v
    if total is None:
        return None
    return pd.Series(total, index=mf.index, name="offset")


def model_weights(mf: ModelFrame):
    return model_extract(mf, "weights")


def model_extract(mf: ModelFrame, component: str):
    """R's ``model.extract(frame, component)``."""
    if component == "response":
        return model_response(mf)
    if component == "offset":
        return model_offset(mf)
    key = "(%s)" % component
    if key in mf.extras:
        return _to_output(mf.extras[key], mf.index, component)
    return None


# ---------------------------------------------------------------------------
# Model matrix
# ---------------------------------------------------------------------------

class _VarInfo:
    __slots__ = ("value", "nlevs", "ncols", "contr1", "contr2", "is_factor", "name")


def _contrast_matrix(f: Factor, spec) -> ContrastMatrix:
    if spec is None:
        spec = OPTIONS["contrasts"][1 if f.ordered else 0]
    if isinstance(spec, ContrastMatrix):
        if spec.matrix.shape[0] != f.nlevels:
            raise RError("wrong number of contrast matrix rows")
        return ContrastMatrix(spec.matrix, f.levels, spec.colnames)
    return contrast_from_spec(spec, f.levels)


def _prepare_model_data(terms: Terms, mf: ModelFrame, contrasts_arg):
    names = list(mf.variables)
    data = OrderedDict()
    resp = terms.response
    contrasts_used = OrderedDict()
    for i, nm in enumerate(names):
        v = mf.variables[nm]
        if isinstance(v, Character):
            v = as_factor(v)
        data[nm] = v
    isF = OrderedDict()
    for i, nm in enumerate(names):
        v = data[nm]
        isF[nm] = (isinstance(v, (Factor, Logical))) and not (resp and i == resp - 1)
    for nm, flag in isF.items():
        if not flag:
            continue
        v = data[nm]
        if isinstance(v, Logical):
            f = as_factor(v, levels=["FALSE", "TRUE"])
            f = Factor(f.codes, ["FALSE", "TRUE"], False, None)
            data[nm] = ("logical", v, f)
            f_obj = f
        else:
            f_obj = v
        if f_obj.contrasts is None:
            if f_obj.nlevels < 2:
                raise RError("contrasts can be applied only to factors with 2 or more levels")
            f_obj = Factor(f_obj.codes, f_obj.levels, f_obj.ordered, OPTIONS["contrasts"][1 if f_obj.ordered else 0])
        if isinstance(data[nm], tuple):
            data[nm] = ("logical", data[nm][1], f_obj)
        else:
            data[nm] = f_obj
    if contrasts_arg:
        if not isinstance(contrasts_arg, Mapping):
            warnings.warn("non-list contrasts argument ignored")
        else:
            for cn, ca in contrasts_arg.items():
                if cn not in data:
                    warnings.warn("variable '%s' is absent, its contrast will be ignored" % cn)
                    continue
                v = data[cn]
                lv = None
                if isinstance(v, tuple):
                    lv = v
                    v = v[2]
                if not isinstance(v, Factor):
                    v = as_factor(v)
                if isinstance(ca, RMatrix):
                    ca = ContrastMatrix(ca.data, v.levels, ca.colnames)
                if not isinstance(ca, (str, ContrastMatrix)) and not callable(ca):
                    cols = None
                    try:
                        import pandas as pd
                        if isinstance(ca, pd.DataFrame):
                            cols = [str(c) for c in ca.columns]
                    except Exception:  # pragma: no cover
                        pass
                    ca = ContrastMatrix(np.asarray(ca, dtype=float), v.levels, cols)
                if isinstance(ca, ContrastMatrix):
                    newf = set_contrasts(v, ca, ca.ncol)
                else:
                    newf = set_contrasts(v, ca)
                data[cn] = ("logical", lv[1], newf) if lv is not None else newf
    for nm, flag in isF.items():
        if flag:
            v = data[nm]
            f = v[2] if isinstance(v, tuple) else v
            contrasts_used[nm] = f.contrasts
    return data, isF, contrasts_used


def _one_hot_block(codes: np.ndarray, contr: np.ndarray, sparse: bool):
    n = len(codes)
    na = codes < 0
    if sparse:
        import scipy.sparse as sp
        good = ~na
        rows = np.nonzero(good)[0]
        ind = sp.csr_matrix((np.ones(len(rows)), (rows, codes[good])), shape=(n, contr.shape[0]))
        block = (ind @ sp.csr_matrix(contr)).tocsc()
        if na.any():
            block = block.tolil()
            block[np.nonzero(na)[0], :] = np.nan
            block = block.tocsc()
        return block
    block = contr[np.where(na, 0, codes)]
    if na.any():
        block = block.copy()
        block[na] = np.nan
    return block


def _khatri_rao(A, B, sparse: bool):
    """Row-wise Kronecker product with columns of A varying fastest."""
    if sparse:
        import scipy.sparse as sp
        A = sp.csc_matrix(A)
        B = sp.csc_matrix(B)
        blocks = [A.multiply(B[:, j]).tocsc() for j in range(B.shape[1])]
        return sp.hstack(blocks, format="csc") if blocks else sp.csc_matrix((A.shape[0], 0))
    n = A.shape[0]
    return np.einsum("ni,nj->nji", A, B).reshape(n, A.shape[1] * B.shape[1])


def build_model_matrix(terms: Terms, mf: ModelFrame, contrasts_arg=None, sparse: bool = False,
                       contrasts_override: Optional[Dict[str, Any]] = None):
    """Return (matrix, colnames, assign, contrasts) for a terms object and model frame."""
    data, isF, contrasts_used = _prepare_model_data(terms, mf, contrasts_arg)
    if contrasts_override:
        for nm, cm in contrasts_override.items():
            if nm in data:
                v = data[nm]
                f = v[2] if isinstance(v, tuple) else v
                f = Factor(f.codes, f.levels, f.ordered, cm)
                data[nm] = ("logical", v[1], f) if isinstance(v, tuple) else f
                contrasts_used[nm] = cm
    names = list(data)
    nvar = len(names)
    n = len(mf)
    factors = terms.factors.copy() if terms.factors is not None and terms.factors.size else np.zeros((nvar, 0), dtype=np.int64)
    nterms = factors.shape[1]
    resp = terms.response
    intercept = terms.intercept
    if factors.shape[0] != nvar and nterms:
        raise RError("model frame and formula mismatch in model.matrix()")

    labmap = dict(zip(terms.variables, getattr(terms, "varnames", terms.variables)))
    infos: List[_VarInfo] = []
    for i, nm in enumerate(names):
        v = data[nm]
        info = _VarInfo()
        info.name = labmap.get(nm, nm)
        info.contr1 = info.contr2 = None
        if isinstance(v, tuple):
            info.value = v[2]
            info.is_factor = True
            info.nlevs = 2
        elif isinstance(v, Factor):
            info.value = v
            info.is_factor = True
            info.nlevs = v.nlevels
            if v.nlevels < 1 and any(factors[i, :]):
                raise RError("variable %d has no levels" % (i + 1))
        elif isinstance(v, RMatrix):
            info.value = v
            info.is_factor = False
            info.nlevs = 0
            info.ncols = v.ncol
        elif isinstance(v, np.ndarray) and v.dtype.kind in "fiu":
            info.value = v.astype(float)
            info.is_factor = False
            info.nlevs = 0
            info.ncols = 1
        elif isinstance(v, Logical):
            info.value = v
            info.is_factor = False
            info.nlevs = 0
            info.ncols = 1
        else:
            if resp and i == resp - 1:
                info.value = v
                info.is_factor = False
                info.nlevs = 0
                info.ncols = 1
            else:
                raise RError("variables of type '%s' are not supported" % type(v).__name__)
        infos.append(info)

    # no intercept: first factor in the first term with a factor gets full coding
    if not intercept and not sparse:
        done = False
        for j in range(nterms):
            for i in range(resp, nvar):
                if infos[i].nlevs > 1 and factors[i, j] > 0:
                    factors[i, j] = 2
                    done = True
                    break
            if done:
                break
    elif not intercept and sparse:
        # Matrix::model.spmatrix: first entry equal to 1 among factor rows (column-major)
        indF = [i for i in range(nvar) if isinstance(data[names[i]], (Factor, tuple, Logical))]
        done = False
        for j in range(nterms):
            for i in indF:
                if factors[i, j] == 1:
                    factors[i, j] = 2
                    done = True
                    break
            if done:
                break

    # contrasts per variable
    for i, info in enumerate(infos):
        if not info.is_factor or nterms == 0 or not factors[i, :].any():
            continue
        f = info.value
        if (factors[i, :] == 1).any():
            info.contr1 = _contrast_matrix(f, f.contrasts)
        if (factors[i, :] == 2).any():
            info.contr2 = ContrastMatrix(np.eye(f.nlevels), f.levels, f.levels)

    # response appearing on the RHS
    rhs_response = -1
    if resp and not sparse:
        for k in range(nterms):
            col = factors[:, k]
            if col[resp - 1] and (col > 0).sum() == 1:
                rhs_response = k
                break

    blocks = []
    colnames: List[str] = []
    assign: List[int] = []
    if intercept:
        if sparse:
            import scipy.sparse as sp
            blocks.append(sp.csc_matrix(np.ones((n, 1))))
        else:
            blocks.append(np.ones((n, 1)))
        colnames.append("(Intercept)")
        assign.append(0)
    if rhs_response >= 0:
        warnings.warn("the response appeared on the right-hand side and was dropped")
        warnings.warn("problem with term %d in model.matrix: no columns are assigned" % (rhs_response + 1))
    for k in range(nterms):
        if k == rhs_response:
            continue
        block = None
        bnames: List[str] = []
        for i in range(nvar):
            code = factors[i, k]
            if not code:
                continue
            info = infos[i]
            vname = info.name
            if info.is_factor:
                cm = info.contr1 if code == 1 else info.contr2
                part = _one_hot_block(info.value.codes, cm.matrix, sparse)
                pnames = [vname + c for c in cm.colnames] if cm.colnames is not None else \
                    [vname + str(j + 1) for j in range(cm.ncol)]
            else:
                val = info.value
                if isinstance(val, RMatrix):
                    arr = val.data
                    if val.colnames is not None:
                        pnames = list(val.colnames) if sparse else [vname + c for c in val.colnames]
                    else:
                        pnames = [vname + str(j + 1) for j in range(arr.shape[1])] if arr.shape[1] > 1 else [vname]
                elif isinstance(val, Logical):
                    arr = val.to_float().reshape(-1, 1)
                    pnames = [vname]
                else:
                    arr = np.asarray(val, dtype=float).reshape(-1, 1)
                    pnames = [vname]
                if sparse:
                    import scipy.sparse as sp
                    part = sp.csc_matrix(arr)
                else:
                    part = arr
            if block is None:
                block = part
                bnames = pnames
            else:
                block = _khatri_rao(block, part, sparse)
                bnames = [b + ":" + p for p in pnames for b in bnames]
        if block is None:
            continue
        if block.shape[1] == 0:
            warnings.warn("problem with term %d in model.matrix: no columns are assigned" % (k + 1))
        blocks.append(block)
        colnames.extend(bnames)
        assign.extend([k + 1] * block.shape[1])
    if sparse:
        import scipy.sparse as sp
        X = sp.hstack(blocks, format="csc") if blocks else sp.csc_matrix((n, 0))
        X.eliminate_zeros()
    else:
        X = np.hstack(blocks) if blocks else np.zeros((n, 0))
    contrasts_out = OrderedDict()
    for i, nm in enumerate(names):
        if isF.get(nm):
            v = data[nm]
            f = v[2] if isinstance(v, tuple) else v
            contrasts_out[nm] = f.contrasts
    return X, colnames, assign, contrasts_out
