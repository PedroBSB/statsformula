"""Vectorised evaluation of R expressions over a data set.

Variables in a formula (``log(x)``, ``I(x^2)``, ``poly(x, 3)``, ``C(f, sum)``
...) are evaluated here with R semantics (recycling, NA propagation,
integer/double distinction, factors, logical vectors).  All operations are
vectorised with numpy.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._bases import (CONTRAST_FUNCTIONS, ContrastMatrix, bs_basis, contrast_from_spec,
                     normalize_contrast_name, ns_basis, poly_fit, poly_predict, spline_design)
from ._parser import (MISSING, Call, Const, Missing, Node, Num, Str, Sym, deparse)
from ._rtypes import (Character, Factor, Logical, RError, RMatrix, as_character, as_double,
                      as_factor, format_num, from_r, is_numeric, isna, r_sort_key, rlen, to_r)


# ---------------------------------------------------------------------------
# global options (R's options(contrasts=..., na.action=...))
# ---------------------------------------------------------------------------

OPTIONS: Dict[str, Any] = {
    "contrasts": ("contr.treatment", "contr.poly"),
    "na.action": "na.omit",
}


def options(**kwargs):
    """Get/set global options, like R's ``options()``.

    ``options(contrasts=("contr.sum", "contr.poly"))``; ``options(na_action="na.exclude")``.
    Returns the previous values of the options that were set.
    """
    old = {}
    for k, v in kwargs.items():
        key = k.replace("_", ".")
        if key not in OPTIONS:
            raise KeyError("unknown option %r" % k)
        old[k] = OPTIONS[key]
        OPTIONS[key] = tuple(v) if key == "contrasts" else v
    if not kwargs:
        return dict(OPTIONS)
    return old


class RFunction:
    """A callable R function value (e.g. ``contr.sum`` passed to ``C()``)."""

    def __init__(self, name: str, fn: Callable):
        self.name = name
        self.fn = fn

    def __call__(self, *a, **k):
        return self.fn(*a, **k)

    def __repr__(self):
        return "<R function %s>" % self.name


# ---------------------------------------------------------------------------
# argument matching
# ---------------------------------------------------------------------------

def match_args(params: Sequence[str], args, fname: str = "", dots: bool = False):
    """R argument matching: exact names, unique partial names, positional."""
    matched: Dict[str, Node] = {}
    rest = []
    named = [(n, a) for n, a in args if n is not None]
    positional = [a for n, a in args if n is None]
    dots_pos = list(params).index("...") if "..." in params else len(params)
    formal = [p for p in params if p != "..."]
    extra_named = []
    for n, a in named:
        if n in formal:
            if n in matched:
                raise RError("formal argument \"%s\" matched by multiple actual arguments" % n)
            matched[n] = a
            continue
        cands = [p for p in params[:dots_pos] if p != "..." and p.startswith(n) and p not in matched]
        if len(cands) == 1:
            matched[cands[0]] = a
        elif dots or "..." in params:
            extra_named.append((n, a))
        else:
            raise RError("unused argument (%s = %s)%s" % (n, deparse(a), " in %s()" % fname if fname else ""))
    free = [p for p in params[:dots_pos] if p not in matched and p != "..."]
    pi = 0
    for a in positional:
        if pi < len(free):
            matched[free[pi]] = a
            pi += 1
        elif "..." in params or dots:
            rest.append((None, a))
        else:
            raise RError("unused argument (%s)%s" % (deparse(a), " in %s()" % fname if fname else ""))
    rest = rest + extra_named
    for k in list(matched):
        if isinstance(matched[k], Missing):
            del matched[k]
    return matched, rest


# ---------------------------------------------------------------------------
# coercion helpers
# ---------------------------------------------------------------------------

def _num(v, what="binary operator") -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, Logical):
        return v.to_float() if v.na.any() else v.values.astype(np.int64)
    if isinstance(v, RMatrix):
        return v.data
    if isinstance(v, Factor):
        warnings.warn("not meaningful for factors")
        return np.full(len(v), np.nan)
    if isinstance(v, Character):
        raise RError("non-numeric argument to %s" % what)
    if v is None:
        return np.zeros(0)
    return np.asarray(v, dtype=float)


def _recycle(a: np.ndarray, b: np.ndarray):
    la, lb = a.shape[0], b.shape[0]
    if la == lb:
        return a, b
    if la == 0 or lb == 0:
        return a[:0], b[:0]
    n = max(la, lb)
    if n % min(la, lb):
        warnings.warn("longer object length is not a multiple of shorter object length")
    if la < n:
        a = np.resize(a, (n,) + a.shape[1:])
    if lb < n:
        b = np.resize(b, (n,) + b.shape[1:])
    return a, b


def _scalar(v, name="argument"):
    if isinstance(v, (int, float, bool, str)):
        return v
    if isinstance(v, Logical):
        if len(v) == 0 or v.na[0]:
            return None
        return bool(v.values[0])
    if isinstance(v, Character):
        return v.values[0] if len(v) else None
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return None
        x = v.reshape(-1)[0]
        return int(x) if v.dtype.kind in "iu" else float(x)
    if isinstance(v, Factor):
        return v.labels()[0]
    return v


def _bool(v) -> bool:
    s = _scalar(v)
    if s is None:
        raise RError("missing value where TRUE/FALSE needed")
    if isinstance(s, str):
        if s in ("TRUE", "true", "T", "True"):
            return True
        if s in ("FALSE", "false", "F", "False"):
            return False
        raise RError("argument is not interpretable as logical")
    return bool(s)


def _as_logical(v) -> Logical:
    if isinstance(v, Logical):
        return v
    if isinstance(v, np.ndarray) and v.dtype.kind in "fiu":
        vv = v.reshape(-1)
        na = np.isnan(vv) if vv.dtype.kind == "f" else np.zeros(len(vv), bool)
        return Logical(np.where(na, False, vv != 0), na)
    if isinstance(v, Character):
        vals = []
        na = []
        for s in v.values:
            if s in ("TRUE", "true", "T", "True"):
                vals.append(True); na.append(False)
            elif s in ("FALSE", "false", "F", "False"):
                vals.append(False); na.append(False)
            else:
                vals.append(False); na.append(True)
        return Logical(vals, na)
    if isinstance(v, Factor):
        return _as_logical(as_character(v))
    if isinstance(v, RMatrix):
        return _as_logical(v.data.reshape(-1, order="F"))
    raise RError("invalid argument type")


def _wrap_like(result: np.ndarray, a, b):
    """Re-attach matrix structure after an elementwise op."""
    for v in (a, b):
        if isinstance(v, RMatrix):
            return RMatrix(result, v.colnames)
    return result


def _matrix_operands(a, b):
    A = _num(a)
    B = _num(b)
    if isinstance(a, RMatrix) or isinstance(b, RMatrix):
        shape = a.data.shape if isinstance(a, RMatrix) else b.data.shape
        def expand(X):
            if X.ndim == 2:
                return X
            n = shape[0] * shape[1]
            if X.shape[0] == 0:
                return np.zeros(shape)
            flat = np.resize(X, n)
            return flat.reshape(shape, order="F")
        return expand(A), expand(B)
    return _recycle(A.reshape(-1), B.reshape(-1))


def arith(op: str, a, b):
    if isinstance(a, Character) or isinstance(b, Character):
        raise RError("non-numeric argument to binary operator")
    A, B = _matrix_operands(a, b)
    both_int = A.dtype.kind in "iu" and B.dtype.kind in "iu"
    with np.errstate(all="ignore"):
        if op == "+":
            r = A + B
        elif op == "-":
            r = A - B
        elif op == "*":
            r = A * B
        elif op == "/":
            r = A.astype(float) / B.astype(float)
        elif op == "^":
            r = np.power(A.astype(float), B.astype(float))
            r = np.where((A == 1) | (B == 0), 1.0, r)
        elif op == "%%":
            if both_int:
                r = np.where(B == 0, np.nan, np.mod(A, np.where(B == 0, 1, B)))
                if not np.isnan(r).any():
                    r = r.astype(np.int64)
            else:
                Af, Bf = A.astype(float), B.astype(float)
                r = Af - np.floor(Af / Bf) * Bf
        elif op == "%/%":
            if both_int:
                r = np.where(B == 0, np.nan, np.floor_divide(A, np.where(B == 0, 1, B)))
                if not np.isnan(r).any():
                    r = r.astype(np.int64)
            else:
                r = np.floor(A.astype(float) / B.astype(float))
        else:  # pragma: no cover
            raise RError("unknown operator %s" % op)
    if not both_int and r.dtype.kind in "iu":
        r = r.astype(float)
    return _wrap_like(r, a, b)


def _cmp_values(op, A, B):
    if op == "==":
        return A == B
    if op == "!=":
        return A != B
    if op == "<":
        return A < B
    if op == ">":
        return A > B
    if op == "<=":
        return A <= B
    if op == ">=":
        return A >= B
    raise RError(op)


def compare(op: str, a, b) -> Logical:
    # factor handling
    if isinstance(a, Factor) or isinstance(b, Factor):
        fa = a if isinstance(a, Factor) else None
        fb = b if isinstance(b, Factor) else None
        if op not in ("==", "!=") and not ((fa is None or fa.ordered) and (fb is None or fb.ordered)):
            warnings.warn("'%s' not meaningful for factors" % op)
            n = max(rlen(a), rlen(b))
            return Logical(np.zeros(n, bool), np.ones(n, bool))
        if op not in ("==", "!="):
            f = fa or fb
            other = b if fa is not None else a
            if isinstance(other, Factor):
                oc = other.codes
            else:
                lab = as_character(other).values
                idx = {l: i for i, l in enumerate(f.levels)}
                oc = np.array([idx.get(s, -1) if s is not None else -1 for s in lab])
            fc = f.codes
            A, B = (fc, oc) if fa is not None else (oc, fc)
            A, B = _recycle(A.astype(float), B.astype(float))
            na = (A < 0) | (B < 0)
            return Logical(np.where(na, False, _cmp_values(op, A, B)), na)
        a = as_character(a) if isinstance(a, Factor) else a
        b = as_character(b) if isinstance(b, Factor) else b
    if isinstance(a, Character) or isinstance(b, Character):
        import pandas as pd
        A = as_character(a).values
        B = as_character(b).values
        A, B = _recycle(A, B)
        na = pd.isna(A) | pd.isna(B)
        if op in ("==", "!="):
            vals = np.asarray(A == B, dtype=bool) & ~na
            if op == "!=":
                vals = ~vals & ~na
        else:
            ka = [r_sort_key(x) if x is not None else None for x in A]
            kb = [r_sort_key(y) if y is not None else None for y in B]
            vals = np.array([_cmp_values(op, x, y) if x is not None and y is not None else False
                             for x, y in zip(ka, kb)], dtype=bool)
        return Logical(vals, na)
    A, B = _matrix_operands(a, b)
    Af, Bf = A.astype(float), B.astype(float)
    na = np.isnan(Af) | np.isnan(Bf)
    with np.errstate(invalid="ignore"):
        vals = _cmp_values(op, Af, Bf)
    vals = np.where(na, False, vals)
    if vals.ndim == 2:
        vals = vals.reshape(-1, order="F")
        na = na.reshape(-1, order="F")
    return Logical(vals, na)


def logic(op: str, a, b) -> Logical:
    A = _as_logical(a)
    B = _as_logical(b)
    av, bv = _recycle(A.values, B.values)
    an, bn = _recycle(A.na, B.na)
    if op in ("&", "&&"):
        false_a = ~av & ~an
        false_b = ~bv & ~bn
        res_false = false_a | false_b
        na = (an | bn) & ~res_false
        vals = av & bv & ~na
    else:
        true_a = av & ~an
        true_b = bv & ~bn
        res_true = true_a | true_b
        na = (an | bn) & ~res_true
        vals = res_true
    if op in ("&&", "||"):
        return Logical(vals[:1], na[:1])
    return Logical(vals, na)


def r_match(x, table) -> np.ndarray:
    """0-based ``match(x, table)`` (first occurrence) or -1; NA matches NA."""
    import pandas as pd
    numeric = lambda v: isinstance(v, np.ndarray) or isinstance(v, Logical)
    if numeric(x) and numeric(table):
        xs, ts = as_double(x), as_double(table)
    else:
        xs, ts = as_character(x).values, as_character(table).values
    idx = pd.Index(ts)
    first = ~idx.duplicated(keep="first")
    pos = np.nonzero(first)[0]
    m = idx[first].get_indexer(pd.Index(xs))
    return np.where(m >= 0, pos[np.maximum(m, 0)], -1).astype(np.int64)


# ---------------------------------------------------------------------------
# evaluation context
# ---------------------------------------------------------------------------

class EvalContext:
    """Evaluates expressions over ``data`` with R lookup rules.

    ``state`` stores the data-dependent parameters of stateful functions
    (``poly``, ``bs``, ``ns``, ``scale``, mgcv smooths) keyed by the deparsed
    call, so that the same transformation can be applied to new data (R's
    ``predvars`` mechanism).  ``fitting`` selects whether to learn the
    parameters (``True``) or reuse them (``False``).
    """

    def __init__(self, data: Optional[Mapping[str, Any]] = None, env: Optional[Mapping[str, Any]] = None,
                 state: Optional[dict] = None, fitting: bool = True, nrows: Optional[int] = None):
        self.data = data if data is not None else {}
        self.env = env if env is not None else {}
        self.state = state if state is not None else {}
        self.fitting = fitting
        self._cache: Dict[str, Any] = {}
        self.nrows = nrows

    # lookup ---------------------------------------------------------------
    def has_column(self, name: str) -> bool:
        try:
            return name in self.data
        except TypeError:
            return False

    def column(self, name: str):
        if name in self._cache:
            return self._cache[name]
        v = to_r(self.data[name])
        self._cache[name] = v
        return v

    def lookup(self, name: str):
        if self.has_column(name):
            return self.column(name)
        if name in self.env:
            v = self.env[name]
            if callable(v) and not isinstance(v, (np.ndarray,)):
                return v
            return to_r(v)
        if name in _CONSTANTS:
            return _CONSTANTS[name]()
        if name in CONTRAST_FUNCTIONS:
            return RFunction(name, CONTRAST_FUNCTIONS[name])
        if name in FUNCTIONS:
            return RFunction(name, FUNCTIONS[name])
        obj = _resolve_dotted(name, self.env)
        if obj is not None:
            return obj if callable(obj) else to_r(obj)
        raise RError("object '%s' not found" % name)

    # evaluation -----------------------------------------------------------
    def eval(self, node: Node):
        if isinstance(node, Num):
            if node.is_int:
                return np.array([int(node.value)], dtype=np.int64)
            return np.array([node.value])
        if isinstance(node, Str):
            return Character([node.value])
        if isinstance(node, Const):
            v = node.value
            if v == "TRUE":
                return Logical([True])
            if v == "FALSE":
                return Logical([False])
            if v == "NULL":
                return None
            if v in ("NA", "NA_integer_"):
                return Logical([False], [True]) if v == "NA" else np.array([np.nan])
            if v == "NA_real_":
                return np.array([np.nan])
            if v == "NA_character_":
                return Character([None])
        if isinstance(node, Sym):
            return self.lookup(node.name)
        if isinstance(node, Missing):
            raise RError("argument is missing, with no default")
        if isinstance(node, Call):
            return self.eval_call(node)
        raise RError("cannot evaluate %r" % (node,))

    def eval_call(self, node: Call):
        fn = node.fn
        if isinstance(fn, Call) and fn.fname in ("::", ":::") and isinstance(fn.argv[1], Sym):
            node = Call(fn.argv[1], node.args)
            fn = node.fn
        if isinstance(fn, Sym):
            name = fn.name
            handler = _OPERATORS.get(name)
            if handler is not None and all(n is None for n, _ in node.args):
                return handler(self, node)
            if name in FUNCTIONS and not (name in self.env and callable(self.env[name]) and name in USER_OVERRIDABLE):
                return FUNCTIONS[name](self, node)
            target = None
            if name in self.env and callable(self.env[name]):
                target = self.env[name]
            elif self.has_column(name):
                raise RError("attempt to apply non-function")
            else:
                target = _resolve_dotted(name, self.env)
            if target is None or not callable(target):
                raise RError("could not find function \"%s\"" % name)
        else:
            target = self.eval(fn)
            if not callable(target):
                raise RError("attempt to apply non-function")
        if isinstance(target, RFunction):
            return _call_rfunction(self, target, node)
        pos = []
        kw = {}
        for n, a in node.args:
            val = from_r(self.eval(a))
            if n is None:
                pos.append(val)
            else:
                kw[n.replace(".", "_") if not n.isidentifier() else n] = val
        return to_r(target(*pos, **kw))


USER_OVERRIDABLE: set = set()


def _call_rfunction(ctx, f: RFunction, node: Call):
    if f.name in FUNCTIONS:
        return FUNCTIONS[f.name](ctx, Call(Sym(f.name), node.args))
    args = [ctx.eval(a) for n, a in node.args if n is None]
    kwargs = {n.replace(".", "_"): ctx.eval(a) for n, a in node.args if n is not None}
    return _contr_call(f.name, args, kwargs)


def _resolve_dotted(name: str, env: Mapping[str, Any]):
    parts = name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        head = ".".join(parts[:i])
        if head in env:
            obj = env[head]
            try:
                for p in parts[i:]:
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                return None
    return None


_CONSTANTS = {
    "pi": lambda: np.array([math.pi]),
    "T": lambda: Logical([True]),
    "F": lambda: Logical([False]),
    "LETTERS": lambda: Character([chr(65 + i) for i in range(26)]),
    "letters": lambda: Character([chr(97 + i) for i in range(26)]),
    "month.name": lambda: Character(["January", "February", "March", "April", "May", "June", "July",
                                     "August", "September", "October", "November", "December"]),
    "month.abb": lambda: Character(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                                    "Oct", "Nov", "Dec"]),
}


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

def _op_arith(op):
    def f(ctx, node):
        a = node.argv
        if len(a) == 1:
            v = ctx.eval(a[0])
            if op == "-":
                if isinstance(v, RMatrix):
                    return RMatrix(-v.data, v.colnames)
                x = _num(v, "unary operator")
                return -x
            if op == "+":
                return _num(v, "unary operator") if not isinstance(v, RMatrix) else v
            raise RError("invalid unary operator")
        return arith(op, ctx.eval(a[0]), ctx.eval(a[1]))
    return f


def _op_cmp(op):
    def f(ctx, node):
        a = node.argv
        return compare(op, ctx.eval(a[0]), ctx.eval(a[1]))
    return f


def _op_logic(op):
    def f(ctx, node):
        a = node.argv
        if op == "!":
            v = ctx.eval(a[0])
            L = _as_logical(v)
            return Logical(~L.values & ~L.na, L.na)
        return logic(op, ctx.eval(a[0]), ctx.eval(a[1]))
    return f


def _op_paren(ctx, node):
    return ctx.eval(node.argv[0])


def _op_in(ctx, node):
    a = node.argv
    m = r_match(ctx.eval(a[0]), ctx.eval(a[1]))
    return Logical(m >= 0)


def _op_colon(ctx, node):
    a = node.argv
    x = ctx.eval(a[0])
    y = ctx.eval(a[1])
    if isinstance(x, (Factor, Character)) or isinstance(y, (Factor, Character)):
        return _interaction([as_factor(x), as_factor(y)], sep=":", lex_order=True, drop=False)
    fr = _scalar(x)
    to = _scalar(y)
    if fr is None or to is None:
        raise RError("NA/NaN argument")
    step = 1 if to >= fr else -1
    n = int(math.floor(abs(to - fr) + 1e-10)) + 1
    seq = fr + step * np.arange(n)
    if float(fr).is_integer():
        return seq.astype(np.int64)
    return seq.astype(float)


def _op_index(ctx, node):
    a = node.args
    x = ctx.eval(a[0][1])
    idx_nodes = [n for _, n in a[1:]]

    def resolve(ix, n):
        if isinstance(ix, Missing):
            return slice(None)
        v = ctx.eval(ix)
        if isinstance(v, Logical):
            m = np.resize(v.values, n)
            return np.nonzero(m)[0]
        if isinstance(v, Character):
            raise RError("character indexing is not supported")
        iv = as_double(v).astype(np.int64)
        if (iv < 0).all():
            keep = np.ones(n, bool)
            keep[-iv - 1] = False
            return np.nonzero(keep)[0]
        return iv[iv > 0] - 1

    if isinstance(x, RMatrix):
        if len(idx_nodes) == 2:
            ri = resolve(idx_nodes[0], x.data.shape[0])
            ci = resolve(idx_nodes[1], x.data.shape[1])
            sub = x.data[ri][:, ci]
            cn = None if x.colnames is None else list(np.array(x.colnames, dtype=object)[ci])
            if sub.ndim == 2 and sub.shape[1] == 1:
                return sub[:, 0]
            return RMatrix(sub, cn)
        flat = x.data.reshape(-1, order="F")
        return flat[resolve(idx_nodes[0], len(flat))]
    if len(idx_nodes) != 1:
        raise RError("incorrect number of dimensions")
    n = rlen(x)
    ii = resolve(idx_nodes[0], n)
    if isinstance(x, (Logical, Character, Factor)):
        if isinstance(ii, slice):
            return x
        return x.take(ii)
    return np.asarray(x)[ii]


def _op_dollar(ctx, node):
    a = node.argv
    base = a[0]
    if isinstance(base, Sym) and isinstance(a[1], (Sym, Str)):
        member = a[1].name if isinstance(a[1], Sym) else a[1].value
        try:
            obj = ctx.lookup(base.name)
        except RError:
            obj = None
        raw = ctx.env.get(base.name) if obj is None or not hasattr(obj, member) else obj
        if raw is not None:
            try:
                return to_r(raw[member])
            except Exception:
                if hasattr(raw, member):
                    return to_r(getattr(raw, member))
    raise RError("'$' is only supported on objects in the evaluation environment")


def _op_tilde(ctx, node):
    raise RError("formulas cannot be evaluated as variables")


_OPERATORS: Dict[str, Callable] = {
    "+": _op_arith("+"), "-": _op_arith("-"), "*": _op_arith("*"), "/": _op_arith("/"),
    "^": _op_arith("^"), "%%": _op_arith("%%"), "%/%": _op_arith("%/%"),
    "==": _op_cmp("=="), "!=": _op_cmp("!="), "<": _op_cmp("<"), ">": _op_cmp(">"),
    "<=": _op_cmp("<="), ">=": _op_cmp(">="),
    "&": _op_logic("&"), "|": _op_logic("|"), "&&": _op_logic("&&"), "||": _op_logic("||"),
    "!": _op_logic("!"), "(": _op_paren, "%in%": _op_in, ":": _op_colon, "[": _op_index,
    "[[": _op_index, "$": _op_dollar, "~": _op_tilde,
}


# ---------------------------------------------------------------------------
# function library
# ---------------------------------------------------------------------------

FUNCTIONS: Dict[str, Callable] = {}


def rfunc(*names):
    def deco(f):
        for n in names:
            FUNCTIONS[n] = f
        return f
    return deco


def _evargs(ctx, node, params, dots=False):
    m, rest = match_args(params, node.args, node.fname or "", dots)
    vals = {k: ctx.eval(v) for k, v in m.items()}
    return vals, [(n, ctx.eval(a)) for n, a in rest], m


def _keep_shape(fn):
    """Apply elementwise numeric fn preserving matrix structure."""
    def apply(v):
        if isinstance(v, RMatrix):
            with np.errstate(all="ignore"):
                return RMatrix(fn(v.data.astype(float)), v.colnames)
        x = _num(v, "mathematical function")
        with np.errstate(all="ignore"):
            r = fn(x.astype(float))
        return r
    return apply


def _math1(np_fn, int_preserving=False):
    def f(ctx, node):
        vals, _, _ = _evargs(ctx, node, ["x"])
        if "x" not in vals:
            raise RError("0 arguments passed to '%s' which requires 1" % node.fname)
        x = vals["x"]
        if isinstance(x, (Character, Factor)):
            raise RError("non-numeric argument to mathematical function")
        if int_preserving and isinstance(x, np.ndarray) and x.dtype.kind in "iu":
            return np_fn(x)
        res = _keep_shape(np_fn)(x)
        if not isinstance(res, RMatrix) and np.any(np.isnan(res) & ~np.isnan(_num(x).astype(float))):
            warnings.warn("NaNs produced")
        return res
    return f


for _n, _f in {
    "sqrt": np.sqrt, "exp": np.exp, "expm1": np.expm1, "log1p": np.log1p, "log2": np.log2,
    "log10": np.log10, "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin,
    "acos": np.arccos, "atan": np.arctan, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh, "floor": np.floor,
    "ceiling": np.ceil, "trunc": np.trunc, "sinpi": lambda x: np.sin(np.pi * x),
    "cospi": lambda x: np.cos(np.pi * x), "tanpi": lambda x: np.tan(np.pi * x),
}.items():
    FUNCTIONS[_n] = _math1(_f)
FUNCTIONS["abs"] = _math1(np.abs, int_preserving=True)
FUNCTIONS["sign"] = _math1(np.sign)


def _vec_special(name):
    def get():
        try:
            import scipy.special as sp
        except ImportError:  # pragma: no cover
            raise RError("%s() requires scipy" % name)
        return sp
    return get


@rfunc("gamma", "lgamma", "digamma", "trigamma", "factorial", "lfactorial")
def _f_gamma(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x"])
    x = _num(vals["x"]).astype(float)
    import scipy.special as sp
    with np.errstate(all="ignore"):
        return {
            "gamma": sp.gamma, "lgamma": lambda v: sp.gammaln(v),
            "digamma": sp.digamma, "trigamma": lambda v: sp.polygamma(1, v),
            "factorial": lambda v: sp.gamma(v + 1), "lfactorial": lambda v: sp.gammaln(v + 1),
        }[node.fname](x)


@rfunc("beta", "lbeta", "choose", "lchoose")
def _f_beta(ctx, node):
    p = ["a", "b"] if node.fname in ("beta", "lbeta") else ["n", "k"]
    vals, _, _ = _evargs(ctx, node, p)
    A, B = _recycle(_num(vals[p[0]]).astype(float), _num(vals[p[1]]).astype(float))
    import scipy.special as sp
    if node.fname == "beta":
        return sp.beta(A, B)
    if node.fname == "lbeta":
        return sp.betaln(A, B)
    if node.fname == "choose":
        return np.round(sp.binom(A, np.round(B)))
    return np.log(np.abs(sp.binom(A, np.round(B))))


@rfunc("log")
def _f_log(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "base"])
    x = vals["x"]
    if isinstance(x, (Character, Factor)):
        raise RError("non-numeric argument to mathematical function")
    if "base" in vals:
        b = float(_scalar(vals["base"]))
        fn = lambda v: np.log(v) / np.log(b)
    else:
        fn = np.log
    res = _keep_shape(fn)(x)
    if not isinstance(res, RMatrix):
        src = _num(x).astype(float)
        if np.any(np.isnan(res) & ~np.isnan(src)):
            warnings.warn("NaNs produced")
    return res


@rfunc("logb")
def _f_logb(ctx, node):
    return _f_log(ctx, node)


@rfunc("round")
def _f_round(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "digits"])
    d = int(_scalar(vals.get("digits", np.array([0]))))
    return _keep_shape(lambda v: np.round(v, d))(vals["x"])


@rfunc("signif")
def _f_signif(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "digits"])
    d = max(1, int(_scalar(vals.get("digits", np.array([6])))))

    def sig(v):
        with np.errstate(all="ignore"):
            mag = np.where(v == 0, 0, np.floor(np.log10(np.abs(np.where(v == 0, 1, v)))))
            fac = 10.0 ** (d - 1 - mag)
            return np.round(v * fac) / fac
    return _keep_shape(sig)(vals["x"])


@rfunc("atan2")
def _f_atan2(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["y", "x"])
    A, B = _recycle(_num(vals["y"]).astype(float), _num(vals["x"]).astype(float))
    return np.arctan2(A, B)


def _cum(fn):
    def f(ctx, node):
        vals, _, _ = _evargs(ctx, node, ["x"])
        x = _num(vals["x"]).astype(float)
        out = fn(x)
        nan = np.isnan(x)
        if nan.any():
            first = np.argmax(nan)
            out[first:] = np.nan
        return out
    return f


FUNCTIONS["cumsum"] = _cum(np.cumsum)
FUNCTIONS["cumprod"] = _cum(np.cumprod)
FUNCTIONS["cummax"] = _cum(np.maximum.accumulate)
FUNCTIONS["cummin"] = _cum(np.minimum.accumulate)


def _summary(fn, name):
    def f(ctx, node):
        vals, rest, _ = _evargs(ctx, node, ["x", "...", "na.rm"], dots=True)
        parts = [vals["x"]] + [v for _, v in rest] if "x" in vals else [v for _, v in rest]
        x = np.concatenate([_num(p).astype(float).reshape(-1) for p in parts]) if parts else np.zeros(0)
        narm = _bool(vals["na.rm"]) if "na.rm" in vals else False
        if narm:
            x = x[~np.isnan(x)]
        elif np.isnan(x).any():
            return np.array([np.nan])
        return np.array([fn(x)])
    return f


FUNCTIONS["sum"] = _summary(np.sum, "sum")
FUNCTIONS["prod"] = _summary(np.prod, "prod")
FUNCTIONS["max"] = _summary(lambda v: np.max(v) if len(v) else -np.inf, "max")
FUNCTIONS["min"] = _summary(lambda v: np.min(v) if len(v) else np.inf, "min")


@rfunc("mean")
def _f_mean(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "...", "na.rm"], dots=True)
    x = _num(vals["x"]).astype(float).reshape(-1)
    if "na.rm" in vals and _bool(vals["na.rm"]):
        x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.array([np.nan])
    m = x.mean()
    if np.isfinite(m):
        m = m + (x - m).mean()  # R's two-pass mean
    return np.array([m])


@rfunc("median")
def _f_median(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "na.rm"])
    x = _num(vals["x"]).astype(float).reshape(-1)
    if "na.rm" in vals and _bool(vals["na.rm"]):
        x = x[~np.isnan(x)]
    elif np.isnan(x).any():
        return np.array([np.nan])
    return np.array([np.median(x)]) if len(x) else np.array([np.nan])


@rfunc("var", "sd")
def _f_var(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "y", "na.rm"])
    x = _num(vals["x"]).astype(float).reshape(-1)
    if "na.rm" in vals and _bool(vals["na.rm"]):
        x = x[~np.isnan(x)]
    if len(x) < 2 or np.isnan(x).any():
        return np.array([np.nan])
    v = np.var(x, ddof=1)
    return np.array([np.sqrt(v) if node.fname == "sd" else v])


@rfunc("quantile")
def _f_quantile(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "probs", "na.rm", "names", "type"])
    x = _num(vals["x"]).astype(float)
    if "na.rm" in vals and _bool(vals["na.rm"]):
        x = x[~np.isnan(x)]
    probs = _num(vals.get("probs", np.array([0, .25, .5, .75, 1.0]))).astype(float)
    return np.quantile(x, probs, method="linear")


@rfunc("length")
def _f_length(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x"])
    x = vals["x"]
    if isinstance(x, RMatrix):
        return np.array([x.data.size], dtype=np.int64)
    return np.array([rlen(x)], dtype=np.int64)


@rfunc("nlevels")
def _f_nlevels(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x"])
    x = vals["x"]
    return np.array([x.nlevels if isinstance(x, Factor) else 0], dtype=np.int64)


@rfunc("range")
def _f_range(ctx, node):
    vals, rest, _ = _evargs(ctx, node, ["x", "...", "na.rm"], dots=True)
    x = _num(vals["x"]).astype(float)
    if "na.rm" in vals and _bool(vals["na.rm"]):
        x = x[~np.isnan(x)]
    return np.array([x.min(), x.max()])


@rfunc("rev")
def _f_rev(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x"])
    x = vals["x"]
    n = rlen(x)
    from ._rtypes import take
    return take(x, np.arange(n)[::-1])


@rfunc("diff")
def _f_diff(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "lag", "differences"])
    x = _num(vals["x"])
    lag = int(_scalar(vals.get("lag", np.array([1]))))
    return x[lag:] - x[:-lag]


@rfunc("pmin", "pmax")
def _f_pminmax(ctx, node):
    vals, rest, _ = _evargs(ctx, node, ["...", "na.rm"], dots=True)
    arrays = [_num(v).astype(float).reshape(-1) for _, v in rest]
    narm = "na.rm" in vals and _bool(vals["na.rm"])
    n = max(len(a) for a in arrays)
    arrays = [np.resize(a, n) for a in arrays]
    st = np.vstack(arrays)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if narm:
            r = np.nanmin(st, axis=0) if node.fname == "pmin" else np.nanmax(st, axis=0)
        else:
            r = st.min(axis=0) if node.fname == "pmin" else st.max(axis=0)
    all_int = all(isinstance(v, np.ndarray) and v.dtype.kind in "iu" for _, v in rest)
    return r.astype(np.int64) if all_int else r


@rfunc("is.na")
def _f_isna(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x"])
    x = vals["x"]
    if isinstance(x, RMatrix):
        return Logical(np.isnan(x.data).reshape(-1, order="F"))
    return Logical(isna(x))


@rfunc("is.finite", "is.infinite", "is.nan")
def _f_isfinite(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x"])
    x = vals["x"]
    if isinstance(x, (Character, Factor)):
        return Logical(np.zeros(rlen(x), bool))
    v = _num(x).astype(float).reshape(-1)
    fn = {"is.finite": np.isfinite, "is.infinite": np.isinf, "is.nan": np.isnan}[node.fname]
    return Logical(fn(v))


@rfunc("ifelse")
def _f_ifelse(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["test", "yes", "no"])
    t = _as_logical(vals["test"])
    n = len(t)
    yes, no = vals["yes"], vals["no"]
    if isinstance(yes, (Character, Factor)) or isinstance(no, (Character, Factor)):
        y = np.resize(as_character(yes).values if not isinstance(yes, Factor) else as_character(yes).values, n)
        nn = np.resize(as_character(no).values, n)
        out = np.where(t.values, y, nn).astype(object)
        out[t.na] = None
        return Character(list(out))
    if isinstance(yes, Logical) and isinstance(no, Logical):
        yv, yn = np.resize(yes.values, n), np.resize(yes.na, n)
        nv, nnn = np.resize(no.values, n), np.resize(no.na, n)
        return Logical(np.where(t.values, yv, nv), np.where(t.values, yn, nnn) | t.na)
    y = np.resize(_num(yes).astype(float).reshape(-1), n)
    nv = np.resize(_num(no).astype(float).reshape(-1), n)
    out = np.where(t.values, y, nv)
    out[t.na] = np.nan
    both_int = all(isinstance(v, np.ndarray) and v.dtype.kind in "iu" for v in (yes, no))
    if both_int and not t.na.any():
        return out.astype(np.int64)
    return out


@rfunc("xor")
def _f_xor(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "y"])
    A, B = _as_logical(vals["x"]), _as_logical(vals["y"])
    av, bv = _recycle(A.values, B.values)
    an, bn = _recycle(A.na, B.na)
    return Logical(av != bv, an | bn)


@rfunc("as.numeric", "as.double")
def _f_asnum(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "..."], dots=True)
    x = vals["x"]
    if isinstance(x, Character):
        r = as_double(x)
        if np.isnan(r).sum() > sum(v is None for v in x.values):
            warnings.warn("NAs introduced by coercion")
        return r
    return as_double(x)


@rfunc("as.integer")
def _f_asint(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "..."], dots=True)
    d = as_double(vals["x"])
    if np.isnan(d).any():
        return np.trunc(d)
    return np.trunc(d).astype(np.int64)


@rfunc("as.logical")
def _f_aslogical(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "..."], dots=True)
    return _as_logical(vals["x"])


@rfunc("as.character")
def _f_aschar(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "..."], dots=True)
    return as_character(vals["x"])


@rfunc("as.vector", "c", "unname", "drop", "as.matrix")
def _f_c(ctx, node):
    if node.fname in ("as.vector", "unname", "drop", "as.matrix"):
        vals, _, _ = _evargs(ctx, node, ["x", "mode"])
        x = vals["x"]
        if node.fname == "as.matrix":
            return x if isinstance(x, RMatrix) else RMatrix(_num(x).astype(float))
        if isinstance(x, RMatrix) and node.fname != "unname":
            if node.fname == "drop" and x.ncol == 1:
                return x.data[:, 0]
            if node.fname == "as.vector":
                return x.data.reshape(-1, order="F")
        if isinstance(x, Factor) and node.fname == "as.vector":
            return as_character(x)
        return x
    parts = [ctx.eval(a) for n, a in node.args]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if any(isinstance(p, (Character, Factor)) for p in parts):
        if all(isinstance(p, Factor) for p in parts):
            levels = []
            for p in parts:
                for l in p.labels():
                    pass
                for l in p.levels:
                    if l not in levels:
                        levels.append(l)
            lab = np.concatenate([p.labels() for p in parts])
            return as_factor(Character(list(lab)), levels=levels)
        vals = []
        for p in parts:
            vals.extend(as_character(p).values if not isinstance(p, Factor) else
                        as_double(p).astype(object))
        return Character(vals)
    if all(isinstance(p, Logical) for p in parts):
        return Logical(np.concatenate([p.values for p in parts]), np.concatenate([p.na for p in parts]))
    arrs = [_num(p).reshape(-1) if not isinstance(p, RMatrix) else p.data.reshape(-1, order="F") for p in parts]
    if all(a.dtype.kind in "iu" for a in arrs):
        return np.concatenate(arrs)
    return np.concatenate([a.astype(float) for a in arrs])


@rfunc("rep")
def _f_rep(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "times", "each", "length.out"])
    x = vals["x"]
    n = rlen(x)
    idx = np.arange(n)
    if "each" in vals:
        idx = np.repeat(idx, int(_scalar(vals["each"])))
    if "times" in vals:
        t = _num(vals["times"]).astype(np.int64).reshape(-1)
        idx = np.repeat(idx, t) if len(t) == len(idx) and len(t) > 1 else np.tile(idx, int(t[0]))
    if "length.out" in vals:
        idx = np.resize(idx, int(_scalar(vals["length.out"])))
    from ._rtypes import take
    return take(x, idx)


@rfunc("seq", "seq.int")
def _f_seq(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["from", "to", "by", "length.out", "along.with"])
    if "along.with" in vals:
        return np.arange(1, rlen(vals["along.with"]) + 1, dtype=np.int64)
    fr = float(_scalar(vals.get("from", np.array([1]))))
    if "to" in vals and "by" in vals:
        to = float(_scalar(vals["to"]))
        by = float(_scalar(vals["by"]))
        n = int(math.floor((to - fr) / by + 1e-10)) + 1
        out = fr + by * np.arange(n)
    elif "length.out" in vals:
        lo = int(_scalar(vals["length.out"]))
        if "to" in vals:
            out = np.linspace(fr, float(_scalar(vals["to"])), lo)
        else:
            by = float(_scalar(vals.get("by", np.array([1]))))
            out = fr + by * np.arange(lo)
    else:
        to = float(_scalar(vals.get("to", np.array([1]))))
        step = 1 if to >= fr else -1
        out = fr + step * np.arange(int(math.floor(abs(to - fr) + 1e-10)) + 1)
    if all(float(v).is_integer() for v in out) and ("by" not in vals or float(_scalar(vals["by"])).is_integer()):
        if float(fr).is_integer():
            return out.astype(np.int64)
    return out


@rfunc("seq_len")
def _f_seqlen(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["length.out"])
    return np.arange(1, int(_scalar(vals["length.out"])) + 1, dtype=np.int64)


@rfunc("seq_along")
def _f_seqalong(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["along.with"])
    return np.arange(1, rlen(vals["along.with"]) + 1, dtype=np.int64)


@rfunc("paste", "paste0")
def _f_paste(ctx, node):
    params = ["...", "sep", "collapse"] if node.fname == "paste" else ["...", "collapse"]
    vals, rest, _ = _evargs(ctx, node, params, dots=True)
    sep = _scalar(vals["sep"]) if "sep" in vals else (" " if node.fname == "paste" else "")
    parts = [as_character(v).values for _, v in rest if v is not None]
    parts = [["NA" if s is None else s for s in p] for p in parts]
    if not parts:
        return Character([])
    n = max(len(p) for p in parts)
    out = [sep.join(str(p[i % len(p)]) for p in parts) for i in range(n)]
    if "collapse" in vals:
        return Character([_scalar(vals["collapse"]).join(out)])
    return Character(out)


@rfunc("nchar")
def _f_nchar(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "type"])
    return np.array([2 if s is None else len(s) for s in as_character(vals["x"]).values], dtype=np.int64)


@rfunc("toupper", "tolower")
def _f_case(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x"])
    up = node.fname == "toupper"
    return Character([None if s is None else (s.upper() if up else s.lower()) for s in as_character(vals["x"]).values])


@rfunc("substr", "substring")
def _f_substr(ctx, node):
    p = ["x", "start", "stop"] if node.fname == "substr" else ["text", "first", "last"]
    vals, _, _ = _evargs(ctx, node, p)
    s = as_character(vals[p[0]]).values
    st = np.resize(_num(vals[p[1]]).astype(int), len(s))
    en = np.resize(_num(vals.get(p[2], np.array([1000000]))).astype(int), len(s))
    return Character([None if x is None else x[max(a, 1) - 1:b] for x, a, b in zip(s, st, en)])


@rfunc("grepl")
def _f_grepl(ctx, node):
    import re
    vals, _, _ = _evargs(ctx, node, ["pattern", "x", "ignore.case", "perl", "fixed", "useBytes"])
    pat = _scalar(vals["pattern"])
    fixed = "fixed" in vals and _bool(vals["fixed"])
    flags = re.IGNORECASE if "ignore.case" in vals and _bool(vals["ignore.case"]) else 0
    xs = as_character(vals["x"]).values
    if fixed:
        return Logical([s is not None and pat in s for s in xs])
    rx = re.compile(pat, flags)
    return Logical([s is not None and rx.search(s) is not None for s in xs])


@rfunc("I")
def _f_I(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x"])
    return vals["x"]


@rfunc("offset")
def _f_offset(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["object"])
    return vals["object"]


@rfunc("identity", "force", "invisible")
def _f_identity(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x"])
    return vals["x"]


# -- factors -----------------------------------------------------------------

def _factor(x, levels=None, labels=None, exclude="NA", ordered=None, contrasts=None) -> Factor:
    if ordered is None:
        ordered = x.ordered if isinstance(x, Factor) else False
    if levels is not None:
        levs = levels if isinstance(levels, list) else list(as_character(levels).values)
        f = as_factor(x if not isinstance(x, Factor) else as_character(x), levels=levs, ordered=ordered)
    else:
        f = as_factor(x, ordered=ordered)
    if exclude is not None and exclude != "NA":
        ex = set(as_character(exclude).values) if not isinstance(exclude, set) else exclude
        keep = [l for l in f.levels if l not in ex]
        f = as_factor(Character(list(f.labels())), levels=keep, ordered=ordered)
    if labels is not None:
        labs = list(as_character(labels).values)
        if len(labs) == 1 and f.nlevels > 1:
            labs = ["%s%d" % (labs[0], i + 1) for i in range(f.nlevels)]
        if len(labs) != f.nlevels:
            raise RError("invalid 'labels'; length %d should be 1 or %d" % (len(labs), f.nlevels))
        uniq = []
        for l in labs:
            if l not in uniq:
                uniq.append(l)
        mapping = np.array([uniq.index(l) for l in labs] + [-1], dtype=np.int64)
        codes = mapping[np.where(f.codes < 0, len(labs), f.codes)]
        f = Factor(codes, uniq, ordered)
    return Factor(f.codes, f.levels, bool(ordered), contrasts)


@rfunc("factor")
def _f_factor(ctx, node):
    vals, _, raw = _evargs(ctx, node, ["x", "levels", "labels", "exclude", "ordered", "nmax"])
    x = vals.get("x", Character([]))
    ordered = _bool(vals["ordered"]) if "ordered" in vals else None
    exclude = "NA"
    if "exclude" in vals:
        exclude = vals["exclude"]
        if isinstance(exclude, Logical) and exclude.na.all():
            exclude = "NA"
        elif exclude is None:
            exclude = None
    return _factor(x, vals.get("levels"), vals.get("labels"), exclude, ordered)


@rfunc("ordered")
def _f_ordered(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "levels", "labels"])
    return _factor(vals["x"], vals.get("levels"), vals.get("labels"), ordered=True)


@rfunc("as.factor")
def _f_asfactor(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x"])
    x = vals["x"]
    if isinstance(x, Factor):
        return x
    return as_factor(x)


@rfunc("as.ordered")
def _f_asordered(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x"])
    x = vals["x"]
    f = x if isinstance(x, Factor) else as_factor(x)
    return Factor(f.codes, f.levels, True, f.contrasts)


@rfunc("relevel")
def _f_relevel(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "ref", "..."], dots=True)
    x = vals["x"]
    f = x if isinstance(x, Factor) else as_factor(x)
    if f.ordered:
        raise RError("'relevel' only for unordered factors")
    ref = vals["ref"]
    if isinstance(ref, Character):
        r = ref.values[0]
        if r not in f.levels:
            raise RError("'ref' must be an existing level")
        ri = f.levels.index(r)
    else:
        ri = int(_scalar(ref)) - 1
        if ri < 0 or ri >= f.nlevels:
            raise RError("ref = %d must be in 1L:%d" % (ri + 1, f.nlevels))
    order = [ri] + [i for i in range(f.nlevels) if i != ri]
    inv = np.empty(f.nlevels + 1, dtype=np.int64)
    inv[order] = np.arange(f.nlevels)
    inv[-1] = -1
    codes = inv[np.where(f.codes < 0, f.nlevels, f.codes)]
    return Factor(codes, [f.levels[i] for i in order], False, None)


@rfunc("droplevels")
def _f_droplevels(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "..."], dots=True)
    x = vals["x"]
    f = x if isinstance(x, Factor) else as_factor(x)
    d = f.drop_unused()
    return Factor(d.codes, d.levels, f.ordered, None)


@rfunc("levels")
def _f_levels(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x"])
    x = vals["x"]
    return Character(x.levels) if isinstance(x, Factor) else None


def _interaction(factors: List[Factor], sep=".", lex_order=False, drop=False) -> Factor:
    ay = None
    ly = None
    for i in range(len(factors) - 1, -1, -1):
        x = factors[i]
        if drop:
            x = x.drop_unused()
        ax = x.codes.copy()
        lx = x.levels
        if ay is None:
            ay, ly = ax, list(lx)
            continue
        nx, ny = len(lx), len(ly)
        na = (ax < 0) | (ay < 0)
        if lex_order:
            ay = ay + ny * ax
            ly = [a + sep + b for a in lx for b in ly]
        else:
            ay = ay * nx + ax
            ly = [a + sep + b for b in ly for a in lx]
        ay = np.where(na, -1, ay)
        if drop:
            present = np.unique(ay[ay >= 0])
            ly = [ly[k] for k in present]
            remap = -np.ones(nx * ny, dtype=np.int64)
            remap[present] = np.arange(len(present))
            ay = np.where(ay < 0, -1, remap[np.where(ay < 0, 0, ay)])
        # merge duplicated level names
        seen = {}
        new_codes = np.arange(len(ly))
        uniq = []
        for k, l in enumerate(ly):
            if l in seen:
                new_codes[k] = seen[l]
            else:
                seen[l] = len(uniq)
                new_codes[k] = len(uniq)
                uniq.append(l)
        if len(uniq) != len(ly):
            ay = np.where(ay < 0, -1, new_codes[np.where(ay < 0, 0, ay)])
            ly = uniq
    return Factor(ay, ly, False)


@rfunc("interaction")
def _f_interaction(ctx, node):
    vals, rest, _ = _evargs(ctx, node, ["...", "drop", "sep", "lex.order"], dots=True)
    facs = [as_factor(v) if not isinstance(v, Factor) else v for _, v in rest]
    if not facs:
        raise RError("No factors specified")
    return _interaction(facs, sep=_scalar(vals["sep"]) if "sep" in vals else ".",
                        lex_order=_bool(vals["lex.order"]) if "lex.order" in vals else False,
                        drop=_bool(vals["drop"]) if "drop" in vals else False)


def _formatC(x: float, digits: int) -> str:
    s = "%.*g" % (digits, x)
    return s


@rfunc("cut")
def _f_cut(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "breaks", "labels", "include.lowest", "right", "dig.lab",
                                     "ordered_result"])
    x = _num(vals["x"]).astype(float)
    breaks = _num(vals["breaks"]).astype(float).reshape(-1)
    right = _bool(vals["right"]) if "right" in vals else True
    incl = _bool(vals["include.lowest"]) if "include.lowest" in vals else False
    dig = int(_scalar(vals["dig.lab"])) if "dig.lab" in vals else 3
    ordered = _bool(vals["ordered_result"]) if "ordered_result" in vals else False
    if len(breaks) == 1:
        nb = int(breaks[0] + 1)
        if nb < 3:
            raise RError("invalid number of intervals")
        finite = x[~np.isnan(x)]
        rx = (finite.min(), finite.max())
        dx = rx[1] - rx[0]
        if dx == 0:
            dx = abs(rx[0]) if rx[0] != 0 else 1
            breaks = np.linspace(rx[0] - dx / 1000, rx[1] + dx / 1000, nb)
        else:
            breaks = np.linspace(rx[0], rx[1], nb)
            breaks[0] = rx[0] - dx / 1000
            breaks[-1] = rx[1] + dx / 1000
    else:
        breaks = np.sort(breaks)
        nb = len(breaks)
    if len(np.unique(breaks)) != nb:
        raise RError("'breaks' are not unique")
    labels = vals.get("labels")
    codes_only = False
    if labels is None:
        ok = False
        for d in range(dig, max(12, dig) + 1):
            ch = [_formatC(0 + b, d) for b in breaks]
            if all(ch[i + 1] != ch[i] for i in range(nb - 1)):
                ok = True
                break
        if ok:
            labs = [("(" if right else "[") + ch[i] + "," + ch[i + 1] + ("]" if right else ")") for i in range(nb - 1)]
            if incl:
                if right:
                    labs[0] = "[" + labs[0][1:]
                else:
                    labs[-1] = labs[-1][:-1] + "]"
        else:
            labs = ["Range_%d" % (i + 1) for i in range(nb - 1)]
    elif isinstance(labels, Logical) and not labels.values[0]:
        codes_only = True
        labs = None
    else:
        labs = list(as_character(labels).values)
        if len(labs) != nb - 1:
            raise RError("number of intervals and length of 'labels' differ")
    # .bincode
    lo, hi = breaks[:-1], breaks[1:]
    if right:
        code = np.searchsorted(breaks, x, side="left")
        inside = (x > breaks[0]) & (x <= breaks[-1])
        if incl:
            inside |= x == breaks[0]
            code = np.where(x == breaks[0], 1, code)
    else:
        code = np.searchsorted(breaks, x, side="right")
        inside = (x >= breaks[0]) & (x < breaks[-1])
        if incl:
            inside |= x == breaks[-1]
            code = np.where(x == breaks[-1], nb - 1, code)
    code = np.where(inside & ~np.isnan(x), code, 0)
    if codes_only:
        out = code.astype(float)
        out[code == 0] = np.nan
        return out
    return Factor(code - 1, labs, ordered)


# -- contrasts and C() ---------------------------------------------------------

def _contr_call(name, args, kwargs):
    fn = CONTRAST_FUNCTIONS[name]
    a = []
    for v in args:
        if isinstance(v, Character):
            a.append(list(v.values))
        elif isinstance(v, Factor):
            a.append(list(v.levels))
        else:
            s = _num(v).reshape(-1)
            a.append(int(s[0]) if s.size == 1 else [format_num(t) for t in s])
    kw = {}
    for k, v in kwargs.items():
        if k == "scores":
            kw[k] = _num(v).astype(float)
        elif k == "contrasts":
            kw[k] = _bool(v)
        elif k == "sparse":
            continue
        else:
            kw[k] = _scalar(v)
    return RMatrix(fn(*a, **kw).matrix, fn(*a, **kw).colnames)


def _make_contr_fn(name):
    def f(ctx, node):
        args = [ctx.eval(a) for n, a in node.args if n is None]
        kwargs = {n: ctx.eval(a) for n, a in node.args if n is not None}
        cm = CONTRAST_FUNCTIONS[name]
        res = _contr_call(name, args, kwargs)
        return res
    return f


for _cn in CONTRAST_FUNCTIONS:
    FUNCTIONS[_cn] = _make_contr_fn(_cn)


def set_contrasts(f: Factor, value, how_many=None) -> Factor:
    """``contrasts(f, how.many) <- value``."""
    if f.nlevels < 2:
        raise RError("contrasts can be applied only to factors with 2 or more levels")
    if isinstance(value, str):
        cm = value
        if how_many is not None:
            cm = contrast_from_spec(value, f.levels, how_many)
    elif value is None:
        cm = None
    else:
        cm = contrast_from_spec(value, f.levels, how_many)
    return Factor(f.codes, f.levels, f.ordered, cm)


@rfunc("C")
def _f_C(ctx, node):
    m, rest = match_args(["object", "contr", "how.many", "..."], node.args, "C", dots=True)
    obj = ctx.eval(m["object"])
    if isinstance(obj, Logical):
        obj = as_factor(obj, levels=["FALSE", "TRUE"])
    f = obj if isinstance(obj, Factor) else as_factor(obj)
    if f.nlevels == 0:
        raise RError("object not interpretable as a factor")
    contr = None
    if "contr" in m:
        cn = m["contr"]
        if isinstance(cn, Sym) and cn.name in ("poly", "helmert", "sum", "treatment", "SAS"):
            contr = normalize_contrast_name(cn.name)
        else:
            contr = ctx.eval(cn)
            if isinstance(contr, Character):
                contr = contr.values[0]
    else:
        contr = OPTIONS["contrasts"][1 if f.ordered else 0]
    how_many = int(_scalar(ctx.eval(m["how.many"]))) if "how.many" in m else None
    if how_many is None and not rest:
        if isinstance(contr, RFunction):
            contr = contr.fn
        if isinstance(contr, RMatrix):
            contr = ContrastMatrix(contr.data, f.levels, contr.colnames)
        return set_contrasts(f, contr)
    extra = {n: ctx.eval(a) for n, a in rest if n is not None}
    fname = contr if isinstance(contr, str) else (contr.name if isinstance(contr, RFunction) else None)
    if fname is not None:
        mat = _contr_call(fname, [Character(f.levels)], extra)
        contr = ContrastMatrix(mat.data, f.levels, mat.colnames)
    elif isinstance(contr, RMatrix):
        contr = ContrastMatrix(contr.data, f.levels, contr.colnames)
    elif callable(contr):
        contr = contr(f.nlevels)
    return set_contrasts(f, contr, how_many)


# -- matrices --------------------------------------------------------------------

@rfunc("cbind")
def _f_cbind(ctx, node):
    cols = []
    names = []
    for n, a in node.args:
        if n in ("deparse.level",):
            continue
        v = ctx.eval(a)
        if v is None:
            continue
        if isinstance(v, RMatrix):
            cols.append(v.data)
            names.extend(v.colnames if v.colnames is not None else [""] * v.ncol)
        else:
            if isinstance(v, (Character, Factor)):
                raise RError("only numeric matrices are supported in model formulas")
            cols.append(_num(v).astype(float).reshape(-1, 1))
            names.append(n if n is not None else (a.name if isinstance(a, Sym) else ""))
    n = max(c.shape[0] for c in cols)
    cols = [np.resize(c, (n, c.shape[1])) if c.shape[0] != n else c for c in cols]
    data = np.hstack(cols)
    return RMatrix(data, names if any(names) else None)


@rfunc("matrix")
def _f_matrix(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["data", "nrow", "ncol", "byrow", "dimnames"])
    d = _num(vals["data"]).astype(float).reshape(-1)
    if "ncol" in vals:
        nc = int(_scalar(vals["ncol"]))
        nr = int(_scalar(vals["nrow"])) if "nrow" in vals else int(math.ceil(len(d) / nc))
    else:
        nr = int(_scalar(vals["nrow"])) if "nrow" in vals else len(d)
        nc = int(math.ceil(len(d) / nr))
    byrow = "byrow" in vals and _bool(vals["byrow"])
    flat = np.resize(d, nr * nc)
    return RMatrix(flat.reshape((nr, nc), order="C" if byrow else "F"))


# -- stateful transforms ---------------------------------------------------------

def _state(ctx, node, compute):
    key = deparse(node)
    if ctx.fitting or key not in ctx.state:
        st = compute()
        ctx.state[key] = st
        return st
    return ctx.state[key]


@rfunc("scale")
def _f_scale(ctx, node):
    vals, _, raw = _evargs(ctx, node, ["x", "center", "scale"])
    x = vals["x"]
    X = x.data.copy() if isinstance(x, RMatrix) else _num(x).astype(float).reshape(-1, 1)
    key = deparse(node)
    if ctx.fitting or key not in ctx.state:
        cen = vals.get("center", Logical([True]))
        if isinstance(cen, Logical):
            center = np.nanmean(X, axis=0) if cen.values[0] else None
        else:
            center = _num(cen).astype(float).reshape(-1)
        Xc = X - center if center is not None else X
        sc = vals.get("scale", Logical([True]))
        if isinstance(sc, Logical):
            if sc.values[0]:
                def f(v):
                    v = v[~np.isnan(v)]
                    return np.sqrt((v ** 2).sum() / max(1, len(v) - 1))
                scale = np.array([f(Xc[:, j]) for j in range(Xc.shape[1])])
            else:
                scale = None
        else:
            scale = _num(sc).astype(float).reshape(-1)
        ctx.state[key] = {"center": center, "scale": scale}
    st = ctx.state[key]
    if st["center"] is not None:
        X = X - st["center"]
    if st["scale"] is not None:
        X = X / st["scale"]
    attrs = {}
    if st["center"] is not None:
        attrs["scaled:center"] = st["center"]
    if st["scale"] is not None:
        attrs["scaled:scale"] = st["scale"]
    cn = x.colnames if isinstance(x, RMatrix) else None
    return RMatrix(X, cn, attrs)


@rfunc("poly")
def _f_poly(ctx, node):
    m, rest = match_args(["x", "...", "degree", "coefs", "raw", "simple"], node.args, "poly", dots=True)
    x = ctx.eval(m["x"])
    degree = int(_scalar(ctx.eval(m["degree"]))) if "degree" in m else 1
    raw = _bool(ctx.eval(m["raw"])) if "raw" in m else False
    extra = [(n, ctx.eval(a)) for n, a in rest]
    if len(extra) == 1 and extra[0][0] is None and rlen(extra[0][1]) == 1:
        degree = int(_scalar(extra[0][1]))
        extra = []
    if extra or isinstance(x, RMatrix):
        xs = [x] + [v for _, v in extra]
        if isinstance(x, RMatrix) and not extra:
            xs = [x.data[:, j] for j in range(x.ncol)]
        return _polym(ctx, node, xs, degree, raw)
    xv = _num(x).astype(float).reshape(-1)
    if raw:
        Z = np.power.outer(xv, np.arange(1, degree + 1, dtype=float))
        return RMatrix(Z, [str(i) for i in range(1, degree + 1)], {"degree": list(range(1, degree + 1))})
    if "coefs" in m:
        raise RError("pass 'coefs' through model specs; explicit coefs are not supported")
    key = deparse(node)
    if ctx.fitting or key not in ctx.state:
        Z, alpha, norm2 = poly_fit(xv, degree)
        ctx.state[key] = {"alpha": alpha, "norm2": norm2, "degree": degree}
    else:
        st = ctx.state[key]
        Z = poly_predict(xv, degree, st["alpha"], st["norm2"])
    st = ctx.state[key]
    return RMatrix(Z, [str(i) for i in range(1, degree + 1)],
                   {"coefs": {"alpha": st["alpha"], "norm2": st["norm2"]}, "degree": list(range(1, degree + 1))})


def _polym(ctx, node, xs, degree, raw):
    import itertools
    nd = len(xs)
    key = deparse(node)
    fitting = ctx.fitting or key not in ctx.state
    bases = []
    states = []
    for j, x in enumerate(xs):
        xv = _num(x).astype(float).reshape(-1)
        if raw:
            Z = np.power.outer(xv, np.arange(1, degree + 1, dtype=float))
            states.append(None)
        elif fitting:
            Z, alpha, norm2 = poly_fit(xv, degree)
            states.append((alpha, norm2))
        else:
            alpha, norm2 = ctx.state[key][j]
            Z = poly_predict(xv, degree, alpha, norm2)
        bases.append(np.column_stack([np.ones(len(xv)), Z]))
    if fitting:
        ctx.state[key] = states
    # expand.grid(rep(list(0:degree), nd)) with first varying fastest
    grid = list(itertools.product(range(degree + 1), repeat=nd))
    grid = [tuple(reversed(g)) for g in grid]
    combos = [g for g in grid if 0 < sum(g) <= degree]
    cols = []
    names = []
    for g in combos:
        c = np.ones(len(bases[0]))
        for j, d in enumerate(g):
            c = c * bases[j][:, d]
        cols.append(c)
        names.append(".".join(str(d) for d in g))
    return RMatrix(np.column_stack(cols), names)


@rfunc("polym")
def _f_polym(ctx, node):
    m, rest = match_args(["...", "degree", "coefs", "raw"], node.args, "polym", dots=True)
    degree = int(_scalar(ctx.eval(m["degree"]))) if "degree" in m else 1
    raw = _bool(ctx.eval(m["raw"])) if "raw" in m else False
    xs = [ctx.eval(a) for n, a in rest]
    return _polym(ctx, node, xs, degree, raw)


def _opt_num(vals, k):
    if k not in vals or vals[k] is None:
        return None
    return _num(vals[k]).astype(float).reshape(-1)


@rfunc("bs")
def _f_bs(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "df", "knots", "degree", "intercept", "Boundary.knots", "warn.outside"])
    x = _num(vals["x"]).astype(float).reshape(-1)
    key = deparse(node)
    degree = int(_scalar(vals["degree"])) if "degree" in vals else 3
    intercept = _bool(vals["intercept"]) if "intercept" in vals else False
    if ctx.fitting or key not in ctx.state:
        df = _scalar(vals["df"]) if "df" in vals else None
        B, st = bs_basis(x, df=df, knots=_opt_num(vals, "knots"), degree=degree, intercept=intercept,
                         boundary_knots=_opt_num(vals, "Boundary.knots"),
                         warn_outside=_bool(vals["warn.outside"]) if "warn.outside" in vals else True)
        ctx.state[key] = st
    else:
        st = ctx.state[key]
        B, _ = bs_basis(x, knots=st["knots"], degree=st["degree"], intercept=st["intercept"],
                        boundary_knots=st["Boundary_knots"])
    return RMatrix(B, [str(i) for i in range(1, B.shape[1] + 1)], dict(st))


@rfunc("ns")
def _f_ns(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "df", "knots", "intercept", "Boundary.knots"])
    x = _num(vals["x"]).astype(float).reshape(-1)
    key = deparse(node)
    intercept = _bool(vals["intercept"]) if "intercept" in vals else False
    if ctx.fitting or key not in ctx.state:
        df = _scalar(vals["df"]) if "df" in vals else None
        B, st = ns_basis(x, df=df, knots=_opt_num(vals, "knots"), intercept=intercept,
                         boundary_knots=_opt_num(vals, "Boundary.knots"))
        ctx.state[key] = st
    else:
        st = ctx.state[key]
        B, _ = ns_basis(x, knots=st["knots"], intercept=st["intercept"], boundary_knots=st["Boundary_knots"])
    return RMatrix(B, [str(i) for i in range(1, B.shape[1] + 1)], dict(st))


@rfunc("splineDesign", "spline.des")
def _f_splinedesign(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["knots", "x", "ord", "derivs", "outer.ok", "sparse"])
    ordv = int(_scalar(vals["ord"])) if "ord" in vals else 4
    derivs = _num(vals["derivs"]).astype(int) if "derivs" in vals else [0]
    D = spline_design(_num(vals["knots"]), _num(vals["x"]), ordv, derivs,
                      _bool(vals["outer.ok"]) if "outer.ok" in vals else False)
    return RMatrix(D)


# -- distributions -----------------------------------------------------------------

@rfunc("plogis", "qlogis", "pnorm", "qnorm", "dnorm", "dlogis")
def _f_dist(ctx, node):
    vals, _, _ = _evargs(ctx, node, ["x", "mean", "sd", "lower.tail", "log.p"])
    x = _num(vals["x"]).astype(float)
    nm = node.fname
    loc = float(_scalar(vals["mean"])) if "mean" in vals else (0.0)
    sc = float(_scalar(vals["sd"])) if "sd" in vals else 1.0
    with np.errstate(all="ignore"):
        if nm == "plogis":
            return 1 / (1 + np.exp(-(x - loc) / sc))
        if nm == "qlogis":
            return loc + sc * np.log(x / (1 - x))
        if nm == "dlogis":
            e = np.exp(-(x - loc) / sc)
            return e / (sc * (1 + e) ** 2)
        import scipy.special as sp
        if nm == "pnorm":
            return sp.ndtr((x - loc) / sc)
        if nm == "qnorm":
            return loc + sc * sp.ndtri(x)
        return np.exp(-0.5 * ((x - loc) / sc) ** 2) / (sc * np.sqrt(2 * np.pi))


def register_function(name: str, fn: Callable, override: bool = False) -> None:
    """Register a Python function callable from formulas.

    ``fn`` receives numpy / pandas objects and should return an array-like.
    With ``override=True`` it replaces a built-in R function of the same name.
    """
    def wrapper(ctx, node):
        pos = []
        kw = {}
        for n, a in node.args:
            v = from_r(ctx.eval(a))
            if n is None:
                pos.append(v)
            else:
                kw[n] = v
        return to_r(fn(*pos, **kw))
    if name in FUNCTIONS and not override:
        raise ValueError("%r is a built-in R function; pass override=True to replace it" % name)
    FUNCTIONS[name] = wrapper
