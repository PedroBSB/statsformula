"""mgcv smooth terms: ``s()``, ``te()``, ``ti()`` in formulas.

A port of the model-matrix side of R's *mgcv* (version 1.9): ``interpret.gam``,
``smoothCon`` (identifiability constraints absorbed exactly as mgcv does, by
Householder QR), ``PredictMat`` and the bases

* ``tp``/``ts`` thin plate regression splines (including mgcv's Lanczos
  eigen-solver, so that the basis - not just its span - matches R),
* ``cr``/``cs`` cubic regression splines, ``cc`` cyclic cubic splines,
* ``ps``/``cp`` (cyclic) P-splines, ``bs`` B-splines, ``re`` random effects,
* tensor products ``te()`` and ``ti()`` of any of the above,
* numeric and factor ``by`` variables.

The resulting design matrix is the one returned by
``model.matrix(gam(...))`` / ``predict.gam(type = "lpmatrix")`` in R.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._bases import spline_design
from ._linalg import LinpackQR
from ._parser import Call, Const, Node, Num, Str, Sym, deparse, deparse_label, is_call, parse
from ._rtypes import Character, Factor, Logical, RError, RMatrix, as_factor

SMOOTH_FUNCS = ("s", "te", "ti", "t2")
DBL_EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# smooth specifications (s, te, ti)
# ---------------------------------------------------------------------------

class SmoothSpec:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return "<SmoothSpec %s bs=%s>" % (getattr(self, "label", "?"), getattr(self, "bs", "tensor"))


def _eval_const(node: Node, ctx):
    v = ctx.eval(node)
    if isinstance(v, Character):
        return list(v.values)
    if isinstance(v, Logical):
        return [None if n else bool(b) for b, n in zip(v.values, v.na)]
    if isinstance(v, np.ndarray):
        return [float(x) for x in v.reshape(-1)]
    if v is None:
        return None
    return v


def _term_label(node: Node) -> str:
    from ._formula import RFormula
    from ._parser import call as mkcall
    t = RFormula(mkcall("~", node)).terms(allow_dot_as_name=True)
    return t.term_labels[0] if t.term_labels else deparse_label(node)


def _first(v, default=None):
    if v is None:
        return default
    if isinstance(v, list):
        return v[0] if v else default
    return v


def _nan_to_none(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def make_s_spec(node: Call, ctx) -> SmoothSpec:
    params = ["k", "fx", "bs", "m", "by", "xt", "id", "sp", "pc"]
    vars_ = [a for n, a in node.args if n is None]
    named = {n: a for n, a in node.args if n is not None}
    for n in list(named):
        if n not in params:
            cand = [p for p in params if p.startswith(n)]
            if len(cand) == 1:
                named[cand[0]] = named.pop(n)
            else:
                raise RError("unused argument (%s)" % n)
    d = len(vars_)
    if d == 0:
        raise RError("s() needs at least one covariate")
    by = deparse_label(named["by"]) if "by" in named else "NA"
    if isinstance(named.get("by"), Const) and named["by"].value == "NA":
        by = "NA"
    if by == ".":
        raise RError("by=. not allowed")
    term = [_term_label(v) for v in vars_]
    if any(t == "." for t in term):
        raise RError("s(.) not supported.")
    if len(set(term)) != d:
        raise RError("Repeated variables as arguments of a smooth are not permitted")
    k = _first(_eval_const(named["k"], ctx), -1) if "k" in named else -1
    k = int(round(k))
    fx = bool(_first(_eval_const(named["fx"], ctx), False)) if "fx" in named else False
    bs = _first(_eval_const(named["bs"], ctx), "tp") if "bs" in named else "tp"
    m = _eval_const(named["m"], ctx) if "m" in named else None
    if isinstance(m, list):
        m = [_nan_to_none(x) for x in m]
        if all(x is None for x in m):
            m = None
    xt = ctx.eval(named["xt"]) if "xt" in named else None
    idv = _first(_eval_const(named["id"], ctx)) if "id" in named else None
    label = "s(" + ",".join(term) + ")"
    return SmoothSpec(kind="s", term=term, bs_dim=k, fixed=fx, dim=d, p_order=m, by=by, label=label,
                      xt=xt, id=idv, bs=bs)


def make_te_spec(node: Call, ctx, inter=False) -> SmoothSpec:
    params = ["k", "bs", "m", "d", "by", "fx", "np", "xt", "id", "sp", "pc", "mc"]
    vars_ = [a for n, a in node.args if n is None]
    named = {n: a for n, a in node.args if n is not None}
    dim = len(vars_)
    by = deparse_label(named["by"]) if "by" in named else "NA"
    if isinstance(named.get("by"), Const) and named["by"].value == "NA":
        by = "NA"
    term = [_term_label(v) for v in vars_]
    dd = _eval_const(named["d"], ctx) if "d" in named else None
    if dd is None or any(x is None or (isinstance(x, float) and math.isnan(x)) for x in dd):
        n_bases = dim
        dd = [1] * dim
    else:
        dd = [int(round(x)) for x in dd]
        if any(x <= 0 for x in dd) or sum(dd) != dim:
            warnings.warn("something wrong with argument d.")
            n_bases = dim
            dd = [1] * dim
        else:
            n_bases = len(dd)
    kk = _eval_const(named["k"], ctx) if "k" in named else None
    if kk is None or any(x is None or (isinstance(x, float) and math.isnan(x)) for x in kk):
        kk = [5 ** x for x in dd]
    else:
        kk = [int(round(x)) for x in kk]
        ok = True
        if any(x < 3 for x in kk):
            ok = False
            warnings.warn("one or more supplied k too small - reset to default")
        if len(kk) == 1 and ok:
            kk = kk * n_bases
        elif len(kk) != n_bases:
            ok = False
        if not ok:
            kk = [5 ** x for x in dd]
    fx = _eval_const(named["fx"], ctx) if "fx" in named else [False]
    if len(fx) == 1:
        fx = fx * n_bases
    bs = _eval_const(named["bs"], ctx) if "bs" in named else ["cr"]
    if len(bs) == 1:
        bs = bs * n_bases
    if len(bs) != n_bases:
        warnings.warn("bs wrong length and ignored.")
        bs = ["cr"] * n_bases
    bs = ["tp" if (dd[i] > 1 and bs[i] in ("cr", "cs", "ps", "cp")) else bs[i] for i in range(n_bases)]
    m = _eval_const(named["m"], ctx) if "m" in named else [None]
    m = [_nan_to_none(x) for x in m]
    if len(m) == 1:
        m = m * n_bases
    if len(m) != n_bases:
        warnings.warn("m wrong length and ignored.")
        m = [0] * n_bases
    m = [0 if (x is not None and x < 0) else x for x in m]
    if len(set(term)) != dim:
        raise RError("Repeated variables as arguments of a smooth are not permitted")
    xt = ctx.eval(named["xt"]) if "xt" in named else None
    margins = []
    j = 0
    for i in range(n_bases):
        tm = term[j:j + dd[i]]
        margins.append(SmoothSpec(kind="s", term=tm, bs_dim=kk[i], fixed=False, dim=dd[i],
                                  p_order=None if m[i] is None else [m[i]], by="NA",
                                  label="s(" + ",".join(tm) + ")", xt=xt, id=None, bs=bs[i]))
        j += dd[i]
    npv = bool(_first(_eval_const(named["np"], ctx), True)) if "np" in named else True
    label = ("ti(" if inter else "te(") + ",".join(term) + ")"
    mc = _eval_const(named["mc"], ctx) if (inter and "mc" in named) else None
    return SmoothSpec(kind="te", margin=margins, term=term, by=by, fx=fx, label=label, dim=dim, np=npv,
                      id=None, inter=inter, mc=mc, bs="tensor", xt=xt)


def make_spec(node: Call, ctx) -> SmoothSpec:
    fn = node.fname
    if fn == "s":
        return make_s_spec(node, ctx)
    if fn == "te":
        return make_te_spec(node, ctx, inter=False)
    if fn == "ti":
        return make_te_spec(node, ctx, inter=True)
    if fn == "t2":
        raise RError("t2() smooths are not supported yet; use te() or ti()")
    raise RError("not a smooth: %s" % fn)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _getvar(data, name):
    v = data.get(name)
    if v is None:
        alt = deparse(parse(name))
        v = data.get(alt)
    if v is None:
        raise RError("object '%s' not found" % name)
    if isinstance(v, RMatrix):
        if v.ncol == 1:
            return v.data[:, 0]
        raise RError("matrix arguments to smooths (summation convention) are not supported")
    if isinstance(v, Logical):
        return v.to_float()
    if isinstance(v, Character):
        return as_factor(v)
    return v


def _num(v):
    if isinstance(v, Factor):
        raise RError("factor used where a numeric smooth covariate was expected")
    return np.asarray(v, dtype=float).reshape(-1)


def r_mean(x):
    """R's mean.default: sequential sum, then a second (correction) pass."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = np.cumsum(x)[-1] / n
    if np.isfinite(m):
        m = m + np.cumsum(x - m)[-1] / n
    return m


def r_colmeans(X):
    """R's colMeans: sequential summation down each column."""
    X = np.asarray(X, dtype=float)
    return np.cumsum(X, axis=0)[-1] / X.shape[0]


def _quantile7(x, probs):
    return np.quantile(np.asarray(x, dtype=float), probs, method="linear")


def _r_unique(x):
    """unique() keeping first-appearance order."""
    _, idx = np.unique(x, return_index=True)
    return x[np.sort(idx)]


# ---------------------------------------------------------------------------
# cubic regression splines (crspl / getFS in mgcv.c)
# ---------------------------------------------------------------------------

def _get_F(xk):
    n = len(xk)
    h = np.diff(xk)
    n2 = n - 2
    D = np.zeros((n2, n))
    i = np.arange(n2)
    D[i, i] = 1 / h[:-1]
    D[i, i + 2] = 1 / h[1:]
    D[i, i + 1] = -D[i, i] - D[i, i + 2]
    B = np.diag((h[:-1] + h[1:]) / 3)
    if n2 > 1:
        off = h[1:-1] / 6
        B[np.arange(n2 - 1), np.arange(1, n2)] = off
        B[np.arange(1, n2), np.arange(n2 - 1)] = off
    BinvD = np.linalg.solve(B, D)
    Mx = np.zeros((n, n))  # row j: second derivative at knot j as linear map of values
    Mx[1:n - 1, :] = BinvD
    return Mx


def _crspl(x, xk, Mx):
    x = np.asarray(x, dtype=float)
    nk = len(xk)
    n = len(x)
    X = np.zeros((n, nk))
    kmin, kmax = xk[0], xk[-1]
    lo = x < kmin
    hi = x > kmax
    inside = ~(lo | hi)
    if inside.any():
        xi = x[inside]
        j = np.clip(np.searchsorted(xk, xi, side="left") - 1, 0, nk - 2)
        xj, xj1 = xk[j], xk[j + 1]
        h = xj1 - xj
        ajm = xj1 - xi
        ajp = xi - xj
        cjm = (ajm * (ajm * ajm / h - h)) / 6
        cjp = (ajp * (ajp * ajp / h - h)) / 6
        rows = cjm[:, None] * Mx[j] + cjp[:, None] * Mx[j + 1]
        r = np.arange(len(xi))
        rows[r, j] += ajm / h
        rows[r, j + 1] += ajp / h
        X[inside] = rows
    if lo.any():
        xi = x[lo]
        h = xk[1] - kmin
        xik = xi - kmin
        cjm = -xik * h / 3
        cjp = -xik * h / 6
        rows = cjm[:, None] * Mx[0] + cjp[:, None] * Mx[1]
        rows[:, 0] += 1 - xik / h
        rows[:, 1] += xik / h
        X[lo] = rows
    if hi.any():
        xi = x[hi]
        j = nk - 1
        h = kmax - xk[j - 1]
        xik = xi - kmax
        cjm = xik * h / 6
        cjp = xik * h / 3
        rows = cjm[:, None] * Mx[j - 1] + cjp[:, None] * Mx[j]
        rows[:, nk - 2] += -xik / h
        rows[:, nk - 1] += 1 + xik / h
        X[hi] = rows
    return X


class CRSmooth:
    def __init__(self, spec, data, knots):
        if len(spec.term) != 1:
            raise RError("Basis only handles 1D smooths")
        x = _num(_getvar(data, spec.term[0]))
        nk = spec.bs_dim if spec.bs_dim >= 0 else 10
        if nk < 3:
            nk = 3
            warnings.warn("basis dimension, k, increased to minimum possible\n")
        xu = _r_unique(x)
        if len(xu) < nk:
            raise RError("%s has insufficient unique values to support %d knots: reduce k." % (spec.term[0], nk))
        k = None if knots is None else knots.get(spec.term[0])
        if k is None:
            k = _quantile7(xu, np.linspace(0, 1, nk))
        k = np.asarray(k, dtype=float)
        if len(k) != nk:
            raise RError("number of supplied knots != k for a cr smooth")
        self.xp = k
        self.Mx = _get_F(k)
        self.bs_dim = nk
        self.X = _crspl(x, k, self.Mx)

    def predict(self, data):
        return _crspl(_num(_getvar(data, self.term[0])), self.xp, self.Mx)


# ---------------------------------------------------------------------------
# cyclic cubic splines
# ---------------------------------------------------------------------------

def _place_knots(x, nk):
    x = np.unique(x)
    n = len(x)
    if nk > n:
        raise RError("more knots than unique data values is not allowed")
    if nk < 2:
        raise RError("too few knots")
    if nk == 2:
        return np.array([x[0], x[-1]])
    delta = (n - 1) / (nk - 1)
    i = np.arange(1, nk - 1)
    lbi = np.floor(delta * i).astype(int) + 1
    frac = delta * i + 1 - lbi
    knot = np.zeros(nk)
    knot[-1] = x[-1]
    knot[0] = x[0]
    knot[1:-1] = x[lbi - 1] * (1 - frac) + x[lbi] * frac
    return knot


def _cwrap(x0, x1, x):
    x = x.copy()
    h = x1 - x0
    hi = x > x1
    if hi.any():
        x[hi] = x0 + np.mod(x[hi] - x1, h)
    lo = x < x0
    if lo.any():
        x[lo] = x1 - np.mod(x0 - x[lo], h)
    return x


class CCSmooth:
    def __init__(self, spec, data, knots):
        if len(spec.term) != 1:
            raise RError("Basis only handles 1D smooths")
        x = _num(_getvar(data, spec.term[0]))
        nk = spec.bs_dim if spec.bs_dim >= 0 else 10
        if nk < 4:
            nk = 4
            warnings.warn("basis dimension, k, increased to minimum possible\n")
        k = None if knots is None else knots.get(spec.term[0])
        if k is None:
            k = _place_knots(x, nk)
        k = np.asarray(k, dtype=float)
        if len(k) == 2:
            k = _place_knots(np.concatenate([k, x]), nk)
        if len(k) != nk:
            raise RError("number of supplied knots != k for a cc smooth")
        n = len(k)
        h = np.diff(k)
        nn = n - 1
        B = np.zeros((nn, nn))
        D = np.zeros((nn, nn))
        B[0, 0] = (h[nn - 1] + h[0]) / 3
        B[0, 1] = h[0] / 6
        B[0, nn - 1] = h[nn - 1] / 6
        D[0, 0] = -(1 / h[0] + 1 / h[nn - 1])
        D[0, 1] = 1 / h[0]
        D[0, nn - 1] = 1 / h[nn - 1]
        for i in range(1, nn - 1):
            B[i, i - 1] = h[i - 1] / 6
            B[i, i] = (h[i - 1] + h[i]) / 3
            B[i, i + 1] = h[i] / 6
            D[i, i - 1] = 1 / h[i - 1]
            D[i, i] = -(1 / h[i - 1] + 1 / h[i])
            D[i, i + 1] = 1 / h[i]
        B[nn - 1, nn - 2] = h[nn - 2] / 6
        B[nn - 1, nn - 1] = (h[nn - 2] + h[nn - 1]) / 3
        B[nn - 1, 0] = h[nn - 1] / 6
        D[nn - 1, nn - 2] = 1 / h[nn - 2]
        D[nn - 1, nn - 1] = -(1 / h[nn - 2] + 1 / h[nn - 1])
        D[nn - 1, 0] = 1 / h[nn - 1]
        self.BD = np.linalg.solve(B, D)
        self.xp = k
        self.bs_dim = nk
        self.X = self._pred(x)

    def _pred(self, x):
        knots = self.xp
        BD = self.BD
        n = len(knots)
        h = np.diff(knots)
        if x.max() > knots.max() or x.min() < knots.min():
            x = _cwrap(knots.min(), knots.max(), x)
        j = np.full(len(x), 0)
        jj = np.searchsorted(knots, x, side="left")  # first i with knots[i] >= x
        j = np.clip(jj, 1, n - 1) + 1  # 1-based index i with x <= knots[i] (smallest)
        j1 = j - 1
        hj = j - 1
        j = np.where(j == n, 1, j)
        I = np.eye(n - 1)
        a = (knots[j1] - x)  # knots[j1+1] in R 1-based == knots[j1] 0-based
        b = (x - knots[j1 - 1])
        hh = h[hj - 1]
        X = (BD[j1 - 1] * (a ** 3 / (6 * hh))[:, None] + BD[j - 1] * (b ** 3 / (6 * hh))[:, None]
             - BD[j1 - 1] * (hh * a / 6)[:, None] - BD[j - 1] * (hh * b / 6)[:, None]
             + I[j1 - 1] * (a / hh)[:, None] + I[j - 1] * (b / hh)[:, None])
        return X

    def predict(self, data):
        return self._pred(_num(_getvar(data, self.term[0])))


# ---------------------------------------------------------------------------
# P-splines, cyclic P-splines and B-splines
# ---------------------------------------------------------------------------

def _p_order(spec, default_second):
    m = spec.p_order
    if m is None:
        m = [None]
    m = list(m)
    if len(m) == 1:
        m = m * 2
    return m


def _c_spline_des(x, knots, ord_=4, derivs=0):
    nk = len(knots)
    if ord_ < 2:
        raise RError("order too low")
    if nk < ord_:
        raise RError("too few knots")
    knots = np.sort(knots)
    k1 = knots[0]
    if x.min() < k1 or x.max() > knots[-1]:
        raise RError("x out of range")
    xc = knots[nk - ord_]
    knots = np.concatenate([k1 - (knots[-1] - knots[nk - ord_:nk - 1]), knots])
    ind = x > xc
    X1 = spline_design(knots, x, ord_, [derivs], outer_ok=True)
    if ind.any():
        x2 = x[ind] - knots.max() + k1
        X2 = spline_design(knots, x2, ord_, [derivs], outer_ok=True)
        X1[ind] += X2
    return X1


class PSSmooth:
    """``ps`` (and, with ``bspline=True``, mgcv's ``bs``) bases."""

    def __init__(self, spec, data, knots, bspline=False):
        x = _num(_getvar(data, spec.term[0]))
        if len(spec.term) != 1:
            raise RError("Basis only handles 1D smooths")
        self.bspline = bspline
        if bspline:
            m = spec.p_order
            if m is None:
                m = [None]
            m = list(m)
            if len(m) == 1:
                m = [m[0], None if m[0] is None else max(0, m[0] - 1)]
            if m[0] is None:
                m = [3, 2] if m[1] is None else [m[1] + 1, m[1]]
            if m[1] is None:
                m[1] = max(0, m[0] - 1)
            m = [int(m[0]), int(m[1])]
            bs_dim = spec.bs_dim if spec.bs_dim >= 0 else max(10, m[0])
            nk = bs_dim - m[0] + 1
            if nk <= 0:
                raise RError("basis dimension too small for b-spline order")
            k = None if knots is None else knots.get(spec.term[0])
            xl, xu = x.min(), x.max()
            if k is not None and len(k) == 2:
                xl, xu = min(k), max(k)
                if xl > x.min() or xu < x.max():
                    raise RError("knot range does not include data")
            if k is not None and len(k) == 4 and len(k) < nk + 2 * m[0]:
                k = np.sort(np.asarray(k, dtype=float))
                dx = (k[3] - k[0]) / (nk - 1)
                ko = (k[0] - dx * m[0], k[3] + dx * m[0])
                k = np.concatenate([np.linspace(ko[0], k[0], m[0] + 1), np.linspace(k[1], k[2], max(0, nk - 2)),
                                    np.linspace(k[3], ko[1], m[0] + 1)])
            elif k is None or len(k) == 2:
                xr = xu - xl
                xl, xu = xl - xr * 0.001, xu + xr * 0.001
                dx = (xu - xl) / (nk - 1)
                k = np.linspace(xl - dx * m[0], xu + dx * m[0], nk + 2 * m[0])
            else:
                if len(k) != nk + 2 * m[0]:
                    raise RError("there should be %d supplied knots" % (nk + 2 * m[0]))
            self.knots = np.asarray(k, dtype=float)
            self.m = [m[0] - 1, m[1]]  # Predict.matrix.Bspline uses m-1
            self.X = spline_design(self.knots, x, m[0] + 1, [0])
            self.bs_dim = bs_dim
        else:
            m = _p_order(spec, 2)
            m = [2 if v is None else int(v) for v in m]
            bs_dim = spec.bs_dim if spec.bs_dim >= 0 else max(10, m[0] + 1)
            nk = bs_dim - m[0]
            if nk <= 0:
                raise RError("basis dimension too small for b-spline order")
            k = None if knots is None else knots.get(spec.term[0])
            xl, xu = x.min(), x.max()
            if k is not None and len(k) == 2:
                xl, xu = min(k), max(k)
                if xl > x.min() or xu < x.max():
                    raise RError("knot range does not include data")
            if k is None or len(k) == 2:
                xr = xu - xl
                xl, xu = xl - xr * 0.001, xu + xr * 0.001
                dx = (xu - xl) / (nk - 1)
                k = np.linspace(xl - dx * (m[0] + 1), xu + dx * (m[0] + 1), nk + 2 * m[0] + 2)
            elif len(k) != nk + 2 * m[0] + 2:
                raise RError("there should be %d supplied knots" % (nk + 2 * m[0] + 2))
            self.knots = np.asarray(k, dtype=float)
            self.m = m
            self.X = spline_design(self.knots, x, m[0] + 2, [0])
            self.bs_dim = bs_dim
        if (self.X.sum(axis=0) == 0).any():
            warnings.warn("there is *no* information about some basis coefficients")
        if len(np.unique(x)) < self.bs_dim:
            warnings.warn("basis dimension is larger than number of unique covariates")

    def predict(self, data):
        x = _num(_getvar(data, self.term[0]))
        m = self.m[0] + 1
        ll = self.knots[m]
        ul = self.knots[len(self.knots) - m - 1]
        m = m + 1
        n = len(x)
        ind = (x <= ul) & (x >= ll)
        if ind.all():
            return spline_design(self.knots, x, m, [0])
        D = spline_design(self.knots, np.array([ll, ll, ul, ul]), m, [0, 1, 0, 1])
        X = np.zeros((n, D.shape[1]))
        if ind.any():
            X[ind] = spline_design(self.knots, x[ind], m, [0])
        lo = x < ll
        if lo.any():
            X[lo] = np.column_stack([np.ones(lo.sum()), x[lo] - ll]) @ D[0:2]
        hi = x > ul
        if hi.any():
            X[hi] = np.column_stack([np.ones(hi.sum()), x[hi] - ul]) @ D[2:4]
        return X


class CPSmooth:
    def __init__(self, spec, data, knots):
        m = _p_order(spec, 2)
        m = [2 if v is None else int(v) for v in m]
        bs_dim = spec.bs_dim if spec.bs_dim >= 0 else max(10, m[0])
        nk = bs_dim + 1
        if nk <= m[0]:
            raise RError("basis dimension too small for b-spline order")
        x = _num(_getvar(data, spec.term[0]))
        k = None if knots is None else knots.get(spec.term[0])
        if k is None:
            x0, x1 = x.min(), x.max()
        elif len(k) == 2:
            x0, x1 = min(k), max(k)
            if x0 > x.min() or x1 < x.max():
                raise RError("knot range does not include data")
        if k is None or len(k) == 2:
            k = np.linspace(x0, x1, nk)
        elif len(k) != nk:
            raise RError("there should be %d supplied knots" % nk)
        self.knots = np.asarray(k, dtype=float)
        self.m = m
        self.bs_dim = bs_dim
        self.X = _c_spline_des(x, self.knots, m[0] + 2)

    def predict(self, data):
        x = _num(_getvar(data, self.term[0]))
        k0, k1 = self.knots.min(), self.knots.max()
        if x.min() < k0 or x.max() > k1:
            x = _cwrap(k0, k1, x)
        return _c_spline_des(x, self.knots, self.m[0] + 2)


# ---------------------------------------------------------------------------
# random effects
# ---------------------------------------------------------------------------

class RESmooth:
    def __init__(self, spec, data, knots):
        self.C = np.zeros((0, 0))
        self.levels = {}
        self.X = self._design(data, fit=True)
        self.bs_dim = self.X.shape[1]
        self.noterp = True

    def _design(self, data, fit):
        blocks = []
        for t in self.term:
            v = _getvar(data, t)
            if isinstance(v, Factor):
                if fit:
                    self.levels[t] = list(v.levels)
                    f = v
                else:
                    f = as_factor(Character(list(v.labels())), levels=self.levels[t])
                B = np.zeros((len(f), f.nlevels))
                ok = f.codes >= 0
                B[np.nonzero(ok)[0], f.codes[ok]] = 1.0
            else:
                B = np.asarray(v, dtype=float).reshape(-1, 1)
            blocks.append(B)
        X = blocks[0]
        for B in blocks[1:]:
            n = X.shape[0]
            X = np.einsum("ni,nj->nji", X, B).reshape(n, -1)
        return X

    def predict(self, data):
        X = self._design(data, fit=False)
        X[~np.isfinite(X)] = 0
        return X


# ---------------------------------------------------------------------------
# thin plate regression splines (tprs.c, Rlanczos in mat.c)
# ---------------------------------------------------------------------------

def null_space_dimension(d, m):
    if 2 * m < d + 1:
        m = 1
        while 2 * m < d + 2:
            m += 1
    M = 1
    for i in range(d):
        M *= d + m - 1 - i
    for i in range(2, d + 1):
        M //= i
    return M


def _eta_const(m, d):
    pi = math.pi
    ghalf = math.sqrt(pi)
    d2 = d // 2
    if 2 * m <= d:
        raise RError("You must have 2m>d for a thin plate spline.")
    if d % 2 == 0:
        f = -1.0 if (m + 1 + d2) % 2 else 1.0
        for _ in range(2 * m - 1):
            f /= 2
        for _ in range(d2):
            f /= pi
        for i in range(2, m):
            f /= i
        for i in range(2, m - d2 + 1):
            f /= i
    else:
        f = ghalf
        k = m - (d - 1) // 2
        for i in range(k):
            f /= -0.5 - i
        for _ in range(m):
            f /= 4
        for _ in range(d2):
            f /= pi
        f /= ghalf
        for i in range(2, m):
            f /= i
    return f


def _eta(r2, m, d, f):
    d2 = d // 2
    out = np.zeros_like(r2)
    pos = r2 > 0
    r = r2[pos]
    if d % 2 == 0:
        val = f * np.log(r) * 0.5 * r ** (m - d2)
    else:
        val = f * r ** (m - d2 - 1) * np.sqrt(r)
    out[pos] = val
    return out


def _poly_powers(M, m, d):
    idx = [0] * d
    out = []
    for _ in range(M):
        out.append(list(idx))
        s = sum(idx)
        if s < m - 1:
            idx[0] += 1
        else:
            s -= idx[0]
            idx[0] = 0
            for j in range(1, d):
                idx[j] += 1
                s += 1
                if s == m:
                    s -= idx[j]
                    idx[j] = 0
                else:
                    break
    return out


def _tps_T(X, m, d):
    M = null_space_dimension(d, m) if 2 * m > d else None
    Mm = 1
    for i in range(d):
        Mm *= d + m - 1 - i
    for i in range(2, d + 1):
        Mm //= i
    pw = _poly_powers(Mm, m, d)
    T = np.ones((X.shape[0], Mm))
    for j, p in enumerate(pw):
        for kk in range(d):
            if p[kk]:
                T[:, j] *= X[:, kk] ** p[kk]
    return T, pw


def _lanczos_biggest(A, k, tol=None):
    """Port of mgcv's Rlanczos(..., lm = -1): k largest magnitude eigenpairs."""
    import scipy.linalg.lapack as lapack
    n = A.shape[0]
    m, lm = k, 0
    eps_stop = DBL_EPS ** 0.7 if tol is None else tol
    f_check = (m + lm) // 2
    if f_check < 10:
        f_check = 10
    kk = n // 10
    if kk < 1:
        kk = 1
    if kk < f_check:
        f_check = kk
    q = np.zeros((n + 1, n))
    jran = 1
    b0 = np.zeros(n)
    for i in range(n):
        jran = (jran * 106 + 1283) % 6075
        b0[i] = jran / 6075 - 0.5
    q[0] = b0 / np.sqrt((b0 ** 2).sum())
    a = np.zeros(n)
    b = np.zeros(n)
    d = v = None
    j = 0
    jj = 0
    while jj < n:
        j = jj
        z = A @ q[j]
        xx = q[j] @ z
        a[j] = xx
        if j == 0:
            z = z - xx * q[j]
        else:
            z = z - (xx * q[j] + b[j - 1] * q[j - 1])
            for _ in range(2):
                for i in range(j + 1):
                    z = z - (z @ q[i]) * q[i]
        b[j] = np.sqrt((z ** 2).sum())
        if j < n - 1:
            q[j + 1] = z / b[j]
        if (j >= m + lm and j % f_check == 0) or j == n - 1:
            dd = a[:j + 1].copy()
            gg = b[:j].copy()
            from ._lapack import dstedc
            w, vv = dstedc(dd, gg)
            d = w[::-1].copy()
            v = vv[:, ::-1].copy()  # column kk is eigenvector kk
            normTj = max(abs(d[0]), abs(d[j]))
            err = np.abs(b[j] * v[j, :])
            if j >= m + lm:
                max_err = normTj * eps_stop
                pi = ni = 0
                converged = True
                while pi + ni < m:
                    if abs(d[pi]) >= abs(d[j - ni]):
                        if err[pi] > max_err:
                            converged = False
                            break
                        pi += 1
                    else:
                        if err[ni] > max_err:  # (sic) as in mgcv
                            converged = False
                            break
                        ni += 1
                if converged:
                    m, lm = pi, ni
                    j += 1
                    break
        jj += 1
    else:
        j = n
    Q = q[:j]
    U = np.zeros((n, m + lm))
    D = np.zeros(m + lm)
    for kq in range(m):
        D[kq] = d[kq]
        U[:, kq] = v[:j, kq] @ Q
    for kq in range(m, lm + m):
        kd = j - (lm + m - kq)
        D[kq] = d[kd]
        U[:, kq] = v[:j, kd] @ Q
    return U, D


def _QT(A):
    """mgcv QT(Q, A, fullQ=0): returns (rows u_i of Householder vectors)."""
    A = A.copy()
    Ar, Ac = A.shape
    Q = np.zeros((Ar, Ac))
    for i in range(Ar):
        L = Ac - i
        p = A[i, :L]
        mm = np.abs(p).max() if L else 0.0
        if mm:
            p = p / mm
        lsq = np.sqrt((p ** 2).sum())
        if p[L - 1] < 0.0:
            lsq = -lsq
        p[L - 1] += lsq
        g = 1 / (lsq * p[L - 1]) if lsq else 0.0
        lsq *= mm
        A[i, :L] = p
        for jr in range(i + 1, Ar):
            x = g * (p @ A[jr, :L])
            A[jr, :L] -= x * p
        Q[i, :L] = p * np.sqrt(g)
        A[i, L - 1] = -lsq
        A[i, :L - 1] = 0.0
    return Q


def _QT_full(A):
    A = A.copy()
    Ar, Ac = A.shape
    Q = np.eye(Ac)
    for i in range(Ar):
        L = Ac - i
        p = A[i, :L]
        mm = np.abs(p).max() if L else 0.0
        if mm:
            p = p / mm
        lsq = np.sqrt((p ** 2).sum())
        if p[L - 1] < 0.0:
            lsq = -lsq
        p[L - 1] += lsq
        g = 1 / (lsq * p[L - 1]) if lsq else 0.0
        lsq *= mm
        A[i, :L] = p
        for jr in range(i + 1, Ar):
            x = g * (p @ A[jr, :L])
            A[jr, :L] -= x * p
        x = g * (Q[:, :L] @ p)
        Q[:, :L] -= np.outer(x, p)
        A[i, L - 1] = -lsq
        A[i, :L - 1] = 0.0
    return Q


def _HQmult_post(C, U):
    """C <- C Q with Q = H_1 H_2 ... (HQmult(C, U, 0, 0))."""
    C = C.copy()
    for kq in range(U.shape[0]):
        u = U[kq]
        C -= np.outer(C @ u, u)
    return C


def _uniquecombs_ordered(x):
    """mgcv::uniquecombs(x, ordered = TRUE) (rows of a numeric matrix)."""
    from ._rtypes import format_num
    if x.shape[1] == 1:
        return np.unique(x[:, 0]).reshape(-1, 1)
    txt = ["*".join(format_num(v) for v in row) for row in x]
    seen = {}
    rows = []
    for i, t in enumerate(txt):
        if t not in seen:
            seen[t] = i
            rows.append((t, i))
    rows.sort(key=lambda r: r[0].encode("utf-8"))
    return x[[i for _, i in rows]]


class TPSmooth:
    def __init__(self, spec, data, knots, shrink=None):
        d = spec.dim
        xs = [_num(_getvar(data, t)) for t in spec.term]
        n = len(xs[0])
        self.shift = np.array([r_mean(x) for x in xs])
        x = np.column_stack([xs[i] - self.shift[i] for i in range(d)])
        knt = None
        if knots is not None and all(knots.get(t) is not None for t in spec.term):
            knt = np.column_stack([np.asarray(knots[t], dtype=float) - self.shift[i] for i, t in enumerate(spec.term)])
            if knt.shape[0] > n:
                knt = None
                warnings.warn("more knots than data in a tp term: knots ignored.")
        xt = spec.xt if isinstance(spec.xt, dict) else {}
        max_knots = int(xt.get("max.knots", 2000))
        seed = int(xt.get("seed", 1))
        if knt is None and n > max_knots:
            xu = _uniquecombs_ordered(x)
            if xu.shape[0] > max_knots:
                from ._rrng import r_sample
                ind = r_sample(xu.shape[0], max_knots, seed)
                knt = xu[np.asarray(ind) - 1]
        p_order = spec.p_order if spec.p_order is not None else [0]
        p_order = [0 if v is None else int(v) for v in p_order]
        m = p_order[0]
        M = null_space_dimension(d, m)
        if len(p_order) > 1 and p_order[1] == 0:
            raise RError("tp with m = c(m, 0) (drop.null) is not supported")
        def_k = [8, 27, 100]
        dd = min(d, len(def_k))
        k = spec.bs_dim if spec.bs_dim >= 0 else M + def_k[dd - 1]
        if k < M + 1:
            k = M + 1
            warnings.warn("basis dimension, k, increased to minimum possible\n")
        self.bs_dim = k
        self.p_order = m
        nk = 0 if knt is None else knt.shape[0]
        X, UZ, Xu, mm = self._setup(x, knt if nk >= k else None, m, d, n, k)
        self.X = X
        self.UZ = UZ
        self.Xu = Xu
        self.m_used = mm
        self.M = M
        self.d = d

    @staticmethod
    def _setup(x, knt, m, d, n, k):
        pts = x if knt is None else knt
        Xu, inv = np.unique(pts, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        if Xu.shape[0] < k:
            raise RError("A term has fewer unique covariate combinations than specified maximum degrees of freedom")
        if 2 * m <= d:
            m = 0
            while 2 * m < d + 2:
                m += 1
        f = _eta_const(m, d)
        diff = Xu[:, None, :] - Xu[None, :, :]
        E = _eta((diff ** 2).sum(axis=2), m, d, f)
        T, pw = _tps_T(Xu, m, d)
        M = T.shape[1]
        if k < M + 1:
            k = M + 1
        nu = Xu.shape[0]
        pure_knot = nu == k
        if pure_knot:
            Qfull = _QT_full(T.T)
            Z = _QT(T.T)
            UZ = np.zeros((nu + M, nu))
            UZ[:nu, :] = Qfull
        else:
            U, v = _lanczos_biggest(E, k)
            TU = T.T @ U
            Z = _QT(TU)
            UQ = _HQmult_post(U, Z)
            UZ = np.zeros((nu + M, k))
            UZ[:nu, :k - M] = UQ[:, :k - M]
        UZ[:nu, k - M:] = 0.0
        UZ[nu:, k - M:] = np.eye(M)
        if knt is None and not pure_knot:
            X1 = U * v[None, :]
            X1 = _HQmult_post(X1, Z)
            X1[:, k - M:] = T
            X = X1[inv]
        else:
            B = TPSmooth._tps_b(x, Xu, m, d, pw, f)
            X = B @ UZ
        w = np.sqrt(np.cumsum(X ** 2, axis=0)[-1] / X.shape[0])
        X = X / w
        UZ = UZ / w
        return X, UZ, Xu, m

    @staticmethod
    def _tps_b(x, Xu, m, d, pw, f):
        diff = x[:, None, :] - Xu[None, :, :]
        Eb = _eta((diff ** 2).sum(axis=2), m, d, f)
        P = np.ones((x.shape[0], len(pw)))
        for j, p in enumerate(pw):
            for kk in range(d):
                if p[kk]:
                    P[:, j] *= x[:, kk] ** p[kk]
        return np.hstack([Eb, P])

    def predict(self, data):
        xs = [_num(_getvar(data, t)) for t in self.term]
        x = np.column_stack([xs[i] - self.shift[i] for i in range(self.d)])
        m = self.m_used
        f = _eta_const(m, self.d)
        T, pw = _tps_T(self.Xu[:1], m, self.d)
        B = self._tps_b(x, self.Xu, m, self.d, pw, f)
        return B @ self.UZ


# ---------------------------------------------------------------------------
# tensor products
# ---------------------------------------------------------------------------

def _tensor_prod(Xs):
    X1 = Xs[-1]
    n = X1.shape[0]
    for X in reversed(Xs[:-1]):
        X1 = np.einsum("ni,nj->nij", X, X1).reshape(n, -1)
    return X1


class TensorSmooth:
    def __init__(self, spec, data, knots):
        m = len(spec.margin)
        inter = spec.inter
        if inter:
            self.mc = [True] * m if spec.mc is None else [bool(v) for v in spec.mc]
        else:
            self.mc = [False] * m
        self.margins = []
        Xm = []
        for i, ms in enumerate(spec.margin):
            if self.mc[i]:
                sm = smooth_con(ms, data, knots, absorb_cons=True)[0]
            else:
                sm = smooth_construct(ms, data, knots)
            self.margins.append(sm)
            Xm.append(sm.X)
        self.XP = [None] * m
        if spec.np:
            for i, sm in enumerate(self.margins):
                if sm.spec.dim == 1 and not getattr(sm.basis, "noterp", False) and not isinstance(sm.basis, (CRSmooth, CCSmooth, RESmooth)):
                    x = _getvar(data, sm.spec.term[0])
                    npc = Xm[i].shape[1]
                    if isinstance(x, Factor):
                        raise RError("factor margins in te() are not supported")
                    knt = np.linspace(np.min(x), np.max(x), npc)
                    pd = {sm.spec.term[0]: knt}
                    P = predict_mat(sm, pd) if self.mc[i] else sm.basis.predict(pd)
                    u, s, vt = np.linalg.svd(P)
                    if s[-1] / s[0] < DBL_EPS ** 0.66:
                        warnings.warn("reparameterization unstable for margin: not done")
                        continue
                    XP = vt.T @ (u.T[:npc] / s[:, None])
                    self.XP[i] = XP
                    Xm[i] = Xm[i] @ XP
        self.X = _tensor_prod(Xm)
        self.C = None
        if inter or any(getattr(sm, "C", None) is not None and np.asarray(sm.C).shape[0] == 0
                        for sm in self.margins if isinstance(sm, _Constructed)):
            self.C = np.zeros((0, 0))
        self.bs_dim = self.X.shape[1]

    def predict(self, data):
        Xs = []
        for i, sm in enumerate(self.margins):
            X = predict_mat(sm, data) if self.mc[i] else sm.basis.predict(data)
            if self.XP[i] is not None:
                X = X @ self.XP[i]
            Xs.append(X)
        return _tensor_prod(Xs)


_BASES = {
    "tp": TPSmooth, "ts": TPSmooth, "cr": CRSmooth, "cs": CRSmooth, "cc": CCSmooth,
    "ps": PSSmooth, "bs": lambda spec, data, knots: PSSmooth(spec, data, knots, bspline=True),
    "cp": CPSmooth, "re": RESmooth,
}


class _Constructed:
    """A constructed smooth (``smooth.construct`` result)."""

    def __init__(self, spec, basis):
        self.spec = spec
        self.basis = basis
        self.X = basis.X
        self.C = getattr(basis, "C", None)


def smooth_construct(spec: SmoothSpec, data, knots=None) -> _Constructed:
    if spec.kind == "te":
        basis = TensorSmooth.__new__(TensorSmooth)
        basis.term = spec.term
        TensorSmooth.__init__(basis, spec, data, knots)
        return _Constructed(spec, basis)
    cls = _BASES.get(spec.bs)
    if cls is None:
        raise RError("smooth basis '%s' is not supported (available: %s)" % (spec.bs, ", ".join(sorted(_BASES))))
    if isinstance(cls, type):
        basis = cls.__new__(cls)
        basis.term = spec.term
        cls.__init__(basis, spec, data, knots)
    else:
        basis = PSSmooth.__new__(PSSmooth)
        basis.term = spec.term
        PSSmooth.__init__(basis, spec, data, knots, bspline=True)
    return _Constructed(spec, basis)


# ---------------------------------------------------------------------------
# smoothCon / PredictMat
# ---------------------------------------------------------------------------

class Smooth:
    """A smooth term ready for the model matrix (one per ``by`` level)."""

    def __init__(self, con: _Constructed, label, by, by_level, qrc, indi, ncons):
        self.con = con
        self.spec = con.spec
        self.basis = con.basis
        self.label = label
        self.by = by
        self.by_level = by_level
        self.qrc = qrc
        self.indi = indi
        self.ncons = ncons
        self.X = None

    @property
    def term(self):
        return self.spec.term


def _apply_constraint(X, qrc, indi, j):
    if qrc is None or j == 0:
        return X
    k = X.shape[1]
    if indi is None:
        return qrc.qty(X.T)[j:k, :].T
    nx = len(indi)
    nz = nx - j
    X = X.copy()
    if nz > 0:
        X[:, indi[:nz]] = qrc.qty(X[:, indi].T)[j:nx, :].T
    keep = [c for c in range(k) if c not in set(indi[nz:nx])]
    return X[:, keep]


def smooth_con(spec: SmoothSpec, data, knots=None, absorb_cons=True) -> List[Smooth]:
    """Port of mgcv's ``smoothCon`` (model matrix side, ``sparse.cons = 0``)."""
    con = smooth_construct(spec, data, knots)
    X = con.X
    C = con.C
    if C is None:
        C = r_colmeans(X).reshape(1, -1)
    by = spec.by
    byv = None
    if by != "NA":
        byv = _getvar(data, by)
        if not isinstance(byv, Factor):
            bv = np.asarray(byv, dtype=float)
            if np.std(bv, ddof=1) > np.mean(bv) * DBL_EPS * 1000:
                C = np.zeros((0, 1))
    # by variables
    items = []
    if byv is not None and isinstance(byv, Factor):
        lev = list(byv.levels)
        if byv.ordered and len(lev) > 1:
            lev = lev[1:]
        lab = byv.labels()
        for L in lev:
            dum = (lab == L).astype(float)
            items.append((dum[:, None] * X, spec.label + ":" + by + L, L))
    elif byv is not None:
        bv = np.asarray(byv, dtype=float)
        if len(bv) != X.shape[0]:
            raise RError("`by' variable must be same dimension as smooth arguments")
        items.append((bv[:, None] * X, spec.label + ":" + by, None))
    else:
        items.append((X, spec.label, None))
    out = []
    qrc = None
    indi = None
    j = C.shape[0] if C.ndim == 2 else 0
    if absorb_cons and j > 0 and C.shape[1] == X.shape[1]:
        nzcols = np.nonzero(C.sum(axis=0) != 0)[0]
        nx = len(nzcols)
        if nx < C.shape[1]:
            indi = list(nzcols)
            qrc = LinpackQR(C[:, nzcols].T)
        else:
            qrc = LinpackQR(C.T)
    else:
        j = 0
    for Xi, label, lev in items:
        sm = Smooth(con, label, by, lev, qrc, indi, j)
        sm.X = _apply_constraint(Xi, qrc, indi, j) if absorb_cons else Xi
        out.append(sm)
    return out


def predict_mat(sm, data) -> np.ndarray:
    """Port of mgcv's ``PredictMat`` (``sm`` from :func:`smooth_con`)."""
    if isinstance(sm, _Constructed):
        return sm.basis.predict(data)
    X = sm.basis.predict(data)
    if sm.by != "NA":
        byv = _getvar(data, sm.by)
        if isinstance(byv, Factor):
            X = (byv.labels() == sm.by_level).astype(float)[:, None] * X
        else:
            X = np.asarray(byv, dtype=float)[:, None] * X
    if sm.qrc is not None and sm.ncons > 0:
        bad = np.isnan(X).any(axis=1)
        if bad.any():
            Y = np.full((X.shape[0], X.shape[1] - sm.ncons), np.nan)
            Y[~bad] = _apply_constraint(X[~bad], sm.qrc, sm.indi, sm.ncons)
            return Y
        X = _apply_constraint(X, sm.qrc, sm.indi, sm.ncons)
    return X


# ---------------------------------------------------------------------------
# interpret.gam and the gam model matrix
# ---------------------------------------------------------------------------

class GamFormula:
    """Result of :func:`interpret_gam` (R's ``split.gam.formula``)."""

    def __init__(self, pf, pfok, smooth_spec, fake_formula, response, fake_names, pred_names):
        self.pf = pf
        self.pfok = pfok
        self.smooth_spec = smooth_spec
        self.fake_formula = fake_formula
        self.response = response
        self.fake_names = fake_names
        self.pred_names = pred_names

    def __repr__(self):
        return "<GamFormula pf=%s smooths=%s fake=%s>" % (self.pf, [s.label for s in self.smooth_spec], self.fake_formula)


def has_smooths(formula) -> bool:
    from ._formula import RFormula, coerce_formula
    f = coerce_formula(formula)
    if not isinstance(f, RFormula):
        return False
    t = f.terms(specials=list(SMOOTH_FUNCS), allow_dot_as_name=True)
    return any(t.specials.get(s) for s in SMOOTH_FUNCS)


def interpret_gam(formula, env=None) -> GamFormula:
    """Port of mgcv's ``interpret.gam``."""
    from ._eval import EvalContext
    from ._formula import RFormula, coerce_formula
    f = coerce_formula(formula, env)
    if not isinstance(f, RFormula):
        raise RError("multi-part formulas are not supported by gam")
    tf = f.terms(specials=list(SMOOTH_FUNCS))
    terms = tf.term_labels
    nt = len(terms)
    response = deparse(f.lhs) if tf.response else None
    vtab = tf.factors
    smooth_term_idx = set()
    for s in SMOOTH_FUNCS:
        for vi in (tf.specials.get(s) or []):
            for ti in range(nt):
                if vtab[vi - 1, ti]:
                    smooth_term_idx.add(ti)
    ctx = EvalContext({}, env or getattr(f, "env", {}) or {})
    av = []
    smooth_spec = []
    for i in range(nt):
        if i in smooth_term_idx:
            node = parse(terms[i])
            smooth_spec.append(make_spec(node, ctx))
        else:
            av.append(terms[i])
    if tf.offset:
        av.extend(tf.variables[o - 1] for o in tf.offset)
    kp = len(av) + 1
    pf = (response + " ~ " if response else "~ ") + " + ".join(av)
    if not tf.intercept:
        pf = pf + "-1"
        pfok = 1 if kp > 1 else 0
    else:
        pfok = 1
        if kp == 1:
            pf = pf + " 1"
    fake = pf
    for sp in smooth_spec:
        fake = fake + " + " + "+".join(sp.term)
        if sp.by != "NA":
            fake = fake + " + " + sp.by
            av.extend(sp.term + [sp.by])
        else:
            av.extend(sp.term)
    from ._parser import all_vars
    pav = []
    for a in av:
        for v in all_vars(parse(a)):
            if v not in pav:
                pav.append(v)
    return GamFormula(RFormula(pf, f.env), pfok, smooth_spec, RFormula(fake, f.env), response, av, pav)


class GamSpec:
    """Stores everything needed to rebuild the gam model matrix on new data
    (equivalent of ``predict.gam(..., type = "lpmatrix")``)."""

    def __init__(self, gp, pterms, param_spec, smooths, columns, env):
        self.gp = gp
        self.pterms = pterms
        self.param_spec = param_spec
        self.smooths = smooths
        self.columns = columns
        self.env = env

    def get_model_matrix(self, data, output: str = "pandas"):
        import pandas as pd
        from ._model import model_frame_from_terms
        from ._formula import delete_response
        Xp = self.param_spec.get_model_matrix(data, na_action="na.pass")
        ft = delete_response(self.gp.fake_formula.terms())
        mf = model_frame_from_terms(ft, data, env=self.env, na_action="na.pass")
        datad = dict(mf.variables)
        blocks = [Xp.to_numpy()]
        for sm in self.smooths:
            blocks.append(predict_mat(sm, datad))
        X = np.hstack(blocks)
        if output == "numpy":
            return X
        return pd.DataFrame(X, index=Xp.index, columns=self.columns)


def gam_model_matrix(formula, data, knots=None, na_action: Optional[str] = None, env=None,
                     output: str = "pandas", drop_unused_levels: bool = True):
    """The model matrix of an mgcv GAM: ``model.matrix(gam(formula, data))``.

    Parametric terms follow ``model.matrix``; each smooth contributes the
    columns ``"<label>.1"``, ``"<label>.2"``, ... with mgcv's identifiability
    constraints absorbed.  The returned frame carries a ``gam_spec``
    attribute whose ``get_model_matrix(newdata)`` reproduces
    ``predict.gam(type = "lpmatrix")``.
    """
    import pandas as pd
    from ._api import ModelMatrix, _get_env, model_matrix
    from ._model import model_frame_from_terms
    env = _get_env(env, formula, 1)
    gp = interpret_gam(formula, env)
    ft = gp.fake_formula.terms(data=data)
    mf = model_frame_from_terms(ft, data, env=env, na_action=na_action, drop_unused_levels=drop_unused_levels)
    pterms = gp.pf.terms()
    Xp = model_matrix(pterms, mf, env=env)
    datad = dict(mf.variables)
    if knots is not None:
        knots = {k: np.asarray(v, dtype=float) for k, v in knots.items()}
    smooths = []
    for sp in gp.smooth_spec:
        smooths.extend(smooth_con(sp, datad, knots, absorb_cons=True))
    _check_side(smooths)
    blocks = [Xp.to_numpy()]
    cols = list(Xp.columns)
    assign = list(Xp.assign)
    for i, sm in enumerate(smooths):
        blocks.append(sm.X)
        cols.extend("%s.%d" % (sm.label, j + 1) for j in range(sm.X.shape[1]))
        assign.extend([len(pterms.term_labels) + i + 1] * sm.X.shape[1])
    X = np.hstack(blocks)
    spec = GamSpec(gp, pterms, Xp.model_spec, smooths, cols, env)
    if output == "numpy":
        return X
    mm = ModelMatrix(X, index=mf.index, columns=pd.Index(cols, dtype=object))
    object.__setattr__(mm, "assign", assign)
    object.__setattr__(mm, "contrasts", Xp.contrasts)
    object.__setattr__(mm, "model_spec", spec)
    return mm


def _check_side(smooths):
    names = []
    for sm in smooths:
        vn = [t + (sm.by if sm.by != "NA" else "") + (sm.by_level or "") for t in sm.term]
        names.extend(vn)
    if len(names) != len(set(names)):
        warnings.warn("smooths share covariates: mgcv would apply gam.side() identifiability "
                      "side-constraints, which statsformula does not implement; columns may differ from R")


# make s(), te(), ti() usable as plain functions inside formulas evaluated outside gam
def _smooth_as_variable(ctx, node):
    raise RError("%s() terms define smooths; use statsformula.gam_model_matrix() (or model_matrix, "
                 "which dispatches to it automatically)" % node.fname)


def _register():
    from ._eval import FUNCTIONS
    for nm in SMOOTH_FUNCS:
        FUNCTIONS.setdefault(nm, _smooth_as_variable)


_register()
