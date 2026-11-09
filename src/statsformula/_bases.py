"""Contrast matrices and data dependent bases: ``contr.*``, ``poly``, ``bs``, ``ns``,
``splineDesign`` and ``scale``.  All follow the R implementations line by line."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ._linalg import LinpackQR, gram_schmidt_basis
from ._rtypes import RError, RMatrix, format_num


# ---------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------

class ContrastMatrix:
    """A contrast matrix with row names (levels) and optional column names."""

    def __init__(self, matrix, rownames: Sequence[str], colnames: Optional[Sequence[str]] = None):
        self.matrix = np.asarray(matrix, dtype=float)
        if self.matrix.ndim == 1:
            self.matrix = self.matrix.reshape(-1, 1)
        self.rownames = [str(r) for r in rownames]
        self.colnames = None if colnames is None else [str(c) for c in colnames]

    @property
    def ncol(self):
        return self.matrix.shape[1]

    def __repr__(self):
        return "ContrastMatrix(%r, rownames=%r, colnames=%r)" % (self.matrix.tolist(), self.rownames, self.colnames)

    def to_pandas(self):
        import pandas as pd
        cols = self.colnames if self.colnames is not None else list(range(1, self.ncol + 1))
        return pd.DataFrame(self.matrix, index=self.rownames, columns=cols)


def _levels_arg(n):
    """Return (levels, n) for a contr.* ``n`` argument (count or level names)."""
    if isinstance(n, (int, np.integer)) or (isinstance(n, float) and float(n).is_integer()):
        return None, int(n)
    if isinstance(n, np.ndarray) and n.dtype.kind in "fiu" and n.size == 1:
        return None, int(n.reshape(-1)[0])
    levs = [str(x) for x in (list(n) if not isinstance(n, str) else [n])]
    return levs, len(levs)


def contr_treatment(n, base: int = 1, contrasts: bool = True) -> ContrastMatrix:
    levels, k = _levels_arg(n)
    if levels is None:
        if k > 1:
            levels = [str(i) for i in range(1, k + 1)]
        else:
            raise RError("not enough degrees of freedom to define contrasts")
    m = np.eye(k)
    if not contrasts:
        return ContrastMatrix(m, levels, levels)
    if k < 2:
        raise RError("contrasts not defined for %d degrees of freedom" % (k - 1))
    base = int(base)
    if base < 1 or base > k:
        raise RError("baseline group number out of range")
    keep = [j for j in range(k) if j != base - 1]
    return ContrastMatrix(m[:, keep], levels, [levels[j] for j in keep])


def contr_SAS(n, contrasts: bool = True) -> ContrastMatrix:
    levels, k = _levels_arg(n)
    return contr_treatment(n, base=k, contrasts=contrasts)


def contr_sum(n, contrasts: bool = True) -> ContrastMatrix:
    levels, k = _levels_arg(n)
    if levels is None:
        if k > 1:
            levels = [str(i) for i in range(1, k + 1)]
        else:
            raise RError("not enough degrees of freedom to define contrasts")
    elif k <= 1:
        raise RError("not enough degrees of freedom to define contrasts")
    m = np.eye(k)
    if not contrasts:
        return ContrastMatrix(m, levels, levels)
    m = m[:, :-1]
    m[-1, :] = -1
    return ContrastMatrix(m, levels, None)


def contr_helmert(n, contrasts: bool = True) -> ContrastMatrix:
    levels, k = _levels_arg(n)
    if levels is None:
        if k > 1:
            levels = [str(i) for i in range(1, k + 1)]
        else:
            raise RError("not enough degrees of freedom to define contrasts")
    elif k <= 1:
        raise RError("not enough degrees of freedom to define contrasts")
    if not contrasts:
        return ContrastMatrix(np.eye(k), levels, levels)
    m = -np.ones((k, k - 1))
    for i in range(k):
        for j in range(k - 1):
            if j <= i - 2:
                m[i, j] = 0
            elif j == i - 1:
                m[i, j] = j + 1
    return ContrastMatrix(m, levels, None)


def _make_poly(n: int, scores: np.ndarray) -> np.ndarray:
    y = scores - scores.mean()
    X = np.power.outer(y, np.arange(n, dtype=float))
    raw = gram_schmidt_basis(X)
    return raw / np.sqrt((raw ** 2).sum(axis=0))


def contr_poly(n, scores=None, contrasts: bool = True) -> ContrastMatrix:
    levels, k = _levels_arg(n)
    if levels is None:
        levels = [str(i) for i in range(1, k + 1)]
    if k < 2:
        raise RError("contrasts not defined for %d degrees of freedom" % (k - 1))
    if k > 95:
        raise RError("orthogonal polynomials cannot be represented accurately enough for %d degrees of freedom" % (k - 1))
    sc = np.arange(1, k + 1, dtype=float) if scores is None else np.asarray(scores, dtype=float).reshape(-1)
    if len(sc) != k:
        raise RError("'scores' argument is of the wrong length")
    if len(np.unique(sc)) != len(sc):
        raise RError("'scores' must all be different numbers")
    Z = _make_poly(k, sc)
    names = ["^%d" % i for i in range(k)]
    if contrasts:
        for i, nm in enumerate([".L", ".Q", ".C"][:min(3, k - 1)]):
            names[i + 1] = nm
        return ContrastMatrix(Z[:, 1:], levels, names[1:])
    Z[:, 0] = 1
    return ContrastMatrix(Z, levels, names)


CONTRAST_FUNCTIONS = {
    "contr.treatment": contr_treatment,
    "contr.sum": contr_sum,
    "contr.helmert": contr_helmert,
    "contr.poly": contr_poly,
    "contr.SAS": contr_SAS,
}

_CONTRAST_ALIASES = {"poly": "contr.poly", "helmert": "contr.helmert", "sum": "contr.sum",
                     "treatment": "contr.treatment", "SAS": "contr.SAS"}


def normalize_contrast_name(name: str) -> str:
    return _CONTRAST_ALIASES.get(name, name)


def contrast_from_spec(spec, levels: Sequence[str], how_many: Optional[int] = None) -> ContrastMatrix:
    """Evaluate a contrast specification (``contrasts<-`` semantics)."""
    nl = len(levels)
    if isinstance(spec, str):
        fn = CONTRAST_FUNCTIONS.get(normalize_contrast_name(spec))
        if fn is None:
            raise RError("could not find function \"%s\"" % spec)
        cm = fn(list(levels))
        if how_many is not None:
            return _complete_contrast(cm, levels, how_many)
        return cm
    if callable(spec) and not isinstance(spec, ContrastMatrix):
        res = spec(list(levels))
        if not isinstance(res, ContrastMatrix):
            arr = np.asarray(res, dtype=float)
            res = ContrastMatrix(arr, levels, None)
        return _complete_contrast(res, levels, how_many)
    if isinstance(spec, ContrastMatrix):
        return _complete_contrast(spec, levels, how_many)
    if isinstance(spec, RMatrix):
        return _complete_contrast(ContrastMatrix(spec.data, levels, spec.colnames), levels, how_many)
    arr = np.asarray(spec, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = None
    try:
        import pandas as pd
        if isinstance(spec, pd.DataFrame):
            cols = [str(c) for c in spec.columns]
    except Exception:  # pragma: no cover
        pass
    return _complete_contrast(ContrastMatrix(arr, levels, cols), levels, how_many)


def _complete_contrast(value: ContrastMatrix, levels, how_many) -> ContrastMatrix:
    nlevs = len(levels)
    if value.matrix.shape[0] != nlevs:
        raise RError("wrong number of contrast matrix rows")
    n1 = nlevs - 1 if how_many is None else int(how_many)
    nc = value.ncol
    if nc < n1:
        X = np.column_stack([np.ones(nlevs), value.matrix])
        qr = LinpackQR(X)
        if qr.rank != nc + 1:
            raise RError("singular contrast matrix")
        cm = qr.qy(np.eye(nlevs))[:, 1:nlevs]
        cm[:, :nc] = value.matrix
        cm = cm[:, :n1] if cm.shape[1] > n1 else cm
        cols = None
        if value.colnames is not None:
            cols = list(value.colnames) + [""] * (n1 - nc)
        return ContrastMatrix(cm, levels, cols)
    cols = None if value.colnames is None else value.colnames[:n1]
    return ContrastMatrix(value.matrix[:, :n1], levels, cols)


# ---------------------------------------------------------------------------
# Orthogonal polynomials
# ---------------------------------------------------------------------------

def poly_fit(x: np.ndarray, degree: int):
    """Return (Z, alpha, norm2) exactly as R's ``poly`` (non-raw)."""
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        raise RError("missing values are not allowed in 'poly'")
    if degree < 1:
        raise RError("'degree' must be at least 1")
    if degree >= len(np.unique(x)):
        raise RError("'degree' must be less than number of unique points")
    xbar = x.mean()
    xc = x - xbar
    X = np.power.outer(xc, np.arange(degree + 1, dtype=float))
    Z = gram_schmidt_basis(X)
    norm2 = (Z ** 2).sum(axis=0)
    alpha = ((xc[:, None] * Z ** 2).sum(axis=0) / norm2 + xbar)[:degree]
    norm2 = np.concatenate([[1.0], norm2])
    Z = Z / np.sqrt(norm2[1:])
    return Z[:, 1:], alpha, norm2


def poly_predict(x: np.ndarray, degree: int, alpha, norm2) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    Z = np.ones((len(x), degree + 1))
    Z[:, 1] = x - alpha[0]
    for i in range(2, degree + 1):
        Z[:, i] = (x - alpha[i - 1]) * Z[:, i - 1] - (norm2[i] / norm2[i - 1]) * Z[:, i - 2]
    Z = Z / np.sqrt(np.asarray(norm2)[1:])
    return Z[:, 1:]


# ---------------------------------------------------------------------------
# B-splines (port of splines.c)
# ---------------------------------------------------------------------------

def _spline_basis(knots: np.ndarray, order: int, x: np.ndarray, derivs: Sequence[int]):
    """Vectorised port of ``spline_basis`` in R's splines.c.

    Returns the ``order`` non-zero basis values per x and the column offsets.
    """
    knots = np.asarray(knots, dtype=float)
    x = np.asarray(x, dtype=float)
    nk = len(knots)
    ordm1 = order - 1
    nx = len(x)
    der = np.asarray([int(d) for d in derivs], dtype=np.int64)
    der = der[np.arange(nx) % len(der)] if nx else der[:0]
    if np.any(der >= order):
        raise RError("derivs = %d >= ord = %d, but should be in {0,..,ord-1}" % (der.max(), order))
    # set_cursor
    curs = np.searchsorted(knots, x, side="right")
    at_end = curs >= nk
    curs = np.where(at_end, np.where(x == knots[-1], nk - 1, -1), curs)
    last = nk - order
    boundary = (curs > last) & (x == knots[last])
    curs = np.where(boundary, last, curs)
    offsets = curs - order
    val = np.full((nx, order), np.nan)
    ok = (curs >= order) & (curs <= nk - order)
    # fast path: values (derivative 0)
    m0 = ok & (der == 0)
    if m0.any():
        xs = x[m0]
        cs = curs[m0]
        rdel = np.stack([knots[cs + r] - xs for r in range(ordm1)], axis=1) if ordm1 else np.zeros((len(xs), 0))
        ldel = np.stack([xs - knots[cs - (r + 1)] for r in range(ordm1)], axis=1) if ordm1 else np.zeros((len(xs), 0))
        b = np.zeros((len(xs), order))
        b[:, 0] = 1.0
        for j in range(1, ordm1 + 1):
            saved = np.zeros(len(xs))
            for r in range(j):
                den = rdel[:, r] + ldel[:, j - 1 - r]
                nz = den != 0
                term = np.divide(b[:, r], den, out=np.zeros(len(xs)), where=nz)
                keep_zero = (~nz) & (r == 0) & (rdel[:, r] == 0.0)
                new_b = np.where(nz, saved + rdel[:, r] * term, np.where(keep_zero, b[:, r], saved))
                saved = np.where(nz, ldel[:, j - 1 - r] * term, 0.0)
                b[:, r] = new_b
            b[:, j] = saved
        val[m0] = b
    # derivatives
    for d in np.unique(der[ok & (der > 0)]):
        md = ok & (der == d)
        xs = x[md]
        cs = curs[md]
        bd = boundary[md]
        out = np.zeros((len(xs), order))
        for ii in range(order):
            a = np.zeros((len(xs), order + 1))
            a[:, ii] = 1.0
            outer = ordm1
            nd_ = int(d)
            while nd_ > 0:
                nd_ -= 1
                for idx in range(outer):
                    lpt = cs - outer + idx
                    a[:, idx] = outer * (a[:, idx + 1] - a[:, idx]) / (knots[lpt + outer] - knots[lpt])
                outer -= 1
            rdel = [knots[cs + r] - xs for r in range(outer)]
            ldel = [xs - knots[cs - (r + 1)] for r in range(outer)]
            while outer > 0:
                outer -= 1
                for inner in range(outer + 1):
                    lp = ldel[outer - inner]
                    rp = rdel[inner]
                    a[:, inner] = (a[:, inner + 1] * lp + a[:, inner] * rp) / (rp + lp)
            res = a[:, 0]
            if d == ordm1:
                res = np.where(bd, 0.0, res)
            out[:, ii] = res
        val[md] = out
    return val, offsets


def spline_design(knots, x, ord: int = 4, derivs=(0,), outer_ok: bool = False) -> np.ndarray:
    """R's ``splines::splineDesign``."""
    knots = np.sort(np.asarray(knots, dtype=float).reshape(-1))
    nk = len(knots)
    if nk <= 0:
        raise RError("must have at least 'ord' knots")
    x = np.asarray(x, dtype=float).reshape(-1)
    nx = len(x)
    derivs = [int(d) for d in np.atleast_1d(derivs)]
    if len(derivs) > nx:
        raise RError("length of 'derivs' is larger than length of 'x'")
    ord = int(ord)
    if ord > nk or ord < 1:
        raise RError("'ord' must be positive integer, at most the number of knots")
    if not outer_ok and nk < 2 * ord - 1:
        raise RError("need at least 2*ord -1 (=%d) knots" % (2 * ord - 1))
    degree = ord - 1
    ncoef = nk - ord
    need_outer = bool(np.any((x < knots[ord - 1]) | (knots[nk - degree - 1] < x)))
    in_x = np.ones(nx, dtype=bool)
    kn = knots
    if need_outer:
        if not outer_ok:
            raise RError("the 'x' data must be in the range %g to %g unless you set 'outer.ok = TRUE'" % (knots[ord - 1], knots[nk - degree - 1]))
        in_x = (knots[0] <= x) & (x <= knots[-1])
        dkn = np.diff(knots)[::-1]
        pos = np.nonzero(dkn > 0)[0]
        m = (pos[0] + 1) if len(pos) else 0
        kn = np.concatenate([np.repeat(knots[0], degree), knots, np.repeat(knots[-1], max(0, ord - m))])
    xs = x[in_x]
    vals, offs = _spline_basis(kn, ord, xs, derivs)
    design = np.zeros((nx, ncoef))
    rows = np.nonzero(in_x)[0]
    if len(rows):
        cols = offs[:, None] + np.arange(1, ord + 1)[None, :]
        if need_outer:
            cols = cols - degree
        c0 = cols - 1
        valid = (c0 >= 0) & (c0 < ncoef)
        rr = np.broadcast_to(rows[:, None], c0.shape)
        design[rr[valid], c0[valid]] = vals[valid]
    return design


def _quantile7(x, probs):
    return np.quantile(np.asarray(x, dtype=float), probs, method="linear")


def _shove_knots(knots, boundary, is_ns):
    import warnings
    if not len(knots):
        return knots
    lr = [knots.min() in boundary, knots.max() in boundary]
    if not any(lr):
        return knots
    knots = knots.copy()
    aE = aE2 = False
    if lr[0]:
        piv = boundary[0]
        i = knots == piv
        aE = i.all()
        if aE:
            if is_ns:
                raise RError("all interior knots match left boundary knot")
            warnings.warn("all interior knots match left boundary knot")
        else:
            knots[i] = knots[i] + (knots[knots > piv].min() - piv) / 8
    if lr[1]:
        piv = boundary[1]
        i = knots == piv
        aE2 = i.all()
        if aE2:
            if is_ns:
                raise RError("all interior knots match right boundary knot")
            warnings.warn("all interior knots match right boundary knot")
        else:
            knots[i] = knots[i] - (piv - knots[knots < piv].max()) / 8
    if is_ns or not ((lr[0] and aE) or (lr[1] and aE2)):
        warnings.warn("shoving 'interior' knots matching boundary knots to inside")
    return knots


def bs_basis(x, df=None, knots=None, degree=3, intercept=False, boundary_knots=None, warn_outside=True):
    """R's ``splines::bs``; returns (matrix, state)."""
    import warnings
    x = np.asarray(x, dtype=float).reshape(-1)
    degree = int(degree)
    ord_ = 1 + degree
    if ord_ <= 1:
        raise RError("'degree' must be integer >= 1")
    nax = np.isnan(x)
    xx = x[~nax]
    bk_given = boundary_knots is not None
    if bk_given:
        bk = np.sort(np.asarray(boundary_knots, dtype=float).reshape(-1))
        ol = xx < bk[0]
        orr = xx > bk[1]
        outside = ol | orr
    else:
        bk = np.array([xx.min(), xx.max()])
        outside = np.zeros(len(xx), dtype=bool)
        ol = orr = outside
    mk = df is not None and knots is None
    if mk:
        nIk = int(df) - ord_ + (1 - int(bool(intercept)))
        if nIk < 0:
            nIk = 0
            warnings.warn("'df' was too small; have used %d" % (ord_ - (1 - int(bool(intercept)))))
        if nIk > 0:
            probs = np.linspace(0, 1, nIk + 2)[1:-1]
            knots = _quantile7(xx[~outside], probs)
        else:
            knots = np.zeros(0)
    else:
        knots = np.zeros(0) if knots is None else np.asarray(knots, dtype=float).reshape(-1)
        if not np.all(np.isfinite(knots)):
            raise RError("non-finite knots")
    if mk and len(knots):
        knots = _shove_knots(knots, bk, False)
    Aknots = np.sort(np.concatenate([np.repeat(bk, ord_), knots]))
    if outside.any():
        if warn_outside:
            warnings.warn("some 'x' values beyond boundary knots may cause ill-conditioned bases")
        derivs = list(range(degree + 1))
        from math import gamma
        scalef = np.array([gamma(i) for i in range(1, ord_ + 1)])
        basis = np.zeros((len(xx), len(Aknots) - degree - 1))
        e = 0.25
        if ol.any():
            kp = (1 - e) * bk[0] + e * Aknots[ord_]
            xl = np.column_stack([np.ones(ol.sum()), np.power.outer(xx[ol] - kp, np.arange(1, degree + 1))])
            tt = spline_design(Aknots, np.repeat(kp, ord_), ord_, derivs)
            basis[ol, :] = xl @ (tt / scalef[:, None])
        if orr.any():
            kp = (1 - e) * bk[1] + e * Aknots[len(Aknots) - ord_ - 1]
            xr = np.column_stack([np.ones(orr.sum()), np.power.outer(xx[orr] - kp, np.arange(1, degree + 1))])
            tt = spline_design(Aknots, np.repeat(kp, ord_), ord_, derivs)
            basis[orr, :] = xr @ (tt / scalef[:, None])
        inside = ~outside
        if inside.any():
            basis[inside, :] = spline_design(Aknots, xx[inside], ord_)
    else:
        basis = spline_design(Aknots, xx, ord_)
    if not intercept:
        basis = basis[:, 1:]
    out = np.full((len(x), basis.shape[1]), np.nan)
    out[~nax, :] = basis
    state = dict(degree=degree, knots=knots, Boundary_knots=bk, intercept=bool(intercept))
    return out, state


def ns_basis(x, df=None, knots=None, intercept=False, boundary_knots=None):
    """R's ``splines::ns``; returns (matrix, state)."""
    import warnings
    x = np.asarray(x, dtype=float).reshape(-1)
    nax = np.isnan(x)
    xx = x[~nax]
    bk_given = boundary_knots is not None
    if bk_given:
        bk = np.sort(np.asarray(boundary_knots, dtype=float).reshape(-1))
        ol = xx < bk[0]
        orr = xx > bk[1]
        outside = ol | orr
    else:
        if len(xx) == 1:
            bk = xx[0] * np.array([7.0, 9.0]) / 8
        else:
            bk = np.array([xx.min(), xx.max()])
        outside = np.zeros(len(xx), dtype=bool)
        ol = orr = outside
    mk = df is not None and knots is None
    if mk:
        nIk = int(df) - 1 - int(bool(intercept))
        if nIk < 0:
            nIk = 0
            warnings.warn("'df' was too small; have used %d" % (1 + int(bool(intercept))))
        if nIk > 0:
            probs = np.linspace(0, 1, nIk + 2)[1:-1]
            knots = _quantile7(xx[~outside], probs)
        else:
            knots = np.zeros(0)
    else:
        knots = np.zeros(0) if knots is None else np.asarray(knots, dtype=float).reshape(-1)
        if not np.all(np.isfinite(knots)):
            raise RError("non-finite knots")
        nIk = len(knots)
    if mk and len(knots):
        knots = _shove_knots(knots, bk, True)
    Aknots = np.sort(np.concatenate([np.repeat(bk, 4), knots]))
    nIk = len(knots)
    if outside.any():
        basis = np.zeros((len(xx), nIk + 4))
        if ol.any():
            kp = bk[0]
            xl = np.column_stack([np.ones(ol.sum()), xx[ol] - kp])
            tt = spline_design(Aknots, [kp, kp], 4, [0, 1])
            basis[ol, :] = xl @ tt
        if orr.any():
            kp = bk[1]
            xr = np.column_stack([np.ones(orr.sum()), xx[orr] - kp])
            tt = spline_design(Aknots, [kp, kp], 4, [0, 1])
            basis[orr, :] = xr @ tt
        inside = ~outside
        if inside.any():
            basis[inside, :] = spline_design(Aknots, xx[inside], 4)
    else:
        basis = spline_design(Aknots, xx, 4)
    const = spline_design(Aknots, bk, 4, [2, 2])
    if not intercept:
        const = const[:, 1:]
        basis = basis[:, 1:]
    qr = LinpackQR(const.T)
    basis = qr.qty(basis.T).T[:, 2:]
    out = np.full((len(x), basis.shape[1]), np.nan)
    out[~nax, :] = basis
    state = dict(knots=knots, Boundary_knots=bk, intercept=bool(intercept))
    return out, state
