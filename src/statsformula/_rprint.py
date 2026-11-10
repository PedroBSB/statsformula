"""R's number formatting and coefficient-table printing, for the ``summary()`` methods.

Ports of ``format()`` for doubles (``formatReal``/``scientific`` in R's
``format.c``), ``formatC``, ``format.pval``, ``printCoefmat`` and the printing of
named vectors and character matrices (``print.default``), so that summaries
print like R's.
"""

import math
from typing import List, Optional, Sequence

import numpy as np

KP_MAX = 22
_TBL = [10.0 ** i for i in range(KP_MAX + 1)]
EPS = float(np.finfo(float).eps)
SIGNIF_LEGEND = "0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1"


def _scientific(x: float, digits: int):
    """(neg, kpower, nsig, roundingwidens) of R's ``scientific()``."""
    if x == 0.0:
        return 0, 0, 1, False
    neg = 1 if x < 0 else 0
    r = -x if neg else x
    R = digits
    kp = int(math.floor(math.log10(r))) - R + 1
    r_prec = r
    if abs(kp) < 10:
        if kp > 0:
            r_prec /= _TBL[kp]
        elif kp < 0:
            r_prec *= _TBL[-kp]
    elif kp <= -308:
        r_prec = (r * 1e303) / 10.0 ** (kp + 303)
    else:
        r_prec /= 10.0 ** kp
    if r_prec < _TBL[R - 1]:
        r_prec *= 10.0
        kp -= 1
    alpha = float(np.rint(r_prec))
    nsig = R
    for _ in range(R):
        alpha /= 10.0
        if alpha == math.floor(alpha):
            nsig -= 1
        else:
            break
    if nsig == 0 and R > 0:
        nsig = 1
        kp += 1
    kpower = kp + R - 1
    rgt = min(max(R - kpower, 0), KP_MAX)
    fuzz = 0.5 / _TBL[rgt]
    widens = 0 < kpower <= KP_MAX and r < _TBL[kpower] - fuzz
    return neg, kpower, nsig, widens


def format_real_spec(x: Sequence[float], digits: int = 7, nsmall: int = 0, scipen: int = 0):
    """(width, decimals, exponent digits) chosen by R's ``formatReal``; e == 0: fixed."""
    neg = 0
    rgt = mxl = mxsl = mxns = -(10 ** 9)
    mnl = 10 ** 9
    naw = 0
    for v in x:
        v = float(v)
        if not math.isfinite(v):
            naw = max(naw, 4 if v == -math.inf else 3 if (v == math.inf or math.isnan(v)) else 2)
            continue
        neg_i, kpower, nsig, widens = _scientific(v, digits)
        left = kpower + 1
        if widens:
            left -= 1
        sleft = neg_i + (1 if left <= 0 else left)
        right = nsig - left
        if neg_i:
            neg = 1
        rgt = max(rgt, right)
        mxl = max(mxl, left)
        mnl = min(mnl, left)
        mxsl = max(mxsl, sleft)
        mxns = max(mxns, nsig)
    if digits == 0:
        rgt = 0
    if mxl < 0:
        mxsl = 1 + neg
    if rgt < 0:
        rgt = 0
    wF = mxsl + rgt + (rgt != 0)
    e = 2 if (mxl > 100 or mnl <= -99) else 1
    if mxns != -(10 ** 9):
        d = mxns - 1
        w = neg + (d > 0) + d + 4 + e
        if wF <= w + scipen:
            e = 0
            if nsmall > rgt:
                rgt = nsmall
                wF = mxsl + rgt + (rgt != 0)
            d = rgt
            w = wF
    else:
        w = d = e = 0
    return max(w, naw), d, e


def _encode(v: float, w: int, d: int, e: int, na: str = "NA") -> str:
    if math.isnan(v):
        return na.rjust(w)
    if math.isinf(v):
        return ("Inf" if v > 0 else "-Inf").rjust(w)
    if e:
        return (("%#.*e" if d > 0 else "%.*e") % (d, v)).rjust(w)
    return ("%.*f" % (d, v)).rjust(w)


def format_num(x, digits: int = 7, nsmall: int = 0, na: str = "NA") -> List[str]:
    """``format(x, digits = digits)`` for a numeric vector (common width)."""
    x = [float(v) for v in np.ravel(np.asarray(x, dtype=float))]
    w, d, e = format_real_spec(x, digits, nsmall)
    w = max(w, len(na) if any(math.isnan(v) for v in x) else 0)
    return [_encode(v, w, d, e, na) for v in x]


def format_c(x: float, digits: Optional[int] = None, width: int = 0, flag: str = "") -> str:
    """``formatC(x, digits, width, flag)`` for a double (C ``%g``)."""
    if digits is None:
        s = "%g" % x
    else:
        s = "%.*g" % (digits, x)
    return s.ljust(width) if "-" in flag else s.rjust(width)


def signif(x: float, digits: int) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return float("%.*e" % (digits - 1, x))


def format_pval(pv, digits: int = 5, eps: float = EPS, na: str = "NA") -> List[str]:
    """``format.pval(pv, digits, eps)``."""
    pv = [float(v) for v in np.ravel(np.asarray(pv, dtype=float))]
    out: List[Optional[str]] = [None] * len(pv)
    ok = [i for i, v in enumerate(pv) if not math.isnan(v)]
    is0 = [i for i in ok if pv[i] < eps]
    rest = [i for i in ok if pv[i] >= eps]
    rr = {}
    if rest:
        expo = {i: math.floor(math.log10(pv[i] if pv[i] > 0 else 1e-50)) for i in rest}
        fixp = [i for i in rest if expo[i] >= -3 or (expo[i] == -4 and digits > 1)]
        sci = [i for i in rest if i not in fixp]
        for grp in (fixp, sci):
            if grp:
                for i, s in zip(grp, format_num([pv[i] for i in grp], digits)):
                    rr[i] = s
        for i in rest:
            out[i] = rr[i]
    if is0:
        dg = max(1, digits - 2)
        if rest:
            nc = max(len(s) for s in rr.values())
            if dg > 1 and dg + 6 > nc:
                dg = max(1, nc - 7)
            sep = "" if (dg == 1 and nc <= 6) else " "
        else:
            sep = "" if dg == 1 else " "
        s = "<" + sep + format_num([eps], dg)[0]
        for i in is0:
            out[i] = s
    for i, v in enumerate(pv):
        if math.isnan(v):
            out[i] = na
    return [s for s in out]  # type: ignore[misc]


def _round(v: float, d: int) -> float:
    if not math.isfinite(v):
        return v
    return float(round(v, d))


def signif_stars(pv) -> List[str]:
    out = []
    for p in pv:
        if p is None or math.isnan(p):
            out.append("")
        elif p <= 0.001:
            out.append("***")
        elif p <= 0.01:
            out.append("**")
        elif p <= 0.05:
            out.append("*")
        elif p <= 0.1:
            out.append(".")
        else:
            out.append(" ")
    return out


def print_matrix(cells: List[List[str]], rownames: Sequence[str], colnames: Sequence[str],
                 right: bool = True, width: int = 80) -> str:
    """``print.default`` of a character matrix (``quote = FALSE``)."""
    rlabw = max([len(r) for r in rownames] + [0])
    nc = len(colnames)
    w = [max([len(colnames[j])] + [len(row[j]) for row in cells]) for j in range(nc)]
    lines = []
    j0 = 0
    while j0 < nc or (nc == 0 and j0 == 0):
        tot = rlabw
        j1 = j0
        while j1 < nc and (j1 == j0 or tot + 1 + w[j1] <= width):
            tot += 1 + w[j1]
            j1 += 1
        just = (lambda s, n: s.rjust(n)) if right else (lambda s, n: s.ljust(n))
        lines.append(" " * rlabw + "".join(" " + just(colnames[j], w[j]) for j in range(j0, j1)))
        for r, row in zip(rownames, cells):
            lines.append(r.ljust(rlabw) + "".join(" " + just(row[j], w[j]) for j in range(j0, j1)))
        if nc == 0:
            break
        j0 = j1
    return "\n".join(lines)


def print_named(values, names: Sequence[str], digits: int = 7, width: int = 80) -> str:
    """``print()`` of a named numeric vector."""
    s = format_num(values, digits)
    w = max([len(v) for v in s] + [len(n) for n in names])
    per = max(1, width // (w + 1))
    lines = []
    for i in range(0, len(s), per):
        lines.append("".join(n.rjust(w) + " " for n in names[i:i + per]))
        lines.append("".join(v.rjust(w) + " " for v in s[i:i + per]))
    return "\n".join(lines)


def print_coefmat(x, rownames: Sequence[str], colnames: Sequence[str], digits: int = 4,
                  signif_stars_: bool = True, cs_ind: Optional[Sequence[int]] = None,
                  tst_ind: Optional[Sequence[int]] = None, has_pvalue: Optional[bool] = None,
                  dig_tst: Optional[int] = None, na_print: str = "NA", legend: bool = True) -> str:
    """R's ``printCoefmat()`` (column indices are 0-based)."""
    xm = np.array(x, dtype=float).reshape(len(rownames), len(colnames))
    nr, nc = xm.shape
    if has_pvalue is None:
        has_pvalue = nc >= 4 and colnames[-1][:3] in ("Pr(", "p-v")
    if dig_tst is None:
        dig_tst = max(1, min(5, digits - 1))
    k = nc - int(has_pvalue) - (1 if tst_ind is None else len(tst_ind))
    if cs_ind is None:
        cs_ind = list(range(k))
    if tst_ind is None:
        tst_ind = [k]
    Cf = [["" for _ in range(nc)] for _ in range(nr)]
    ina = np.isnan(xm)
    if len(cs_ind):
        cs = xm[:, cs_ind]
        acs = np.abs(cs)
        fin = np.isfinite(acs)
        if fin.any():
            nz = acs[fin & (acs != 0)]
            digmin = 1 + (np.floor(np.log10([nz.min(), nz.max()])).astype(int) if nz.size else np.array([0]))
            # round(coef.se, max(1L, digits - digmin)): one number of decimals for the block
            dround = int(max(1, np.max(digits - digmin)))
            flat = cs.ravel(order="F")
            rounded = [_round(v, dround) for v in flat]
            s = format_num(rounded, digits)
            for idx, val in enumerate(s):
                i, j = idx % nr, idx // nr
                Cf[i][cs_ind[j]] = val
    for j in tst_ind:
        s = format_num([_round(v, dig_tst) for v in xm[:, j]], digits)
        for i in range(nr):
            Cf[i][j] = s[i]
    pcol = nc - 1 if has_pvalue else None
    for j in range(nc):
        if j in cs_ind or j in tst_ind or j == pcol:
            continue
        s = format_num(xm[:, j], digits)
        for i in range(nr):
            Cf[i][j] = s[i]
    # entries shown as zero that are not (or vice versa) get digits - 1
    ok_cols = [j for j in range(nc) if j not in tst_ind and j != pcol]
    redo = []
    for j in ok_cols:
        for i in range(nr):
            if ina[i, j]:
                continue
            try:
                shown0 = float(Cf[i][j]) == 0
            except ValueError:
                continue
            if (xm[i, j] == 0) != shown0:
                redo.append((i, j))
    if redo:
        s = format_num([xm[i, j] for i, j in redo], max(1, digits - 1))
        for (i, j), v in zip(redo, s):
            Cf[i][j] = v
    for i in range(nr):
        for j in range(nc):
            if ina[i, j]:
                Cf[i][j] = na_print
    cols = list(colnames)
    stars = False
    if has_pvalue:
        okp = [i for i in range(nr) if not ina[i, pcol]]
        if okp:
            s = format_pval([xm[i, pcol] for i in okp], dig_tst)
            for i, v in zip(okp, s):
                Cf[i][pcol] = v
            pv = xm[:, pcol]
            stars = signif_stars_ and bool(np.any(pv[okp] < 0.1))
            if stars:
                sy = signif_stars(pv)
                wd = max(len(v) for v in sy)
                for i in range(nr):
                    Cf[i].append(sy[i].ljust(wd))
                cols.append("")
    out = print_matrix(Cf, list(rownames), cols)
    if stars and legend:
        out += "\n---\nSignif. codes:  " + SIGNIF_LEGEND
    return out
