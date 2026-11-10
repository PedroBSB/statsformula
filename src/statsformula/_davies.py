"""Davies' (1980) algorithm for the distribution of a linear combination of chi^2
variables, and mgcv's smooth-term test (``testStat``) used by ``summary.gam``.

Translated from mgcv 1.9-3 ``src/davies.c`` (S. N. Wood's C translation of
Davies' Algol 60 code) and ``R/mgcv.r`` (``psum.chisq``, ``liu2``, ``testStat``).
"""

import math
import warnings
from typing import Optional

import numpy as np
from scipy import stats

_LN28 = math.log(2.0) / 8.0
_PI = math.pi


def _ln1(x: float, first: bool) -> float:
    return math.log1p(x) if first else math.log1p(x) - x


class _Davies:
    def __init__(self, lb, nc, n):
        self.lb = lb
        self.nc = nc
        self.n = n
        self.r = len(lb)

    def errbd(self, u, sigsq):
        lb, nc, n = self.lb, self.nc, self.n
        cx = u * sigsq
        sum1 = u * cx
        u = u * 2
        for j in range(self.r - 1, -1, -1):
            nj, lj, ncj = n[j], lb[j], nc[j]
            x = u * lj
            y = 1.0 - x
            cx += lj * (ncj / y + nj) / y
            xy = x / y
            sum1 = sum1 + ncj * xy * xy + nj * (x * xy + _ln1(-x, False))
        return math.exp(-0.5 * sum1), cx

    def ctff(self, accx, upn, mean, lmin, lmax, sigsq):
        u2 = upn
        u1 = 0.0
        c1 = mean
        rb = 2 * lmax if u2 > 0 else 2 * lmin
        while True:
            e, c2 = self.errbd(u2 / (1 + u2 * rb), sigsq)
            if e <= accx:
                break
            u1, c1, u2 = u2, c2, u2 * 2
        while True:
            if (c1 - mean) / (c2 - mean) >= 0.9:
                break
            u = (u1 + u2) * 0.5
            e, cst = self.errbd(u / (1 + u * rb), sigsq)
            if e > accx:
                u1, c1 = u, cst
            else:
                u2, c2 = u, cst
        return c2, u2

    def truncation(self, u, tausq, sigsq):
        lb, nc, n = self.lb, self.nc, self.n
        sum1 = prod2 = prod3 = 0.0
        s = 0
        sum2 = (sigsq + tausq) * u * u
        prod1 = 2 * sum2
        u = 2 * u
        for j in range(self.r):
            lj, ncj, nj = lb[j], nc[j], n[j]
            x = u * lj
            x = x * x
            sum1 += ncj * x / (1 + x)
            if x > 1:
                prod2 += nj * math.log(x)
                prod3 += nj * _ln1(x, True)
                s += nj
            else:
                prod1 += nj * _ln1(x, True)
        sum1 *= 0.5
        prod2 += prod1
        prod3 += prod1
        x = math.exp(-sum1 - 0.25 * prod2) / _PI
        y = math.exp(-sum1 - 0.25 * prod3) / _PI
        err1 = 1.0 if s == 0 else 2.0 * x / s
        err2 = 2.5 * y if prod3 > 1 else 1.0
        if err2 < err1:
            err1 = err2
        x = 0.5 * sum2
        err2 = 1.0 if x <= y else y / x
        return err1 if err1 < err2 else err2

    def findu(self, utx, accx, sigsq):
        ut = utx
        u = ut * 0.25
        if self.truncation(u, 0, sigsq) > accx:
            while self.truncation(ut, 0, sigsq) > accx:
                ut *= 4
        else:
            ut = u
            u = u / 4
            while self.truncation(u, 0, sigsq) <= accx:
                ut = u
                u = u / 4
        for a in (2.0, 1.4, 1.2, 1.1):
            u = ut / a
            if self.truncation(u, 0, sigsq) <= accx:
                ut = u
        return ut

    def integrate(self, nterm, interv, tausq, main, c, sigsq):
        lb, nc, n = self.lb, self.nc, self.n
        inpi = interv / _PI
        intl = ersm = 0.0
        for k in range(nterm, -1, -1):
            u = (k + 0.5) * interv
            sum1 = -2 * u * c
            sum2 = abs(sum1)
            sum3 = -0.5 * sigsq * u * u
            for j in range(self.r - 1, -1, -1):
                nj = n[j]
                x = 2 * lb[j] * u
                y = x * x
                sum3 = sum3 - 0.25 * nj * _ln1(y, True)
                y = nc[j] * x / (1 + y)
                z = nj * math.atan(x) + y
                sum1 += z
                sum2 += abs(z)
                sum3 += -0.5 * x * y
            x = inpi * math.exp(sum3) / u
            if not main:
                x = x * (1 - math.exp(-0.5 * tausq * u * u))
            intl += math.sin(0.5 * sum1) * x
            ersm += 0.5 * sum2 * x
        return intl, ersm

    def cfe(self, x, th):
        lb, nc, n = self.lb, self.nc, self.n
        axl = abs(x)
        sxl = -1 if x < 0 else 1
        sum1 = 0.0
        for j in range(self.r - 1, -1, -1):
            t = th[j]
            if lb[t] * sxl > 0.0:
                lj = abs(lb[t])
                axl1 = axl - lj * (n[t] + nc[t])
                axl2 = lj / _LN28
                if axl1 > axl2:
                    axl = axl1
                else:
                    if axl > axl2:
                        axl = axl2
                    sum1 = (axl - axl1) / lj
                    for k in range(j - 1, -1, -1):
                        sum1 += n[th[k]] + nc[th[k]]
                    break
        if sum1 > 100.0:
            return 1.0, True
        return 2.0 ** (sum1 * 0.25) / (_PI * axl * axl), False


def davies(lb, nc, n, sigma: float, c: float, lim: int = 100000, acc: float = 2e-5):
    """``Pr(Q < c)`` for ``Q = sum_j lb_j X_j + sigma X_0``, ``X_j ~ chi^2_{n_j}(nc_j)``.

    Returns ``(value, ifault)`` as the C routine (``ifault`` 0 ok, 1 accuracy not
    reached, 2 round-off possibly significant, 3 invalid parameters, 4 failure).
    """
    lb = [float(v) for v in lb]
    nc = [float(v) for v in nc]
    n = [int(v) for v in n]
    D = _Davies(lb, nc, n)
    r = len(lb)
    ifault = 0
    intl = ersm = 0.0
    acc1 = acc
    # th = order(abs(lb), decreasing = TRUE)  (R's revsort)
    th = _revsort_order([abs(v) for v in lb])
    sd = sigsq = sigma * sigma
    lmax = lmin = 0.0
    mean = 0.0
    for j in range(r):
        nj, lj, ncj = n[j], lb[j], nc[j]
        if nj < 0 or ncj < 0:
            return c, 3
        sd += lj * lj * (2.0 * nj + 4.0 * ncj)
        mean += lj * (nj + ncj)
        if lmax < lj:
            lmax = lj
        elif lmin > lj:
            lmin = lj
    if sd == 0.0:
        return (1.0 if c > 0.0 else 0.0), 0
    if lmin == 0.0 and lmax == 0.0 and sigma == 0.0:
        return c, 3
    sd = math.sqrt(sd)
    almx = -lmin if lmax < -lmin else lmax
    utx = 16.0 / sd
    up = 4.5 / sd
    un = -up
    utx = D.findu(utx, 0.5 * acc1, sigsq)
    if c != 0.0 and almx > 0.07 * sd:
        cf, fail = D.cfe(c, th)
        tausq = 0.25 * acc1 / cf
        if not fail:
            if D.truncation(utx, tausq, sigsq) < 0.2 * acc1:
                sigsq = sigsq + tausq
                utx = D.findu(utx, 0.25 * acc1, sigsq)
    acc1 = 0.5 * acc1
    while True:
        cu, up = D.ctff(acc1, up, mean, lmin, lmax, sigsq)
        d1 = cu - c
        if d1 < 0.0:
            return 1.0, ifault
        cl, un = D.ctff(acc1, un, mean, lmin, lmax, sigsq)
        d2 = c - cl
        if d2 < 0.0:
            return 0.0, ifault
        intv = 2 * _PI / d1 if d1 > d2 else 2 * _PI / d2
        x = utx / intv
        nt = int(math.floor(x))
        if x - nt > 0.5:
            nt += 1
        x = 3.0 / math.sqrt(acc1)
        ntm = int(math.floor(x))
        if x - ntm > 0.5:
            ntm += 1
        if nt > ntm * 1.5:
            intv1 = utx / ntm
            x = 2 * _PI / intv1
            if x <= abs(c):
                break
            cf1, f1 = D.cfe(c - x, th)
            cf2, f2 = D.cfe(c + x, th)
            fail = f2  # the C code keeps only the second call's flag
            tausq = 0.33 * acc1 / (1.1 * (cf1 + cf2))
            if fail:
                break
            acc1 = acc1 * 0.67
            if ntm > lim:
                return -1.0, ifault
            a, b = D.integrate(ntm, intv1, tausq, False, c, sigsq)
            intl += a
            ersm += b
            lim = lim - ntm
            sigsq = sigsq + tausq
            utx = D.findu(utx, 0.25 * acc1, sigsq)
            acc1 = 0.75 * acc1
        else:
            break
    if nt > lim:
        return -1.0, 1
    a, b = D.integrate(nt, intv, 0.0, True, c, sigsq)
    intl += a
    ersm += b
    val = 0.5 - intl
    x = ersm + acc / 10.0
    j = 1
    for _ in range(4):
        if j * x == j * ersm:
            ifault = 2
        j *= 2
    return val, ifault


def _revsort_order(a):
    """The permutation R's ``revsort(a, ib, n)`` applies (sort decreasing, heapsort)."""
    a = list(a)
    ib = list(range(len(a)))
    n = len(a)
    if n <= 1:
        return ib
    # R's revsort (sort.c): heapsort into decreasing order, permuting ib alongside
    lo = (n >> 1) + 1
    ir = n
    # 1-based indexing as in the C code
    ra_arr = [0.0] + a
    ib_arr = [0] + ib
    while True:
        if lo > 1:
            lo -= 1
            ra = ra_arr[lo]
            ii = ib_arr[lo]
        else:
            ra = ra_arr[ir]
            ii = ib_arr[ir]
            ra_arr[ir] = ra_arr[1]
            ib_arr[ir] = ib_arr[1]
            ir -= 1
            if ir == 1:
                ra_arr[1] = ra
                ib_arr[1] = ii
                return ib_arr[1:]
        i = lo
        j = lo << 1
        while j <= ir:
            if j < ir and ra_arr[j] > ra_arr[j + 1]:
                j += 1
            if ra > ra_arr[j]:
                ra_arr[i] = ra_arr[j]
                ib_arr[i] = ib_arr[j]
                i = j
                j += j
            else:
                j = ir + 1
        ra_arr[i] = ra
        ib_arr[i] = ii


def liu2(x, lam, h=None, lower_tail: bool = False):
    """mgcv's ``liu2``: Liu et al. (2009) approximation to ``Pr(sum lam_j chi^2_{h_j} > x)``."""
    lam = np.asarray(lam, dtype=float)
    h = np.ones_like(lam) if h is None else np.asarray(h, dtype=float)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    lh = lam * h
    muQ = lh.sum()
    lh = lh * lam
    c2 = lh.sum()
    lh = lh * lam
    c3 = lh.sum()
    xpos = x > 0
    res = np.ones_like(x)
    if xpos.sum() == 0 or c2 <= 0:
        return res
    s1 = c3 / c2 ** 1.5
    s2 = np.sum(lh * lam) / c2 ** 2
    sigQ = math.sqrt(2 * c2)
    t = (x[xpos] - muQ) / sigQ
    if s1 ** 2 > s2:
        a = 1 / (s1 - math.sqrt(s1 ** 2 - s2))
        delta = s1 * a ** 3 - a ** 2
        dfx = a ** 2 - 2 * delta
    else:
        a = 1 / s1
        delta = 0.0
        if c3 == 0:
            return res
        dfx = c2 ** 3 / c3 ** 2
    muX = dfx + delta
    sigX = math.sqrt(2) * a
    q = t * sigX + muX
    if delta == 0:
        res[xpos] = stats.chi2.cdf(q, dfx) if lower_tail else stats.chi2.sf(q, dfx)
    else:
        res[xpos] = stats.ncx2.cdf(q, dfx, delta) if lower_tail else stats.ncx2.sf(q, dfx, delta)
    return res


def psum_chisq(q: float, lb, df=None, nc=None, sigz: float = 0.0, lower_tail: bool = False,
               tol: float = 2e-5, nlim: int = 100000) -> float:
    """mgcv's ``psum.chisq``: ``Pr(sum lb_j chi^2_{df_j}(nc_j) + sigz Z > q)`` by Davies' method."""
    lb = np.asarray(lb, dtype=float)
    r = len(lb)
    df = np.ones(r) if df is None else np.broadcast_to(np.asarray(df, dtype=float), (r,))
    nc = np.zeros(r) if nc is None else np.broadcast_to(np.asarray(nc, dtype=float), (r,))
    df = np.round(df)
    if np.any(df < 1):
        raise ValueError("df must be positive integers")
    if np.all(lb == 0):
        raise ValueError("at least one element of lb must be non-zero")
    sigz = max(sigz, 0.0)
    val, ifault = davies(lb, nc, df.astype(int), sigz, float(q), nlim, tol)
    if ifault != 0:
        if ifault == 2:
            warnings.warn("danger of round-off error", RuntimeWarning, stacklevel=2)
            return val if lower_tail else 1 - val
        warnings.warn("failure of Davies method, falling back on Liu et al approximtion",
                      RuntimeWarning, stacklevel=2)
        return float(liu2(q, lb, h=df)[0]) if np.all(nc == 0) else float("nan")
    return val if lower_tail else 1 - val


def _sym_eigen(V):
    """``eigen(V, symmetric = TRUE)``: decreasing eigenvalues; vectors sign-normalized later."""
    ev, vec = np.linalg.eigh(V)
    o = np.argsort(-ev, kind="stable")
    return ev[o], vec[:, o]


def smooth_test_stat(p, X, V, rank: Optional[float] = None, type: int = 0, res_df: float = -1):
    """mgcv's ``testStat``: Wood (2013) test of a smooth's coefficients ``p``.

    ``X``: the smooth's columns of the model's ``R`` factor (or of any matrix with
    the same cross-product); ``V``: the coefficients' covariance (``Vp``);
    ``rank``: the reference degrees of freedom (``min(ncol(X), edf1)``).
    Returns ``(stat, pval, rank)``.
    """
    p = np.asarray(p, dtype=float)
    X = np.asarray(X, dtype=float)
    R = np.linalg.qr(X, mode="r")
    V = R @ np.asarray(V, dtype=float) @ R.T
    V = (V + V.T) / 2
    values, vectors = _sym_eigen(V)
    siv = np.sign(vectors[0, :])
    siv[siv == 0] = 1
    vectors = vectors * siv
    k = max(0, int(math.floor(rank)))
    nu = abs(rank - k)
    if type == 1:
        if rank > k + 0.05 or k == 0:
            k = k + 1
        nu = 0
        rank = k
    k1 = k + 1 if nu > 0 else k
    r_est = int(np.sum(values > values.max() * np.finfo(float).eps ** 0.9))
    if r_est < k1:
        k1 = k = r_est
        nu = 0
        rank = r_est
    vec = vectors.copy()
    if k1 < vec.shape[1]:
        vec = vec[:, :k1]
    Rp = R @ p
    if nu > 0 and k > 0:
        if k > 1:
            vec[:, :k - 1] = vec[:, :k - 1] / np.sqrt(values[:k - 1])
        b12 = 0.5 * nu * (1 - nu)
        b12 = math.sqrt(max(b12, 0.0))
        B = np.array([[1.0, b12], [b12, nu]])
        ev = np.diag(values[k - 1:k1] ** -0.5)
        B = ev @ B @ ev
        eb_val, eb_vec = _sym_eigen(B)
        rB = eb_vec @ np.diag(np.sqrt(eb_val)) @ eb_vec.T
        vec1 = vec.copy()
        vec1[:, k - 1:k1] = (rB @ np.diag([-1.0, 1.0]) @ vec[:, k - 1:k1].T).T
        vec[:, k - 1:k1] = (rB @ vec[:, k - 1:k1].T).T
    else:
        if k == 0:
            vec = vec * math.sqrt(1 / values[0])
        else:
            vec = vec / np.sqrt(values[:k])
        vec1 = vec
        if k == 1:
            rank = 1
    d = float(np.sum((vec.T @ Rp) ** 2))
    d1 = float(np.sum((vec1.T @ Rp) ** 2))
    rank1 = rank
    if nu > 0:
        if k1 == 1:
            rank1 = 1
            val = np.array([1.0])
        else:
            val = np.ones(k1)
            rp = nu + 1
            val[k - 1] = (rp + math.sqrt(rp * (2 - rp))) / 2
            val[k1 - 1] = rp - val[k - 1]
        if res_df <= 0:
            pval = (psum_chisq(d, val) + psum_chisq(d1, val)) / 2
        else:
            k0 = max(1, int(_rround(res_df)))
            dfv = np.concatenate([np.ones(len(val)), [k0]])
            pval = (psum_chisq(0, np.concatenate([val, [-d / k0]]), df=dfv)
                    + psum_chisq(0, np.concatenate([val, [-d1 / k0]]), df=dfv)) / 2
    else:
        pval = 2.0
    if pval > 1:
        if res_df <= 0:
            pval = (stats.chi2.sf(d, rank1) + stats.chi2.sf(d1, rank1)) / 2
        else:
            pval = (stats.f.sf(d / rank1, rank1, res_df) + stats.f.sf(d1 / rank1, rank1, res_df)) / 2
    return d, min(1.0, float(pval)), rank


def _rround(x: float) -> float:
    """R's ``round()`` to an integer (half to even)."""
    return float(np.rint(x))
