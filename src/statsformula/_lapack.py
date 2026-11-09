"""Reference LAPACK (3.12) symmetric tridiagonal eigen-solver ported to Python.

R links the reference LAPACK, while numpy/scipy may use vendor libraries
(Apple Accelerate, MKL) whose eigenvector *signs* differ.  mgcv's thin plate
regression spline basis depends on those signs (through its Lanczos solver),
so ``DSTEQR`` / ``DSTEDC`` are reproduced here exactly as in the reference
implementation.
"""

from __future__ import annotations

import math

import numpy as np

_EPS = np.finfo(float).eps * 0.5  # DLAMCH('E') (relative machine precision, rounding)
_SAFMIN = np.finfo(float).tiny     # DLAMCH('S')
_SAFMAX = 1.0 / _SAFMIN
_RTMIN = math.sqrt(_SAFMIN)
_RTMAX = math.sqrt(_SAFMAX / 2)


def _sign(a, b):
    return abs(a) if b >= 0 else -abs(a)


def dlartg(f, g):
    f1 = abs(f)
    g1 = abs(g)
    if g == 0.0:
        return 1.0, 0.0, f
    if f == 0.0:
        return 0.0, _sign(1.0, g), g1
    if _RTMIN < f1 < _RTMAX and _RTMIN < g1 < _RTMAX:
        d = math.sqrt(f * f + g * g)
        c = f1 / d
        r = _sign(d, f)
        s = g / r
        return c, s, r
    u = min(_SAFMAX, max(_SAFMIN, f1, g1))
    fs = f / u
    gs = g / u
    d = math.sqrt(fs * fs + gs * gs)
    c = abs(fs) / d
    r = _sign(d, f)
    s = gs / r
    return c, s, r * u


def dlapy2(x, y):
    xa, ya = abs(x), abs(y)
    w = max(xa, ya)
    z = min(xa, ya)
    if z == 0.0 or w > 1.79e308:
        return w
    return w * math.sqrt(1.0 + (z / w) ** 2)


def dlaev2(a, b, c):
    sm = a + c
    df = a - c
    adf = abs(df)
    tb = b + b
    ab = abs(tb)
    if abs(a) > abs(c):
        acmx, acmn = a, c
    else:
        acmx, acmn = c, a
    if adf > ab:
        rt = adf * math.sqrt(1.0 + (ab / adf) ** 2)
    elif adf < ab:
        rt = ab * math.sqrt(1.0 + (adf / ab) ** 2)
    else:
        rt = ab * math.sqrt(2.0)
    if sm < 0.0:
        rt1 = 0.5 * (sm - rt)
        sgn1 = -1
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b
    elif sm > 0.0:
        rt1 = 0.5 * (sm + rt)
        sgn1 = 1
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b
    else:
        rt1 = 0.5 * rt
        rt2 = -0.5 * rt
        sgn1 = 1
    if df >= 0.0:
        cs = df + rt
        sgn2 = 1
    else:
        cs = df - rt
        sgn2 = -1
    acs = abs(cs)
    if acs > ab:
        ct = -tb / cs
        sn1 = 1.0 / math.sqrt(1.0 + ct * ct)
        cs1 = ct * sn1
    else:
        if ab == 0.0:
            cs1 = 1.0
            sn1 = 0.0
        else:
            tn = -cs / tb
            cs1 = 1.0 / math.sqrt(1.0 + tn * tn)
            sn1 = tn * cs1
    if sgn1 == sgn2:
        tn = cs1
        cs1 = -sn1
        sn1 = tn
    return rt1, rt2, cs1, sn1


def _dlasr_rvb(Z, l, mm, c, s):
    """DLASR('R','V','B') on columns l..l+mm-1 (1-based) of Z (0-based array)."""
    for j in range(mm - 1, 0, -1):
        ct, st = c[j - 1], s[j - 1]
        if ct != 1.0 or st != 0.0:
            a = l - 1 + j - 1
            temp = Z[:, a + 1].copy()
            Z[:, a + 1] = ct * temp - st * Z[:, a]
            Z[:, a] = st * temp + ct * Z[:, a]


def _dlasr_rvf(Z, l, mm, c, s):
    for j in range(1, mm):
        ct, st = c[j - 1], s[j - 1]
        if ct != 1.0 or st != 0.0:
            a = l - 1 + j - 1
            temp = Z[:, a + 1].copy()
            Z[:, a + 1] = ct * temp - st * Z[:, a]
            Z[:, a] = st * temp + ct * Z[:, a]


def dsteqr(d_in, e_in, Z=None):
    """DSTEQR(COMPZ='I' or 'V'): eigenvalues (ascending) and eigenvectors."""
    n = len(d_in)
    D = [0.0] + [float(x) for x in d_in]
    E = [0.0] + [float(x) for x in e_in] + [0.0]
    if Z is None:
        Z = np.eye(n)
    else:
        Z = np.array(Z, dtype=float, copy=True)
    if n <= 1:
        return np.array(D[1:]), Z
    eps = _EPS
    eps2 = eps ** 2
    safmin = _SAFMIN
    ssfmax = math.sqrt(_SAFMAX) / 3.0
    ssfmin = math.sqrt(safmin) / eps2
    nmaxit = n * 30
    jtot = 0
    l1 = 1
    nm1 = n - 1
    work_c = [0.0] * (n + 1)
    work_s = [0.0] * (n + 1)
    while True:
        # label 10
        if l1 > n:
            break
        if l1 > 1:
            E[l1 - 1] = 0.0
        m = n
        if l1 <= nm1:
            found = False
            for mm_ in range(l1, nm1 + 1):
                tst = abs(E[mm_])
                if tst == 0.0:
                    m = mm_
                    found = True
                    break
                if tst <= (math.sqrt(abs(D[mm_])) * math.sqrt(abs(D[mm_ + 1]))) * eps:
                    E[mm_] = 0.0
                    m = mm_
                    found = True
                    break
            if not found:
                m = n
        l = l1
        lsv = l
        lend = m
        lendsv = lend
        l1 = m + 1
        if lend == l:
            continue
        anorm = max([abs(D[i]) for i in range(l, lend + 1)] + [abs(E[i]) for i in range(l, lend)])
        iscale = 0
        if anorm == 0.0:
            continue
        if anorm > ssfmax:
            iscale = 1
            f = ssfmax / anorm
            for i in range(l, lend + 1):
                D[i] *= f
            for i in range(l, lend):
                E[i] *= f
        elif anorm < ssfmin:
            iscale = 2
            f = ssfmin / anorm
            for i in range(l, lend + 1):
                D[i] *= f
            for i in range(l, lend):
                E[i] *= f
        if abs(D[lend]) < abs(D[l]):
            lend = lsv
            l = lendsv
        if lend > l:
            # QL iteration
            while True:
                # label 40
                m = lend
                if l != lend:
                    for mm_ in range(l, lend):
                        tst = abs(E[mm_]) ** 2
                        if tst <= (eps2 * abs(D[mm_])) * abs(D[mm_ + 1]) + safmin:
                            m = mm_
                            break
                if m < lend:
                    E[m] = 0.0
                p = D[l]
                if m == l:
                    D[l] = p
                    l += 1
                    if l <= lend:
                        continue
                    break
                if m == l + 1:
                    rt1, rt2, c, s = dlaev2(D[l], E[l], D[l + 1])
                    _dlasr_rvb(Z, l, 2, [c], [s])
                    D[l] = rt1
                    D[l + 1] = rt2
                    E[l] = 0.0
                    l += 2
                    if l <= lend:
                        continue
                    break
                if jtot == nmaxit:
                    break
                jtot += 1
                g = (D[l + 1] - p) / (2.0 * E[l])
                r = dlapy2(g, 1.0)
                g = D[m] - p + (E[l] / (g + _sign(r, g)))
                s = 1.0
                c = 1.0
                p = 0.0
                for i in range(m - 1, l - 1, -1):
                    f = s * E[i]
                    b = c * E[i]
                    c, s, r = dlartg(g, f)
                    if i != m - 1:
                        E[i + 1] = r
                    g = D[i + 1] - p
                    r = (D[i] - g) * s + 2.0 * c * b
                    p = s * r
                    D[i + 1] = g + p
                    g = c * r - b
                    work_c[i] = c
                    work_s[i] = -s
                mm = m - l + 1
                _dlasr_rvb(Z, l, mm, work_c[l:m], work_s[l:m])
                D[l] = D[l] - p
                E[l] = g
        else:
            # QR iteration
            while True:
                # label 90
                m = lend
                if l != lend:
                    for mm_ in range(l, lend, -1):
                        tst = abs(E[mm_ - 1]) ** 2
                        if tst <= (eps2 * abs(D[mm_])) * abs(D[mm_ - 1]) + safmin:
                            m = mm_
                            break
                if m > lend:
                    E[m - 1] = 0.0
                p = D[l]
                if m == l:
                    D[l] = p
                    l -= 1
                    if l >= lend:
                        continue
                    break
                if m == l - 1:
                    rt1, rt2, c, s = dlaev2(D[l - 1], E[l - 1], D[l])
                    _dlasr_rvf(Z, l - 1, 2, [c], [s])
                    D[l - 1] = rt1
                    D[l] = rt2
                    E[l - 1] = 0.0
                    l -= 2
                    if l >= lend:
                        continue
                    break
                if jtot == nmaxit:
                    break
                jtot += 1
                g = (D[l - 1] - p) / (2.0 * E[l - 1])
                r = dlapy2(g, 1.0)
                g = D[m] - p + (E[l - 1] / (g + _sign(r, g)))
                s = 1.0
                c = 1.0
                p = 0.0
                for i in range(m, l):
                    f = s * E[i]
                    b = c * E[i]
                    c, s, r = dlartg(g, f)
                    if i != m:
                        E[i - 1] = r
                    g = D[i] - p
                    r = (D[i + 1] - g) * s + 2.0 * c * b
                    p = s * r
                    D[i] = g + p
                    g = c * r - b
                    work_c[i] = c
                    work_s[i] = s
                mm = l - m + 1
                _dlasr_rvf(Z, m, mm, work_c[m:l], work_s[m:l])
                D[l] = D[l] - p
                E[l - 1] = g
        # label 140
        if iscale == 1:
            f = anorm / ssfmax
            for i in range(lsv, lendsv + 1):
                D[i] *= f
            for i in range(lsv, lendsv):
                E[i] *= f
        elif iscale == 2:
            f = anorm / ssfmin
            for i in range(lsv, lendsv + 1):
                D[i] *= f
            for i in range(lsv, lendsv):
                E[i] *= f
        if jtot >= nmaxit:
            raise ArithmeticError("DSTEQR failed to converge")
    # label 160: selection sort
    for ii in range(2, n + 1):
        i = ii - 1
        k = i
        p = D[i]
        for j in range(ii, n + 1):
            if D[j] < p:
                k = j
                p = D[j]
        if k != i:
            D[k] = D[i]
            D[i] = p
            Z[:, [i - 1, k - 1]] = Z[:, [k - 1, i - 1]]
    return np.array(D[1:]), Z


def dstedc(d, e):
    """DSTEDC(COMPZ='I'): for n <= 25 (SMLSIZ) reference LAPACK calls DSTEQR."""
    n = len(d)
    if n <= 25:
        return dsteqr(d, e)
    from ._lapack_dc import dstedc_dc
    return dstedc_dc(d, e)
