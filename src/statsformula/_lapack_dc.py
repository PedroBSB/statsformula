"""Divide and conquer path of reference LAPACK ``DSTEDC`` (COMPZ='I', n > 25).

Ports DLAED0, DLAED1, DLAED2 (deflation), DLAED3 (eigenvectors), DLAED5 and
DLAMRG literally so that eigenvector *signs* are identical to reference
LAPACK; the secular equation (DLAED4) is solved by vectorised bisection in
shifted coordinates (signs do not depend on the last bits of the roots).
"""

from __future__ import annotations

import math

import numpy as np

from ._lapack import _EPS, dlapy2, dsteqr

SMLSIZ = 25


def _dlamrg(n1, n2, a, dtrd1, dtrd2):
    """Return 1-based permutation merging two sorted sub-lists of ``a`` (1-based list)."""
    index = []
    ind1 = 1 if dtrd1 > 0 else n1
    ind2 = 1 + n1 if dtrd2 > 0 else n1 + n2
    n1sv, n2sv = n1, n2
    while n1sv > 0 and n2sv > 0:
        if a[ind1] <= a[ind2]:
            index.append(ind1)
            ind1 += dtrd1
            n1sv -= 1
        else:
            index.append(ind2)
            ind2 += dtrd2
            n2sv -= 1
    if n1sv == 0:
        for _ in range(n2sv):
            index.append(ind2)
            ind2 += dtrd2
    else:
        for _ in range(n1sv):
            index.append(ind1)
            ind1 += dtrd1
    return index


def _idamax(x):
    return int(np.argmax(np.abs(np.asarray(x[1:])))) + 1


def _dlaed5(i, d, z, rho):
    """d, z: 1-based lists of length 2."""
    dl = d[2] - d[1]
    if i == 1:
        w = 1.0 + 2.0 * rho * (z[2] * z[2] - z[1] * z[1]) / dl
        if w > 0.0:
            b = dl + rho * (z[1] * z[1] + z[2] * z[2])
            c = rho * z[1] * z[1] * dl
            tau = 2.0 * c / (b + math.sqrt(abs(b * b - 4.0 * c)))
            dlam = d[1] + tau
            d1 = -z[1] / tau
            d2 = z[2] / (dl - tau)
        else:
            b = -dl + rho * (z[1] * z[1] + z[2] * z[2])
            c = rho * z[2] * z[2] * dl
            if b > 0.0:
                tau = -2.0 * c / (b + math.sqrt(b * b + 4.0 * c))
            else:
                tau = (b - math.sqrt(b * b + 4.0 * c)) / 2.0
            dlam = d[2] + tau
            d1 = -z[1] / (dl + tau)
            d2 = -z[2] / tau
    else:
        b = -dl + rho * (z[1] * z[1] + z[2] * z[2])
        c = rho * z[2] * z[2] * dl
        if b > 0.0:
            tau = (b + math.sqrt(b * b + 4.0 * c)) / 2.0
        else:
            tau = 2.0 * c / (-b + math.sqrt(b * b + 4.0 * c))
        dlam = d[2] + tau
        d1 = -z[1] / (dl + tau)
        d2 = -z[2] / tau
    temp = math.sqrt(d1 * d1 + d2 * d2)
    return dlam, [d1 / temp, d2 / temp]


def _secular_all(dl, w, rho):
    """All roots of 1/rho + sum w_i^2/(d_i - lam) = 0 (d ascending, rho > 0).

    Returns (lam, delta) with delta[i, j] = d_i - lam_j computed in shifted
    coordinates (as DLAED4 does).
    """
    K = len(dl)
    w2 = w * w
    rhoinv = 1.0 / rho
    origin = np.arange(K)
    lo = np.zeros(K)
    hi = np.zeros(K)
    for j in range(K):
        if j < K - 1:
            mid = (dl[j + 1] - dl[j]) / 2.0
            f = rhoinv + np.sum(w2 / ((dl - dl[j]) - mid))
            if f > 0:
                origin[j] = j
                lo[j], hi[j] = 0.0, mid
            else:
                origin[j] = j + 1
                lo[j], hi[j] = -mid, 0.0
        else:
            origin[j] = K - 1
            lo[j], hi[j] = 0.0, rho * w2.sum()
    shifts = dl[None, :] - dl[origin][:, None]  # [j, i] = d_i - d_origin(j)
    for _ in range(1100):
        mid = 0.5 * (lo + hi)
        den = shifts - mid[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            f = rhoinv + np.sum(w2[None, :] / den, axis=1)
        up = (f > 0) | ~np.isfinite(f)
        newhi = np.where(up, mid, hi)
        newlo = np.where(up, lo, mid)
        if np.array_equal(newhi, hi) and np.array_equal(newlo, lo):
            break
        lo, hi = newlo, newhi
    tau = 0.5 * (lo + hi)
    tau = np.where(tau == 0.0, np.where(hi != 0.0, hi, lo), tau)
    lam = dl[origin] + tau
    delta = (shifts - tau[:, None]).T  # [i, j]
    return lam, delta


def _dlaed3(K, n, n1, D, Q, rho, dlamda, q2, indx, ctot, w):
    """D: 1-based list (len n+1); Q: n x n numpy (modified)."""
    if K == 0:
        return
    if K == 1:
        D[1] = dlamda[1] + rho * w[1] * w[1]
        Q[0, 0] = 1.0
    elif K == 2:
        for j in (1, 2):
            dlam, dv = _dlaed5(j, dlamda, w, rho)
            D[j] = dlam
            Q[0, j - 1] = dv[0]
            Q[1, j - 1] = dv[1]
        for j in (1, 2):
            w1 = Q[0, j - 1]
            w2 = Q[1, j - 1]
            vals = [None, w1, w2]
            Q[0, j - 1] = vals[indx[1]]
            Q[1, j - 1] = vals[indx[2]]
    else:
        dl = np.array(dlamda[1:K + 1])
        wv = np.array(w[1:K + 1])
        lam, delta = _secular_all(dl, wv, rho)
        for j in range(K):
            D[j + 1] = lam[j]
        Qd = delta  # Q(i,j) = delta_i(j)
        s = wv.copy()
        ww = np.diag(Qd).copy()
        for j in range(K):
            for i in range(K):
                if i != j:
                    ww[i] = ww[i] * (Qd[i, j] / (dl[i] - dl[j]))
        ww = np.where(s >= 0, np.sqrt(-ww), -np.sqrt(-ww))
        idx = np.array(indx[1:K + 1]) - 1
        for j in range(K):
            sv = ww / Qd[:, j]
            temp = np.sqrt((sv ** 2).sum())
            Q[:K, j] = sv[idx] / temp
    n2 = n - n1
    n12 = ctot[1] + ctot[2]
    n23 = ctot[2] + ctot[3]
    S = Q[ctot[1]:ctot[1] + n23, :K].copy()
    iq2 = n1 * n12
    if n23 != 0:
        Q2b = np.asarray(q2[iq2:iq2 + n2 * n23]).reshape((n2, n23), order="F")
        Q[n1:n, :K] = Q2b @ S
    else:
        Q[n1:n, :K] = 0.0
    S = Q[:n12, :K].copy()
    if n12 != 0:
        Q2a = np.asarray(q2[0:n1 * n12]).reshape((n1, n12), order="F")
        Q[:n1, :K] = Q2a @ S
    else:
        Q[:n1, :K] = 0.0


def _dlaed2(n, n1, D, Q, indxq, rho, z):
    """Returns (K, rho, dlamda, w, q2, indxc, ctot). D, Q, z, indxq modified (1-based lists)."""
    n2 = n - n1
    n1p1 = n1 + 1
    if rho < 0.0:
        for i in range(n1p1, n + 1):
            z[i] = -z[i]
    t = 1.0 / math.sqrt(2.0)
    for i in range(1, n + 1):
        z[i] *= t
    rho = abs(2.0 * rho)
    for i in range(n1p1, n + 1):
        indxq[i] += n1
    dlamda = [0.0] * (n + 1)
    for i in range(1, n + 1):
        dlamda[i] = D[indxq[i]]
    indxc = [0] + _dlamrg(n1, n2, dlamda, 1, 1)
    indx = [0] * (n + 1)
    for i in range(1, n + 1):
        indx[i] = indxq[indxc[i]]
    imax = _idamax(z)
    jmax = _idamax(D)
    eps = _EPS
    tol = 8.0 * eps * max(abs(D[jmax]), abs(z[imax]))
    w = [0.0] * (n + 1)
    if rho * abs(z[imax]) <= tol:
        Qn = np.zeros_like(Q)
        for j in range(1, n + 1):
            i = indx[j]
            Qn[:, j - 1] = Q[:, i - 1]
            dlamda[j] = D[i]
        Q[:, :] = Qn
        for j in range(1, n + 1):
            D[j] = dlamda[j]
        return 0, rho, dlamda, w, None, indxc, [0, 0, 0, 0, 0]
    coltyp = [0] * (n + 1)
    for i in range(1, n1 + 1):
        coltyp[i] = 1
    for i in range(n1p1, n + 1):
        coltyp[i] = 3
    indxp = [0] * (n + 2)
    K = 0
    k2 = n + 1
    pj = None
    j = 1
    done = False
    while j <= n:
        nj = indx[j]
        if rho * abs(z[nj]) <= tol:
            k2 -= 1
            coltyp[nj] = 4
            indxp[k2] = nj
            if j == n:
                done = True
                break
            j += 1
        else:
            pj = nj
            break
    if not done:
        while True:
            j += 1
            if j > n:
                break
            nj = indx[j]
            if rho * abs(z[nj]) <= tol:
                k2 -= 1
                coltyp[nj] = 4
                indxp[k2] = nj
            else:
                s = z[pj]
                c = z[nj]
                tau = dlapy2(c, s)
                tt = D[nj] - D[pj]
                c = c / tau
                s = -s / tau
                if abs(tt * c * s) <= tol:
                    z[nj] = tau
                    z[pj] = 0.0
                    if coltyp[nj] != coltyp[pj]:
                        coltyp[nj] = 2
                    coltyp[pj] = 4
                    x = Q[:, pj - 1].copy()
                    y = Q[:, nj - 1].copy()
                    Q[:, pj - 1] = c * x + s * y
                    Q[:, nj - 1] = c * y - s * x
                    tt = D[pj] * c ** 2 + D[nj] * s ** 2
                    D[nj] = D[pj] * s ** 2 + D[nj] * c ** 2
                    D[pj] = tt
                    k2 -= 1
                    i = 1
                    while True:
                        if k2 + i <= n:
                            if D[pj] < D[indxp[k2 + i]]:
                                indxp[k2 + i - 1] = indxp[k2 + i]
                                indxp[k2 + i] = pj
                                i += 1
                                continue
                            else:
                                indxp[k2 + i - 1] = pj
                        else:
                            indxp[k2 + i - 1] = pj
                        break
                    pj = nj
                else:
                    K += 1
                    dlamda[K] = D[pj]
                    w[K] = z[pj]
                    indxp[K] = pj
                    pj = nj
        K += 1
        dlamda[K] = D[pj]
        w[K] = z[pj]
        indxp[K] = pj
    ctot = [0, 0, 0, 0, 0]
    for jj in range(1, n + 1):
        ctot[coltyp[jj]] += 1
    psm = [0, 1, 1 + ctot[1], 1 + ctot[1] + ctot[2], 1 + ctot[1] + ctot[2] + ctot[3]]
    K = n - ctot[4]
    indx2 = [0] * (n + 1)
    indxc2 = [0] * (n + 1)
    for jj in range(1, n + 1):
        js = indxp[jj]
        ct = coltyp[js]
        indx2[psm[ct]] = js
        indxc2[psm[ct]] = jj
        psm[ct] += 1
    q2 = []
    zz = [0.0] * (n + 1)
    i = 1
    blk1 = []
    blk2 = []
    blk4 = []
    for _ in range(ctot[1]):
        js = indx2[i]
        blk1.append(Q[:n1, js - 1].copy())
        zz[i] = D[js]
        i += 1
    for _ in range(ctot[2]):
        js = indx2[i]
        blk1.append(Q[:n1, js - 1].copy())
        blk2.append(Q[n1:n, js - 1].copy())
        zz[i] = D[js]
        i += 1
    for _ in range(ctot[3]):
        js = indx2[i]
        blk2.append(Q[n1:n, js - 1].copy())
        zz[i] = D[js]
        i += 1
    for _ in range(ctot[4]):
        js = indx2[i]
        blk4.append(Q[:, js - 1].copy())
        zz[i] = D[js]
        i += 1
    for col in blk1:
        q2.extend(col.tolist())
    for col in blk2:
        q2.extend(col.tolist())
    if K < n:
        for c_i, col in enumerate(blk4):
            Q[:, K + c_i] = col
        for jj in range(K + 1, n + 1):
            D[jj] = zz[jj]
    return K, rho, dlamda, w, q2, indxc2, ctot


def _dlaed1(n, D, Q, indxq, rho, cutpnt):
    z = [0.0] * (n + 1)
    for i in range(1, cutpnt + 1):
        z[i] = Q[cutpnt - 1, i - 1]
    for i in range(cutpnt + 1, n + 1):
        z[i] = Q[cutpnt, i - 1]
    K, rho, dlamda, w, q2, indxc, ctot = _dlaed2(n, cutpnt, D, Q, indxq, rho, z)
    if K != 0:
        _dlaed3(K, n, cutpnt, D, Q, rho, dlamda, q2, indxc, ctot, w)
        idx = _dlamrg(K, n - K, D, 1, -1)
        for i in range(1, n + 1):
            indxq[i] = idx[i - 1]
    else:
        for i in range(1, n + 1):
            indxq[i] = i


def _dlaed0(n, d, e):
    """ICOMPQ = 2.  d, e numpy; returns (d, Q)."""
    D = [0.0] + list(map(float, d))
    E = [0.0] + list(map(float, e)) + [0.0]
    Q = np.zeros((n, n))
    iw = [0, n]
    subpbs = 1
    while iw[subpbs] > SMLSIZ:
        new = [0] * (2 * subpbs + 1)
        for j in range(subpbs, 0, -1):
            new[2 * j] = (iw[j] + 1) // 2
            new[2 * j - 1] = iw[j] // 2
        iw = new
        subpbs *= 2
    for j in range(2, subpbs + 1):
        iw[j] = iw[j] + iw[j - 1]
    for i in range(1, subpbs):
        submat = iw[i] + 1
        smm1 = submat - 1
        D[smm1] = D[smm1] - abs(E[smm1])
        D[submat] = D[submat] - abs(E[smm1])
    indxq = [0] * (n + 1)
    for i in range(0, subpbs):
        if i == 0:
            submat = 1
            matsiz = iw[1]
        else:
            submat = iw[i] + 1
            matsiz = iw[i + 1] - iw[i]
        w_, Z = dsteqr(D[submat:submat + matsiz], E[submat:submat + matsiz - 1])
        for k in range(matsiz):
            D[submat + k] = w_[k]
        Q[submat - 1:submat - 1 + matsiz, submat - 1:submat - 1 + matsiz] = Z
        k = 1
        for j in range(submat, iw[i + 1] + 1):
            indxq[j] = k
            k += 1
    while subpbs > 1:
        for i in range(0, subpbs - 1, 2):
            if i == 0:
                submat = 1
                matsiz = iw[2]
                msd2 = iw[1]
            else:
                submat = iw[i] + 1
                matsiz = iw[i + 2] - iw[i]
                msd2 = matsiz // 2
            sl = slice(submat - 1, submat - 1 + matsiz)
            Db = [0.0] + D[submat:submat + matsiz]
            Qb = Q[sl, sl].copy()
            ib = [0] + indxq[submat:submat + matsiz]
            _dlaed1(matsiz, Db, Qb, ib, E[submat + msd2 - 1], msd2)
            D[submat:submat + matsiz] = Db[1:]
            Q[sl, sl] = Qb
            indxq[submat:submat + matsiz] = ib[1:]
            iw[i // 2 + 1] = iw[i + 2]
        subpbs //= 2
    out_d = np.zeros(n)
    out_q = np.zeros((n, n))
    for i in range(1, n + 1):
        j = indxq[i]
        out_d[i - 1] = D[j]
        out_q[:, i - 1] = Q[:, j - 1]
    return out_d, out_q


def dstedc_dc(d, e):
    n = len(d)
    d = np.array(d, dtype=float)
    e = np.array(e, dtype=float)
    Z = np.eye(n)
    orgnrm = max(np.abs(d).max(), np.abs(e).max() if len(e) else 0.0)
    if orgnrm == 0.0:
        return d, Z
    eps = _EPS
    start = 1
    while start <= n:
        finish = start
        while finish < n:
            tiny = eps * math.sqrt(abs(d[finish - 1])) * math.sqrt(abs(d[finish]))
            if abs(e[finish - 1]) > tiny:
                finish += 1
            else:
                break
        m = finish - start + 1
        if m == 1:
            start = finish + 1
            continue
        sl = slice(start - 1, start - 1 + m)
        if m > SMLSIZ:
            nrm = max(np.abs(d[sl]).max(), np.abs(e[start - 1:start - 1 + m - 1]).max())
            ds = d[sl] / nrm
            es = e[start - 1:start - 1 + m - 1] / nrm
            w_, Qb = _dlaed0(m, ds, es)
            d[sl] = w_ * nrm
            Z[sl, sl] = Qb
        else:
            w_, Qb = dsteqr(d[sl], e[start - 1:start - 1 + m - 1])
            d[sl] = w_
            Z[sl, sl] = Qb
        start = finish + 1
    for ii in range(2, n + 1):
        i = ii - 1
        k = i
        p = d[i - 1]
        for j in range(ii, n + 1):
            if d[j - 1] < p:
                k = j
                p = d[j - 1]
        if k != i:
            d[k - 1] = d[i - 1]
            d[i - 1] = p
            Z[:, [i - 1, k - 1]] = Z[:, [k - 1, i - 1]]
    return d, Z
