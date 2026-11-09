"""LINPACK style Householder QR (``dqrdc2`` / ``dqrsl``) as used by R's ``qr()``.

Reproducing R's sign conventions exactly matters for bases such as
``splines::ns`` whose columns are built from ``qr.qty``.
"""

from __future__ import annotations

import numpy as np


class LinpackQR:
    def __init__(self, x, tol: float = 1e-7):
        x = np.array(x, dtype=float, copy=True)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n, p = x.shape
        qraux = np.zeros(p)
        pivot = np.arange(p)
        work0 = np.sqrt((x ** 2).sum(axis=0)) if n else np.zeros(p)
        work1 = work0.copy()
        qraux[:] = work0
        work1[work1 == 0] = 1.0
        k = p + 1
        lup = min(n, p)
        # dqrdc2 with limited column pivoting
        l = 0
        while l < lup:
            # cycle columns with negligible norm to the end
            while l + 1 < k and qraux[l] < work1[l] * tol:
                lp1 = l + 1
                for i in range(n):
                    t = x[i, l]
                    x[i, l:p - 1] = x[i, l + 1:p]
                    x[i, p - 1] = t
                i = pivot[l]
                t = qraux[l]
                tt = work0[l]
                ttt = work1[l]
                pivot[l:p - 1] = pivot[l + 1:p].copy()
                qraux[l:p - 1] = qraux[l + 1:p].copy()
                work0[l:p - 1] = work0[l + 1:p].copy()
                work1[l:p - 1] = work1[l + 1:p].copy()
                pivot[p - 1] = i
                qraux[p - 1] = t
                work0[p - 1] = tt
                work1[p - 1] = ttt
                k -= 1
            if l == n - 1:
                break
            nrmxl = np.sqrt((x[l:, l] ** 2).sum())
            if nrmxl != 0.0:
                if x[l, l] != 0.0:
                    nrmxl = abs(nrmxl) * (1.0 if x[l, l] >= 0 else -1.0)
                x[l:, l] /= nrmxl
                x[l, l] += 1.0
                for j in range(l + 1, p):
                    t = -np.dot(x[l:, l], x[l:, j]) / x[l, l]
                    x[l:, j] += t * x[l:, l]
                    if qraux[j] != 0.0:
                        tt = 1.0 - (abs(x[l, j]) / qraux[j]) ** 2
                        tt = max(tt, 0.0)
                        t = tt
                        if abs(t) < 1e-6:
                            qraux[j] = np.sqrt((x[l + 1:, j] ** 2).sum())
                            work0[j] = qraux[j]
                        else:
                            qraux[j] = qraux[j] * np.sqrt(t)
                qraux[l] = x[l, l]
                x[l, l] = -nrmxl
            l += 1
        rank = min(k - 1, n)
        self.qr = x
        self.qraux = qraux
        self.rank = rank
        self.pivot = pivot
        self.n, self.p = n, p

    def _apply(self, y, transpose: bool):
        y = np.array(y, dtype=float, copy=True)
        vec = y.ndim == 1
        if vec:
            y = y.reshape(-1, 1)
        n = self.n
        ju = min(self.rank, n - 1)
        rng = range(ju) if transpose else range(ju - 1, -1, -1)
        for j in rng:
            if self.qraux[j] == 0.0:
                continue
            u = self.qr[j:, j].copy()
            u[0] = self.qraux[j]
            t = -(u @ y[j:, :]) / u[0]
            y[j:, :] += np.outer(u, t)
        return y[:, 0] if vec else y

    def qty(self, y):
        return self._apply(y, transpose=True)

    def qy(self, y):
        return self._apply(y, transpose=False)


def gram_schmidt_basis(X):
    """``qr.qy(QR, diag(R))``: Q times the diagonal of R (sign invariant)."""
    qr = LinpackQR(X)
    z = np.zeros_like(qr.qr)
    d = np.diag(qr.qr)
    for i in range(min(z.shape)):
        z[i, i] = d[i]
    return qr.qy(z)
