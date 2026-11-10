"""Maximum likelihood for linear path models (simultaneous equations).

A path model whose equations are linear in the endogenous variables is the system

    A y = G x + zeta,     A = I - B,     zeta ~ N(0, Psi),

where ``y`` holds the ``k`` endogenous variables, ``B`` their direct effects on
each other, ``x`` the (exogenous) design columns of every equation (including the
intercept column) and ``Psi`` the disturbance covariance (diagonal unless
residual covariances ``y1 ~~ y2`` are added).  The exogenous columns are treated
as fixed (lavaan's ``fixed.x = TRUE``), so the likelihood is the conditional one

    l = n log|det A| - n/2 log|Psi| - 1/2 tr(Psi^-1 Z'Z),    Z = Y A' - X G'.

For a recursive model with diagonal ``Psi`` it factorises into one regression per
equation (OLS is ML); feedback loops (non-recursive models) and correlated
disturbances need the joint fit.  Standard errors use the expected information
(lavaan's default ``information = "expected"``) or the observed information (the
Hessian, ``information = "observed"``).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["LinearSystem"]

_LOG2PI = float(np.log(2 * np.pi))


class LinearSystem:
    """The linear system of a path model.

    Parameters
    ----------
    endo : names of the ``k`` endogenous variables (one equation each).
    Y : (n, k) values of the endogenous variables.
    eq_columns : for every equation, the names of its design columns.
    eq_X : for every equation, its (n, p_i) design matrix.
    covariances : pairs ``(a, b)`` of endogenous variables with correlated disturbances.

    A design column named like an endogenous variable is that variable itself
    (a ``B`` entry); every other column is exogenous (a ``G`` entry).  Each
    equation has its own block of exogenous columns (so equal column names in
    different equations never clash).
    """

    def __init__(self, endo: Sequence[str], Y, eq_columns: Sequence[Sequence[str]], eq_X,
                 covariances: Sequence[Tuple[str, str]] = ()):
        self.endo = list(endo)
        k = self.k = len(self.endo)
        self.Y = np.asarray(Y, dtype=float).reshape(-1, k)
        self.n = self.Y.shape[0]
        idx = {v: i for i, v in enumerate(self.endo)}
        blocks, xnames = [], []
        # par entries: (kind, i, j, lhs, op, rhs)  kind in "B", "G", "P"
        par = []
        for i, (cols, Xi) in enumerate(zip(eq_columns, eq_X)):
            Xi = np.asarray(Xi, dtype=float).reshape(self.n, -1)
            for c, name in enumerate(cols):
                if name in idx:
                    if idx[name] == i:
                        raise ValueError("%s cannot be a predictor of itself" % name)
                    par.append(("B", i, idx[name], self.endo[i], "~", name))
                else:
                    blocks.append(Xi[:, c])
                    xnames.append(name)
                    j = len(xnames) - 1
                    if name == "(Intercept)":
                        par.append(("G", i, j, self.endo[i], "~1", ""))
                    else:
                        par.append(("G", i, j, self.endo[i], "~", name))
        self.X = np.column_stack(blocks) if blocks else np.zeros((self.n, 0))
        self.xnames = xnames
        self.q = self.X.shape[1]
        for i in range(k):
            par.append(("P", i, i, self.endo[i], "~~", self.endo[i]))
        seen = set()
        for a, b in covariances:
            ia, ib = idx[a], idx[b]
            if ia == ib or (min(ia, ib), max(ia, ib)) in seen:
                continue
            seen.add((min(ia, ib), max(ia, ib)))
            par.append(("P", max(ia, ib), min(ia, ib), a, "~~", b))
        self.par = par
        self.kind = np.array([p[0] for p in par])
        self.pi = np.array([p[1] for p in par], dtype=int)
        self.pj = np.array([p[2] for p in par], dtype=int)
        self.npar = len(par)
        self.XtX = self.X.T @ self.X
        self.recursive_diagonal = self._is_recursive() and len(seen) == 0

    # -- structure -------------------------------------------------------------
    def _is_recursive(self) -> bool:
        M = np.zeros((self.k, self.k), dtype=bool)
        sel = self.kind == "B"
        M[self.pi[sel], self.pj[sel]] = True
        R = M.copy()
        for _ in range(self.k):
            R = R | ((R.astype(int) @ M.astype(int)) > 0)
        return not np.any(np.diag(R))

    def unpack(self, theta):
        B = np.zeros((self.k, self.k))
        G = np.zeros((self.k, self.q))
        P = np.zeros((self.k, self.k))
        for kind, M in (("B", B), ("G", G), ("P", P)):
            sel = self.kind == kind
            M[self.pi[sel], self.pj[sel]] = theta[sel]
        P = P + np.tril(P, -1).T
        return B, G, P

    def pack(self, B, G, P):
        theta = np.empty(self.npar)
        for kind, M in (("B", B), ("G", G), ("P", P)):
            sel = self.kind == kind
            theta[sel] = M[self.pi[sel], self.pj[sel]]
        return theta

    # -- likelihood ------------------------------------------------------------
    def loglik(self, theta, grad: bool = True):
        """Conditional log-likelihood (and its gradient)."""
        B, G, P = self.unpack(theta)
        A = np.eye(self.k) - B
        sign, logdetA = np.linalg.slogdet(A)
        try:
            L = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            return (-np.inf, None) if grad else -np.inf
        if sign == 0 or not np.isfinite(logdetA):
            return (-np.inf, None) if grad else -np.inf
        n, k = self.n, self.k
        Z = self.Y @ A.T - self.X @ G.T
        S = Z.T @ Z
        Pinv = np.linalg.inv(P)
        logdetP = 2.0 * np.sum(np.log(np.diag(L)))
        ll = n * logdetA - 0.5 * n * logdetP - 0.5 * np.sum(Pinv * S) - 0.5 * n * k * _LOG2PI
        if not grad:
            return ll
        gB = -n * np.linalg.inv(A).T + Pinv @ Z.T @ self.Y
        gG = Pinv @ Z.T @ self.X
        Q = 0.5 * (-n * Pinv + Pinv @ S @ Pinv)
        gP = 2.0 * Q - np.diag(np.diag(Q))
        g = np.empty(self.npar)
        for kind, M in (("B", gB), ("G", gG), ("P", gP)):
            sel = self.kind == kind
            g[sel] = M[self.pi[sel], self.pj[sel]]
        return ll, g

    def expected_information(self, theta):
        """Fisher information ``n * E[-d2 l / d theta2]`` with the exogenous moments at
        their sample values (lavaan's structured expected information, fixed.x)."""
        B, G, P = self.unpack(theta)
        k = self.k
        A = np.eye(k) - B
        Ainv = np.linalg.inv(A)
        Om = Ainv @ P @ Ainv.T
        Omi = np.linalg.inv(Om)
        Pinv = np.linalg.inv(P)
        Pi = Ainv @ G                      # reduced form: E[y | x] = Pi x
        M2 = self.XtX / self.n
        # mean part: dm_p = A^-1 e_{i_p} (w_p' x)
        mean = (self.kind == "B") | (self.kind == "G")
        W = np.zeros((self.npar, self.q))
        selG = self.kind == "G"
        W[np.where(selG)[0], self.pj[selG]] = 1.0
        selB = self.kind == "B"
        W[selB] = Pi[self.pj[selB]]
        I = np.zeros((self.npar, self.npar))
        Im = (W @ M2 @ W.T) * Pinv[np.ix_(self.pi, self.pi)]
        I[np.ix_(mean, mean)] = Im[np.ix_(mean, mean)]
        # covariance part: 1/2 tr(Om^-1 dOm_p Om^-1 dOm_q)
        cov = np.where((self.kind == "B") | (self.kind == "P"))[0]
        D = np.zeros((len(cov), k, k))
        for r, p in enumerate(cov):
            i, j = self.pi[p], self.pj[p]
            if self.kind[p] == "B":
                E = np.zeros((k, k))
                E[i, j] = 1.0
                T = Ainv @ E @ Om
                D[r] = T + T.T
            else:
                E = np.zeros((k, k))
                E[i, j] = 1.0
                E[j, i] = 1.0
                D[r] = Ainv @ E @ Ainv.T
        K = np.einsum("ab,rbc->rac", Omi, D)
        I[np.ix_(cov, cov)] += 0.5 * np.einsum("rab,sba->rs", K, K)
        return self.n * I

    def observed_information(self, theta):
        """Hessian of ``-l`` (central differences of the analytic gradient)."""
        H = np.zeros((self.npar, self.npar))
        for p in range(self.npar):
            h = 1e-6 * max(1.0, abs(theta[p]))
            tp, tm = theta.copy(), theta.copy()
            tp[p] += h
            tm[p] -= h
            gp = self.loglik(tp)[1]
            gm = self.loglik(tm)[1]
            H[:, p] = -(gp - gm) / (2 * h)
        return (H + H.T) / 2

    # -- estimation --------------------------------------------------------------
    def starts(self) -> List[np.ndarray]:
        """Starting values: ``B = 0`` (lavaan's start, ``det(I - B) = 1``), two-stage
        least squares per equation (instruments: every exogenous column) and OLS.
        The likelihood is -inf where ``det(I - B) = 0``, so a non-recursive model can
        have optima in several regions; every start is tried."""
        if self.recursive_diagonal:  # OLS is the ML estimate
            return [self._ols_start()]
        k = self.k
        out = []
        for mode in ("zero", "2sls", "ols"):
            B = np.zeros((k, k))
            G = np.zeros((k, self.q))
            for i in range(k):
                sel = self.pi == i
                bj = self.pj[sel & (self.kind == "B")]
                gj = self.pj[sel & (self.kind == "G")]
                y = self.Y[:, i]
                Xg = self.X[:, gj]
                coef = None
                if mode == "zero":
                    g = np.linalg.lstsq(Xg, y, rcond=None)[0] if len(gj) else np.zeros(0)
                    coef = np.concatenate([np.zeros(len(bj)), g])
                else:
                    Wm = np.column_stack([self.Y[:, bj], Xg])
                    if mode == "2sls" and len(bj) and self.q:
                        What = self.X @ np.linalg.lstsq(self.X, Wm, rcond=None)[0]
                        if np.linalg.matrix_rank(What) == Wm.shape[1]:
                            coef = np.linalg.lstsq(What, y, rcond=None)[0]
                    if coef is None:
                        coef = np.linalg.lstsq(Wm, y, rcond=None)[0] if Wm.shape[1] else np.zeros(0)
                B[i, bj] = coef[:len(bj)]
                G[i, gj] = coef[len(bj):]
            Z = self.Y @ (np.eye(k) - B).T - self.X @ G.T
            P = np.diag(np.maximum(np.mean(Z ** 2, axis=0), 1e-8 * np.var(self.Y, axis=0) + 1e-300))
            out.append(self.pack(B, G, P))
        return out

    def _ols_start(self):
        k = self.k
        B = np.zeros((k, k))
        G = np.zeros((k, self.q))
        for i in range(k):
            sel = self.pi == i
            bj = self.pj[sel & (self.kind == "B")]
            gj = self.pj[sel & (self.kind == "G")]
            Wm = np.column_stack([self.Y[:, bj], self.X[:, gj]])
            coef = np.linalg.lstsq(Wm, self.Y[:, i], rcond=None)[0] if Wm.shape[1] else np.zeros(0)
            B[i, bj] = coef[:len(bj)]
            G[i, gj] = coef[len(bj):]
        Z = self.Y @ (np.eye(k) - B).T - self.X @ G.T
        return self.pack(B, G, np.diag(np.mean(Z ** 2, axis=0)))

    def start(self) -> np.ndarray:
        return self.starts()[0]

    def fit(self, theta0: Optional[np.ndarray] = None, maxiter: int = 1000):
        """Maximum likelihood by Fisher scoring with step halving, from ``theta0`` or
        from every start of :meth:`starts` (the best converged optimum is kept).

        Returns ``(theta, loglik, converged, iterations)``."""
        starts = [np.asarray(theta0, dtype=float)] if theta0 is not None else self.starts()
        best = None
        for th in starts:
            res = self._scoring(th.copy(), maxiter)
            # prefer converged fits, then the higher likelihood
            if best is None or (res[2], res[1]) > (best[2], best[1] + 1e-9 * abs(best[1])):
                best = res
        return best

    def _scoring(self, theta, maxiter):
        ll, g = self.loglik(theta)
        if not np.isfinite(ll):
            return theta, -np.inf, False, 0
        converged = False
        it = 0
        for it in range(1, maxiter + 1):
            info = self.expected_information(theta)
            try:
                step = np.linalg.solve(info, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(info, g, rcond=None)[0]
            t = 1.0
            while True:
                cand = theta + t * step
                lc, gc = self.loglik(cand)
                if np.isfinite(lc) and lc >= ll - 1e-12 * abs(ll):
                    break
                t *= 0.5
                if t < 1e-12:
                    lc = None
                    break
            if lc is None:
                break
            dtheta = cand - theta
            dl = lc - ll
            theta, ll, g = cand, lc, gc
            if np.all(np.abs(dtheta) <= 1e-10 * (1.0 + np.abs(theta))) and abs(dl) <= 1e-12 * (1.0 + abs(ll)):
                converged = True
                break
        return theta, ll, converged, it

    # -- inference -----------------------------------------------------------------
    def vcov(self, theta, information: str = "expected"):
        info = self.expected_information(theta) if information == "expected" else self.observed_information(theta)
        ev = np.linalg.eigvalsh(info)
        if ev.min() <= ev.max() * 1e-12:
            import warnings
            warnings.warn("the information matrix is singular: the path model is not identified "
                          "(standard errors are not available)", RuntimeWarning, stacklevel=3)
            return np.full_like(info, np.nan)
        return np.linalg.inv(info)

    def names(self) -> List[Tuple[str, str, str]]:
        return [(p[3], p[4], p[5]) for p in self.par]
