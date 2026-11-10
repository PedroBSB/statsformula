"""Penalized least squares fitting of Gaussian GAMs, as ``mgcv::gam(family = gaussian())``.

The coefficients minimise ``||sqrt(W)(y - X b)||^2 + sum_j sp_j b' S_j b`` where the
``S_j`` are mgcv's penalty matrices (``G$S`` of ``gam(..., fit = FALSE)``).  Smoothing
parameters are fixed or estimated by REML (``method = "REML"``, scale profiled out) or
GCV (``method = "GCV.Cp"``, unknown scale), with exact gradients and L-BFGS-B.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt

__all__ = ["PenalizedFit", "fit_penalized", "GamFit", "gam"]


def _blocks(penalties):
    """Group penalties acting on the same coefficient block (e.g. the margins of a te())."""
    blocks = {}
    for j, pen in enumerate(penalties):
        key = (pen.off, pen.S.shape[0])
        blocks.setdefault(key, []).append(j)
    return [(off, size, idx) for (off, size), idx in blocks.items()]


class _Problem:
    def __init__(self, X, y, penalties, weights, gamma):
        self.n, self.p = X.shape
        sw = np.sqrt(weights) if weights is not None else np.ones(self.n)
        self.Xw = X * sw[:, None]
        self.yw = y * sw
        self.XtX = self.Xw.T @ self.Xw
        self.Xty = self.Xw.T @ self.yw
        self.yty = float(self.yw @ self.yw)
        # REML is the likelihood of y, whose covariance is scale / w
        self.logw = 0.0 if weights is None else float(np.sum(np.log(weights[weights > 0])))
        self.pens = penalties
        self.gamma = gamma
        self.blocks = []
        for off, size, idx in _blocks(penalties):
            tot = sum(penalties[j].S / np.linalg.norm(penalties[j].S) for j in idx)
            ev = np.linalg.eigvalsh(tot)
            rank = int(np.sum(ev > ev.max() * np.finfo(float).eps ** 0.75))
            self.blocks.append((slice(off, off + size), idx, rank))
        self.Mp = self.p - sum(b[2] for b in self.blocks)

    def Sl(self, sp):
        P = np.zeros((self.p, self.p))
        for lam, pen in zip(sp, self.pens):
            P[pen.columns, pen.columns] += lam * pen.S
        return P

    def solve(self, sp):
        Sl = self.Sl(sp)
        H = self.XtX + Sl
        try:
            cf = sla.cho_factor(H, lower=False)
            Hinv = sla.cho_solve(cf, np.eye(self.p))
            logdetH = 2.0 * np.sum(np.log(np.diag(cf[0])))
        except np.linalg.LinAlgError:
            ev, V = np.linalg.eigh(H)
            ev = np.maximum(ev, ev.max() * 1e-14)
            Hinv = (V / ev) @ V.T
            logdetH = float(np.sum(np.log(ev)))
        beta = Hinv @ self.Xty
        rss = max(self.yty - 2 * beta @ self.Xty + beta @ self.XtX @ beta, 0.0)
        return Sl, Hinv, logdetH, beta, rss

    def _logdetS(self, sp, want_grad):
        val = 0.0
        grad = np.zeros(len(self.pens))
        for sl, idx, rank in self.blocks:
            B = sum(sp[j] * self.pens[j].S for j in idx)
            ev, V = np.linalg.eigh((B + B.T) / 2)
            ev, V = ev[::-1][:rank], V[:, ::-1][:, :rank]
            ev = np.maximum(ev, np.finfo(float).tiny)
            val += float(np.sum(np.log(ev)))
            if want_grad:
                Bp = (V / ev) @ V.T
                for j in idx:
                    grad[j] = sp[j] * np.sum(Bp * self.pens[j].S)
        return val, grad

    def reml(self, rho):
        sp = np.exp(rho)
        Sl, Hinv, logdetH, beta, rss = self.solve(sp)
        Dp = rss + beta @ Sl @ beta
        nm = self.n - self.Mp
        ldS, gS = self._logdetS(sp, True)
        V = 0.5 * nm * np.log(Dp / nm) + 0.5 * logdetH - 0.5 * ldS + 0.5 * nm * (1 + np.log(2 * np.pi)) - 0.5 * self.logw
        g = np.empty(len(sp))
        for j, pen in enumerate(self.pens):
            c = pen.columns
            bj = beta[c]
            g[j] = 0.5 * nm * sp[j] * (bj @ pen.S @ bj) / Dp + 0.5 * sp[j] * np.sum(Hinv[c, c] * pen.S) - 0.5 * gS[j]
        return V, g

    def gcv(self, rho):
        sp = np.exp(rho)
        Sl, Hinv, _, beta, rss = self.solve(sp)
        A = Hinv @ self.XtX
        tau = np.trace(A)
        den = self.n - self.gamma * tau
        V = self.n * rss / den ** 2
        Slb = Sl @ beta
        HSlb = Hinv @ Slb
        g = np.empty(len(sp))
        for j, pen in enumerate(self.pens):
            c = pen.columns
            dD = 2 * sp[j] * (HSlb[c] @ pen.S @ beta[c])
            dtau = -sp[j] * np.sum((A[c, :] @ Hinv[:, c]) * pen.S.T)
            g[j] = self.n * dD / den ** 2 + 2 * self.n * self.gamma * rss * dtau / den ** 3
        return V, g


class PenalizedFit:
    """Result of :func:`fit_penalized` (one response column)."""

    def __init__(self, coef, sp, sp_names, fitted, edf, scale, criterion, method, Vp, rss, converged):
        self.coef = coef
        self.sp = sp
        self.sp_names = sp_names
        self.fitted = fitted
        self.edf = edf
        self.scale = scale
        self.criterion = criterion
        self.method = method
        self.Vp = Vp
        self.rss = rss
        self.converged = converged

    def __repr__(self):
        sp = ", ".join("%s=%.4g" % kv for kv in zip(self.sp_names, self.sp))
        return "<PenalizedFit %s: edf=%.3f scale=%.4g sp: %s>" % (self.method, self.edf.sum(), self.scale, sp)


def _initial_rho(prob):
    """mgcv-like starting values: balance each penalty against the data Hessian."""
    d = np.diag(prob.XtX)
    rho = np.zeros(len(prob.pens))
    for j, pen in enumerate(prob.pens):
        c = pen.columns
        ds = np.diag(pen.S)
        ds = ds[ds > ds.max() * np.finfo(float).eps ** 0.5] if ds.max() > 0 else ds
        rho[j] = np.log(max(d[c].mean(), 1e-300) / max(ds.mean(), 1e-300)) if ds.size else 0.0
    return rho


def _check_identifiable(prob):
    """Warn when the penalized design [X; total penalty] is rank deficient (mgcv then
    sets the unidentifiable coefficients to zero, e.g. ``y ~ x + s(x)``)."""
    P = np.zeros((prob.p, prob.p))
    for pen in prob.pens:
        P[pen.columns, pen.columns] += pen.S / max(np.linalg.norm(pen.S), 1e-300)
    xs = max(np.linalg.norm(prob.XtX), 1e-300)
    H = prob.XtX / xs + P
    ev = np.linalg.eigvalsh((H + H.T) / 2)
    r = int(np.sum(ev > ev.max() * 1e-11))
    if r < prob.p:
        import warnings
        warnings.warn("the model is not identifiable: the penalized design has rank %d < %d coefficients "
                      "(e.g. a parametric term duplicated by a smooth, as in y ~ x + s(x)); "
                      "standard errors are meaningless and mgcv would set %d coefficient(s) to zero"
                      % (r, prob.p, prob.p - r), RuntimeWarning, stacklevel=3)
    return r


def fit_penalized(X, y, penalties, method: str = "REML", sp=None, weights=None, gamma: float = 1.0,
                  sp_names: Optional[Sequence[str]] = None) -> PenalizedFit:
    """Fit ``y ~ X`` by penalized least squares with mgcv's penalties.

    ``sp``: smoothing parameters; ``None`` or negative entries are estimated (as in
    ``mgcv::gam(sp = )``).  ``method``: ``"REML"`` or ``"GCV.Cp"``.  ``gamma`` inflates
    the effective degrees of freedom in the GCV score.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    w = None if weights is None else np.asarray(weights, dtype=float).ravel()
    if method not in ("REML", "GCV.Cp", "GCV"):
        raise ValueError("method must be 'REML' or 'GCV.Cp'")
    method = "GCV.Cp" if method == "GCV" else method
    prob = _Problem(X, y, list(penalties), w, gamma)
    rank = _check_identifiable(prob)
    m = len(prob.pens)
    names = list(sp_names) if sp_names is not None else [p.name for p in prob.pens]
    if sp is None:
        sp_in = -np.ones(m)
    else:
        sp_in = np.broadcast_to(np.asarray(sp, dtype=float), (m,)).copy()
    free = sp_in < 0
    rho = np.where(free, 0.0, np.log(np.where(free, 1.0, np.maximum(sp_in, 1e-300))))
    converged = True
    if free.any():
        rho0 = _initial_rho(prob)
        crit = prob.reml if method == "REML" else prob.gcv

        def f(r):
            full = rho.copy()
            full[free] = r
            v, g = crit(full)
            return v, g[free]

        best = None
        # beyond ~exp(15) times the data scale a penalty has fully suppressed its range
        # space; larger values only add rounding error to log|H| - log|S|+
        bounds = [(r0 - 20.0, r0 + 15.0) for r0 in rho0[free]]
        for start in (rho0[free], rho0[free] + 5.0, rho0[free] - 5.0):
            res = sopt.minimize(f, start, jac=True, method="L-BFGS-B", bounds=bounds,
                                options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000})
            if best is None or res.fun < best.fun - 1e-12:
                best = res
        rho[free] = best.x
        converged = bool(best.success)
    sp_hat = np.exp(rho)
    Sl, Hinv, _, beta, rss = prob.solve(sp_hat)
    F = Hinv @ prob.XtX
    edf = np.diag(F).copy()
    tau = edf.sum()
    if method == "REML" and free.any():
        nm = prob.n - prob.Mp
        scale = (rss + beta @ Sl @ beta) / nm
        crit_val = prob.reml(rho)[0]
    else:
        scale = rss / (prob.n - tau)
        crit_val = prob.gcv(rho)[0] if method == "GCV.Cp" else prob.reml(rho)[0]
    fitted = X @ beta
    out = PenalizedFit(beta, sp_hat, names, fitted, edf, scale, crit_val, method, Hinv * scale, rss, converged)
    # for summary(): mgcv's edf1 = diag(2F - FF), the R factor of sqrt(W) X, the data
    out.edf1 = 2.0 * edf - np.sum(F * F.T, axis=1)
    out.R = np.linalg.qr(prob.Xw, mode="r")
    out.y = y
    out.weights = w
    out.rank = rank
    out.penalties = list(prob.pens)
    return out


class GamFit:
    """A Gaussian GAM fitted by :func:`gam` (``mgcv::gam(family = gaussian())``).

    Attributes: ``coefficients`` (Series), ``sp`` (Series of smoothing parameters,
    named like ``b$sp``), ``fitted_values``, ``edf`` (per coefficient), ``scale``
    (``b$sig2``), ``Vp`` (Bayesian posterior covariance), ``penalties``
    (list of :class:`GamPenalty`), ``model_matrix``; ``predict(newdata)``.
    """

    def __init__(self, formula, X, y, fit, method):
        import pandas as pd
        self.formula = formula
        self.model_matrix = X
        self.y = y
        self.method = method
        self.penalties = X.model_spec.penalties
        self.coefficients = pd.Series(fit.coef, index=list(X.columns))
        self.sp = pd.Series(fit.sp, index=fit.sp_names, dtype=float)
        self.fitted_values = pd.Series(fit.fitted, index=X.index)
        self.residuals = self.y - self.fitted_values
        self.edf = pd.Series(fit.edf, index=list(X.columns))
        self.scale = fit.scale
        self.Vp = fit.Vp
        self.criterion = fit.criterion
        self.converged = fit.converged
        self.weights = None  # prior weights (set by gam())
        self._fit = fit

    def penalty_matrix(self, sp=None):
        """``sum_j sp_j S_j`` over all coefficients (defaults to the fitted ``sp``)."""
        from ._mgcv import penalty_matrix
        return penalty_matrix(self.penalties, self.model_matrix.shape[1], self.sp.to_numpy() if sp is None else sp)

    @property
    def residual_df(self) -> float:
        """Residual degrees of freedom ``n - sum(edf)`` (as in ``summary.gam``)."""
        return float(len(self.fitted_values) - self.edf.sum())

    def predict(self, newdata=None, type: str = "response", se_fit: bool = False):
        """Predictions (``type="response"``/``"link"``) or the ``"lpmatrix"`` for new data.

        ``se_fit=True`` returns a DataFrame with columns ``fit`` and ``se_fit``, the
        standard errors ``sqrt(diag(Xp Vp Xp'))`` of ``predict.gam(se.fit = TRUE)``
        (Bayesian posterior covariance; identical for both types with the Gaussian
        identity link).
        """
        import pandas as pd
        if type not in ("response", "link", "lpmatrix"):
            raise ValueError("type must be 'response', 'link' or 'lpmatrix'")
        if newdata is None:
            Xn = self.model_matrix
        else:
            Xn = self.model_matrix.model_spec.get_model_matrix(newdata)
        if type == "lpmatrix":
            return Xn
        A = Xn.to_numpy()
        fit = pd.Series(A @ self.coefficients.to_numpy(), index=Xn.index)
        if not se_fit:
            return fit
        se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", A, self.Vp, A), 0.0))
        return pd.DataFrame({"fit": fit, "se_fit": se}, index=Xn.index)

    def summary(self) -> "GamSummary":
        """``summary.gam``: parametric coefficients (``p_table``), approximate
        significance of the smooth terms (``s_table``), parametric term tests
        (``pterms_table``) and the fit statistics; prints like R."""
        return GamSummary(self._fit, self.model_matrix.model_spec, list(self.model_matrix.columns),
                          self.formula, y=self.y.to_numpy(dtype=float),
                          fitted=self.fitted_values.to_numpy(dtype=float),
                          assign=getattr(self.model_matrix, "assign", None))

    def __repr__(self):
        return "<GamFit %s %s: edf=%.3f scale=%.4g>" % (self.formula, self.method, self.edf.sum(), self.scale)


def _null_space_zero(sm, penalties) -> bool:
    """mgcv's ``null.space.dim == 0``: the smooth's total penalty (after the
    identifiability constraints are absorbed) has full rank, as for ``cs``, ``ts``,
    ``cc``, ``re`` or ``m = 1`` smooths."""
    Ss = [p.S for p in penalties if p.smooth == sm.label]
    if not Ss:
        return False
    tot = sum(S / np.linalg.norm(S) for S in Ss)
    ev = np.linalg.eigvalsh((tot + tot.T) / 2)
    return int(np.sum(ev > ev.max() * np.finfo(float).eps ** 0.75)) == tot.shape[0]


def _is_random(sm) -> bool:
    return getattr(sm.spec, "kind", "s") == "s" and getattr(sm.spec, "bs", None) == "re"


def _mroot(S):
    ev, vec = np.linalg.eigh((S + S.T) / 2)
    keep = ev > max(ev.max(), 0) * np.finfo(float).eps ** 0.8
    return vec[:, keep] * np.sqrt(ev[keep])


class GamSummary:
    """``summary(gam_fit)`` (mgcv's ``summary.gam``), printed like R.

    Attributes: ``p_table`` (parametric coefficients: ``Estimate``, ``Std. Error``,
    ``t value``, ``Pr(>|t|)``), ``s_table`` (smooth terms: ``edf``, ``Ref.df``,
    ``F``, ``p-value``, Wood's (2013) tests), ``pterms_table`` (Wald F test of each
    parametric term, as in ``anova.gam``), ``edf`` (per smooth), ``residual_df``,
    ``scale``, ``r_sq`` (adjusted R^2), ``dev_expl`` (deviance explained), ``n``,
    ``rank``, ``method``, ``criterion``.
    """

    def __init__(self, fit: "PenalizedFit", spec, columns, formula, y=None, fitted=None, assign=None):
        import pandas as pd
        from scipy import stats
        from ._davies import smooth_test_stat
        k = spec.nsdf
        coef = np.asarray(fit.coef, dtype=float)
        Vp = np.asarray(fit.Vp, dtype=float)
        n = len(fit.fitted)
        edf_all = np.asarray(fit.edf, dtype=float)
        df = float(n - edf_all.sum())
        se_all = np.sqrt(np.maximum(np.diag(Vp), 0.0))
        est, se = coef[:k], se_all[:k]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = est / se
        p = 2.0 * stats.t.sf(np.abs(t), df)
        self.p_table = pd.DataFrame({"Estimate": est, "Std. Error": se, "t value": t, "Pr(>|t|)": p},
                                    index=list(columns[:k]))
        # parametric terms (anova.gam's pTerms.table)
        rows = []
        pterms = getattr(spec, "pterms", None)
        labels = list(getattr(pterms, "term_labels", []) or [])
        if assign is not None and labels:
            asg = np.asarray(list(assign)[:k])
            for i, lab in enumerate(labels):
                ind = np.where(asg == i + 1)[0]
                if not len(ind):
                    continue
                b = coef[ind]
                V = Vp[np.ix_(ind, ind)]
                if len(b) == 1:
                    nb, chi = 1, float(b[0] ** 2 / V[0, 0])
                else:
                    ev, vec = np.linalg.eigh((V + V.T) / 2)
                    o = np.argsort(-ev)
                    ev, vec = ev[o], vec[:, o]
                    nb = int(min(len(b), np.sum(ev > np.finfo(float).eps ** 0.5 * ev[0])))
                    inv = np.zeros_like(ev)
                    inv[:nb] = 1 / ev[:nb]
                    chi = float(b @ (vec * inv) @ vec.T @ b)
                rows.append((lab, nb, chi / nb, stats.f.sf(chi / nb, nb, df)))
        self.pterms_table = pd.DataFrame([r[1:] for r in rows], index=[r[0] for r in rows],
                                         columns=["df", "F", "p-value"])
        # smooth terms
        R = fit.R
        sig2 = float(fit.scale)
        sp = np.asarray(fit.sp, dtype=float)
        pens = list(getattr(fit, "penalties", spec.penalties))
        p_all = len(coef)
        recs, edf_s = [], {}
        start = k
        firsts = []
        for sm in spec.smooths:
            firsts.append((start, start + sm.X.shape[1]))
            start += sm.X.shape[1]
        for i, (sm, (a, b)) in enumerate(zip(spec.smooths, firsts)):
            ind = np.arange(a, b)
            edfi = float(edf_all[a:b].sum())
            edf1i = float(np.asarray(fit.edf1)[a:b].sum())
            edf_s[sm.label] = edfi
            fixed = bool(getattr(sm.spec, "fixed", False))
            if not fixed and _null_space_zero(sm, pens):
                stat, pval, rank = self._re_test(i, ind, firsts, spec, coef, Vp, R, sig2, sp, pens, df, p_all)
            else:
                stat, pval, rank = smooth_test_stat(coef[a:b], R[:, a:b], Vp[a:b, a:b],
                                             min(b - a, edf1i), type=0, res_df=df)
            recs.append((sm.label, edfi, rank, stat / rank if rank else np.nan, pval))
        self.s_table = pd.DataFrame([r[1:] for r in recs], index=[r[0] for r in recs],
                                    columns=["edf", "Ref.df", "F", "p-value"])
        self.edf = pd.Series(edf_s, dtype=float)
        self.residual_df = df
        self.scale = sig2
        self.n = n
        y = np.asarray(fit.y if y is None else y, dtype=float)
        mu = np.asarray(fit.fitted if fitted is None else fitted, dtype=float)
        w = np.ones_like(y) if fit.weights is None else np.asarray(fit.weights, dtype=float)
        ybar = np.sum(w * y) / np.sum(w)
        sw = np.sqrt(w)
        self.r_sq = float(1.0 - np.var(sw * (y - mu), ddof=1) * (n - 1) / (np.var(sw * (y - ybar), ddof=1) * df))
        self.deviance = float(np.sum(w * (y - mu) ** 2))
        self.null_deviance = float(np.sum(w * (y - ybar) ** 2))
        self.dev_expl = (self.null_deviance - self.deviance) / self.null_deviance
        self.method = fit.method
        self.criterion = float(fit.criterion)
        self.formula = formula
        self.rank = int(getattr(fit, "rank", p_all))
        self.np = p_all
        self.family, self.link = "gaussian", "identity"

    @staticmethod
    def _re_test(m, ind, firsts, spec, coef, Vp, R, sig2, sp, pens, df_res, p):
        """mgcv's ``reTest`` (with ``recov``) for smooths without a penalty null space."""
        rind = np.zeros(p, dtype=bool)
        for i, sm in enumerate(spec.smooths):
            if i != m and _is_random(sm):
                rind[firsts[i][0]:firsts[i][1]] = True
        if not rind.any():
            S1 = np.zeros((p, p))
            for lam, pen in zip(sp, pens):
                S1[pen.columns, pen.columns] += lam * pen.S
            Ve = Vp @ (R.T @ R) @ Vp / sig2
            Ve = (Ve + Ve.T) / 2
            LRB = np.vstack([R, _mroot(S1).T])
        else:
            mp = np.zeros(p, dtype=int)
            p2 = int(rind.sum())
            mp[rind] = np.arange(p2)
            mp[~rind] = np.arange(p - p2)
            R1, R2 = R[:, ~rind], R[:, rind]
            S1 = np.zeros((p - p2, p - p2))
            S2 = np.zeros((p2, p2))
            for lam, pen in zip(sp, pens):
                cols = np.arange(pen.off, pen.off + pen.S.shape[0])
                if rind[cols[0]]:
                    S2[np.ix_(mp[cols], mp[cols])] += lam * pen.S
                else:
                    S1[np.ix_(mp[cols], mp[cols])] += lam * pen.S
            ev, vec = np.linalg.eigh((S2 + S2.T) / 2)
            keep = ev > ev.max() * np.finfo(float).eps ** 0.8
            inv = np.where(keep, 1 / np.where(keep, ev, 1), 0.0)
            S2r = np.sqrt(inv)[:, None] * vec.T
            L = np.linalg.cholesky(np.eye(R.shape[0]) + (S2r @ R2.T).T @ (S2r @ R2.T)).T
            Ve = (L @ R @ Vp).T @ (L @ R @ Vp) / sig2
            LRB = np.vstack([L @ R1, _mroot(S1).T])
            ind_m = mp[ind]
            other = np.setdiff1d(np.arange(LRB.shape[1]), ind_m)
            Rm = np.linalg.qr(LRB[:, np.concatenate([other, ind_m])], mode="r")[-len(ind):, -len(ind):]
            return GamSummary._re_finish(Rm, Ve, ind, coef, sig2, df_res)
        other = np.setdiff1d(np.arange(p), ind)
        Rm = np.linalg.qr(LRB[:, np.concatenate([other, ind])], mode="r")[-len(ind):, -len(ind):]
        return GamSummary._re_finish(Rm, Ve, ind, coef, sig2, df_res)

    @staticmethod
    def _re_finish(Rm, Ve, ind, coef, sig2, df_res):
        from ._davies import psum_chisq
        B = _mroot(Ve[np.ix_(ind, ind)])
        d = Rm @ coef[ind]
        stat = float(np.sum(d ** 2) / sig2)
        RB = Rm @ B
        ev = np.linalg.eigvalsh(RB.T @ RB / sig2)[::-1]
        ev[ev < 0] = 0
        rank = int(np.sum(ev > ev.max() * np.finfo(float).eps ** 0.8))
        kk = max(1, int(np.rint(df_res)))
        pval = psum_chisq(0, np.concatenate([ev, [-stat / kk]]), df=np.concatenate([np.ones(len(ev)), [kk]]))
        return stat, pval, rank

    def __str__(self):
        from ._rprint import format_c, print_coefmat
        out = ["", "Family: %s " % self.family, "Link function: %s " % self.link, "", "Formula:", str(self.formula)]
        if len(self.p_table):
            out += ["", "Parametric coefficients:",
                    print_coefmat(self.p_table.to_numpy(), list(self.p_table.index), list(self.p_table.columns))]
        out.append("")
        if len(self.s_table):
            out += ["Approximate significance of smooth terms:",
                    print_coefmat(self.s_table.to_numpy(), list(self.s_table.index), list(self.s_table.columns),
                                  has_pvalue=True, cs_ind=[0])]
        out.append("")
        if self.rank < self.np:
            out.append("Rank: %d/%d" % (self.rank, self.np))
        meth = {"REML": "-REML", "ML": "-ML", "GCV.Cp": "GCV"}.get(self.method, self.method)
        out.append("R-sq.(adj) =  %s   Deviance explained = %s%%" % (format_c(self.r_sq, 3, 5),
                                                                    format_c(self.dev_expl * 100, 3, 4)))
        out.append("%s = %s  Scale est. = %s  n = %d" % (meth, format_c(self.criterion, 5),
                                                          format_c(self.scale, 5, 8, "-"), self.n))
        return "\n".join(out)

    __repr__ = __str__


def gam(formula, data, method: str = "REML", sp=None, weights=None, gamma: float = 1.0, knots=None,
        na_action: Optional[str] = None, env=None) -> GamFit:
    """Fit a Gaussian additive model like ``mgcv::gam(formula, data, method = method)``.

    The design matrix and penalties are mgcv's (``gam(..., fit = FALSE)``); the
    smoothing parameters are estimated by ``"REML"`` or ``"GCV.Cp"`` unless given
    in ``sp`` (negative entries are estimated).  ``weights``: prior weights (a
    column name or an array).
    """
    import pandas as pd
    from ._api import _get_env
    from ._formula import coerce_formula
    from ._mgcv import gam_model_matrix, interpret_gam
    from ._model import model_frame_from_terms, model_response
    env = _get_env(env, formula, 1)
    f = coerce_formula(formula, env)
    X = gam_model_matrix(f, data, knots=knots, na_action=na_action, env=env)
    ft = interpret_gam(f, env).fake_formula.terms(data=data)
    mf = model_frame_from_terms(ft, data, env=env, na_action=na_action, drop_unused_levels=True)
    y = model_response(mf)
    if y is None:
        raise ValueError("gam() needs a response")
    y = pd.Series(np.asarray(y, dtype=float).ravel(), index=X.index)
    from ._model import model_offset
    o = model_offset(mf)
    yfit = y if o is None else y - np.asarray(o, dtype=float)
    w = None
    if weights is not None:
        w = np.asarray(data[weights] if isinstance(weights, str) else weights, dtype=float)
        w = pd.Series(w, index=data.index).loc[X.index].to_numpy()
    fit = fit_penalized(X.to_numpy(), yfit.to_numpy(), X.model_spec.penalties, method=method, sp=sp,
                        weights=w, gamma=gamma)
    res = GamFit(str(f), X, y, fit, method)
    res.weights = w
    if o is not None:
        res.fitted_values = res.fitted_values + np.asarray(o, dtype=float)
        res.residuals = res.y - res.fitted_values
    return res
