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
    return PenalizedFit(beta, sp_hat, names, fitted, edf, scale, crit_val, method, Hinv * scale, rss, converged)


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
        self._fit = fit

    def penalty_matrix(self, sp=None):
        """``sum_j sp_j S_j`` over all coefficients (defaults to the fitted ``sp``)."""
        from ._mgcv import penalty_matrix
        return penalty_matrix(self.penalties, self.model_matrix.shape[1], self.sp.to_numpy() if sp is None else sp)

    def predict(self, newdata=None, type: str = "response"):
        """Predictions (``type="response"``/``"link"``) or the ``"lpmatrix"`` for new data."""
        import pandas as pd
        if newdata is None:
            Xn = self.model_matrix
        else:
            Xn = self.model_matrix.model_spec.get_model_matrix(newdata)
        if type == "lpmatrix":
            return Xn
        return pd.Series(Xn.to_numpy() @ self.coefficients.to_numpy(), index=Xn.index)

    def __repr__(self):
        return "<GamFit %s %s: edf=%.3f scale=%.4g>" % (self.formula, self.method, self.edf.sum(), self.scale)


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
    if o is not None:
        res.fitted_values = res.fitted_values + np.asarray(o, dtype=float)
        res.residuals = res.y - res.fitted_values
    return res
