"""``summary()`` objects of the fitted models: R's ``summary.lm`` for least-squares
formula estimators, a generic summary for other scikit-learn estimators, and the
lavaan-style summary of path models.  Each prints like R and keeps its tables as
DataFrames."""

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ._rprint import format_c, format_num, format_pval, print_coefmat, print_named, signif


def _zapsmall(x, digits):
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    if not ok.any():
        return x
    mx = np.max(np.abs(x[ok]))
    d = max(0, digits - math.log10(mx)) if mx > 0 else digits
    d = int(math.floor(d + 0.5))
    return np.array([round(v, d) for v in x])


def _quantile7(x):
    return np.quantile(np.asarray(x, dtype=float), [0, 0.25, 0.5, 0.75, 1.0])


def _deleted(n: int) -> str:
    if not n:
        return ""
    return "%d observation%s deleted due to missingness" % (n, "" if n == 1 else "s")


class LmSummary:
    """``summary(lm(...))`` of one response, printed like R's ``print.summary.lm``.

    Attributes: ``coefficients`` (DataFrame ``Estimate``, ``Std. Error``,
    ``t value``, ``Pr(>|t|)``; aliased coefficients are left out, as in R, and
    listed in ``aliased``), ``sigma`` (residual standard error), ``df``
    (``(rank, residual df, columns)``), ``r_squared``, ``adj_r_squared``,
    ``fstatistic`` (``(value, numdf, dendf)`` or None), ``f_pvalue``,
    ``residuals`` (weighted residuals), ``cov_unscaled`` (``(X'X)^-1``),
    ``loglik``, ``aic``, ``bic`` (``logLik``/``AIC``/``BIC`` of the ``lm``),
    ``nobs``.
    """

    def __init__(self, X, y, columns: Sequence[str], intercept: bool, weights=None, rownames=None,
                 n_deleted: int = 0, call: str = "", response: Optional[str] = None):
        from ._linalg import LinpackQR
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, ncol = X.shape
        w = None if weights is None else np.asarray(weights, dtype=float).ravel()
        sw = np.ones(n) if w is None else np.sqrt(w)
        qr = LinpackQR(X * sw[:, None], tol=1e-7)
        p = qr.rank
        piv = qr.pivot
        qty = qr.qty(y * sw)
        coef = np.full(ncol, np.nan)
        if p:
            Rm = np.triu(qr.qr[:p, :p])
            coef[piv[:p]] = np.linalg.solve(Rm, qty[:p]) if p else []
        # residuals as lm.fit: Q applied to the effects with the first rank entries zeroed
        eff = qty.copy()
        eff[:p] = 0.0
        r = qr.qy(eff) / sw
        fitted = y - r
        rdf = n - p
        if w is None:
            mss = np.sum((fitted - fitted.mean()) ** 2) if intercept else np.sum(fitted ** 2)
            rss = float(np.sum(r ** 2))
        else:
            m = np.sum(w * fitted / np.sum(w))
            mss = np.sum(w * (fitted - m) ** 2) if intercept else np.sum(w * fitted ** 2)
            rss = float(np.sum(w * r ** 2))
            r = np.sqrt(w) * r
        resvar = rss / rdf if rdf > 0 else np.nan
        self.call = call
        self.response = response
        self.columns = list(columns)
        self.aliased = pd.Series(np.isnan(coef), index=self.columns)
        self.residuals = pd.Series(r, index=None if rownames is None else list(rownames))
        self.fitted_values = fitted
        self.weighted = w is not None and float(np.ptp(w)) != 0
        self.df = (p, rdf, ncol)
        self.nobs = n
        self.n_deleted = n_deleted
        self.sigma = math.sqrt(resvar) if np.isfinite(resvar) else np.nan
        if p:
            Rinv = np.linalg.inv(Rm)
            XtXi = Rinv @ Rinv.T
            se = np.sqrt(np.diag(XtXi) * resvar)
            est = coef[piv[:p]]
            with np.errstate(divide="ignore", invalid="ignore"):
                tval = est / se
            pv = 2 * stats.t.sf(np.abs(tval), rdf) if rdf > 0 else np.full(p, np.nan)
            names = [self.columns[i] for i in piv[:p]]
            self.coefficients = pd.DataFrame({"Estimate": est, "Std. Error": se, "t value": tval,
                                              "Pr(>|t|)": pv}, index=names)
            self.cov_unscaled = pd.DataFrame(XtXi, index=names, columns=names)
        else:
            self.coefficients = pd.DataFrame(columns=["Estimate", "Std. Error", "t value", "Pr(>|t|)"],
                                             dtype=float)
            self.cov_unscaled = pd.DataFrame(dtype=float)
        df_int = 1 if intercept else 0
        if p and p != df_int:
            self.r_squared = float(mss / (mss + rss))
            self.adj_r_squared = float(1 - (1 - self.r_squared) * ((n - df_int) / rdf))
            fval = float((mss / (p - df_int)) / resvar)
            self.fstatistic = (fval, p - df_int, rdf)
            self.f_pvalue = float(stats.f.sf(fval, p - df_int, rdf))
        else:
            self.r_squared = self.adj_r_squared = 0.0
            self.fstatistic = None
            self.f_pvalue = None
        # logLik.lm (weights enter as in R: 0.5 * sum(log(w)) for non-zero weights)
        ww = np.ones(n) if w is None else w
        ll = 0.5 * (np.sum(np.log(ww)) - n * (math.log(2 * math.pi) + 1 - math.log(n) + math.log(rss)))
        self.loglik = float(ll)
        k = p + 1
        self.aic = float(-2 * ll + 2 * k)
        self.bic = float(-2 * ll + math.log(n) * k)

    @property
    def rsquared(self):
        return self.r_squared

    def __str__(self):
        digits = 4
        out = ["", "Call:", self.call, ""]
        p, rdf, ncol = self.df
        out.append(("Weighted " if self.weighted else "") + "Residuals:")
        resid = self.residuals.to_numpy()
        if rdf > 5:
            rq = _zapsmall(_quantile7(resid), digits + 1)
            out.append(print_named(rq, ["Min", "1Q", "Median", "3Q", "Max"], digits))
        elif rdf > 0:
            names = [str(i) for i in self.residuals.index]
            out.append(print_named(resid, names, digits))
        else:
            out.append("ALL %d residuals are 0: no residual degrees of freedom!" % p)
        if len(self.aliased) == 0:
            out += ["", "No Coefficients"]
        else:
            nsing = ncol - p
            out.append("")
            out.append("Coefficients: (%d not defined because of singularities)" % nsing if nsing
                       else "Coefficients:")
            cf = np.full((ncol, 4), np.nan)
            names = list(self.aliased.index)
            for i, nm in enumerate(names):
                if nm in self.coefficients.index and not self.aliased.iloc[i]:
                    cf[i] = self.coefficients.loc[nm].to_numpy(dtype=float)
            if not nsing:
                names = list(self.coefficients.index)
                cf = self.coefficients.to_numpy(dtype=float)
            out.append(print_coefmat(cf, names, ["Estimate", "Std. Error", "t value", "Pr(>|t|)"], digits=digits))
        out.append("")
        out.append("Residual standard error: %s on %d degrees of freedom"
                   % (format_num([signif(self.sigma, digits)], 7)[0], rdf))
        if self.n_deleted:
            out.append("  (%s)" % _deleted(self.n_deleted))
        if self.fstatistic is not None:
            f, d1, d2 = self.fstatistic
            out.append("Multiple R-squared:  %s,\tAdjusted R-squared:  %s "
                       % (format_c(self.r_squared, digits), format_c(self.adj_r_squared, digits)))
            out.append("F-statistic: %s on %d and %d DF,  p-value: %s"
                       % (format_c(f, digits), d1, d2, format_pval([self.f_pvalue], digits)[0]))
        out.append("")
        return "\n".join(out)

    __repr__ = __str__


class MlmSummary(list):
    """``summary()`` of a multi-response least-squares fit: one :class:`LmSummary` per
    response (R's ``summary.mlm``)."""

    def __str__(self):
        parts = []
        for s in self:
            parts.append("Response %s :" % s.response)
            parts.append(str(s))
            parts.append("")
        return "\n".join(parts)

    __repr__ = __str__


class EstimatorSummary:
    """Summary of a formula estimator that is not an unpenalized least-squares fit
    (no standard errors): the estimator, the data used and the coefficients when
    the estimator has them."""

    def __init__(self, formula, estimator, nobs, n_deleted, columns, coef, intercept, score, score_name):
        self.formula = formula
        self.estimator = estimator
        self.nobs = nobs
        self.n_deleted = n_deleted
        self.columns = list(columns)
        self.coefficients = coef
        self.intercept = intercept
        self.score = score
        self.score_name = score_name

    def __str__(self):
        out = ["", "Formula: %s" % self.formula, "Estimator: %r" % (self.estimator,),
               "Observations: %d%s" % (self.nobs, ("   (%s)" % _deleted(self.n_deleted)) if self.n_deleted else ""),
               "Design columns: %d" % len(self.columns)]
        if self.score is not None:
            out.append("%s (training data): %s" % (self.score_name, format_c(self.score, 4)))
        if self.coefficients is not None:
            out += ["", "Coefficients:"]
            C = self.coefficients
            cells = [[v for v in format_num(C.iloc[:, j].to_numpy(dtype=float), 4)] for j in range(C.shape[1])]
            rows = [[cells[j][i] for j in range(C.shape[1])] for i in range(C.shape[0])]
            from ._rprint import print_matrix
            out.append(print_matrix(rows, [str(i) for i in C.index], [str(c) for c in C.columns]))
            out += ["", "(no standard errors: %s is not an unpenalized least-squares fit)"
                    % type(self.estimator).__name__]
        out.append("")
        return "\n".join(out)

    __repr__ = __str__


def _kv(label: str, value) -> str:
    v = value if isinstance(value, str) else ("NA" if value is None or (isinstance(value, float) and math.isnan(value))
                                              else ("%d" % value if isinstance(value, (int, np.integer)) else "%.3f" % value))
    return "  " + label + v.rjust(max(52 - len(label), len(v) + 1))


def _f3(v) -> str:
    return "NA" if v is None or not np.isfinite(v) else "%.3f" % v


def lavaan_blocks(P, test: str = "z", dotted=(), name_w: Optional[int] = None):
    """The parameter blocks of lavaan's ``summary()`` (Latent Variables, Regressions,
    Covariances, Intercepts, Variances, Defined Parameters) for a table with
    columns ``lhs, op, rhs, est`` and optionally ``label, se, <test>, pvalue``."""
    dotted = set(dotted)
    has_se = "se" in P.columns
    labels = P["label"].fillna("").astype(str) if "label" in P.columns else pd.Series([""] * len(P), index=P.index)
    longest = max([len(str(v)) + 6 + (len(lab) + 3 if lab else 0) for v, lab in zip(P["rhs"], labels)]
                  + [len(str(v)) + 6 for v in P["lhs"]] + [0])
    name_w = max(18, longest) if name_w is None else name_w
    cols = 4 if has_se else 1
    width = name_w + 9 * cols
    hdr = " " * name_w + ("%9s%9s%9s%9s" % ("Estimate", "Std.Err", "%s-value" % test, "P(>|%s|)" % test)
                          if has_se else "%9s" % "Estimate")

    def row(name, r, lab):
        lab = "(%s)" % lab if lab else ""
        head = name.ljust(max(name_w - len(lab), len(name) + 1)) + lab if lab else name.ljust(name_w)
        if not has_se:
            return head + "%9s" % _f3(r["est"])
        return head + "%9s%9s%9s%9s" % (_f3(r["est"]), _f3(r.get("se", np.nan)), _f3(r.get(test, np.nan)),
                                        _f3(r.get("pvalue", np.nan)))

    def dot(v):
        return ".%s" % v if v in dotted else v

    out = []
    for op, title in (("=~", "Latent Variables:"), ("~", "Regressions:")):
        sub = P[P.op == op]
        if len(sub):
            out += ["", title, hdr]
            for lhs in dict.fromkeys(sub.lhs):
                out.append(("  %s %s" % (lhs, op)).ljust(width))
                for i, r in sub[sub.lhs == lhs].iterrows():
                    out.append(row("    %s" % r["rhs"], r, labels[i]))
    cov = P[(P.op == "~~") & (P.lhs != P.rhs)]
    if len(cov):
        out += ["", "Covariances:", hdr]
        for lhs in dict.fromkeys(cov.lhs):
            out.append(((" .%s ~~" if lhs in dotted else "  %s ~~") % lhs).ljust(width))
            for i, r in cov[cov.lhs == lhs].iterrows():
                out.append(row(("   .%s" if r["rhs"] in dotted else "    %s") % r["rhs"], r, labels[i]))
    for title, sub in (("Intercepts:", P[P.op == "~1"]), ("Variances:", P[(P.op == "~~") & (P.lhs == P.rhs)])):
        if len(sub):
            out += ["", title, hdr]
            for i, r in sub.iterrows():
                out.append(row(("   .%s" if r["lhs"] in dotted else "    %s") % r["lhs"], r, labels[i]))
    dfn = P[P.op == ":="]
    if len(dfn):
        out += ["", "Defined Parameters:", hdr]
        for i, r in dfn.iterrows():
            out.append(row("    %s" % r["lhs"], r, ""))
    return out, name_w


class SemSummary:
    """lavaan-style summary of a :class:`statsformula.sem.SemFit`: the backend, the
    fit statistics it provides (semopy: chi-square test, baseline model, CFI, TLI,
    RMSEA) and the parameter estimates with standard errors when available."""

    def __init__(self, fit):
        P = fit.estimates.copy()
        if fit.defined:
            extra = pd.DataFrame({"lhs": list(fit.defined), "op": ":=", "rhs": "", "label": "",
                                  "free": 0, "est": list(fit.defined.values())})
            P = pd.concat([P, extra], ignore_index=True)
        self.parameters = P
        self.fit_stats = fit.fit_stats or {}
        self.backend = fit.name

    def __str__(self):
        P, fs = self.parameters, self.fit_stats
        lines = ["SEM fitted with %s (lavaan parameter table)" % self.backend, ""]
        lines.append(_kv("Number of model parameters", int(fs.get("npar", int((P.free > 0).sum())))))
        lines.append("")
        if "ntotal" in fs:
            lines.append(_kv("Number of observations", int(fs["ntotal"])))
        if "chisq" in fs:
            lines += ["", "Model Test User Model:", "",
                      _kv("Test statistic", fs["chisq"]), _kv("Degrees of freedom", int(round(fs["df"])))]
            if fs["df"] > 0:
                lines.append(_kv("P-value (Chi-square)", fs["pvalue"]))
            lines += ["", "Model Test Baseline Model:", "",
                      _kv("Test statistic", fs["baseline.chisq"]),
                      _kv("Degrees of freedom", int(round(fs["baseline.df"]))),
                      "", "User Model versus Baseline Model:", "",
                      _kv("Comparative Fit Index (CFI)", fs["cfi"]), _kv("Tucker-Lewis Index (TLI)", fs["tli"]),
                      "", "Root Mean Square Error of Approximation:", "", _kv("RMSEA", fs["rmsea"])]
        lines += ["", "Parameter Estimates:", ""]
        if "se" in P.columns:
            lines.append(_kv("Standard errors", self.backend))
        else:
            lines.append(_kv("Standard errors", "none (%s)" % self.backend))
        dotted = set(P.lhs[P.op == "~"]) | set(P.rhs[P.op == "=~"])
        blocks, _ = lavaan_blocks(P, "z", dotted)
        lines += blocks
        lines.append("")
        return "\n".join(lines)

    __repr__ = __str__


class PathSummary:
    """Summary of a :class:`~statsformula.sklearn.PathModel`, laid out like lavaan's
    ``summary(fit, fit.measures = TRUE, rsquare = TRUE)``.

    Attributes: ``parameters`` (``coefficients()``), ``fit_measures`` (Series, or
    None when the model is not one linear system), ``rsquare`` (Series),
    ``effects`` (dict of the indirect/total effect tables, if requested),
    ``estimation``, ``information``, ``nobs``.
    """

    def __init__(self, model, parameters, fit_measures, rsquare, effects=None, smooth=None):
        self.smooth = smooth or {}
        if self.smooth:
            drop = np.zeros(len(parameters), dtype=bool)
            for resp, (_, cols) in self.smooth.items():
                drop |= ((parameters.lhs == resp) & (parameters.op == "~") & parameters.rhs.isin(cols)).to_numpy()
            parameters = parameters[~drop].reset_index(drop=True)
        self.parameters = parameters
        self.fit_measures = fit_measures
        self.rsquare = rsquare
        self.effects = effects
        self.estimation = model.estimation_
        self.information = model.information_
        self.converged = bool(getattr(model, "converged_", True))
        self.nobs = model.nobs_
        self.test = "t" if model.se == "lm" else "z"
        self.npar = int(len(parameters)) + sum(len(c) for _, c in self.smooth.values())
        self.level = model.level

    def __str__(self):
        P = self.parameters
        fm = self.fit_measures
        lines = []
        ml_eq = self.estimation != "ML" and fm is not None   # recursive, linear: OLS is the ML fit
        if self.estimation == "ML":
            lines.append("PathModel %s after Fisher scoring (full-information ML)"
                         % ("converged" if self.converged else "did NOT converge"))
        elif ml_eq:
            lines.append("PathModel fitted equation by equation (least squares = ML for this recursive model)")
        else:
            lines.append("PathModel fitted equation by equation")
        lines.append("")
        lines.append(_kv("Estimator", "ML" if (self.estimation == "ML" or ml_eq) and self.test == "z" else
                         "OLS + GAM (REML/GCV)" if self.smooth else "OLS"))
        lines.append(_kv("Optimization method", "Fisher scoring" if self.estimation == "ML" else
                         "least squares per equation"))
        lines.append(_kv("Number of model parameters", self.npar))
        lines.append("")
        if isinstance(self.nobs, dict):
            ns = {k: len(v) for k, v in self.nobs.items()}
            if len(set(ns.values())) == 1:
                lines.append(_kv("Number of observations", next(iter(ns.values()))))
            else:
                lines.append("  Number of observations per equation:")
                for k, v in ns.items():
                    lines.append(_kv("  " + k, v))
        else:
            lines.append(_kv("Number of observations", int(self.nobs)))
        if fm is not None:
            lines += ["", "Model Test User Model:", "",
                      _kv("Test statistic", float(fm["chisq"])), _kv("Degrees of freedom", int(fm["df"]))]
            if fm["df"] > 0:
                lines.append(_kv("P-value (Chi-square)", float(fm["pvalue"])))
            lines += ["", "Model Test Baseline Model:", "",
                      _kv("Test statistic", float(fm["baseline.chisq"])),
                      _kv("Degrees of freedom", int(fm["baseline.df"])),
                      _kv("P-value", float(fm["baseline.pvalue"])),
                      "", "User Model versus Baseline Model:", "",
                      _kv("Comparative Fit Index (CFI)", float(fm["cfi"])),
                      _kv("Tucker-Lewis Index (TLI)", float(fm["tli"])),
                      "", "Loglikelihood and Information Criteria:", "",
                      _kv("Loglikelihood user model (H0)", float(fm["logl"])),
                      _kv("Loglikelihood unrestricted model (H1)", float(fm["unrestricted.logl"])),
                      "",
                      _kv("Akaike (AIC)", float(fm["aic"])),
                      _kv("Bayesian (BIC)", float(fm["bic"])),
                      _kv("Sample-size adjusted Bayesian (SABIC)", float(fm["bic2"])),
                      "", "Root Mean Square Error of Approximation:", "",
                      _kv("RMSEA", float(fm["rmsea"])),
                      _kv("90 Percent confidence interval - lower", float(fm["rmsea.ci.lower"])),
                      _kv("90 Percent confidence interval - upper", float(fm["rmsea.ci.upper"])),
                      _kv("P-value H_0: RMSEA <= 0.050", float(fm["rmsea.pvalue"])),
                      _kv("P-value H_0: RMSEA >= 0.080", float(fm["rmsea.notclose.pvalue"])),
                      "", "Standardized Root Mean Square Residual:", "",
                      _kv("SRMR", float(fm["srmr"]))]
        info = {"expected": ("Standard", "Expected"), "observed": ("Standard", "Observed"),
                "observed (per equation)": ("Standard", "Observed (per equation)"),
                "lm": ("lm (t tests)", "Per equation"), "none": ("None", "")}.get(self.information, ("", ""))
        lines += ["", "Parameter Estimates:", "", _kv("Standard errors", info[0])]
        if info[1]:
            lines.append(_kv("Information", info[1]))
        dotted = set(P.lhs[P.op.isin(["~", "~~", "~1"])])
        blocks, name_w = lavaan_blocks(P, self.test, dotted)
        lines += blocks
        if self.smooth:
            from ._rprint import print_coefmat
            lines += ["", "Smooth terms (mgcv summary.gam; Wood 2013 tests):"]
            for resp, (tab, _) in self.smooth.items():
                lines.append("  %s ~" % resp)
                txt = print_coefmat(tab.to_numpy(), list(tab.index), list(tab.columns), has_pvalue=True,
                                    cs_ind=[0], legend=False)
                lines += ["    " + ln for ln in txt.split("\n")]
        if self.rsquare is not None and len(self.rsquare):
            lines += ["", "R-Square:", " " * name_w + "%9s" % "Estimate"]
            for k, v in self.rsquare.items():
                lines.append(("    %s" % k).ljust(name_w) + "%9s" % _f3(v))
        if self.effects:
            labs = [len("    %s <- %s" % (r["lhs"], r["rhs"])) + 1 for E in self.effects.values() if E is not None
                    for _, r in E.iterrows()]
            name_w = max([name_w] + labs)
            for kind in ("indirect", "total"):
                E = self.effects.get(kind)
                if E is None or not len(E):
                    continue
                lines += ["", "%s Effects:" % kind.capitalize(),
                          " " * name_w + "%9s%9s%9s%9s" % ("Estimate", "Std.Err", "z-value", "P(>|z|)")]
                for _, r in E.iterrows():
                    lab = "    %s <- %s" % (r["lhs"], r["rhs"])
                    lines.append(lab.ljust(name_w) + "%9s%9s%9s%9s" % (_f3(r["est"]), _f3(r["se"]), _f3(r["z"]),
                                                                       _f3(r["pvalue"])))
        lines.append("")
        return "\n".join(lines)

    __repr__ = __str__
