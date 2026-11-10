"""Bridges to Python SEM packages: semopy, pySEM and pysemforge.

None of these packages evaluates R expressions inside the model, and they differ in how much
lavaan syntax they understand.  This module

1. rewrites a lavaan model whose terms may be R expressions (``log(w)``, factors,
   ``poly(z, 2)``, ``x:z``, ``I(x^2)``, ...) into plain lavaan syntax, building the columns with
   statsformula (stateful: :meth:`SemData.transform` re-applies them to new data);
2. expands it with :func:`statsformula.lavaanify` into lavaan's full parameter table, so every
   backend fits the *same* model lavaan would (first loading fixed to 1, residual variances,
   exogenous covariances, ``auto.cov.y``, ...);
3. translates that table into each backend's input, fits it, and maps the estimates back to a
   lavaan-style table (``lhs``, ``op``, ``rhs``, ``label``, ``est``) with ``:=`` parameters
   evaluated.

::

    from statsformula.sem import fit_semopy, fit_pysem, fit_pysemforge, sem_data

    model = '''
      ind =~ x1 + x2 + x3
      m ~ a*ind + log(w) + f        # R expressions are fine
      y ~ b*m + ind
      ab := a*b
    '''
    fit = fit_semopy(model, df)     # or fit_pysem(model, df), fit_pysemforge(model, df)
    fit.estimates                   # lhs/op/rhs/label/est, as lavaan::parameterEstimates()
    fit.defined                     # {"ab": ...}

The backends are optional: ``pip install semopy``, ``pip install pysemforge``, pySEM from
https://github.com/twoertwein/pySEM (needs PyTorch).
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._path import _depth, _split_top, _strip_comment
from ._rtypes import RError

__all__ = ["SemData", "SemFit", "sem_data", "sem_partable", "to_semopy", "fit_semopy", "to_pysem",
           "fit_pysem", "to_pysemforge", "fit_pysemforge", "evaluate_defined"]

# lavaan operators, longest first
_OPS = ["=~", "~~", ":=", "==", "<~", "~*~", "|~", "~", "<", ">"]
_IDENT = re.compile(r"^[A-Za-z.][A-Za-z0-9._]*$")
_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_MODIFIER_CALL = re.compile(r"^(start|equal|label|c|NA)\s*\(")


# ---------------------------------------------------------------------------
# statement splitting and operator detection
# ---------------------------------------------------------------------------

def _find_op(stmt: str) -> Tuple[Optional[str], int]:
    d, quote, i = 0, None, 0
    while i < len(stmt):
        ch = stmt[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
        elif d == 0:
            for op in _OPS:
                if stmt.startswith(op, i):
                    return op, i
        i += 1
    return None, -1


def _statements(model: str) -> List[str]:
    lines = []
    for raw in model.splitlines():
        for piece in _split_top(_strip_comment(raw), ";"):
            if piece.strip():
                lines.append(piece.strip())
    out: List[str] = []
    for ln in lines:
        cont = out and (_depth(out[-1]) > 0 or re.search(r"(\+|\*|-|/|:|\^|,|\(|~|=)\s*$", out[-1])
                        or re.match(r"^\s*(\+|\*|/|:|\^|,|\))", ln) or _find_op(ln)[0] is None)
        if cont:
            out[-1] = out[-1] + " " + ln
        else:
            out.append(ln)
    return out


def _split_modifier(term: str, data_cols, latent) -> Tuple[Optional[str], str]:
    """``a*x`` -> ("a", "x") when ``a`` is a lavaan modifier (label, number, NA, start(),
    equal(), c(...)) rather than a variable; R crossings like ``x*z`` are left alone."""
    parts = _split_top(term, "*")
    if len(parts) < 2:
        return None, term.strip()
    mod, rest = parts[0].strip(), "*".join(parts[1:]).strip()
    if _NUM.match(mod) or mod == "NA" or mod[:1] in "\"'" or _MODIFIER_CALL.match(mod):
        return mod, rest
    if _IDENT.match(mod) and mod not in data_cols and mod not in latent:
        return mod, rest
    return None, term.strip()


def _sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    if not s or s[0].isdigit():
        s = "v_" + s
    return s


# ---------------------------------------------------------------------------
# SemData: rewritten model + data
# ---------------------------------------------------------------------------

class SemData:
    """A lavaan model rewritten with plain variable names plus the matching data.

    Attributes: ``model`` (plain lavaan syntax, ``:=`` lines included), ``data``,
    ``defined`` (list of ``(name, expression)``), ``constraints`` (``==``, ``<``, ``>``
    statements), ``columns`` (term -> new column names), ``observed`` (data columns used).
    """

    def __init__(self, model, data, defined, constraints, columns, specs, plain, observed):
        self.model = model
        self.data = data
        self.defined = defined
        self.constraints = constraints
        self.columns = columns
        self._specs = specs
        self._plain = plain
        self.observed = observed

    @property
    def model_without_defined(self) -> str:
        return "\n".join(ln for ln in self.model.splitlines() if ":=" not in ln)

    def transform(self, newdata) -> pd.DataFrame:
        """The same columns for new data (factor levels, ``poly``/``ns``/``scale`` reused)."""
        newdata = pd.DataFrame(newdata)
        out = {c: newdata[c] for c in self._plain if c in newdata.columns}
        for spec, rename in self._specs:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="contrasts dropped from factor")
                M = spec.get_model_matrix(newdata, na_action="na.pass")
            for old, new in rename.items():
                out[new] = np.asarray(M[old], dtype=float)
        return pd.DataFrame(out, index=newdata.index)[[c for c in self.observed if c in out]]

    def __repr__(self):
        return "<SemData %d columns>\n%s" % (self.data.shape[1], self.model)


def sem_data(model: str, data, contrasts=None, na_action: str = "na.omit", env=None) -> SemData:
    """Rewrite R-expression terms of a lavaan model into plain columns (see module docs)."""
    from ._api import model_matrix
    data = pd.DataFrame(data)
    env = {} if env is None else env
    stmts = _statements(model)
    parsed = []
    latent = set()
    for st in stmts:
        op, i = _find_op(st)
        if op is None:
            raise RError("not a lavaan statement: %r" % st)
        lhs, rhs = st[:i].strip(), st[i + len(op):].strip()
        parsed.append((op, lhs, rhs))
        if op in ("=~", "<~"):
            latent.add(lhs)
    data_cols = set(map(str, data.columns))
    used_names = set(data_cols) | latent
    numeric = {c for c in data_cols if pd.api.types.is_numeric_dtype(data[c]) and not
               pd.api.types.is_bool_dtype(data[c])}
    out_lines, defined, constraints = [], [], []
    columns: Dict[str, List[str]] = {}
    specs = []
    new_cols: Dict[str, np.ndarray] = {}
    plain = []

    def use_plain(v):
        if v in data_cols and v not in plain:
            plain.append(v)

    def expand(terms: List[str], allow_multi: bool) -> List[str]:
        """Return rewritten terms; expression terms become new columns."""
        items = []
        exprs = []
        for t in terms:
            mod, expr = _split_modifier(t, data_cols, latent)
            is_plain = (_IDENT.match(expr) and (expr in latent or expr in numeric or expr not in data_cols)) \
                or expr == "1" or _NUM.match(expr)
            items.append((mod, expr, bool(is_plain)))
            if not is_plain and expr not in columns:
                exprs.append(expr)
        if exprs:
            f = "~ " + " + ".join(exprs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                M = model_matrix(f, data, contrasts=contrasts, na_action="na.pass", env=env)
            labels = M.model_spec.terms.term_labels
            rename = {}
            label_cols = {}
            for k, lab in enumerate(labels, start=1):
                cols = [c for c, a in zip(M.columns, M.assign) if a == k]
                names = []
                for c in cols:
                    if lab == c and c in numeric:  # a plain numeric variable (x in x*z): reuse it
                        use_plain(c)
                        names.append(c)
                        continue
                    nm = _sanitize(c)
                    base, j = nm, 2
                    while nm in used_names:
                        nm = "%s_%d" % (base, j)
                        j += 1
                    used_names.add(nm)
                    rename[c] = nm
                    new_cols[nm] = np.asarray(M[c], dtype=float)
                    names.append(nm)
                label_cols[lab] = names
            # each user term (``x*z`` expands to x, z, x:z) -> the columns of its R terms
            from ._api import formula as _formula
            for e in exprs:
                cols_e = []
                for lab in _formula("~ " + e).terms().term_labels:
                    for nm in label_cols.get(lab, []):
                        if nm not in cols_e:
                            cols_e.append(nm)
                columns[e] = cols_e
            columns.update(label_cols)
            specs.append((M.model_spec, rename))
        out = []
        for mod, expr, is_plain in items:
            if is_plain:
                use_plain(expr)
                out.append(expr if mod is None else "%s*%s" % (mod, expr))
                continue
            names = columns.get(expr)
            if names is None:
                raise RError("could not expand term %r" % expr)
            if mod is not None and len(names) > 1:
                raise RError("modifier %r applies to %r, which expands to %d columns (%s); "
                             "label each column instead" % (mod, expr, len(names), ", ".join(names)))
            if len(names) > 1 and not allow_multi:
                raise RError("%r expands to several columns; not allowed here" % expr)
            out.extend(names if mod is None else ["%s*%s" % (mod, names[0])])
        return out

    for op, lhs, rhs in parsed:
        if op == ":=":
            defined.append((lhs, rhs))
            out_lines.append("%s := %s" % (lhs, rhs))
            continue
        if op in ("==", "<", ">"):
            constraints.append("%s %s %s" % (lhs, op, rhs))
            out_lines.append(constraints[-1])
            continue
        if op in ("~", "=~", "<~"):
            lhs_terms = _split_top(lhs, "+")
            new_lhs = expand([t.strip() for t in lhs_terms], allow_multi=False) if op == "~" else [lhs]
            rhs_terms = [t.strip() for t in _split_top(rhs, "+") if t.strip()]
            new_rhs = expand(rhs_terms, allow_multi=True)
            out_lines.append("%s %s %s" % (" + ".join(new_lhs), op, " + ".join(new_rhs)))
            continue
        if op == "~~":
            for v in (lhs,):
                if not _IDENT.match(v):
                    raise RError("covariances (~~) need variable names, not %r" % v)
            rhs_new = []
            for t in _split_top(rhs, "+"):
                mod, expr = _split_modifier(t.strip(), data_cols, latent)
                expr = columns.get(expr, [expr])[0] if expr in columns else expr
                use_plain(expr)
                rhs_new.append(expr if mod is None else "%s*%s" % (mod, expr))
            use_plain(lhs)
            out_lines.append("%s ~~ %s" % (columns.get(lhs, [lhs])[0], " + ".join(rhs_new)))
            continue
        out_lines.append("%s %s %s" % (lhs, op, rhs))
    frame = {c: data[c] for c in plain}
    for nm, v in new_cols.items():
        frame[nm] = v
    df = pd.DataFrame(frame, index=data.index)
    observed = list(df.columns)
    if na_action == "na.omit":
        df = df.dropna()
    elif na_action == "na.fail" and df.isna().any().any():
        raise RError("missing values in the model variables")
    return SemData("\n".join(out_lines), df, defined, constraints, columns, specs, plain, observed)


def sem_partable(sd: SemData, preset: str = "sem", **options) -> pd.DataFrame:
    """lavaan's full parameter table for the rewritten model (``lavaanify``)."""
    from ._lavaan import lavaanify
    return lavaanify(sd.model, preset=preset, **options)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

def evaluate_defined(defined, label_values: Dict[str, float]) -> Dict[str, float]:
    """Evaluate ``:=`` expressions (in order, so later ones may use earlier ones)."""
    from ._eval import EvalContext
    from ._parser import parse
    vals = dict(label_values)
    out = {}
    for name, expr in defined:
        v = EvalContext({k: np.array([x]) for k, x in vals.items()}, {}).eval(parse(expr))
        out[name] = float(np.asarray(v, dtype=float).ravel()[0])
        vals[name] = out[name]
    return out


class SemFit:
    """A model fitted by a SEM backend, with lavaan-style estimates.

    ``estimates``: DataFrame ``lhs, op, rhs, label, free, est`` (lavaan naming, one row per
    lavaan parameter); ``defined``: ``:=`` values; ``backend``: the backend's own fitted
    object; ``sem_data``: the rewritten model and data.
    """

    def __init__(self, estimates, defined, backend, sem_data, name, fit_stats=None):
        self.estimates = estimates
        self.defined = defined
        self.backend = backend
        self.sem_data = sem_data
        self.name = name
        self.fit_stats = fit_stats

    def summary(self):
        """lavaan-style summary: fit statistics (when the backend provides them) and the
        estimates grouped as Latent Variables, Regressions, Covariances, Intercepts,
        Variances and Defined Parameters, with the backend's standard errors and z
        tests where it computes them (semopy)."""
        from ._summary import SemSummary
        return SemSummary(self)

    def __repr__(self):
        return "<SemFit %s: %d parameters>" % (self.name, len(self.estimates))


def _finish(pt, values, sd, backend, name, se=None, fit_stats=None):
    est = pt[pt.op.isin(["=~", "~", "~~", "~1"])][["lhs", "op", "rhs", "label", "free"]].copy()
    est["est"] = values
    if se is not None:
        from scipy import stats
        est["se"] = se
        with np.errstate(divide="ignore", invalid="ignore"):
            est["z"] = est["est"] / est["se"]
        est["pvalue"] = 2 * stats.norm.sf(np.abs(est["z"]))
    est = est.reset_index(drop=True)
    labels = {}
    for lab, v in zip(est.label, est.est):
        if lab:
            labels[lab] = v
    defined = evaluate_defined(sd.defined, labels) if sd.defined else {}
    return SemFit(est, defined, backend, sd, name, fit_stats)


def _check_constraints(sd, pt, backend):
    if sd.constraints:
        raise RError("%s does not support constraints (%s)" % (backend, "; ".join(sd.constraints)))


# ---------------------------------------------------------------------------
# semopy
# ---------------------------------------------------------------------------

def _num(v):
    return repr(float(v))


def to_semopy(model: str, data, preset: str = "sem", contrasts=None, na_action="na.omit", env=None,
              **options) -> Tuple[str, pd.DataFrame, SemData, pd.DataFrame]:
    """``(semopy_description, data, sem_data, partable)``: lavaan's full model written explicitly
    in semopy syntax (labels, fixed values and every (co)variance spelled out)."""
    sd = sem_data(model, data, contrasts=contrasts, na_action=na_action, env=env)
    pt = sem_partable(sd, preset=preset, **options)
    _check_constraints(sd, pt, "semopy")
    if (pt.op == "~1").any():
        raise RError("semopy.Model has no mean structure; use meanstructure=False")
    lines = []
    for _, r in pt[pt.op.isin(["=~", "~", "~~"])].iterrows():
        if r.free == 0 and not np.isnan(r.ustart):
            term = "%s*%s" % (_num(r.ustart), r.rhs)
        elif r.label:
            term = "%s*%s" % (r.label, r.rhs)
        elif r.op == "=~":
            # semopy fixes the first loading of a factor whose loadings are all unlabelled;
            # lavaan's table already says which loadings are free (std.lv, NA*x), so label them
            term = "sf_p%d*%s" % (r.id, r.rhs)
        else:
            term = r.rhs
        lines.append("%s %s %s" % (r.lhs, r.op, term))
    return "\n".join(lines), sd.data, sd, pt


def fit_semopy(model: str, data, preset: str = "sem", obj: str = "MLW", solver: str = "SLSQP",
               solver_options: Optional[dict] = None, contrasts=None, na_action="na.omit", env=None,
               **options) -> SemFit:
    """Fit with ``semopy.Model`` (ML by default) and return lavaan-style estimates.

    ``solver_options`` go to scipy's optimiser; the default tightens SLSQP's tolerance
    (``ftol=1e-12``) so the estimates agree with lavaan's to ~1e-5 rather than ~1e-3.
    ``**options`` are lavaanify options (``std_lv=True``, ``auto_cov_y=False``, ...).
    """
    import semopy
    desc, df, sd, pt = to_semopy(model, data, preset=preset, contrasts=contrasts, na_action=na_action,
                                 env=env, **options)
    m = semopy.Model(desc)
    # semopy 2.3 loops forever computing start values for a factor with no fixed loading
    # (std.lv = TRUE): its first_manifs entry is None.  The entry only seeds start values.
    fm = getattr(m, "first_manifs", None)
    if fm is not None:
        for lat in dict.fromkeys(pt.lhs[pt.op == "=~"]):
            if fm.get(lat) is None:
                inds = [v for v in pt.rhs[(pt.op == "=~") & (pt.lhs == lat)] if v in m.vars["observed"]]
                if inds:
                    fm[lat] = inds[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        so = {"ftol": 1e-12, "maxiter": 10000} if solver_options is None and solver == "SLSQP" else solver_options
        m.fit(df, obj=obj, solver=solver, options=so)
        ins = m.inspect()
    look, look_se = {}, {}
    for _, r in ins.iterrows():
        look[(r["lval"], r["op"], r["rval"])] = float(r["Estimate"])
        try:
            look_se[(r["lval"], r["op"], r["rval"])] = float(r["Std. Err"])
        except (TypeError, ValueError, KeyError):
            pass
    vals, ses = [], []
    for _, r in pt[pt.op.isin(["=~", "~", "~~", "~1"])].iterrows():
        if r.op == "=~":
            keys = [(r.rhs, "~", r.lhs), (r.lhs, "=~", r.rhs)]
        elif r.op == "~~":
            keys = [(r.lhs, "~~", r.rhs), (r.rhs, "~~", r.lhs)]
        else:
            keys = [(r.lhs, r.op, r.rhs)]
        v = next((look[k] for k in keys if k in look), None)
        if v is None:
            v = float(r.ustart) if not np.isnan(r.ustart) else np.nan
        vals.append(v)
        ses.append(next((look_se[k] for k in keys if k in look_se), np.nan) if r.free else np.nan)
    stats_ = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = semopy.calc_stats(m).iloc[0]
        stats_ = {"chisq": float(st["chi2"]), "df": float(st["DoF"]), "pvalue": float(st["chi2 p-value"]),
                  "baseline.chisq": float(st["chi2 Baseline"]), "baseline.df": float(st["DoF Baseline"]),
                  "cfi": float(st["CFI"]), "tli": float(st["TLI"]), "rmsea": float(st["RMSEA"]),
                  "ntotal": int(len(df)), "npar": int(len(m.param_vals))}
    except Exception:
        pass
    return _finish(pt, vals, sd, m, "semopy", se=ses, fit_stats=stats_)


# ---------------------------------------------------------------------------
# pySEM (twoertwein/pySEM): LISREL all-y matrices
# ---------------------------------------------------------------------------

def to_pysem(model: str, data, preset: str = "sem", meanstructure: bool = False, contrasts=None,
             na_action="na.omit", env=None, **options) -> Tuple[dict, SemData, pd.DataFrame, dict]:
    """``(kwargs, sem_data, partable, names)`` where ``pysem.sem.SEM(**kwargs)`` is lavaan's model.

    pySEM regresses latent variables only (LISREL all-y), so every observed variable taking part
    in a regression or in a covariance with a latent variable gets a single-indicator stand-in
    ``"eta:<name>"`` (loading 1, residual variance 0).  ``names`` maps lavaan names to pySEM's.
    """
    sd = sem_data(model, data, contrasts=contrasts, na_action=na_action, env=env)
    if na_action != "na.omit" and sd.data.isna().any().any():
        meanstructure = True  # pySEM switches to FIML: model the means
    pt = sem_partable(sd, preset=preset, meanstructure=meanstructure, **options)
    _check_constraints(sd, pt, "pySEM")
    labs = [lab for lab in pt.label if lab]
    dup = sorted({lab for lab in labs if labs.count(lab) > 1})
    if dup:
        raise RError("pySEM has no equality constraints (labels used twice: %s)" % ", ".join(dup))
    rows = pt[pt.op.isin(["=~", "~", "~~", "~1"])]
    latent = list(dict.fromkeys(rows.lhs[rows.op == "=~"]))
    lat_set = set(latent)
    observed = [c for c in sd.data.columns]
    in_reg = set(rows.lhs[rows.op == "~"]) | set(rows.rhs[rows.op == "~"])
    phantom = [v for v in observed if v in in_reg]
    for _, r in rows[rows.op == "~~"].iterrows():
        a, b = r.lhs, r.rhs
        if (a in lat_set or a in phantom) and b in observed and b not in phantom:
            phantom.append(b)
        if (b in lat_set or b in phantom) and a in observed and a not in phantom:
            phantom.append(a)
    pname = {v: "eta:" + v for v in phantom}

    def L(v):
        return pname.get(v, v)

    lam, beta, psi, theta, alpha, nu = {}, {}, {}, {}, {}, {}
    nan = float("nan")

    def val(r):
        if r.free > 0 or (r.free == 0 and np.isnan(r.ustart)):  # exogenous fixed.x rows: free (same ML value)
            return nan
        return float(r.ustart)

    for v in phantom:
        lam[(v, pname[v])] = 1.0
        theta[(v, v)] = 0.0
        if meanstructure:
            nu[v] = 0.0
    for _, r in rows.iterrows():
        if r.op == "=~":
            if r.rhs in lat_set or r.rhs in phantom:
                beta[(L(r.rhs), r.lhs)] = val(r)
            else:
                lam[(r.rhs, r.lhs)] = val(r)
        elif r.op == "~":
            beta[(L(r.lhs), L(r.rhs))] = val(r)
        elif r.op == "~~":
            a_lat = r.lhs in lat_set or r.lhs in pname
            b_lat = r.rhs in lat_set or r.rhs in pname
            if a_lat and b_lat:
                psi[(L(r.lhs), L(r.rhs))] = val(r)
            elif not a_lat and not b_lat:
                theta[(r.lhs, r.rhs)] = val(r)
            else:
                raise RError("cannot express %s ~~ %s in pySEM" % (r.lhs, r.rhs))
        elif r.op == "~1":
            if r.lhs in lat_set or r.lhs in pname:
                alpha[L(r.lhs)] = val(r)
            else:
                nu[r.lhs] = val(r)
    lat_all = latent + [pname[v] for v in phantom]
    for v in lat_all:  # pySEM defaults latent variances to 1: make the table's choice explicit
        psi.setdefault((v, v), 0.0)
    kwargs = dict(observed=observed, latent=lat_all, data=sd.data, lambda_y=lam, beta=beta, psi=psi,
                  theta=theta)
    if meanstructure:
        for v in lat_all:
            alpha.setdefault(v, 0.0)
        kwargs.update(alpha=alpha, nu=nu)
    return kwargs, sd, pt, pname


def _lbfgs(sem, max_iter, warmup=1500):
    """Minimise pySEM's ML objective: a short Adam warm-up (pySEM's own optimiser, stepping back
    with a smaller rate whenever the implied covariance stops being positive definite), then
    L-BFGS to convergence."""
    import torch
    params = list(sem.parameters())
    lr = 0.01
    adam = torch.optim.Adam(params, lr=lr)
    good = [p.detach().clone() for p in params]
    for _ in range(warmup):
        adam.zero_grad()
        try:
            loss = sem.forward()
        except Exception:
            with torch.no_grad():
                for p, g in zip(params, good):
                    p.copy_(g)
            lr /= 2
            adam = torch.optim.Adam(params, lr=lr)
            continue
        good = [p.detach().clone() for p in params]
        loss.backward()
        adam.step()
    with torch.no_grad():
        for p, g in zip(params, good):
            p.copy_(g)
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, max_eval=2 * max_iter, tolerance_grad=1e-10,
                            tolerance_change=1e-15, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        try:
            loss = sem.forward()
        except Exception:  # torch.linalg.LinAlgError: step left the positive definite region
            return sum((p * 0).sum() for p in params) + 1e10
        loss.backward()
        return loss

    for _ in range(5):
        before = float(sem.forward().detach())
        opt.step(closure)
        if abs(before - float(sem.forward().detach())) < 1e-13:
            break
    return float(sem.forward().detach())


def _pysem_starts(sem, kwargs, pname):
    """Start the stand-ins' (co)variances at the sample values (pySEM starts psi at I), so the
    implied covariance starts close to the data."""
    import torch
    S = _sample_cov_frame(kwargs["data"])
    lat = kwargs["latent"]
    idx = {v: i for i, v in enumerate(lat)}
    inv = {p: v for v, p in pname.items()}
    with torch.no_grad():
        for a in pname.values():
            for b in pname.values():
                sem.psi.data[idx[a], idx[b]] = float(S.loc[inv[a], inv[b]])


def _sample_cov_frame(df):
    return df.cov() * (len(df) - 1) / len(df)


def fit_pysem(model: str, data, preset: str = "sem", meanstructure: bool = False, optimizer: str = "lbfgs",
              max_iter: int = 5000, epochs: int = 15000, lr: float = 0.002, contrasts=None, na_action="na.omit",
              env=None, **options) -> SemFit:
    """Fit with pySEM (``pysem.sem.SEM``) and return lavaan-style estimates.

    ``optimizer="lbfgs"`` (default) minimises pySEM's ML objective with PyTorch's L-BFGS to
    convergence; ``"adam"`` calls pySEM's own ``fit(epochs, lr)``.
    """
    from pysem.sem import SEM
    kwargs, sd, pt, pname = to_pysem(model, data, preset=preset, meanstructure=meanstructure,
                                     contrasts=contrasts, na_action=na_action, env=env, **options)
    sem = SEM(**kwargs)
    _pysem_starts(sem, kwargs, pname)
    if optimizer == "lbfgs":
        _lbfgs(sem, max_iter)
    else:
        if not hasattr(np, "Inf"):
            raise RError("pySEM's own fit() uses np.Inf, removed in NumPy 2; use optimizer='lbfgs'")
        sem.fit(epochs=epochs, lr=lr)
    mats = sem.summary(verbose=False)
    lat_set = set(pt.lhs[pt.op == "=~"])

    def L(v):
        return pname.get(v, v)

    def is_lat(v):
        return v in lat_set or v in pname

    vals = []
    for _, r in pt[pt.op.isin(["=~", "~", "~~", "~1"])].iterrows():
        if r.op == "=~":
            v = mats["beta"].loc[L(r.rhs), r.lhs] if is_lat(r.rhs) else mats["lambda_y"].loc[r.rhs, r.lhs]
        elif r.op == "~":
            v = mats["beta"].loc[L(r.lhs), L(r.rhs)]
        elif r.op == "~~":
            v = (mats["psi"].loc[L(r.lhs), L(r.rhs)] if is_lat(r.lhs)
                 else mats["theta"].loc[r.lhs, r.rhs])
        else:
            v = (mats["alpha"].loc[L(r.lhs)].iloc[0] if is_lat(r.lhs) else mats["nu"].iloc[
                list(mats["nu"].index).index(r.lhs), 0]) if "alpha" in mats else np.nan
        vals.append(float(v))
    return _finish(pt, vals, sd, sem, "pySEM", fit_stats={"ntotal": int(len(sd.data))})


# ---------------------------------------------------------------------------
# pysemforge
# ---------------------------------------------------------------------------

def to_pysemforge(model: str, data, contrasts=None, na_action="na.omit", env=None) -> Tuple[str, pd.DataFrame, SemData]:
    """``(syntax, data, sem_data)`` for ``pysemforge.Model(syntax).fit(data)``.

    pysemforge's parser reads plain variable names only, so labels and fixed values
    (``a*x``, ``1*x``) and ``:=`` lines are removed (``:=`` is evaluated by
    :func:`fit_pysemforge` from the semopy-equivalent estimates instead).
    """
    sd = sem_data(model, data, contrasts=contrasts, na_action=na_action, env=env)
    out = []
    for st in sd.model.splitlines():
        op, i = _find_op(st)
        if op in (":=",):
            continue
        if op in ("==", "<", ">"):
            raise RError("pysemforge does not support the constraint %r" % st)
        lhs, rhs = st[:i].strip(), st[i + len(op):].strip()
        terms = []
        for t in _split_top(rhs, "+"):
            mod, expr = _split_modifier(t.strip(), set(sd.data.columns), set())
            if mod is not None and (_NUM.match(mod) or mod == "NA" or _MODIFIER_CALL.match(mod)):
                raise RError("pysemforge cannot fix or start parameters (%r)" % t.strip())
            terms.append(expr)
        out.append("%s %s %s" % (lhs, op, " + ".join(terms)))
    return "\n".join(out), sd.data, sd


def fit_pysemforge(model: str, data, estimator: str = "ML", group: Optional[str] = None, contrasts=None,
                   na_action="na.omit", env=None, **fit_kwargs):
    """Fit with ``pysemforge.Model`` on the rewritten model; returns pysemforge's ``SEMResult``
    with the rewritten :class:`SemData` attached as ``.sem_data``.

    pysemforge reports fit indices and an unnamed parameter vector; for named estimates and
    ``:=`` parameters of the same ML model use :func:`fit_semopy`.
    """
    import pysemforge
    from pysemforge import Model
    if getattr(pysemforge, "__version__", "0") in ("0.0.0", "0.1.0"):
        warnings.warn("pysemforge %s: its semopy backend refits a random sample drawn from the covariance "
                      "matrix (not your data) and reports chi-square/df/CFI/TLI/RMSEA as 0/0/1/1/0 because "
                      "semopy.calc_stats fails with a NameError; use statsformula.sem.fit_semopy for the same "
                      "ML model on the actual data" % pysemforge.__version__, RuntimeWarning, stacklevel=2)
    syntax, df, sd = to_pysemforge(model, data, contrasts=contrasts, na_action=na_action, env=env)
    if group is not None:
        df = df.assign(**{group: pd.DataFrame(data).loc[df.index, group]})
    res = Model(syntax).fit(df, estimator=estimator, group=group, **fit_kwargs)
    res.sem_data = sd
    return res
