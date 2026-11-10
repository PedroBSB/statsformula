"""Path models: a system of regression equations written as a multi-line formula.

::

    model = '''
      # mediation
      m  ~ x + z          # one equation per line
      y  ~ m + x +
           log(w)         # continuation lines (trailing / leading operator, open parentheses)
      y2 + y3 ~ m         # lavaan style: one equation per left-hand-side variable
    '''

Every line is an ordinary R formula (transformations, factors, interactions,
``poly``/``ns``/``bs``, mgcv smooths, ...).  As in lavaan, ``y1 + y2 ~ x`` (and
``cbind(y1, y2) ~ x``) define one equation per response, and several lines with the
same response are merged (``y ~ x1`` and ``y ~ x2`` give ``y ~ x1 + x2``).
Residual covariances between endogenous variables are written as in lavaan,
``y1 ~~ y2`` (or ``y1 ~~ y2 + y3``); they are returned by :func:`parse_path_system`
(:func:`parse_path_model` rejects them).  Latent variables (``=~``), defined
parameters (``:=``) and constraints are not supported.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ._parser import Call, Sym, all_vars, deparse, is_call, parse
from ._rtypes import RError

__all__ = ["PathEquation", "parse_path_model", "parse_path_system", "path_order"]

_BINARY_END = re.compile(r"(\+|-|\*|/|:|\^|~|,|\||%in%|\()\s*$")
_BINARY_START = re.compile(r"^\s*(\+|\*|/|:|\^|\||%in%|,|\))")
_NOT_REGRESSION = [("=~", "latent variable definitions (=~)"), (":=", "defined parameters (:=)"),
                   ("<~", "composites (<~)"), ("~*~", "scaling factors (~*~)")]


class PathEquation:
    """One equation of a path model: ``response ~ rhs``."""

    __slots__ = ("response", "lhs", "rhs", "formula", "predictors")

    def __init__(self, response: str, lhs: str, rhs: str, predictors: List[str]):
        self.response = response      # variable name (or the deparsed transformed lhs)
        self.lhs = lhs                # deparsed left-hand side, e.g. "log(y)"
        self.rhs = rhs
        self.formula = "%s ~ %s" % (lhs, rhs)
        self.predictors = predictors  # variables read by the right-hand side

    def __repr__(self):
        return "<PathEquation %s>" % self.formula


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
            out.append(ch)
        elif ch in "#!":
            break
        else:
            out.append(ch)
    return "".join(out)


def _depth(s: str) -> int:
    d, quote = 0, None
    for ch in s:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
    return d


def _split_top(s: str, sep: str) -> List[str]:
    parts, buf, d, quote = [], [], 0, None
    for ch in s:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
        elif ch == sep and d <= 0:  # d < 0: closing a parenthesis opened on an earlier line
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return parts


def split_statements(model: str) -> List[str]:
    """Split a multi-line model into statements (comments removed, continuation lines joined)."""
    lines = []
    for raw in model.splitlines():
        for piece in _split_top(_strip_comment(raw), ";"):
            if piece.strip():
                lines.append(piece.strip())
    stmts: List[str] = []
    for ln in lines:
        if stmts and (_depth(stmts[-1]) > 0 or _BINARY_END.search(stmts[-1]) or _BINARY_START.match(ln)
                      or "~" not in ln):
            stmts[-1] = stmts[-1] + " " + ln
        else:
            stmts.append(ln)
    return stmts


def _lhs_names(node) -> Optional[List[str]]:
    """Variables of an lhs ``y1 + y2 + ...`` or ``cbind(y1, y2)``; None for other expressions."""
    if isinstance(node, Sym):
        return [node.name]
    if is_call(node, "+") and len(node.args) == 2:
        a, b = _lhs_names(node.argv[0]), _lhs_names(node.argv[1])
        return None if a is None or b is None else a + b
    if is_call(node, "cbind") and all(isinstance(a, Sym) for a in node.argv):
        return [a.name for a in node.argv]
    return None


def _cov_terms(rhs: str, st: str) -> List[str]:
    names = []
    for part in rhs.split("+"):
        part = part.strip()
        if not re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*|`[^`]+`", part):
            raise RError("a covariance statement is 'y1 ~~ y2 + y3' with plain variable names: %r" % st)
        names.append(part.strip("`"))
    return names


def parse_path_system(model: str) -> Tuple[List[PathEquation], List[Tuple[str, str]]]:
    """The equations and the residual covariances ``(a, b)`` (from ``a ~~ b``) of a
    multi-line path model."""
    if not isinstance(model, str):
        raise TypeError("a path model is a string with one formula per line")
    order: List[str] = []
    eqs: Dict[str, Tuple[str, List[str]]] = {}
    covs: List[Tuple[str, str]] = []
    for st in split_statements(model):
        if "~~" in st:
            lhs, rhs = st.split("~~", 1)
            a = _cov_terms(lhs, st)
            if len(a) != 1:
                raise RError("a covariance statement has one variable on the left: %r" % st)
            covs.extend((a[0], b) for b in _cov_terms(rhs, st) if b != a[0])
            continue
        for op, what in _NOT_REGRESSION:
            if op in st:
                raise RError("path models contain regressions and residual covariances only; %s are not "
                             "supported: %r (use statsformula.sem for full lavaan models)" % (what, st))
        node = parse(st)
        if not (is_call(node, "~") and len(node.args) == 2):
            raise RError("each statement of a path model must be a formula 'response ~ predictors': %r" % st)
        lhs, rhs = node.argv
        names = _lhs_names(lhs)
        rhs_txt = deparse(rhs)
        targets = [(n, n) for n in names] if names is not None else [(deparse(lhs), deparse(lhs))]
        for resp, lhs_txt in targets:
            if resp in eqs:
                eqs[resp] = (eqs[resp][0], eqs[resp][1] + [rhs_txt])
            else:
                order.append(resp)
                eqs[resp] = (lhs_txt, [rhs_txt])
    if not order:
        raise RError("the path model has no equations")
    out = []
    for resp in order:
        lhs_txt, rhss = eqs[resp]
        rhs = " + ".join(rhss)
        f = parse("%s ~ %s" % (lhs_txt, rhs))
        preds = [v for v in all_vars(f.argv[1])]
        out.append(PathEquation(resp, lhs_txt, rhs, preds))
    endo = set(order)
    for a, b in covs:
        bad = [v for v in (a, b) if v not in endo]
        if bad:
            raise RError("residual covariances are between endogenous variables (responses); %s is not one "
                         "(covariances of exogenous variables are taken from the data)" % bad[0])
    return out, covs


def parse_path_model(model: str) -> List[PathEquation]:
    """The equations of a multi-line path model (see the module docstring)."""
    eqs, covs = parse_path_system(model)
    if covs:
        raise RError("path models contain regressions only here; residual covariances (~~) are supported by "
                     "parse_path_system() and PathModel")
    return eqs


def path_order(equations: List[PathEquation]) -> Optional[List[int]]:
    """Topological order of the equations (upstream first); ``None`` if the model is
    non-recursive (feedback loops)."""
    idx = {e.response: i for i, e in enumerate(equations)}
    deps = [[idx[v] for v in e.predictors if v in idx and v != e.response] for e in equations]
    if any(e.response in e.predictors for e in equations):
        return None
    done, order, state = set(), [], {}

    def visit(i):
        if state.get(i) == 1:
            return False
        if i in done:
            return True
        state[i] = 1
        for j in deps[i]:
            if not visit(j):
                return False
        state[i] = 2
        done.add(i)
        order.append(i)
        return True

    for i in range(len(equations)):
        if not visit(i):
            return None
    return order
