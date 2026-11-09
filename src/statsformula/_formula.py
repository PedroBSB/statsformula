"""Formula objects.

* :class:`RFormula` mirrors a base R ``formula`` (``y ~ a*b + log(x)``).
* :class:`Formula` mirrors the R package *Formula* (multi-part formulas
  ``y1 | y2 ~ x1 + x2 | z1``; multiple responses ``y1 + y2 ~ x``).
"""

from __future__ import annotations

import sys
from collections import ChainMap
from typing import Any, List, Optional, Sequence, Union

from ._parser import (Call, Const, Node, Num, Str, Sym, all_names as _all_names,
                      all_vars as _all_vars, call, deparse, is_call, node_key, parse)
from ._rtypes import RError
from ._terms import Terms, terms_formula


# ---------------------------------------------------------------------------
# environment capture (like R's formula environment / patsy EvalEnvironment)
# ---------------------------------------------------------------------------

def capture_env(depth: int = 1):
    """Capture the namespace ``depth`` frames above the caller."""
    frame = sys._getframe(depth + 1)
    return ChainMap(dict(frame.f_locals), frame.f_globals)


def _resolve_env(env, depth: int):
    if env is None:
        return capture_env(depth + 1)
    if isinstance(env, int):
        return capture_env(depth + 1 + env)
    return env


# ---------------------------------------------------------------------------
# base R formula
# ---------------------------------------------------------------------------

class RFormula:
    """A base R formula, e.g. ``RFormula("y ~ x + log(z)")``.

    Use :func:`statsformula.formula` to create one with the caller's
    environment attached.
    """

    __slots__ = ("node", "env")

    def __init__(self, x: Union[str, Node, "RFormula"], env=None):
        if isinstance(x, RFormula):
            self.node = x.node
            self.env = env if env is not None else x.env
            return
        if isinstance(x, str):
            node = parse(x)
        elif isinstance(x, Node):
            node = x
        else:
            raise TypeError("cannot make a formula from %r" % (type(x).__name__,))
        if not is_call(node, "~"):
            raise RError("invalid formula %s: not a call to '~'" % deparse(node))
        self.node = node
        self.env = env if env is not None else {}

    # structure ------------------------------------------------------------
    @property
    def lhs(self) -> Optional[Node]:
        return self.node.argv[0] if len(self.node.args) == 2 else None

    @property
    def rhs(self) -> Node:
        return self.node.argv[-1]

    @property
    def has_response(self) -> bool:
        return len(self.node.args) == 2

    def __len__(self):
        return 1 + len(self.node.args)

    def __str__(self):
        return deparse(self.node)

    def __repr__(self):
        return "<formula: %s>" % deparse(self.node)

    def __eq__(self, other):
        if isinstance(other, str):
            other = RFormula(other)
        if not isinstance(other, RFormula):
            return NotImplemented
        return node_key(self.node) == node_key(other.node)

    def __hash__(self):
        return hash(node_key(self.node))

    # R methods ------------------------------------------------------------
    def terms(self, data=None, keep_order: bool = False, specials=None, simplify: bool = False,
              allow_dot_as_name: bool = False) -> Terms:
        from ._model import _data_names
        names = _data_names(data) if data is not None else None
        t = terms_formula(self.node, names, keep_order, specials, allow_dot_as_name, formula_obj=self)
        if simplify:
            t.formula = _fix_formula_object(t, self)
        return t

    def all_vars(self) -> List[str]:
        return _all_vars(self.node)

    def all_names(self, functions: bool = True, unique: bool = False) -> List[str]:
        return _all_names(self.node, functions, unique)

    def update(self, new) -> "RFormula":
        return update_formula(self, new)

    def delete_response(self) -> "RFormula":
        return RFormula(call("~", self.rhs), self.env)


def _fix_formula_object(t: Terms, form: RFormula) -> RFormula:
    """``terms.formula(simplify = TRUE)``: rebuild the formula from term labels."""
    tmp = ["( %s )" % l if "|" in l else l for l in t.term_labels]
    if t.offset:
        tmp = tmp + [t.variables[i - 1] for i in t.offset]
    rhs = " + ".join(tmp) if tmp else "1"
    if not t.intercept:
        rhs = rhs + " - 1"
    if form.has_response:
        new = parse("lhs ~ " + rhs)
        node = Call(Sym("~"), ((None, form.lhs), (None, new.argv[1])))
    else:
        node = parse("~ " + rhs)
    return RFormula(node, form.env)


# ---------------------------------------------------------------------------
# update.formula (C function updateform + ExpandDots)
# ---------------------------------------------------------------------------

from ._parser import expand_dots as _expand_dots


def _updateform(old: RFormula, new: RFormula) -> Node:
    lhs = old.lhs
    rhs = old.rhs
    if new.has_response:
        nl = new.lhs
        nr = new.rhs
        if lhs is not None:
            nl = _expand_dots(nl, lhs)
        nr = _expand_dots(nr, rhs)
        return Call(Sym("~"), ((None, nl), (None, nr)))
    nr = _expand_dots(new.rhs, rhs)
    if lhs is not None:
        return Call(Sym("~"), ((None, lhs), (None, nr)))
    return Call(Sym("~"), ((None, nr),))


def update_formula(old, new) -> RFormula:
    """R's ``update.formula``."""
    old = old if isinstance(old, RFormula) else RFormula(old)
    new = new if isinstance(new, RFormula) else RFormula(new)
    tmp = RFormula(_updateform(old, new), old.env)
    t = tmp.terms(simplify=True)
    return t.formula


# ---------------------------------------------------------------------------
# reformulate / drop.terms / delete.response
# ---------------------------------------------------------------------------

def reformulate(termlabels: Sequence[str], response=None, intercept: bool = True, env=None) -> RFormula:
    """R's ``reformulate``."""
    if isinstance(termlabels, str):
        termlabels = [termlabels]
    termlabels = list(termlabels)
    if intercept and not termlabels:
        termlabels = ["1"]
    termtext = "+".join(termlabels)
    if not intercept:
        termtext = termtext + " - 1" if termtext else " - 1"
    rhs = parse(termtext)
    if response is None:
        node = Call(Sym("~"), ((None, rhs),))
    else:
        if isinstance(response, str):
            try:
                resp = parse(response)
            except Exception:
                import warnings
                warnings.warn("Unparseable 'response' \"%s\"; using as.name(response)" % response)
                resp = Sym(response)
        elif isinstance(response, Node):
            resp = response
        else:
            raise TypeError("response must be a string or expression")
        node = Call(Sym("~"), ((None, resp), (None, rhs)))
    return RFormula(node, env or {})


def delete_response(t: Terms) -> Terms:
    """R's ``delete.response``."""
    import numpy as np
    if not t.response:
        return t
    y = t.response
    nt = t.copy()
    f = t.formula
    if f is not None:
        nt.formula = RFormula(call("~", f.rhs), f.env)
    nt.response = 0
    nt.variable_nodes = t.variable_nodes[:y - 1] + t.variable_nodes[y:]
    nt.variables = t.variables[:y - 1] + t.variables[y:]
    nt.varnames = t.varnames[:y - 1] + t.varnames[y:]
    if t.factors is not None and t.factors.size:
        nt.factors = np.delete(t.factors, y - 1, axis=0)
    nt.term_bits = [((b >> y) << (y - 1)) | (b & ((1 << (y - 1)) - 1)) for b in t.term_bits]
    if t.offset:
        nt.offset = [o - 1 if o > y else o for o in t.offset]
    if t.specials:
        nt.specials = {k: (None if v is None else [b - 1 if b > y else b for b in v]) for k, v in t.specials.items()}
    if t.data_classes:
        nt.data_classes = {k: v for i, (k, v) in enumerate(t.data_classes.items()) if i != y - 1}
    return nt


def drop_terms(t: Terms, dropx: Optional[Sequence[int]] = None, keep_response: bool = False) -> Terms:
    """R's ``drop.terms`` (``dropx`` are 1-based term indices)."""
    if dropx is None:
        return t
    if isinstance(dropx, int):
        dropx = [dropx]
    drop = set(int(i) for i in dropx)
    labels = [l for i, l in enumerate(t.term_labels, start=1) if i not in drop]
    resp = t.formula.lhs if (keep_response and t.response and t.formula is not None) else None
    env = t.formula.env if t.formula is not None else {}
    nf = reformulate(labels, response=resp, intercept=bool(t.intercept), env=env)
    specials = list(t.specials) if t.specials else None
    return nf.terms(specials=specials)


# ---------------------------------------------------------------------------
# Formula package
# ---------------------------------------------------------------------------

def _extract_parts(x: Optional[Node], sep: str = "|") -> List[Node]:
    if x is None:
        return []
    rval: List[Node] = []
    while is_call(x, sep) and len(x.args) == 2:
        rval.insert(0, x.argv[1])
        x = x.argv[0]
    return [x] + rval


def _c_formula(f1: Optional[Node], f2: Optional[Node], sep: str) -> Node:
    if sep == "~":
        if f1 is None:
            return Call(Sym("~"), ((None, f2),))
        return Call(Sym("~"), ((None, f1), (None, f2)))
    return Call(Sym(sep), ((None, f1), (None, f2)))


def _paste_formula(lhs: Sequence[Node], rhs: Sequence[Node], lsep="|", rsep="|") -> Node:
    lsep = [lsep] * max(0, len(lhs) - 1) if isinstance(lsep, str) else list(lsep)
    rsep = [rsep] * max(0, len(rhs) - 1) if isinstance(rsep, str) else list(rsep)
    lval = lhs[0] if lhs else None
    for i in range(1, len(lhs)):
        lval = _c_formula(lval, lhs[i], lsep[i - 1])
    rval = rhs[0] if rhs else Num(0.0)
    for i in range(1, len(rhs)):
        rval = _c_formula(rval, rhs[i], rsep[i - 1])
    return _c_formula(lval, rval, "~")


def _select(n: int, idx) -> List[int]:
    """R index semantics on ``1:n`` (``None`` = all, ``0`` = none, negatives drop)."""
    if idx is None:
        return list(range(n))
    if isinstance(idx, (int,)) and not isinstance(idx, bool):
        idx = [idx]
    idx = list(idx)
    if any(i < 0 for i in idx):
        drop = {-i - 1 for i in idx if i < 0}
        return [i for i in range(n) if i not in drop]
    out = []
    for i in idx:
        if i == 0:
            continue
        if i > n:
            import warnings
            warnings.warn("subscript out of bounds, not all parts available")
            continue
        out.append(i - 1)
    return out


class Formula:
    """Multi-part formula from the R package *Formula*.

    ``Formula("y1 | y2 ~ x1 + x2 | z")`` has 2 LHS parts and 2 RHS parts.
    Part indices (``lhs=``, ``rhs=``) are 1-based as in R; ``0`` selects no
    part and ``None`` selects all parts.
    """

    def __init__(self, x, *more, env=None):
        if isinstance(x, Formula) and not more:
            self.lhs_parts = list(x.lhs_parts)
            self.rhs_parts = list(x.rhs_parts)
            self.env = env if env is not None else x.env
            return
        forms = [x] + list(more)
        lhs: List[Node] = []
        rhs: List[Node] = []
        envs = None
        for f in forms:
            if isinstance(f, Formula):
                lhs += f.lhs_parts
                rhs += f.rhs_parts
                envs = envs or f.env
                continue
            rf = f if isinstance(f, RFormula) else RFormula(f)
            envs = envs or rf.env
            lhs += _extract_parts(rf.lhs)
            rhs += _extract_parts(rf.rhs)
        self.lhs_parts = lhs
        self.rhs_parts = rhs
        self.env = env if env is not None else (envs or {})

    # basic properties -----------------------------------------------------
    def __len__(self):
        raise TypeError("use .length for (n_lhs, n_rhs)")

    @property
    def length(self):
        return (len(self.lhs_parts), len(self.rhs_parts))

    def __str__(self):
        return str(self.formula())

    def __repr__(self):
        return "<Formula: %s>" % self

    def __eq__(self, other):
        if isinstance(other, str):
            other = Formula(other)
        if not isinstance(other, Formula):
            return NotImplemented
        return ([node_key(n) for n in self.lhs_parts] == [node_key(n) for n in other.lhs_parts] and
                [node_key(n) for n in self.rhs_parts] == [node_key(n) for n in other.rhs_parts])

    # R methods ------------------------------------------------------------
    def formula(self, lhs=None, rhs=None, collapse=False, update: bool = False) -> RFormula:
        li = _select(len(self.lhs_parts), lhs)
        ri = _select(len(self.rhs_parts), rhs)
        if isinstance(collapse, (list, tuple)):
            cl, cr = bool(collapse[0]), bool(collapse[-1])
        else:
            cl = cr = bool(collapse)
        node = _paste_formula([self.lhs_parts[i] for i in li], [self.rhs_parts[i] for i in ri],
                              "+" if cl else "|", "+" if cr else "|")
        rval = RFormula(node, self.env)
        if cl and cr and update:
            rval = update_formula(rval, ". ~ ." if rval.has_response else "~ .")
        return rval

    def _is_lhs_extended(self) -> bool:
        if len(self.lhs_parts) > 1:
            return True
        if len(self.lhs_parts) < 1:
            return False
        t = RFormula(_paste_formula([], self.lhs_parts, rsep="+")).terms(allow_dot_as_name=True)
        return len(t.term_labels) > 1

    def simplify_to_formula(self, lhs=None, rhs=None) -> RFormula:
        form = self.formula(lhs=lhs, rhs=rhs)
        F = Formula(form)
        ext_lhs = F._is_lhs_extended()
        if ext_lhs or len(F.rhs_parts) > 1:
            if ext_lhs:
                if len(F.rhs_parts) == 1 and isinstance(F.rhs_parts[0], Num) and F.rhs_parts[0].value == 0:
                    node = _paste_formula([], F.lhs_parts, rsep="+")
                else:
                    node = _paste_formula([], F.lhs_parts + F.rhs_parts, rsep="+")
            else:
                node = _paste_formula(F.lhs_parts, F.rhs_parts, rsep="+")
            return RFormula(node, self.env)
        return form

    def has_dot(self) -> bool:
        return any("." in _all_names(p) for p in self.lhs_parts + self.rhs_parts)

    def terms(self, lhs=None, rhs=None, data=None, dot: str = "separate", **kwargs) -> Terms:
        form = self.simplify_to_formula(lhs, rhs)
        if "." in _all_names(form.node) and data is not None:
            if dot not in ("separate", "sequential", "previous"):
                raise ValueError("dot must be 'separate', 'sequential' or 'previous'")
            ll = self.formula(rhs=0, collapse=True).lhs
            rr = list(self.rhs_parts)
            for i in range(len(rr)):
                if dot == "sequential" and i > 0:
                    ll = _c_formula(ll, self.rhs_parts[i - 1], "+") if ll is not None else self.rhs_parts[i - 1]
                fi = RFormula(_paste_formula([ll] if ll is not None else [], [rr[i]]), self.env)
                if dot == "previous" and i > 0:
                    prev = RFormula(_paste_formula([ll] if ll is not None else [], [rr[i - 1]]), self.env)
                    rr[i] = update_formula(prev, fi).rhs
                else:
                    expanded = fi.terms(data=data)
                    rr[i] = update_formula(_terms_to_formula(expanded, fi), ". ~ ." if fi.has_response else "~ .").rhs
            nx = Formula(self)
            nx.rhs_parts = rr
            form = nx.simplify_to_formula(lhs, rhs)
        return form.terms(data=data, **kwargs)

    def all_vars(self, lhs=None, rhs=None) -> List[str]:
        return self.formula(lhs=lhs, rhs=rhs).all_vars()

    def update(self, new) -> "Formula":
        new = new if isinstance(new, Formula) else Formula(new)
        o_l, o_r = self.lhs_parts, self.rhs_parts
        n_l, n_r = new.lhs_parts, new.rhs_parts

        def comp(x, y):
            xf = RFormula(Call(Sym("~"), ((None, x),)))
            yf = RFormula(Call(Sym("~"), ((None, y),)))
            return update_formula(xf, yf).rhs

        lhs = []
        for i in range(max(len(o_l), len(n_l))):
            if len(o_l) <= i:
                lhs.append(n_l[i])
            elif len(n_l) <= i:
                lhs.append(o_l[i])
            else:
                lhs.append(comp(o_l[i], n_l[i]))
        rhs = []
        for i in range(max(len(o_r), len(n_r))):
            if len(o_r) <= i:
                rhs.append(n_r[i])
            elif len(n_r) <= i:
                rhs.append(o_r[i])
            else:
                rhs.append(comp(o_r[i], n_r[i]))
        return Formula(RFormula(_paste_formula(lhs, rhs), self.env))


def _terms_to_formula(t: Terms, orig: RFormula) -> RFormula:
    """``formula(terms(...))``: the formula with ``.`` expanded."""
    labels = t.term_labels
    if t.offset:
        labels = labels + [t.variables[i - 1] for i in t.offset]
    rhs = " + ".join(labels) if labels else "1"
    if not t.intercept:
        rhs += " - 1"
    node = parse("~ " + rhs)
    if orig.has_response:
        node = Call(Sym("~"), ((None, orig.lhs), (None, node.argv[0])))
    return RFormula(node, orig.env)


def as_formula(x, env=None) -> RFormula:
    if isinstance(x, RFormula):
        return x
    if isinstance(x, Formula):
        return x.formula()
    return RFormula(x, env)


def coerce_formula(x, env=None):
    """Turn strings into :class:`RFormula`, or :class:`Formula` when they use ``|`` parts."""
    if isinstance(x, (RFormula, Formula, Terms)):
        return x
    if isinstance(x, str):
        rf = RFormula(x, env)
        lhs_parts = _extract_parts(rf.lhs)
        rhs_parts = _extract_parts(rf.rhs)
        if len(lhs_parts) > 1 or len(rhs_parts) > 1:
            return Formula(rf, env=env)
        return rf
    if isinstance(x, (list, tuple)):
        return Formula(*x, env=env)
    raise TypeError("expected a formula string, RFormula, Formula or Terms; got %r" % type(x).__name__)
