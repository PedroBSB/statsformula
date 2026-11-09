"""Port of R's ``terms.formula`` (C function ``termsform`` in ``src/library/stats/src/model.c``).

Terms are bitsets over the variable list.  The operators are implemented as
in R:

* ``a + b``     union (duplicates removed)
* ``a:b``       interaction: pairwise OR of terms
* ``a * b``     ``a + b + a:b``
* ``a / b``     nesting: ``a + (all of a):b``
* ``b %in% a``  ``b`` terms OR'ed with all of ``a``
* ``(a+b)^k``   all interactions up to order ``k``
* ``a - b``     removes terms (``- 1`` removes the intercept)
* ``0`` / ``1`` intercept control, ``offset()`` terms are removed
* ``.``         every column of ``data`` not used on the LHS
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ._parser import (Call, Const, Node, Num, Str, Sym, deparse, deparse_label, expand_dots,
                      is_call, node_key, parse)
from ._rtypes import RError


def _bitcount(x: int) -> int:
    return bin(x).count("1")


def _is_numeric_literal(n: Node) -> bool:
    return isinstance(n, Num) or (isinstance(n, Const) and n.value in ("TRUE", "FALSE"))


def _as_real(n: Node) -> float:
    if isinstance(n, Num):
        return n.value
    return 1.0 if n.value == "TRUE" else 0.0


def _is_zero_one(n: Node) -> bool:
    return _is_numeric_literal(n) and _as_real(n) in (0.0, 1.0)


def _is_one(n: Node) -> bool:
    return _is_numeric_literal(n) and _as_real(n) == 1.0


def _is_zero(n: Node) -> bool:
    return _is_numeric_literal(n) and _as_real(n) == 0.0


def _trim_repeats(terms: List[int]) -> List[int]:
    seen = set()
    out = []
    for t in terms:
        if t == 0 or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _lhs_symbols(node: Node) -> List[str]:
    out = []

    def walk(n):
        if isinstance(n, Sym):
            out.append(n.name)
        elif isinstance(n, Call):
            walk(n.fn)
            for _, a in n.args:
                walk(a)

    walk(node)
    return out


class _TermsForm:
    def __init__(self, framenames: Optional[Sequence[str]], allow_dot_as_name: bool):
        self.varlist: List[Node] = []
        self.keys: Dict[object, int] = {}
        self.intercept = True
        self.parity = True
        self.response = False
        self.have_dot = False
        self.framenames = list(framenames) if framenames is not None else None
        self.allow_dot = allow_dot_as_name

    # -- variables ---------------------------------------------------------
    def install_var(self, node: Node) -> int:
        if not isinstance(node, (Sym, Call)) and not _is_zero_one(node):
            raise RError("invalid term in model formula")
        k = node_key(node)
        if k in self.keys:
            return self.keys[k]
        self.varlist.append(node)
        self.keys[k] = len(self.varlist) - 1
        return self.keys[k]

    def _dot_names(self) -> List[str]:
        seen = set()
        for nm in self.framenames:
            if nm in seen:
                raise RError("duplicated name '%s' in data frame using '.'" % nm)
            seen.add(nm)
        return list(self.framenames)

    def extract_vars(self, f: Node) -> None:
        """Port of R 4.5 ``ExtractVars`` (installs every variable)."""
        if f is None or _is_zero_one(f):
            return
        if isinstance(f, Sym):
            if f.name == ".":
                self.have_dot = True
            if self.have_dot and self.framenames is not None:
                first = node_key(self.varlist[0]) if self.varlist else None
                for nm in self.framenames:
                    if node_key(Sym(nm)) != first:
                        self.install_var(Sym(nm))
            else:
                self.install_var(f)
            return
        if isinstance(f, Call) and isinstance(f.fn, Sym):
            op = f.fn.name
            a = f.argv
            n = len(a)
            if op == "~":
                if self.response:
                    raise RError("invalid model formula")
                if n == 1:
                    self.response = False
                    self.extract_vars(a[0])
                else:
                    self.response = True
                    self.install_var(a[0])
                    self.extract_vars(a[1])
                return
            if op == "+":
                for x in a[:2]:
                    self.extract_vars(x)
                return
            if op == "^" and n == 2:
                if not _is_numeric_literal(a[1]):
                    raise RError("invalid power in formula")
                self.extract_vars(a[0])
                return
            if op in (":", "*", "%in%", "/") and n == 2:
                self.extract_vars(a[0])
                self.extract_vars(a[1])
                return
            if op == "-":
                for x in a[:2]:
                    self.extract_vars(x)
                return
            if op == "(" and n == 1:
                self.extract_vars(a[0])
                return
            self.install_var(f)
            return
        if isinstance(f, Call):
            self.install_var(f)
            return
        raise RError("invalid model formula in ExtractVars")

    # -- encoding ----------------------------------------------------------
    def bit(self, node: Node) -> int:
        return 1 << self.install_var(node)

    def encode(self, f: Node) -> List[int]:
        if f is None:
            return []
        if _is_one(f):
            self.intercept = bool(self.parity)
            return []
        if _is_zero(f):
            self.intercept = not self.parity
            return []
        if isinstance(f, Sym):
            if f.name == "." and self.framenames is not None:
                seen = set()
                out = []
                for nm in self.framenames:
                    if nm in seen:
                        raise RError("duplicated name '%s' in data frame using '.'" % nm)
                    seen.add(nm)
                    out.append(self.bit(Sym(nm)))
                return out
            return [self.bit(f)]
        if isinstance(f, Call) and isinstance(f.fn, Sym):
            op = f.fn.name
            a = f.argv
            n = len(a)
            if op == "~":
                return self.encode(a[0] if n == 1 else a[1])
            if op == "+":
                if n == 1:
                    return self.encode(a[0])
                left = self.encode(a[0])
                right = self.encode(a[1])
                return _trim_repeats(left + right)
            if op == ":" and n == 2:
                left = self.encode(a[0])
                right = self.encode(a[1])
                return _trim_repeats([l | r for l in left for r in right])
            if op == "*" and n == 2:
                # CrossTerms: R returns `left` after listAppend(left, right), so an
                # empty left operand (0, 1, -x) discards everything (as in R).
                left = self.encode(a[0])
                right = self.encode(a[1])
                if not left:
                    return []
                prod = [l | r for l in left for r in right]
                return _trim_repeats(left + right + prod)
            if op == "%in%" and n == 2:
                left = self.encode(a[0])
                right = self.encode(a[1])
                allr = 0
                for t in right:
                    allr |= t
                return _trim_repeats([t | allr for t in left])
            if op == "/" and n == 2:
                # NestTerms: same `return left` behaviour as CrossTerms in R.
                left = self.encode(a[0])
                right = self.encode(a[1])
                if not left:
                    return []
                alll = 0
                for t in left:
                    alll |= t
                return _trim_repeats(left + [t | alll for t in right])
            if op == "^" and n == 2:
                if not _is_numeric_literal(a[1]):
                    raise RError("invalid power in formula")
                val = _as_real(a[1])
                if val != val:
                    raise RError("invalid power in formula")
                ip = int(val)
                if ip <= 1:
                    raise RError("invalid power in formula")
                left = self.encode(a[0])
                right = left
                term = left
                for _ in range(1, ip):
                    term = _trim_repeats([l | r for l in left for r in right])
                    right = term
                return term
            if op == "-":
                if n == 1:
                    return self._delete([], a[0], left_encoded=True)
                left = self.encode(a[0])
                return self._delete(left, a[1], left_encoded=True)
            if op == "(" and n == 1:
                return self.encode(a[0])
            return [self.bit(f)]
        if isinstance(f, Call):
            return [self.bit(f)]
        raise RError("invalid model formula in EncodeVars")

    def _delete(self, left: List[int], right_node: Node, left_encoded: bool) -> List[int]:
        self.parity = not self.parity
        right = self.encode(right_node)
        self.parity = not self.parity
        for t in right:
            if t == 0:
                self.intercept = False
            left = [l for l in left if l != t]
        return left


def _term_code(termlist: List[int], idx: int, whichbit: int) -> int:
    term = termlist[idx] & ~(1 << whichbit)
    if term == 0:
        return 1
    for t in termlist[:idx]:
        if (~t) & term == 0:
            return 1
    return 2


class Terms:
    """Equivalent of an R ``terms`` object.

    Attributes mirror R: ``variables`` (deparsed names), ``factors`` (2-D
    integer array, variables x terms), ``term_labels``, ``order``,
    ``intercept``, ``response``, ``offset`` (1-based variable indices, or
    ``None``), ``specials`` and, after building a model frame,
    ``data_classes`` and ``predvars``.
    """

    def __init__(self, formula, variable_nodes, factors, term_bits, term_labels, order,
                 intercept, response, offset, specials, have_dot=False):
        self.formula = formula
        self.variable_nodes: List[Node] = list(variable_nodes)
        self.variables: List[str] = [deparse(v) for v in self.variable_nodes]
        self.varnames: List[str] = [deparse_label(v) for v in self.variable_nodes]
        self.factors = factors
        self.term_bits = list(term_bits)
        self.term_labels: List[str] = list(term_labels)
        self.order: List[int] = list(order)
        self.intercept: int = int(intercept)
        self.response: int = int(response)
        self.offset = offset
        self.specials = specials
        self.have_dot = have_dot
        self.data_classes: Optional[Dict[str, str]] = None
        self.predvars = None

    # R-like accessors -----------------------------------------------------
    @property
    def term_variables(self) -> List[List[int]]:
        """For each term, the 0-based indices of the variables it contains."""
        nv = len(self.variables)
        return [[i for i in range(nv) if (b >> i) & 1] for b in self.term_bits]

    def factors_frame(self):
        import pandas as pd
        return pd.DataFrame(self.factors, index=self.variables, columns=self.term_labels)

    def copy(self):
        t = Terms(self.formula, self.variable_nodes, None if self.factors is None else self.factors.copy(),
                  self.term_bits, self.term_labels, self.order, self.intercept, self.response,
                  None if self.offset is None else list(self.offset),
                  None if self.specials is None else dict(self.specials), self.have_dot)
        t.varnames = list(self.varnames)
        t.data_classes = None if self.data_classes is None else dict(self.data_classes)
        t.predvars = self.predvars
        return t

    def __repr__(self):
        lines = ["Terms: %s" % (self.formula,),
                 "  variables:   %s" % (self.variables,),
                 "  term.labels: %s" % (self.term_labels,),
                 "  order:       %s" % (self.order,),
                 "  intercept:   %d" % self.intercept,
                 "  response:    %d" % self.response]
        if self.offset:
            lines.append("  offset:      %s" % (self.offset,))
        if self.specials:
            lines.append("  specials:    %s" % (self.specials,))
        return "\n".join(lines)


def terms_formula(formula_node: Node, data_names: Optional[Sequence[str]] = None,
                  keep_order: bool = False, specials: Optional[Sequence[str]] = None,
                  allow_dot_as_name: bool = False, formula_obj=None) -> Terms:
    """Port of R's ``termsform`` (R 4.5): compute the terms of a formula AST."""
    import warnings
    if not is_call(formula_node, "~") or len(formula_node.args) not in (1, 2):
        raise RError("argument is not a valid model")
    framenames = None
    had_framenames = False
    if data_names is not None:
        framenames = list(data_names)
        had_framenames = len(framenames) > 0
        if len(formula_node.args) == 2:
            lhs_syms = set(_lhs_symbols(formula_node.argv[0]))
            framenames = [n for n in framenames if n not in lhs_syms]
    tf = _TermsForm(framenames, allow_dot_as_name)
    tf.extract_vars(formula_node)
    nvar0 = len(tf.varlist)
    term_list = tf.encode(formula_node)
    nvar = len(tf.varlist)
    if nvar != nvar0:
        warnings.warn("'varlist' has changed (from nvar=%d) to new %d after EncodeVars() -- should no longer happen!"
                      % (nvar0, nvar))
    labnames = [deparse_label(v) for v in tf.varlist]
    resp = 1 if tf.response else 0

    # Step 2b: offsets
    offset_idx = [l for l in range(resp, nvar) if labnames[l].startswith("offset(")]
    offset = [l + 1 for l in offset_idx] if offset_idx else None
    if offset_idx:
        omask = 0
        for l in offset_idx:
            omask |= 1 << l
        term_list = [t for t in term_list if not (t & omask)]

    # Step 3: reorder by order
    if not keep_order:
        term_list = sorted(term_list, key=_bitcount)

    # Step 4: factor pattern
    nterm = len(term_list)
    if nterm > 0:
        factors = np.zeros((nvar, nterm), dtype=np.int64)
        for k, t in enumerate(term_list):
            for i in range(nvar):
                if (t >> i) & 1:
                    factors[i, k] = _term_code(term_list, k, i)
    else:
        factors = np.zeros((0, 0), dtype=np.int64)
    # Step 5: labels
    labels = [":".join(labnames[i] for i in range(nvar) if (t >> i) & 1) for t in term_list]
    order = [_bitcount(t) for t in term_list]

    spec = None
    if specials:
        spec = {}
        for s in specials:
            idx = [i + 1 for i, v in enumerate(labnames) if v.startswith(s + "(")]
            spec[s] = idx or None

    # Step 6: substitute the dot in the returned formula
    out_node = formula_node
    if tf.have_dot:
        if framenames:
            rhs = Sym(framenames[0])
            for nm in framenames[1:]:
                rhs = Call(Sym("+"), ((None, rhs), (None, Sym(nm))))
            if len(formula_node.args) == 2:
                out_node = Call(formula_node.fn, ((None, formula_node.argv[0]),
                                                  (None, expand_dots(formula_node.argv[1], rhs))))
            else:
                out_node = Call(formula_node.fn, ((None, expand_dots(formula_node.argv[0], rhs)),))
        elif not allow_dot_as_name and not had_framenames:
            raise RError("'.' in formula and no 'data' argument")
    if formula_obj is not None and out_node is not formula_node:
        from ._formula import RFormula
        formula_obj = RFormula(out_node, formula_obj.env)
    t = Terms(formula_obj, tf.varlist, factors, term_list, labels, order,
              tf.intercept, resp, offset, spec, tf.have_dot)
    t.varnames = labnames
    return t
