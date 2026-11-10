"""lavaan model syntax: a port of ``lavaan::lavParseModelString()`` and ``lavaan::lavaanify()``.

This is a line-by-line port of the "open" parser of lavaan 0.7.3
(``lav_parse_model_string_open`` and helpers) and of the parameter-table builder
(``lav_model_pt`` / ``lav_pt_flat`` / ``lav_pt_vnames`` / ``lav_pt_labels``), validated
against R (see ``tools/make_lavaan_fixtures.py``).

>>> import statsformula as sf
>>> sf.lavaan_parse("f =~ x1 + a*x2").to_dict("list")["label"]
['', 'a']
"""
from __future__ import annotations

import math
import re
import warnings
from itertools import combinations

__all__ = ["LavaanSyntaxError", "LavaanWarning", "lavaan_parse", "lavaanify", "lavaan_partable",
           "lavaan_names"]


class LavaanSyntaxError(ValueError):
    """Error in lavaan model syntax (mirrors lavaan's ``lav_msg_stop`` messages)."""


class LavaanWarning(UserWarning):
    """Warning emitted where lavaan emits a ``lav_msg_warn`` warning."""


# --------------------------------------------------------------------------------------
# configuration (lav_parse_options)
# --------------------------------------------------------------------------------------
_OPERATORS = ["=~", "<~", "~*~", "~~", "~", "|~", "==", "<", ">", ":=", ":", "|", "%", ":~"]
# (name, side, expect)
_MODIFIERS = [("efa", "l", "char"), ("fixed", "r", "num"), ("label", "r", "char"),
              ("start", "r", "num"), ("lower", "r", "num"), ("upper", "r", "num"),
              ("rv", "r", "char"), ("equal", "r", "char")]
_MOD_NAMES = [m[0] for m in _MODIFIERS]
_MOD_EXPECT = {m[0]: m[2] for m in _MODIFIERS}
_GROUPINGS = ["group", "level", "class", "block"]
_CON_OPS = ("==", "<", ">", ":=")

T_ID, T_NUM, T_STR, T_SYM, T_OP, T_NL = 1, 2, 3, 4, 5, 6

_R_RESERVED = {"if", "else", "repeat", "while", "function", "for", "next", "break", "TRUE",
               "FALSE", "NULL", "Inf", "NaN", "NA", "NA_integer_", "NA_real_",
               "NA_character_", "NA_complex_", "in"}


def _warn(msg: str) -> None:
    warnings.warn("lavaan WARNING: " + msg, LavaanWarning, stacklevel=3)


def _view(xs) -> str:
    """lav_msg_view(x, "none")."""
    return ", ".join('"%s"' % x for x in xs)


def _rseq(a: int, b: int) -> list[int]:
    """R's ``seq.int(a, b)`` (descending when ``a > b``)."""
    return list(range(a, b + 1)) if a <= b else list(range(a, b - 1, -1))


def _txtloc(src: str, position: int) -> tuple[str, str]:
    """``lav_parse_txtloc``; ``position`` is 1-based."""
    if not (len(src) >= position > 0):
        return "", ""
    nl = [m.start() + 1 for m in re.finditer("\n", src + "\n")]
    lijn = next(i for i, p in enumerate(nl, 1) if p >= position)
    if lijn == 1:
        pos = position
        line = src[0:nl[0]]
    else:
        pos = position - nl[lijn - 2]
        line = src[nl[lijn - 2]:nl[lijn - 1]]
    line = "" if len(line) == 1 else line[:-1]
    if re.match(r"^[ \t]*\n", src):
        lijn -= 1
    return (" at line %d, pos %d" % (lijn, pos), line + "\n" + " " * (pos - 1) + "^\n")


def _stop(msg: str, src: str | None = None, pos: int | None = None):
    if src is not None and pos is not None:
        a, b = _txtloc(src, pos)
        msg = msg + a + ("\n" + b if b else "")
    raise LavaanSyntaxError(msg)


def _warn_at(msg: str, src: str, pos: int) -> None:
    a, b = _txtloc(src, pos)
    _warn(msg + a + ("\n" + b if b else ""))


def _make_names_ok(x: str) -> bool:
    """``make.names(x) == x``."""
    if x in _R_RESERVED:
        return False
    if not re.fullmatch(r"(?:[^\W\d_]|\.(?!\d))[\w.]*|\.", x):
        return False
    return True


# --------------------------------------------------------------------------------------
# R number formatting (as.character for doubles: 15 significant digits)
# --------------------------------------------------------------------------------------
def _r_num2str(x) -> str:
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "TRUE" if x else "FALSE"
    x = float(x)
    if math.isnan(x):
        return "NA" if _is_na(x) else "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == 0:
        return "0"
    ref = float("%.14e" % x)
    for d in range(1, 16):
        if float("%.*e" % (d - 1, x)) == ref:
            break
    mant, ex = ("%.*e" % (d - 1, ref)).split("e")
    e10 = int(ex)
    sci = mant + "e" + ("-" if e10 < 0 else "+") + "%02d" % abs(e10)
    nd = max(0, d - 1 - e10)
    fixed = "%.*f" % (nd, ref)
    return fixed if len(fixed) <= len(sci) else sci


# NA sentinel for doubles: a NaN with a distinct identity is impractical, so R NA (double)
# is represented by ``None`` inside R vectors and by float('nan') in output tables.
def _is_na(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# --------------------------------------------------------------------------------------
# A tiny R evaluator for modifier expressions (c(), start(), label(), arithmetic, ...)
# --------------------------------------------------------------------------------------
class _RVec:
    """An atomic R vector: ``mode`` in {'logical', 'double', 'character'}; NA is None."""

    __slots__ = ("mode", "values", "tiepe")

    def __init__(self, mode, values, tiepe=None):
        self.mode, self.values, self.tiepe = mode, list(values), tiepe


class _RParseError(Exception):
    pass


def _r_tokenize(s: str):
    toks = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in " \t\n":
            i += 1
            continue
        if c in "\"'":
            j = s.find(c, i + 1)
            if j < 0:
                raise _RParseError("unterminated string")
            toks.append(("str", s[i + 1:j]))
            i = j + 1
            continue
        m = re.match(r"(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?L?", s[i:])
        if m:
            t = m.group(0)
            if t.endswith("L"):
                t = t[:-1]
            toks.append(("num", float(t)))
            i += len(m.group(0))
            continue
        m = re.match(r"(?:[^\W\d_]|\.(?!\d))[\w.]*|\.", s[i:])
        if m:
            toks.append(("name", m.group(0)))
            i += len(m.group(0))
            continue
        if c in "()+-*/^,":
            toks.append((c, c))
            i += 1
            continue
        raise _RParseError("unexpected '%s'" % c)
    return toks


class _RExprParser:
    """Recursive-descent parser for the subset of R used in modifiers."""

    def __init__(self, s):
        self.toks = _r_tokenize(s)
        self.i = 0

    def peek(self):
        return self.toks[self.i][0] if self.i < len(self.toks) else None

    def take(self, kind=None):
        if self.i >= len(self.toks):
            raise _RParseError("unexpected end")
        t = self.toks[self.i]
        if kind is not None and t[0] != kind:
            raise _RParseError("expected %s" % kind)
        self.i += 1
        return t

    def parse(self):
        if not self.toks:
            raise _RParseError("empty")
        e = self.expr()
        if self.i != len(self.toks):
            raise _RParseError("trailing tokens")
        return e

    # precedence: + - < * / < unary - < ^
    def expr(self):
        e = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()[0]
            e = ("bin", op, e, self.term())
        return e

    def term(self):
        e = self.unary()
        while self.peek() in ("*", "/"):
            op = self.take()[0]
            e = ("bin", op, e, self.unary())
        return e

    def unary(self):
        if self.peek() in ("-", "+"):
            op = self.take()[0]
            return ("un", op, self.unary())
        return self.power()

    def power(self):
        e = self.atom()
        if self.peek() == "^":
            self.take()
            e = ("bin", "^", e, self.unary())
        return e

    def atom(self):
        k, v = self.take()
        if k == "num":
            return ("num", v)
        if k == "str":
            return ("str", v)
        if k == "(":
            e = self.expr()
            self.take(")")
            return ("paren", e)
        if k == "name":
            if self.peek() == "(":
                self.take()
                args = []
                if self.peek() != ")":
                    while True:
                        args.append(self.expr())
                        if self.peek() == ",":
                            self.take()
                            continue
                        break
                self.take(")")
                return ("call", v, args)
            return ("name", v)
        raise _RParseError("unexpected token")


_R_CONST = {"TRUE": _RVec("logical", [True]), "FALSE": _RVec("logical", [False]),
            "NA": _RVec("logical", [None]), "NULL": _RVec("logical", [])}


def _r_c(args):
    vals = args
    if any(a.mode == "character" for a in vals):
        out = []
        for a in vals:
            if a.mode == "character":
                out += a.values
            elif a.mode == "double":
                out += [None if v is None else _r_num2str(v) for v in a.values]
            else:
                out += [None if v is None else ("TRUE" if v else "FALSE") for v in a.values]
        return _RVec("character", out)
    if any(a.mode == "double" for a in vals):
        out = []
        for a in vals:
            out += [None if v is None else float(v) for v in a.values]
        return _RVec("double", out)
    out = []
    for a in vals:
        out += a.values
    return _RVec("logical", out)


def _as_double(a):
    if a.mode == "character":
        raise _RParseError("non-numeric argument")
    return [None if v is None else float(v) for v in a.values]


def _arith(op, x, y=None):
    if y is None:
        xs = _as_double(x)
        return _RVec("double", [None if v is None else (-v if op == "-" else v) for v in xs])
    xs, ys = _as_double(x), _as_double(y)
    if not xs or not ys:
        return _RVec("double", [])
    n = max(len(xs), len(ys))
    out = []
    for k in range(n):
        a, b = xs[k % len(xs)], ys[k % len(ys)]
        if a is None or b is None:
            out.append(None)
            continue
        try:
            if op == "+":
                r = a + b
            elif op == "-":
                r = a - b
            elif op == "*":
                r = a * b
            elif op == "/":
                r = a / b if b != 0 else (math.nan if a == 0 else math.copysign(math.inf, a) *
                                          (1 if math.copysign(1, b) > 0 else -1))
            else:
                r = a ** b
        except (OverflowError, ZeroDivisionError):
            r = math.inf
        out.append(r)
    return _RVec("double", out)


def _r_eval(node, modfuns):
    k = node[0]
    if k == "num":
        return _RVec("double", [node[1]])
    if k == "str":
        return _RVec("character", [node[1]])
    if k == "paren":
        return _r_eval(node[1], modfuns)
    if k == "name":
        v = node[1]
        if v in _R_CONST:
            c = _R_CONST[v]
            return _RVec(c.mode, c.values)
        if v == "pi":
            return _RVec("double", [math.pi])
        return _RVec("character", [v])  # all.vars() are assigned their own name
    if k == "un":
        return _arith(node[1], _r_eval(node[2], modfuns))
    if k == "bin":
        return _arith(node[1], _r_eval(node[2], modfuns), _r_eval(node[3], modfuns))
    if k == "call":
        fn, args = node[1], [_r_eval(a, modfuns) for a in node[2]]
        if fn in modfuns:
            res = _r_c(args)
            res.tiepe = modfuns[fn]
            return res
        if fn == "c":
            return _r_c(args)
        if fn in ("sqrt", "exp", "log", "abs") and len(args) == 1:
            f = {"sqrt": math.sqrt, "exp": math.exp, "log": math.log, "abs": abs}[fn]
            return _RVec("double", [None if v is None else f(v) for v in _as_double(args[0])])
        if fn == "rep" and 1 <= len(args) <= 2:
            times = int(_as_double(args[1])[0]) if len(args) == 2 else 1
            return _RVec(args[0].mode, args[0].values * times)
        if fn == "as.numeric" and len(args) == 1:
            return _RVec("double", _as_double(args[0]))
        raise _RParseError("could not find function %s" % fn)
    raise _RParseError("bad node")


def _r_eval_string(txt, modfuns):
    return _r_eval(_RExprParser(txt).parse(), modfuns)


# --------------------------------------------------------------------------------------
# tokenizer (lav_parse_tokens_open)
# --------------------------------------------------------------------------------------
class _Formula:
    __slots__ = ("pos", "type", "text")

    def __init__(self, pos, typ, text):
        self.pos, self.type, self.text = list(pos), list(typ), list(text)

    def sub(self, idx):  # idx: 1-based indices
        return _Formula([self.pos[i - 1] for i in idx], [self.type[i - 1] for i in idx],
                        [self.text[i - 1] for i in idx])

    def __len__(self):
        return len(self.type)


def _tokens(src: str):
    """Tokenize; returns (pos[1-based], type, text, formula_number)."""
    w = list(src + "\n")
    elems = []  # (pos1, type, text)

    def ws():
        return "".join(w)

    for m in re.finditer(r'"[^"]*?["\n]', ws()):
        p, ln = m.start(), m.end() - m.start()
        for k in range(p + 1, p + ln - 1):
            w[k] = " "
        elems.append([p + 1, T_STR, src[p + 1:p + ln - 1]])
    for m in re.finditer(r"[#!][^\n]*?\n", ws()):
        c, ln = m.start() + 1, m.end() - m.start()
        for k in range(c - 1, c - 1 + ln - 1):
            w[k] = " "
        for e in elems:
            if c < e[0] < c + ln:
                e[1] = 0
    w = [" " if ch in "\t\r" else ch for ch in w]
    for k, ch in enumerate(w):
        if ch in ";\n":
            w[k] = "\n"
            elems.append([k + 1, T_NL, "\n"])
    for opr in [o for o in _OPERATORS if len(o) == 2]:
        rx = "([^=~%%])[%s]( +)[%s]([^=~%%])" % (re.escape(opr[0]), re.escape(opr[1]))
        s = ws()
        m = re.search(rx, s)
        if m:
            waar = m.start() + 2
            s = re.sub(rx, lambda mm: mm.group(1) + opr + mm.group(2) + mm.group(3), s)
            w = list(s)
            _warn_at("splitting of operator '%s' deprecated" % opr, src, waar)
    # operators: POSIX leftmost-longest alternation
    ops_sorted = sorted(_OPERATORS, key=len, reverse=True)
    s = ws()
    k = 0
    found = []
    while k < len(s):
        for o in ops_sorted:
            if s.startswith(o, k):
                found.append((k, o))
                k += len(o)
                break
        else:
            k += 1
    for p, o in found:
        elems.append([p + 1, T_OP, o])
        for q in range(p, p + len(o)):
            w[q] = " "
    s = ws()
    symbols = [m.start() for m in re.finditer(r"[,()/*?^']", s)]
    symbols += [m.start() for m in re.finditer(r"[-+][^.0-9]", s)]
    symbols += [m.end() - 2 for m in re.finditer(r"[._0-9a-df-zA-DF-Z)] *[-+][.0-9]", s)]
    symbols += [m.end() - 2 for m in re.finditer(r"[^.0-9][eE] *[-+][.0-9]", s)]
    for p in symbols:
        w[p] = " "
        elems.append([p + 1, T_SYM, src[p] if p < len(src) else "\n"])
    s = " " + ws()
    for m in re.finditer(r"([ \n][-+][.0-9]|[ \n]\.[0-9]|[ \n][0-9])[-+.0-9eE]*", s):
        p, ln = m.start(), m.end() - m.start() - 1
        for q in range(p, p + ln):
            w[q] = " "
        elems.append([p + 1, T_NUM, src[p:p + ln]])
    s = " " + ws()
    for m in re.finditer(r"[ \n](?:[_.]|[^\W\d_])[\w.]*", s):
        p, ln = m.start(), m.end() - m.start() - 1
        for q in range(p, p + ln):
            w[q] = " "
        elems.append([p + 1, T_ID, src[p:p + ln]])
    m = re.search(r'[^"\n ]', ws())
    if m:
        _stop("unexpected character", src, m.start() + 1)
    elems = [e for e in elems if e[1] > 0]
    elems.sort(key=lambda e: e[0])
    pos = [e[0] for e in elems]
    typ = [e[1] for e in elems]
    txt = [e[2] for e in elems]
    i = len(pos) - 1
    while i > 0:
        if typ[i] in (T_ID, T_NUM) and typ[i - 1] == T_ID:
            spaces = pos[i] - pos[i - 1] - 1  # lavaan uses length(), not nchar()
            txt[i - 1] = txt[i - 1] + " " * max(spaces, 0) + txt[i]
            typ[i] = 0
        i -= 1
    keep = [k for k in range(len(typ)) if typ[k] > 0]
    pos = [pos[k] for k in keep]
    typ = [typ[k] for k in keep]
    txt = [txt[k] for k in keep]
    # formula numbers (lav_parse_formula_numbers)
    fnum = [0] * len(typ)
    frm, hasefa, lastplus, incr = 1, False, False, False
    contops = ("+", "*", "/", "=~", "-", "<~", "~*~", "~~", "~", "|~", "|", "%")
    for i in range(len(typ)):
        fnum[i] = frm
        if typ[i] == T_ID and txt[i] == "efa":
            hasefa = True
        if txt[i] in contops:
            if incr:
                frm -= 1
                fnum[i] = frm
                incr = False
            lastplus = True
        else:
            if typ[i] in (T_STR, T_ID, T_NUM, T_SYM):
                lastplus = False
            if i > 0 and typ[i] != T_NL and typ[i - 1] == T_OP:
                hasefa = False
        if typ[i] == T_NL:
            if i > 0 and typ[i - 1] != T_NL:
                if not hasefa and not lastplus:
                    frm += 1
                    incr = True
                else:
                    hasefa = False
        else:
            incr = False
    return pos, typ, txt, fnum


def _formulas(pos, typ, txt, fnum, src):
    """lav_parse_formulas_open: group tokens in formulas and split on '+'."""
    relational = [o for o in _OPERATORS if o not in (":", ":=", ":~", "==", "<", ">")]
    nums = []
    for i in range(len(typ)):
        if typ[i] != T_NL and fnum[i] not in nums:
            nums.append(fnum[i])
    out = []
    for s in nums:
        idx = [i for i in range(len(typ)) if fnum[i] == s and typ[i] != T_NL]
        f = _Formula([pos[i] for i in idx], [typ[i] for i in idx], [txt[i] for i in idx])
        opis = [k + 1 for k, t in enumerate(f.type) if t == T_OP]
        nelem = len(f)
        if not opis:
            _stop("formula without valid operator", src, f.pos[0])
        opi = opis[0]
        if f.text[opi - 1] in relational and "+" in f.text:
            if opi == 1 or opi == nelem:
                _stop("formula with missing left- or right-hand side", src, f.pos[opi - 1])
            lhplus = []
            depth = 0
            for jj in range(1, opi):
                t = f.text[jj - 1]
                if t == "(":
                    depth += 1
                elif t == ")":
                    depth -= 1
                elif t == "+" and depth == 0:
                    lhplus.append(jj)
            lhplus.append(opi)
            plus = []
            depth = 0
            for jj in range(opi + 1, nelem + 1):
                t = f.text[jj - 1]
                if t == "(":
                    depth += 1
                elif t == ")":
                    depth -= 1
                elif t == "+" and depth == 0:
                    plus.append(jj)
            plus.append(nelem + 1)
            for j in range(len(lhplus)):
                j0 = 1 if j == 0 else lhplus[j - 1] + 1
                j1 = lhplus[j] - 1
                if j1 < j0:
                    continue
                for k in range(len(plus)):
                    k0 = opi + 1 if k == 0 else plus[k - 1] + 1
                    k1 = plus[k] - 1
                    if k1 < k0:
                        continue
                    out.append(f.sub(list(range(j0, j1 + 1)) + [opi] + list(range(k0, k1 + 1))))
        else:
            out.append(f)
    return out


# --------------------------------------------------------------------------------------
# formula handling (lav_parse_handle_formule and friends)
# --------------------------------------------------------------------------------------
def _check_name(f, ind, src):
    if ind < 1:
        _stop("formula with missing left-hand side", src, f.pos[0])
    item = f.text[ind - 1].replace(" ", "_")
    if not _make_names_ok(item):
        _stop("identifier is either a reserved word (in R) or contains an illegal character",
              src, f.pos[ind - 1])


def _check_relational(f, src, opi, nelem, op):
    for k in range(nelem):
        if f.type[k] == T_ID and " " in f.text[k]:
            _warn_at("having identifiers with spaces ('%s') is deprecated" % f.text[k], src,
                     f.pos[k])
            break
    _check_name(f, opi - 1, src)
    if opi >= nelem:
        _stop("Last element of rhs part expected to be an identifier or, for operator ~, "
              "|~ or =~, a numeric literal!", src, f.pos[nelem - 1])
    for j in range(opi + 1, nelem + 1):
        if f.type[j - 1] == T_ID and f.text[j - 1] != "NA":
            _check_name(f, j, src)
    if f.type[nelem - 1] != T_ID and (f.type[nelem - 1] != T_NUM or op not in ("~", "|~", "=~")):
        _stop("Last element of rhs part expected to be an identifier or, for operator ~, "
              "|~ or =~, a numeric literal!", src, f.pos[nelem - 1])


def _update_relational(f, src, opi, op):
    nelem = len(f)
    if f.text[nelem - 1] == "0" and op == "~" and opi == nelem - 1:
        f = _Formula(f.pos + [f.pos[nelem - 1]] * 2, f.type + [T_SYM, T_NUM], f.text + ["*", "1"])
        nelem = len(f)
    if f.type[nelem - 1] == T_NUM and op == "=~":
        f = _Formula(f.pos[:nelem - 1] + [f.pos[nelem - 1]] * 6,
                     f.type[:nelem - 1] + [T_ID, T_SYM, T_NUM, T_SYM, T_SYM, T_ID],
                     f.text[:nelem - 1] + ["fixed", "(", f.text[nelem - 1], ")", "*",
                                           f.text[opi - 2]])
        nelem = len(f)
    colons = [k for k in range(1, nelem) if f.text[k - 1] == ":" and f.type[k] == T_ID]
    if len(colons) > 2 or (len(colons) == 2 and (colons[0] > opi or colons[1] < opi)):
        _stop("Three-way or higher-order interaction terms (using multiple colons) are not "
              "supported in the lavaan syntax; please manually construct the product terms "
              "yourself in the data.frame, give them an appropriate name, and then you can use "
              "these interaction variables as any other (observed) variable in the model "
              "syntax.", src, f.pos[colons[1] - 1])
    if colons:
        c = colons[0]
        f.text[c - 2] = "".join(f.text[c - 2:c + 1])
        f = f.sub([k for k in range(1, nelem + 1) if k not in (c, c + 1)])
        nelem = len(f)
        if c < opi:
            opi -= 2
            if len(colons) == 2:
                colons[1] -= 2
    if len(colons) == 2:
        c = colons[1]
        f.text[c - 2] = "".join(f.text[c - 2:c + 1])
        f = f.sub([k for k in range(1, nelem + 1) if k not in (c, c + 1)])
        nelem = len(f)
    return f, opi, nelem


def _modifier(formul, opi, src, rme, rmeprev):
    """lav_parse_modifier_open: returns {modname: list-of-values} and the R vector."""
    n = len(formul)
    if rme < opi:
        lhs = True
        welke = _rseq(rmeprev + 1, rme) + _rseq(opi - 1, n)
    else:
        lhs = False
        welke = _rseq(1, opi) + _rseq(rmeprev + 1, rme) + [n]
    f1 = formul.sub(welke)
    nelem = len(f1)
    opi = next(k + 1 for k, t in enumerate(f1.type) if t == T_OP)
    check_more = True
    while check_more and nelem > 4:
        check_more = False
        for pi_ in range(3, nelem):
            if f1.text[pi_ - 2] == "(" and f1.text[pi_] == ")" and f1.type[pi_ - 3] != T_ID:
                f1.type[pi_ - 2] = 0
                f1.type[pi_] = 0
                check_more = True
        if check_more:
            f1 = f1.sub([k + 1 for k, t in enumerate(f1.type) if t > 0])
            nelem = len(f1)  # note: lavaan does not recompute opi here
    rng = _rseq(1, opi - 3) if lhs else _rseq(opi + 1, nelem - 2)
    strings = []
    for x in rng:
        if x < 1 or x > nelem:
            strings.append("NA")
        elif f1.type[x - 1] == T_STR:
            strings.append('"' + f1.text[x - 1] + '"')
        else:
            strings.append(f1.text[x - 1])
    txt = "".join(strings)
    if not lhs and f1.text[nelem - 2] == "?":
        f1.text[nelem - 2] = "*"
        txt = "start(" + txt + ")"
    errpos = f1.pos[opi] if opi < nelem else f1.pos[-1]
    if lhs:
        modfuns = {"efa": "efa"}
    else:
        modfuns = {m: m for m, side, _ in _MODIFIERS if side == "r"}
        modfuns["equal"] = "label"
    try:
        if txt in ("c", "t"):
            mod = _RVec("character", [txt], "label")
        else:
            mod = _r_eval_string(txt, modfuns)
            if mod.tiepe is None:
                mod.tiepe = "fixed" if mod.mode in ("double", "logical") else "label"
    except (_RParseError, ValueError, OverflowError, IndexError):
        _stop("invalid modifier specification", src, errpos)
    mdtype = mod.tiepe
    expect = _MOD_EXPECT[mdtype]
    if expect == "char" and mod.mode != "character":
        _stop("invalid %s modifier (should be character)" % mdtype, src, errpos)
    if expect == "num" and mod.mode != "double" and not all(v is None for v in mod.values):
        _stop("invalid %s modifier (should be numeric)" % mdtype, src, errpos)
    return mdtype, mod


def _mod_str(vec: _RVec) -> str:
    if vec.mode == "character":
        return ";".join("NA" if v is None else v for v in vec.values)
    if vec.mode == "logical":
        return ";".join("NA" if v is None else ("TRUE" if v else "FALSE") for v in vec.values)
    return ";".join(_r_num2str(v) for v in vec.values)


def _mod_value(vec: _RVec):
    """Python value of a modifier as stored in the 'modifiers' attribute."""
    if vec.mode == "character":
        vals = list(vec.values)
    elif vec.mode == "logical":
        vals = [None if v is None else bool(v) for v in vec.values]
    else:
        vals = [None if v is None else float(v) for v in vec.values]
    return vals


def _parse(model: str):
    """Port of lav_parse_model_string_open. Returns (flat dict of lists, modifiers, constraints)."""
    if isinstance(model, (list, tuple)):
        model = "\n".join(str(m) + " " for m in model)
        src = model.replace("˜", "~")
    else:
        src = (str(model) + " ").replace("˜", "~")
    if not re.search(r"[^\s;]", re.sub(r"[#!][^\n]*", "", src)):
        _stop("Model syntax is empty.")
    pos, typ, txt, fnum = _tokens(src)
    formulas = _formulas(pos, typ, txt, fnum, src)
    flat = {"lhs": [], "op": [], "rhs": [], "mod_idx": [], "block": []}
    for m in _MOD_NAMES:
        flat[m] = []
    modlist: list[dict] = []
    constraints: list[dict] = []
    block, block_op = 1, False
    for f in formulas:
        nelem = len(f)
        opis = [k + 1 for k, t in enumerate(f.type) if t == T_OP]
        opii = 0
        while f.text[opis[opii] - 1] == ":" and opii < len(opis) - 1:
            opii += 1
        opi = opis[opii]
        op = f.text[opi - 1]
        if op in ("==", "<", ">", ":=", ":~"):
            if opi <= 1 or opi >= nelem:
                _stop("Missing or empty %s of operator '%s'." % (
                    "left-hand side" if opi <= 1 else "right-hand side", op), src, f.pos[opi - 1])
            if op == ":~" and (opi != 2 or f.type[0] != T_ID):
                _stop("left-hand side of the ':~' operator must be a single parameter label.",
                      src, f.pos[0])
            constraints.append({"op": op, "lhs": "".join(f.text[:opi - 1]),
                                "rhs": "".join(f.text[opi:]), "user": 1})
            continue
        if op == ":":
            correct = ('The correct syntax is: "LHS : RHS", where LHS is block identifier '
                       '("group", "level", "class" or "block") and RHS is the block number or '
                       'label.')
            if opi == 1:
                _stop("Missing block identifier. " + correct, src, f.pos[0])
            if opi > 2 or f.text[0].lower() not in _GROUPINGS:
                _stop("Invalid block identifier. " + correct, src, f.pos[0])
            if nelem != 3 or f.type[2] not in (T_STR, T_ID, T_NUM):
                _stop("Syntax contains block identifier with missing or invalid number/label. "
                      + correct, src, f.pos[0])
            if block_op:
                block += 1
            elif len(flat["lhs"]) != 0:
                _warn_at("First block defined after other formula's", src, f.pos[0])
            flat["lhs"].append(f.text[0])
            flat["op"].append(op)
            flat["rhs"].append(f.text[2])
            flat["mod_idx"].append(0)
            flat["block"].append(block)
            for m in _MOD_NAMES:
                flat[m].append("")
            block_op = True
            continue
        _check_relational(f, src, opi, nelem, op)
        f, opi, nelem = _update_relational(f, src, opi, op)
        lhs = f.text[opi - 2]
        rhs = f.text[nelem - 1]
        already = [i for i in range(len(flat["lhs"]))
                   if flat["lhs"][i] == lhs and flat["op"][i] == op and flat["block"][i] == block
                   and (flat["rhs"][i] == rhs or (flat["rhs"][i] == "" and op in ("~", "|~")
                                                   and f.type[nelem - 1] == T_NUM))]
        if len(already) == 1:
            idx = already[0]
        else:
            flat["lhs"].append(lhs)
            flat["op"].append(op)
            flat["rhs"].append("" if (f.type[nelem - 1] == T_NUM and op in ("~", "|~")) else rhs)
            flat["mod_idx"].append(0)
            flat["block"].append(block)
            for m in _MOD_NAMES:
                flat[m].append("")
            idx = len(flat["lhs"]) - 1
        modelems = [k + 1 for k in range(nelem) if f.type[k] == T_SYM and f.text[k] in ("*", "?")]
        ok = []
        for me in modelems:
            r = _rseq(opi, me)
            seg = [f.text[k - 1] for k in r]
            if seg.count("(") == seg.count(")"):
                ok.append(me)
        modelems = ok
        if modelems:
            leftfirst = rightfirst = True
            for k, rme in enumerate(modelems):
                if rme < opi:
                    rmeprev = 0 if leftfirst else modelems[k - 1]
                    leftfirst = False
                else:
                    rmeprev = opi if rightfirst else modelems[k - 1]
                    rightfirst = False
                mdtype, vec = _modifier(f, opi, src, rme, rmeprev)
                flat[mdtype][idx] = _mod_str(vec)
                modnu = {mdtype: _mod_value(vec)}
                if flat["mod_idx"][idx] == 0:
                    modlist.append(modnu)
                    flat["mod_idx"][idx] = len(modlist)
                else:
                    cur = modlist[flat["mod_idx"][idx] - 1]
                    if mdtype in cur:
                        _warn_at("modifier %s specified multiple times, overwritten" % mdtype,
                                 src, f.pos[rmeprev])
                    cur.update(modnu)
        elif opi != nelem - 1:
            _stop("invalid formula right hand side", src, f.pos[opi])
        if f.text[opi - 1] == "~" and f.text[opi - 2] == f.text[nelem - 1]:
            if not re.match(r"^0\.?0*$", flat["fixed"][idx]):
                _stop("a variable cannot be regressed on itself", src, f.pos[opi - 1])
    # lav_parse_final_operations
    flat["mod.idx"] = flat.pop("mod_idx")
    for i in range(len(flat["op"])):
        if flat["op"][i] == "~" and flat["rhs"][i] == "":
            flat["op"][i] = "~1"
        if flat["op"][i] == "|~" and flat["rhs"][i] == "":
            flat["op"][i] = "|~1"
    for j in range(len(constraints) - 1, -1, -1):
        con = constraints[j]
        if con["op"] in ("<", ">") and con["lhs"] in flat["label"]:
            numbound = _const_bound(con["rhs"])
            if numbound is not None and not math.isnan(numbound):
                for nr in [i for i, l in enumerate(flat["label"]) if l == con["lhs"]]:
                    if flat["mod.idx"][nr] > 0:
                        nrm = flat["mod.idx"][nr]
                    else:  # unreachable in practice: labels always come with a modifier
                        modlist.append({"label": [con["lhs"]]})
                        nrm = len(modlist)
                        flat["mod.idx"][nr] = nrm
                    key = "upper" if con["op"] == "<" else "lower"
                    flat[key][nr] = _r_num2str(numbound)
                    modlist[nrm - 1][key] = [numbound]
                del constraints[j]
    _covariance_reorder(flat)
    for m in _MOD_NAMES:
        if all(v == "" for v in flat[m]):
            del flat[m]
    # move mod.idx to the end (as.data.frame column order)
    flat["mod.idx"] = flat.pop("mod.idx")
    if ":" in flat["op"]:
        op_idx = [i for i, o in enumerate(flat["op"]) if o == ":"]
        if len(op_idx) < 2:
            _warn("syntax contains only a single block identifier!")
        elif flat["lhs"][op_idx[0]] == "level" and flat["lhs"][op_idx[1]] == "group":
            _stop("groups can not be nested within levels!")
    return flat, modlist, constraints


def _const_bound(rhs: str):
    """Numeric value of a constraint rhs, or None (str2lang + eval as in lavaan)."""
    try:
        node = _RExprParser(rhs).parse()
    except _RParseError:
        return None
    if node[0] in ("name", "str"):
        return None
    if node[0] == "num":
        return node[1]
    try:
        v = _r_eval(node, {})
    except (_RParseError, ValueError, OverflowError, ZeroDivisionError):
        return None
    if v.mode != "double" or not v.values or v.values[0] is None:
        return None
    return v.values[0]


def _covariance_reorder(pt, ov_names=None, lv_names=None):
    cov_idx = [i for i in range(len(pt["op"])) if pt["op"][i] == "~~" and pt["lhs"][i] != pt["rhs"][i]]
    if not cov_idx:
        return
    if ov_names is None:
        ov_names = _vnames(pt, "ov")
    if lv_names is None:
        lv_names = _vnames(pt, "lv")
        rv = pt.get("rv")
        if rv is not None and any(len(x) > 0 for x in rv):
            lv_names = lv_names + _unique([x for x in rv if len(x) > 0])
    names = list(lv_names) + list(ov_names)
    pos = {}
    for k, n in enumerate(names):
        pos.setdefault(n, k)
    for i in cov_idx:
        a, b = pos.get(pt["lhs"][i]), pos.get(pt["rhs"][i])
        if a is not None and b is not None and a > b:
            pt["lhs"][i], pt["rhs"][i] = pt["rhs"][i], pt["lhs"][i]


def _unique(x):
    seen = set()
    out = []
    for v in x:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# --------------------------------------------------------------------------------------
# parameter-table helpers
# --------------------------------------------------------------------------------------
def _n(pt) -> int:
    return len(pt["lhs"])


def _col(pt, name, default):
    v = pt.get(name)
    return v if v is not None else [default] * _n(pt)


def _isnum(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _eq(a, b) -> bool:
    """R ``a == b`` for scalars, NA-propagating to False (as used inside which())."""
    if _is_na(a) or _is_na(b):
        return False
    return a == b


def _block_values(pt):
    if pt.get("block") is None:
        return [1]
    return _unique([b for b, o in zip(pt["block"], pt["op"])
                    if b is not None and b > 0 and o not in _CON_OPS])


def _level_values(pt):
    if ":" in pt["op"]:
        return _unique([r for l, o, r in zip(pt["lhs"], pt["op"], pt["rhs"])
                        if o == ":" and l.lower() == "level"])
    lv = pt.get("level")
    if lv is None:
        return [1]
    if all(_isnum(x) or x is None for x in lv):
        return _unique([x for x, o in zip(lv, pt["op"]) if x is not None and x > 0
                        and o not in _CON_OPS])
    return _unique([x for x, o in zip(lv, pt["op"]) if x and o not in _CON_OPS])


def _group_values(pt):
    if ":" in pt["op"]:
        return _unique([r for l, o, r in zip(pt["lhs"], pt["op"], pt["rhs"])
                        if o == ":" and l.lower() == "group"])
    g = pt.get("group")
    if g is None:
        return [1]
    if all(_isnum(x) or x is None for x in g):
        return _unique([x for x, o in zip(g, pt["op"]) if x is not None and x > 0
                        and o not in _CON_OPS])
    return _unique([x for x, o in zip(g, pt["op"]) if x and o not in _CON_OPS])


def _efa_values(pt):
    efa = pt.get("efa")
    if efa is None:
        return []
    return _unique([e for e, o in zip(efa, pt["op"]) if e and o not in _CON_OPS])


def _vnames_block(pt, b, nlevels, force_warn=False):
    """All variable-name sets of lav_pt_vnames for block ``b`` (a dict type -> list)."""
    n = _n(pt)
    lhs, op, rhs = pt["lhs"], pt["op"], pt["rhs"]
    block = pt["block"]
    bi = [block[i] == b for i in range(n)]
    rv = pt.get("rv")
    rv_names = _unique([x for x in rv if x]) if rv is not None else []
    out = {}
    lv_names = _unique([lhs[i] for i in range(n) if bi[i] and op[i] in ("=~", "<~")])
    lv_names2 = _unique(lv_names + rv_names)
    int_names = _unique([rhs[i] for i in range(n) if bi[i] and ":" in rhs[i]])
    lv_interaction = [x for x in int_names if any(t in lv_names for t in x.split(":"))]
    lv_names = lv_names + lv_interaction
    lv_names2 = lv_names2 + lv_interaction
    out["lv"] = lv_names
    out["lv.regular"] = _unique([lhs[i] for i in range(n) if bi[i] and op[i] == "=~"
                                 and lhs[i] != rhs[i] and lhs[i] not in rv_names])
    out["lv.interaction"] = lv_interaction
    form = _unique([lhs[i] for i in range(n) if bi[i] and lhs[i] == rhs[i] and op[i] == "=~"])
    keep = []
    for v in form:
        var_idx = [i for i in range(n) if bi[i] and op[i] == "~~" and lhs[i] == v and rhs[i] == v]
        if not var_idx:
            keep.append(v)
            continue
        i = var_idx[0]
        if pt.get("mod.idx") is not None and pt.get("fixed") is not None:
            if pt["fixed"][i] != "0":
                continue
        elif pt.get("free") is not None and pt.get("ustart") is not None:
            us = pt["ustart"][i]
            if pt["free"][i] > 0 or (not _is_na(us) and us != 0):
                continue
        keep.append(v)
    out["lv.formative"] = keep
    out["lv.composite"] = _unique([lhs[i] for i in range(n) if bi[i] and op[i] == "<~"])
    if pt.get("efa") is None:
        lv_efa = []
    else:
        sets = _efa_values(pt)
        lv_efa = _unique([lhs[i] for i in range(n) if op[i] == "=~" and bi[i]
                          and pt["efa"][i] in sets])
    out["lv.efa"] = lv_efa
    out["lv.rv"] = [] if rv is None else _unique([lhs[i] for i in range(n) if op[i] == "=~"
                                                  and bi[i] and lhs[i] in rv_names])
    out["lv.ind"] = _unique([rhs[i] for i in range(n) if bi[i] and op[i] == "=~"
                             and rhs[i] in lv_names and lhs[i] != rhs[i]])
    eqs_y = _unique([lhs[i] for i in range(n) if bi[i] and op[i] == "~"])
    eqs_x = _unique([rhs[i] for i in range(n) if bi[i] and op[i] in ("~", "<~")])
    out["eqs.y"], out["eqs.x"] = eqs_y, eqs_x
    v_ind = _unique([rhs[i] for i in range(n) if bi[i] and op[i] == "=~"])
    v_cind = _unique([rhs[i] for i in range(n) if bi[i] and op[i] == "<~"])
    ov_ind = [v for v in v_ind if v not in lv_names2]
    ov_cind = [v for v in v_cind if v not in lv_names2]
    ov_y = [v for v in eqs_y if v not in lv_names2 + ov_ind + ov_cind]
    if nlevels > 1 and b > 1:
        tmp_eqs_y = _unique([lhs[i] for i in range(n) if op[i] == "~"])
        ov_x = [v for v in eqs_x if v not in lv_names2 + ov_ind + ov_cind + tmp_eqs_y]
    else:
        ov_x = [v for v in eqs_x if v not in lv_names2 + ov_ind + ov_cind + ov_y]
    bad = [k for k, v in enumerate(ov_x) if ":" in v and any(t in eqs_y for t in v.split(":"))]
    if bad:
        ov_y = _unique(ov_y + [ov_x[k] for k in bad])
        ov_x = [v for k, v in enumerate(ov_x) if k not in bad]
    ov_cov = [lhs[i] for i in range(n) if bi[i] and op[i] == "~~" and lhs[i] not in lv_names2] + \
             [rhs[i] for i in range(n) if bi[i] and op[i] == "~~" and rhs[i] not in lv_names2]
    ov_int = [lhs[i] for i in range(n) if bi[i] and op[i] in ("~1", "|")
              and lhs[i] not in lv_names2]
    ov_tmp = ov_ind + ov_cind + ov_y + ov_x
    ov_extra = _unique(ov_cov + ov_int)
    ov_names = ov_tmp + [v for v in ov_extra if v not in ov_tmp]
    out["ov"], out["ov.ind"], out["ov.cind"] = ov_names, ov_ind, ov_cind
    out["ov.orphan"] = ov_extra
    oin = [v for v in ov_names if ":" in v]
    noint = [v for v in ov_names if v not in oin]
    out["ov.interaction"] = [v for v in oin if all(t in noint for t in v.split(":"))]
    out["ov.efa"] = _unique([rhs[i] for i in range(n) if op[i] == "=~" and bi[i]
                             and rhs[i] in ov_ind and lhs[i] in lv_efa])
    user = _col(pt, "user", 1)
    vars_ = [lhs[i] for i in range(n) if bi[i] and op[i] == "~1" and user[i] == 1] + \
            [lhs[i] for i in range(n) if bi[i] and op[i] == "~~" and user[i] == 1] + \
            [rhs[i] for i in range(n) if bi[i] and op[i] == "~~" and user[i] == 1]
    no_x = [v for v in ov_x if v in vars_]
    if no_x:
        if force_warn:
            _warn("model syntax contains variance/covariance/intercept formulas involving (an) "
                  "exogenous variable(s): [%s]; these variables will now be treated as random "
                  "introducing additional free parameters. If you wish to treat those variables "
                  "as fixed, remove these formulas from the model syntax. Otherwise, consider "
                  "adding the fixed_x = FALSE option." % _view(no_x))
        ov_x = [v for v in ov_x if v not in vars_]
    ov_tmp_x = list(ov_x)
    if pt.get("exo") is not None:
        exo = pt["exo"]
        oc = [lhs[i] for i in range(n) if bi[i] and op[i] == "~~" and exo[i] == 1] + \
             [rhs[i] for i in range(n) if bi[i] and op[i] == "~~" and exo[i] == 1]
        oi = [lhs[i] for i in range(n) if bi[i] and op[i] == "~1" and exo[i] == 1]
        ox = _unique(oc + oi)
        ov_tmp_x = ov_tmp_x + [v for v in ox if v not in ov_tmp_x]
    out["ov.x"] = ov_tmp_x
    ov_nox = [v for v in ov_names if v not in ov_tmp_x]
    out["ov.nox"] = ov_nox
    exo = _col(pt, "exo", None)
    if any(bi[i] and op[i] == "~" and exo[i] == 1 for i in range(n)):
        out["ov.model"] = ov_nox
    else:
        out["ov.model"] = ov_names
    thl = _unique([lhs[i] for i in range(n) if bi[i] and op[i] == "|"])
    ord_names = [v for v in ov_names if v in thl]
    out["ov.ord"] = ord_names
    ov_num = [v for v in ov_names if v not in ord_names]
    out["ov.num"] = ov_num
    lv_x = [v for v in lv_names if v not in v_ind + eqs_y]
    out["lv.x"] = lv_x
    tmp = [v for v in eqs_y if v not in v_ind + eqs_x + lv_names]
    out["ov.y"] = [v for v in ov_names if v in tmp]
    tmp = [v for v in eqs_y if v not in v_ind + eqs_x and v in lv_names]
    out["lv.y"] = [v for v in lv_names if v in tmp]
    out["lv.nox"] = [v for v in lv_names if v not in lv_x]
    marker = []
    if "free" in pt and "ustart" in pt:
        free, ustart = pt["free"], pt["ustart"]
        for lv in lv_names:
            midx = [i for i in range(n) if bi[i] and op[i] == "=~" and lhs[i] == lv
                    and rhs[i] in v_ind and _eq(ustart[i], 1) and free[i] == 0]
            valid = []
            for i in midx:
                other = [k for k in range(n) if bi[k] and op[k] == "=~" and lhs[k] != lv
                         and rhs[k] == rhs[i] and free[k] == 0]
                if not other or all(_eq(ustart[k], 0) for k in other):
                    valid.append(rhs[i])
            marker.append(valid[0] if valid else "")
    else:
        marker = [""] * len(lv_names)
    out["lv.marker"] = marker
    return out


def _vnames(pt, type_, force_warn=False, **sel):
    """lav_pt_vnames(partable, type) for a single type (dict-of-lists partable)."""
    if _n(pt) == 0:
        return []
    if pt.get("block") is None:
        pt = dict(pt)
        pt["block"] = [1] * _n(pt)
    block_select = _block_values(pt)
    if sel:
        rows = [True] * _n(pt)
        for k, v in sel.items():
            vals = v if isinstance(v, (list, tuple)) else [v]
            col = pt.get(k)
            if col is None:
                if k == "group" and vals == [1]:
                    col = [0 if bb == 0 else 1 for bb in pt["block"]]
                    rows = [r and c in vals for r, c in zip(rows, col)]
                    continue
                _stop("selection variable '%s' not found in the parameter table." % k)
            rows = [r and o not in _CON_OPS and c in vals for r, c, o in zip(rows, col, pt["op"])]
        block_select = _unique([bb for bb, r in zip(pt["block"], rows) if r])
    nlev = len(_level_values(pt))
    res = []
    for b in block_select:
        res.append(_vnames_block(pt, b, nlev, force_warn)[type_])
    if sel and len(block_select) == 1:
        return res[0]
    flat = [x for r in res for x in r]
    return flat if type_ == "lv.marker" else _unique(flat)


NA = float("nan")


def _pt_flat(flat, blocks=("group",), block_id=None, meanstructure=False, int_ov_free=False,
             int_lv_free=False, orthogonal=False, orthogonal_y=False, orthogonal_x=False,
             orthogonal_efa=False, std_lv=False, correlation=False,
             correlation_delta_free=False, composites=True, composites_cov_free=False,
             conditional_x=False, fixed_x=True, parameterization="delta",
             auto_fix_first=False, marker=None, auto_fix_single=False, auto_var=False,
             auto_cov_lv_x=False, auto_cov_x=False, auto_cov_y=False, auto_th=False,
             auto_delta=False, auto_efa=False, group_equal=None, group_w_free=False,
             ngroups=1, nthresholds=None, ov_names_x_block=None):
    """Port of lav_pt_flat (without var_table support)."""
    group_equal = list(group_equal or [])
    categorical = False
    if auto_cov_x:
        auto_cov_lv_x = True
    V = lambda t: _vnames(flat, t)  # noqa: E731
    lv_names = V("lv")
    if composites:
        lv_names_f = []
        lv_names_c = V("lv.composite")
        ov_ind_c = V("ov.cind")
        lv_names_noc = [v for v in lv_names if v not in lv_names_c]
    else:
        lv_names_c, ov_ind_c = [], []
        lv_names_f = V("lv.formative")
        lv_names_noc = lv_names
    ov_names = V("ov")
    ov_names_x = V("ov.x")
    lv_names_int = V("lv.interaction")
    if ov_names_x_block is None:
        ov_names_x_block = ov_names_x
    ov_names_nox = V("ov.nox")
    lv_names_x = V("lv.x")
    ov_names_y = V("ov.y")
    lv_names_y = V("lv.y")
    lvov_names_y = lv_names_y + ov_names_y
    ov_names_ord1 = V("ov.ord")
    if ov_names_ord1:
        bad = [v for v in ov_names_ord1 if v in ov_names_x]
        if bad:
            _warn("thresholds are defined for exogenous variables: " + _view(bad))
    ov_names_ord3 = []
    nth_names = None
    if nthresholds is not None:
        nth_names = list(nthresholds.keys()) if isinstance(nthresholds, dict) else None
        if nth_names is not None:
            ov_names_ord3 = nth_names
        else:
            ov_names_ord3 = ov_names_nox
    tmp = _unique(ov_names_ord1 + ov_names_ord3)
    ov_names_ord = [v for v in ov_names if v in tmp]
    if ov_names_ord:
        categorical = True
    if flat.get("efa") is not None and auto_efa:
        lv_names_efa = _unique([l for l, o, e in zip(flat["lhs"], flat["op"], flat["efa"])
                                if o == "=~" and e])
    else:
        lv_names_efa = []
    lhs, rhs = [], []
    nth = 0
    for o in ov_names_ord:
        if nthresholds is not None:
            if nth_names is None:
                nth = int(nthresholds)
            else:
                if o not in nthresholds:
                    _stop("ordered variable %s not found in the named vector nthresholds." % o)
                nth = int(nthresholds[o])
        if nth < 1:
            continue
        lhs += [o] * nth
        rhs += ["t%d" % k for k in range(1, nth + 1)]
    nth = len(lhs)
    lhs += ov_names_nox + lv_names
    rhs += ov_names_nox + lv_names
    if auto_cov_lv_x and len(lv_names_x) > 1:
        for a, b in combinations(lv_names_x, 2):
            lhs.append(a)
            rhs.append(b)
    if auto_cov_x and not conditional_x and lv_names_x and ov_names_x_block:
        for a in lv_names_x:
            for b in ov_names_x_block:
                lhs.append(a)
                rhs.append(b)
    if auto_cov_y and len(lvov_names_y) > 1:
        for a, b in combinations(lvov_names_y, 2):
            lhs.append(a)
            rhs.append(b)

    def lower_tri(x):
        for c in range(len(x)):
            for r in range(c, len(x)):
                lhs.append(x[c])
                rhs.append(x[r])

    if ov_names_x:
        if conditional_x:
            x1 = [v for v in ov_names_x if v not in ov_names_x_block]
            lower_tri(x1)
            lower_tri(list(ov_names_x_block))
        else:
            lower_tri(ov_names_x)
    if len(ov_ind_c) > 0:
        cidx = [i for i, o in enumerate(flat["op"]) if o == "<~"]
        w1 = [[0] * len(lv_names_c) for _ in ov_ind_c]
        for i in cidx:
            w1[ov_ind_c.index(flat["rhs"][i])][lv_names_c.index(flat["lhs"][i])] = 1
        if len(ov_ind_c) > 1:
            m = len(ov_ind_c)
            for c in range(m):
                for r in range(c + 1, m):
                    if sum(w1[r][k] * w1[c][k] for k in range(len(lv_names_c))) != 0:
                        lhs.append(ov_ind_c[c])
                        rhs.append(ov_ind_c[r])
    if auto_efa and len(lv_names_efa) > 1:
        for st in _efa_values(flat):
            this = _unique([l for l, o, e in zip(flat["lhs"], flat["op"], flat["efa"])
                            if o == "=~" and l not in lv_names_x and l not in lv_names_y
                            and e == st])
            if this:
                if len(this) < 2:
                    _stop("efa set %s has only a single factor" % st)
                for a, b in combinations(this, 2):
                    lhs.append(a)
                    rhs.append(b)
    op = ["~~"] * len(lhs)
    for k in range(nth):
        op[k] = "|"
    if ov_names_ord:
        lhs += ov_names_ord
        rhs += ov_names_ord
        op += ["~*~"] * len(ov_names_ord)
    if not categorical and (correlation is True or isinstance(correlation, (list, tuple))):
        cor_ov = [v for v in ov_names if v in correlation] if isinstance(
            correlation, (list, tuple)) else list(ov_names)
        if conditional_x:
            cor_ov = [v for v in cor_ov if v not in ov_names_x]
        lhs += cor_ov
        rhs += cor_ov
        op += ["~*~"] * len(cor_ov)
    if meanstructure:
        int_lhs = ov_names + lv_names
        lhs += int_lhs
        rhs += [""] * len(int_lhs)
        op += ["~1"] * len(int_lhs)
    if conditional_x:
        x1 = [v for v in ov_names_x if v not in ov_names_x_block]
        x2 = list(ov_names_x_block)
        if x1 and x2:
            for b in x2:
                for a in x1:
                    lhs.append(a)
                    op.append("~")
                    rhs.append(b)
    if group_w_free:
        lhs.append("group")
        rhs.append("w")
        op.append("%")
    default = list(zip(lhs, op, rhs))
    # user part
    user_rows = list(zip(flat["lhs"], flat["op"], flat["rhs"], flat["mod.idx"]))
    seen = set()
    dups = []
    kept = []
    for r in user_rows:
        key = r[:3]
        if key in seen:
            dups.append(key)
            continue
        seen.add(key)
        kept.append(r)
    if dups:
        _warn("duplicated elements in model syntax have been ignored: " +
              _view("    %s %s %s" % d for d in dups))
    user_rows = kept
    ukeys = set(r[:3] for r in user_rows)
    # duplicated(fromLast=TRUE) on rbind(default, user): drop defaults that occur later
    dd = []
    for k, d in enumerate(default):
        if d in ukeys or d in default[k + 1:]:
            continue
        dd.append(d)
    default = dd
    lhs = [r[0] for r in user_rows] + [d[0] for d in default]
    op = [r[1] for r in user_rows] + [d[1] for d in default]
    rhs = [r[2] for r in user_rows] + [d[2] for d in default]
    user = [1] * len(user_rows) + [0] * len(default)
    mod_idx = [r[3] for r in user_rows] + [0] * len(default)
    N = len(lhs)
    free = [1] * N
    ustart = [NA] * N
    label = [""] * N
    exo = [0] * N
    rng = range(N)
    if not auto_th:
        for i in rng:
            if op[i] == "|" and user[i] == 0:
                free[i] = 0
    if not auto_var:
        for i in rng:
            if op[i] == "~~" and lhs[i] == rhs[i] and lhs[i] not in ov_ind_c and user[i] == 0:
                ustart[i], free[i] = 0.0, 0
    elif lv_names_f:
        for i in rng:
            if op[i] == "~~" and lhs[i] == rhs[i] and lhs[i] in lv_names_f and user[i] == 0:
                ustart[i], free[i] = 0.0, 0
    if ov_ind_c:
        for i in rng:
            if composites_cov_free:
                if op[i] == "~~" and lhs[i] in ov_ind_c and rhs[i] in ov_ind_c:
                    ustart[i], free[i] = NA, 1
            elif op[i] == "~~" and lhs[i] in ov_ind_c:
                ustart[i], free[i] = NA, 0
    if lv_names_c:
        for i in rng:
            if op[i] == "~~" and lhs[i] in lv_names_c and lhs[i] == rhs[i]:
                ustart[i], free[i] = NA, 0
    for i in rng:
        if op[i] == "|~":
            ustart[i], free[i] = 0.0, 0
    if std_lv:
        for i in rng:
            if op[i] == "~~" and lhs[i] in lv_names_noc and lhs[i] == rhs[i]:
                ustart[i], free[i] = 1.0, 0
    if auto_efa and lv_names_efa:
        for i in rng:
            if op[i] == "~~" and lhs[i] in lv_names_efa and lhs[i] == rhs[i]:
                ustart[i], free[i] = 1.0, 0

    def fix_markers(mm_idx):
        if marker is None:
            seen_l = set()
            for i in mm_idx:
                if lhs[i] not in seen_l:
                    seen_l.add(lhs[i])
                    ustart[i], free[i] = 1.0, 0
        else:
            for lv in _unique([lhs[i] for i in mm_idx]):
                rows = [i for i in mm_idx if lhs[i] == lv]
                mrow = rows[0]
                if lv in marker and marker[lv] is not None:
                    t = [i for i in rows if rhs[i] == marker[lv]]
                    if t:
                        mrow = t[0]
                ustart[mrow], free[mrow] = 1.0, 0

    if auto_fix_first:
        fix_markers([i for i in rng if op[i] == "=~" and lhs[i] not in lv_names_efa])
    if composites and lv_names_c and (auto_fix_first or std_lv):
        fix_markers([i for i in rng if op[i] == "<~"])
    if auto_var and auto_fix_single:
        mm = [i for i in rng if op[i] == "=~"]
        cnt = {}
        for i in mm:
            cnt[lhs[i]] = cnt.get(lhs[i], 0) + 1
        singles = [k for k, v in cnt.items() if v == 1]
        if singles:
            rhs_mm = [rhs[i] for i in mm]
            single_ind = [rhs[i] for i in mm if lhs[i] in singles and lhs[i] != rhs[i]
                          and rhs_mm.count(rhs[i]) == 1]
            if single_ind:
                for i in rng:
                    if op[i] == "~~" and lhs[i] in single_ind and lhs[i] == rhs[i] and user[i] == 0:
                        ustart[i], free[i] = 0.0, 0

    def fix_cov(cond):
        for i in rng:
            if op[i] == "~~" and lhs[i] != rhs[i] and user[i] == 0 and cond(i):
                ustart[i], free[i] = 0.0, 0

    if orthogonal:
        fix_cov(lambda i: lhs[i] in lv_names and rhs[i] in lv_names)
    if orthogonal_y:
        fix_cov(lambda i: lhs[i] in lv_names_y and rhs[i] in lv_names_y)
    if orthogonal_x:
        fix_cov(lambda i: lhs[i] in lv_names_x and rhs[i] in lv_names_x + list(ov_names_x_block))
    if orthogonal_efa:
        fix_cov(lambda i: lhs[i] in lv_names_efa and rhs[i] in lv_names_efa)
    if meanstructure:
        for i in rng:
            if op[i] != "~1":
                continue
            if categorical and lhs[i] in ov_names_ord and user[i] == 0:
                ustart[i], free[i] = 0.0, 0
            if not int_ov_free and lhs[i] in ov_names and user[i] == 0:
                ustart[i], free[i] = 0.0, 0
            if not int_lv_free and lhs[i] in lv_names and user[i] == 0:
                ustart[i], free[i] = 0.0, 0
        rvc = flat.get("rv")
        if rvc is not None and any(rvc):
            lv_rv = V("lv.rv")
            for i in rng:
                if op[i] == "~1" and lhs[i] in lv_rv and user[i] == 0:
                    ustart[i], free[i] = NA, 1
        if lv_names_int:
            for i in rng:
                if op[i] == "~1" and lhs[i] in lv_names_int and user[i] == 0:
                    ustart[i], free[i] = NA, 1
        if lv_names_c:
            for i in rng:
                if op[i] == "~1" and lhs[i] in lv_names_c:
                    ustart[i], free[i] = NA, 0
    if not conditional_x and fixed_x and ov_names_x_block:
        for i in rng:
            if (op[i] == "~~" and rhs[i] in ov_names_x_block and lhs[i] in ov_names_x_block
                    and user[i] == 0) or (op[i] == "~1" and lhs[i] in ov_names_x_block
                                          and user[i] == 0):
                ustart[i], free[i], exo[i] = NA, 0, 1
    if conditional_x and ov_names_x:
        for i in rng:
            if (op[i] == "~~" and rhs[i] in ov_names_x and lhs[i] in ov_names_x and user[i] == 0) \
                    or (op[i] == "~1" and lhs[i] in ov_names_x and user[i] == 0):
                if fixed_x:
                    ustart[i], free[i] = NA, 0
                exo[i] = 1
            if op[i] in ("~", "<~") and lhs[i] in lv_names + ov_names_nox and rhs[i] in ov_names_x:
                exo[i] = 1
            if op[i] in ("~", "<~") and lhs[i] in ov_names_x and rhs[i] in ov_names_x:
                if fixed_x:
                    ustart[i], free[i] = NA, 0
                exo[i] = 1
    if ov_names_ord:
        for i in rng:
            if lhs[i] in ov_names_ord and op[i] == "~~" and user[i] == 0 and lhs[i] == rhs[i]:
                ustart[i], free[i] = 1.0, 0
        for i in rng:
            if op[i] == "~*~" and user[i] == 0:
                ustart[i], free[i] = 1.0, 0
    if correlation is True or isinstance(correlation, (list, tuple)):
        cor_ov = [v for v in ov_names if v in correlation] if isinstance(
            correlation, (list, tuple)) else list(ov_names)
        for i in rng:
            if lhs[i] in cor_ov and op[i] == "~~" and user[i] == 0 and lhs[i] == rhs[i]:
                ustart[i], free[i] = 1.0, 0
        for i in rng:
            if op[i] == "~*~" and user[i] == 0:
                if correlation_delta_free:
                    ustart[i], free[i] = NA, 1
                    if fixed_x and lhs[i] in ov_names_x:
                        free[i] = 0
                else:
                    ustart[i], free[i] = 1.0, 0
    if group_w_free:
        for i in rng:
            if lhs[i] == "group" and op[i] == "%":
                free[i], ustart[i] = 1, NA
    group = [1] * N
    if ngroups > 1:
        if not std_lv and "loadings" in group_equal:
            lv_marker = _vnames({"id": list(range(1, N + 1)), "lhs": lhs, "op": op, "rhs": rhs,
                                 "free": free, "ustart": ustart, "block": [1] * N}, "lv.marker")
        group = [g for g in range(1, ngroups + 1) for _ in range(N)]
        user = user * ngroups
        lhs = lhs * ngroups
        op = op * ngroups
        rhs = rhs * ngroups
        free = free * ngroups
        ustart = ustart * ngroups
        mod_idx = mod_idx * ngroups
        label = label * ngroups
        exo = exo * ngroups
        M = len(lhs)
        nth_ord = {x: sum(1 for i in range(M) if lhs[i] == x and op[i] == "|" and group[i] == 1)
                   for x in ov_names_ord}
        for g in range(2, ngroups + 1):
            for i in range(M):
                if group[i] != g:
                    continue
                if meanstructure and op[i] == "~1" and lhs[i] in lv_names_noc and user[i] == 0:
                    if not int_lv_free and "intercepts" in group_equal and \
                            "means" not in group_equal:
                        free[i], ustart[i] = 1, NA
                if meanstructure and ov_names_ord and op[i] == "~1" and lhs[i] in ov_names_ord \
                        and user[i] == 0:
                    if not int_lv_free and "thresholds" in group_equal:
                        free[i], ustart[i] = 1, NA
                if std_lv and "loadings" in group_equal and "lv.variances" not in group_equal:
                    if op[i] == "~~" and lhs[i] in lv_names and lhs[i] not in lv_names_efa and \
                            lhs[i] == rhs[i] and user[i] == 0:
                        free[i], ustart[i] = 1, NA
                elif not std_lv and "loadings" in group_equal:
                    if op[i] == "=~" and rhs[i] in lv_marker and free[i] == 0 and \
                            _eq(ustart[i], 1):
                        free[i], ustart[i] = 1, NA
                if lv_names_efa and auto_efa and "loadings" in group_equal and \
                        "lv.variances" not in group_equal:
                    if op[i] == "~~" and lhs[i] in lv_names_efa and lhs[i] == rhs[i] and \
                            user[i] == 0:
                        free[i], ustart[i] = 1, NA
            if ov_names_ord:
                notbin = [x for x in ov_names_ord if nth_ord[x] > 1]
                has_delta = any(op[i] == "~*~" and group[i] == g for i in range(M))
                if auto_delta and parameterization == "delta":
                    if has_delta and "thresholds" in group_equal:
                        for i in range(M):
                            if op[i] == "~*~" and group[i] == g and lhs[i] in notbin:
                                free[i], ustart[i] = 1, NA
                elif parameterization == "theta":
                    if has_delta and "thresholds" in group_equal:
                        for i in range(M):
                            if op[i] == "~~" and group[i] == g and lhs[i] in notbin and \
                                    lhs[i] == rhs[i]:
                                free[i], ustart[i] = 1, NA
            if group_w_free:
                for i in range(M):
                    if lhs[i] == "group" and op[i] == "%" and group[i] == g:
                        free[i], ustart[i] = 1, NA
    M = len(lhs)
    out = {"id": list(range(1, M + 1)), "lhs": lhs, "op": op, "rhs": rhs, "user": user}
    out["block"] = [block_id] * M if block_id is not None else list(group)
    for b in blocks:
        out[b] = list(group) if b == "group" else [0] * M
    out["mod.idx"] = mod_idx
    out["free"] = free
    out["ustart"] = ustart
    out["exo"] = exo
    out["label"] = label
    return out


# --------------------------------------------------------------------------------------
# lavaanify (lav_model_pt)
# --------------------------------------------------------------------------------------
_DEFAULTS = dict(meanstructure=False, int_ov_free=False, int_lv_free=False, marker_int_zero=False,
                 orthogonal=False, orthogonal_y=False, orthogonal_x=False, orthogonal_efa=False,
                 std_lv=False, correlation=False, correlation_delta_free=False, composites=True,
                 composites_cov_free=False, effect_coding="", conditional_x=False, fixed_x=False,
                 parameterization="delta", constraints=None, ceq_simple=False, auto=False,
                 model_type="sem", auto_fix_first=False, marker=None, auto_fix_single=False,
                 auto_var=False, auto_cov_lv_x=False, auto_cov_x=False, auto_cov_y=False,
                 auto_th=False, auto_delta=False, auto_efa=False, ngroups=1, nthresholds=None,
                 group_equal=None, group_partial=None, group_w_free=False, warn=True)

_PRESETS = {"cfa": dict(auto=True, model_type="cfa", fixed_x=True),
            "sem": dict(auto=True, model_type="sem", fixed_x=True),
            "growth": dict(auto=True, model_type="growth", fixed_x=True, meanstructure=True)}


def _norm_options(options):
    out = {}
    for k, v in options.items():
        key = k.replace(".", "_")
        if key not in _DEFAULTS:
            raise TypeError("lavaanify() got an unexpected option %r" % k)
        out[key] = v
    return out


def _str_cols(pt):
    return {k for k, v in pt.items() if v and all(isinstance(x, str) for x in v)}


def _append_rows(pt, n_new, char_cols):
    for k in pt:
        pt[k] = pt[k] + ([""] * n_new if k in char_cols else [None] * n_new)


def _labels(pt, blocks, group_equal, group_partial):
    """lav_pt_labels (type = "user")."""
    n = _n(pt)
    if n == 0:
        return []
    label = [a + b + c for a, b, c in zip(pt["lhs"], pt["op"], pt["rhs"])]
    group_equal = list(group_equal or [])
    group_partial = list(group_partial or [])
    ngroups = 1
    if "group" in blocks:
        g = pt.get("group")
        if g is not None and any(isinstance(x, str) for x in g):
            group_label = [x for x in _unique(g) if x]
            ngroups = len(group_label)
        else:
            ngroups = len(_group_values(pt))
            group_label = list(range(1, ngroups + 1))
        for gi in range(2, ngroups + 1):
            for i in range(n):
                if g[i] == group_label[gi - 1]:
                    label[i] = label[i] + ".g%d" % gi
    if ngroups > 1 and group_equal:
        def per_group(t):
            return [_unique(_vnames(pt, t, group=gi) or []) for gi in range(1, ngroups + 1)]

        def in_group(names_list):
            return [pt["group"][i] in range(1, ngroups + 1) and
                    pt["lhs"][i] in names_list[pt["group"][i] - 1] for i in range(n)]
        if {"intercepts", "residuals", "residual.covariances"} & set(group_equal):
            ov_nox = in_group(per_group("ov.nox"))
        if "thresholds" in group_equal:
            ov_ord = in_group(per_group("ov.ord"))
        if {"means", "lv.variances", "lv.covariances"} & set(group_equal):
            lvn = in_group(per_group("lv"))
        op, lhs, rhs = pt["op"], pt["lhs"], pt["rhs"]
        eq = [False] * n
        for i in range(n):
            if "loadings" in group_equal and op[i] == "=~":
                eq[i] = True
            if "composite.weights" in group_equal and op[i] == "<~":
                eq[i] = True
            if "intercepts" in group_equal and op[i] == "~1" and ov_nox[i]:
                eq[i] = True
            if "thresholds" in group_equal and op[i] == "|" and ov_ord[i]:
                eq[i] = True
            if "means" in group_equal and op[i] == "~1" and lvn[i]:
                eq[i] = True
            if "regressions" in group_equal and op[i] == "~":
                eq[i] = True
            if "residuals" in group_equal and op[i] == "~~" and ov_nox[i] and lhs[i] == rhs[i]:
                eq[i] = True
            if "residual.covariances" in group_equal and op[i] == "~~" and ov_nox[i] and \
                    lhs[i] != rhs[i]:
                eq[i] = True
            if "lv.variances" in group_equal and op[i] == "~~" and lvn[i] and lhs[i] == rhs[i]:
                eq[i] = True
            if "lv.covariances" in group_equal and op[i] == "~~" and lvn[i] and lhs[i] != rhs[i]:
                eq[i] = True
        if "composite.loadings" in group_equal:
            if "<~" in op:
                _warn("composite.loadings are in fact composite weights; better use "
                      "composite.weights")
                for i in range(n):
                    if op[i] == "<~":
                        eq[i] = True
            else:
                lvf = _vnames(pt, "lv.formative")
                for i in range(n):
                    if op[i] == "~" and lhs[i] in lvf:
                        eq[i] = True
        if group_partial:
            for i in range(n):
                if lhs[i] + op[i] + rhs[i] in group_partial:
                    eq[i] = False
        first = {}
        for i in range(n):
            if eq[i]:
                key = lhs[i] + op[i] + rhs[i]
                if key in first:
                    label[i] = label[first[key]]
                else:
                    first[key] = i
    for b in blocks:
        if b == "group":
            continue
        if b == "level" and pt.get(b) is not None:
            lev1 = _level_values(pt)[0]
            for i in range(n):
                if pt[b][i] != lev1:
                    label[i] = label[i] + ".l" + str(pt[b][i])
        elif pt.get(b) is not None:
            label = [lab + "." + b + str(v) for lab, v in zip(label, pt[b])]
    for i in range(n):
        if pt["label"][i] is None or pt["label"][i]:
            label[i] = pt["label"][i]
    return label


def _efa_con(pt, orthogonal_efa=False, group_equal=()):
    """lav_pt_efa_con."""
    n = _n(pt)
    sets = _efa_values(pt)
    lhs, op, blk, efa = pt["lhs"], pt["op"], pt["block"], pt["efa"]
    label_set1 = label_group1 = None
    for b in range(1, len(_block_values(pt)) + 1):
        for s, st in enumerate(sets, 1):
            lvs = _unique([lhs[i] for i in range(n) if op[i] == "=~" and blk[i] == b and efa[i] == st])
            if len(lvs) == 1:
                continue
            fix_to_zero = True
            set_idx = [i for i in range(n) if op[i] == "=~" and blk[i] == b and lhs[i] in lvs]
            this = [pt["label"][i] for i in set_idx]
            if s == 1:
                label_set1 = this
            elif all(len(x) > 0 for x in this) and all(x in label_set1 for x in this):
                fix_to_zero = False
            if b == 1:
                label_group1 = this
            elif "loadings" in group_equal:
                fix_to_zero = False
            elif all(len(x) > 0 for x in this) and all(x in label_group1 for x in this):
                fix_to_zero = False
            for f in range(2, len(lvs) + 1):
                nzero = f - 1
                ind_idx = [i for i in range(n) if op[i] == "=~" and blk[i] == b and lhs[i] == lvs[f - 1]]
                if len(ind_idx) < nzero:
                    _stop("efa factor %s has not enough indicators for echelon pattern" % lvs[f - 1])
                for i in ind_idx[:nzero]:
                    if fix_to_zero:
                        pt["free"][i], pt["ustart"][i], pt["user"][i] = 0, 0.0, 7
                    else:
                        pt["user"][i] = 77
            if not orthogonal_efa:
                for i in range(n):
                    if op[i] == "~~" and blk[i] == b and pt["user"][i] == 0 and lhs[i] in lvs and \
                            pt["rhs"][i] in lvs and lhs[i] != pt["rhs"][i]:
                        if fix_to_zero:
                            pt["free"][i], pt["ustart"][i], pt["user"][i] = 0, 0.0, 7
                        else:
                            pt["user"][i] = 77
    return pt


def lavaan_partable(model, preset=None, **options):
    """Build the lavaan parameter table as a dict of lists (see :func:`lavaanify`)."""
    opts = dict(_DEFAULTS)
    if preset is not None:
        if preset not in _PRESETS:
            raise ValueError("preset must be one of 'cfa', 'sem', 'growth'")
        opts.update(_PRESETS[preset])
    opts.update(_norm_options(options))
    if not opts.pop("warn"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LavaanWarning)
            return _model_pt(model, **opts)
    return _model_pt(model, **opts)


def _model_pt(model, meanstructure, int_ov_free, int_lv_free, marker_int_zero, orthogonal,
              orthogonal_y, orthogonal_x, orthogonal_efa, std_lv, correlation,
              correlation_delta_free, composites, composites_cov_free, effect_coding,
              conditional_x, fixed_x, parameterization, constraints, ceq_simple, auto,
              model_type, auto_fix_first, marker, auto_fix_single, auto_var, auto_cov_lv_x,
              auto_cov_x, auto_cov_y, auto_th, auto_delta, auto_efa, ngroups, nthresholds,
              group_equal, group_partial, group_w_free):
    if effect_coding not in ("", None, False) and effect_coding != []:
        raise NotImplementedError("effect_coding is not supported")
    if isinstance(group_equal, str):
        group_equal = [group_equal]
    if isinstance(group_partial, str):
        group_partial = [group_partial]
    group_equal = list(group_equal or [])
    ngroups = int(ngroups)
    flat, tmp_mod, tmp_con = _parse(model)
    tmp_mod = [dict(m) for m in tmp_mod]
    tmp_con = [dict(c) for c in tmp_con]
    if constraints:
        cons = [constraints] if isinstance(constraints, str) else list(constraints)
        if any(len(c) > 0 for c in cons):
            _, _, con2 = _parse(cons)
            tmp_con = tmp_con + con2
    if tmp_con:
        for c in tmp_con:
            c["user"] = 1
        if any(c["op"] in ("==", "<", ">", ":~") for c in tmp_con):
            ceq_simple = False
    if any(c["op"] == ":~" for c in tmp_con):
        raise NotImplementedError("the ':~' operator (data-defined parameters) is not supported")
    if fixed_x:
        _vnames(flat, "ov.x", force_warn=True)
    if auto:
        if model_type not in ("sem", "cfa", "growth"):
            raise ValueError("model_type must be 'sem', 'cfa' or 'growth'")
        int_ov_free = model_type != "growth"
        int_lv_free = model_type == "growth"
        auto_fix_first = not std_lv
        auto_fix_single = auto_var = auto_cov_lv_x = auto_cov_y = True
        auto_th = auto_delta = auto_efa = True
    if auto_cov_x:
        auto_cov_lv_x = True
        if conditional_x:
            auto_cov_x = False
    if "~1" in flat["op"]:
        meanstructure = True
    if nthresholds is None and "|" in flat["op"]:
        pairs = _unique([(l, r) for l, o, r in zip(flat["lhs"], flat["op"], flat["rhs"]) if o == "|"])
        nthresholds = {}
        for l, _ in pairs:
            nthresholds[l] = nthresholds.get(l, 0) + 1
    common = dict(meanstructure=meanstructure, int_ov_free=int_ov_free, int_lv_free=int_lv_free,
                  orthogonal=orthogonal, orthogonal_y=orthogonal_y, orthogonal_x=orthogonal_x,
                  orthogonal_efa=orthogonal_efa, std_lv=std_lv, correlation=correlation,
                  correlation_delta_free=correlation_delta_free, composites=composites,
                  composites_cov_free=composites_cov_free, conditional_x=conditional_x,
                  fixed_x=fixed_x, parameterization=parameterization,
                  auto_fix_first=auto_fix_first, marker=marker, auto_fix_single=auto_fix_single,
                  auto_var=auto_var, auto_cov_lv_x=auto_cov_lv_x, auto_cov_x=auto_cov_x,
                  auto_cov_y=auto_cov_y, auto_th=auto_th, auto_delta=auto_delta,
                  auto_efa=auto_efa, group_w_free=group_w_free, nthresholds=nthresholds)
    nf = len(flat["lhs"])
    block_op_idx = [i for i in range(nf) if flat["op"][i] == ":"]
    n_block_flat = len(block_op_idx)
    tmp_blocks_lhs = []
    if n_block_flat > 0:
        tmp_blocks_lhs = _unique([flat["lhs"][i].lower() for i in block_op_idx])
        if "group" in tmp_blocks_lhs and "level" in tmp_blocks_lhs:
            if tmp_blocks_lhs.index("group") > tmp_blocks_lhs.index("level"):
                _stop("levels must be nested within groups (not the other way around).")
        if len([b for b in tmp_blocks_lhs if "group" in b]) > 1:
            _warn("ambiguous block identifiers for group: " +
                  _view(b for b in tmp_blocks_lhs if "group" in b))
        if "group" in tmp_blocks_lhs:
            n_group_flat = len(_unique([flat["rhs"][i] for i in block_op_idx
                                        if flat["lhs"][i] == "group"]))
            if n_group_flat > 0 and n_group_flat != ngroups:
                _stop("syntax defines %d groups; data (or argument ngroups) suggests %d groups"
                      % (n_group_flat, ngroups))
        if conditional_x:
            raise NotImplementedError("conditional_x with block syntax is not supported")
        tmp_block_rhs = ["0"] * len(tmp_blocks_lhs)
        block_info = []
        idx1 = block_op_idx + [nf]
        for k in range(n_block_flat):
            blhs = flat["lhs"][idx1[k]]
            if blhs in tmp_blocks_lhs:
                tmp_block_rhs[tmp_blocks_lhs.index(blhs)] = flat["rhs"][idx1[k]]
            if idx1[k + 1] - idx1[k] == 1:
                continue
            block_info.append({"rhs": list(tmp_block_rhs),
                               "idx": list(range(idx1[k] + 1, idx1[k + 1]))})
        labels_ = [".".join(b["rhs"]) for b in block_info]
        merged = []
        for lab, b in zip(labels_, block_info):
            for m in merged:
                if m[0] == lab:
                    m[1]["idx"] = m[1]["idx"] + b["idx"]
                    break
            else:
                merged.append((lab, b))
        block_info = [m[1] for m in merged]
        nblocks = len(block_info)
        tmp_list = None
        has_rv = flat.get("rv") is not None and any(flat["rv"])
        for bnum in range(1, nblocks + 1):
            info = block_info[bnum - 1]
            block_lhs = tmp_blocks_lhs[-1]
            fb = {k: [v[i] for i in info["idx"]] for k, v in flat.items() if k != "block"}
            if block_lhs == "level" and bnum > 1 and has_rv:
                lv_rv = _unique([x for x in flat["rv"] if x])
                mx = max(flat["mod.idx"])
                for i, rvn in enumerate(lv_rv, 1):
                    for k in fb:
                        fb[k].append(fb[k][0] if fb[k] else "")
                    fb["lhs"][-1], fb["op"][-1], fb["rhs"][-1] = rvn, "=~", rvn
                    fb["mod.idx"][-1] = mx + i
                    for k in ("fixed", "start", "lower", "upper", "label", "prior", "efa"):
                        if k in fb:
                            fb[k][-1] = "0" if k == "fixed" else ""
                    fb["rv"][-1] = rvn
                    tmp_mod.append({"fixed": [0.0]})
            if fixed_x and block_lhs == "level":
                all_x = _vnames(flat, "ov.x")
                bx = _vnames(fb, "ov.x")
                bad = [v for v in bx if v not in all_x]
                if bad:
                    _warn("the variable(s) [%s] are exogenous at one level, but endogenous at "
                          "another level. These variables will be treated as endogenous, and "
                          "their variances/intercepts will be freely estimated. To remove this "
                          "warning, use fixed_x = FALSE." % _view(bad))
            lb = _pt_flat(fb, blocks=tmp_blocks_lhs, block_id=bnum, group_equal=None, ngroups=1,
                          ov_names_x_block=None, **common)
            for bl, br in zip(tmp_blocks_lhs, info["rhs"]):
                lb[bl] = [br] * len(lb["lhs"])
            if tmp_list is None:
                tmp_list = lb
            else:
                off = max(tmp_list["id"])
                lb["id"] = [x + off for x in lb["id"]]
                for k in tmp_list:
                    tmp_list[k] = tmp_list[k] + lb[k]
        for bl in tmp_blocks_lhs:
            vals = tmp_list[bl]
            if all(re.fullmatch(r"\s*[-+]?\d+\s*", v) for v in vals):
                tmp_list[bl] = [int(v) for v in vals]
    else:
        tmp_list = _pt_flat(flat, blocks=("group",), group_equal=group_equal, ngroups=ngroups,
                            **common)
    pt = tmp_list
    n = _n(pt)
    ov_ord = _vnames(pt, "ov.ord")
    categorical = len(ov_ord) > 0
    if categorical and parameterization == "delta":
        bad = [pt["lhs"][i] for i in range(n) if pt["op"][i] == "~~" and pt["lhs"][i] == pt["rhs"][i]
               and pt["lhs"][i] in ov_ord and pt["user"][i] == 1]
        if bad:
            _warn('variances of ordered variables are ignored when parameterization = "delta"; '
                  'please remove them from the model syntax or use parameterization = "theta"; '
                  'variables involved are: ' + " ".join(bad))
    cor_var_flag = False
    cor_ov = []
    if not categorical and (correlation is True or isinstance(correlation, (list, tuple))):
        ov_all = _vnames(pt, "ov")
        cor_ov = [v for v in ov_all if v in correlation] if isinstance(
            correlation, (list, tuple)) else ov_all
        bad = [pt["lhs"][i] for i in range(n) if pt["op"][i] == "~~" and
               pt["lhs"][i] == pt["rhs"][i] and pt["lhs"][i] in cor_ov and pt["user"][i] == 1]
        if bad:
            cor_var_flag = True
            _warn("(residual) variances of observed variables are ignored in a correlation "
                  "structure: the total variances are fixed to unity, and the residual variances "
                  "are derived from this constraint; please remove them from the model syntax; "
                  "variables involved are: " + " ".join(_unique(bad)))
    multilevel = False
    if pt.get("level") is not None and len(_level_values(pt)) > 1:
        multilevel = True
    if multilevel and "~1" in pt["op"]:
        ov_names = _vnames(pt, "ov")
        lev = _level_values(pt)
        other = [pt["lhs"][i] for i in range(n) if pt["op"][i] == "~1" and pt["level"][i] in lev[1:]
                 and pt["lhs"][i] in ov_names]
        for i in range(n):
            if pt["op"][i] == "~1" and pt["level"][i] == lev[0] and pt["lhs"][i] in other:
                pt["free"][i], pt["ustart"][i] = 0, 0.0
    if multilevel and "|" in pt["op"]:
        lev = _level_values(pt)
        for i in range(n):
            if pt["op"][i] == "|" and pt["level"][i] == lev[0]:
                pt["free"][i], pt["ustart"][i] = 0, 0.0
            if pt["op"][i] == "~*~" and pt["level"][i] in lev[1:]:
                pt["free"][i], pt["ustart"][i] = 0, 1.0
            if parameterization == "theta" and pt["op"][i] == "~~" and \
                    pt["lhs"][i] == pt["rhs"][i] and pt["lhs"][i] in ov_ord and \
                    pt["level"][i] in lev[1:] and pt["user"][i] == 0:
                pt["free"][i], pt["ustart"][i] = 1, NA
    warn_single = False
    for el, mod in enumerate(tmp_mod, 1):
        idx = [i for i in range(n) if pt["mod.idx"][i] == el]
        if not idx:
            continue
        m = {k: list(v) for k, v in mod.items()}
        if ngroups > 1 and len(idx) > 1:
            for k in list(m):
                if len(m[k]) == 1 and k != "equal":
                    m[k] = m[k] * ngroups
                    if k == "label" and not group_equal:
                        warn_single = True
        for k in ("fixed", "start", "lower", "upper", "label", "prior", "efa", "rv"):
            if k in m and len(m[k]) != len(idx):
                i0 = idx[0]
                _stop("wrong number of arguments in modifier (%s) of element %s %s %s" % (
                    " ".join(str(x) for x in m.get("label", [])), pt["lhs"][i0], pt["op"][i0],
                    pt["rhs"][i0]))
        if "fixed" in m:
            for i, v in zip(idx, m["fixed"]):
                if _is_na(v):
                    pt["free"][i], pt["ustart"][i] = 1, NA
                else:
                    pt["ustart"][i], pt["free"][i] = float(v), 0
        if "start" in m:
            for i, v in zip(idx, m["start"]):
                pt["ustart"][i] = NA if _is_na(v) else float(v)
        for k in ("prior", "efa", "rv"):
            if k in m:
                if k not in pt:
                    pt[k] = [""] * n
                for i, v in zip(idx, m[k]):
                    pt[k][i] = v  # None is R's NA_character_
                if k == "rv":
                    for i in idx:
                        pt["free"][i], pt["ustart"][i] = 0, NA
        for k, inf in (("lower", -math.inf), ("upper", math.inf)):
            if k in m:
                if k not in pt:
                    pt[k] = [inf] * n
                for i, v in zip(idx, m[k]):
                    pt[k][i] = NA if _is_na(v) else float(v)
        if "label" in m:
            for i, v in zip(idx, m["label"]):
                pt["label"][i] = v  # None is R's NA_character_ (nchar(NA) == 2)
    del pt["mod.idx"]
    if categorical and parameterization == "delta":
        for i in range(n):
            if pt["op"][i] == "~~" and pt["lhs"][i] == pt["rhs"][i] and pt["lhs"][i] in ov_ord \
                    and pt["user"][i] == 1:
                pt["free"][i], pt["ustart"][i] = 0, 1.0
    if cor_var_flag:
        for i in range(n):
            if pt["op"][i] == "~~" and pt["lhs"][i] == pt["rhs"][i] and pt["lhs"][i] in cor_ov \
                    and pt["user"][i] == 1:
                pt["free"][i], pt["ustart"][i] = 0, 1.0
    if warn_single:
        _warn("using a single label per parameter in a multiple group setting implies imposing "
              "equality constraints across all the groups; If this is not intended, either "
              "remove the label(s), or use a vector of labels (one for each group); See the "
              "Multiple groups section in the man page of model.syntax.")
    for k in ("lower", "upper"):
        if k in pt:
            for i in range(n):
                if pt["free"][i] == 0:
                    pt[k][i] = pt["ustart"][i]
    if "rv" in pt:
        for rvn in _unique([x for x in pt["rv"] if x]):
            for i in range(n):
                if pt["lhs"][i] == rvn and pt["op"][i] == "=~":
                    pt["rv"][i] = rvn
    blocks = tmp_blocks_lhs if n_block_flat > 1 else ["group"]
    label = _labels(pt, blocks, group_equal, group_partial)
    if auto_efa and "efa" in pt:
        pt = _efa_con(pt, orthogonal_efa=orthogonal_efa, group_equal=group_equal)
    if ceq_simple and "efa" in pt and any(pt["efa"]):
        ceq_simple = False
    pt["plabel"] = [".p%d." % i for i in pt["id"]]
    counts = {}
    for lab in label:
        counts[lab] = counts.get(lab, 0) + 1
    eq_labels = _unique([lab for lab in label if counts[lab] > 1])
    eq_id = [0] * n
    for eq_label in eq_labels:
        all_idx = [i for i in range(n) if label[i] == eq_label]
        ref, others = all_idx[0], all_idx[1:]
        for i in all_idx:
            eq_id[i] = ref + 1
        if "lower" in pt:
            fin = [pt["lower"][i] for i in all_idx if not _is_na(pt["lower"][i])
                   and math.isfinite(pt["lower"][i])]
            if fin:
                for i in all_idx:
                    pt["lower"][i] = min(fin)
        if "upper" in pt:
            fin = [pt["upper"][i] for i in all_idx if not _is_na(pt["upper"][i])
                   and math.isfinite(pt["upper"][i])]
            if fin:
                for i in all_idx:
                    pt["upper"][i] = max(fin)
        fixed_all = [i for i in all_idx if pt["free"][i] == 0]
        if fixed_all:
            u1 = pt["ustart"][fixed_all[0]]
            if not all(_is_na(pt["ustart"][i]) for i in fixed_all):
                if any(not _eq(u1, pt["ustart"][i]) for i in fixed_all):
                    _warn("equality constraints involve fixed parameters with different values; "
                          "only the first one will be used")
            for i in all_idx:
                pt["ustart"][i] = u1
                pt["free"][i] = 0
                eq_id[i] = 0
            if any(pt["user"][i] in (7, 77) for i in all_idx):
                for o in others:
                    tmp_con.append({"op": "==", "lhs": pt["plabel"][ref], "rhs": pt["plabel"][o],
                                    "user": 2})
        else:
            if not ceq_simple:
                for o in others:
                    tmp_con.append({"op": "==", "lhs": pt["plabel"][ref], "rhs": pt["plabel"][o],
                                    "user": 2})
            else:
                for o in others:
                    pt["free"][o] = 0
            for i in all_idx:
                if pt["label"][i] == "":
                    pt["label"][i] = pt["plabel"][ref]
    if tmp_con:
        char_cols = {"lhs", "op", "rhs", "label", "plabel", "efa", "prior", "rv"}
        for b in ("group", "level", "class"):
            if b in pt and any(isinstance(x, str) for x in pt[b]):
                char_cols.add(b)
        n0 = n
        nc = len(tmp_con)
        _append_rows(pt, nc, char_cols)
        for k, c in enumerate(tmp_con):
            i = n0 + k
            pt["id"][i] = i + 1
            pt["lhs"][i], pt["op"][i], pt["rhs"][i] = c["lhs"], c["op"], c["rhs"]
            pt["user"][i] = c["user"]
            pt["free"][i] = pt["exo"][i] = pt["block"][i] = 0
            for b in ("group", "level", "class"):
                if b in pt:
                    pt[b][i] = "" if b in char_cols else 0
        n = _n(pt)
    def_lhs = [pt["lhs"][i] for i in range(n) if pt["op"][i] == ":="]
    if def_lhs:
        dup = [x for k, x in enumerate(def_lhs) if x in def_lhs[:k]]
        if dup:
            _warn("at least one defined variable (using the := operator) has been duplicated, "
                  "and will be overwritten by the last one: " + " ".join(dup))
        for i in range(n):
            if pt["op"][i] == ":=":
                pt["label"][i] = pt["lhs"][i]
    if meanstructure and marker_int_zero:
        for b in range(1, len(_block_values(pt)) + 1):
            lvn = _vnames(pt, "lv.regular", block=b)
            lvm = _vnames(pt, "lv.marker", block=b)
            ovn = _vnames(pt, "ov.num", block=b)
            lvm = [x for x in lvm if x in ovn]
            if not lvm:
                continue
            for i in range(n):
                if pt["op"][i] == "~1" and pt["block"][i] == b and pt["user"][i] == 0:
                    if pt["lhs"][i] in lvm:
                        pt["free"][i], pt["ustart"][i] = 0, 0.0
                    elif pt["lhs"][i] in lvn:
                        pt["free"][i], pt["ustart"][i] = 1, NA
    k = 0
    for i in range(n):
        if pt["free"][i] is not None and pt["free"][i] > 0:
            k += 1
            pt["free"][i] = k
    if ceq_simple:
        for i in range(len(eq_id)):
            if eq_id[i] > 0:
                pt["free"][i] = pt["free"][eq_id[i] - 1]
    return pt


def lavaanify(model, preset=None, **options):
    """Equivalent of ``lavaan::lavaanify()``: the full lavaan parameter table.

    Parameters
    ----------
    model : str or list of str
        lavaan model syntax.
    preset : {"cfa", "sem", "growth"}, optional
        Use the automatic options that ``lavaan::cfa()``, ``sem()`` and ``growth()`` pass to
        ``lavaanify()`` (``auto = TRUE``, ``model.type``, ``fixed.x = TRUE``, and
        ``meanstructure = TRUE`` for growth models). Further options override the preset.
    **options
        lavaanify options; both R-style (``**{"int.ov.free": True}``) and Python-style
        (``int_ov_free=True``) names are accepted: ``meanstructure``, ``int_ov_free``,
        ``int_lv_free``, ``marker_int_zero``, ``orthogonal``, ``orthogonal_y``, ``orthogonal_x``,
        ``orthogonal_efa``, ``std_lv``, ``correlation``, ``conditional_x``, ``fixed_x``,
        ``parameterization``, ``constraints``, ``ceq_simple``, ``auto``, ``model_type``,
        ``auto_fix_first``, ``marker``, ``auto_fix_single``, ``auto_var``, ``auto_cov_lv_x``,
        ``auto_cov_x``, ``auto_cov_y``, ``auto_th``, ``auto_delta``, ``auto_efa``, ``ngroups``,
        ``nthresholds``, ``group_equal``, ``group_partial``, ``group_w_free``, ``composites``,
        ``warn``.

    Returns
    -------
    pandas.DataFrame
        Columns ``id, lhs, op, rhs, user, block, group, [level,] free, ustart, exo, label,
        [efa, lower, upper, ...,] plabel`` exactly as in R; R ``NA`` becomes ``NaN``.
    """
    import pandas as pd
    pt = lavaan_partable(model, preset=preset, **options)
    return _to_frame(pt, pd)


def _to_frame(pt, pd):
    data = {}
    for k, v in pt.items():
        if k in ("ustart", "lower", "upper"):
            data[k] = pd.Series([NA if _is_na(x) else float(x) for x in v], dtype=float)
        elif k in ("lhs", "op", "rhs", "label", "plabel", "efa", "prior", "rv") or \
                any(isinstance(x, str) for x in v):
            data[k] = pd.Series(v, dtype=object)
        elif any(x is None for x in v):
            data[k] = pd.Series(v, dtype="Int64")
        else:
            data[k] = pd.Series(v, dtype="int64")
    return pd.DataFrame(data, columns=list(pt.keys()))


def lavaan_parse(model):
    """Port of lavaan's ``lavParseModelString()`` returning a data frame (lavaan's data-frame form).

    Returns a :class:`pandas.DataFrame` with columns ``lhs, op, rhs, block``, the non-empty
    modifier columns among ``efa, fixed, label, start, lower, upper, rv`` and ``mod.idx``.
    As in lavaan 0.7.3, constraints and defined parameters (``:=``, ``==``, ``<``, ``>``,
    ``:~``) are *not* rows: they are stored in ``df.attrs["constraints"]`` (a list of dicts
    with keys ``op, lhs, rhs, user``); simple bounds on labels (``a > 0``) are turned into
    ``lower``/``upper`` modifiers. ``df.attrs["modifiers"]`` holds the parsed modifier
    values (one dict per ``mod.idx``; R ``NA`` is ``None``).
    """
    import pandas as pd
    flat, mods, cons = _parse(model)
    data = {}
    for k, v in flat.items():
        data[k] = pd.Series(v, dtype="int64" if k in ("block", "mod.idx") else object)
    df = pd.DataFrame(data, columns=list(flat.keys()))
    df.attrs["modifiers"] = mods
    df.attrs["constraints"] = cons
    return df


def lavaan_names(model, type="ov", **options):
    """Variable names of a lavaan model, like ``lavaan::lavNames(lavaanify(model), type)``.

    ``type`` is one of lavaan's name types (``"ov"``, ``"lv"``, ``"ov.x"``, ``"ov.nox"``,
    ``"ov.y"``, ``"lv.x"``, ``"lv.y"``, ``"ov.ord"``, ``"ov.ind"``, ``"lv.regular"``, ...).
    """
    pt = lavaan_partable(model, **options)
    return _vnames(pt, type.replace("_", "."))
