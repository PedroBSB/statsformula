"""Parser and deparser for R expressions (the subset used by model formulas).

The parser follows R's grammar and operator precedence (see ``?Syntax``):

====================  ===========================
operator              meaning
====================  ===========================
``::``                namespace
``$ @``               component / slot
``[ [[``              indexing
``^`` (right)         exponentiation (``**`` is translated to ``^``)
``- +`` (unary)       unary minus / plus
``:``                 sequence / interaction
``%any%`` ``|>``      special operators
``* /``               multiply, divide
``+ -`` (binary)      add, subtract
``< > <= >= == !=``   comparison
``!``                 negation
``& &&``              and
``| ||``              or
``~``                 formula
``<-``                assignment (parsed, not evaluated)
====================  ===========================

Expressions are represented exactly like R language objects: every operator
application is a :class:`Call` whose function is a :class:`Sym`, and
parentheses are kept as calls to ``(``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ._rtypes import RError, format_num


class FormulaSyntaxError(RError, SyntaxError):
    """Raised when a formula string cannot be parsed."""


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ()


@dataclass(frozen=True)
class Sym(Node):
    name: str

    def __repr__(self):
        return "Sym(%r)" % self.name


@dataclass(frozen=True)
class Num(Node):
    value: float
    is_int: bool = False

    def __repr__(self):
        return "Num(%r%s)" % (self.value, "L" if self.is_int else "")


@dataclass(frozen=True)
class Str(Node):
    value: str


@dataclass(frozen=True)
class Const(Node):
    """``TRUE``, ``FALSE``, ``NULL``, ``NA`` (and typed NA variants)."""
    value: str


@dataclass(frozen=True)
class Call(Node):
    fn: Node
    args: Tuple[Tuple[Optional[str], Node], ...] = field(default_factory=tuple)

    @property
    def fname(self) -> Optional[str]:
        return self.fn.name if isinstance(self.fn, Sym) else None

    @property
    def argv(self) -> List[Node]:
        return [a for _, a in self.args]

    def __repr__(self):
        return "Call(%r, %r)" % (self.fn, list(self.args))


class Missing(Node):
    """Empty argument, as in ``x[, 1]``."""
    __slots__ = ()

    def __eq__(self, other):
        return isinstance(other, Missing)

    def __hash__(self):
        return 0

    def __repr__(self):
        return "Missing()"


MISSING = Missing()


def call(fname: str, *args: Node) -> Call:
    return Call(Sym(fname), tuple((None, a) for a in args))


def is_call(node, *names) -> bool:
    return isinstance(node, Call) and isinstance(node.fn, Sym) and (not names or node.fn.name in names)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_RESERVED = {"if", "else", "repeat", "while", "function", "for", "next", "break",
             "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA", "NA_integer_", "NA_real_",
             "NA_character_", "NA_complex_", "in"}

_TOKEN_RE = re.compile(r"""
    (?P<ws>[ \t\r\n\f]+|\#[^\n]*)
  | (?P<num>0[xX][0-9a-fA-F]+L?|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?L?)
  | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<bq>`(?:[^`\\]|\\.)*`)
  | (?P<name>(?:[^\W\d_]|\.)[\w.]*)
  | (?P<op>%[^%\n]*%|\*\*|<=|>=|==|!=|&&|\|\||<<-|<-|->>|->|:::|::|\[\[|\|>|[~+\-*/^:(),=<>!&|$@\[\]?{};])
""", re.VERBOSE | re.UNICODE)


@dataclass
class Tok:
    kind: str
    value: str
    pos: int


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "'": "'",
            "`": "`", "a": "\a", "b": "\b", "f": "\f", "v": "\v"}


def _unescape(body: str) -> str:
    out = []
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            n = body[i + 1]
            if n == "x":
                m = re.match(r"[0-9a-fA-F]{1,2}", body[i + 2:])
                out.append(chr(int(m.group(0), 16)))
                i += 2 + len(m.group(0))
                continue
            if n in "uU":
                m = re.match(r"\{?([0-9a-fA-F]{1,8})\}?", body[i + 2:])
                out.append(chr(int(m.group(1), 16)))
                i += 2 + len(m.group(0))
                continue
            out.append(_ESCAPES.get(n, n))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def tokenize(text: str) -> List[Tok]:
    toks: List[Tok] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise FormulaSyntaxError("unexpected input at position %d in %r" % (pos, text))
        kind = m.lastgroup
        val = m.group(kind)
        if kind != "ws":
            if kind == "op" and val == "**":
                val = "^"
            toks.append(Tok(kind, val, pos))
        pos = m.end()
    toks.append(Tok("eof", "", n))
    return toks


# ---------------------------------------------------------------------------
# Pratt parser
# ---------------------------------------------------------------------------

_BINARY_BP = {
    "?": 1, "=": 2, "<-": 3, "<<-": 3, "->": 4, "->>": 4, "~": 5,
    "||": 6, "|": 6, "&&": 7, "&": 7,
    "==": 9, "!=": 9, "<": 9, ">": 9, "<=": 9, ">=": 9,
    "+": 10, "-": 10, "*": 11, "/": 11, "|>": 12, ":": 13, "^": 15,
    "$": 17, "@": 17, "::": 18, ":::": 18,
}
_RIGHT_ASSOC = {"^", "<-", "<<-", "="}
_POSTFIX_BP = 16


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.toks = tokenize(text)
        self.i = 0

    def peek(self, k: int = 0) -> Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self) -> Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def error(self, msg: str, tok: Optional[Tok] = None):
        tok = tok or self.peek()
        raise FormulaSyntaxError("%s at position %d in %r" % (msg, tok.pos, self.text))

    def expect(self, value: str):
        t = self.next()
        if t.kind != "op" or t.value != value:
            self.error("expected '%s' but found %r" % (value, t.value or "end of input"), t)
        return t

    def binary_bp(self, tok: Tok) -> int:
        if tok.kind != "op":
            return -1
        v = tok.value
        if v.startswith("%") and v.endswith("%") and len(v) >= 2:
            return 12
        if v in ("(", "[", "[["):
            return _POSTFIX_BP
        return _BINARY_BP.get(v, -1)

    def parse(self) -> Node:
        if self.peek().kind == "eof":
            self.error("empty expression")
        node = self.expr(0)
        if self.peek().kind != "eof":
            self.error("unexpected %r" % self.peek().value)
        return node

    def nud(self) -> Node:
        t = self.next()
        if t.kind == "num":
            return _make_num(t.value)
        if t.kind == "str":
            return Str(_unescape(t.value[1:-1]))
        if t.kind == "bq":
            return Sym(_unescape(t.value[1:-1]))
        if t.kind == "name":
            v = t.value
            if v in ("TRUE", "FALSE", "NULL", "NA", "NA_integer_", "NA_real_", "NA_character_"):
                return Const(v)
            if v == "Inf":
                return Num(float("inf"))
            if v == "NaN":
                return Num(float("nan"))
            if v in _RESERVED:
                self.error("unsupported R keyword %r in formula" % v, t)
            return Sym(v)
        if t.kind == "op":
            v = t.value
            if v == "(":
                inner = self.expr(0)
                self.expect(")")
                return Call(Sym("("), ((None, inner),))
            if v in ("-", "+"):
                operand = self.expr(14)
                return Call(Sym(v), ((None, operand),))
            if v == "!":
                operand = self.expr(8)
                return Call(Sym("!"), ((None, operand),))
            if v == "~":
                operand = self.expr(5)
                return Call(Sym("~"), ((None, operand),))
            if v == "?":
                operand = self.expr(1)
                return Call(Sym("?"), ((None, operand),))
        self.error("unexpected %r" % (t.value or "end of input"), t)

    def expr(self, min_bp: int) -> Node:
        left = self.nud()
        while True:
            tok = self.peek()
            bp = self.binary_bp(tok)
            if bp < 0 or bp <= min_bp:
                break
            op = tok.value
            self.next()
            if op == "(":
                args = self.arglist(")")
                left = Call(left, tuple(args))
                continue
            if op in ("[", "[["):
                args = self.arglist("]")
                if op == "[[":
                    self.expect("]")
                left = Call(Sym(op), ((None, left),) + tuple(args))
                continue
            if op in ("$", "@"):
                t = self.next()
                if t.kind == "name":
                    rhs = Sym(t.value)
                elif t.kind == "bq":
                    rhs = Sym(_unescape(t.value[1:-1]))
                elif t.kind == "str":
                    rhs = Str(_unescape(t.value[1:-1]))
                else:
                    self.error("unexpected %r after '%s'" % (t.value, op), t)
                left = Call(Sym(op), ((None, left), (None, rhs)))
                continue
            if op in ("->", "->>"):
                right = self.expr(bp)
                left = Call(Sym("<-" if op == "->" else "<<-"), ((None, right), (None, left)))
                continue
            right = self.expr(bp - 1 if op in _RIGHT_ASSOC else bp)
            left = Call(Sym(op), ((None, left), (None, right)))
        return left

    def arglist(self, closing: str):
        args = []
        if self.peek().kind == "op" and self.peek().value == closing:
            self.next()
            return args
        while True:
            tok = self.peek()
            if tok.kind == "op" and tok.value in (",", closing):
                args.append((None, MISSING))
            else:
                nxt = self.peek(1)
                if tok.kind in ("name", "str", "bq") and nxt.kind == "op" and nxt.value == "=":
                    self.next()
                    self.next()
                    name = tok.value if tok.kind == "name" else _unescape(tok.value[1:-1])
                    t2 = self.peek()
                    if t2.kind == "op" and t2.value in (",", closing):
                        args.append((name, MISSING))
                    else:
                        args.append((name, self.expr(2)))
                else:
                    args.append((None, self.expr(2)))
            t = self.next()
            if t.kind == "op" and t.value == ",":
                continue
            if t.kind == "op" and t.value == closing:
                return args
            self.error("expected ',' or '%s'" % closing, t)


def _make_num(text: str) -> Num:
    is_int = text.endswith("L")
    body = text[:-1] if is_int else text
    if body.lower().startswith("0x"):
        value = float(int(body, 16))
    else:
        value = float(body)
    if is_int and value != int(value):
        is_int = False
    return Num(value, is_int)


def parse(text: str) -> Node:
    """Parse an R expression string into an AST."""
    if not isinstance(text, str):
        raise TypeError("expected a string")
    return _Parser(text).parse()


# ---------------------------------------------------------------------------
# Deparser
# ---------------------------------------------------------------------------

_SYNTACTIC_RE = re.compile(r"^(?:(?:[^\W\d_]|\.(?![0-9]))[\w.]*|\.)$", re.UNICODE)
_NOSPACE_BINARY = {"^", ":", "/", "%%", "%/%", "$", "@", "::", ":::"}
_UNARY_OPS = {"-", "+", "!", "~", "?"}
_BINARY_OPS = set(_BINARY_BP) - {"$", "@", "::", ":::"}


def is_syntactic(name: str) -> bool:
    if name == "...":
        return True
    return bool(_SYNTACTIC_RE.match(name)) and name not in _RESERVED


def deparse_name(name: str) -> str:
    if is_syntactic(name):
        return name
    return "`" + name.replace("\\", "\\\\").replace("`", "\\`") + "`"


def _quote_str(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 32:
            out.append("\\%03o" % ord(ch))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _op_bp(node: Node) -> Optional[int]:
    """Precedence of a node if it is an operator call, else None."""
    if not isinstance(node, Call) or not isinstance(node.fn, Sym):
        return None
    op = node.fn.name
    n = len(node.args)
    if n == 1 and op in ("-", "+"):
        return 14
    if n == 1 and op == "!":
        return 8
    if n == 1 and op == "~":
        return 5
    if n == 2:
        if op.startswith("%") and op.endswith("%") and len(op) >= 2:
            return 12
        if op in _BINARY_BP:
            return _BINARY_BP[op]
    return None


def deparse(node: Node) -> str:
    """Deparse like R's ``deparse()`` (single line): bare symbols are not
    backquoted, integer literals keep their ``L`` suffix."""
    if isinstance(node, Sym):
        return node.name
    return _dp(node, True)


def deparse_label(node: Node) -> str:
    """Deparse like R's internal ``deparse1line(x, FALSE)`` used for term labels and
    model matrix column names: symbols are backquoted when needed and integer
    literals lose their ``L`` suffix."""
    return _dp(node, False)


def _dp(node: Node, ki: bool) -> str:
    if isinstance(node, Sym):
        return deparse_name(node.name)
    if isinstance(node, Num):
        if node.is_int:
            return format_num(node.value, is_int=True) + ("L" if ki else "")
        return format_num(node.value)
    if isinstance(node, Str):
        return _quote_str(node.value)
    if isinstance(node, Const):
        return node.value
    if isinstance(node, Missing):
        return ""
    if isinstance(node, Call):
        return _deparse_call(node, ki)
    raise TypeError("cannot deparse %r" % (node,))


def _wrap(child: Node, parent_bp: int, side: str, op: str, ki: bool) -> str:
    s = _dp(child, ki)
    cbp = _op_bp(child)
    if cbp is None:
        return s
    if cbp < parent_bp:
        if len(child.args) == 1 and side == "right":
            return s
        return "(" + s + ")"
    if cbp == parent_bp:
        right_assoc = op in _RIGHT_ASSOC
        if (side == "right" and not right_assoc) or (side == "left" and right_assoc):
            if len(child.args) == 2:
                return "(" + s + ")"
    return s


def _deparse_args(args, ki: bool) -> str:
    parts = []
    for name, a in args:
        if name is not None:
            parts.append("%s = %s" % (deparse_name(name), _dp(a, ki)) if not isinstance(a, Missing) else "%s = " % deparse_name(name))
        else:
            parts.append(_dp(a, ki))
    return ", ".join(parts)


def _deparse_call(node: Call, ki: bool) -> str:
    fn = node.fn
    args = node.args
    if isinstance(fn, Sym):
        op = fn.name
        anon = all(n is None for n, _ in args)
        if op == "(" and len(args) == 1 and anon:
            return "(" + _dp(args[0][1], ki) + ")"
        if op in ("[", "[[") and len(args) >= 1 and args[0][0] is None:
            close = "]" if op == "[" else "]]"
            return _wrap(args[0][1], _POSTFIX_BP, "left", op, ki) + op + _deparse_args(args[1:], ki) + close
        if op in ("$", "@") and len(args) == 2 and anon:
            rhs = args[1][1]
            r = _dp(rhs, ki) if not isinstance(rhs, Str) else _quote_str(rhs.value)
            return _wrap(args[0][1], 17, "left", op, ki) + op + r
        if op in ("::", ":::") and len(args) == 2 and anon:
            return _dp(args[0][1], ki) + op + _dp(args[1][1], ki)
        is_special = op.startswith("%") and op.endswith("%") and len(op) >= 2
        if len(args) == 1 and anon and op in _UNARY_OPS:
            bp = _op_bp(node)
            inner = args[0][1]
            s = _dp(inner, ki)
            cbp = _op_bp(inner)
            if cbp is not None and cbp < bp and len(inner.args) == 2:
                s = "(" + s + ")"
            return op + s
        if len(args) == 2 and anon and (op in _BINARY_OPS or is_special):
            bp = _op_bp(node)
            left = _wrap(args[0][1], bp, "left", op, ki)
            right = _wrap(args[1][1], bp, "right", op, ki)
            if op in _NOSPACE_BINARY:
                return left + op + right
            return left + " " + op + " " + right
        return deparse_name(op) + "(" + _deparse_args(args, ki) + ")"
    head = _dp(fn, ki)
    if isinstance(fn, Call) and fn.fname not in ("::", ":::", "$", "@") and _op_bp(fn) is not None:
        head = "(" + head + ")"
    return head + "(" + _deparse_args(args, ki) + ")"


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------

def node_key(node: Node):
    """Key implementing R's ``MatchVar`` equality of variables."""
    if isinstance(node, Sym):
        return ("s", node.name)
    if isinstance(node, Num):
        return ("n", node.value)
    if isinstance(node, Const):
        if node.value == "TRUE":
            return ("n", 1.0)
        if node.value == "FALSE":
            return ("n", 0.0)
        return ("c", node.value)
    if isinstance(node, Str):
        return ("str", node.value)
    if isinstance(node, Missing):
        return ("m",)
    if isinstance(node, Call):
        return ("call", node_key(node.fn), tuple((n, node_key(a)) for n, a in node.args))
    raise TypeError(node)


def all_names(node: Node, functions: bool = True, unique: bool = False) -> List[str]:
    """R's ``all.names``."""
    out: List[str] = []

    def walk(n):
        if isinstance(n, Sym):
            out.append(n.name)
        elif isinstance(n, Call):
            if isinstance(n.fn, Sym):
                if functions:
                    out.append(n.fn.name)
            else:
                walk(n.fn)
            for _, a in n.args:
                walk(a)

    walk(node)
    if unique:
        seen = set()
        res = []
        for x in out:
            if x not in seen:
                seen.add(x)
                res.append(x)
        return res
    return out


def all_vars(node: Node) -> List[str]:
    """R's ``all.vars``."""
    return all_names(node, functions=False, unique=True)


def expand_dots(obj: Node, value: Node) -> Node:
    """Port of R's ``ExpandDots`` (model.c): substitute ``.`` by ``value``,
    adding parentheses exactly where R does."""
    if isinstance(obj, Sym):
        return value if obj.name == "." else obj
    if not isinstance(obj, Call):
        return obj
    op = value.fname if isinstance(value, Call) else None
    fn = obj.fname
    args = list(obj.args)
    n = len(args)

    def is_dot(a):
        return isinstance(a, Sym) and a.name == "."

    def put(i, wrap_ops):
        name, a = args[i]
        if is_dot(a) and op in wrap_ops:
            args[i] = (name, Call(Sym("("), ((None, expand_dots(a, value)),)))
        else:
            args[i] = (name, expand_dots(a, value))

    pm = ("+", "-")
    if fn == "+":
        if n not in (1, 2):
            raise RError("invalid formula in 'update'")
        for i in range(n):
            put(i, ())
    elif fn == "-":
        if n not in (1, 2):
            raise RError("invalid formula in 'update'")
        for i in range(n):
            put(i, pm)
    elif fn in ("*", "/"):
        if n != 2:
            raise RError("invalid formula in 'update'")
        put(0, pm)
        put(1, pm)
    elif fn == ":":
        if n != 2:
            raise RError("invalid formula in 'update'")
        put(0, ("+", "-", "*", "/"))
        put(1, pm)
    elif fn == "^":
        if n != 2:
            raise RError("invalid formula in 'update'")
        put(0, ("+", "-", "*", "/", ":"))
        put(1, pm)
    else:
        new_fn = expand_dots(obj.fn, value)
        return Call(new_fn, tuple((nm, expand_dots(a, value)) for nm, a in args))
    return Call(obj.fn, tuple(args))
