"""R value types used while evaluating formula variables.

R distinguishes numeric vectors, logical vectors (with ``NA``), character
vectors, factors (with levels, ordering and an optional ``contrasts``
attribute) and matrices (with column names).  This module provides light
weight Python equivalents and conversions from/to numpy / pandas objects.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

try:  # pandas is a hard dependency, but keep imports local-friendly
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


class RError(Exception):
    """Error raised where R would signal an error."""


# ---------------------------------------------------------------------------
# Number formatting (R's formatReal with 15 significant digits)
# ---------------------------------------------------------------------------

def _sci_parts(x: float, digits: int):
    s = "%.*e" % (digits - 1, abs(x))
    mant, exp = s.split("e")
    mant = mant.replace(".", "").rstrip("0") or "0"
    return mant, int(exp)


def format_num(x: float, digits: int = 15, is_int: bool = False) -> str:
    """Format a single number the way R's ``as.character``/``deparse`` do."""
    if is_int:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "NA"
        return str(int(x))
    if x is None:
        return "NA"
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == 0:
        return "0"
    neg = 1 if x < 0 else 0
    mant, kp = _sci_parts(x, digits)
    nsig = len(mant)
    if kp >= 0:
        left = kp + 1
    else:
        left = 1
    rgt = max(0, nsig - kp - 1)
    fixed_w = neg + left + (rgt + 1 if rgt > 0 else 0)
    sci_w = neg + (nsig + 1 if nsig > 1 else 1) + (4 if abs(kp) < 100 else 5)
    if fixed_w <= sci_w:
        return "%.*f" % (rgt, x)
    m = mant[0] + ("." + mant[1:] if nsig > 1 else "")
    return "%s%se%s%02d" % ("-" if neg else "", m, "-" if kp < 0 else "+", abs(kp))


def format_num_vec(values: np.ndarray, is_int: bool = False) -> List[str]:
    return [format_num(v, is_int=is_int) for v in values]


# ---------------------------------------------------------------------------
# Collation (approximates ICU root collation used by R on macOS/Windows)
# ---------------------------------------------------------------------------

_PUNCT_ORDER = "_-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$"
_COLLATION = "icu"


def set_collation(kind: str) -> None:
    """Choose string collation used when sorting factor levels.

    ``"icu"`` (default) mimics R using ICU (macOS / Windows builds, and Linux
    builds with ICU); ``"C"`` mimics ``Sys.setlocale("LC_COLLATE", "C")``.
    """
    global _COLLATION
    if kind not in ("icu", "C"):
        raise ValueError("collation must be 'icu' or 'C'")
    _COLLATION = kind


def get_collation() -> str:
    return _COLLATION


def _char_keys(ch: str):
    if ch.isspace() or unicodedata.category(ch).startswith("C"):
        return (0, ord(ch)), 0, 0
    if ch.isdigit():
        return (3, unicodedata.digit(ch, 0)), 0, 0
    if ch.isalpha():
        decomp = unicodedata.normalize("NFD", ch)
        base = decomp[0]
        accent = sum(ord(c) for c in decomp[1:])
        tert = 1 if base.isupper() else 0
        return (4, ord(base.lower())), accent, tert
    pos = _PUNCT_ORDER.find(ch)
    if pos >= 0:
        return (1, pos), 0, 0
    return (2, ord(ch)), 0, 0


def r_sort_key(s: str):
    if _COLLATION == "C":
        return s.encode("utf-8")
    prim, sec, ter = [], [], []
    for ch in s:
        p, a, t = _char_keys(ch)
        prim.append(p)
        sec.append(a)
        ter.append(t)
    return (tuple(prim), tuple(sec), tuple(ter), s)


def r_sort_strings(values: Iterable[str]) -> List[str]:
    return sorted(values, key=r_sort_key)


# ---------------------------------------------------------------------------
# R vector classes
# ---------------------------------------------------------------------------

class Logical:
    """Logical vector with NA support."""

    __slots__ = ("values", "na")

    def __init__(self, values, na=None):
        self.values = np.asarray(values, dtype=bool).reshape(-1)
        if na is None:
            na = np.zeros(self.values.shape, dtype=bool)
        self.na = np.asarray(na, dtype=bool).reshape(-1)
        self.values = np.where(self.na, False, self.values)

    def __len__(self):
        return len(self.values)

    def isna(self):
        return self.na.copy()

    def take(self, idx):
        return Logical(self.values[idx], self.na[idx])

    def to_float(self):
        out = self.values.astype(float)
        out[self.na] = np.nan
        return out

    def __repr__(self):
        return "Logical(%r)" % ([None if n else bool(v) for v, n in zip(self.values, self.na)],)


class Character:
    """Character vector; ``None`` marks NA (stored as a numpy object array)."""

    __slots__ = ("values",)

    def __init__(self, values):
        arr = np.array(values, dtype=object).reshape(-1) if not isinstance(values, np.ndarray) \
            else values.astype(object, copy=True).reshape(-1)
        if len(arr):
            na = pd.isna(arr)
            if na.any():
                arr[na] = None
            if pd.api.types.infer_dtype(arr, skipna=True) not in ("string", "empty"):
                ok = ~na
                arr[ok] = np.array([str(x) for x in arr[ok]], dtype=object)
        self.values = arr

    def __len__(self):
        return len(self.values)

    def isna(self):
        return pd.isna(self.values)

    def take(self, idx):
        c = Character.__new__(Character)
        c.values = self.values[idx]
        return c

    def __repr__(self):
        return "Character(%r)" % (list(self.values),)


class Factor:
    """R factor: integer codes (``-1`` = NA) plus levels."""

    __slots__ = ("codes", "levels", "ordered", "contrasts")

    def __init__(self, codes, levels, ordered=False, contrasts=None):
        self.codes = np.asarray(codes, dtype=np.int64).reshape(-1)
        self.levels = [str(l) for l in levels]
        self.ordered = bool(ordered)
        self.contrasts = contrasts

    def __len__(self):
        return len(self.codes)

    @property
    def nlevels(self):
        return len(self.levels)

    def isna(self):
        return self.codes < 0

    def labels(self):
        out = np.empty(len(self.codes), dtype=object)
        lev = np.array(self.levels + [None], dtype=object)
        out[:] = lev[np.where(self.codes < 0, len(self.levels), self.codes)]
        return out

    def take(self, idx):
        return Factor(self.codes[idx], self.levels, self.ordered, self.contrasts)

    def drop_unused(self):
        used = np.unique(self.codes[self.codes >= 0])
        if len(used) == len(self.levels):
            return self
        mapping = -np.ones(len(self.levels) + 1, dtype=np.int64)
        mapping[used] = np.arange(len(used))
        codes = np.where(self.codes < 0, -1, mapping[np.where(self.codes < 0, 0, self.codes)])
        return Factor(codes, [self.levels[i] for i in used], self.ordered, None)

    def __repr__(self):
        return "Factor(%r, levels=%r)" % (list(self.labels()), self.levels)


class RMatrix:
    """Numeric matrix with optional column names and attributes."""

    __slots__ = ("data", "colnames", "attrs")

    def __init__(self, data, colnames=None, attrs=None):
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.data = data
        self.colnames = None if colnames is None else [str(c) for c in colnames]
        self.attrs = dict(attrs or {})

    def __len__(self):
        return self.data.shape[0]

    @property
    def ncol(self):
        return self.data.shape[1]

    def isna(self):
        return np.isnan(self.data).any(axis=1)

    def take(self, idx):
        return RMatrix(self.data[idx], self.colnames, self.attrs)

    def __repr__(self):
        return "RMatrix(shape=%r, colnames=%r)" % (self.data.shape, self.colnames)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def is_numeric(v) -> bool:
    return isinstance(v, np.ndarray) and v.dtype.kind in "fiu"


def rlen(v) -> int:
    if v is None:
        return 0
    if isinstance(v, (Logical, Character, Factor, RMatrix)):
        return len(v)
    if isinstance(v, np.ndarray):
        return v.shape[0] if v.ndim else 1
    return len(v)


def isna(v) -> np.ndarray:
    if isinstance(v, (Logical, Character, Factor, RMatrix)):
        return v.isna()
    arr = np.asarray(v)
    if arr.dtype.kind == "f":
        return np.isnan(arr) if arr.ndim == 1 else np.isnan(arr).any(axis=1)
    return np.zeros(arr.shape[0] if arr.ndim else 1, dtype=bool)


def take(v, idx):
    if isinstance(v, (Logical, Character, Factor, RMatrix)):
        return v.take(idx)
    return np.asarray(v)[idx]


def as_double(v) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v.astype(float)
    if isinstance(v, Logical):
        return v.to_float()
    if isinstance(v, Factor):
        out = (v.codes + 1).astype(float)
        out[v.codes < 0] = np.nan
        return out
    if isinstance(v, RMatrix):
        return v.data.reshape(-1, order="F")
    if isinstance(v, Character):
        out = np.empty(len(v))
        for i, s in enumerate(v.values):
            try:
                out[i] = float(s) if s is not None else np.nan
            except ValueError:
                out[i] = np.nan
        return out
    if v is None:
        return np.zeros(0)
    return np.asarray(v, dtype=float).reshape(-1)


def as_character(v) -> Character:
    if isinstance(v, Character):
        return v
    if isinstance(v, Factor):
        return Character(list(v.labels()))
    if isinstance(v, Logical):
        return Character([None if n else ("TRUE" if b else "FALSE") for b, n in zip(v.values, v.na)])
    if isinstance(v, np.ndarray):
        isint = v.dtype.kind in "iu"
        vals = v.reshape(-1)
        if isint:
            return Character(vals.astype(str).astype(object))
        nan = np.isnan(vals)
        u, inv = np.unique(vals[~nan], return_inverse=True)
        strs = np.array([format_num(x) for x in u], dtype=object)
        out = np.empty(len(vals), dtype=object)
        out[~nan] = strs[inv]
        out[nan] = None
        c = Character.__new__(Character)
        c.values = out
        return c
    if isinstance(v, RMatrix):
        return as_character(v.data.reshape(-1, order="F"))
    raise RError("cannot coerce to character")


def _codes_for(values: np.ndarray, levels) -> np.ndarray:
    """Vectorised level matching (hash based), -1 for NA / unknown."""
    if not len(values):
        return np.zeros(0, dtype=np.int64)
    return pd.Categorical(values, categories=pd.Index(list(levels), dtype=object)).codes.astype(np.int64)


def as_factor(v, levels=None, ordered=None) -> Factor:
    """Equivalent of ``factor(x)`` / ``as.factor(x)``."""
    if isinstance(v, Factor):
        if levels is None:
            f = v.drop_unused()
            if ordered is not None:
                f = Factor(f.codes, f.levels, ordered)
            return f
        v = as_character(v)
    if isinstance(v, Logical):
        if levels is None:
            present = sorted(set(np.unique(v.values[~v.na]).tolist()))
            levels = ["TRUE" if b else "FALSE" for b in present]
        codes = np.where(v.na, -1, v.values.astype(np.int64))
        lv = [str(l) for l in levels]
        mapping = np.array([lv.index("FALSE") if "FALSE" in lv else -1, lv.index("TRUE") if "TRUE" in lv else -1])
        codes = np.where(v.na, -1, mapping[np.where(v.na, 0, codes)])
        return Factor(codes, lv, bool(ordered))
    if isinstance(v, RMatrix):
        return as_factor(v.data.reshape(-1, order="F"), levels, ordered)
    if isinstance(v, np.ndarray) and v.dtype.kind in "fiu" and levels is None:
        isint = v.dtype.kind in "iu"
        vals = v.reshape(-1)
        if isint:
            u, inv = np.unique(vals, return_inverse=True)
            codes = inv.astype(np.int64)
        else:
            nan = np.isnan(vals)
            u, inv = np.unique(vals[~nan], return_inverse=True)
            codes = np.full(len(vals), -1, dtype=np.int64)
            codes[~nan] = inv
        return Factor(codes, [format_num(x, is_int=isint) for x in u], bool(ordered))
    if not isinstance(v, Character):
        v = as_character(v) if isinstance(v, np.ndarray) else to_r(v)
        if not isinstance(v, Character):
            return as_factor(v, levels, ordered)
    if levels is None:
        uniq = pd.unique(v.values[~pd.isna(v.values)]) if len(v.values) else []
        levels = r_sort_strings([str(x) for x in uniq])
    levels = [str(l) for l in levels]
    return Factor(_codes_for(v.values, levels), levels, bool(ordered))


# ---------------------------------------------------------------------------
# Conversion from Python objects
# ---------------------------------------------------------------------------

def _fmt_category(c) -> str:
    if isinstance(c, (bool, np.bool_)):
        return "TRUE" if c else "FALSE"
    if isinstance(c, (int, np.integer)):
        return str(int(c))
    if isinstance(c, (float, np.floating)):
        return format_num(float(c))
    return str(c)


def _object_to_r(arr: np.ndarray):
    kind = pd.api.types.infer_dtype(arr, skipna=True)
    na = pd.isna(arr)
    if kind == "boolean":
        vals = np.zeros(len(arr), dtype=bool)
        vals[~na] = arr[~na].astype(bool)
        return Logical(vals, na)
    if kind in ("integer", "floating", "mixed-integer-float", "decimal"):
        out = np.full(len(arr), np.nan)
        out[~na] = arr[~na].astype(float)
        return out
    return Character(arr)


def _is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if pd is not None:
        try:
            if x is pd.NA or x is pd.NaT:
                return True
        except Exception:
            pass
    return False


def to_r(obj: Any):
    """Convert a Python / numpy / pandas object to an R-like value."""
    if obj is None:
        return None
    if isinstance(obj, (Logical, Character, Factor, RMatrix)):
        return obj
    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            return RMatrix(obj.to_numpy(dtype=float), list(map(str, obj.columns)))
        if isinstance(obj, (pd.Series, pd.Index)):
            s = obj if isinstance(obj, pd.Series) else pd.Series(obj)
            dt = s.dtype
            if isinstance(dt, pd.CategoricalDtype):
                cat = s.cat
                return Factor(cat.codes.to_numpy(), [_fmt_category(c) for c in cat.categories], bool(cat.ordered))
            if dt == bool:
                return Logical(s.to_numpy())
            if str(dt) == "boolean":
                na = s.isna().to_numpy()
                return Logical(s.fillna(False).to_numpy(dtype=bool), na)
            if dt.kind in "iu":
                return s.to_numpy().astype(np.int64)
            if str(dt) in ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"):
                if s.isna().any():
                    return s.to_numpy(dtype=float, na_value=np.nan)
                return s.to_numpy(dtype=np.int64)
            if dt.kind == "f" or str(dt).startswith("Float"):
                return s.to_numpy(dtype=float, na_value=np.nan)
            if dt.kind == "M" or dt.kind == "m":
                raise RError("date/time columns are not supported in formulas; convert them to numeric first")
            return _object_to_r(s.to_numpy(dtype=object))
        if isinstance(obj, pd.Categorical):
            return to_r(pd.Series(obj))
    if isinstance(obj, (bool, np.bool_)):
        return Logical([bool(obj)])
    if isinstance(obj, (int, np.integer)):
        return np.array([int(obj)], dtype=np.int64)
    if isinstance(obj, (float, np.floating)):
        return np.array([float(obj)])
    if isinstance(obj, str):
        return Character([obj])
    if isinstance(obj, (list, tuple)):
        if len(obj) and all(isinstance(x, str) or x is None for x in obj):
            return Character(list(obj))
        return to_r(np.asarray(obj, dtype=object if any(x is None for x in obj) else None))
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2:
            if obj.dtype.kind in "fiub":
                return RMatrix(obj.astype(float))
            raise RError("only numeric matrices are supported")
        obj = obj.reshape(-1)
        if obj.dtype.kind == "b":
            return Logical(obj)
        if obj.dtype.kind in "iu":
            return obj.astype(np.int64)
        if obj.dtype.kind == "f":
            return obj.astype(float)
        if obj.dtype.kind in "US":
            return Character(list(obj))
        return _object_to_r(obj)
    try:
        arr = np.asarray(obj)
    except Exception as exc:  # pragma: no cover
        raise RError("cannot convert %r to an R value" % (type(obj),)) from exc
    if arr.dtype == object and arr.ndim == 0:
        raise RError("cannot convert %r to an R value" % (type(obj),))
    return to_r(arr)


def from_r(v):
    """Convert an R-like value to a numpy / pandas object for Python callables."""
    if isinstance(v, Factor):
        return pd.Categorical.from_codes(v.codes, v.levels, ordered=v.ordered)
    if isinstance(v, Logical):
        if v.na.any():
            return pd.array([None if n else bool(b) for b, n in zip(v.values, v.na)], dtype="boolean")
        return v.values.copy()
    if isinstance(v, Character):
        return v.values.copy()
    if isinstance(v, RMatrix):
        return v.data.copy()
    return v


def class_of(v) -> str:
    """R ``.MFclass`` of a model-frame variable."""
    if isinstance(v, Logical):
        return "logical"
    if isinstance(v, Factor):
        return "ordered" if v.ordered else "factor"
    if isinstance(v, Character):
        return "character"
    if isinstance(v, RMatrix):
        return "nmatrix.%d" % v.ncol
    if isinstance(v, np.ndarray) and v.dtype.kind in "fiu":
        return "numeric"
    return "other"
