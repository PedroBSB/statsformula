"""lavaan model syntax: compare lavaan_parse() / lavaanify() with lavaan 0.7.3 (R).

Reference results are produced by tools/make_lavaan_fixtures.py.
"""
import json
import math
import os
import re
import warnings

import pytest

import statsformula as sf
from statsformula._lavaan import LavaanSyntaxError, LavaanWarning, _r_num2str

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "r_lavaan.json")
with open(FIX) as fh:
    CASES = json.load(fh)["cases"]


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _core(msg):
    """Main message of a lavaan error/warning (without location and source excerpt)."""
    msg = re.sub(r"^lavaan( ERROR| WARNING)?:\s*", "", msg)
    msg = re.sub(r"^lavaan->[\w.]+\(\):\s*", "", msg)
    return _norm(msg.split(" at line ")[0])


def _rval(v):
    if v is None:
        return float("nan")
    if v == "Inf":
        return float("inf")
    if v == "-Inf":
        return float("-inf")
    return v


def _same(a, b):
    if isinstance(b, float) or isinstance(a, float):
        if a is None or b is None:
            return False
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= 1e-10 * max(1.0, abs(b))
    return a == b


def _compare_frame(df, ref, what):
    assert list(df.columns) == ref["columns"], what
    for col in ref["columns"]:
        rv = [_rval(v) for v in ref["data"][col]]
        pv = df[col].tolist()
        assert len(pv) == len(rv), (what, col)
        for i, (a, b) in enumerate(zip(pv, rv)):
            if ref["data"][col][i] is None and (a is None or isinstance(a, str)):
                assert a is None, (what, col, i, a)  # NA_character_
                continue
            if isinstance(b, float) and not isinstance(a, float):
                a = float("nan") if a is None or (hasattr(a, "__class__") and str(a) == "<NA>") else a
            assert _same(a, b), (what, col, i, a, b)


def _ids(cases):
    return [c["name"] for c in cases]


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_parse(case):
    ref = case["parse"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if "error" in ref:
            with pytest.raises(LavaanSyntaxError) as ei:
                sf.lavaan_parse(case["model"])
            if "lavaan->" in ref["error"]:
                assert _core(str(ei.value)) in _norm(ref["error"])
                loc = re.search(r"at line \d+, pos \d+", ref["error"])
                if loc:
                    assert loc.group(0) in str(ei.value)
            return
        df = sf.lavaan_parse(case["model"])
    _compare_frame(df, ref, "parse")
    # warnings
    lw = [str(x.message) for x in w if issubclass(x.category, LavaanWarning)]
    assert len(lw) == len(ref["warnings"])
    for a, b in zip(lw, ref["warnings"]):
        assert _core(a) in _norm(b)
    # modifiers attribute
    mods = df.attrs["modifiers"]
    assert len(mods) == len(ref["modifiers"])
    for pm, rm in zip(mods, ref["modifiers"]):
        assert sorted(pm) == sorted(rm)
        for k in rm:
            assert len(pm[k]) == len(rm[k])
            for a, b in zip(pm[k], rm[k]):
                if b is None:
                    assert a is None
                elif isinstance(b, str) and not isinstance(_rval(b), float):
                    assert a == b
                else:
                    assert _same(float(a), float(_rval(b)))
    # constraints attribute
    cons = df.attrs["constraints"]
    assert len(cons) == len(ref["constraints"])
    for pc, rc in zip(cons, ref["constraints"]):
        assert pc == {k: rc[k] for k in ("op", "lhs", "rhs", "user")}


LAVAANIFY = [(c, k) for c in CASES for k in range(len(c["lavaanify"]))]


@pytest.mark.parametrize("case,k", LAVAANIFY,
                         ids=["%s-%d" % (c["name"], k) for c, k in LAVAANIFY])
def test_lavaanify(case, k):
    ref = case["lavaanify"][k]
    opts = ref["options"] if isinstance(ref["options"], dict) else {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if "error" in ref:
            with pytest.raises(LavaanSyntaxError) as ei:
                sf.lavaanify(case["model"], **opts)
            assert _core(str(ei.value)) in _norm(ref["error"])
            return
        df = sf.lavaanify(case["model"], **opts)
    _compare_frame(df, ref, "lavaanify %r" % opts)
    # every warning R gave must be given too (R may give fewer: it caches parser results)
    pw = [_core(str(x.message)) for x in w if issubclass(x.category, LavaanWarning)]
    for rw in ref["warnings"]:
        assert any(p in _norm(rw) for p in pw), rw


PRESETS = [c for c in CASES if "preset" in c]


@pytest.mark.parametrize("case", PRESETS, ids=_ids(PRESETS))
def test_preset(case):
    ref = case["preset"]
    assert "error" not in ref
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LavaanWarning)
        df = sf.lavaanify(case["model"], preset=ref["preset"])
    _compare_frame(df, ref, "preset " + ref["preset"])


def test_option_names():
    m = "f =~ x1 + x2 + x3\ny ~ f"
    a = sf.lavaanify(m, **{"auto.var": True, "int.ov.free": True, "meanstructure": True})
    b = sf.lavaanify(m, auto_var=True, int_ov_free=True, meanstructure=True)
    assert a.equals(b)
    with pytest.raises(TypeError):
        sf.lavaanify(m, not_an_option=True)
    with pytest.raises(ValueError):
        sf.lavaanify(m, preset="mplus")


def test_preset_override():
    m = "f =~ x1 + x2 + x3"
    df = sf.lavaanify(m, preset="cfa", std_lv=True)
    assert df.loc[0, "free"] == 1 and df.loc[df.lhs.eq("f") & df.op.eq("~~"), "ustart"].item() == 1


def test_lavaan_names():
    m = "f =~ x1 + x2 + x3\ny ~ f + z"
    assert sf.lavaan_names(m, "ov") == ["x1", "x2", "x3", "y", "z"]
    assert sf.lavaan_names(m, "lv") == ["f"]
    assert sf.lavaan_names(m, "ov.x") == ["z"]
    assert sf.lavaan_names(m, "ov_y") == ["y"]


def test_list_input():
    a = sf.lavaan_parse(["f =~ x1 + x2", "y ~ f"])
    b = sf.lavaan_parse("f =~ x1 + x2\ny ~ f")
    assert a.equals(b)


@pytest.mark.parametrize("x,s", [(1.0, "1"), (0.5, "0.5"), (-0.25, "-0.25"), (1e5, "1e+05"),
                                 (123456.0, "123456"), (1e-4, "1e-04"), (0.00012, "0.00012"),
                                 (1 / 3, "0.333333333333333"), (0.1 + 0.2, "0.3"),
                                 (float("inf"), "Inf"), (None, "NA"), (2e-1, "0.2")])
def test_r_num2str(x, s):
    assert _r_num2str(x) == s
