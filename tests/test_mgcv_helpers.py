"""Helpers for the mgcv comparisons in test_r_reference.py."""
import numpy as np

import statsformula as sf


def run_gampen_case(cs, DATA):
    from statsformula._mgcv import make_spec, smooth_con
    from statsformula._eval import EvalContext
    from statsformula._parser import parse
    from statsformula._model import model_frame_from_terms
    spec = make_spec(parse(cs["smooth"]), EvalContext({}, {}))
    data = DATA[cs["data"]]
    datad = {c: sf._rtypes.to_r(data[c]) for c in data.columns}
    return smooth_con(spec, datad, None, absorb_cons=True)


def check_gampen_case(sms, r, cs):
    from test_r_reference import num
    assert [s.label for s in sms] == [x["label"] for x in r["smooths"]]
    # "cs": mgcv shrinks the two null-space eigenvalues of the cr penalty to different
    # values; the null-space basis returned by LAPACK's eigen solver is arbitrary, so
    # that part of the penalty agrees with R only to ~1e-3 (model matrices are exact).
    rtol = 2e-3 if '"cs"' in cs["smooth"] else 1e-7
    for sm, rs in zip(sms, r["smooths"]):
        assert len(sm.S) == len(rs["S"]), cs["smooth"]
        for S, RS in zip(sm.S, rs["S"]):
            RS = np.column_stack([num(c) for c in RS])
            np.testing.assert_allclose(S, RS, rtol=rtol, atol=rtol * 1e-2 * np.abs(RS).max(), err_msg=cs["smooth"])


def run_gam_case(cs, DATA):
    X = sf.gam_model_matrix(cs["formula"], DATA[cs["data"]], knots=cs.get("knots"))
    Xn = X.model_spec.get_model_matrix(DATA[cs["newdata"]]) if cs.get("newdata") else None
    return X, Xn


def uses_tprs(formula):
    """Thin plate (tp/ts) bases: R's eigenvector signs depend on LAPACK rounding."""
    import re
    f = formula.replace(" ", "")
    for m in re.finditer(r"\b(s|te|ti)\((.*?)\)(?=\+|$|-)", f):
        args = m.group(2)
        bs = re.search(r'bs=(?:c\()?"(\w+)"', args)
        if m.group(1) == "s" and (bs is None or bs.group(1) in ("tp", "ts")):
            return True
        if m.group(1) in ("te", "ti") and bs is not None and "tp" in args:
            return True
    return False


def _align_signs(A, R):
    """Flip whole columns of A whose sign is opposite to R (tp bases only)."""
    A = A.copy()
    for j in range(A.shape[1]):
        if np.allclose(A[:, j], -R[:, j], rtol=1e-8, atol=1e-9) and not np.allclose(A[:, j], R[:, j], rtol=1e-8, atol=1e-9):
            A[:, j] = -A[:, j]
    return A


def check_gam_case(res, r, cs):
    from test_r_reference import assert_matrix, num
    X, Xn = res
    A = X.to_numpy()
    rm = np.column_stack([num(c) for c in r["fit"]["columns"]])
    tp = False  # exact comparison (reference LAPACK sign conventions are reproduced)
    if tp:
        A = _align_signs(A, rm)
    assert_matrix(A, X.columns, r["fit"], cs["formula"])
    if Xn is not None:
        B = Xn.to_numpy()
        if tp:
            # use the same column flips as the fit matrix
            flips = np.array([np.allclose(X.to_numpy()[:, j], -rm[:, j], rtol=1e-8, atol=1e-9) and
                              not np.allclose(X.to_numpy()[:, j], rm[:, j], rtol=1e-8, atol=1e-9)
                              for j in range(A.shape[1])])
            B = np.where(flips[None, :], -B, B)
        assert_matrix(B, Xn.columns, r["new"], cs["formula"] + " (newdata)")
