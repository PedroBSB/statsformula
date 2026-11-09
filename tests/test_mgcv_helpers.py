"""Helpers for the mgcv comparisons in test_r_reference.py."""
import numpy as np

import statsformula as sf


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
