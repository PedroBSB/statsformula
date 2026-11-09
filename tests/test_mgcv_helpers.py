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
    check_gam_penalties(X, r, cs)


def check_gam_penalties(X, r, cs):
    from test_r_reference import num
    pens = X.model_spec.penalties
    rp = r.get("pen", [])
    assert [p.name for p in pens] == [x["name"] for x in rp], cs["formula"]
    # cs: the shrunk null space depends on the platform LAPACK's arbitrary eigenbasis
    rtol = 1e-2 if '"cs"' in cs["formula"] else 1e-7
    for p, x in zip(pens, rp):
        assert p.off + 1 == x["off"], (cs["formula"], p.name)
        assert p.rank == x["rank"], (cs["formula"], p.name)
        RS = np.column_stack([num(c) for c in x["S"]])
        np.testing.assert_allclose(p.S, RS, rtol=rtol, atol=rtol * 1e-2 * np.abs(RS).max(),
                                   err_msg="%s %s" % (cs["formula"], p.name))


def run_gamfit_case(cs, DATA):
    d = DATA[cs["data"]]
    return sf.gam(cs["formula"], d, method=cs["method"], sp=cs.get("sp"), weights=cs.get("weights"),
                  gamma=cs.get("gamma", 1.0)), DATA[cs["newdata"]]


def check_gamfit_case(res, r, cs):
    from test_r_reference import num
    b, newdata = res
    yscale = np.abs(num(r["fitted"])).max()
    tol = 1e-5
    if r["sp"]:  # R leaves b$sp empty when every sp is fixed
        assert list(b.sp.index) == r["sp_names"], cs
        lsp, rlsp = np.log(b.sp.to_numpy()), np.log(num(r["sp"]))
        # A smoothing parameter running off to infinity (the smooth reduced to its null
        # space) stops wherever the optimiser's tolerance is met: R at ~1e5, here at the
        # upper bound.  Those are not compared; the fit then agrees to ~1e-4.
        finite = rlsp < 8
        if not finite.all():
            tol = 1e-4
        # cs: mgcv's shrinkage of the null space depends on an arbitrary LAPACK
        # eigenbasis (see check_gampen_case), which moves log|S|+ and the REML score
        cs_ = '"cs"' in cs["formula"]
        np.testing.assert_allclose(lsp[finite], rlsp[finite], err_msg=str(cs),
                                   atol=3e-2 if cs_ else 5e-3 if not finite.all() else 2e-3)
        assert (lsp[~finite] > 8).all()
    if '"cs"' in cs["formula"]:
        tol = 5e-3
    np.testing.assert_allclose(b.criterion, r["criterion"], rtol=2e-2 if '"cs"' in cs["formula"] else 10 * tol)
    np.testing.assert_allclose(b.fitted_values.to_numpy(), num(r["fitted"]), atol=tol * yscale, err_msg=str(cs))
    np.testing.assert_allclose(b.predict(newdata).to_numpy(), num(r["pred"]), atol=tol * yscale, err_msg=str(cs))
    np.testing.assert_allclose(b.edf.sum(), r["edf"], rtol=10 * tol)
    np.testing.assert_allclose(b.scale, r["scale"], rtol=tol)
    np.testing.assert_allclose(b.coefficients.to_numpy(), num(r["coef"]), rtol=1e-3, atol=10 * tol * yscale)
    # standard errors (Vp) and the parametric table of summary.gam
    se_tol = 5e-3 if '"cs"' in cs["formula"] else 50 * tol
    pr = b.predict(newdata, se_fit=True)
    np.testing.assert_allclose(pr["fit"].to_numpy(), num(r["pred"]), atol=tol * yscale, err_msg=str(cs))
    np.testing.assert_allclose(pr["se_fit"].to_numpy(), num(r["se_fit"]), rtol=se_tol, err_msg=str(cs))
    sm = b.summary()
    pt = r["ptable"]
    assert list(sm.p_table.index) == pt["names"], cs
    np.testing.assert_allclose(sm.p_table["Estimate"], num(pt["est"]), rtol=1e-3, atol=10 * tol * yscale)
    np.testing.assert_allclose(sm.p_table["Std. Error"], num(pt["se"]), rtol=se_tol, err_msg=str(cs))
    np.testing.assert_allclose(sm.p_table["t value"], num(pt["t"]), rtol=se_tol, atol=1e-3, err_msg=str(cs))
    np.testing.assert_allclose(sm.p_table["Pr(>|t|)"], num(pt["p"]), rtol=se_tol, atol=1e-12, err_msg=str(cs))
    np.testing.assert_allclose(sm.residual_df, r["residual_df"], rtol=10 * tol)
    np.testing.assert_allclose(sm.r_sq, r["r_sq"], rtol=10 * tol)
