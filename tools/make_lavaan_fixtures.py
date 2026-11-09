"""Generate lavaan reference results (tests/fixtures/r_lavaan.json).

For every model string R (lavaan) computes
  * ``lavParseModelString(model, as.data.frame. = TRUE)`` plus its ``modifiers`` and
    ``constraints`` attributes (or the error message),
  * ``lavaanify(model, ...)`` under a few option sets,
  * for some models, the parameter table that ``cfa()``/``sem()``/``growth()`` build
    (``lavaan(..., do.fit = FALSE)`` on an identity covariance matrix).

Usage: python tools/make_lavaan_fixtures.py   (needs Rscript with lavaan + jsonlite)
"""
import json, os, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "tests", "fixtures", "r_lavaan.json")

HS = """
# Holzinger & Swineford three-factor CFA
visual  =~ x1 + x2 + x3
textual =~ x4 + x5 + x6
speed   =~ x7 + x8 + x9
"""

PD = """
  # measurement model
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + a*y2 + b*y3 + c*y4
    dem65 =~ y5 + a*y6 + b*y7 + c*y8
  # regressions
    dem60 ~ ind60
    dem65 ~ ind60 + dem60
  # residual correlations
    y1 ~~ y5
    y2 ~~ y4 + y6
    y3 ~~ y7
    y4 ~~ y8
    y6 ~~ y8
"""

GROWTH = """
  i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4
  s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4
"""

GROWTH_COV = """
  i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4
  s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4
  # time-invariant covariates
  i ~ x1 + x2
  s ~ x1 + x2
  # time-varying covariates
  t1 ~ c1
  t2 ~ c2
  t3 ~ c3
  t4 ~ c4
"""

MEDIATION = """
  # direct effect
    Y ~ c*X
  # mediator
    M ~ a*X
    Y ~ b*M
  # indirect effect (a*b)
    ab := a*b
  # total effect
    total := c + (a*b)
"""

CASES = [
    # name, model, option sets, preset (or None)
    ("hs_cfa", HS, "std", "cfa"),
    ("pd_sem", PD, "std", "sem"),
    ("growth_basic", GROWTH, "growth", "growth"),
    ("growth_cov", GROWTH_COV, "growth", "growth"),
    ("mediation", MEDIATION, "std", "sem"),
    ("single_line", "f =~ x1 + x2 + x3", "std", "cfa"),
    ("semicolons", "f1 =~ x1 + x2 + x3; f2 =~ x4 + x5 + x6; f1 ~~ f2", "std", "cfa"),
    ("bang_comments", "! a comment\nf =~ x1 + x2 ! trailing\n  + x3 # another\n", "std", "cfa"),
    ("blank_lines", "\n\n  f =~ x1 + x2 + x3\n\n\n  g =~ x4 + x5 + x6\n\n", "std", "cfa"),
    ("continuation_plus", "f =~ x1 +\n  x2 +\n  x3\ng =~\n  x4 + x5 + x6", "std", "cfa"),
    ("continuation_star", "f =~ x1 + 0.5*\n x2 + x3", "std", None),
    ("tabs_cr", "f\t=~\tx1 +\tx2 + x3\r\ny ~ f", "std", "sem"),
    ("intercepts", "f =~ x1 + x2 + x3\nx1 ~ 1\nf ~ 0*1\nx2 ~ 0.5*1", "std", "cfa"),
    ("intercept_fixed0", "y ~ x\ny ~ 0", "std", "sem"),
    ("fixed_values", "f =~ 1*x1 + 0.5*x2 + -1*x3 + -0.25*x4 + 1e-3*x5", "std", "cfa"),
    ("free_na", "f =~ NA*x1 + x2 + x3\nf ~~ 1*f", "std", "cfa"),
    ("start_values", "f =~ x1 + start(0.8)*x2 + 0.7?x3\nx1 ~~ start(0.3)*x1", "std", "cfa"),
    ("labels_eq", "f =~ x1 + a*x2 + a*x3 + b*x4\nx2 ~~ v*x2\nx3 ~~ v*x3", "std", "cfa"),
    ("label_fun", 'f =~ x1 + label("lab2")*x2 + equal("f=~x2")*x3', "std", "cfa"),
    ("quoted_label", "f =~ x1 + 'q'*x2 + \"r\"*x3", "std", "cfa"),
    ("combined_mods", "f =~ x1 + start(1)*a*x2 + b*start(0.5)*x3", "std", "cfa"),
    ("bounds_mod", "f =~ x1 + lower(0)*x2 + upper(2)*x3 + lower(0.1)*upper(0.9)*x4", "std", "cfa"),
    ("constraints_bounds", "f =~ x1 + a*x2 + b*x3 + c*x4\na > 0\nb < 1\nc > -0.5\na == b", "std", "cfa"),
    ("constraints_nonlin", "f =~ x1 + a*x2 + b*x3 + c*x4\na == 2*b\nb^2 < c\nd := a/b\nc > sqrt(2)", "std", "cfa"),
    ("defined_only", "y ~ a*x1 + b*x2\nab := a + b\nab2 := a*b\nratio := a/(b+1)", "std", "sem"),
    ("thresholds", "f =~ u1 + u2 + u3 + u4\nu1 | t1 + t2\nu2 | t1\nu3 | a*t1 + t2 + t3", "std", None),
    ("thresholds_only", "u1 | t1 + t2 + t3\nu2 | t1\nf =~ u1 + u2 + x3", "std", None),
    ("scaling", "f =~ u1 + u2 + u3\nu1 | t1\nu2 | t1\nu3 | t1\nu1 ~*~ 1*u1\nu2 ~*~ u2", "std", None),
    ("formative", "f =~ y1 + y2 + y3\nc1 <~ x1 + x2 + x3\nf ~ c1", "std", None),
    ("formative_comp", "c1 <~ 1*x1 + x2 + x3\nc2 <~ x2 + x4\ny ~ c1 + c2", "std", None),
    ("multigroup_c", "f =~ x1 + c(a1, a2)*x2 + c(1, NA)*x3\nx1 ~~ c(v, v)*x1", "mg", None),
    ("multigroup_c_fixed", "f =~ x1 + c(0.5, 0.6)*x2 + x3\nf ~~ c(1, NA)*f\nx2 ~ c(i1, i2)*1", "mg", None),
    ("multigroup_single_label", "f =~ x1 + a*x2 + x3", "mg", None),
    ("multigroup_three", "f =~ x1 + c(a, b, a)*x2 + x3", "mg3", None),
    ("group_blocks", "group: 1\n f =~ x1 + x2 + x3\ngroup: 2\n f =~ x1 + a*x2 + x3", "mgb", None),
    ("group_blocks_labels", "group: A\n f =~ x1 + x2 + x3\n group: B\n f =~ x1 + x2 + x3\n x1 ~~ x2", "mgb", None),
    ("level_blocks", "level: 1\n fw =~ y1 + y2 + y3\n fw ~ x1\nlevel: 2\n fb =~ y1 + y2 + y3\n fb ~ w1", "std", None),
    ("level_blocks_int", "level: 1\n y ~ x\nlevel: 2\n y ~ w\n y ~ 1\n x ~ 1", "std", None),
    ("group_level", "group: 1\nlevel: 1\n fw =~ y1 + y2\nlevel: 2\n fb =~ y1 + y2\ngroup: 2\nlevel: 1\n fw =~ y1 + y2\nlevel: 2\n fb =~ y1 + y2", "mgb", None),
    ("level_within_between", "level: within\n f =~ y1 + y2 + y3\nlevel: between\n f =~ y1 + y2 + y3", "std", None),
    ("interaction", "y ~ x1 + x2 + x1:x2", "std", "sem"),
    ("interaction_lv", "f =~ x1 + x2 + x3\ng =~ x4 + x5 + x6\ny ~ f + g + f:g", "std", None),
    ("efa_block", 'efa("efa1")*f1 +\nefa("efa1")*f2 =~ x1 + x2 + x3 + x4 + x5 + x6', "std", None),
    ("efa_two_sets", 'efa("a")*f1 + efa("a")*f2 =~ x1 + x2 + x3 + x4\nefa("b")*f3 + efa("b")*f4 =~ x5 + x6 + x7 + x8\nf3 ~ f1', "std", None),
    ("efa_plus_cfa", 'efa("e")*f1 + efa("e")*f2 =~ x1 + x2 + x3 + x4 + x5\nf3 =~ x6 + x7 + x8', "std", None),
    ("cov_reorder", "f =~ x1 + x2\ng =~ x3 + x4\ng ~~ f\nx4 ~~ x1\nx2 ~~ x3", "std", "cfa"),
    ("lhs_plus", "f + g =~ x1 + x2\nx1 + x2 ~ z", "std", None),
    ("multi_lhs_cov", "x1 + x2 ~~ x3 + x4", "std", None),
    ("duplicate_rows", "f =~ x1 + x2 + x3\nf =~ x2\ny ~ f", "std", None),
    ("higher_order", "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nf3 =~ x7 + x8 + x9\ng =~ f1 + f2 + f3", "std", "cfa"),
    ("single_indicator", "f =~ x1\ng =~ x2 + x3 + x4\nf ~ g", "std", "sem"),
    ("phantom_lv", "f =~ x1 + x2 + x3\nph =~ 0\nph ~~ 1*ph", "std", None),
    ("regressions_only", "y1 ~ x1 + x2\ny2 ~ y1 + x3\ny1 ~~ y2", "std", "sem"),
    ("exo_cov_user", "y ~ x1 + x2\nx1 ~~ x2", "std", None),
    ("odd_whitespace", "   f=~x1+x2   +x3\n\n   y~f;y~~y   ", "std", "sem"),
    ("long_names", "Visual.Factor_1 =~ item.01 + item_02 + .item3\n.item3 ~~ item_02", "std", None),
    ("numbers_exp", "f =~ 1.5e+0*x1 + 2E-1*x2 + .5*x3 + 5.*x4", "std", None),
    ("fixed_zero_cov", "f =~ x1 + x2 + x3\ng =~ x4 + x5 + x6\nf ~~ 0*g", "std", "cfa"),
    ("orthogonal_case", "f =~ x1 + x2 + x3\ng =~ x4 + x5 + x6\nh =~ x7 + x8 + x9", "orth", "cfa"),
    ("std_lv_case", "f =~ x1 + x2 + x3\ng =~ x4 + x5 + x6\ng ~ f", "stdlv", "sem"),
    ("residual_variance_fixed", "f =~ x1 + x2 + x3\nx1 ~~ 0*x1\nf ~~ 0.5*f", "std", "cfa"),
    ("bounds_and_eq", "f =~ x1 + a*x2 + a*x3\na > 0.1\na < 3", "std", None),
    ("constraint_file_style", "y ~ b1*x1 + b2*x2 + b3*x3\nb1 == (b2 + b3)^2\nb1 > exp(b2 + b3)", "std", None),
    ("mixed_mods_order", "f =~ x1 + a*start(0.5)*x2 + NA*b*x3", "std", None),
    ("rhs_number_eq", "f =~ x1 + x2\ng =~ 1", "std", None),
    ("group_equal_loadings", "f =~ x1 + x2 + x3\ng =~ x4 + x5 + x6", "ge", None),
    ("tilde_unicode", "f =˜ x1 + x2 + x3", "std", None),
    ("colon_in_mod_line", "f =~ x1 +\n\n x2\n\n + x3", "std", None),
    ("many_comments", "# header\n#\n f =~ x1 # first\n # middle comment\n  + x2 ! second\n\n  + x3 # last", "std", None),
    ("label_c_t", "y ~ c*x1 + t*x2", "std", None),
    ("growth_quadratic", "i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4\ns =~ 0*t1 + 1*t2 + 2*t3 + 3*t4\nq =~ 0*t1 + 1*t2 + 4*t3 + 9*t4", "growth", "growth"),
    ("ordinal_mg", "f =~ u1 + u2 + u3\nu1 | t1 + t2\nu2 | t1 + t2\nu3 | t1", "mgord", None),
    ("theta_param", "f =~ u1 + u2 + u3\nu1 | t1\nu2 | t1\nu3 | t1", "theta", None),
    # edge cases
    ("label_na", "f =~ x1 + c(a, NA)*x2 + x3", "mg", None),
    ("label_vec_fun", 'f =~ x1 + label(c("a", "b"))*x2 + start(c(0.5, 0.6))*x3', "mg", None),
    ("string_with_hash", 'f =~ x1 + label("a#b")*x2 + equal("f=~x2")*x3 # real comment', "std", None),
    ("math_mods", "f =~ pi*x1 + 2^2*x2 + sqrt(4)*x3 + (1+1)*x4 + -2*x5 + +3*x6", "std", None),
    ("name_like_consts", "f =~ x1 + T*x2 + Inf*x3 + F*x4", "std", None),
    ("bound_inf", "f =~ x1 + a*x2 + b*x3\na < 1/0\nb > -1\na > b", "std", None),
    ("bound_label_mg", "f =~ x1 + c(a1, a2)*x2 + x3\na1 > 0\na2 < 5", "mg", None),
    ("unicode_names", "fé =~ x1 + größe + x3", "std", None),
    ("cov_multi", "x1 ~~ x1 + x2 + x3\nx2 ~~ 0.5*x3", "std", None),
    ("start_and_fixed", "y ~ 0.5*x + start(1)*x", "std", None),
    ("rhs_minus_mod", "y ~ -1*x1 + -0.5*x2", "std", None),
    ("group_w", "f =~ x1 + x2 + x3", "misc", None),
    ("misc_sem", "f =~ x1 + x2 + x3\ng =~ x4 + x5 + x6\nh =~ x7 + x8\ng ~ f + z1\nh ~ g + z2\nx1 ~~ x4", "misc", None),
    ("misc_comp", "c1 <~ x1 + x2 + x3\nf =~ y1 + y2 + y3\nf ~ c1\nc2 <~ x3 + x4", "comp", None),
    ("cond_x", "f =~ y1 + y2 + y3\nf ~ x1 + x2\ny1 ~ x3", "condx", None),
    ("err_group_count", "group: 1\n f =~ x1 + x2 + x3", "mgb", None),
    ("err_upper_inf", "f =~ x1 + upper(Inf)*x2", "none", None),
    ("err_label_num", "f =~ x1 + label(1)*x2", "none", None),
    ("efa_echelon", 'efa("e")*f1 + efa("e")*f2 + efa("e")*f3 =~ x1 + x2 + x3 + x4 + x5 + x6 + x7', "efa", None),
    ("multi_line_efa", 'efa("e")*f1 +\n  efa("e")*f2 =~\n  x1 + x2 +\n  x3 + x4', "efa", None),
    ("interaction_lhs_cov", "y ~ a*x1 + b*x2 + c*x1:x2\nx1:x2 ~~ x1", "std", None),
    ("duplicate_mod_overwrite", "f =~ x1 + 0.5*x2\nf =~ 0.7*x2", "std", None),
    ("ordered_exo_warn", "y ~ u1\nu1 | t1", "std", None),
    ("double_semicolon", "f =~ x1 + x2 ;; f ~~ f ;\n;", "std", None),
    ("quote_in_comment", 'f =~ x1 + x2 # say "hi\ny ~ f', "std", None),
    ("quoted_group_labels", 'group: "A"\n f =~ x1 + x2 + x3\ngroup: "B"\n f =~ x1 + x2 + x3', "mgb", None),
    ("block_keyword", "block: 1\n y ~ x\nblock: 2\n y ~ x", "std", None),
    ("class_keyword", "class: 1\n y ~ x\nclass: 2\n y ~ x", "std", None),
    ("formula_before_block", "y ~ x\ngroup: 1\n y ~ z\ngroup: 2\n y ~ z", "mgb", None),
    ("op_at_line_end", "f =~\n\n x1 + x2\ny ~\n f", "std", None),
    ("lhs_continuation", "f1 +\n f2 =~ x1 + x2 + x3", "std", None),
    ("scaling_intercepts", "f =~ u1 + u2\nu1 | t1 + t2\nu2 | t1 + t2\nu1 ~ 0.3*1\nu2 ~~ 0.5*u2", "theta", None),
    ("reversed_cov_labels", "f =~ x1 + x2 + x3\nx3 ~~ r*x1\ny ~ f\nf ~~ y", "std", None),
    ("growth_free_loadings", "i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4\ns =~ 0*t1 + t2 + t3 + 1*t4\ni ~~ 0*s", "growth", "growth"),
    ("level_mg_labels", "level: 1\n y ~ a*x\nlevel: 2\n y ~ b*w\n a == b", "std", None),
    ("err_level_group_order", "level: 1\n y ~ x\nlevel: 2\n y ~ w\ngroup: 2\n y ~ x", "none", None),
    ("err_def_lhs", ":= a*b", "none", None),
    ("err_block_no_label", "group:\n y ~ x", "none", None),
    ("dv_operator_parse", "y ~ b*x\nb :~ b0 + b1*data(z)", "none", None),
    ("percent_operator_parse", "f =~ x1 + x2\ngroup % w", "none", None),
    # error cases (parser)
    ("err_no_operator", "f =~ x1 + x2\n x3 x4", "none", None),
    ("err_unexpected_char", "f =~ x1 + x2 + $x3", "none", None),
    ("err_self_regression", "y ~ y", "none", None),
    ("err_reserved", "f =~ x1 + if", "none", None),
    ("err_start_char", "f =~ start(a)*x1", "none", None),
    ("err_unknown_fun", "f =~ foo(1)*x1", "none", None),
    ("err_block_id", "group 1: \n f=~x", "none", None),
    ("err_nested", "level: 1\n f =~ x\n group: 1\n f=~x", "none", None),
    ("err_def_rhs", "a :=", "none", None),
    ("err_trailing_star", "x ~ 2*", "none", None),
    ("err_three_way", "y ~ x1:x2:x3", "none", None),
    ("err_empty", "", "none", None),
    ("err_only_comments", "# nothing\n ! here\n", "none", None),
    ("err_parens", "f =~ (x1)", "none", None),
    ("err_lhs_missing", "=~ x1", "none", None),
    ("err_mg_args", "f =~ x1 + c(1,2,3)*x2 + x3", "mg", None),
    ("warn_spaces", "f =~ x1 + x2 x3", "std", None),
    ("warn_split_op", "f = ~ x1 + x2", "std", None),
    ("warn_mod_twice", "f =~ 1*x1 + 2*x2*x3", "std", None),
]

OPTION_SETS = {
    "none": [],
    "std": [
        {},
        {"auto": True, "model.type": "sem"},
        {"meanstructure": True, "int.ov.free": True, "auto.fix.first": True, "auto.var": True,
         "auto.fix.single": True, "auto.cov.lv.x": True, "auto.cov.y": True, "auto.th": True,
         "auto.delta": True, "fixed.x": True},
    ],
    "growth": [
        {},
        {"auto": True, "model.type": "growth", "meanstructure": True, "fixed.x": True},
        {"meanstructure": True, "int.lv.free": True, "auto.var": True, "auto.cov.lv.x": True,
         "auto.cov.y": True, "auto.fix.first": True, "fixed.x": False},
    ],
    "mg": [
        {"ngroups": 2},
        {"ngroups": 2, "auto": True, "model.type": "cfa", "meanstructure": True},
    ],
    "mg3": [
        {"ngroups": 3, "auto": True, "model.type": "cfa"},
    ],
    "mgb": [
        {"ngroups": 2},
        {"ngroups": 2, "auto": True, "model.type": "cfa", "meanstructure": True},
    ],
    "orth": [
        {"auto": True, "model.type": "cfa", "orthogonal": True},
        {"auto": True, "model.type": "cfa", "orthogonal": True, "std.lv": True},
    ],
    "stdlv": [
        {"auto": True, "model.type": "sem", "std.lv": True},
        {"auto.var": True, "std.lv": True, "auto.cov.lv.x": True, "orthogonal": True},
    ],
    "ge": [
        {"ngroups": 2, "auto": True, "model.type": "cfa", "meanstructure": True,
         "group.equal": ["loadings", "intercepts"]},
        {"ngroups": 2, "auto": True, "model.type": "cfa", "std.lv": True,
         "group.equal": ["loadings", "lv.variances", "residuals"],
         "group.partial": ["f=~x2"]},
        {"ngroups": 3, "auto": True, "model.type": "cfa", "meanstructure": True,
         "group.equal": ["means", "lv.covariances", "residual.covariances"]},
    ],
    "mgord": [
        {"ngroups": 2, "auto": True, "model.type": "cfa", "meanstructure": True},
        {"ngroups": 2, "auto": True, "model.type": "cfa", "meanstructure": True,
         "group.equal": ["loadings", "thresholds"]},
        {"ngroups": 2, "auto": True, "model.type": "cfa", "meanstructure": True,
         "parameterization": "theta", "group.equal": ["thresholds"]},
    ],
    "misc": [
        {"auto": True, "model.type": "sem", "orthogonal.y": True, "orthogonal.x": True,
         "auto.cov.x": True, "fixed.x": True},
        {"auto": True, "model.type": "sem", "meanstructure": True, "marker.int.zero": True},
        {"auto": True, "model.type": "cfa", "ngroups": 2, "group.w.free": True,
         "meanstructure": True, "int.lv.free": True},
        {"auto": True, "model.type": "cfa", "correlation": True},
        {"auto": True, "model.type": "cfa", "ngroups": 2, "group.equal": ["loadings"],
         "ceq.simple": True},
    ],
    "comp": [
        {"auto": True, "model.type": "sem"},
        {"auto": True, "model.type": "sem", "composites": False},
        {"auto": True, "model.type": "sem", "meanstructure": True, "composites.cov.free": True},
    ],
    "condx": [
        {"auto": True, "model.type": "sem", "conditional.x": True, "fixed.x": True,
         "meanstructure": True},
        {"auto": True, "model.type": "sem", "conditional.x": True, "fixed.x": False},
        {"auto": True, "model.type": "sem", "fixed.x": False, "auto.cov.x": True},
    ],
    "efa": [
        {"auto": True, "model.type": "cfa"},
        {"auto": True, "model.type": "cfa", "orthogonal.efa": True, "meanstructure": True},
        {"auto": True, "model.type": "cfa", "auto.efa": False},
        {"auto": True, "model.type": "cfa", "ngroups": 2, "group.equal": ["loadings"]},
    ],
    "theta": [
        {"auto": True, "model.type": "cfa", "meanstructure": True, "parameterization": "theta"},
        {"auto": True, "model.type": "cfa", "meanstructure": True, "auto.th": False},
    ],
}

RCODE = r'''
suppressPackageStartupMessages({library(lavaan); library(jsonlite)})
work <- commandArgs(TRUE)[1]
cases <- fromJSON(file.path(work, "cases.json"), simplifyVector = FALSE)
enc_col <- function(x) {
  if (is.numeric(x)) {
    lapply(x, function(v) if (is.na(v)) NULL else if (is.infinite(v)) (if (v > 0) "Inf" else "-Inf") else v)
  } else as.list(x)
}
enc_df <- function(df) {
  out <- lapply(as.list(df), enc_col)
  list(columns = names(df), data = out)
}
capture <- function(expr) {
  warns <- character(0)
  val <- tryCatch(withCallingHandlers(expr, warning = function(w) {
    warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }),
    error = function(e) structure(conditionMessage(e), class = "err"))
  list(val = val, warns = warns)
}
res <- lapply(cases, function(cs) {
  m <- cs$model
  out <- list(name = cs$name)
  p <- capture(lavParseModelString(m, as.data.frame. = TRUE))
  if (inherits(p$val, "err")) {
    out$parse <- list(error = unclass(p$val), warnings = as.list(p$warns))
  } else {
    df <- p$val
    mods <- lapply(attr(df, "modifiers"), function(mm) lapply(mm, function(v) enc_col(v)))
    cons <- lapply(attr(df, "constraints"), function(cc) cc)
    out$parse <- c(enc_df(df), list(modifiers = mods, constraints = cons, warnings = as.list(p$warns)))
  }
  out$lavaanify <- lapply(cs$options, function(op) {
    args <- c(list(model = m), lapply(op, function(v) if (is.list(v)) unlist(v) else v))
    r <- capture(do.call(lavaanify, args))
    if (inherits(r$val, "err")) list(options = op, error = unclass(r$val), warnings = as.list(r$warns))
    else c(list(options = op), enc_df(as.data.frame(r$val)), list(warnings = as.list(r$warns)))
  })
  if (!is.null(cs$preset)) {
    fun <- get(cs$preset, asNamespace("lavaan"))
    r <- capture({
      ov <- lavNames(lavaanify(m), "ov"); S <- diag(length(ov)); dimnames(S) <- list(ov, ov)
      args <- list(model = m, sample.cov = S, sample.nobs = 100, do.fit = FALSE)
      if (cs$preset == "growth") args$sample.mean <- setNames(rep(0, length(ov)), ov)
      fit <- do.call(fun, args)
      pt <- parTable(fit); pt[, setdiff(names(pt), c("start", "est", "se"))]
    })
    out$preset <- if (inherits(r$val, "err")) list(preset = cs$preset, error = unclass(r$val))
                  else c(list(preset = cs$preset), enc_df(r$val))
  }
  out
})
writeLines(toJSON(res, auto_unbox = TRUE, null = "null", digits = NA), file.path(work, "out.json"))
'''


def main():
    payload = []
    for name, model, optset, preset in CASES:
        payload.append({"name": name, "model": model, "options": OPTION_SETS[optset],
                        "preset": preset})
    work = tempfile.mkdtemp()
    with open(os.path.join(work, "cases.json"), "w") as fh:
        json.dump(payload, fh)
    with open(os.path.join(work, "run.R"), "w") as fh:
        fh.write(RCODE)
    subprocess.run(["Rscript", os.path.join(work, "run.R"), work], check=True)
    with open(os.path.join(work, "out.json")) as fh:
        res = json.load(fh)
    for case, r in zip(payload, res):
        r["model"] = case["model"]
    with open(OUT, "w") as fh:
        json.dump({"lavaan_version": "0.7.3", "cases": res}, fh, indent=0)
    print("wrote", OUT, len(res), "cases")


if __name__ == "__main__":
    main()
