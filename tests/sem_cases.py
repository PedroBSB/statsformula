"""SEM bridge cases (statsformula.sem) compared with lavaan (tools/make_sem_fixtures.py)."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def datasets():
    hs = pd.read_csv(os.path.join(HERE, "fixtures", "HolzingerSwineford1939.csv"))
    hs["school"] = pd.Categorical(hs["school"], categories=["Grant-White", "Pasteur"])
    pdm = pd.read_csv(os.path.join(HERE, "fixtures", "PoliticalDemocracy.csv"))
    rng = np.random.default_rng(20261111)
    n = 400
    F = rng.normal(size=n)
    syn = pd.DataFrame({"x1": F + rng.normal(0, .5, n), "x2": 0.8 * F + rng.normal(0, .5, n),
                        "x3": 1.2 * F + rng.normal(0, .5, n), "w": rng.uniform(1, 3, n),
                        "z": rng.normal(size=n),
                        "f": pd.Categorical(rng.choice(list("abc"), n), categories=list("abc"))})
    syn["m"] = 0.5 * F + np.log(syn.w) + (syn.f == "b") * 0.5 + 0.3 * syn.z + rng.normal(0, .5, n)
    syn["y"] = 0.7 * syn.m + 0.3 * F + 0.2 * syn.z ** 2 + rng.normal(0, .5, n)
    syn = syn.round(6)
    return {"hs": hs, "pd": pdm, "syn": syn}


HS3 = "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9"
CASES = [
    {"name": "hs_cfa", "data": "hs", "model": HS3, "preset": "cfa"},
    {"name": "hs_cfa_stdlv", "data": "hs", "model": HS3, "preset": "cfa", "options": {"std.lv": True}},
    {"name": "hs_second_order", "data": "hs", "model": HS3 + "\ng =~ visual + textual + speed", "preset": "cfa"},
    {"name": "hs_mimic_exprs", "data": "hs", "preset": "sem",
     "model": "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\n"
              "visual ~ a*ageyr + school   # factor covariate\n"
              "textual ~ b*visual + log(ageyr)\nab := a*b"},
    {"name": "pd_sem", "data": "pd", "preset": "sem",
     "model": "ind60 =~ x1 + x2 + x3\ndem60 =~ y1 + y2 + y3 + y4\ndem65 =~ y5 + y6 + y7 + y8\n"
              "dem60 ~ a*ind60\ndem65 ~ c*ind60 + b*dem60\n"
              "y1 ~~ y5\ny2 ~~ y4 + y6\ny3 ~~ y7\ny4 ~~ y8\ny6 ~~ y8\n"
              "indirect := a*b\ntotal := c + a*b"},
    {"name": "syn_latent_exprs", "data": "syn", "preset": "sem",
     "model": "ind =~ x1 + x2 + x3\nm ~ a*ind + log(w) + f + z\ny ~ b*m + ind + I(z^2)\nab := a*b"},
    {"name": "syn_path", "data": "syn", "preset": "sem",
     "model": "m ~ a*z + log(w) +\n    f\ny ~ b*m + poly(x1, 2)\nab := a*b"},
]
