"""Path-model cases compared with R (tools/make_path_fixtures.py)."""
import numpy as np
import pandas as pd


def make_data():
    rng = np.random.default_rng(20261110)
    n = 250
    x = np.round(rng.normal(size=n), 5)
    z = np.round(rng.normal(size=n), 5)
    w = np.round(rng.uniform(1, 4, n), 5)
    f = pd.Categorical(rng.choice(["a", "b", "c"], n), categories=["a", "b", "c"])
    fe = np.array([0.0, 0.6, -0.4])[f.codes]
    m = 0.5 * x + 0.3 * z + fe + rng.normal(0, 0.5, n)
    m1 = 0.6 * x + rng.normal(0, 0.5, n)
    m2 = 0.4 * x + 0.5 * m1 + rng.normal(0, 0.5, n)
    y = 0.7 * m + 0.2 * x + np.log(w) + 0.3 * m2 + rng.normal(0, 0.5, n)
    y2 = -0.4 * m + 0.5 * z + rng.normal(0, 0.5, n)
    d = pd.DataFrame({"x": x, "z": z, "w": w, "f": f, "m": np.round(m, 5), "m1": np.round(m1, 5),
                      "m2": np.round(m2, 5), "y": np.round(y, 5), "y2": np.round(y2, 5)})
    nd = d.iloc[:15].copy().reset_index(drop=True)
    nd["x"] = np.round(np.linspace(-2, 2, 15), 5)
    nd["m"] = 99.0   # endogenous values in new data are ignored when propagating
    return d, nd


CASES = [
    {"model": "m ~ x + z\ny ~ m + x", "lavaan": True},
    {"model": "\n  # serial mediation\n  m1 ~ x\n  m2 ~ x + m1\n  y ~ m1 + m2 + z\n", "lavaan": True},
    {"model": "y + y2 ~ m + x\nm ~ z", "lavaan": True},
    {"model": "m ~ x + z; y ~ m + x\ny ~ z   # merged with the line above", "lavaan": True},
    {"model": "y ~ m2 + m1 +\n    x\nm2 ~ m1 + x\nm1 ~ x", "lavaan": True},
    {"model": "m ~ x + f\ny ~ m + log(w) + poly(z, 2)"},
    {"model": "m ~ x * f\ny ~ m:x + z + I(m^2)\ny2 ~ m + ns(z, 3)"},
    {"model": "m1 ~ x\nm2 ~ m1 + f\ny ~ m2 + exp(m1 / 5) + scale(w)"},
]
