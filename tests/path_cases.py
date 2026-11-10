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


def make_feedback_data():
    """Non-recursive systems: a two-variable loop y1 <-> y2 with correlated disturbances
    and a three-variable cycle v1 -> v2 -> v3 -> v1."""
    rng = np.random.default_rng(20261111)
    n = 300
    x1, x2, x3 = (np.round(rng.normal(size=n), 5) for _ in range(3))
    e = rng.multivariate_normal([0, 0], [[1.0, 0.4], [0.4, 1.0]], n)
    A = np.array([[1.0, -0.5], [0.3, 1.0]])         # y1 = .5 y2 + x1 ;  y2 = -.3 y1 + .8 x2 + .3 x3
    Y = np.linalg.solve(A, np.column_stack([1 + x1, 2 + 0.8 * x2 + 0.3 * x3]).T + e.T).T
    C = np.array([[1.0, 0.0, -0.4], [-0.5, 1.0, 0.0], [0.0, 0.3, 1.0]])  # v1 <- .4 v3, v2 <- .5 v1, v3 <- -.3 v2
    V = np.linalg.solve(C, np.column_stack([x1, x2, x3]).T + rng.normal(0, 0.7, (3, n))).T
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y1": np.round(Y[:, 0], 5), "y2": np.round(Y[:, 1], 5),
                         "v1": np.round(V[:, 0], 5), "v2": np.round(V[:, 1], 5), "v3": np.round(V[:, 2], 5)})


# standard errors against lavaan (information = "expected" / "observed") and summary(lm())
# "lavaan": the same model in lavaan syntax with labels; "defined": label -> (effects kind, lhs, rhs)
SE_CASES = [
    {"data": "d", "model": "m ~ x + z\ny ~ m + x", "lm": True,
     "lavaan": "m ~ a*x + z\ny ~ b*m + c*x\nind := a*b\ntot := c + a*b",
     "defined": {"ind": ["indirect", "y", "x"], "tot": ["total", "y", "x"]}},
    {"data": "d", "model": "m1 ~ x\nm2 ~ x + m1\ny ~ m1 + m2 + z", "lm": True,
     "lavaan": "m1 ~ a*x\nm2 ~ b*x + c*m1\ny ~ d*m1 + e*m2 + z\n"
               "ind := a*d + b*e + a*c*e\nind1 := c*e\ntot1 := d + c*e",
     "defined": {"ind": ["indirect", "y", "x"], "ind1": ["indirect", "y", "m1"], "tot1": ["total", "y", "m1"]}},
    {"data": "d", "model": "m ~ x + log(w) + f\ny ~ m + x + z", "lm": True,
     "lavaan": "m ~ x + logw + fb + fc\ny ~ m + x + z", "rename": {"logw": "log(w)"}},
    {"data": "fb", "model": "y1 ~ y2 + x1\ny2 ~ y1 + x2 + x3",
     "lavaan": "y1 ~ b1*y2 + g1*x1\ny2 ~ b2*y1 + x2 + x3\n"
               "t11 := g1/(1 - b1*b2)\nt21 := b2*g1/(1 - b1*b2)\ni11 := g1/(1 - b1*b2) - g1",
     "defined": {"t11": ["total", "y1", "x1"], "t21": ["total", "y2", "x1"], "i11": ["indirect", "y1", "x1"]}},
    {"data": "fb", "model": "y1 ~ y2 + x1\ny2 ~ y1 + x2 + x3\ny1 ~~ y2",
     "lavaan": "y1 ~ b1*y2 + g1*x1\ny2 ~ b2*y1 + x2 + x3\ny1 ~~ y2\nt21 := b2*g1/(1 - b1*b2)",
     "defined": {"t21": ["total", "y2", "x1"]}},
    {"data": "fb", "model": "y1 ~ x1 + x2\ny2 ~ x2 + x3\ny1 ~~ y2",
     "lavaan": "y1 ~ x1 + x2\ny2 ~ x2 + x3\ny1 ~~ y2"},
    {"data": "fb", "model": "v1 ~ v3 + x1\nv2 ~ v1 + x2\nv3 ~ v2 + x3",
     "lavaan": "v1 ~ a*v3 + g*x1\nv2 ~ b*v1 + x2\nv3 ~ c*v2 + x3\nt31 := c*b*g/(1 - a*b*c)",
     "defined": {"t31": ["total", "v3", "x1"]}},
]
