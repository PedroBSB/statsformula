"""Run every script in examples/ (the README shows their output), so the documentation
cannot drift from the code."""

import glob
import os
import runpy
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = sorted(glob.glob(os.path.join(ROOT, "examples", "*.py")))


@pytest.mark.parametrize("path", SCRIPTS, ids=[os.path.basename(p) for p in SCRIPTS])
def test_example_runs(path, capsys):
    import re
    source = open(path, encoding="utf-8").read()
    for mod in ("sklearn", "statsmodels", "semopy"):
        if re.search(r"^(import|from) %s\b" % mod, source, re.M):
            pytest.importorskip(mod)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runpy.run_path(path, run_name="__main__")
    out = capsys.readouterr().out
    assert len(out.splitlines()) > 5


def test_example_numbers():
    """Key numbers quoted in the README."""
    pytest.importorskip("sklearn")
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LinearRegression
    from statsformula.sklearn import FormulaEstimator
    raw = load_iris(as_frame=True)
    iris = raw.frame.drop(columns="target")
    iris.columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    iris["Species"] = pd.Categorical.from_codes(raw.target, ["setosa", "versicolor", "virginica"])
    s = FormulaEstimator("Sepal.Length ~ Petal.Length * Species",
                         LinearRegression(fit_intercept=False)).fit(iris).summary()
    # R: summary(lm(Sepal.Length ~ Petal.Length * Species, iris))
    assert round(s.r_squared, 4) == 0.8405 and round(s.sigma, 4) == 0.3365
    assert round(s.fstatistic[0], 1) == 151.7


def test_readme_examples_run(capsys):
    """The code blocks of the README's "Examples with real data" section run in order."""
    import re
    pytest.importorskip("sklearn")
    pytest.importorskip("semopy")
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    section = text[text.index("## Examples with real data"):text.index("## Why")]
    ns = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, block in enumerate(re.findall(r"```python\n(.*?)```", section, re.S)):
            exec(compile(block, "README block %d" % i, "exec"), ns)
    assert ns["search"].best_params_["formula"] == "progression ~ (sex + bmi + bp + ltg)^2 + hdl"
