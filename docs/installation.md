# Installation

```bash
pip install statsformula                  # core: numpy, pandas, scipy
pip install "statsformula[sklearn]"       # scikit-learn estimators, pipelines, summaries
pip install "statsformula[sem]"           # semopy backend for structural equation models
pip install "statsformula[statsmodels]"   # statsmodels, used by the econometrics examples
```

| requirement | version |
|---|---|
| Python | 3.10 – 3.13 |
| numpy, pandas, scipy | core dependencies |
| scikit-learn | ≥ 1.6 (optional, `statsformula.sklearn`) |
| semopy | ≥ 2.3 (optional, `statsformula.sem`) |
| pySEM (twoertwein/pySEM), pysemforge | optional SEM backends |

R is **not** required at run time. R is needed only to regenerate the reference fixtures of
the test suite (see [Validation](validation.md)).

## Development install

```bash
git clone https://github.com/PedroBSB/statsformula
cd statsformula
python -m pip install -e ".[test,docs]"
python -m pytest                     # ~2,000 tests, about 15 s
python -m sphinx -b html docs docs/_build/html
```

The package is pure Python, with no compiled extensions. The numerical kernels that
determine R's exact output (LINPACK `dqrdc2`, LAPACK `DSTEQR`/`DSTEDC`, mgcv's Lanczos
iteration, R's Mersenne-Twister, Davies' algorithm) are translated to Python, so the results
do not depend on the BLAS/LAPACK build installed with numpy.
