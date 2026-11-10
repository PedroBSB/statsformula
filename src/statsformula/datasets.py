"""Textbook datasets that are not available from other Python packages.

>>> import statsformula as sf
>>> sf.load_data()                     # the catalogue                     # doctest: +SKIP
>>> kmenta = sf.load_data("kmenta")    # a pandas DataFrame                 # doctest: +SKIP
>>> kmenta.attrs["source"]                                                  # doctest: +SKIP

Datasets that already ship with scikit-learn (``load_diabetes``, ``load_iris``, ...),
statsmodels (``sm.datasets.grunfeld``, ``ccard``, ``spector``, ``longley``, ``macrodata``) or
semopy (``semopy.examples.political_democracy``, ``holzinger39``) are not duplicated here.
The CSV files are in ``statsformula/data``; they were taken from the R package systemfit
1.1-30 (GPL >= 2).
"""

from importlib import resources
from typing import Optional

import pandas as pd

_DATASETS = {
    "kmenta": {
        "file": "kmenta.csv",
        "title": "Kmenta's supply and demand for food (20 years)",
        "description": (
            "Partly artificial market data for food: consump (food consumption per capita), "
            "price (ratio of food prices to general consumer prices), income (disposable income "
            "per capita), farmPrice (ratio of prices received by farmers to general consumer "
            "prices, lagged one year), trend (1, ..., 20). Price and consumption are jointly "
            "determined: the standard example of a simultaneous-equations (supply and demand) model."),
        "source": "Kmenta, J. (1986). Elements of Econometrics, 2nd ed. Macmillan, New York. "
                  "Table 13-1, p. 687 (via R package systemfit).",
    },
    "klein": {
        "file": "klein.csv",
        "title": "Klein's Model I of the US economy, 1920-1941",
        "description": (
            "Annual data, billions of 1934 dollars: consump (consumption), corpProf (profits), "
            "corpProfLag (profits lagged), privWage (private wage bill), invest (net investment), "
            "capitalLag (capital stock lagged), gnp (private output), gnpLag, govWage (government "
            "wage bill), govExp (government non-wage spending), taxes, wages (privWage + govWage), "
            "trend (year - 1931). The lagged values are missing in 1920."),
        "source": "Klein, L. R. (1950). Economic Fluctuations in the United States, 1921-1941. Wiley. "
                  "Greene, W. H. (2003). Econometric Analysis, 5th ed. Prentice Hall. Appendix F, "
                  "Table F15.1 (via R package systemfit).",
    },
    "grunfeld_greene": {
        "file": "grunfeld_greene.csv",
        "title": "Grunfeld's investment data, Greene's five-firm version, 1935-1954",
        "description": (
            "Panel of 5 US firms (General Motors, Chrysler, General Electric, Westinghouse, US "
            "Steel) x 20 years: invest (gross investment), value (market value of the firm), "
            "capital (stock of plant and equipment), firm (factor), year. This is the version "
            "used in Greene's panel-data and SUR examples; statsmodels' grunfeld is the 11-firm "
            "data of Kleiber and Zeileis (2008), with some different values."),
        "source": "Grunfeld, Y. (1958). The Determinants of Corporate Investment. PhD thesis, "
                  "University of Chicago. Greene, W. H. (2003). Econometric Analysis, 5th ed. "
                  "Prentice Hall. Appendix F, Table F13.1 (via R package systemfit).",
        "factors": {"firm": ["General Motors", "Chrysler", "General Electric", "Westinghouse", "US Steel"]},
    },
}


def load_data(name: Optional[str] = None) -> pd.DataFrame:
    """Load a bundled dataset as a DataFrame (R's ``data(name)``).

    Without ``name``, returns the catalogue (``name``, ``title``, ``rows``, ``columns``,
    ``source``).  The returned frame carries ``attrs["title"]``, ``attrs["description"]``
    and ``attrs["source"]``; factors are pandas ``Categorical`` with R's level order.
    """
    if name is None:
        rows = []
        for key, meta in _DATASETS.items():
            df = load_data(key)
            rows.append({"name": key, "title": meta["title"], "rows": df.shape[0],
                         "columns": df.shape[1], "source": meta["source"]})
        return pd.DataFrame(rows)
    key = name.lower().replace("-", "_")
    if key not in _DATASETS:
        raise KeyError("unknown dataset %r; available: %s" % (name, ", ".join(_DATASETS)))
    meta = _DATASETS[key]
    with resources.files("statsformula").joinpath("data", meta["file"]).open("r", encoding="utf-8") as fh:
        df = pd.read_csv(fh)
    for col, levels in meta.get("factors", {}).items():
        df[col] = pd.Categorical(df[col], categories=levels)
    df.attrs.update({"title": meta["title"], "description": meta["description"], "source": meta["source"]})
    return df
