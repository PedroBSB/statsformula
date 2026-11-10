# Datasets

`sf.load_data(name)` returns a bundled dataset as a pandas DataFrame, the equivalent of R's
`data(name)`. Only textbook datasets that are not available from another Python package are
bundled. The returned frame has `attrs["title"]`, `attrs["description"]` and
`attrs["source"]`, and factors are `Categorical` with R's level order. Called with no
argument, `sf.load_data()` returns the catalogue.

| name | rows × cols | description | source |
|---|---|---|---|
| `kmenta` | 20 × 5 | supply and demand for food: `consump`, `price`, `income`, `farmPrice`, `trend` | Kmenta (1986), *Elements of Econometrics*, Table 13-1 |
| `klein` | 22 × 14 | Klein's Model I, US 1920–1941: consumption, profits, wages, investment, capital, output, government, taxes | Greene (2003), *Econometric Analysis*, Table F15.1 |
| `grunfeld_greene` | 100 × 5 | investment, market value and capital of 5 US firms, 1935–1954 | Greene (2003), *Econometric Analysis*, Table F13.1 |

The files were taken from the R package systemfit (GPL ≥ 2).

Datasets that are available elsewhere and used in the examples:

| dataset | loader |
|---|---|
| Fisher's iris, diabetes progression (Efron et al. 2004), Wisconsin breast cancer | `sklearn.datasets.load_iris`, `load_diabetes(scaled=False)`, `load_breast_cancer` |
| Greene's credit card data, Spector & Mazzeo, Longley, Grunfeld (11 firms), US macro data | `statsmodels.api.datasets.ccard`, `spector`, `longley`, `grunfeld`, `macrodata` |
| Holzinger & Swineford (1939), Bollen's Political Democracy (1989) | `semopy.examples.holzinger39`, `political_democracy` |
