"""Generate R reference results for path models (tests/fixtures/r_path.json).

For every multi-line model R computes
  * lm() per equation (coefficients), and the propagated predictions on new data
    (exogenous variables observed, endogenous ones replaced by their predictions,
    upstream first);
  * for models written in plain lavaan syntax, lavaan::sem() (meanstructure = TRUE,
    auto.cov.y = FALSE: uncorrelated disturbances) regression/intercept estimates and the
    total effects (I - B)^-1 - I from lavInspect(fit, "est")$beta.

Usage: python tools/make_path_fixtures.py   (needs Rscript with lavaan + jsonlite)
"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from statsformula._path import parse_path_model, path_order  # noqa: E402
from path_cases import CASES, make_data  # noqa: E402

OUT = os.path.join(ROOT, "tests", "fixtures", "r_path.json")

R = r'''
suppressMessages({library(jsonlite); library(lavaan); library(splines)})
args <- commandArgs(TRUE)
cases <- fromJSON(args[1], simplifyVector = FALSE)
rd <- function(p) { d <- read.csv(p, stringsAsFactors = FALSE); d$f <- factor(d$f, levels = c("a", "b", "c")); d }
d <- rd(args[2]); nd <- rd(args[3])
res <- lapply(cases, function(cs) {
  out <- list()
  fits <- list()
  for (eq in cs$equations) {
    fit <- lm(as.formula(eq$formula), data = d)
    fits[[eq$response]] <- fit
    out$lm[[eq$response]] <- list(names = as.list(names(coef(fit))), coef = as.list(unname(coef(fit))))
  }
  nd2 <- nd
  pred <- list()
  for (r in unlist(cs$order)) {
    p <- unname(predict(fits[[r]], newdata = nd2))
    nd2[[r]] <- p
    pred[[r]] <- as.list(p)
  }
  out$pred <- pred
  if (isTRUE(cs$lavaan)) {
    fit <- sem(cs$model, data = d, meanstructure = TRUE, auto.cov.y = FALSE)
    pe <- parameterEstimates(fit)
    pe <- pe[pe$op %in% c("~", "~1") & pe$lhs %in% unlist(cs$endogenous), ]
    out$pe <- list(lhs = as.list(pe$lhs), op = as.list(pe$op), rhs = as.list(pe$rhs), est = as.list(pe$est))
    B <- lavInspect(fit, "est")$beta
    Tt <- solve(diag(nrow(B)) - B) - diag(nrow(B))
    out$total <- list(names = as.list(rownames(B)), m = lapply(seq_len(ncol(Tt)), function(j) as.list(unname(Tt[, j]))))
  }
  out
})
writeLines(toJSON(list(results = res, R = R.version.string, lavaan = as.character(packageVersion("lavaan"))),
                  auto_unbox = TRUE, digits = NA, null = "null"), args[4])
'''


def main():
    d, nd = make_data()
    cases = []
    for c in CASES:
        eqs = parse_path_model(c["model"])
        order = path_order(eqs)
        cases.append({"model": c["model"], "lavaan": c.get("lavaan", False),
                      "equations": [{"response": e.response, "formula": e.formula} for e in eqs],
                      "endogenous": [e.response for e in eqs], "order": [eqs[i].response for i in order]})
    with tempfile.TemporaryDirectory() as tmp:
        paths = [os.path.join(tmp, n) for n in ("cases.json", "d.csv", "nd.csv", "out.json", "run.R")]
        json.dump(cases, open(paths[0], "w"))
        d.to_csv(paths[1], index=False)
        nd.to_csv(paths[2], index=False)
        open(paths[4], "w").write(R)
        subprocess.run(["Rscript", paths[4], paths[0], paths[1], paths[2], paths[3]], check=True)
        out = json.load(open(paths[3]))
    json.dump({"cases": CASES, "results": out["results"], "R": out["R"], "lavaan": out["lavaan"]},
              open(OUT, "w"), indent=0)
    print("wrote", OUT, len(CASES), "cases;", out["R"], "lavaan", out["lavaan"])


if __name__ == "__main__":
    main()
