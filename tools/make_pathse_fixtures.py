"""Generate R reference standard errors for path models (tests/fixtures/r_pathse.json).

For every case of tests/path_cases.py:SE_CASES R computes lavaan::sem(meanstructure = TRUE,
auto.cov.y = FALSE, fixed.x = TRUE) with information = "expected" and "observed"
(parameterEstimates(): est, se, z, pvalue, ci; defined parameters := by the delta
method) and, for "lm" cases, summary(lm()) of every equation.

Usage: python tools/make_pathse_fixtures.py   (needs Rscript with lavaan + jsonlite)
"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from statsformula._path import parse_path_system  # noqa: E402
from path_cases import SE_CASES, make_data, make_feedback_data  # noqa: E402

OUT = os.path.join(ROOT, "tests", "fixtures", "r_pathse.json")

R = r'''
suppressMessages({library(jsonlite); library(lavaan)})
args <- commandArgs(TRUE)
cases <- fromJSON(args[1], simplifyVector = FALSE)
d <- read.csv(args[2], stringsAsFactors = FALSE)
d$f <- factor(d$f, levels = c("a", "b", "c"))
d$logw <- log(d$w); d$fb <- as.numeric(d$f == "b"); d$fc <- as.numeric(d$f == "c")
fb <- read.csv(args[3])
col <- function(x) as.list(unname(x))
res <- lapply(cases, function(cs) {
  dat <- if (cs$data == "d") d else fb
  out <- list()
  for (inf in c("expected", "observed")) {
    fit <- sem(cs$lavaan, data = dat, meanstructure = TRUE, auto.cov.y = FALSE, fixed.x = TRUE, information = inf)
    pe <- parameterEstimates(fit)
    endo <- unlist(cs$endogenous)
    keep <- pe$op %in% c("~", "~1", ":=") | (pe$op == "~~" & pe$lhs %in% endo & pe$rhs %in% endo)
    keep <- keep & (pe$op != "~1" | pe$lhs %in% endo)
    pe <- pe[keep, ]
    out[[inf]] <- list(lhs = col(pe$lhs), op = col(pe$op), rhs = col(pe$rhs), label = col(pe$label),
                       est = col(pe$est), se = col(pe$se), z = col(pe$z), pvalue = col(pe$pvalue),
                       ci.lower = col(pe$ci.lower), ci.upper = col(pe$ci.upper),
                       converged = lavInspect(fit, "converged"), loglik = as.numeric(logLik(fit)),
                       fm = as.list(fitMeasures(fit, c("npar", "chisq", "df", "pvalue", "baseline.chisq",
                              "baseline.df", "baseline.pvalue", "cfi", "tli", "logl", "unrestricted.logl", "aic",
                              "bic", "bic2", "ntotal", "rmsea", "rmsea.ci.lower", "rmsea.ci.upper", "rmsea.pvalue", "srmr"))),
                       r2 = as.list(lavInspect(fit, "r2")),
                       printed = as.list(capture.output(summary(fit, rsquare = TRUE, fit.measures = TRUE))))
  }
  if (isTRUE(cs$lm)) {
    out$lm <- lapply(cs$equations, function(eq) {
      s <- summary(lm(as.formula(eq), data = dat))
      ct <- coef(s)
      ci <- confint(lm(as.formula(eq), data = dat))
      list(names = col(rownames(ct)), est = col(ct[, 1]), se = col(ct[, 2]), t = col(ct[, 3]), p = col(ct[, 4]),
           ci.lower = col(ci[, 1]), ci.upper = col(ci[, 2]), df = s$df[2])
    })
  }
  out
})
writeLines(toJSON(list(results = res, R = R.version.string, lavaan = as.character(packageVersion("lavaan"))),
                  auto_unbox = TRUE, digits = NA, null = "null", na = "null"), args[4])
'''


def main():
    d, _ = make_data()
    fb = make_feedback_data()
    cases = []
    for c in SE_CASES:
        eqs, _ = parse_path_system(c["model"])
        cases.append({"data": c["data"], "lavaan": c["lavaan"], "lm": c.get("lm", False),
                      "endogenous": [e.response for e in eqs], "equations": [e.formula for e in eqs]})
    with tempfile.TemporaryDirectory() as tmp:
        paths = [os.path.join(tmp, n) for n in ("cases.json", "d.csv", "fb.csv", "out.json", "run.R")]
        json.dump(cases, open(paths[0], "w"))
        d.to_csv(paths[1], index=False)
        fb.to_csv(paths[2], index=False)
        open(paths[4], "w").write(R)
        subprocess.run(["Rscript", paths[4], *paths[:4]], check=True)
        out = json.load(open(paths[3]))
    json.dump({"cases": SE_CASES, "results": out["results"], "R": out["R"], "lavaan": out["lavaan"]},
              open(OUT, "w"), indent=0)
    print("wrote", OUT, len(SE_CASES), "cases;", out["R"], "lavaan", out["lavaan"])


if __name__ == "__main__":
    main()
