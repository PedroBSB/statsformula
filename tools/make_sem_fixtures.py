"""lavaan reference estimates for the SEM bridges (tests/fixtures/r_sem.json).

Each case's model is first rewritten by statsformula.sem.sem_data (R expressions -> plain
columns); lavaan then fits the rewritten model on the rewritten data, exactly what semopy /
pySEM receive.  Usage: python tools/make_sem_fixtures.py  (needs Rscript with lavaan, jsonlite)
"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from statsformula.sem import sem_data  # noqa: E402
from sem_cases import CASES, datasets  # noqa: E402

OUT = os.path.join(ROOT, "tests", "fixtures", "r_sem.json")
R = r'''
suppressMessages({library(jsonlite); library(lavaan)})
a <- commandArgs(TRUE)
cases <- fromJSON(a[1], simplifyVector = FALSE)
res <- lapply(cases, function(cs) {
  d <- read.csv(cs$csv)
  opts <- if (is.null(cs$options)) list() else cs$options
  fun <- if (cs$preset == "cfa") cfa else sem
  fit <- do.call(fun, c(list(model = cs$model, data = d), opts))
  pe <- parameterEstimates(fit)
  list(lhs = as.list(pe$lhs), op = as.list(pe$op), rhs = as.list(pe$rhs), label = as.list(pe$label),
       est = as.list(pe$est), fmin = as.numeric(fitMeasures(fit, "fmin")), chisq = as.numeric(fitMeasures(fit, "chisq")),
       df = as.numeric(fitMeasures(fit, "df")))
})
writeLines(toJSON(list(results = res, lavaan = as.character(packageVersion("lavaan"))), auto_unbox = TRUE,
                  digits = NA), a[2])
'''


def main():
    D = datasets()
    with tempfile.TemporaryDirectory() as tmp:
        cases = []
        for c in CASES:
            sd = sem_data(c["model"], D[c["data"]])
            csv = os.path.join(tmp, c["name"] + ".csv")
            sd.data.to_csv(csv, index=False, float_format="%.17g")
            cases.append({"model": sd.model, "csv": csv, "preset": c["preset"], "options": c.get("options")})
        cj, oj, rs = (os.path.join(tmp, n) for n in ("cases.json", "out.json", "run.R"))
        json.dump(cases, open(cj, "w"))
        open(rs, "w").write(R)
        subprocess.run(["Rscript", rs, cj, oj], check=True)
        out = json.load(open(oj))
    json.dump({"cases": CASES, "results": out["results"], "lavaan": out["lavaan"]}, open(OUT, "w"), indent=0)
    print("wrote", OUT, len(CASES), "cases; lavaan", out["lavaan"])


if __name__ == "__main__":
    main()
