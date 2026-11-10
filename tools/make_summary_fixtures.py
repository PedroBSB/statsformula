"""Generate R reference values for the summary() machinery (tests/fixtures/r_summary.json).

* ``davies``: mgcv's ``psum.chisq`` (Davies' algorithm) on random linear combinations of chi^2
  variables, positive and mixed-sign weights, ``q = 0`` (the form used by summary.gam);
* ``format``: ``format(x, digits = 4)`` and ``format.pval(x, digits = 3)`` of assorted vectors;
* ``coefmat``: ``capture.output(printCoefmat(...))`` of a few coefficient tables.

Usage: python tools/make_summary_fixtures.py   (needs Rscript with mgcv + jsonlite)
"""
import json
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "tests", "fixtures", "r_summary.json")

R = r'''
suppressMessages({library(jsonlite); library(mgcv)})
set.seed(3)
dav <- list()
for (i in 1:60) {
  r <- sample(1:7, 1)
  lb <- round(rexp(r), 3)
  if (i %% 3 == 0) lb[r] <- -abs(lb[r]) * runif(1, 0.05, 2)
  df <- sample(1:4, r, TRUE)
  q <- if (i %% 4 == 0) 0 else round(runif(1, 0, 3 * sum(abs(lb) * df)), 3)
  dav[[i]] <- list(lb = lb, df = df, q = q, p = psum.chisq(q, lb, df))
}
vals <- list(c(1.23456789, 0.000123, 12345.6789), c(0.1, 0.25, 100), c(9996, 1), c(1e-10, 2),
             c(-0.5, 3.14159), c(123456789, 1), 0.0001234, c(1e5, 2e5), c(0.000012, 0.00034), 2.2e-16,
             c(0.99999, 1e-20), c(-1.5e-7, 3), c(0.05, 0.049999), c(1e-300, 1), c(1e15, 1.5), c(0, 0),
             c(-0, 12.5), c(99999, 0.00001), c(0.12345, 123.45), c(7, 7.25, -7.125))
fmt <- lapply(vals, function(v) list(x = v, format = format(v, digits = 4), pval = format.pval(abs(v), digits = 3)))
mk <- function(m, rn, cn) { m <- matrix(m, nrow = length(rn), byrow = TRUE, dimnames = list(rn, cn)); m }
cm <- list(
  list(x = c(0.478132, 0.044191, 10.82, 1e-20, -1.2345, 0.5, -2.469, 0.0142, 12.3, 0.0012, 10250, 0.3),
       rn = c("(Intercept)", "x", "factor(g)b"), cn = c("Estimate", "Std. Error", "t value", "Pr(>|t|)"), cs = NULL),
  list(x = c(5.627, 6.791, 52.66, 1e-30, 1.0001, 1.0002, 0.123, 0.726),
       rn = c("s(x)", "s(z)"), cn = c("edf", "Ref.df", "F", "p-value"), cs = 1),
  list(x = c(123.4, 56.7, 2.176, 0.0312, -0.00012, 0.00034, -0.353, 0.7245),
       rn = c("a", "b"), cn = c("Estimate", "Std. Error", "z value", "Pr(>|z|)"), cs = NULL),
  list(x = c(1e-8, 2e-9, 5, 6e-7, 3, 1, 3, 0.002), rn = c("tiny", "big"),
       cn = c("Estimate", "Std. Error", "t value", "Pr(>|t|)"), cs = NULL))
coefmat <- lapply(cm, function(c) {
  m <- mk(c$x, c$rn, c$cn)
  out <- if (is.null(c$cs)) capture.output(printCoefmat(m, digits = 4))
         else capture.output(printCoefmat(m, digits = 4, has.Pvalue = TRUE, cs.ind = c$cs))
  list(x = c$x, rn = c$rn, cn = c$cn, cs = if (is.null(c$cs)) NULL else c$cs - 1, printed = out)
})
writeLines(toJSON(list(davies = dav, format = fmt, coefmat = coefmat, R = R.version.string,
                       mgcv = as.character(packageVersion("mgcv"))),
                  auto_unbox = FALSE, digits = NA, null = "null"), commandArgs(TRUE)[1])
'''


def main():
    with tempfile.TemporaryDirectory() as tmp:
        script, out = os.path.join(tmp, "run.R"), os.path.join(tmp, "out.json")
        open(script, "w").write(R)
        subprocess.run(["Rscript", script, out], check=True)
        res = json.load(open(out))
    json.dump(res, open(OUT, "w"), indent=0)
    print("wrote", OUT, res["R"][0], "mgcv", res["mgcv"][0])


if __name__ == "__main__":
    main()
