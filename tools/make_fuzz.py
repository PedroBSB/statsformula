"""Generate random formulas and their R ``terms()`` (tests/fixtures/r_fuzz.json)."""
import json, os, random, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "tests", "fixtures", "r_fuzz.json")
VARS = ["a", "b", "c", "d", "log(e)", "I(f^2)", "poly(g, 2)", "offset(h)"]


def expr(rng, depth):
    if depth <= 0 or rng.random() < 0.3:
        r = rng.random()
        if r < 0.06:
            return rng.choice(["0", "1"])
        return rng.choice(VARS[:-1] if rng.random() < 0.95 else VARS)
    op = rng.choice(["+", "+", "*", ":", "/", "%in%", "-", "^", "()", "-u"])
    if op == "()":
        return "(" + expr(rng, depth - 1) + ")"
    if op == "-u":
        return "-" + expr(rng, depth - 1)
    if op == "^":
        return "(" + expr(rng, depth - 1) + ")^" + str(rng.choice([2, 3]))
    return expr(rng, depth - 1) + " " + op + " " + expr(rng, depth - 1)


def main(n=500, seed=7):
    rng = random.Random(seed)
    forms = []
    for i in range(n):
        lhs = rng.choice(["y ", "", "log(y) ", "cbind(y, z) "])
        forms.append(lhs + "~ " + expr(rng, rng.randint(1, 4)))
    work = tempfile.mkdtemp()
    with open(os.path.join(work, "f.json"), "w") as fh:
        json.dump(forms, fh)
    rcode = r'''
    library(jsonlite); f <- fromJSON(file.path(commandArgs(TRUE)[1], "f.json"))
    dep <- function(x) paste(deparse(x, width.cutoff = 500L), collapse = " ")
    out <- lapply(f, function(s) tryCatch({ tt <- suppressWarnings(terms(as.formula(s), keep.order = FALSE)); fac <- attr(tt, "factors")
      list(formula = dep(as.formula(s)), labels = as.list(attr(tt, "term.labels")), intercept = attr(tt, "intercept"),
           response = attr(tt, "response"), order = as.list(attr(tt, "order")),
           variables = as.list(vapply(as.list(attr(tt, "variables"))[-1], dep, "")),
           offset = if (is.null(attr(tt, "offset"))) NULL else as.list(attr(tt, "offset")),
           factors = if (length(fac)) lapply(seq_len(ncol(fac)), function(j) as.list(unname(fac[, j]))) else list())
    }, error = function(e) list(error = conditionMessage(e))))
    writeLines(toJSON(list(formulas = f, results = out), auto_unbox = TRUE, null = "null"), file.path(commandArgs(TRUE)[1], "out.json"))
    '''
    with open(os.path.join(work, "run.R"), "w") as fh:
        fh.write(rcode)
    subprocess.run(["Rscript", os.path.join(work, "run.R"), work], check=True)
    with open(os.path.join(work, "out.json")) as fh:
        data = json.load(fh)
    with open(OUT, "w") as fh:
        json.dump(data, fh)
    print("wrote", OUT, len(forms))


if __name__ == "__main__":
    main()
