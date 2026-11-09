"""Random model.matrix() cases vs R (tests/fixtures/r_fuzz_mm.json)."""
import json, os, random, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))
import rcases  # noqa

OUT = os.path.join(os.path.dirname(HERE), "tests", "fixtures", "r_fuzz_mm.json")
VARS = ["f", "g", "o", "l", "h", "x", "z", "log(w)", "C(f, sum)", "poly(x, 2)", "I(x > 5)"]


def expr(rng, depth):
    if depth <= 0 or rng.random() < 0.35:
        return rng.choice(VARS)
    op = rng.choice(["+", "+", "*", ":", ":", "/", "%in%", "-", "^"])
    if op == "^":
        return "(" + expr(rng, depth - 1) + " + " + expr(rng, depth - 1) + ")^2"
    return expr(rng, depth - 1) + " " + op + " " + expr(rng, depth - 1)


def main(n=400, seed=3):
    rng = random.Random(seed)
    forms = []
    for _ in range(n):
        f = "y ~ " + expr(rng, rng.randint(1, 3))
        r = rng.random()
        if r < 0.2:
            f += " - 1"
        elif r < 0.3:
            f = f.replace("y ~ ", "y ~ 0 + ")
        forms.append(f)
    work = tempfile.mkdtemp()
    d1 = rcases.make_datasets()["d1"]
    d1.to_csv(os.path.join(work, "d1.csv"), index=False)
    with open(os.path.join(work, "f.json"), "w") as fh:
        json.dump({"formulas": forms, "schema": rcases.schema(d1)}, fh)
    rcode = r'''
    library(jsonlite); wd <- commandArgs(TRUE)[1]; J <- fromJSON(file.path(wd, "f.json"), simplifyVector = FALSE)
    d <- read.csv(file.path(wd, "d1.csv"), stringsAsFactors = FALSE)
    for (nm in names(J$schema)) { s <- J$schema[[nm]]; v <- d[[nm]]
      d[[nm]] <- switch(s$type, factor = factor(v, levels = unlist(s$levels)), ordered = factor(v, levels = unlist(s$levels), ordered = TRUE),
        logical = as.logical(v), integer = as.integer(v), numeric = as.numeric(v), character = as.character(v)) }
    w <- seq_len(nrow(d))
    out <- lapply(J$formulas, function(s) tryCatch({ m <- suppressWarnings(model.matrix(as.formula(s), d))
      list(colnames = as.list(colnames(m)), assign = as.list(attr(m, "assign")),
           m1 = as.list(unname(colSums(m * w))), m2 = as.list(unname(colSums(m^2 * w))))
    }, error = function(e) list(error = conditionMessage(e))))
    writeLines(toJSON(list(formulas = unlist(J$formulas), results = out), auto_unbox = TRUE, digits = NA, null = "null"), file.path(wd, "out.json"))
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
    main(*(int(a) for a in sys.argv[1:]))
