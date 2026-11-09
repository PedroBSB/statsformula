# Compute R reference results for the statsformula test-suite.
# Usage: Rscript r_reference.R <workdir>   (workdir holds cases.json, schema.json, <data>.csv)
suppressPackageStartupMessages({
  library(jsonlite); library(splines); library(Formula); library(Matrix); library(mgcv)
})
args <- commandArgs(trailingOnly = TRUE)
wd <- args[[1]]
cases <- fromJSON(file.path(wd, "cases.json"), simplifyVector = FALSE)
schema <- fromJSON(file.path(wd, "schema.json"), simplifyVector = FALSE)

load_data <- function(name) {
  sc <- schema[[name]]
  d <- read.csv(file.path(wd, paste0(name, ".csv")), check.names = FALSE, stringsAsFactors = FALSE,
                na.strings = "NA")
  for (nm in names(sc)) {
    s <- sc[[nm]]
    v <- d[[nm]]
    d[[nm]] <- switch(s$type,
      factor = factor(v, levels = unlist(s$levels)),
      ordered = factor(v, levels = unlist(s$levels), ordered = TRUE),
      logical = as.logical(v),
      integer = as.integer(v),
      numeric = as.numeric(v),
      character = as.character(v))
  }
  d
}
datasets <- list()
getd <- function(name) { if (is.null(datasets[[name]])) datasets[[name]] <<- load_data(name); datasets[[name]] }

dep <- function(x) paste(deparse(x, width.cutoff = 500L), collapse = " ")

terms_info <- function(tt) {
  fac <- attr(tt, "factors")
  list(
    formula = dep(formula(tt)),
    term_labels = as.list(attr(tt, "term.labels")),
    order = as.list(attr(tt, "order")),
    intercept = attr(tt, "intercept"),
    response = attr(tt, "response"),
    variables = as.list(vapply(as.list(attr(tt, "variables"))[-1], dep, "")),
    factors = if (length(fac)) lapply(seq_len(ncol(fac)), function(j) as.list(unname(fac[, j]))) else list(),
    offset = if (is.null(attr(tt, "offset"))) NULL else as.list(attr(tt, "offset")),
    specials = if (is.null(attr(tt, "specials"))) NULL else lapply(attr(tt, "specials"), function(s) if (is.null(s)) NULL else as.list(s))
  )
}

contr_info <- function(cl) {
  if (is.null(cl)) return(NULL)
  lapply(cl, function(v) if (is.character(v)) unname(v) else list(matrix = lapply(seq_len(ncol(v)), function(j) as.list(unname(v[, j]))),
                                                            colnames = if (is.null(colnames(v))) NULL else as.list(colnames(v))))
}

mat_info <- function(m) {
  m <- as.matrix(m)
  list(colnames = as.list(colnames(m)), nrow = nrow(m),
       rownames = as.list(rownames(m)),
       columns = lapply(seq_len(ncol(m)), function(j) as.list(unname(m[, j]))))
}

col_info <- function(v) {
  if (is.matrix(v)) return(list(type = "matrix", colnames = if (is.null(colnames(v))) NULL else as.list(colnames(v)),
                                columns = lapply(seq_len(ncol(v)), function(j) as.list(unname(as.numeric(v[, j]))))))
  if (is.factor(v)) return(list(type = if (is.ordered(v)) "ordered" else "factor", levels = as.list(levels(v)),
                                values = as.list(as.character(v))))
  if (is.logical(v)) return(list(type = "logical", values = as.list(v)))
  if (is.character(v)) return(list(type = "character", values = as.list(v)))
  list(type = "numeric", values = as.list(as.numeric(v)))
}

mf_info <- function(mf) {
  list(names = as.list(names(mf)), rownames = as.list(rownames(mf)),
       columns = lapply(mf, col_info),
       terms = terms_info(attr(mf, "terms")),
       dataClasses = as.list(attr(attr(mf, "terms"), "dataClasses")),
       na_action = if (is.null(attr(mf, "na.action"))) NULL else list(class = class(attr(mf, "na.action")), rows = as.list(names(attr(mf, "na.action")))))
}

run_case <- function(cs) {
  k <- cs$kind
  if (!is.null(cs$options_contrasts)) { old <- options(contrasts = unlist(cs$options_contrasts)); on.exit(options(old)) }
  if (k == "terms") {
    f <- as.formula(cs$formula)
    tt <- terms(f, data = if (!is.null(cs$data)) getd(cs$data) else NULL,
                keep.order = isTRUE(cs$keep_order), specials = unlist(cs$specials))
    return(terms_info(tt))
  }
  if (k == "mm" || k == "sparse") {
    d <- getd(cs$data)
    ca <- NULL
    if (!is.null(cs$contrasts)) ca <- lapply(cs$contrasts, identity)
    if (!is.null(cs$contrasts_matrix)) ca <- lapply(cs$contrasts_matrix, function(m) do.call(rbind, lapply(m, unlist)))
    f <- as.formula(cs$formula)
    if (k == "sparse") {
      m <- sparse.model.matrix(f, d)
      return(c(mat_info(m), list(assign = as.list(attr(m, "assign")))))
    }
    m <- if (is.null(ca)) model.matrix(f, d) else model.matrix(f, d, contrasts.arg = ca)
    return(c(mat_info(m), list(assign = as.list(attr(m, "assign")), contrasts = contr_info(attr(m, "contrasts")))))
  }
  if (k == "update") return(list(formula = dep(update(as.formula(cs$old), as.formula(cs$new)))))
  if (k == "reformulate") {
    f <- reformulate(if (length(cs$labels)) unlist(cs$labels) else character(0), response = cs$response, intercept = cs$intercept)
    return(list(formula = dep(f)))
  }
  if (k == "drop_terms") {
    tt <- drop.terms(terms(as.formula(cs$formula)), unlist(cs$dropx), keep.response = cs$keep_response)
    return(terms_info(tt))
  }
  if (k == "delete_response") return(terms_info(delete.response(terms(as.formula(cs$formula), specials = unlist(cs$specials)))))
  if (k == "simplify") return(list(formula = dep(formula(terms(as.formula(cs$formula), simplify = TRUE)))))
  if (k == "allvars") { f <- as.formula(cs$formula); return(list(vars = as.list(all.vars(f)), names = as.list(all.names(f)))) }
  if (k == "Formula") {
    F <- Formula(as.formula(cs$formula)); d <- getd(cs$data)
    len <- length(F)
    mf <- model.frame(F, d)
    res <- list(print = dep(formula(F)), length = as.list(len), collapse = dep(formula(F, collapse = TRUE)),
                allvars = as.list(all.vars(formula(F))), mf_names = as.list(names(mf)),
                terms = terms_info(terms(F, data = d)), parts = list(), lhs = list())
    for (i in seq_len(len[2])) {
      mm <- model.matrix(F, mf, rhs = i)
      res$parts[[i]] <- c(list(formula = dep(formula(F, lhs = 0, rhs = i)), formula_l1 = dep(formula(F, lhs = 1, rhs = i))),
                          mat_info(mm), list(assign = as.list(attr(mm, "assign"))))
    }
    for (i in seq_len(len[1])) {
      mp <- model.part(F, mf, lhs = i)
      res$lhs[[i]] <- list(names = as.list(names(mp)), columns = lapply(mp, col_info))
    }
    return(res)
  }
  if (k == "Formula_update") return(list(formula = dep(formula(update(Formula(as.formula(cs$old)), as.formula(cs$new))))))
  if (k == "Formula_combine") {
    F <- do.call(as.Formula, lapply(cs$parts, as.formula))
    return(list(formula = dep(formula(F)), length = as.list(length(F))))
  }
  if (k == "predict") {
    d <- getd(cs$data); nd <- getd(cs$newdata)
    f <- as.formula(cs$formula)
    mf <- model.frame(f, d)
    X <- model.matrix(attr(mf, "terms"), mf)
    tt <- delete.response(attr(mf, "terms"))
    mf2 <- model.frame(tt, nd, xlev = .getXlevels(attr(mf, "terms"), mf))
    X2 <- model.matrix(tt, mf2, contrasts.arg = attr(X, "contrasts"))
    return(list(fit = mat_info(X), new = mat_info(X2)))
  }
  if (k == "mf") {
    d <- getd(cs$data)
    f <- as.formula(cs$formula)
    a <- list(formula = f, data = d)
    if (!is.null(cs$na_action)) a$na.action <- get(cs$na_action)
    if (!is.null(cs$weights)) a$weights <- d[[cs$weights]]
    if (!is.null(cs$offset)) a$offset <- d[[cs$offset]]
    if (!is.null(cs$subset)) a$subset <- eval(parse(text = cs$subset), d)
    if (isTRUE(cs$drop_unused_levels)) a$drop.unused.levels <- TRUE
    mf <- do.call(model.frame, a)
    res <- mf_info(mf)
    res$response <- if (attr(attr(mf, "terms"), "response")) col_info(model.response(mf)) else NULL
    off <- model.offset(mf); res$offset <- if (is.null(off)) NULL else as.list(off)
    w <- model.weights(mf); res$weights <- if (is.null(w)) NULL else as.list(w)
    return(res)
  }
  if (k == "gam") return(gam_case(cs))
  stop("unknown case kind ", k)
}

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "r_reference_mgcv.R"))

out <- lapply(cases, function(cs) {
  warns <- character(0)
  res <- tryCatch(withCallingHandlers(run_case(cs), warning = function(w) {
    warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }),
    error = function(e) list(error = conditionMessage(e)))
  res$warnings <- as.list(warns)
  res
})
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, null = "null", na = "null"), file.path(wd, "r_reference.json"))
cat("wrote", length(out), "cases\n")
