gam_case <- function(cs) {
  d <- getd(cs$data)
  f <- as.formula(cs$formula)
  knots <- if (!is.null(cs$knots)) lapply(cs$knots, unlist) else NULL
  G <- gam(f, data = d, knots = knots, fit = FALSE)
  X <- G$X
  colnames(X) <- G$term.names
  res <- list(fit = mat_info(X))
  res$pen <- lapply(seq_along(G$S), function(i) list(
    name = names(G$sp)[i], off = G$off[i], rank = G$rank[i],
    S = lapply(seq_len(ncol(G$S[[i]])), function(j) as.list(unname(G$S[[i]][, j])))))
  if (!is.null(cs$newdata)) {
    b <- gam(f, data = d, knots = knots)
    Xp <- predict(b, newdata = getd(cs$newdata), type = "lpmatrix")
    res$new <- mat_info(Xp)
  }
  res
}

gampen_case <- function(cs) {
  d <- getd(cs$data)
  sml <- smoothCon(eval(parse(text = cs$smooth)), data = d, absorb.cons = TRUE)
  lapply(sml, function(sm) list(label = sm$label,
                                S = lapply(sm$S, function(S) lapply(seq_len(ncol(S)), function(j) as.list(unname(S[, j]))))))
}

gamfit_case <- function(cs) {
  d <- getd(cs$data)
  f <- as.formula(cs$formula)
  sp <- if (!is.null(cs$sp)) unlist(cs$sp) else NULL
  w <- if (!is.null(cs$weights)) d[[cs$weights]] else NULL
  b <- if (is.null(w)) gam(f, data = d, method = cs$method, sp = sp, gamma = if (is.null(cs$gamma)) 1 else cs$gamma)
       else gam(f, data = d, method = cs$method, sp = sp, weights = w)
  list(sp = as.list(unname(b$sp)), sp_names = as.list(names(b$sp)), coef = as.list(unname(coef(b))),
       fitted = as.list(unname(fitted(b))), edf = sum(b$edf), scale = b$sig2,
       pred = as.list(unname(predict(b, newdata = getd(cs$newdata)))),
       criterion = as.numeric(b$gcv.ubre))
}
