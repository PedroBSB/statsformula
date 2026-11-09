gam_case <- function(cs) {
  d <- getd(cs$data)
  f <- as.formula(cs$formula)
  knots <- if (!is.null(cs$knots)) lapply(cs$knots, unlist) else NULL
  G <- gam(f, data = d, knots = knots, fit = FALSE)
  X <- G$X
  colnames(X) <- G$term.names
  res <- list(fit = mat_info(X))
  if (!is.null(cs$newdata)) {
    b <- gam(f, data = d, knots = knots)
    Xp <- predict(b, newdata = getd(cs$newdata), type = "lpmatrix")
    res$new <- mat_info(Xp)
  }
  res
}
