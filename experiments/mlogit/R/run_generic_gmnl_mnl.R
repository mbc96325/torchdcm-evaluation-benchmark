#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(mlogit))
suppressPackageStartupMessages(library(gmnl))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx)) return(default)
  args[[idx + 1]]
}

data_path <- get_arg("--data")
spec_path <- get_arg("--spec")
result_out <- get_arg("--result-output")

if (is.null(data_path) || is.null(spec_path) || is.null(result_out)) {
  stop("Usage: run_generic_gmnl_mnl.R --data data.csv --spec spec.json --result-output result.json")
}

aligned <- read.csv(data_path)
spec <- fromJSON(spec_path, simplifyVector = FALSE)
parameters <- unlist(spec$parameters)

if ("availability" %in% names(aligned)) {
  aligned <- aligned[as.logical(aligned$availability), ]
}

formula_text <- paste("choice ~ 0 +", paste(parameters, collapse = " + "), "| 0")
mdata <- mlogit.data(aligned, shape = "long", chid.var = "obs_id", alt.var = "alt", choice = "choice")

estimate_start <- proc.time()[["elapsed"]]
model <- gmnl(as.formula(formula_text), data = mdata, model = "mnl", print.level = 0)
estimate_seconds <- proc.time()[["elapsed"]] - estimate_start

covariance_start <- proc.time()[["elapsed"]]
covariance <- vcov(model)
covariance_seconds <- proc.time()[["elapsed"]] - covariance_start

payload <- list(
  backend = "gmnl",
  loglike = as.numeric(logLik(model)),
  estimate_seconds = estimate_seconds,
  covariance_seconds = covariance_seconds,
  total_seconds = estimate_seconds + covariance_seconds,
  params = as.list(coef(model)),
  covariance_names = colnames(covariance),
  covariance = unname(covariance)
)
write(toJSON(payload, auto_unbox = TRUE, digits = 16), result_out)
