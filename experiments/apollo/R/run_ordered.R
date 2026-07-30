#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) {
    stop(paste("Missing required argument", flag))
  }
  args[[index + 1]]
}

data_path <- get_arg("--data")
spec_path <- get_arg("--spec")
output_path <- get_arg("--output")

suppressPackageStartupMessages(library(apollo))
suppressPackageStartupMessages(library(jsonlite))

spec <- fromJSON(spec_path, simplifyVector = FALSE)
database <- read.csv(data_path, check.names = FALSE)

apollo_initialise()

apollo_control <- list(
  modelName = spec$model_name,
  modelDescr = paste("TorchDCM", spec$kind, "ordered-response benchmark"),
  indivID = spec$id_col,
  weights = spec$weight_col,
  nCores = 1,
  noDiagnostics = TRUE,
  outputDirectory = dirname(output_path)
)

apollo_beta <- unlist(spec$parameters)
apollo_fixed <- c()

parameter_names <- names(spec$variables)
variable_names <- unlist(spec$variables, use.names = FALSE)
threshold_names <- unlist(spec$thresholds, use.names = FALSE)
identifiers <- c(
  parameter_names,
  variable_names,
  threshold_names,
  spec$outcome_col
)
if (any(!grepl("^[A-Za-z][A-Za-z0-9_.]*$", identifiers))) {
  stop("Ordered benchmark specification contains an invalid R identifier")
}
if (!(spec$kind %in% c("logit", "probit"))) {
  stop(paste("Unsupported ordered model kind:", spec$kind))
}

utility_expression <- paste(
  paste(parameter_names, variable_names, sep = " * "),
  collapse = " + "
)
threshold_expression <- paste(threshold_names, collapse = ", ")
category_expression <- paste(unlist(spec$categories), collapse = ", ")
apollo_component <- if (spec$kind == "logit") "apollo_ol" else "apollo_op"

probability_source <- sprintf(
  paste0(
    "function(apollo_beta, apollo_inputs, functionality = 'estimate') {",
    "apollo_attach(apollo_beta, apollo_inputs);",
    "on.exit(apollo_detach(apollo_beta, apollo_inputs));",
    "P <- list();",
    "ordered_settings <- list(",
    "outcomeOrdered = %s, utility = %s, tau = list(%s), coding = c(%s));",
    "P[['model']] <- %s(ordered_settings, functionality);",
    "P <- apollo_weighting(P, apollo_inputs, functionality);",
    "P <- apollo_prepareProb(P, apollo_inputs, functionality);",
    "return(P)",
    "}"
  ),
  spec$outcome_col,
  utility_expression,
  threshold_expression,
  category_expression,
  apollo_component
)
apollo_probabilities <- eval(parse(text = probability_source))

apollo_inputs <- apollo_validateInputs(silent = TRUE)
estimate_settings <- list(
  maxIterations = spec$max_iter,
  writeIter = FALSE,
  silent = TRUE
)

estimate_start <- proc.time()[["elapsed"]]
model <- apollo_estimate(
  apollo_beta,
  apollo_fixed,
  apollo_probabilities,
  apollo_inputs,
  estimate_settings = estimate_settings
)
estimate_seconds <- proc.time()[["elapsed"]] - estimate_start

covariance_start <- proc.time()[["elapsed"]]
covariance <- unname(model$varcov)
covariance_seconds <- proc.time()[["elapsed"]] - covariance_start

out <- list(
  backend = "apollo",
  model = paste0("ordered_", spec$kind),
  apollo_version = as.character(packageVersion("apollo")),
  loglike = as.numeric(model$maximum),
  estimates = as.list(model$estimate),
  covariance = covariance,
  covariance_names = names(model$estimate),
  timing = list(
    estimate_seconds = estimate_seconds,
    covariance_seconds = covariance_seconds
  ),
  convergence = list(
    status = model$code,
    message = model$message
  )
)

write_json(out, output_path, pretty = TRUE, auto_unbox = TRUE, digits = NA)
