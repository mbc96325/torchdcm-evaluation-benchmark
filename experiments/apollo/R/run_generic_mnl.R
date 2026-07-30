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
database <- read.csv(data_path)

apollo_initialise()

apollo_control <- list(
  modelName = spec$model_name,
  modelDescr = "TorchDCM generic MNL benchmark",
  indivID = "obs_id",
  outputDirectory = dirname(output_path)
)

apollo_beta <- unlist(spec$parameters)
apollo_fixed <- c()

apollo_probabilities <- function(apollo_beta, apollo_inputs, functionality = "estimate") {
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))

  P <- list()
  V <- list()
  av <- list()
  alternatives <- list()

  for (alt in spec$alternatives) {
    item <- spec$utility[[alt]]
    utility <- rep(0, length(get(spec$choice_col)))
    if (!is.null(item$asc)) {
      utility <- utility + get(item$asc)
    }
    for (param in names(item$variables)) {
      utility <- utility + get(param) * get(item$variables[[param]])
    }
    V[[alt]] <- utility
    av[[alt]] <- get(item$availability)
    alternatives[[alt]] <- item$code
  }

  mnl_settings <- list(
    alternatives = alternatives,
    avail = av,
    choiceVar = get(spec$choice_col),
    V = V
  )

  P[["model"]] <- apollo_mnl(mnl_settings, functionality)
  P <- apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

apollo_inputs <- apollo_validateInputs()
estimate_start <- proc.time()[["elapsed"]]
model <- apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs)
estimate_seconds <- proc.time()[["elapsed"]] - estimate_start

covariance_start <- proc.time()[["elapsed"]]
covariance <- model$varcov
covariance_seconds <- proc.time()[["elapsed"]] - covariance_start

out <- list(
  backend = "apollo",
  model = "generic_mnl",
  apollo_version = as.character(packageVersion("apollo")),
  loglike = as.numeric(model$maximum),
  estimates = as.list(model$estimate),
  covariance_names = colnames(covariance),
  covariance = unname(covariance),
  timing = list(
    estimate_seconds = estimate_seconds,
    covariance_seconds = covariance_seconds
  ),
  convergence = list(
    status = model$code,
    message = model$message
  )
)

write_json(out, output_path, pretty = TRUE, auto_unbox = TRUE, digits = 16)
