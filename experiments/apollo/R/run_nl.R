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
  modelDescr = "TorchDCM NL benchmark",
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
    if (!is.null(item$variables)) {
      for (param_name in names(item$variables)) {
        utility <- utility + get(param_name) * get(item$variables[[param_name]])
      }
    } else {
      utility <- utility + get("B_TIME") * get(item$time) + get("B_COST") * get(item$cost)
    }
    V[[alt]] <- utility
    av[[alt]] <- get(item$availability)
    alternatives[[alt]] <- item$code
  }

  nlNests <- list(root = 1)
  nlStructure <- list(root = c())
  for (nest_name in names(spec$nests)) {
    nest <- spec$nests[[nest_name]]
    nlStructure[["root"]] <- c(nlStructure[["root"]], nest_name)
    nlStructure[[nest_name]] <- unlist(nest$alternatives)
    if (is.null(nest$lambda_param)) {
      nlNests[[nest_name]] <- nest$lambda_value
    } else if (!is.null(nest$lambda_raw_param)) {
      raw_lambda <- get(nest$lambda_raw_param)
      lambda_min <- nest$lambda_min
      nlNests[[nest_name]] <- lambda_min + (1 - lambda_min) / (1 + exp(-raw_lambda))
    } else {
      nlNests[[nest_name]] <- get(nest$lambda_param)
    }
  }

  nl_settings <- list(
    alternatives = alternatives,
    avail = av,
    choiceVar = get(spec$choice_col),
    V = V,
    nlNests = nlNests,
    nlStructure = nlStructure
  )

  P[["model"]] <- apollo_nl(nl_settings, functionality)
  P <- apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

apollo_inputs <- apollo_validateInputs()
estimate_start <- proc.time()[["elapsed"]]
model <- apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs)
estimate_seconds <- proc.time()[["elapsed"]] - estimate_start

covariance_start <- proc.time()[["elapsed"]]
estimates <- model$estimate
covariance <- model$varcov
robust_covariance <- model$robvarcov
if (!is.null(spec$nests)) {
  for (nest_name in names(spec$nests)) {
    nest <- spec$nests[[nest_name]]
    if (!is.null(nest$lambda_raw_param)) {
      raw_name <- nest$lambda_raw_param
      lambda_name <- nest$lambda_param
      raw_value <- estimates[[raw_name]]
      sigmoid <- 1 / (1 + exp(-raw_value))
      lambda_value <- nest$lambda_min + (1 - nest$lambda_min) * sigmoid
      deriv <- (1 - nest$lambda_min) * sigmoid * (1 - sigmoid)
      index <- match(raw_name, names(estimates))
      estimates[[index]] <- lambda_value
      names(estimates)[[index]] <- lambda_name
      if (!is.null(covariance)) {
        jac <- diag(length(model$estimate))
        jac[index, index] <- deriv
        covariance <- jac %*% covariance %*% t(jac)
        rownames(covariance) <- names(estimates)
        colnames(covariance) <- names(estimates)
      }
      if (!is.null(robust_covariance)) {
        robust_jac <- diag(length(model$estimate))
        robust_jac[index, index] <- deriv
        robust_covariance <- robust_jac %*% robust_covariance %*% t(robust_jac)
        rownames(robust_covariance) <- names(estimates)
        colnames(robust_covariance) <- names(estimates)
      }
    }
  }
}
covariance_seconds <- proc.time()[["elapsed"]] - covariance_start

out <- list(
  backend = "apollo",
  model = "nl",
  apollo_version = as.character(packageVersion("apollo")),
  loglike = as.numeric(model$maximum),
  estimates = as.list(estimates),
  se = as.list(model$se),
  robust_se = as.list(model$robse),
  covariance = unname(covariance),
  covariance_names = names(estimates),
  robust_covariance = unname(robust_covariance),
  robust_covariance_names = names(estimates),
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
