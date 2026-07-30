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
options(expressions = 500000)

spec <- fromJSON(spec_path, simplifyVector = FALSE)
database <- read.csv(data_path, check.names = FALSE)
kind <- spec$kind
shared_draws <- as.numeric(unlist(spec$draws))
n_draws <- length(shared_draws)
positive_parameters <- as.character(unlist(spec$positive_parameters))
n_database_rows <- nrow(database)
has_four_alternatives <- "x_D" %in% names(database)

softplus <- function(value) {
  ifelse(value > 20, value, log1p(exp(value)))
}

inverse_softplus <- function(value) {
  ifelse(value > 20, value, log(expm1(value)))
}

apollo_initialise()

apollo_control <- list(
  modelName = spec$model_name,
  modelDescr = paste("TorchDCM aligned full estimation for", kind),
  indivID = spec$id_col,
  mixing = kind %in% c("hybrid_choice", "panel_likelihood"),
  nCores = 1,
  analyticGrad = FALSE,
  noDiagnostics = TRUE,
  outputDirectory = dirname(output_path)
)

natural_starts <- unlist(spec$parameters)
apollo_beta <- natural_starts
if (length(positive_parameters) > 0) {
  for (name in positive_parameters) {
    phi_name <- paste0("PHI_", name)
    apollo_beta[[phi_name]] <- inverse_softplus(apollo_beta[[name]])
    apollo_beta <- apollo_beta[names(apollo_beta) != name]
  }
}
apollo_fixed <- c()

positive_value <- function(name) {
  if (name %in% positive_parameters) {
    return(softplus(get(paste0("PHI_", name))))
  }
  get(name)
}

if (apollo_control$mixing) {
  apollo_draws <- list(
    interDrawsType = "halton",
    interNDraws = n_draws,
    interUnifDraws = c(),
    interNormDraws = c("unused_draw"),
    intraDrawsType = "",
    intraNDraws = 0,
    intraUnifDraws = c(),
    intraNormDraws = c()
  )

  apollo_randCoeff <- function(apollo_beta, apollo_inputs) {
    randcoeff <- list()
    eta <- matrix(
      rep(shared_draws, each = n_database_rows),
      nrow = n_database_rows,
      ncol = n_draws
    )
    if (kind == "hybrid_choice") {
      randcoeff[["LV"]] <- G_Q * q + softplus(PHI_SIGMA_LV) * eta
    } else {
      randcoeff[["B_X_RND"]] <- B_X + softplus(PHI_SIGMA_B_X) * eta
    }
    randcoeff
  }
}

apollo_probabilities <- function(apollo_beta, apollo_inputs, functionality = "estimate") {
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  P <- list()

  if (kind == "latent_class") {
    V1 <- list(
      A = B_X_C1 * x_A,
      B = ASC_B_C1 + B_X_C1 * x_B,
      C = ASC_C_C1 + B_X_C1 * x_C
    )
    V2 <- list(
      A = B_X_C2 * x_A,
      B = ASC_B_C2 + B_X_C2 * x_B,
      C = ASC_C_C2 + B_X_C2 * x_C
    )
    common <- list(
      alternatives = c(A = 1, B = 2, C = 3),
      avail = list(A = av_A, B = av_B, C = av_C),
      choiceVar = choice_code
    )
    P[["class_1"]] <- apollo_mnl(
      c(common, list(utilities = V1, componentName = "class_1")),
      functionality
    )
    P[["class_2"]] <- apollo_mnl(
      c(common, list(utilities = V2, componentName = "class_2")),
      functionality
    )
    P[["allocation"]] <- apollo_classAlloc(
      list(
        alternatives = c(class_1 = 1, class_2 = 2),
        avail = 1,
        choiceVar = NA,
        utilities = list(
          class_1 = 0,
          class_2 = CLASS_2 + CLASS_2_Z * z
        ),
        componentName = "allocation"
      )
    )
    P[["model"]] <- P[["allocation"]][["class_1"]] * P[["class_1"]] +
      P[["allocation"]][["class_2"]] * P[["class_2"]]
    P[["class_1"]] <- NULL
    P[["class_2"]] <- NULL
    P[["allocation"]] <- NULL
  }

  if (kind == "hybrid_choice") {
    sigma_y1 <- softplus(PHI_SIGMA_Y1)
    sigma_y2 <- softplus(PHI_SIGMA_Y2)
    P[["indicator_1"]] <- apollo_normalDensity(
      list(
        outcomeNormal = y1,
        xNormal = LV,
        mu = 0,
        sigma = sigma_y1,
        componentName = "indicator_1"
      ),
      functionality
    )
    P[["indicator_2"]] <- apollo_normalDensity(
      list(
        outcomeNormal = y2,
        xNormal = A2 + L2 * LV,
        mu = 0,
        sigma = sigma_y2,
        componentName = "indicator_2"
      ),
      functionality
    )
    P[["choice"]] <- apollo_mnl(
      list(
        alternatives = c(A = 1, B = 2),
        avail = list(A = 1, B = 1),
        choiceVar = choice_code,
        utilities = list(A = 0, B = ASC_B + B_X * x + B_ATT * LV),
        componentName = "choice"
      ),
      functionality
    )
    P <- apollo_combineModels(P, apollo_inputs, functionality)
    P <- apollo_avgInterDraws(P, apollo_inputs, functionality)
  }

  if (kind == "panel_likelihood") {
    V <- list(
      A = B_X_RND * x_A,
      B = ASC_B + B_X_RND * x_B,
      C = ASC_C + B_X_RND * x_C
    )
    alternatives <- c(A = 1, B = 2, C = 3)
    availability <- list(A = 1, B = 1, C = 1)
    if (has_four_alternatives) {
      V[["D"]] <- ASC_D + B_X_RND * x_D
      alternatives <- c(alternatives, D = 4)
      availability[["D"]] <- 1
    }
    P[["model"]] <- apollo_mnl(
      list(
        alternatives = alternatives,
        avail = availability,
        choiceVar = choice_code,
        utilities = V,
        componentName = "choice"
      ),
      functionality
    )
    P <- apollo_panelProd(P, apollo_inputs, functionality)
    P <- apollo_avgInterDraws(P, apollo_inputs, functionality)
  }

  P <- apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

apollo_inputs <- apollo_validateInputs(silent = TRUE)
estimate_settings <- list(
  maxIterations = as.integer(spec$max_iter),
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
covariance <- model$varcov
covariance_available <- !is.null(covariance) && all(is.finite(covariance))
covariance_seconds <- proc.time()[["elapsed"]] - covariance_start

estimates <- model$estimate
if (length(positive_parameters) > 0) {
  for (name in positive_parameters) {
    phi_name <- paste0("PHI_", name)
    estimates[[name]] <- softplus(estimates[[phi_name]])
    estimates <- estimates[names(estimates) != phi_name]
  }
}
estimates <- estimates[names(natural_starts)]

out <- list(
  backend = "apollo",
  model = kind,
  apollo_version = as.character(packageVersion("apollo")),
  loglike = as.numeric(model$maximum),
  estimates = as.list(estimates),
  covariance_available = covariance_available,
  timing = list(
    estimate_seconds = estimate_seconds,
    covariance_seconds = covariance_seconds,
    total_seconds = estimate_seconds + covariance_seconds
  ),
  convergence = list(
    status = model$code,
    message = model$message,
    successful_estimation = tryCatch(
      as.logical(model$successfulEstimation),
      error = function(e) NULL
    ),
    iterations = tryCatch(as.integer(model$iterations), error = function(e) NULL)
  )
)

write_json(out, output_path, pretty = TRUE, auto_unbox = TRUE, digits = 16)
