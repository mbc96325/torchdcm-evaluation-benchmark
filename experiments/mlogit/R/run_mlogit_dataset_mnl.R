suppressPackageStartupMessages(library(mlogit))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx)) return(default)
  args[[idx + 1]]
}
has_flag <- function(flag) {
  flag %in% args
}

dataset <- get_arg("--dataset")
data_input <- get_arg("--data-input")
data_out <- get_arg("--data-output")
result_out <- get_arg("--result-output")

if (is.null(dataset) || is.null(data_input) || is.null(data_out) || is.null(result_out)) {
  stop("Usage: run_mlogit_dataset_mnl.R --dataset DATASET --data-input source.csv --data-output data.csv --result-output result.json")
}

source_data <- read.csv(data_input, check.names = FALSE, stringsAsFactors = FALSE)

wide_to_long <- function(df, alternatives, choice_col, variables, column_fun) {
  rows <- list()
  cursor <- 1
  for (i in seq_len(nrow(df))) {
    chosen <- as.character(df[[choice_col]][[i]])
    for (alt in alternatives) {
      row <- data.frame(obs_id = i, alt = as.character(alt), choice = chosen == as.character(alt))
      for (var in variables) {
        row[[var]] <- df[[column_fun(var, alt)]][[i]]
      }
      rows[[cursor]] <- row
      cursor <- cursor + 1
    }
  }
  do.call(rbind, rows)
}

fit_long <- function(aligned, variables, availability_col = NULL) {
  if (!is.null(availability_col)) {
    aligned <- aligned[aligned[[availability_col]], ]
  }
  formula_text <- paste("choice ~ 0 +", paste(variables, collapse = " + "), "| 0")
  mdata <- mlogit.data(aligned, shape = "long", chid.var = "obs_id", alt.var = "alt", choice = "choice")
  estimate_start <- proc.time()[["elapsed"]]
  model <- mlogit(as.formula(formula_text), mdata, print.level = 0)
  estimate_seconds <- proc.time()[["elapsed"]] - estimate_start
  covariance_start <- proc.time()[["elapsed"]]
  covariance <- vcov(model)
  covariance_seconds <- proc.time()[["elapsed"]] - covariance_start
  list(model = model, estimate_seconds = estimate_seconds, covariance_seconds = covariance_seconds, covariance = covariance)
}

make_case <- function(dataset) {
  if (dataset == "catsup") {
    Catsup <- source_data
    alternatives <- c("heinz41", "heinz32", "heinz28", "hunts32")
    variables <- c("disp", "feat", "price")
    aligned <- wide_to_long(Catsup, alternatives, "choice", variables, function(var, alt) paste0(var, ".", alt))
    return(list(aligned = aligned, variables = variables))
  }
  if (dataset == "cracker") {
    Cracker <- source_data
    alternatives <- c("sunshine", "kleebler", "nabisco", "private")
    variables <- c("disp", "feat", "price")
    aligned <- wide_to_long(Cracker, alternatives, "choice", variables, function(var, alt) paste0(var, ".", alt))
    return(list(aligned = aligned, variables = variables))
  }
  if (dataset == "electricity") {
    Electricity <- source_data
    alternatives <- as.character(1:4)
    variables <- c("pf", "cl", "loc", "wk", "tod", "seas")
    aligned <- wide_to_long(Electricity, alternatives, "choice", variables, function(var, alt) paste0(var, alt))
    return(list(aligned = aligned, variables = variables))
  }
  if (dataset == "fishing") {
    Fishing <- source_data
    alternatives <- c("beach", "pier", "boat", "charter")
    variables <- c("price", "catch")
    aligned <- wide_to_long(Fishing, alternatives, "mode", variables, function(var, alt) paste0(var, ".", alt))
    return(list(aligned = aligned, variables = variables))
  }
  if (dataset == "hc") {
    HC <- source_data
    alternatives <- c("gcc", "ecc", "erc", "hpc", "gc", "ec", "er")
    variables <- c("ich", "och")
    aligned <- wide_to_long(HC, alternatives, "depvar", variables, function(var, alt) paste0(var, ".", alt))
    return(list(aligned = aligned, variables = variables))
  }
  if (dataset == "heating") {
    Heating <- source_data
    alternatives <- c("gc", "gr", "ec", "er", "hp")
    variables <- c("ic", "oc")
    aligned <- wide_to_long(Heating, alternatives, "depvar", variables, function(var, alt) paste0(var, ".", alt))
    return(list(aligned = aligned, variables = variables))
  }
  if (dataset == "mode") {
    Mode <- source_data
    alternatives <- c("car", "carpool", "bus", "rail")
    variables <- c("cost", "time")
    aligned <- wide_to_long(Mode, alternatives, "choice", variables, function(var, alt) paste0(var, ".", alt))
    return(list(aligned = aligned, variables = variables))
  }
  if (dataset == "modecanada") {
    ModeCanada <- source_data
    variables <- c("cost", "ivt", "ovt", "freq")
    aligned <- data.frame(obs_id = ModeCanada$case, alt = ModeCanada$alt, choice = as.logical(ModeCanada$choice))
    for (var in variables) aligned[[var]] <- ModeCanada[[var]]
    return(list(aligned = aligned, variables = variables))
  }
  if (dataset == "nox") {
    NOx <- source_data
    variables <- c("post", "vcost", "kcost")
    aligned <- data.frame(obs_id = NOx$chid, alt = NOx$alt, choice = as.logical(NOx$choice), availability = as.logical(NOx$available))
    for (var in variables) aligned[[var]] <- NOx[[var]]
    return(list(aligned = aligned, variables = variables, availability_col = "availability"))
  }
  if (dataset == "risky_transport") {
    RiskyTransport <- source_data
    variables <- c("cost", "risk", "seats", "noise", "crowdness", "convloc", "clientele")
    aligned <- data.frame(obs_id = RiskyTransport$chid, alt = RiskyTransport$mode, choice = as.logical(RiskyTransport$choice))
    for (var in variables) aligned[[var]] <- RiskyTransport[[var]]
    return(list(aligned = aligned, variables = variables))
  }
  if (dataset == "train") {
    Train <- source_data
    alternatives <- c("A", "B")
    variables <- c("price", "time", "change", "comfort")
    aligned <- wide_to_long(Train, alternatives, "choice", variables, function(var, alt) paste0(var, "_", alt))
    return(list(aligned = aligned, variables = variables))
  }
  stop(paste("Unsupported dataset:", dataset))
}

case <- make_case(dataset)
aligned <- case$aligned
variables <- case$variables
availability_col <- case$availability_col

for (var in variables) {
  aligned[[var]] <- as.numeric(aligned[[var]])
}
aligned <- aligned[complete.cases(aligned[, c("obs_id", "alt", "choice", variables)]), ]

if (has_flag("--data-only")) {
  write.csv(aligned, data_out, row.names = FALSE)
  payload <- list(
    backend = "mlogit",
    dataset = dataset,
    data_only = TRUE,
    n_obs = length(unique(aligned$obs_id)),
    n_rows = nrow(aligned),
    n_parameters = length(variables),
    variables = variables
  )
  write(toJSON(payload, auto_unbox = TRUE, digits = 16), result_out)
  quit(status = 0)
}

fit <- fit_long(aligned, variables, availability_col)
write.csv(aligned, data_out, row.names = FALSE)

payload <- list(
  backend = "mlogit",
  dataset = dataset,
  n_obs = length(unique(aligned$obs_id)),
  n_rows = nrow(aligned),
  n_parameters = length(variables),
  variables = variables,
  loglike = as.numeric(logLik(fit$model)),
  estimate_seconds = fit$estimate_seconds,
  covariance_seconds = fit$covariance_seconds,
  total_seconds = fit$estimate_seconds + fit$covariance_seconds,
  params = as.list(coef(fit$model)),
  covariance_names = colnames(fit$covariance),
  covariance = unname(fit$covariance)
)
write(toJSON(payload, auto_unbox = TRUE, digits = 16), result_out)
