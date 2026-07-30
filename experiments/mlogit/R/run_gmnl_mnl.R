suppressPackageStartupMessages(library(mlogit))
suppressPackageStartupMessages(library(gmnl))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx)) return(default)
  args[[idx + 1]]
}

dataset <- get_arg("--dataset")
data_input <- get_arg("--data-input")
result_out <- get_arg("--result-output")

if (is.null(dataset) || is.null(data_input) || is.null(result_out)) {
  stop("Usage: run_gmnl_mnl.R --dataset DATASET --data-input source.csv --result-output result.json")
}

source_data <- read.csv(data_input, check.names = FALSE, stringsAsFactors = FALSE)

if (dataset == "fishing") {
  Fishing <- source_data
  mdata <- mlogit.data(Fishing, varying = 2:9, shape = "wide", choice = "mode")
  estimate_start <- proc.time()[["elapsed"]]
  model <- gmnl(mode ~ price + catch, data = mdata, model = "mnl", print.level = 0)
  estimate_seconds <- proc.time()[["elapsed"]] - estimate_start
} else if (dataset == "modecanada") {
  ModeCanada <- source_data
  mdata <- mlogit.data(ModeCanada, shape = "long", chid.var = "case", alt.var = "alt", choice = "choice")
  estimate_start <- proc.time()[["elapsed"]]
  model <- gmnl(choice ~ cost + ivt + ovt | 0, data = mdata, model = "mnl", print.level = 0)
  estimate_seconds <- proc.time()[["elapsed"]] - estimate_start
} else {
  stop(paste("Unknown dataset:", dataset))
}

covariance_start <- proc.time()[["elapsed"]]
covariance <- vcov(model)
covariance_seconds <- proc.time()[["elapsed"]] - covariance_start

payload <- list(
  backend = "gmnl",
  dataset = dataset,
  loglike = as.numeric(logLik(model)),
  estimate_seconds = estimate_seconds,
  covariance_seconds = covariance_seconds,
  params = as.list(coef(model)),
  covariance_names = colnames(covariance),
  covariance = unname(covariance)
)
write(toJSON(payload, auto_unbox = TRUE, digits = 16), result_out)
