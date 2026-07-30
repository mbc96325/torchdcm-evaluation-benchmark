suppressPackageStartupMessages(library(mlogit))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx)) return(default)
  args[[idx + 1]]
}

dataset <- get_arg("--dataset")
data_out <- get_arg("--data-output")
result_out <- get_arg("--result-output")

if (is.null(dataset) || is.null(data_out) || is.null(result_out)) {
  stop("Usage: run_mlogit_mnl.R --dataset DATASET --data-output data.csv --result-output result.json")
}

if (dataset == "fishing") {
  data(Fishing, package = "mlogit")
  alternatives <- c("beach", "boat", "charter", "pier")
  rows <- list()
  cursor <- 1
  for (i in seq_len(nrow(Fishing))) {
    for (alt in alternatives) {
      rows[[cursor]] <- data.frame(
        obs_id = i,
        alt = alt,
        choice = Fishing$mode[[i]] == alt,
        price = Fishing[[paste0("price.", alt)]][[i]],
        catch = Fishing[[paste0("catch.", alt)]][[i]]
      )
      cursor <- cursor + 1
    }
  }
  aligned <- do.call(rbind, rows)
  write.csv(aligned, data_out, row.names = FALSE)
  mdata <- mlogit.data(Fishing, varying = 2:9, shape = "wide", choice = "mode")
  estimate_start <- proc.time()[["elapsed"]]
  model <- mlogit(mode ~ price + catch, mdata, print.level = 0)
  estimate_seconds <- proc.time()[["elapsed"]] - estimate_start
  covariance_start <- proc.time()[["elapsed"]]
  covariance <- vcov(model)
  covariance_seconds <- proc.time()[["elapsed"]] - covariance_start
} else if (dataset == "modecanada") {
  data(ModeCanada, package = "mlogit")
  aligned <- data.frame(
    obs_id = ModeCanada$case,
    alt = ModeCanada$alt,
    choice = as.logical(ModeCanada$choice),
    cost = ModeCanada$cost,
    ivt = ModeCanada$ivt,
    ovt = ModeCanada$ovt
  )
  write.csv(aligned, data_out, row.names = FALSE)
  mdata <- mlogit.data(ModeCanada, shape = "long", chid.var = "case", alt.var = "alt", choice = "choice")
  estimate_start <- proc.time()[["elapsed"]]
  model <- mlogit(choice ~ cost + ivt + ovt | 0, mdata, print.level = 0)
  estimate_seconds <- proc.time()[["elapsed"]] - estimate_start
  covariance_start <- proc.time()[["elapsed"]]
  covariance <- vcov(model)
  covariance_seconds <- proc.time()[["elapsed"]] - covariance_start
} else {
  stop(paste("Unknown dataset:", dataset))
}

payload <- list(
  backend = "mlogit",
  dataset = dataset,
  loglike = as.numeric(logLik(model)),
  estimate_seconds = estimate_seconds,
  covariance_seconds = covariance_seconds,
  params = as.list(coef(model)),
  covariance_names = colnames(covariance),
  covariance = unname(covariance)
)
write(toJSON(payload, auto_unbox = TRUE, digits = 16), result_out)
