#!/usr/bin/env Rscript

# Fit an ordinal rhetoric model on Salvi DebateGPT merged data.

suppressPackageStartupMessages({
  library(MASS)
})

POST_COL <- "agreement_post_likert"
PRE_COL <- "agreement_pre_likert"
MEAN_FEATURES <- c("mean_logos_z", "mean_pathos_z", "mean_ethos_z")
TREATMENT_COL <- "treatment_type"
TOPIC_COL <- "topic"

parse_args <- function(args) {
  # Parse CLI args into a named list.
  parsed <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (i == length(args)) {
      stop(paste("Missing value for argument", key))
    }
    value <- args[[i + 1]]
    if (key == "--data") {
      parsed$data <- value
    } else if (key == "--summary") {
      parsed$summary <- value
    } else {
      stop(paste("Unknown argument", key))
    }
    i <- i + 2
  }
  if (is.null(parsed$data)) {
    stop("Missing --data")
  }
  if (is.null(parsed$summary)) {
    stop("Missing --summary")
  }
  parsed
}

has_factor_variation <- function(vec) {
  # Return whether there are at least two observed levels.
  values <- vec[!is.na(vec)]
  if (length(values) == 0) {
    return(FALSE)
  }
  length(unique(values)) > 1
}

build_formula <- function(df) {
  # Build an ordinal formula based on available predictors.
  terms <- c()
  if (has_factor_variation(df[[PRE_COL]])) {
    terms <- c(terms, PRE_COL)
  }
  for (feature in MEAN_FEATURES) {
    if (has_factor_variation(df[[feature]])) {
      terms <- c(terms, feature)
    }
  }
  if (TREATMENT_COL %in% names(df) && has_factor_variation(df[[TREATMENT_COL]])) {
    terms <- c(terms, TREATMENT_COL)
  }
  if (TOPIC_COL %in% names(df) && has_factor_variation(df[[TOPIC_COL]])) {
    terms <- c(terms, TOPIC_COL)
  }
  if (length(terms) == 0) {
    stop("No varying predictors available for ordinal model.")
  }
  rhs <- paste(terms, collapse = " + ")
  as.formula(paste(POST_COL, "~", rhs))
}

safe_write_csv <- function(df, path) {
  # Write CSV and create parent directory when needed.
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write.csv(df, path, row.names = FALSE)
}

extract_summary_table <- function(model, formula_text, nobs_count) {
  # Convert model summary into a tidy coefficient table.
  coef_matrix <- coef(summary(model))
  terms <- rownames(coef_matrix)
  estimates <- as.numeric(coef_matrix[, "Value"])
  std_err <- as.numeric(coef_matrix[, "Std. Error"])
  z_value <- estimates / std_err
  p_value <- 2 * pnorm(abs(z_value), lower.tail = FALSE)
  ci_low <- estimates - 1.96 * std_err
  ci_high <- estimates + 1.96 * std_err
  data.frame(
    term = terms,
    estimate = estimates,
    std_err = std_err,
    z_value = z_value,
    p_value = p_value,
    ci_low = ci_low,
    ci_high = ci_high,
    odds_ratio = exp(estimates),
    odds_ratio_ci_low = exp(ci_low),
    odds_ratio_ci_high = exp(ci_high),
    nobs = nobs_count,
    formula = formula_text,
    stringsAsFactors = FALSE
  )
}

main <- function() {
  # Entrypoint for Salvi ordinal model fitting.
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  df <- read.csv(args$data, stringsAsFactors = FALSE)
  required <- c(POST_COL, PRE_COL, MEAN_FEATURES)
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    stop(paste("Missing required columns:", paste(missing, collapse = ", ")))
  }

  for (feature in MEAN_FEATURES) {
    df[[feature]] <- as.numeric(df[[feature]])
  }
  df[[PRE_COL]] <- as.numeric(df[[PRE_COL]])
  df[[POST_COL]] <- ordered(as.integer(df[[POST_COL]]), levels = 1:5)

  if (TREATMENT_COL %in% names(df)) {
    df[[TREATMENT_COL]] <- factor(df[[TREATMENT_COL]])
  }
  if (TOPIC_COL %in% names(df)) {
    df[[TOPIC_COL]] <- factor(df[[TOPIC_COL]])
  }

  model_cols <- c(POST_COL, PRE_COL, MEAN_FEATURES)
  if (TREATMENT_COL %in% names(df)) {
    model_cols <- c(model_cols, TREATMENT_COL)
  }
  if (TOPIC_COL %in% names(df)) {
    model_cols <- c(model_cols, TOPIC_COL)
  }
  df <- df[complete.cases(df[, model_cols]), , drop = FALSE]
  if (nrow(df) == 0) {
    stop("No complete rows available for ordinal model.")
  }

  formula <- build_formula(df)
  formula_text <- paste(deparse(formula), collapse = " ")
  message("Fitting ordinal model with formula: ", formula_text)
  summary_df <- tryCatch(
    {
      model <- MASS::polr(formula, data = df, Hess = TRUE, method = "logistic")
      extract_summary_table(
        model = model,
        formula_text = formula_text,
        nobs_count = nrow(df)
      )
    },
    error = function(err) {
      data.frame(
        term = "__error__",
        estimate = NA_real_,
        std_err = NA_real_,
        z_value = NA_real_,
        p_value = NA_real_,
        ci_low = NA_real_,
        ci_high = NA_real_,
        odds_ratio = NA_real_,
        odds_ratio_ci_low = NA_real_,
        odds_ratio_ci_high = NA_real_,
        nobs = nrow(df),
        formula = formula_text,
        error = conditionMessage(err),
        stringsAsFactors = FALSE
      )
    }
  )
  safe_write_csv(summary_df, args$summary)
}

main()
