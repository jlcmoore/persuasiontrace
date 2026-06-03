#!/usr/bin/env Rscript

# Fit mixed-effects models on annotation-derived features.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})

response_var <- "delta_changed"
min_rows_per_condition <- 5
PERSUADER_TYPES <- c("human", "llm")
MEAN_FEATURES <- c("mean_logos_z", "mean_pathos_z", "mean_ethos_z")
PERSUADER_TYPE_COL <- "persuader_type"
PARTICIPANT_PROP_COL <- "participant_proposition"
BASELINE_BELIEF_COL <- "baseline_belief_z"

parse_args <- function(args) {
  # Parse CLI arguments into a named list.
  parsed <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--no-per-condition") {
      parsed$no_per_condition <- TRUE
      i <- i + 1
      next
    }
    if (i == length(args)) {
      stop(paste("Missing value for argument", key))
    }
    value <- args[[i + 1]]
    if (key == "--data") {
      parsed$data <- value
    } else if (key == "--summary") {
      parsed$summary <- value
    } else if (key == "--marginal") {
      parsed$marginal <- value
    } else if (key == "--contrast") {
      parsed$contrast <- value
    } else if (key == "--slopes") {
      parsed$slopes <- value
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
  if (is.null(parsed$marginal)) {
    parsed$marginal <- "analysis/data/annotation_regression_marginals.csv"
  }
  if (is.null(parsed$contrast)) {
    parsed$contrast <- "analysis/data/annotation_regression_contrasts.csv"
  }
  if (is.null(parsed$slopes)) {
    parsed$slopes <- "analysis/data/annotation_regression_slopes.csv"
  }
  if (is.null(parsed$no_per_condition)) {
    parsed$no_per_condition <- FALSE
  }
  parsed
}

has_valid_random_effect <- function(vec, n_obs) {
  # Only keep random effects with multiple levels but not one-per-row.
  values <- vec[!is.na(vec)]
  if (length(values) == 0) {
    return(FALSE)
  }
  level_counts <- table(values)
  level_count <- length(level_counts)
  level_count > 1 && level_count < n_obs && any(level_counts > 1)
}

has_factor_variation <- function(vec) {
  # Require at least two observed levels.
  values <- vec[!is.na(vec)]
  if (length(values) == 0) {
    return(FALSE)
  }
  if (is.factor(values)) {
    return(nlevels(droplevels(values)) > 1)
  }
  length(unique(values)) > 1
}

normalize_bool <- function(vec) {
  # Normalize string or logical vectors into TRUE/FALSE/NA.
  if (is.logical(vec)) {
    return(vec)
  }
  values <- tolower(as.character(vec))
  is_true <- values %in% c("true", "t", "1", "yes")
  is_false <- values %in% c("false", "f", "0", "no")
  out <- rep(NA, length(values))
  out[is_true] <- TRUE
  out[is_false] <- FALSE
  out
}

normalize_term_label <- function(term) {
  # Move mean_* terms to the end for readability.
  if (!grepl("mean_", term)) {
    return(term)
  }
  parts <- unlist(strsplit(term, ":", fixed = TRUE))
  mean_parts <- parts[grepl("^mean_", parts)]
  other_parts <- parts[!grepl("^mean_", parts)]
  if (length(mean_parts) != 1 || length(other_parts) == 0) {
    return(term)
  }
  paste(c(other_parts, mean_parts), collapse = ":")
}

normalize_term_labels <- function(terms) {
  # Apply label normalization to all coefficient terms.
  vapply(terms, normalize_term_label, character(1))
}

write_csv_safe <- function(df, path) {
  # Write CSV after ensuring the parent directory exists.
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write.csv(df, path, row.names = FALSE)
}

build_formula <- function(df, include_condition_fe) {
  # Build fixed and random effects formula based on available columns.
  factor_terms <- c()
  if (PERSUADER_TYPE_COL %in% names(df) &&
        has_factor_variation(df[[PERSUADER_TYPE_COL]])) {
    factor_terms <- c(factor_terms, PERSUADER_TYPE_COL)
  }
  if (PARTICIPANT_PROP_COL %in% names(df) &&
        has_factor_variation(df[[PARTICIPANT_PROP_COL]])) {
    factor_terms <- c(factor_terms, PARTICIPANT_PROP_COL)
  }

  # Feature groups for interaction construction.
  mean_terms <- MEAN_FEATURES
  # NOTE: Length terms are intentionally omitted because they covary with
  # annotation magnitudes (humans write less). We want total effects here.
  # Re-enable msg_count_z/avg_chars_z terms for direct-effect sensitivity
  # checks.
  fixed_terms <- c()
  if (length(factor_terms) == 0) {
    # No categorical factors: use main effects only.
    fixed_terms <- c(mean_terms)
  } else if (length(factor_terms) == 1) {
    # Single factor: include interactions with each feature group.
    fixed_terms <- c(
      paste(mean_terms, factor_terms[1], sep = " * ")
    )
  } else {
    # Multiple factors: use full interaction across factors.
    interaction <- paste(factor_terms, collapse = " * ")
    fixed_terms <- c(
      paste(mean_terms, interaction, sep = " * ")
    )
  }
  # Always include baseline belief.
  fixed_terms <- c(fixed_terms, BASELINE_BELIEF_COL)
  fixed <- paste(fixed_terms, collapse = " + ")
  if (include_condition_fe &&
        length(factor_terms) == 0 &&
        "condition" %in% names(df)) {
    # Add condition fixed effects when not already modeled via factors.
    fixed <- paste(fixed, "+ condition")
  }

  re_terms <- c()
  # NOTE: target_id random effect is intentionally omitted because targets
  # are mostly one-off in the current dataset (no repeated measures). Add
  # back when the same target appears across multiple rounds.
  # NOTE: persuader_id random effect is intentionally omitted because LLM rows
  # are dominated by a single model and human persuaders are mostly one-off.
  # Add back when there are repeated persuaders across multiple rounds.
  # NOTE: proposition_id random effect is intentionally omitted because current
  # non-ppt-prop data use a single proposition. Add back when more propositions
  # are available to avoid confounding topic with condition.
  rhs <- fixed
  has_random <- length(re_terms) > 0
  if (has_random) {
    # Combine fixed and random effects for the final RHS.
    rhs <- paste(rhs, "+", paste(re_terms, collapse = " + "))
  }
  list(
    formula = as.formula(paste(response_var, "~", rhs)),
    has_random = has_random
  )
}

extract_coefficients <- function(model, model_label, model_type) {
  # Standardize coefficient output across model types.
  summary_fit <- summary(model)
  coefs <- summary_fit$coefficients
  terms <- normalize_term_labels(rownames(coefs))
  pvalue <- rep(NA_real_, nrow(coefs))
  if ("Pr(>|t|)" %in% colnames(coefs)) {
    pvalue <- coefs[, "Pr(>|t|)"]
  } else if ("Pr(>|z|)" %in% colnames(coefs)) {
    pvalue <- coefs[, "Pr(>|z|)"]
  }
  data.frame(
    model_label = model_label,
    model_type = model_type,
    term = terms,
    estimate = coefs[, "Estimate"],
    std_err = coefs[, "Std. Error"],
    t_value = coefs[, "t value"],
    pvalue = pvalue,
    stringsAsFactors = FALSE
  )
}

fit_model <- function(df, include_condition_fe) {
  # Fit with lmerTest for mixed models and lm otherwise.
  formula_info <- build_formula(df, include_condition_fe)
  formula_text <- paste(deparse(formula_info$formula), collapse = " ")
  message("Fitting overall with formula: ", formula_text)
  message("")
  if (formula_info$has_random) {
    lmerTest::lmer(formula_info$formula, data = df)
  } else {
    lm(formula_info$formula, data = df)
  }
}

fit_model_safe <- function(df, model_label, include_condition_fe) {
  # Fit with lmerTest for mixed models and lm otherwise.
  formula_info <- build_formula(df, include_condition_fe)
  formula_text <- paste(deparse(formula_info$formula), collapse = " ")
  message("Fitting ", model_label, " with formula: ", formula_text)
  message("")
  result <- tryCatch(
    {
      if (formula_info$has_random) {
        # lmerTest provides p-values for mixed models.
        model <- lmerTest::lmer(formula_info$formula, data = df)
        extract_coefficients(model, model_label, "lmerTest")
      } else {
        model <- lm(formula_info$formula, data = df)
        extract_coefficients(model, model_label, "lm")
      }
    },
    error = function(err) {
      # Return a row describing the failure instead of stopping.
      data.frame(
        model_label = model_label,
        model_type = "lmer",
        term = "__error__",
        estimate = NA_real_,
        std_err = NA_real_,
        t_value = NA_real_,
        pvalue = NA_real_,
        error = conditionMessage(err),
        stringsAsFactors = FALSE
      )
    }
  )
  result
}

write_emmeans_exports <- function(model, df, output_paths) {
  # Export marginal means, contrasts, and slopes to CSV files.
  if (!("persuader_type" %in% names(df))) {
    stop("Missing persuader_type for emmeans exports.")
  }
  terms_in_model <- all.vars(formula(model))
  has_ppt <- "participant_proposition" %in% terms_in_model &&
    "participant_proposition" %in% names(df) &&
    has_factor_variation(df$participant_proposition)
  if (!has_ppt) {
    stop("participant_proposition has no variation; check input data filters.")
  }

  emm <- emmeans::emmeans(
    model,
    ~ persuader_type * participant_proposition,
    data = df
  )
  emm_summary <- summary(emm, infer = c(TRUE, TRUE))
  cell_counts <- as.data.frame(
    table(df$persuader_type, df$participant_proposition),
    stringsAsFactors = FALSE
  )
  names(cell_counts) <- c(
    "persuader_type",
    "participant_proposition",
    "n"
  )
  marginals <- data.frame(
    persuader_type = emm_summary$persuader_type,
    participant_proposition = emm_summary$participant_proposition,
    estimate = emm_summary$emmean,
    std_err = emm_summary$SE,
    df = emm_summary$df,
    conf_low = emm_summary$lower.CL,
    conf_high = emm_summary$upper.CL
  )
  marginals <- merge(
    marginals,
    cell_counts,
    by = c("persuader_type", "participant_proposition"),
    all.x = TRUE
  )
  write_csv_safe(marginals, output_paths$marginal)

  contrast_list <- list("llm - human" = c(-1, 1))
  contrasts <- emmeans::contrast(
    emm,
    method = contrast_list,
    by = "participant_proposition"
  )
  contrast_summary <- summary(contrasts, infer = c(TRUE, TRUE))
  contrast_df <- data.frame(
    participant_proposition = contrast_summary$participant_proposition,
    contrast = contrast_summary$contrast,
    estimate = contrast_summary$estimate,
    std_err = contrast_summary$SE,
    conf_low = contrast_summary$lower.CL,
    conf_high = contrast_summary$upper.CL,
    p_value = contrast_summary$p.value
  )
  write_csv_safe(contrast_df, output_paths$contrast)

  trend_features <- MEAN_FEATURES
  slope_rows <- list()
  for (feature in trend_features) {
    trends <- emmeans::emtrends(
      model,
      ~ persuader_type * participant_proposition,
      var = feature,
      data = df
    )
    trend_summary <- summary(trends, infer = c(TRUE, TRUE))
    if (nrow(trend_summary) == 0) {
      next
    }
    trend_col <- paste0(feature, ".trend")
    if (!(trend_col %in% names(trend_summary))) {
      stop(paste("Expected trend column missing for", feature))
    }
    trend_df <- data.frame(
      feature = feature,
      persuader_type = trend_summary$persuader_type,
      participant_proposition = trend_summary$participant_proposition,
      estimate = trend_summary[[trend_col]],
      std_err = trend_summary$SE,
      conf_low = trend_summary$lower.CL,
      conf_high = trend_summary$upper.CL,
      p_value = trend_summary$p.value
    )
    slope_rows[[length(slope_rows) + 1]] <- trend_df
  }
  slopes <- do.call(rbind, slope_rows)
  write_csv_safe(slopes, output_paths$slopes)
}

main <- function() {
  # Entry point: load data, fit models, write summary.
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  df <- read.csv(args$data, stringsAsFactors = FALSE)
  if (nrow(df) == 0) {
    write_csv_safe(df, args$summary)
    quit(status = 0)
  }

  # Column groups for validation and modeling.
  response_cols <- c(response_var)
  mean_feature_cols <- MEAN_FEATURES
  length_feature_cols <- c("msg_count_z", "avg_chars_z")
  factor_cols <- c(PERSUADER_TYPE_COL, PARTICIPANT_PROP_COL)
  required <- c(
    response_cols,
    mean_feature_cols,
    BASELINE_BELIEF_COL,
    length_feature_cols,
    factor_cols
  )
  for (col in required) {
    if (!col %in% names(df)) {
      stop(paste("Missing required column:", col))
    }
  }

  df[[PARTICIPANT_PROP_COL]] <- normalize_bool(df[[PARTICIPANT_PROP_COL]])
  df[[PERSUADER_TYPE_COL]] <- as.character(df[[PERSUADER_TYPE_COL]])
  df[[PERSUADER_TYPE_COL]] <- tolower(df[[PERSUADER_TYPE_COL]])
  df[[PERSUADER_TYPE_COL]][!df[[PERSUADER_TYPE_COL]] %in% PERSUADER_TYPES] <- NA

  # Drop rows with missing values in required columns.
  df <- df[complete.cases(df[, required]), ]
  df[[PARTICIPANT_PROP_COL]] <- factor(
    df[[PARTICIPANT_PROP_COL]],
    levels = c(FALSE, TRUE)
  )
  df[[PERSUADER_TYPE_COL]] <- factor(
    df[[PERSUADER_TYPE_COL]],
    levels = PERSUADER_TYPES
  )

  model <- fit_model(df, TRUE)
  summary_rows <- extract_coefficients(model, "overall", class(model)[1])
  write_emmeans_exports(
    model,
    df,
    list(
      marginal = args$marginal,
      contrast = args$contrast,
      slopes = args$slopes
    )
  )

  if (!args$no_per_condition && "condition" %in% names(df)) {
    # Fit per-condition models when requested.
    conditions <- unique(df$condition)
    subset_rows_list <- list()
    for (cond in conditions) {
      subset_df <- df[df$condition == cond, ]
      if (nrow(subset_df) < min_rows_per_condition) {
        next
      }
      subset_rows_list[[length(subset_rows_list) + 1]] <- fit_model_safe(
        subset_df,
        paste("condition:", cond, sep = ""),
        FALSE
      )
    }
    if (length(subset_rows_list) > 0) {
      summary_rows <- rbind(summary_rows, do.call(rbind, subset_rows_list))
    }
  }

  write_csv_safe(summary_rows, args$summary)
}

main()
