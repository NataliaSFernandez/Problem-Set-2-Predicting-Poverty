# nolint start
#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/NaiveBayes_feature.R
#==============================================================================
# ALGORITMO : Naive Bayes (e1071) con variables seleccionadas por Random Forest
# ESPECIFICACIONES (3 cortes):
#   NB_RF_top20, NB_RF_top40, NB_RF_top60
#==============================================================================

# -- 0. Paquetes -------------------------------------------------------------
library(tidyverse)
library(e1071)
library(caret)
library(pROC)
library(fs)

dir_outputs_nb  <- "02_outputs/models/NB/feature"
dir_submissions <- "03_submissions"
registry_path   <- "02_outputs/model_registry.csv"

fs::dir_create(dir_outputs_nb,  recurse = TRUE)
fs::dir_create(dir_submissions, recurse = TRUE)
fs::dir_create("02_outputs",    recurse = TRUE)

# -- 1. Carga de datos -------------------------------------------------------
message("== Cargando datos ==")

train   <- readRDS("00_data/processed/train_final.rds")
test    <- readRDS("00_data/processed/test_final.rds")
y_train <- train$pobre   # 0/1 numérico

AUTOR   <- "Natalia"
K_FOLDS <- 5

message("  train: ", nrow(train), " x ", ncol(train))
message("  Prevalencia pobre: ", round(mean(y_train) * 100, 1), "%")

cat("\n-- Verificacion codificacion y_train --\n")
cat("  Valor 0:", sum(y_train == 0), "| Valor 1:", sum(y_train == 1), "\n")
cat("  Proporcion 1 (pobres):", round(mean(y_train == 1) * 100, 1),
    "% — esperado ~20%\n")

# -- 2. Variables de RF ------------------------------------------------------
message("\n== Cargando importancia de RF ==")

rf_imp <- read_csv(
  "02_outputs/models/RandomForest/RF_001/feature_matrix.csv",
  show_col_types = FALSE
) |>
  filter(importancia > 0) |>
  arrange(rank)

message("  Variables con importancia > 0: ", nrow(rf_imp))

CUTS <- list(
  NB_RF_top20 = rf_imp |> head(20) |> pull(variable),
  NB_RF_top40 = rf_imp |> head(40) |> pull(variable),
  NB_RF_top60 = rf_imp |> head(60) |> pull(variable)
)

cat("\n-- Top 10 variables (RF) --\n")
print(rf_imp |> head(10) |> select(rank, variable, importancia))

# -- 3. Folds estratificados -------------------------------------------------
set.seed(42)
fold_ids <- caret::createFolds(y = factor(y_train), k = K_FOLDS, list = FALSE)

message("\nFolds estratificados (semilla 42):")
for (k in 1:K_FOLDS) {
  message(sprintf("  Fold %d: %.1f%% pobres", k,
                  mean(y_train[fold_ids == k] == 1) * 100))
}

# =============================================================================
# FUNCIÓN run_nb_rf()
# =============================================================================
run_nb_rf <- function(model_id, vars_sel) {

  message("\n", strrep("=", 62))
  message("  INICIANDO: ", model_id, " (", length(vars_sel), " variables)")

  dir_model <- file.path(dir_outputs_nb, model_id)
  fs::dir_create(dir_model, recurse = TRUE)

  # Variables disponibles en ambos datasets
  vars_ok      <- intersect(vars_sel, intersect(names(train), names(test)))
  vars_missing <- setdiff(vars_sel, vars_ok)
  if (length(vars_missing) > 0)
    message("  Vars no encontradas: ", paste(vars_missing, collapse = ", "))
  message("  Variables efectivas: ", length(vars_ok))

  X_train_raw <- train |> select(all_of(vars_ok))
  X_test_raw  <- test  |> select(all_of(vars_ok))

  # ── Preprocesamiento ──────────────────────────────────────────────────────
  preprocess_nb <- function(df, ref_medians = NULL) {

    # character → factor (incluye ciudad, dpto, cotiza_pension)
    df <- df |> mutate(across(where(is.character), as.factor))

    # logical → integer (para kernel density)
    df <- df |> mutate(across(where(is.logical), as.integer))

    # Imputar NAs numéricos con mediana
    if (is.null(ref_medians)) {
      ref_medians <- df |>
        summarise(across(where(is.numeric), ~ median(., na.rm = TRUE))) |>
        as.list()
    }
    for (col in names(ref_medians)) {
      if (col %in% names(df) && anyNA(df[[col]]))
        df[[col]][is.na(df[[col]])] <- ref_medians[[col]]
    }
    # Imputar NAs en factores con la moda
    for (col in names(df)) {
      if (is.factor(df[[col]]) && anyNA(df[[col]])) {
        moda <- names(sort(table(df[[col]]), decreasing = TRUE))[1]
        df[[col]][is.na(df[[col]])] <- moda
      }
    }

    list(data = df, medians = ref_medians)
  }

  # ── CV-5 OOF ─────────────────────────────────────────────────────────────
  oof_probs <- rep(NA_real_, nrow(X_train_raw))
  message("\n--- CV-5 OOF ---")

  for (k in seq_len(K_FOLDS)) {
    idx_val   <- which(fold_ids == k)
    idx_train <- which(fold_ids != k)

    prep_tr  <- preprocess_nb(X_train_raw[idx_train, ])
    medianas <- prep_tr$medians
    df_vl    <- preprocess_nb(X_train_raw[idx_val, ],
                               ref_medians = medianas)$data

    y_tr_k <- factor(y_train[idx_train], levels = c(0, 1))
    stopifnot("Nivel positivo debe ser '1'" = levels(y_tr_k)[2] == "1")

    modelo_k <- naiveBayes(x = prep_tr$data, y = y_tr_k,
                            laplace = 1, usekernel = TRUE)

    prob_mat_k         <- predict(modelo_k, newdata = df_vl, type = "raw")
    oof_probs[idx_val] <- prob_mat_k[, "1"]

    message(sprintf("  Fold %d/%d: P(pobre) en [%.3f, %.3f]",
                    k, K_FOLDS,
                    min(prob_mat_k[, "1"], na.rm = TRUE),
                    max(prob_mat_k[, "1"], na.rm = TRUE)))
  }

  cat(sprintf("\n--- Diagnostico OOF — %s ---\n", model_id))
  print(summary(oof_probs))
  if (median(oof_probs, na.rm = TRUE) > 0.5)
    warning("Mediana OOF > 0.5 — verificar codificacion de y_train")

  oof_probs_cl <- pmax(pmin(oof_probs, 1), 0)

  # ── Threshold óptimo ─────────────────────────────────────────────────────
  th_grid <- seq(0.05, 0.95, by = 0.01)

  th_results <- map_dfr(th_grid, function(th) {
    pred_c <- factor(if_else(oof_probs_cl >= th, "Pobre", "NoPobre"),
                     levels = c("NoPobre", "Pobre"))
    obs_c  <- factor(if_else(y_train == 1, "Pobre", "NoPobre"),
                     levels = c("NoPobre", "Pobre"))
    cm <- confusionMatrix(pred_c, obs_c, positive = "Pobre")
    tibble(threshold = th,
           F1        = cm$byClass["F1"],
           Precision = cm$byClass["Precision"],
           Recall    = cm$byClass["Recall"])
  })

  best_th   <- th_results |> filter(!is.na(F1)) |>
    slice_max(F1, n = 1, with_ties = FALSE)
  THRESHOLD <- best_th$threshold

  cat(sprintf("\n--- Threshold optimo — %s ---\n", model_id))
  print(best_th)
  if (THRESHOLD > 0.80)
    message("  AVISO: threshold > 0.80 — posible problema de calibracion")

  cm_cv <- confusionMatrix(
    factor(if_else(oof_probs_cl >= THRESHOLD, "Pobre", "NoPobre"),
           levels = c("NoPobre", "Pobre")),
    factor(if_else(y_train == 1, "Pobre", "NoPobre"),
           levels = c("NoPobre", "Pobre")),
    positive = "Pobre"
  )
  cv_F1        <- cm_cv$byClass["F1"]
  cv_Precision <- cm_cv$byClass["Precision"]
  cv_Recall    <- cm_cv$byClass["Recall"]

  FN <- cm_cv$table["NoPobre", "Pobre"]
  FP <- cm_cv$table["Pobre",   "NoPobre"]
  cat(sprintf("  CV F1: %.4f | Precision: %.4f | Recall: %.4f\n",
              cv_F1, cv_Precision, cv_Recall))
  cat(sprintf("  Exclusion (FN): %d (%.1f%%) | Filtracion (FP): %d (%.1f%%)\n",
              FN, FN / sum(y_train == 1) * 100,
              FP, FP / sum(y_train == 0) * 100))

  roc_obj <- pROC::roc(response = y_train, predictor = oof_probs_cl, quiet = TRUE)
  auc_roc <- as.numeric(pROC::auc(roc_obj))
  message("  AUC-ROC (OOF): ", round(auc_roc, 4))

  # ── Modelo final ──────────────────────────────────────────────────────────
  prep_full     <- preprocess_nb(X_train_raw)
  X_train_f     <- prep_full$data
  medianas_full <- prep_full$medians

  model_final <- naiveBayes(x = X_train_f,
                              y = factor(y_train, levels = c(0, 1)),
                              laplace = 1, usekernel = TRUE)

  probs_tr    <- predict(model_final, newdata = X_train_f, type = "raw")[, "1"]
  obs_c_train <- factor(if_else(y_train == 1, "Pobre", "NoPobre"),
                         levels = c("NoPobre", "Pobre"))
  cm_train <- confusionMatrix(
    factor(if_else(probs_tr >= THRESHOLD, "Pobre", "NoPobre"),
           levels = c("NoPobre", "Pobre")),
    obs_c_train, positive = "Pobre"
  )
  train_F1        <- cm_train$byClass["F1"]
  train_Precision <- cm_train$byClass["Precision"]
  train_Recall    <- cm_train$byClass["Recall"]

  # ── Gráficos ──────────────────────────────────────────────────────────────
  th_label_x     <- if_else(THRESHOLD > 0.80, THRESHOLD - 0.04, THRESHOLD + 0.02)
  th_label_hjust <- if_else(THRESHOLD > 0.80, 1, 0)

  p_threshold <- th_results |>
    filter(!is.na(F1)) |>
    pivot_longer(c(F1, Precision, Recall), names_to = "metrica") |>
    ggplot(aes(threshold, value, color = metrica)) +
    geom_line(linewidth = 0.9) +
    geom_vline(xintercept = THRESHOLD, linetype = "dashed", color = "gray30") +
    geom_vline(xintercept = 0.50,      linetype = "dotted", color = "gray60") +
    annotate("text", x = th_label_x, y = 0.08,
             label = sprintf("th* = %.2f\n(max F1)", THRESHOLD),
             hjust = th_label_hjust, size = 3.2, color = "gray25") +
    scale_color_manual(
      values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3"),
      labels = c(F1 = "F1", Precision = "Precision", Recall = "Recall")
    ) +
    labs(title    = paste0("NB vars RF: threshold — ", model_id),
         subtitle = sprintf("%d vars | CV-5 OOF | usekernel=TRUE | laplace=1",
                            length(vars_ok)),
         x = "Threshold", y = "Metrica", color = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top")

  ggsave(file.path(dir_model, "threshold.png"),
         p_threshold, width = 8, height = 5, dpi = 150)

  roc_df <- tibble(fpr = 1 - roc_obj$specificities, tpr = roc_obj$sensitivities)

  p_roc <- ggplot(roc_df, aes(fpr, tpr)) +
    geom_line(color = "#534AB7", linewidth = 0.9) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
    annotate("text", x = 0.55, y = 0.15,
             label = sprintf("AUC-ROC = %.4f", auc_roc),
             size = 3.5, color = "#534AB7") +
    scale_x_continuous(labels = function(x) paste0(round(x * 100), "%")) +
    scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
    labs(title    = paste0("NB vars RF: ROC — ", model_id),
         subtitle = sprintf("%d vars | CV-5 OOF", length(vars_ok)),
         x = "Tasa de Falsos Positivos (1 - Specificity)",
         y = "Tasa de Verdaderos Positivos (Recall)") +
    coord_equal() +
    theme_minimal(base_size = 12)

  ggsave(file.path(dir_model, "roc.png"), p_roc, width = 6, height = 6, dpi = 150)

  pr_df <- tibble(
    TP = roc_obj$sensitivities * sum(y_train == 1),
    FP = (1 - roc_obj$specificities) * sum(y_train == 0),
    recall = roc_obj$sensitivities,
    precision = TP / (TP + FP)
  ) |> filter(is.finite(precision), is.finite(recall))

  p_pr <- ggplot(pr_df, aes(recall, precision)) +
    geom_line(color = "#534AB7", linewidth = 0.9) +
    geom_hline(yintercept = mean(y_train), linetype = "dashed", color = "gray60") +
    annotate("text", x = 0.85, y = mean(y_train) + 0.015,
             label = sprintf("Azar (%.0f%%)", mean(y_train) * 100),
             size = 3, color = "gray50") +
    labs(title    = paste0("NB vars RF: PR — ", model_id),
         subtitle = sprintf("AUC-ROC: %.4f | %d vars | CV-5 OOF",
                            auc_roc, length(vars_ok)),
         x = "Recall", y = "Precision") +
    theme_minimal(base_size = 12)

  ggsave(file.path(dir_model, "prcurve.png"), p_pr, width = 6, height = 5, dpi = 150)
  message("  Graficos guardados en: ", dir_model)

  # ── Submission ────────────────────────────────────────────────────────────
  X_test_f   <- preprocess_nb(X_test_raw, ref_medians = medianas_full)$data
  probs_test <- predict(model_final, newdata = X_test_f, type = "raw")[, "1"]
  preds_test <- as.integer(probs_test >= THRESHOLD)

  cat(sprintf("  %d pobres (%.1f%%) | threshold=%.2f\n",
              sum(preds_test), mean(preds_test) * 100, THRESHOLD))

  sub_file <- file.path(dir_submissions, paste0("submission_", model_id, ".csv"))
  write_csv(tibble(id = test$id, pobre = preds_test), sub_file)
  message("  Submission: ", sub_file)
  message("  -> Subir a Kaggle y anotar kaggle_public_F1")

  # ── Diagnósticos y registry ───────────────────────────────────────────────
  saveRDS(
    list(model_id = model_id, vars_usadas = vars_ok,
         n_features = length(vars_ok), fold_ids = fold_ids,
         oof_probs = oof_probs_cl, threshold = THRESHOLD,
         cm_cv = cm_cv, cm_train = cm_train,
         cv_F1 = cv_F1, cv_Precision = cv_Precision, cv_Recall = cv_Recall,
         train_F1 = train_F1, train_Precision = train_Precision,
         train_Recall = train_Recall, auc_roc = auc_roc),
    file.path(dir_model, "diagnostics.rds")
  )

  nueva_fila <- tibble(
    model_id           = model_id,
    fecha              = Sys.Date(),
    autor              = AUTOR,
    algoritmo          = "NaiveBayes",
    n_features         = length(vars_ok),
    imbalance_strategy = "none",
    cv_folds           = K_FOLDS,
    cv_F1              = round(cv_F1,           4),
    cv_Precision       = round(cv_Precision,    4),
    cv_Recall          = round(cv_Recall,       4),
    auc_roc            = round(auc_roc,         4),
    kaggle_public_F1   = NA_real_,
    threshold          = THRESHOLD,
    notas              = paste0("NB top-", length(vars_ok),
                                " vars RF_001. usekernel=TRUE, laplace=1.",
                                " Ciudad/dpto incluidos si en top-N.",
                                " Binarias como distribucion (no factor).",
                                " Threshold CV-5 OOF."),
    cp                 = NA_real_,
    maxdepth           = NA_real_,
    train_F1           = round(train_F1,        4),
    train_Precision    = round(train_Precision, 4),
    train_Recall       = round(train_Recall,    4)
  )

  este_id  <- model_id
  registry <- if (file.exists(registry_path)) {
    existing <- read_csv(registry_path, show_col_types = FALSE)
    existing <- existing |> filter(model_id != este_id)
    bind_rows(existing, nueva_fila)
  } else {
    nueva_fila
  }
  write_csv(registry, registry_path)
  message("  Registry actualizado.")
  message(strrep("=", 62))

  tibble(
    model_id   = model_id,
    n_features = length(vars_ok),
    cv_F1      = round(cv_F1,    4),
    auc_roc    = round(auc_roc,  4),
    threshold  = THRESHOLD,
    train_F1   = round(train_F1, 4)
  )
}

# =============================================================================
# EJECUTAR LOS TRES CORTES
# =============================================================================
result_20 <- run_nb_rf("NB_RF_top20", CUTS$NB_RF_top20)
result_40 <- run_nb_rf("NB_RF_top40", CUTS$NB_RF_top40)
result_60 <- run_nb_rf("NB_RF_top60", CUTS$NB_RF_top60)

# =============================================================================
# COMPARACIÓN FINAL
# =============================================================================
comparacion <- bind_rows(result_20, result_40, result_60)

cat("\n", strrep("=", 62), "\n")
cat("  COMPARACION: NB con variables de RF\n")
cat(strrep("=", 62), "\n\n")
print(comparacion)

ganador <- comparacion |> slice_max(cv_F1, n = 1, with_ties = FALSE)
cat(sprintf("\nMejor: %s — F1=%.4f | AUC=%.4f | %d vars | th=%.2f\n",
            ganador$model_id, ganador$cv_F1, ganador$auc_roc,
            ganador$n_features, ganador$threshold))

# nolint end