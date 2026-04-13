# nolint start
#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/NaiveBayes_feature.R
#==============================================================================
# ALGORITMO : Naive Bayes (e1071) con variables seleccionadas por Random Forest
#
# MOTIVACION:
#   NB sufre con variables correlacionadas (viola el supuesto de independencia)
#   y con variables de alta cardinalidad (ciudad, dpto → celdas vacías).
#   Usar las variables ordenadas por importancia de RF permite:
#     1. Eliminar variables irrelevantes que añaden ruido
#     2. Reducir multicolinealidad usando solo las más discriminantes
#     3. Comparar directamente con la especificación de RF en el registry
#
# ESPECIFICACIONES (3 cortes de variables):
#   NB_RF_top20 — top 20 variables por importancia RF
#   NB_RF_top40 — top 40 variables por importancia RF
#   NB_RF_top60 — top 60 variables por importancia RF
#   (Se excluyen variables con importancia <= 0 y alta cardinalidad)
#
# OPTIMIZACIONES:
#   - Folds estratificados: caret::createFolds() — conserva ~20% pobres/fold
#   - Threshold: CV-5 OOF (barrido 0.05 a 0.95)
#   - usekernel = TRUE — distribución no paramétrica (mejor con vars bimodales)
#   - Binarias 0/1 integer → factor (evita calibración errónea con kernel)
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#   02_outputs/models/RandomForest/RF_001/feature_matrix.csv
#
# OUTPUTS:
#   03_submissions/submission_NB_RF_top20.csv
#   03_submissions/submission_NB_RF_top40.csv
#   03_submissions/submission_NB_RF_top60.csv
#   02_outputs/models/NB/NB_RF_top{20,40,60}/{threshold,roc,prcurve}.png
#   02_outputs/model_registry.csv  (append, una fila por especificación)
#==============================================================================

# -- 0. Paquetes -------------------------------------------------------------
library(tidyverse)
library(e1071)
library(caret)
library(pROC)
library(fs)

dir_outputs_nb  <- "02_outputs/models/NB/feature"  # NB con variables seleccionadas por RF
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

# -- 2. Variables de RF ------------------------------------------------------
message("\n== Cargando importancia de variables (RF) ==")

rf_imp <- read_csv(
  "02_outputs/models/RandomForest/RF_001/feature_matrix.csv",
  show_col_types = FALSE
)

# Excluir variables con importancia <= 0 (no aportan o perjudican)
# Excluir ciudad y dpto por alta cardinalidad (problema específico de NB:

rf_imp_clean <- rf_imp |>
  filter(importancia > 0) |>
  filter(!variable %in% c("ciudad", "dpto")) |>
  arrange(rank)

message("  Variables con importancia > 0: ", nrow(rf_imp_clean))
message("  (alguno_intereses excluida por importancia negativa)")

# Tres cortes de variables
CUTS <- list(
  NB_RF_top20 = rf_imp_clean |> head(20) |> pull(variable),
  NB_RF_top40 = rf_imp_clean |> head(40) |> pull(variable),
  NB_RF_top60 = rf_imp_clean |> head(60) |> pull(variable)
)

cat("\n-- Top 10 variables (RF) --\n")
print(rf_imp_clean |> head(10) |> select(rank, variable, importancia))

# -- 3. Folds estratificados -------------------------------------------------
set.seed(42)
fold_ids <- caret::createFolds(y = factor(y_train), k = K_FOLDS, list = FALSE)

message("\nFolds estratificados (semilla 42):")
for (k in 1:K_FOLDS) {
  prop_k <- mean(y_train[fold_ids == k] == 1)
  message(sprintf("  Fold %d: %.1f%% pobres", k, prop_k * 100))
}

# =============================================================================
# FUNCIÓN run_nb_rf()
# =============================================================================
run_nb_rf <- function(model_id, vars_sel) {

  message("\n", strrep("=", 62))
  message("  INICIANDO: ", model_id, " (", length(vars_sel), " variables)")

  dir_model <- file.path(dir_outputs_nb, model_id)
  fs::dir_create(dir_model, recurse = TRUE)

  # -- Variables que realmente existen en el dataset -------------------------
  vars_en_train <- intersect(vars_sel, names(train))
  vars_en_test  <- intersect(vars_sel, names(test))
  vars_ok       <- intersect(vars_en_train, vars_en_test)
  vars_missing  <- setdiff(vars_sel, vars_ok)

  if (length(vars_missing) > 0)
    message("  Vars del ranking no encontradas en datos: ",
            paste(vars_missing, collapse = ", "))

  message("  Variables usadas efectivamente: ", length(vars_ok))

  X_train_raw <- train |> select(all_of(vars_ok))
  X_test_raw  <- test  |> select(all_of(vars_ok))

  # -- Preprocesamiento ------------------------------------------------------
  preprocess_nb <- function(df, ref_medians = NULL) {

    # character → factor
    df <- df |> mutate(across(where(is.character), as.factor))

    # logical → factor (TRUE/FALSE no son continuos)
    df <- df |> mutate(across(where(is.logical), as.factor))

    # integer binario 0/1 → factor (son categóricas, no continuas)
    # Si entran como numéricas, kernel density las trata como distribución
    # continua y produce probabilidades extremas
    cols_bin <- names(df)[sapply(df, function(x) {
      is.integer(x) &&
        length(unique(na.omit(x))) == 2 &&
        all(unique(na.omit(x)) %in% c(0L, 1L))
    })]
    if (length(cols_bin) > 0)
      df <- df |> mutate(across(all_of(cols_bin), as.factor))

    # Medianas de referencia para imputar NAs 
    if (is.null(ref_medians)) {
      ref_medians <- df |>
        summarise(across(where(is.numeric), ~ median(., na.rm = TRUE))) |>
        as.list()
    }
    for (col in names(ref_medians)) {
      if (col %in% names(df) && anyNA(df[[col]]))
        df[[col]][is.na(df[[col]])] <- ref_medians[[col]]
    }

    list(data = df, medians = ref_medians)
  }

  # -- CV-5 OOF --------------------------------------------------------------
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

    modelo_k <- naiveBayes(x = prep_tr$data, y = y_tr_k,
                            laplace = 0, usekernel = FALSE)

    prob_mat_k         <- predict(modelo_k, newdata = df_vl, type = "raw")
    oof_probs[idx_val] <- prob_mat_k[, "1"]

    message(sprintf("  Fold %d/%d: P(pobre) en [%.3f, %.3f]",
                    k, K_FOLDS,
                    min(prob_mat_k[, "1"], na.rm = TRUE),
                    max(prob_mat_k[, "1"], na.rm = TRUE)))
  }

  cat(sprintf("\n--- Diagnostico OOF — %s ---\n", model_id))
  print(summary(oof_probs))
  oof_probs_cl <- pmax(pmin(oof_probs, 1), 0)

  # -- Threshold óptimo (CV-OOF) ---------------------------------------------
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

  # Métricas CV con threshold óptimo
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
  cat(sprintf("  Exclusion  (FN): %d pobres (%.1f%%)\n",
              FN, FN / sum(y_train == 1) * 100))
  cat(sprintf("  Filtracion (FP): %d no-pobres (%.1f%%)\n",
              FP, FP / sum(y_train == 0) * 100))

  roc_obj <- pROC::roc(response = y_train, predictor = oof_probs_cl, quiet = TRUE)
  auc_roc <- as.numeric(pROC::auc(roc_obj))
  message("  AUC-ROC (OOF): ", round(auc_roc, 4))

  # -- Modelo final (todo el train) ------------------------------------------
  prep_full     <- preprocess_nb(X_train_raw)
  X_train_f     <- prep_full$data
  medianas_full <- prep_full$medians

  model_final <- naiveBayes(x = X_train_f,
                              y = factor(y_train, levels = c(0, 1)),
                              laplace = 0, usekernel = TRUE)

  # Métricas en train (en muestra — optimistas)
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

  # -- Gráficos --------------------------------------------------------------
  th_label_x     <- if_else(THRESHOLD > 0.80, THRESHOLD - 0.04, THRESHOLD + 0.02)
  th_label_hjust <- if_else(THRESHOLD > 0.80, 1, 0)

  p_threshold <- th_results |>
    filter(!is.na(F1)) |>
    pivot_longer(c(F1, Precision, Recall), names_to = "metrica") |>
    ggplot(aes(threshold, value, color = metrica)) +
    geom_line(linewidth = 0.9) +
    geom_vline(xintercept = THRESHOLD,  linetype = "dashed", color = "gray30") +
    geom_vline(xintercept = 0.50,       linetype = "dotted", color = "gray60") +
    annotate("text", x = th_label_x, y = 0.08,
             label = sprintf("th* = %.2f\n(max F1)", THRESHOLD),
             hjust = th_label_hjust, size = 3.2, color = "gray25") +
    scale_color_manual(
      values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3"),
      labels = c(F1        = "F1 (balance)",
                 Precision = "Precision (menos filtracion)",
                 Recall    = "Recall (menos exclusion)")
    ) +
    labs(title    = paste0("NB con vars RF: threshold — ", model_id),
         subtitle = sprintf("%d vars RF | %d-fold CV OOF | usekernel=TRUE",
                            length(vars_ok), K_FOLDS),
         x = "Threshold", y = "Metrica", color = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top")

  ggsave(file.path(dir_model, "threshold.png"),
         p_threshold, width = 8, height = 5, dpi = 150)

  roc_df <- tibble(fpr = 1 - roc_obj$specificities,
                   tpr = roc_obj$sensitivities)

  p_roc <- ggplot(roc_df, aes(fpr, tpr)) +
    geom_line(color = "#534AB7", linewidth = 0.9) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
    annotate("text", x = 0.55, y = 0.15,
             label = sprintf("AUC-ROC = %.4f", auc_roc),
             size = 3.5, color = "#534AB7") +
    scale_x_continuous(labels = function(x) paste0(round(x * 100), "%")) +
    scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
    labs(title    = paste0("NB con vars RF: ROC — ", model_id),
         subtitle = sprintf("%d vars | CV-5 OOF", length(vars_ok)),
         x = "Tasa de Falsos Positivos (1 - Specificity)",
         y = "Tasa de Verdaderos Positivos (Recall)") +
    coord_equal() +
    theme_minimal(base_size = 12)

  ggsave(file.path(dir_model, "roc.png"), p_roc, width = 6, height = 6, dpi = 150)

  pr_df <- tibble(
    TP        = roc_obj$sensitivities * sum(y_train == 1),
    FP        = (1 - roc_obj$specificities) * sum(y_train == 0),
    recall    = roc_obj$sensitivities,
    precision = TP / (TP + FP)
  ) |> filter(is.finite(precision), is.finite(recall))

  p_pr <- ggplot(pr_df, aes(recall, precision)) +
    geom_line(color = "#534AB7", linewidth = 0.9) +
    geom_hline(yintercept = mean(y_train), linetype = "dashed", color = "gray60") +
    annotate("text", x = 0.85, y = mean(y_train) + 0.015,
             label = sprintf("Azar (%.0f%%)", mean(y_train) * 100),
             size = 3, color = "gray50") +
    labs(title    = paste0("NB con vars RF: PR — ", model_id),
         subtitle = sprintf("AUC-ROC: %.4f | %d vars | CV-5 OOF",
                            auc_roc, length(vars_ok)),
         x = "Recall", y = "Precision") +
    theme_minimal(base_size = 12)

  ggsave(file.path(dir_model, "prcurve.png"), p_pr, width = 6, height = 5, dpi = 150)

  message("  Graficos guardados en: ", dir_model)

  # -- Submission ------------------------------------------------------------
  X_test_f   <- preprocess_nb(X_test_raw, ref_medians = medianas_full)$data
  probs_test <- predict(model_final, newdata = X_test_f, type = "raw")[, "1"]
  preds_test <- as.integer(probs_test >= THRESHOLD)

  cat(sprintf("  %d pobres predichos (%.1f%%) | threshold = %.2f\n",
              sum(preds_test), mean(preds_test) * 100, THRESHOLD))

  sub_file <- file.path(dir_submissions, paste0("submission_", model_id, ".csv"))
  write_csv(tibble(id = test$id, pobre = preds_test), sub_file)
  message("  Submission: ", sub_file)
  message("  -> Subir a Kaggle y anotar kaggle_public_F1 en model_registry.csv")

  # -- Diagnósticos ----------------------------------------------------------
  saveRDS(
    list(model_id = model_id, vars_usadas = vars_ok,
         n_features = length(vars_ok), fold_ids = fold_ids,
         oof_probs = oof_probs_cl, threshold = THRESHOLD,
         cm_cv = cm_cv, cm_train = cm_train,
         cv_F1 = cv_F1, cv_Precision = cv_Precision, cv_Recall = cv_Recall,
         train_F1 = train_F1, auc_roc = auc_roc),
    file.path(dir_model, "diagnostics.rds")
  )

  # -- Registry --------------------------------------------------------------
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
    notas              = paste0("NB con top-", length(vars_ok),
                                " vars de RF_001. usekernel=TRUE. ",
                                "Folds estratificados. Threshold CV-5 OOF."),
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

  # Devuelve resumen para comparación final
  tibble(
    model_id   = model_id,
    n_features = length(vars_ok),
    cv_F1      = round(cv_F1,   4),
    auc_roc    = round(auc_roc, 4),
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
cat(sprintf("\nMejor corte: %s\n  F1 CV = %.4f | AUC = %.4f | %d vars | th = %.2f\n",
            ganador$model_id, ganador$cv_F1, ganador$auc_roc,
            ganador$n_features, ganador$threshold))

# nolint end