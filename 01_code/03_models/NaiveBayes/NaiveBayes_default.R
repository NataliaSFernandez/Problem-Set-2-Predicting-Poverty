# nolint start
#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 03_modelos/naive_bayes.R
#==============================================================================
# ALGORITMO : Naive Bayes (e1071)
# DOS ESPECIFICACIONES:
#   NB_001A — Sin variables compuestas (misma lógica que LPM_002A)
#   NB_001B — Con variables compuestas (misma lógica que LPM_002B)
#
# OPTIMIZACIONES ACTIVAS:
#   - Threshold: CV-5 OOF (barrido 0.05 a 0.95)   ← YA ESTÁ
#   - Hiperparámetros: usekernel=TRUE               ← YA ESTÁ (máximo impacto)
#     laplace y adjust tienen impacto mínimo con datos grandes → no se tunean
#
# COLUMNAS DE REGISTRY (mismas que CART para uniformidad del equipo):
#   model_id, fecha, autor, algoritmo, n_features, imbalance_strategy,
#   cv_folds, cv_F1, cv_Precision, cv_Recall, auc_roc, kaggle_public_F1,
#   threshold, notas, cp, maxdepth, train_F1, train_Precision, train_Recall
#   (cp y maxdepth son NA para NB — no aplican a este algoritmo)
#==============================================================================

# -- 0. Paquetes -------------------------------------------------------------
library(tidyverse)
library(e1071)
library(caret)
library(pROC)
library(fs)

dir_outputs_nb  <- "02_outputs/models/NB/Baseline"
dir_submissions <- "03_submissions"
registry_path   <- "02_outputs/model_registry.csv"

fs::dir_create(dir_outputs_nb,  recurse = TRUE)
fs::dir_create(dir_submissions, recurse = TRUE)
fs::dir_create("02_outputs",    recurse = TRUE)

set.seed(42)

# -- 1. Carga de datos -------------------------------------------------------
message("== Cargando datos ==")

train   <- readRDS("00_data/processed/train_final.rds")
test    <- readRDS("00_data/processed/test_final.rds")
ids     <- test$id
y_train <- train$pobre   # vector numérico 0/1

AUTOR   <- "Natalia"
K_FOLDS <- 5

# =============================================================================
# LISTAS DE EXCLUSIÓN
# =============================================================================
VARS_EXCLUIR_A <- c(
  "id", "pobre",
  "zona", "bogota", "doble_ingreso", "brecha_educ",
  "indice_formalidad", "vulnerabilidad_lab", "n_fuentes_no_lab",
  "prop_contributivo", "costa_caribe", "region_pacifico", "eje_cafetero",
  "n_privaciones",
  "estrato_hog", "estrato_bajo", "estrato_x_zona",
  "ciudad", "dpto"
)

VARS_EXCLUIR_B <- c(
  "id", "pobre",
  "ciudad", "dpto",
  "sin_ocupados", "un_solo_ocupado",
  "nivel_educ_max", "nivel_educ_jefe",
  "prop_asalariado", "prop_cotiza_pension", "prop_prima_serv", "prop_sub_transp",
  "prop_domestico", "prop_fam_sin_rem", "prop_jornalero",
  "alguno_pension_jub", "alguno_remesas", "alguno_subsidio", "alguno_transf_nac",
  "alguno_intereses", "alguno_cesantias", "alguno_agropec", "alguno_arriendos",
  "prop_afil_salud", "prop_subsidiado",
  "dep_educacion", "dep_vivienda", "dep_empleo_formal",
  "dep_proteccion", "dep_dependencia",
  "estrato_hog", "estrato_bajo", "estrato_x_zona"
)

# =============================================================================
# FOLDS DE CV (una sola vez, compartidos entre NB_001A y NB_001B)
# =============================================================================
fold_ids <- sample(rep(1:K_FOLDS, length.out = nrow(train)))
message("\nFolds de CV generados (semilla 42).")

# =============================================================================
# FUNCIÓN run_nb()
# =============================================================================
run_nb <- function(model_id, notas, vars_excluir) {

  message("\n", strrep("=", 62))
  message("  INICIANDO MODELO: ", model_id)

  dir_model <- file.path(dir_outputs_nb, model_id)
  fs::dir_create(dir_model, recurse = TRUE)

  # ── PASO 1: Matrices X ────────────────────────────────────────────────────
  X_train_raw <- train |> select(-any_of(vars_excluir))
  X_test_raw  <- test  |> select(-any_of(vars_excluir))

  cols_comunes <- intersect(names(X_train_raw), names(X_test_raw))
  solo_train   <- setdiff(names(X_train_raw), names(X_test_raw))
  if (length(solo_train) > 0)
    message("  Solo en train (se eliminan): ", paste(solo_train, collapse = ", "))

  X_train_raw <- X_train_raw |> select(all_of(cols_comunes))
  X_test_raw  <- X_test_raw  |> select(all_of(cols_comunes))

  message("\nPredictores: ", ncol(X_train_raw))

  # ── Preprocesamiento ──────────────────────────────────────────────────────
  preprocess_nb <- function(df, ref_medians = NULL) {
    if ("zona" %in% names(df))
      df <- df |> mutate(zona = as.numeric(as.character(zona)))
    df <- df |> mutate(across(where(is.character), as.factor))
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

  # ── PASO 2: CV-5 OOF ─────────────────────────────────────────────────────
  oof_probs <- rep(NA_real_, nrow(X_train_raw))
  message("\n--- CV-5 OOF — ", model_id, " ---")

  for (k in seq_len(K_FOLDS)) {
    idx_val   <- which(fold_ids == k)
    idx_train <- which(fold_ids != k)

    prep_tr  <- preprocess_nb(X_train_raw[idx_train, ])
    df_tr    <- prep_tr$data
    medianas <- prep_tr$medians
    df_vl    <- preprocess_nb(X_train_raw[idx_val, ],
                               ref_medians = medianas)$data

    y_tr_k <- factor(y_train[idx_train], levels = c(0, 1))

    modelo_k <- naiveBayes(x = df_tr, y = y_tr_k,
                            laplace = 0, usekernel = TRUE)

    prob_mat_k         <- predict(modelo_k, newdata = df_vl, type = "raw")
    oof_probs[idx_val] <- prob_mat_k[, "1"]

    message(sprintf("  Fold %d/%d: P(pobre) en [%.3f, %.3f]",
                    k, K_FOLDS,
                    min(prob_mat_k[, "1"], na.rm = TRUE),
                    max(prob_mat_k[, "1"], na.rm = TRUE)))
  }

  if (any(is.na(oof_probs))) stop("NAs en oof_probs — revisar loop CV")

  cat(sprintf("\n--- Diagnóstico OOF — %s ---\n", model_id))
  print(summary(oof_probs))
  oof_probs_cl <- pmax(pmin(oof_probs, 1), 0)

  # ── PASO 3: Threshold óptimo (CV-OOF) ────────────────────────────────────
  th_grid <- seq(0.05, 0.95, by = 0.01)

  th_results <- map_dfr(th_grid, function(th) {
    pred_class <- factor(if_else(oof_probs_cl >= th, "Pobre", "NoPobre"),
                         levels = c("NoPobre", "Pobre"))
    obs_class  <- factor(if_else(y_train == 1, "Pobre", "NoPobre"),
                         levels = c("NoPobre", "Pobre"))
    cm <- confusionMatrix(pred_class, obs_class, positive = "Pobre")
    tibble(threshold = th,
           F1        = cm$byClass["F1"],
           Precision = cm$byClass["Precision"],
           Recall    = cm$byClass["Recall"])
  })

  best_th_row   <- th_results |> filter(!is.na(F1)) |>
    slice_max(F1, n = 1, with_ties = FALSE)
  THRESHOLD_OPT <- best_th_row$threshold
  THRESHOLD     <- THRESHOLD_OPT

  cat(sprintf("\n--- Threshold óptimo (CV-OOF) — %s ---\n", model_id))
  print(best_th_row)

  th_label_x     <- if_else(THRESHOLD_OPT > 0.80, THRESHOLD_OPT - 0.04,
                              THRESHOLD_OPT + 0.02)
  th_label_hjust <- if_else(THRESHOLD_OPT > 0.80, 1, 0)

  p_threshold <- th_results |>
    filter(!is.na(F1)) |>
    pivot_longer(c(F1, Precision, Recall), names_to = "metrica") |>
    ggplot(aes(threshold, value, color = metrica)) +
    geom_line(linewidth = 0.9) +
    geom_vline(xintercept = THRESHOLD_OPT, linetype = "dashed", color = "gray30") +
    geom_vline(xintercept = 0.50,          linetype = "dotted", color = "gray60") +
    annotate("text", x = th_label_x, y = 0.08,
             label = sprintf("th* = %.2f\n(max F1)", THRESHOLD_OPT),
             hjust = th_label_hjust, size = 3.2, color = "gray25") +
    scale_color_manual(
      values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3"),
      labels = c(F1        = "F1 (balance)",
                 Precision = "Precision (menos filtracion)",
                 Recall    = "Recall (menos exclusion)")
    ) +
    labs(title    = paste0("Naive Bayes: metricas por threshold — ", model_id),
         subtitle = sprintf("%d-fold CV OOF | n = %d | usekernel=TRUE",
                            K_FOLDS, nrow(X_train_raw)),
         x = "Threshold", y = "Metrica", color = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top")

  ggsave(file.path(dir_model, "threshold.png"),
         p_threshold, width = 8, height = 5, dpi = 150)
  message("Grafico guardado: ", model_id, "/threshold.png")

  # ── PASO 4: Modelo final (todo el train) ──────────────────────────────────
  message("\n--- Modelo final — ", model_id, " ---")

  prep_full     <- preprocess_nb(X_train_raw)
  X_train_f     <- prep_full$data
  medianas_full <- prep_full$medians

  model_final <- naiveBayes(
    x         = X_train_f,
    y         = factor(y_train, levels = c(0, 1)),
    laplace   = 0,
    usekernel = TRUE
  )

  message("  Entrenado sobre ", nrow(X_train_f), " hogares | ",
          ncol(X_train_f), " predictores")

  # ── PASO 5: Métricas CV y en train ────────────────────────────────────────

  # Métricas CV-OOF (honestas — estas van al registry como cv_*)
  oof_clase <- factor(if_else(oof_probs_cl >= THRESHOLD, "Pobre", "NoPobre"),
                      levels = c("NoPobre", "Pobre"))
  obs_clase <- factor(if_else(y_train == 1, "Pobre", "NoPobre"),
                      levels = c("NoPobre", "Pobre"))
  cm_cv <- confusionMatrix(oof_clase, obs_clase, positive = "Pobre")

  cv_F1        <- cm_cv$byClass["F1"]
  cv_Precision <- cm_cv$byClass["Precision"]
  cv_Recall    <- cm_cv$byClass["Recall"]

  # Métricas en train (en muestra — optimistas, van como train_* en registry)
  probs_train  <- predict(model_final, newdata = X_train_f, type = "raw")[, "1"]
  pred_train   <- factor(if_else(probs_train >= THRESHOLD, "Pobre", "NoPobre"),
                          levels = c("NoPobre", "Pobre"))
  cm_train     <- confusionMatrix(pred_train, obs_clase, positive = "Pobre")

  train_F1        <- cm_train$byClass["F1"]
  train_Precision <- cm_train$byClass["Precision"]
  train_Recall    <- cm_train$byClass["Recall"]

  cat(sprintf("\n--- Métricas — %s ---\n", model_id))
  cat(sprintf("  CV-OOF   F1: %.4f | Precision: %.4f | Recall: %.4f\n",
              cv_F1, cv_Precision, cv_Recall))
  cat(sprintf("  Train    F1: %.4f | Precision: %.4f | Recall: %.4f  (optimistas)\n",
              train_F1, train_Precision, train_Recall))

  FN <- cm_cv$table["NoPobre", "Pobre"]
  FP <- cm_cv$table["Pobre",   "NoPobre"]
  cat(sprintf("  Exclusion  (FN): %d pobres excluidos (%.1f%%)\n",
              FN, FN / sum(y_train == 1) * 100))
  cat(sprintf("  Filtracion (FP): %d no-pobres con beneficio (%.1f%%)\n",
              FP, FP / sum(y_train == 0) * 100))

  # ROC
  roc_obj <- pROC::roc(response = y_train, predictor = oof_probs_cl, quiet = TRUE)
  auc_roc <- as.numeric(pROC::auc(roc_obj))
  message("  AUC-ROC (OOF): ", round(auc_roc, 4))

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
    labs(title    = paste0("Naive Bayes: curva ROC — ", model_id),
         subtitle = "Predicciones CV-5 out-of-fold | usekernel=TRUE",
         x = "Tasa de Falsos Positivos (1 - Specificity)",
         y = "Tasa de Verdaderos Positivos (Recall)") +
    coord_equal() +
    theme_minimal(base_size = 12)

  ggsave(file.path(dir_model, "roc.png"), p_roc, width = 6, height = 6, dpi = 150)
  message("Grafico guardado: ", model_id, "/roc.png")

  pr_df <- tibble(
    TP        = roc_obj$sensitivities * sum(y_train == 1),
    FP        = (1 - roc_obj$specificities) * sum(y_train == 0),
    recall    = roc_obj$sensitivities,
    precision = TP / (TP + FP)
  ) |> filter(is.finite(precision), is.finite(recall))

  p_prcurve <- ggplot(pr_df, aes(recall, precision)) +
    geom_line(color = "#534AB7", linewidth = 0.9) +
    geom_hline(yintercept = mean(y_train), linetype = "dashed", color = "gray60") +
    annotate("text", x = 0.85, y = mean(y_train) + 0.015,
             label = sprintf("Azar (%.0f%%)", mean(y_train) * 100),
             size = 3, color = "gray50") +
    labs(title    = paste0("Naive Bayes: curva PR — ", model_id),
         subtitle = sprintf("AUC-ROC: %.4f | CV-5 OOF | usekernel=TRUE", auc_roc),
         x = "Recall", y = "Precision") +
    theme_minimal(base_size = 12)

  ggsave(file.path(dir_model, "prcurve.png"),
         p_prcurve, width = 6, height = 5, dpi = 150)
  message("Grafico guardado: ", model_id, "/prcurve.png")

  # ── PASO 6: Submission ────────────────────────────────────────────────────
  X_test_f <- preprocess_nb(X_test_raw, ref_medians = medianas_full)$data

  prob_mat_test <- predict(model_final, newdata = X_test_f, type = "raw")
  probs_test    <- prob_mat_test[, "1"]
  preds_test    <- as.integer(probs_test >= THRESHOLD)

  cat(sprintf("\n--- Predicciones test — %s ---\n", model_id))
  cat(sprintf("%d hogares predichos como pobres (%.1f%%) | threshold = %.2f\n",
              sum(preds_test), mean(preds_test) * 100, THRESHOLD))

  submission_file <- file.path(dir_submissions,
                                paste0("submission_", model_id, ".csv"))
  write_csv(tibble(id = test$id, pobre = preds_test), submission_file)
  message("Submission guardada: ", submission_file)
  message("-> Subir a Kaggle y anotar kaggle_public_F1 en model_registry.csv")

  # ── PASO 7: Diagnósticos y registry ──────────────────────────────────────
  saveRDS(
    list(model_id = model_id, autor = AUTOR, notas = notas,
         fecha = Sys.Date(), vars_excluir = vars_excluir,
         model_final = model_final, medianas_full = medianas_full,
         oof_probs = oof_probs, oof_probs_cl = oof_probs_cl,
         oof_actuals = y_train, fold_ids = fold_ids,
         th_results = th_results, threshold_opt = THRESHOLD_OPT,
         threshold_usado = THRESHOLD,
         cm_cv = cm_cv, cm_train = cm_train,
         cv_F1 = cv_F1, cv_Precision = cv_Precision, cv_Recall = cv_Recall,
         train_F1 = train_F1, train_Precision = train_Precision,
         train_Recall = train_Recall,
         auc_roc = auc_roc, predictores = names(X_train_f),
         n_features = ncol(X_train_f)),
    file.path(dir_model, "diagnostics.rds")
  )
  message("Diagnosticos guardados: ", model_id, "/diagnostics.rds")

  # Registry con exactamente las mismas columnas que CART
  # cp y maxdepth son NA porque no aplican a Naive Bayes
  nueva_fila <- tibble(
    model_id           = model_id,
    fecha              = Sys.Date(),
    autor              = AUTOR,
    algoritmo          = "NaiveBayes",
    n_features         = ncol(X_train_f),
    imbalance_strategy = "none",
    cv_folds           = K_FOLDS,
    cv_F1              = round(cv_F1,           4),
    cv_Precision       = round(cv_Precision,    4),
    cv_Recall          = round(cv_Recall,       4),
    auc_roc            = round(auc_roc,         4),
    kaggle_public_F1   = NA_real_,
    threshold          = THRESHOLD,
    notas              = notas,
    cp                 = NA_real_,   # no aplica a NB
    maxdepth           = NA_real_,   # no aplica a NB
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
  message("Registro actualizado: ", registry_path)

  message("\n", strrep("=", 62))
  message("  ", model_id, " completado")
  message(strrep("=", 62), "\n")

  tibble(
    model_id       = model_id,
    especificacion = if_else(grepl("A$", model_id), "componentes", "indices"),
    n_features     = ncol(X_train_f),
    cv_F1          = round(cv_F1,        4),
    cv_Precision   = round(cv_Precision, 4),
    cv_Recall      = round(cv_Recall,    4),
    auc_roc        = round(auc_roc,      4),
    threshold      = THRESHOLD
  )
}

# =============================================================================
# EJECUTAR AMBOS MODELOS
# =============================================================================
result_A <- run_nb(
  model_id     = "NB_001A",
  notas        = "Sin indices compuestos — LPM_002A. usekernel=TRUE. Threshold CV-5 OOF.",
  vars_excluir = VARS_EXCLUIR_A
)

result_B <- run_nb(
  model_id     = "NB_001B",
  notas        = "Con indices compuestos — LPM_002B. usekernel=TRUE. Threshold CV-5 OOF.",
  vars_excluir = VARS_EXCLUIR_B
)

# =============================================================================
# COMPARACIÓN FINAL
# =============================================================================
comparacion <- bind_rows(result_A, result_B)

cat("\n", strrep("=", 62), "\n")
cat("  COMPARACION: NB_001A vs NB_001B\n")
cat(strrep("=", 62), "\n\n")
print(comparacion, n = Inf)

ganador    <- comparacion |> slice_max(cv_F1, n = 1, with_ties = FALSE)
diferencia <- abs(comparacion$cv_F1[1] - comparacion$cv_F1[2])

cat(sprintf("\nMejor modelo: %s (F1 = %.4f | AUC = %.4f | %d features)\n",
            ganador$model_id, ganador$cv_F1, ganador$auc_roc, ganador$n_features))
cat(sprintf("Diferencia en F1-CV: %.4f\n", diferencia))

if (diferencia < 0.005) {
  cat("Diferencia < 0.005: preferir el mas parsimonioso —",
      comparacion |> slice_min(n_features, n = 1) |> pull(model_id), "\n")
}

# nolint end