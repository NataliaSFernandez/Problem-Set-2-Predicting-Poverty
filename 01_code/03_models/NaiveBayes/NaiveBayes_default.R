#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 03_modelos/naive_bayes.R
#==============================================================================
# ALGORITMO : Naive Bayes
# PASO      : 1 — Baseline sin tuning
# OUTPUT    : submissions/NaiveBayes_default.csv
#
# SUPUESTO CLAVE:
#   Naive Bayes asume independencia condicional entre predictores dado Y.
#   En la práctica este supuesto se viola, pero el clasificador sigue siendo
#   competitivo con variables débilmente correlacionadas.
#   Para variables continuas asume distribución gaussiana por defecto.
#
# PREPROCESAMIENTO NECESARIO:
#   e1071::naiveBayes() no acepta NAs, characters ni factores de texto.
# QUÉ HACE ESTE SCRIPT:
#   Naive Bayes es un clasificador probabilístico basado en el teorema de Bayes.
#   Estima P(pobre | X) = P(X | pobre) × P(pobre) / P(X), donde el supuesto
#   "naive" es que todos los predictores son independientes entre sí dado Y.
#   Este supuesto se viola en la práctica, pero el clasificador sigue siendo
#   competitivo cuando las correlaciones entre variables son débiles.
#
# DOS ESPECIFICACIONES:
#   NB_001A — Sin variables compuestas (misma lógica que LPM_002A):
#             conserva componentes, elimina derivados y alias.
#
#   NB_001B — Con variables compuestas (misma lógica que LPM_002B):
#             conserva índices agregados, elimina componentes.
# FLUJO:
#   0. Paquetes y configuración
#   1. Identificación de modelos
#   2. Cargar datos
#   3. Listas de exclusión (A y B)
#   4. Generar folds de CV
#   5. Función run_nb()
#   6. Ejecutar NB_001A y NB_001B
#   7. Comparación final
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#
# OUTPUTS:
#   03_submissions/submission_NB_001A.csv
#   03_submissions/submission_NB_001B.csv
#   02_outputs/models/NB/NB_001A/{threshold, prcurve, roc}.png
#   02_outputs/models/NB/NB_001A/diagnostics.rds
#   02_outputs/models/NB/NB_001B/  
#   02_outputs/model_registry.csv  
#
#==============================================================================
# nolint start
# -- 0. Paquetes -------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,
  e1071,   # naiveBayes()
  caret,   # confusionMatrix()
  pROC,    # roc(), auc()
  fs       # dir_create()
)

dir_processed   <- "00_data/processed"
dir_outputs_nb  <- "02_outputs/models/NB"
dir_submissions <- "03_submissions"
registry_path   <- "02_outputs/model_registry.csv"
set.seed(42)
# -- 1. Carga de datos -------------------------------------------------------
message("== Cargando datos ==")
 
train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id
y_train <- train$pobre   # vector numérico 0/1 — lm() necesita numérico

AUTOR   <- "Natalia"   
K_FOLDS <- 5

# =============================================================================
# SECCIÓN 1: IDENTIFICACIÓN DE LOS DOS MODELOS
#Tomados de LPM.R 
# Mismas listas que LPM_002A y LPM_002B.
# Adicionalmente, en ambas listas se excluyen ciudad y dpto por alta
# cardinalidad (problema específico de NB, no del LPM).
VARS_EXCLUIR_A <- c(
  "id", "pobre",
  # Alias del LPM_002A
  "zona", "bogota", "doble_ingreso", "brecha_educ",
  "indice_formalidad", "vulnerabilidad_lab", "n_fuentes_no_lab",
  "prop_contributivo", "costa_caribe", "region_pacifico", "eje_cafetero",
  "n_privaciones",
  # NA en todo el test
  "estrato_hog", "estrato_bajo", "estrato_x_zona",
  # Alta cardinalidad — problemáticas para NB
  "ciudad", "dpto"
)

 
VARS_EXCLUIR_B <- c(
  "id", "pobre",
  # Alias del LPM_002B
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
  # NA en todo el test
  "estrato_hog", "estrato_bajo", "estrato_x_zona"
)

# =============================================================================
# SECCIÓN 2: GENERAR FOLDS DE CV

fold_ids <- sample(rep(1:K_FOLDS, length.out = nrow(train)))
message("\nFolds de CV generados (semilla 42) — mismos para NB_001A y NB_001B.")
 
# SECCIÓN 3: FUNCIÓN run_nb()
# =============================================================================
run_nb <- function(model_id, notas, vars_excluir) {
 
  message("\n", strrep("═", 62))
  message("  INICIANDO MODELO: ", model_id)
 
  dir_model <- file.path(dir_outputs_nb, model_id)

  # ---------------------------------------------------------------------------
  # PASO 1: MATRICES X
  X_train_raw <- train |> select(-any_of(vars_excluir))
  X_test_raw  <- test  |> select(-any_of(vars_excluir))
 
  # Alinear columnas train/test
  cols_comunes <- intersect(names(X_train_raw), names(X_test_raw))
  solo_train   <- setdiff(names(X_train_raw), names(X_test_raw))
  if (length(solo_train) > 0)
    message("  Solo en train (se eliminan): ", paste(solo_train, collapse = ", "))
 
  X_train_raw <- X_train_raw |> select(all_of(cols_comunes))
  X_test_raw  <- X_test_raw  |> select(all_of(cols_comunes))
 
  message("\n━━━ Matrices preparadas (", model_id, ") ━━━")
  message("Predictores: ", ncol(X_train_raw))
 
  # Función de preprocesamiento
  # ref_medians = NULL → calcula medianas desde df (usar en train)
  # ref_medians = lista → aplica medianas externas (usar en val y test)
  preprocess_nb <- function(df, ref_medians = NULL) {
 
    # zona: character "1"/"2" → numérico
    if ("zona" %in% names(df))
      df <- df |> mutate(zona = as.numeric(as.character(zona)))
 
    # character restante → factor
    df <- df |> mutate(across(where(is.character), as.factor))
 
    # Calcular medianas de referencia si no se pasaron
    if (is.null(ref_medians)) {
      ref_medians <- df |>
        summarise(across(where(is.numeric),
                         ~ median(., na.rm = TRUE))) |>
        as.list()
    }
 
    # Imputar NAs con mediana de referencia
    for (col in names(ref_medians)) {
      if (col %in% names(df) && anyNA(df[[col]])) {
        df[[col]][is.na(df[[col]])] <- ref_medians[[col]]
      }
    }
 
    list(data = df, medians = ref_medians)
  }
 
  # ---------------------------------------------------------------------------
  # PASO 2: CROSS-VALIDATION
  # ---------------------------------------------------------------------------
  oof_probs <- rep(NA_real_, nrow(X_train_raw))
 
  message("\n━━━ Cross-Validation (", K_FOLDS, " folds) — ", model_id, " ━━━")
 
  for (k in seq_len(K_FOLDS)) {
 
    idx_val   <- which(fold_ids == k)
    idx_train <- which(fold_ids != k)
 
    # Preprocesar fold train → obtiene medianas
    prep_tr  <- preprocess_nb(X_train_raw[idx_train, ])
    df_tr    <- prep_tr$data
    medianas <- prep_tr$medians
 
    # Preprocesar fold val con medianas del train (sin data leakage)
    df_vl <- preprocess_nb(X_train_raw[idx_val, ],
                            ref_medians = medianas)$data
 
    y_tr_k <- factor(y_train[idx_train], levels = c(0, 1))
 
    modelo_k <- naiveBayes(x = df_tr, y = y_tr_k, laplace = 0)
 
    # predict type="raw" → columna "1" = P(pobre=1|X)
    prob_mat_k          <- predict(modelo_k, newdata = df_vl, type = "raw")
    oof_probs[idx_val]  <- prob_mat_k[, "1"]
 
    message(sprintf("  Fold %d/%d: P(pobre) en [%.3f, %.3f]",
                    k, K_FOLDS,
                    min(prob_mat_k[, "1"], na.rm = TRUE),
                    max(prob_mat_k[, "1"], na.rm = TRUE)))
  }
 
  if (any(is.na(oof_probs))) stop("NAs en oof_probs — revisar loop CV")
 
  cat(sprintf("\n━━━ Diagnóstico predicciones OOF — %s ━━━\n", model_id))
  print(summary(oof_probs))
 
  oof_probs_cl <- pmax(pmin(oof_probs, 1), 0)
 
  # ---------------------------------------------------------------------------
  # PASO 3: OPTIMIZACIÓN DEL THRESHOLD
  # ---------------------------------------------------------------------------
  th_grid <- seq(0.05, 0.95, by = 0.01)
 
  th_results <- map_dfr(th_grid, function(th) {
    pred_class <- factor(if_else(oof_probs_cl >= th, "Pobre", "NoPobre"),
                         levels = c("NoPobre", "Pobre"))
    obs_class  <- factor(if_else(y_train == 1,       "Pobre", "NoPobre"),
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
 
  cat(sprintf("\n━━━ Threshold óptimo — %s ━━━\n", model_id))
  print(best_th_row)
 
  p_threshold <- th_results |>
    filter(!is.na(F1)) |>
    pivot_longer(c(F1, Precision, Recall), names_to = "metrica") |>
    ggplot(aes(threshold, value, color = metrica)) +
    geom_line(linewidth = 0.9) +
    geom_vline(xintercept = THRESHOLD_OPT, linetype = "dashed", color = "gray30") +
    geom_vline(xintercept = 0.50,          linetype = "dotted", color = "gray60") +
    annotate("text", x = THRESHOLD_OPT + 0.02, y = 0.08,
             label = sprintf("th* = %.2f\n(max F1)", THRESHOLD_OPT),
             hjust = 0, size = 3.2, color = "gray25") +
    scale_color_manual(
      values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3"),
      labels = c(F1 = "F1 (balance)",
                 Precision = "Precision (↑ = menos filtración)",
                 Recall    = "Recall (↑ = menos exclusión)")
    ) +
    labs(title    = paste0("Naive Bayes: métricas por threshold — ", model_id),
         subtitle = sprintf("%d-fold CV OOF | n = %d", K_FOLDS, nrow(X_train_raw)),
         x = "Threshold", y = "Métrica", color = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top")
 
  ggsave(file.path(dir_model, "threshold.png"),
         p_threshold, width = 8, height = 5, dpi = 150)
  message("Gráfico guardado: ", model_id, "/threshold.png")
 
  # ---------------------------------------------------------------------------
  # PASO 4: MODELO FINAL
  # ---------------------------------------------------------------------------
  message("\n━━━ Ajustando modelo final — ", model_id, " ━━━")
 
  prep_full     <- preprocess_nb(X_train_raw)
  X_train_f     <- prep_full$data
  medianas_full <- prep_full$medians
 
  model_final <- naiveBayes(
    x       = X_train_f,
    y       = factor(y_train, levels = c(0, 1)),
    laplace = 0
  )
 
  message("  Entrenado sobre ", nrow(X_train_f), " hogares | ",
          ncol(X_train_f), " predictores")
 
  # ---------------------------------------------------------------------------
  # PASO 5: MÉTRICAS Y GRÁFICOS
  # ---------------------------------------------------------------------------
 
  # Confusion Matrix
  oof_clase <- factor(if_else(oof_probs_cl >= THRESHOLD, "Pobre", "NoPobre"),
                      levels = c("NoPobre", "Pobre"))
  obs_clase <- factor(if_else(y_train == 1, "Pobre", "NoPobre"),
                      levels = c("NoPobre", "Pobre"))
  cm_cv <- confusionMatrix(oof_clase, obs_clase, positive = "Pobre")
 
  cat(sprintf("\n━━━ Confusion Matrix (threshold = %.2f) — %s ━━━\n",
              THRESHOLD, model_id))
  print(cm_cv)
 
  cv_F1        <- cm_cv$byClass["F1"]
  cv_Precision <- cm_cv$byClass["Precision"]
  cv_Recall    <- cm_cv$byClass["Recall"]
 
  cat(sprintf("\nF1: %.4f | Precision: %.4f | Recall: %.4f\n",
              cv_F1, cv_Precision, cv_Recall))
 
  FN <- cm_cv$table["NoPobre", "Pobre"]
  FP <- cm_cv$table["Pobre",   "NoPobre"]
  cat(sprintf("Exclusión  (FN): %d hogares pobres excluidos (%.1f%%)\n",
              FN, FN / sum(y_train == 1) * 100))
  cat(sprintf("Filtración (FP): %d no-pobres con beneficio indebido (%.1f%%)\n",
              FP, FP / sum(y_train == 0) * 100))
 
  # ROC y AUC
  roc_obj <- pROC::roc(response = y_train, predictor = oof_probs_cl, quiet = TRUE)
  auc_roc <- as.numeric(pROC::auc(roc_obj))
 
  roc_df <- tibble(fpr = 1 - roc_obj$specificities,
                   tpr = roc_obj$sensitivities)
 
  p_roc <- ggplot(roc_df, aes(fpr, tpr)) +
    geom_line(color = "#534AB7", linewidth = 0.9) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
    annotate("text", x = 0.55, y = 0.15,
             label = sprintf("AUC-ROC = %.4f", auc_roc),
             size = 3.5, color = "#534AB7") +
    scale_x_continuous(labels = scales::percent_format()) +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(title    = paste0("Naive Bayes: curva ROC — ", model_id),
         subtitle = "Predicciones CV out-of-fold",
         x = "Tasa de Falsos Positivos (1 − Specificity)",
         y = "Tasa de Verdaderos Positivos (Recall)") +
    coord_equal() +
    theme_minimal(base_size = 12)
 
  ggsave(file.path(dir_model, "roc.png"), p_roc, width = 6, height = 6, dpi = 150)
  message("Gráfico guardado: ", model_id, "/roc.png")
 
  # Curva Precision-Recall
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
         subtitle = sprintf("AUC-ROC: %.4f | CV OOF", auc_roc),
         x = "Recall", y = "Precision") +
    theme_minimal(base_size = 12)
 
  ggsave(file.path(dir_model, "prcurve.png"),
         p_prcurve, width = 6, height = 5, dpi = 150)
  message("Gráfico guardado: ", model_id, "/prcurve.png")
 
  # ---------------------------------------------------------------------------
  # PASO 6: PREDICCIÓN EN TEST Y SUBMISSION
  # ---------------------------------------------------------------------------
  X_test_f <- preprocess_nb(X_test_raw,
                              ref_medians = medianas_full)$data
 
  prob_mat_test <- predict(model_final, newdata = X_test_f, type = "raw")
  probs_test    <- prob_mat_test[, "1"]
  preds_test    <- as.integer(probs_test >= THRESHOLD)
 
  cat(sprintf("\n━━━ Predicciones test — %s ━━━\n", model_id))
  cat(sprintf("%d hogares predichos como pobres (%.1f%%) | threshold = %.2f\n",
              sum(preds_test), mean(preds_test) * 100, THRESHOLD))
 
  submission <- tibble(id = test$id, pobre = preds_test)
  submission_file <- file.path(dir_submissions,
                                paste0("submission_", model_id, ".csv"))
  write_csv(submission, submission_file)
  message("Submission guardada: ", submission_file)
  message("→ Subir a Kaggle y registrar public F1 en model_registry.csv")
 
  # ---------------------------------------------------------------------------
  # PASO 7: DIAGNÓSTICOS Y REGISTRO
  # ---------------------------------------------------------------------------
  saveRDS(
    list(model_id = model_id, autor = AUTOR, notas = notas,
         fecha = Sys.Date(), vars_excluir = vars_excluir,
         model_final = model_final, medianas_full = medianas_full,
         oof_probs = oof_probs, oof_probs_cl = oof_probs_cl,
         oof_actuals = y_train, fold_ids = fold_ids,
         th_results = th_results, threshold_opt = THRESHOLD_OPT,
         threshold_usado = THRESHOLD, cm_cv = cm_cv,
         cv_F1 = cv_F1, cv_Precision = cv_Precision, cv_Recall = cv_Recall,
         auc_roc = auc_roc, predictores = names(X_train_f),
         n_features = ncol(X_train_f)),
    file.path(dir_model, "diagnostics.rds")
  )
  message("Diagnósticos guardados: ", model_id, "/diagnostics.rds")
 
  nueva_fila <- tibble(
    model_id           = model_id,
    fecha              = Sys.Date(),
    autor              = AUTOR,
    algoritmo          = "NaiveBayes",
    n_features         = ncol(X_train_f),
    imbalance_strategy = "none",
    cv_folds           = K_FOLDS,
    cv_F1              = round(cv_F1,        4),
    cv_Precision       = round(cv_Precision, 4),
    cv_Recall          = round(cv_Recall,    4),
    auc_roc            = round(auc_roc,      4),
    kaggle_public_F1   = NA_real_,
    threshold          = THRESHOLD,
    notas              = notas
  )
 
  registry <- if (file.exists(registry_path)) {
    bind_rows(read_csv(registry_path, show_col_types = FALSE), nueva_fila)
  } else {
    nueva_fila
  }
  write_csv(registry, registry_path)
  message("Registro actualizado: ", registry_path)
 
  message("\n", strrep("═", 62))
  message("  ", model_id, " completado ✓")
  message(strrep("═", 62), "\n")
 
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
# SECCIÓN 6: EJECUTAR AMBOS MODELOS
# =============================================================================
result_A <- run_nb(
  model_id     = "NB_001A",
  notas        = "Sin índices compuestos — misma especificación que LPM_002A. usekernel=FALSE",
  vars_excluir = VARS_EXCLUIR_A
)
 
result_B <- run_nb(
  model_id     = "NB_001B",
  notas        = "Con índices compuestos — misma especificación que LPM_002B. usekernel=FALSE",
  vars_excluir = VARS_EXCLUIR_B
)
 
# =============================================================================
# SECCIÓN 7: COMPARACIÓN FINAL
# =============================================================================
comparacion <- bind_rows(result_A, result_B)
 
cat("\n", strrep("═", 62), "\n")
cat("  COMPARACIÓN: NB_001A vs NB_001B\n")
cat(strrep("═", 62), "\n\n")
print(comparacion, n = Inf)
 
ganador <- comparacion |> slice_max(cv_F1, n = 1, with_ties = FALSE)
cat(sprintf("\n Mejor modelo: %s (F1 = %.4f | AUC = %.4f | %d features)\n",
            ganador$model_id, ganador$cv_F1, ganador$auc_roc, ganador$n_features))
 
diferencia <- abs(comparacion$cv_F1[1] - comparacion$cv_F1[2])
cat(sprintf(" Diferencia en F1-CV: %.4f\n", diferencia))
if (diferencia < 0.005) {
  cat("Diferencia < 0.005: preferir el más parsimonioso —",
      comparacion |> slice_min(n_features, n=1) |> pull(model_id), "\n")
}
 