#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/ElasticNet/ElasticNet_features.R
#==============================================================================
# ALGORITMO : Elastic Net (glmnet) -- Regresion logistica penalizada
# PARTE     : 3 -- Feature selection usando ranking de importancia del RF_001
#
# QUE HACE ESTE SCRIPT:
#   Reentrena Elastic Net usando subconjuntos de variables seleccionadas por
#   el ranking de importancia del Random Forest (permutacion). Se evaluan
#   tres subconjuntos: top 20, top 40 y top 60 variables.
#   Responde: menos variables mejoran la prediccion?
#
# FUENTE DEL RANKING:
#   02_outputs/models/RandomForest/RF_001/feature_matrix.csv
#   Generado por Jonathan con RF_001 (ranger, mtry=50, 150 arboles).
#   La importancia se mide por permutacion: cuanto cae el desempeno
#   del modelo al aleatorizar cada variable.
#
# HIPERPARAMETROS: misma grilla que EN_001A/B
#   alpha: 0, 0.25, 0.5, 0.75, 1
#   lambda: 10^(-4) a 10^(1), 20 valores
#   Threshold: optimizado sobre OOF maximizando F1
#
# OUTPUT:
#   03_submissions/submission_EN_RF_top20.csv
#   03_submissions/submission_EN_RF_top40.csv
#   03_submissions/submission_EN_RF_top60.csv
#   02_outputs/models/EN/EN_RF_topNN/ (metricas, diagnostics.rds)
#   02_outputs/model_registry.csv (append)
#==============================================================================
# nolint start
# -- 0. Paquetes -------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,
  glmnet,
  caret,
  MLmetrics,
  pROC,
  fs
)

# -- Rutas -------------------------------------------------------------------
registry_path   <- "02_outputs/model_registry.csv"
dir_submissions <- "03_submissions"
feature_path    <- "02_outputs/models/RandomForest/RF_006/feature_matrix.csv"
dir_create(dir_submissions, recurse = TRUE)

# -- 1. Carga de datos -------------------------------------------------------
message("== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id

# -- 2. Cargar ranking de importancia del RF ---------------------------------
message("== Cargando ranking de variables RF_001 ==")

feat_rank <- read_csv(feature_path, show_col_types = FALSE) |>
  arrange(rank)

cat("\n  Top 20 variables por importancia RF:\n")
print(feat_rank[1:20, c("variable", "importancia", "rank")])

top20_vars <- feat_rank$variable[1:20]
top40_vars <- feat_rank$variable[1:40]
top60_vars <- feat_rank$variable[1:60]

# -- 3. Preprocesamiento -----------------------------------------------------
# glmnet necesita numerico sin NAs. Se imputa con mediana.
preparar_X <- function(df, vars_sel) {
  df |>
    select(any_of(vars_sel)) |>
    mutate(across(where(is.character), as.numeric)) |>
    mutate(across(where(is.factor), as.numeric)) |>
    mutate(across(everything(), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
}

niveles     <- c("No", "Yes")
y_train_fct <- factor(ifelse(train$pobre == 1, "Yes", "No"), levels = niveles)

# -- 4. trainControl ---------------------------------------------------------
ctrl_cv <- trainControl(
  method          = "cv",
  number          = 5,
  summaryFunction = twoClassSummary,
  classProbs      = TRUE,
  verboseIter     = TRUE,
  savePredictions = "final"
)

# -- 5. Grilla ---------------------------------------------------------------
tune_grid <- expand.grid(
  alpha  = c(0, 0.25, 0.5, 0.75, 1),
  lambda = 10^seq(-4, 1, length.out = 20)
)

# -- 6. Optimizar threshold --------------------------------------------------
optimizar_threshold <- function(modelo) {
  oof <- modelo$pred |>
    filter(alpha == modelo$bestTune$alpha,
           lambda == modelo$bestTune$lambda)

  thresholds <- seq(0.10, 0.60, by = 0.01)

  resultados <- map_dfr(thresholds, function(th) {
    pred_clase <- ifelse(oof$Yes >= th, "Yes", "No")
    f1 <- tryCatch({
      MLmetrics::F1_Score(
        y_true   = as.character(oof$obs),
        y_pred   = pred_clase,
        positive = "Yes"
      )
    }, error = function(e) NA_real_)
    tibble(threshold = th, F1 = f1)
  })

  mejor <- resultados |> filter(F1 == max(F1, na.rm = TRUE)) |> slice(1)
  list(threshold = mejor$threshold, F1 = mejor$F1, todos = resultados)
}

# -- 7. Funcion principal ----------------------------------------------------
correr_en_feat <- function(vars_sel, model_id, notas) {

  dir_model <- file.path("02_outputs/models/EN", model_id)
  dir_create(dir_model, recurse = TRUE)

  X_train <- preparar_X(train, vars_sel)
  X_test  <- preparar_X(test,  vars_sel)

  message("\n========================================")
  message(model_id, " -- Elastic Net (feature selection RF)")
  message("========================================")
  message("  Predictores: ", ncol(X_train))

  # --- Entrenar ---
  set.seed(42)
  t0 <- Sys.time()

  modelo <- train(
    x          = as.data.frame(X_train),
    y          = y_train_fct,
    method     = "glmnet",
    family     = "binomial",
    trControl  = ctrl_cv,
    tuneGrid   = tune_grid,
    metric     = "ROC",
    preProcess = c("center", "scale")
  )

  t1 <- Sys.time()
  message("  Tiempo CV: ",
          round(as.numeric(difftime(t1, t0, units = "mins")), 1), " min")

  cat("\n-- Mejor modelo --\n")
  cat("  alpha:  ", modelo$bestTune$alpha, "\n")
  cat("  lambda: ", modelo$bestTune$lambda, "\n")
  cat("  ROC CV: ", round(max(modelo$results$ROC, na.rm = TRUE), 4), "\n")

  # --- Threshold ---
  opt <- optimizar_threshold(modelo)
  THRESHOLD <- opt$threshold
  cat("  Threshold optimo: ", THRESHOLD, "\n")
  cat("  F1 OOF: ", round(opt$F1, 4), "\n")

  # --- Metricas OOF ---
  oof <- modelo$pred |>
    filter(alpha == modelo$bestTune$alpha,
           lambda == modelo$bestTune$lambda)
  oof_probs <- oof$Yes
  oof_obs   <- oof$obs

  oof_clase <- factor(ifelse(oof_probs >= THRESHOLD, "Yes", "No"), levels = niveles)
  obs_clase <- factor(as.character(oof_obs), levels = niveles)

  cm_cv <- confusionMatrix(oof_clase, obs_clase, positive = "Yes")

  cv_F1        <- cm_cv$byClass["F1"]
  cv_Precision <- cm_cv$byClass["Precision"]
  cv_Recall    <- cm_cv$byClass["Recall"]

  cat("\n-- Metricas CV (OOF, threshold =", THRESHOLD, ") --\n")
  cat("  F1:        ", round(cv_F1,        4), "\n")
  cat("  Precision: ", round(cv_Precision, 4), "\n")
  cat("  Recall:    ", round(cv_Recall,    4), "\n")

  # --- AUC-ROC ---
  oof_obs_num <- ifelse(as.character(oof_obs) == "Yes", 1, 0)
  roc_obj <- pROC::roc(response = oof_obs_num, predictor = oof_probs, quiet = TRUE)
  auc_roc <- as.numeric(pROC::auc(roc_obj))
  cat("  AUC-ROC:   ", round(auc_roc, 4), "\n")

  # --- Costos de politica ---
  FN <- cm_cv$table["No", "Yes"]
  FP <- cm_cv$table["Yes", "No"]
  cat("\n-- Costos de politica --\n")
  cat("  Falsos negativos (exclusion):  ", FN, "\n")
  cat("  Falsos positivos (filtracion): ", FP, "\n")

  # --- Coeficientes no cero ---
  coefs <- coef(modelo$finalModel, s = modelo$bestTune$lambda)
  coefs_df <- data.frame(
    variable    = rownames(coefs),
    coeficiente = as.numeric(coefs)
  ) |>
    filter(coeficiente != 0, variable != "(Intercept)") |>
    arrange(desc(abs(coeficiente)))

  cat("\n-- Variables con coef != 0: ", nrow(coefs_df),
      " de ", ncol(X_train), " --\n")
  cat("-- Top 15 coeficientes --\n")
  print(head(coefs_df, 15))

  # --- Guardar metricas txt ---
  sink(file.path(dir_model, "metricas.txt"))
  cat("=== ELASTIC NET -- ", model_id, " ===\n\n")
  cat("Predictores:", ncol(X_train), "\n")
  cat("Variables usadas:", paste(names(X_train), collapse = ", "), "\n\n")
  cat("alpha:", modelo$bestTune$alpha, "\n")
  cat("lambda:", modelo$bestTune$lambda, "\n")
  cat("Threshold optimo:", THRESHOLD, "\n")
  cat("F1 OOF:", round(cv_F1, 4), "\n")
  cat("Precision OOF:", round(cv_Precision, 4), "\n")
  cat("Recall OOF:", round(cv_Recall, 4), "\n")
  cat("AUC-ROC:", round(auc_roc, 4), "\n\n")
  cat("-- Confusion Matrix (OOF) --\n")
  print(cm_cv)
  cat("\n-- Coeficientes no cero --\n")
  print(coefs_df)
  sink()

  # --- Guardar diagnostics.rds ---
  diagnostics <- list(
    model_id        = model_id,
    notas           = notas,
    fecha           = Sys.Date(),
    modelo          = modelo,
    bestTune        = modelo$bestTune,
    oof_probs       = oof_probs,
    oof_obs         = oof_obs,
    threshold_opt   = THRESHOLD,
    th_results      = opt$todos,
    cm_cv           = cm_cv,
    cv_F1           = cv_F1,
    cv_Precision    = cv_Precision,
    cv_Recall       = cv_Recall,
    auc_roc         = auc_roc,
    coeficientes    = coefs_df,
    predictores     = names(X_train),
    n_features      = ncol(X_train)
  )
  saveRDS(diagnostics, file.path(dir_model, "diagnostics.rds"))
  message("Diagnosticos guardados: ", model_id, "/diagnostics.rds")

  # --- Submission ---
  prob_test  <- predict(modelo, newdata = as.data.frame(X_test), type = "prob")$Yes
  pred_test  <- as.integer(prob_test >= THRESHOLD)

  submission <- tibble(id = ids, pobre = pred_test)
  submission_file <- file.path(dir_submissions,
                                paste0("submission_", model_id, ".csv"))
  write_csv(submission, submission_file)
  message("Submission guardada: ", submission_file)
  message("  Pobres predichos: ", sum(submission$pobre),
          " (", round(mean(submission$pobre) * 100, 1), "%)")

  # --- Registrar en model_registry.csv ---
  nueva_fila <- tibble(
    model_id           = model_id,
    fecha              = Sys.Date(),
    autor              = "Dani",
    algoritmo          = "ElasticNet",
    n_features         = ncol(X_train),
    imbalance_strategy = "none",
    cv_folds           = 5,
    cv_F1              = round(cv_F1,        4),
    cv_Precision       = round(cv_Precision, 4),
    cv_Recall          = round(cv_Recall,    4),
    auc_roc            = round(auc_roc,      4),
    kaggle_public_F1   = NA_real_,
    threshold          = THRESHOLD,
    notas              = notas
  )

  if (file.exists(registry_path)) {
    registry <- read_csv(registry_path, show_col_types = FALSE)
    registry <- bind_rows(registry, nueva_fila)
  } else {
    registry <- nueva_fila
  }
  write_csv(registry, registry_path)
  message("Registro actualizado: ", registry_path)

  cat("\n-- Fila registrada en model_registry.csv --\n")
  print(nueva_fila)

  invisible(modelo)
}


# =============================================================================
# CORRER LAS TRES ESPECIFICACIONES
# =============================================================================

en_top20 <- correr_en_feat(
  vars_sel = top20_vars,
  model_id = "EN_RF_top20",
  notas    = "EN top-20 vars RF_001. glmnet binomial, center+scale. Threshold optimizado OOF."
)

en_top40 <- correr_en_feat(
  vars_sel = top40_vars,
  model_id = "EN_RF_top40",
  notas    = "EN top-40 vars RF_001. glmnet binomial, center+scale. Threshold optimizado OOF."
)

en_top60 <- correr_en_feat(
  vars_sel = top60_vars,
  model_id = "EN_RF_top60",
  notas    = "EN top-60 vars RF_001. glmnet binomial, center+scale. Threshold optimizado OOF."
)


# =============================================================================
# RESUMEN COMPARATIVO
# =============================================================================
cat("\n========================================\n")
cat("RESUMEN -- Elastic Net Feature Selection\n")
cat("========================================\n")
cat("Comparacion con baseline EN_001A (93 vars, F1 CV = 0.695):\n\n")

cat("  EN_RF_top20: ver metricas arriba\n")
cat("  EN_RF_top40: ver metricas arriba\n")
cat("  EN_RF_top60: ver metricas arriba\n")

cat("\nSubmissions generadas:\n")
cat("  03_submissions/submission_EN_RF_top20.csv\n")
cat("  03_submissions/submission_EN_RF_top40.csv\n")
cat("  03_submissions/submission_EN_RF_top60.csv\n")

cat("\nPROXIMO PASO: subir a Kaggle y comparar F1 vs baseline.\n")
cat("El mejor de estos 4 (baseline + top20/40/60) pasa a Parte 4 (balanceo).\n")
message("\nElastic Net Feature Selection completo!")
# nolint end
