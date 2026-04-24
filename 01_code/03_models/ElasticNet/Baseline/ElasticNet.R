#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY

#==============================================================================
# ALGORITMO : Elastic Net (glmnet) -- Regresion logistica penalizada
#
# QUE HACE ESTE SCRIPT:
#   Entrena un modelo de regresion logistica con penalizacion Elastic Net.
#   Es una combinacion de Lasso (seleccion de variables) y Ridge (manejo de
#   multicolinealidad). El modelo estima P(pobre|X) como funcion lineal de
#   los predictores, pasada por una funcion logistica para mantenerla en [0,1].
#
# DIFERENCIA CON LPM:
#   - LPM usa OLS y puede predecir probabilidades fuera de [0,1]
#   - Elastic Net usa Logit (probabilidades siempre en [0,1]) y ademas
#     penaliza los coeficientes para seleccionar variables automaticamente
#
# HIPERPARAMETROS:
#   alpha: mezcla entre Ridge (0) y Lasso (1)
#     - alpha = 0: Ridge puro -- reduce coeficientes pero no los elimina
#     - alpha = 1: Lasso puro -- puede poner coeficientes en cero (seleccion)
#     - 0 < alpha < 1: Elastic Net -- combina ambas propiedades
#     Rango a explorar: 0, 0.25, 0.5, 0.75, 1
#
#   lambda: intensidad de la penalizacion
#     - lambda grande = modelo mas simple (mas coeficientes eliminados/reducidos)
#     - lambda pequeno = poca penalizacion (se parece a Logit sin penalizacion)
#     Rango a explorar: 10^(-4) a 10^(1), 20 valores en escala logaritmica
#
# PREPROCESAMIENTO:
#   glmnet requiere que todas las variables sean numericas y sin NAs.
#   Se imputan NAs con la mediana y se estandarizan (center + scale).
#
# DOS ESPECIFICACIONES (misma logica que LPM y NB):
#   EN_001A -- Sin indices compuestos (conserva componentes individuales)
#   EN_001B -- Con indices compuestos (elimina componentes, conserva indices)
#
# OPTIMIZACION DEL THRESHOLD:
#   Con desbalance 80/20, el threshold optimo no es 0.5.
#   Se optimiza sobre predicciones out-of-fold del CV.
#
# OUTPUT:
#   03_submissions/submission_EN_001A.csv
#   03_submissions/submission_EN_001B.csv
#   02_outputs/models/EN/EN_001A/ (metricas, coeficientes, diagnostics.rds)
#   02_outputs/models/EN/EN_001B/
#   02_outputs/model_registry.csv (append)
#==============================================================================
# nolint start
# -- 0. Paquetes -------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,
  glmnet,       # Elastic Net
  caret,        # trainControl, train, confusionMatrix
  MLmetrics,    # F1_Score
  pROC,         # roc, auc
  fs            # dir_create
)

# -- Rutas -------------------------------------------------------------------
registry_path   <- "02_outputs/model_registry.csv"
dir_submissions <- "03_submissions"
dir_create(dir_submissions, recurse = TRUE)

# -- 1. Carga de datos -------------------------------------------------------
message("== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id

# -- 2. Preprocesamiento -----------------------------------------------------
# glmnet necesita: todo numerico, sin NAs
message("\n== Preprocesando variables ==")

preparar_X_glmnet <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>
    mutate(zona = as.numeric(zona)) |>
    select(-any_of(c("ciudad", "dpto"))) |>
    mutate(across(where(is.character), as.numeric)) |>
    mutate(across(everything(), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
}

X_train_all <- preparar_X_glmnet(train)
X_test_all  <- preparar_X_glmnet(test)
y_train_num <- train$pobre
niveles     <- c("No", "Yes")
y_train_fct <- factor(ifelse(train$pobre == 1, "Yes", "No"), levels = niveles)

message("  Predictores totales: ", ncol(X_train_all))
message("  NAs restantes train: ", sum(is.na(X_train_all)))

# -- 3. Definir las dos especificaciones -------------------------------------
vars_indices <- c(
  "indice_formalidad", "formal_estricto", "vulnerabilidad_lab",
  "proteccion_social", "desprotegido", "n_fuentes_no_lab",
  "tiene_ingreso_no_lab", "n_privaciones", "multi_privado",
  "privado_severo", "dep_educacion", "dep_vivienda",
  "dep_empleo_formal", "dep_proteccion", "dep_dependencia"
)

vars_componentes <- c(
  "zona", "costa_caribe", "region_pacifico", "eje_cafetero",
  "rural_costa_caribe", "rural_x_cta_propia", "educ_x_rural",
  "prop_contributivo", "prop_cotiza_pension", "prop_prima_serv",
  "prop_asalariado", "prop_sub_transp", "prop_sub_alim",
  "alguno_pension_jub", "alguno_remesas", "alguno_subsidio",
  "alguno_transf_nac", "alguno_intereses", "alguno_cesantias",
  "alguno_agropec", "alguno_arriendos", "solo_subsidio",
  "ingresos_activos", "hacinamiento", "hacinamiento_dorm",
  "tenencia_insegura", "vivienda_propia_pag", "ratio_depend",
  "prop_menores", "prop_adultos_may_p", "menores_por_ocupado",
  "sin_ocupados", "un_solo_ocupado", "doble_ingreso",
  "educ_media_o_mas", "educ_superior_jefe", "brecha_educ",
  "jefe_mayor_pension"
)

# -- 4. trainControl ---------------------------------------------------------
ctrl_cv <- trainControl(
  method          = "cv",
  number          = 5,
  summaryFunction = twoClassSummary,
  classProbs      = TRUE,
  verboseIter     = TRUE,
  savePredictions = "final"
)

# -- 5. Grilla de hiperparametros --------------------------------------------
tune_grid <- expand.grid(
  alpha  = c(0, 0.25, 0.5, 0.75, 1),
  lambda = 10^seq(-4, 1, length.out = 20)
)

message("  Grilla: ", nrow(tune_grid), " combinaciones (alpha x lambda)")

# -- 6. Funcion para optimizar threshold sobre OOF ---------------------------
optimizar_threshold <- function(modelo, y_real_num) {
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

# -- 7. Funcion para correr una especificacion completa ----------------------
correr_elastic_net <- function(X_train, X_test, y_fct, y_num, model_id, notas) {

  dir_model <- file.path("02_outputs/models/EN/Baseline", model_id)
  dir_create(dir_model, recurse = TRUE)

  message("\n========================================")
  message(model_id, " -- Elastic Net")
  message("========================================")
  message("  Predictores: ", ncol(X_train))

  # --- Entrenar con caret ---
  set.seed(42)
  t0 <- Sys.time()

  modelo <- train(
    x          = as.data.frame(X_train),
    y          = y_fct,
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

  # --- Optimizar threshold ---
  opt <- optimizar_threshold(modelo, y_num)
  THRESHOLD <- opt$threshold
  cat("  Threshold optimo: ", THRESHOLD, "\n")
  cat("  F1 con threshold optimo (OOF): ", round(opt$F1, 4), "\n")

  # --- Predicciones OOF con threshold optimo ---
  oof <- modelo$pred |>
    filter(alpha == modelo$bestTune$alpha,
           lambda == modelo$bestTune$lambda)

  oof_probs <- oof$Yes
  oof_obs   <- oof$obs

  oof_clase <- factor(
    ifelse(oof_probs >= THRESHOLD, "Yes", "No"),
    levels = niveles
  )
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
  cat("  Falsos negativos (exclusion):  ", FN, " hogares pobres excluidos\n")
  cat("  Falsos positivos (filtracion): ", FP, " hogares no-pobres con beneficio\n")
  cat("  Tasa exclusion:  ",
      round(FN / sum(oof_obs_num == 1) * 100, 1), "% de los pobres reales\n")
  cat("  Tasa filtracion: ",
      round(FP / sum(oof_obs_num == 0) * 100, 1), "% de los no-pobres reales\n")

  # --- Coeficientes no cero ---
  coefs <- coef(modelo$finalModel, s = modelo$bestTune$lambda)
  coefs_df <- data.frame(
    variable    = rownames(coefs),
    coeficiente = as.numeric(coefs)
  ) |>
    filter(coeficiente != 0, variable != "(Intercept)") |>
    arrange(desc(abs(coeficiente)))

  cat("\n-- Variables seleccionadas (coef != 0): ", nrow(coefs_df),
      " de ", ncol(X_train), " --\n")
  cat("-- Top 20 coeficientes --\n")
  print(head(coefs_df, 20))

  # --- Guardar metricas en txt ---
  sink(file.path(dir_model, "metricas.txt"))
  cat("=== ELASTIC NET -- ", model_id, " ===\n\n")
  cat("Predictores:", ncol(X_train), "\n")
  cat("alpha:", modelo$bestTune$alpha, "\n")
  cat("lambda:", modelo$bestTune$lambda, "\n")
  cat("Threshold optimo:", THRESHOLD, "\n")
  cat("F1 OOF:", round(cv_F1, 4), "\n")
  cat("Precision OOF:", round(cv_Precision, 4), "\n")
  cat("Recall OOF:", round(cv_Recall, 4), "\n")
  cat("AUC-ROC:", round(auc_roc, 4), "\n\n")
  cat("-- Confusion Matrix (OOF) --\n")
  print(cm_cv)
  cat("\n-- Todos los coeficientes no cero --\n")
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
  message("-> Subir a Kaggle y anotar el public F1 en model_registry.csv")

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
# CORRER LAS DOS ESPECIFICACIONES
# =============================================================================

# --- Spec A: sin indices compuestos ---
X_train_A <- X_train_all |> select(-any_of(vars_indices))
X_test_A  <- X_test_all  |> select(-any_of(vars_indices))

en_A <- correr_elastic_net(
  X_train  = X_train_A,
  X_test   = X_test_A,
  y_fct    = y_train_fct,
  y_num    = y_train_num,
  model_id = "EN_001A",
  notas    = "Sin indices compuestos -- misma especificacion que LPM_002A. glmnet binomial, center+scale."
)

# --- Spec B: con indices compuestos ---
X_train_B <- X_train_all |> select(-any_of(vars_componentes))
X_test_B  <- X_test_all  |> select(-any_of(vars_componentes))

en_B <- correr_elastic_net(
  X_train  = X_train_B,
  X_test   = X_test_B,
  y_fct    = y_train_fct,
  y_num    = y_train_num,
  model_id = "EN_001B",
  notas    = "Con indices compuestos -- misma especificacion que LPM_002B. glmnet binomial, center+scale."
)

# =============================================================================
# RESUMEN FINAL
# =============================================================================
cat("\n========================================\n")
cat("RESUMEN -- Elastic Net\n")
cat("========================================\n")
cat("Submissions generadas:\n")
cat("  03_submissions/submission_EN_001A.csv\n")
cat("  03_submissions/submission_EN_001B.csv\n")
cat("Metricas guardadas en:\n")
cat("  02_outputs/models/EN/EN_001A/\n")
cat("  02_outputs/models/EN/EN_001B/\n")
cat("Registro actualizado:\n")
cat("  02_outputs/model_registry.csv\n")
cat("\nPROXIMO PASO: subir los CSV a Kaggle y anotar kaggle_public_F1 en el registry.\n")
message("\nElastic Net completo!")
# nolint end
