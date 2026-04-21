# nolint start
#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/ElasticNet/Balance/ElasticNet_balance.R
#==============================================================================
# ALGORITMO : Elastic Net (glmnet) -- Balanceo de clases
# PARTE     : 4 -- Evaluar si el balanceo mejora el mejor modelo EN
#
# QUE HACE ESTE SCRIPT:
#   Toma el mejor modelo de Elastic Net (EN_001A, 93 vars, F1=0.695)
#   y compara 5 tecnicas de balanceo de clases:
#     (a) baseline       -- sin correccion, threshold optimizado
#     (b) class_weights  -- pesos inversamente proporcionales a la frecuencia
#     (c) downsample     -- reducir clase mayoritaria
#     (d) upsample       -- replicar clase minoritaria
#     (e) smote          -- generar observaciones sinteticas (k=5)
#
#   El balanceo se aplica SOLO al fold de entrenamiento.
#   El fold de validacion siempre refleja la distribucion real (~20% pobres).
#   Threshold se optimiza sobre predicciones OOF.
#
# VARIABLES: todas las disponibles (93 predictores, misma spec que EN_001A)
#
# OUTPUT:
#   02_outputs/models/EN/Balance/cv_balance_results.csv
#   02_outputs/models/EN/Balance/cv_balance_summary.csv
#   02_outputs/models/EN/Balance/metricas.txt
#   02_outputs/models/EN/Balance/diagnostics.rds
#   03_submissions/submission_EN_balance_*.csv
#   02_outputs/model_registry.csv (append)
#==============================================================================

# -- 0. Paquetes -------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,
  glmnet,
  caret,
  MLmetrics,
  pROC,
  themis,    # smote
  fs
)

# -- Configuracion -----------------------------------------------------------
K_FOLDS    <- 5L
TECHNIQUES <- c("baseline", "class_weights", "downsample", "upsample", "smote")

dir_model       <- "02_outputs/models/EN/Balance"
dir_submissions <- "03_submissions"
registry_path   <- "02_outputs/model_registry.csv"

dir_create(dir_model, recurse = TRUE)
dir_create(dir_submissions, recurse = TRUE)

set.seed(42)

# -- 1. Carga de datos -------------------------------------------------------
message("== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id
y_train <- train$pobre

pct_pobre <- round(mean(y_train) * 100, 1)
message("  Balance: ", pct_pobre, "% pobres / ",
        round(100 - pct_pobre, 1), "% no pobres")

# -- 2. Preprocesamiento (misma logica que EN_001A) --------------------------
message("\n== Preprocesando variables ==")

preparar_X <- function(df, ref_medians = NULL) {
  X <- df |>
    select(-any_of(c("id", "pobre"))) |>
    mutate(zona = as.numeric(zona)) |>
    select(-any_of(c("ciudad", "dpto"))) |>
    mutate(across(where(is.character), as.numeric)) |>
    mutate(across(where(is.factor), as.numeric))

  # Excluir variables con varianza cero
  X <- X |> select(-any_of(c(
    "indice_formalidad", "formal_estricto", "vulnerabilidad_lab",
    "proteccion_social", "desprotegido", "n_fuentes_no_lab",
    "tiene_ingreso_no_lab", "n_privaciones", "multi_privado",
    "privado_severo", "dep_educacion", "dep_vivienda",
    "dep_empleo_formal", "dep_proteccion", "dep_dependencia"
  )))

  # Imputar NAs
  if (is.null(ref_medians)) {
    medians <- map_dbl(X, ~ median(.x, na.rm = TRUE))
  } else {
    medians <- ref_medians
  }

  for (col in names(X)) {
    if (any(is.na(X[[col]]))) {
      X[[col]][is.na(X[[col]])] <- medians[col]
    }
  }

  list(data = X, medians = medians)
}

prep_train <- preparar_X(train)
X_train    <- prep_train$data
medians_ref <- prep_train$medians

prep_test <- preparar_X(test, ref_medians = medians_ref)
X_test    <- prep_test$data

message("  Predictores: ", ncol(X_train))

# -- 3. Funcion para buscar threshold optimo ---------------------------------
buscar_th <- function(probs, actuals) {
  thresholds <- seq(0.10, 0.60, by = 0.01)

  resultados <- map_dfr(thresholds, function(th) {
    pred <- ifelse(probs >= th, 1, 0)
    f1 <- tryCatch({
      MLmetrics::F1_Score(y_true = actuals, y_pred = pred, positive = "1")
    }, error = function(e) NA_real_)
    tibble(threshold = th, F1 = f1)
  })

  resultados |> filter(F1 == max(F1, na.rm = TRUE)) |> slice(1)
}

# -- 4. Folds estratificados -------------------------------------------------
message("\n== Creando folds estratificados ==")

fold_ids <- rep(0L, length(y_train))
idx_1 <- which(y_train == 1)
idx_0 <- which(y_train == 0)
fold_ids[idx_1] <- sample(rep(1:K_FOLDS, length.out = length(idx_1)))
fold_ids[idx_0] <- sample(rep(1:K_FOLDS, length.out = length(idx_0)))

for (k in 1:K_FOLDS)
  message(sprintf("  Fold %d: %.1f%% pobres", k,
                   mean(y_train[fold_ids == k] == 1) * 100))

# -- 5. CV-5 por tecnica de balanceo -----------------------------------------
message("\n========================================")
message("CV-5: comparando ", length(TECHNIQUES), " tecnicas de balanceo")
message("  Variables: ", ncol(X_train))
message("  Modelo: glmnet binomial, alpha/lambda optimizados por ROC")
message("========================================")

balance_resultados <- vector("list", K_FOLDS)
t0_bal <- Sys.time()

for (k in seq_len(K_FOLDS)) {
  message("\n-- Fold ", k, " / ", K_FOLDS, " --")

  idx_val   <- which(fold_ids == k)
  idx_train <- which(fold_ids != k)

  X_fold_tr_raw  <- X_train[idx_train, ]
  X_fold_val_raw <- X_train[idx_val, ]
  y_fold_tr      <- y_train[idx_train]
  y_fold_val     <- y_train[idx_val]

  # Preprocesar fold de entrenamiento
  prep_fold  <- preparar_X(train[idx_train, ])
  X_fold_tr  <- prep_fold$data
  med_fold   <- prep_fold$medians

  # Preprocesar fold de validacion con medianas del train
  X_fold_val <- preparar_X(train[idx_val, ], ref_medians = med_fold)$data

  fold_bal_res <- map_dfr(TECHNIQUES, function(tech) {
    seed_tech <- 42 * k + match(tech, TECHNIQUES) * 1000

    X_tr_use <- X_fold_tr
    y_tr_use <- y_fold_tr
    weights_use <- NULL

    if (tech == "class_weights") {
      # glmnet acepta weights: inverso de frecuencia
      n_yes <- sum(y_fold_tr == 1)
      n_no  <- sum(y_fold_tr == 0)
      w <- ifelse(y_fold_tr == 1, n_no / n_yes, 1)
      weights_use <- w
      message(sprintf("    [class_weights] peso pobre=%.2f peso no_pobre=1.00",
                       n_no / n_yes))

    } else if (tech == "downsample") {
      set.seed(seed_tech)
      ds <- caret::downSample(
        x = as.data.frame(X_fold_tr),
        y = factor(y_fold_tr, levels = c(0, 1)),
        yname = "pobre"
      )
      X_tr_use <- ds |> select(-pobre)
      y_tr_use <- as.numeric(as.character(ds$pobre))
      message(sprintf("    [downsample] %d obs | Yes=%d / No=%d",
                       nrow(X_tr_use), sum(y_tr_use == 1), sum(y_tr_use == 0)))

    } else if (tech == "upsample") {
      set.seed(seed_tech)
      us <- caret::upSample(
        x = as.data.frame(X_fold_tr),
        y = factor(y_fold_tr, levels = c(0, 1)),
        yname = "pobre"
      )
      X_tr_use <- us |> select(-pobre)
      y_tr_use <- as.numeric(as.character(us$pobre))
      message(sprintf("    [upsample] %d obs | Yes=%d / No=%d",
                       nrow(X_tr_use), sum(y_tr_use == 1), sum(y_tr_use == 0)))

    } else if (tech == "smote") {
      set.seed(seed_tech)
      df_in <- bind_cols(as.data.frame(X_fold_tr),
                          pobre = factor(y_fold_tr, levels = c(0, 1)))
      df_out <- tryCatch({
        themis::smote(df_in, var = "pobre", k = 5, over_ratio = 1)
      }, error = function(e) {
        message("    [smote] error: ", e$message, " -- usando upsample")
        us <- caret::upSample(
          x = as.data.frame(X_fold_tr),
          y = factor(y_fold_tr, levels = c(0, 1)),
          yname = "pobre"
        )
        us
      })
      X_tr_use <- df_out |> select(-pobre)
      y_tr_use <- as.numeric(as.character(df_out$pobre))
      message(sprintf("    [smote] %d obs | Yes=%d / No=%d",
                       nrow(X_tr_use), sum(y_tr_use == 1), sum(y_tr_use == 0)))
    }

    # Entrenar glmnet con CV interno para elegir lambda
    set.seed(seed_tech)
    y_fct <- factor(ifelse(y_tr_use == 1, "Yes", "No"), levels = c("No", "Yes"))

    modelo_k <- tryCatch({
      train(
        x          = as.data.frame(X_tr_use),
        y          = y_fct,
        method     = "glmnet",
        family     = "binomial",
        weights    = weights_use,
        trControl  = trainControl(
          method = "cv", number = 3,
          summaryFunction = twoClassSummary,
          classProbs = TRUE, verboseIter = FALSE
        ),
        tuneGrid   = expand.grid(
          alpha = c(0.5, 0.75, 1),
          lambda = 10^seq(-4, 0, length.out = 10)
        ),
        metric     = "ROC",
        preProcess = c("center", "scale")
      )
    }, error = function(e) {
      message("    [", tech, "] error en train: ", e$message)
      NULL
    })

    if (is.null(modelo_k)) {
      return(tibble(fold = k, technique = tech, F1 = NA_real_, th = NA_real_))
    }

    # Predecir en fold de validacion
    prob_val <- predict(modelo_k, newdata = as.data.frame(X_fold_val),
                         type = "prob")$Yes

    best_row <- buscar_th(prob_val, y_fold_val)

    message(sprintf("    %-15s -> F1=%.4f | th=%.2f",
                     tech, best_row$F1, best_row$threshold))

    tibble(fold = k, technique = tech,
           F1 = best_row$F1, th = best_row$threshold)
  })

  balance_resultados[[k]] <- fold_bal_res
}

t1_bal <- Sys.time()
message("\nCV completado en ",
        round(as.numeric(difftime(t1_bal, t0_bal, units = "mins")), 1), " min")

# -- 6. Seleccion de mejor tecnica -------------------------------------------
bal_df <- bind_rows(balance_resultados)

bal_summary <- bal_df |>
  group_by(technique) |>
  summarise(
    F1_mean = mean(F1, na.rm = TRUE),
    F1_sd   = sd(F1,   na.rm = TRUE),
    th_mean = mean(th, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(desc(F1_mean))

cat("\n========================================\n")
cat("RESULTADOS POR TECNICA (CV-5)\n")
cat("========================================\n")
print(bal_summary |> mutate(across(where(is.double), ~ round(., 4))))

write_csv(bal_df,      file.path(dir_model, "cv_balance_results.csv"))
write_csv(bal_summary, file.path(dir_model, "cv_balance_summary.csv"))

best_technique <- bal_summary |>
  slice_max(F1_mean, n = 1, with_ties = FALSE) |>
  pull(technique)
best_f1  <- bal_summary |> filter(technique == best_technique) |> pull(F1_mean)
best_th  <- bal_summary |> filter(technique == best_technique) |> pull(th_mean)

message("\n  Mejor tecnica: ", best_technique,
        " (F1 = ", round(best_f1, 4), ", th = ", round(best_th, 2), ")")

# -- 7. Modelo final con mejor tecnica --------------------------------------
message("\n== Entrenando modelo final con: ", best_technique, " ==")

THRESHOLD <- round(best_th, 2)
y_final   <- y_train
X_final   <- X_train
weights_final <- NULL

if (best_technique == "class_weights") {
  n_yes <- sum(y_train == 1)
  n_no  <- sum(y_train == 0)
  weights_final <- ifelse(y_train == 1, n_no / n_yes, 1)

} else if (best_technique == "downsample") {
  set.seed(42)
  ds <- caret::downSample(
    x = as.data.frame(X_train), y = factor(y_train, levels = c(0,1)),
    yname = "pobre"
  )
  X_final <- ds |> select(-pobre)
  y_final <- as.numeric(as.character(ds$pobre))

} else if (best_technique == "upsample") {
  set.seed(42)
  us <- caret::upSample(
    x = as.data.frame(X_train), y = factor(y_train, levels = c(0,1)),
    yname = "pobre"
  )
  X_final <- us |> select(-pobre)
  y_final <- as.numeric(as.character(us$pobre))

} else if (best_technique == "smote") {
  set.seed(42)
  df_in <- bind_cols(as.data.frame(X_train),
                      pobre = factor(y_train, levels = c(0, 1)))
  df_out <- tryCatch({
    themis::smote(df_in, var = "pobre", k = 5, over_ratio = 1)
  }, error = function(e) {
    caret::upSample(x = as.data.frame(X_train),
                     y = factor(y_train, levels = c(0,1)), yname = "pobre")
  })
  X_final <- df_out |> select(-pobre)
  y_final <- as.numeric(as.character(df_out$pobre))
}

y_final_fct <- factor(ifelse(y_final == 1, "Yes", "No"), levels = c("No", "Yes"))

set.seed(42)
modelo_final <- train(
  x          = as.data.frame(X_final),
  y          = y_final_fct,
  method     = "glmnet",
  family     = "binomial",
  weights    = weights_final,
  trControl  = trainControl(
    method = "cv", number = 5,
    summaryFunction = twoClassSummary,
    classProbs = TRUE, verboseIter = TRUE,
    savePredictions = "final"
  ),
  tuneGrid   = expand.grid(
    alpha = c(0, 0.25, 0.5, 0.75, 1),
    lambda = 10^seq(-4, 1, length.out = 20)
  ),
  metric     = "ROC",
  preProcess = c("center", "scale")
)

cat("\n-- Modelo final --\n")
cat("  Tecnica: ", best_technique, "\n")
cat("  alpha:   ", modelo_final$bestTune$alpha, "\n")
cat("  lambda:  ", modelo_final$bestTune$lambda, "\n")
cat("  Threshold: ", THRESHOLD, "\n")

# -- 8. Metricas en train ---------------------------------------------------
prob_train <- predict(modelo_final, newdata = as.data.frame(X_train),
                       type = "prob")$Yes
pred_train <- factor(ifelse(prob_train >= THRESHOLD, "Yes", "No"),
                      levels = c("No", "Yes"))
obs_train  <- factor(ifelse(y_train == 1, "Yes", "No"),
                      levels = c("No", "Yes"))

cm_final <- confusionMatrix(pred_train, obs_train, positive = "Yes")

cv_F1        <- cm_final$byClass["F1"]
cv_Precision <- cm_final$byClass["Precision"]
cv_Recall    <- cm_final$byClass["Recall"]

roc_obj <- pROC::roc(response = y_train, predictor = prob_train, quiet = TRUE)
auc_roc <- as.numeric(pROC::auc(roc_obj))

cat("\n-- Metricas (train, threshold =", THRESHOLD, ") --\n")
cat("  F1:        ", round(cv_F1,        4), "\n")
cat("  Precision: ", round(cv_Precision, 4), "\n")
cat("  Recall:    ", round(cv_Recall,    4), "\n")
cat("  AUC-ROC:   ", round(auc_roc,      4), "\n")

# -- 9. Guardar metricas y diagnosticos -------------------------------------
sink(file.path(dir_model, "metricas.txt"))
cat("=== ELASTIC NET -- BALANCE ===\n\n")
cat("Mejor tecnica:", best_technique, "\n")
cat("Predictores:", ncol(X_train), "\n")
cat("alpha:", modelo_final$bestTune$alpha, "\n")
cat("lambda:", modelo_final$bestTune$lambda, "\n")
cat("Threshold:", THRESHOLD, "\n")
cat("F1:", round(cv_F1, 4), "\n")
cat("AUC-ROC:", round(auc_roc, 4), "\n\n")
cat("-- Comparacion tecnicas (CV-5) --\n")
print(bal_summary |> mutate(across(where(is.double), ~ round(., 4))))
cat("\n-- Confusion Matrix --\n")
print(cm_final)
sink()

diagnostics <- list(
  model_id     = paste0("EN_balance_", best_technique),
  best_technique = best_technique,
  bal_summary  = bal_summary,
  bal_results  = bal_df,
  modelo       = modelo_final,
  threshold    = THRESHOLD,
  cm           = cm_final,
  cv_F1        = best_f1,
  auc_roc      = auc_roc
)
saveRDS(diagnostics, file.path(dir_model, "diagnostics.rds"))

# -- 10. Submission ----------------------------------------------------------
prob_test <- predict(modelo_final, newdata = as.data.frame(X_test),
                      type = "prob")$Yes
pred_test <- as.integer(prob_test >= THRESHOLD)

model_id <- paste0("EN_balance_", best_technique)
submission <- tibble(id = ids, pobre = pred_test)
sub_file <- file.path(dir_submissions,
                       paste0("submission_", model_id, ".csv"))
write_csv(submission, sub_file)
message("Submission guardada: ", sub_file)
message("  Pobres predichos: ", sum(submission$pobre),
        " (", round(mean(submission$pobre) * 100, 1), "%)")

# -- 11. Registrar en model_registry.csv ------------------------------------
nueva_fila <- tibble(
  model_id           = model_id,
  fecha              = Sys.Date(),
  autor              = "Dani",
  algoritmo          = "ElasticNet",
  n_features         = ncol(X_train),
  imbalance_strategy = best_technique,
  cv_folds           = K_FOLDS,
  cv_F1              = round(best_f1, 4),
  cv_Precision       = round(cv_Precision, 4),
  cv_Recall          = round(cv_Recall, 4),
  auc_roc            = round(auc_roc, 4),
  kaggle_public_F1   = NA_real_,
  threshold          = THRESHOLD,
  notas              = paste0(
    "EN Parte 4. Comparacion 5 tecnicas balanceo CV-5. ",
    "Mejor: ", best_technique, " F1_CV5=", round(best_f1, 4), ". ",
    "93 vars (misma spec EN_001A). th=", THRESHOLD, "."
  )
)

if (file.exists(registry_path)) {
  registry <- read_csv(registry_path, show_col_types = FALSE)
  registry <- bind_rows(registry, nueva_fila)
} else {
  registry <- nueva_fila
}
write_csv(registry, registry_path)
message("Registro actualizado: ", registry_path)

cat("\n-- Fila registrada --\n")
print(nueva_fila)

# -- 12. Resumen final -------------------------------------------------------
cat("\n========================================\n")
cat("RESUMEN -- Elastic Net Balanceo\n")
cat("========================================\n")
cat("Tecnicas evaluadas: ", paste(TECHNIQUES, collapse = ", "), "\n")
cat("Mejor tecnica: ", best_technique, "\n")
cat("F1 CV-5: ", round(best_f1, 4), "\n")
cat("Comparar vs EN_001A baseline: F1 = 0.695\n")
cat("\nSi ", best_technique, " > baseline -> el balanceo ayuda\n")
cat("Si ", best_technique, " <= baseline -> no usar balanceo\n")
message("\nElastic Net Balanceo completo!")
# nolint end
