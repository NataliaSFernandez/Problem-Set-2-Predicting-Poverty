#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/XGBoost/XGB_002.R
#==============================================================================
#
# ALGORITMO: XGBoost — Feature Selection por ranking de importancia de RF_001
#
# QUÉ HACE ESTE SCRIPT:
#   Entrena 3 modelos XGBoost con distintos subconjuntos de predictores,
#   seleccionados según el ranking de importancia por permutación calculado en
#   RF_001 (feature_matrix.csv). Los subconjuntos son:
#
#     Modelo A: top 20 variables
#     Modelo B: top 40 variables
#     Modelo C: top 60 variables
#
#   Para cada subconjunto se usa la misma arquitectura que XGB_001: CV-5 con
#   el mejor conjunto de hiperparámetros de XGB_001 (leído de diagnostics.rds),
#   evaluación sobre predicciones OOF (out-of-fold) y threshold óptimo OOF.
#
#   Al final se imprime una tabla comparativa de F1, Precision, Recall y
#   AUC-ROC OOF para los tres modelos más la referencia de XGB_001.
#
# DECISIÓN DE DISEÑO:
#   Se usa el ranking de RF_001 (importancia por permutación) en lugar del
#   Gain de XGBoost porque es una medida más estable y comparable entre
#   algoritmos: permite estudiar si las variables que más ayudan al RF
#   son igualmente útiles para XGBoost.
#
# EVALUACIÓN HONESTA:
#   El threshold óptimo se determina sobre predicciones OOF del CV-5. No hay
#   data leakage: cada observación es predicha por modelos que no la vieron.
#   (1 combo × 5 folds = 5 fits por modelo → 15 fits totales)
#
# FLUJO DEL SCRIPT:
#   0. Paquetes
#   1. Configuración global
#   2. Cargar datos
#   3. Cargar ranking de RF_001 (feature_matrix.csv) y mejores params de XGB_001
#   4. Preprocesamiento (todo numérico) y selección de features
#   5. Función auxiliar: CV-5 con hiperparámetros fijos → métricas OOF
#   6. Entrenar modelos A (top20), B (top40), C (top60)
#   7. Generar submissions
#   8. Comparación final (consola + CSV)
#   9. Registro en model_registry.csv
#
# INPUTS (relativos a la raíz del proyecto):
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#   02_outputs/models/RandomForest/RF_001/feature_matrix.csv   ← ranking vars
#   02_outputs/models/XGBoost/XGB_001/diagnostics.rds          ← best params
#   02_outputs/model_registry.csv
#
# OUTPUTS (relativos a la raíz del proyecto):
#   03_submissions/XGB_top{20,40,60}_*.csv
#   02_outputs/models/XGBoost/XGB_002/comparacion_features.csv
#   02_outputs/model_registry.csv  (append)
#
# REPRODUCIBILIDAD:
#   Correr desde la raíz del proyecto. Semilla global: 42.
#   Requiere haber corrido RF_001 y XGB_001 antes.
#==============================================================================
# nolint start

# =============================================================================
# SECCIÓN 0: PAQUETES
# =============================================================================
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,  # dplyr, ggplot2, purrr, tibble
  caret,      # createFolds() para el CV manual
  xgboost,    # xgb.train() / xgb.DMatrix() — API nativa (compatible con v3.x)
  pROC,       # roc(), auc()
  fs          # dir_create()
)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
AUTOR    <- "Jonathan Melo"
MODEL_ID <- "XGB_002"

set.seed(42)

N_THREADS <- max(1L, parallel::detectCores() - 1L)

dir_model  <- file.path("02_outputs/models/XGBoost", MODEL_ID)
dir_subs   <- "03_submissions"
reg_path   <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model, recurse = TRUE)
fs::dir_create(dir_subs,  recurse = TRUE)

message("  Núcleos disponibles para XGBoost (nthread): ", N_THREADS)

# =============================================================================
# SECCIÓN 2: CARGAR DATOS
# =============================================================================
message("== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id

message("Train: ", nrow(train), " hogares | ", ncol(train), " columnas")
message("Test:  ", nrow(test),  " hogares | ", ncol(test),  " columnas")

niveles <- c("No", "Yes")
y_train <- factor(ifelse(train$pobre == 1, "Yes", "No"), levels = niveles)

# =============================================================================
# SECCIÓN 3: RANKING DE VARIABLES (RF_001) Y MEJORES HIPERPARÁMETROS (XGB_001)
# =============================================================================
message("\n== Cargando ranking de importancia (RF_001) ==")

feat_mat_path <- "02_outputs/models/RandomForest/RF_001/feature_matrix.csv"
if (!file.exists(feat_mat_path)) {
  stop("No se encontró feature_matrix.csv de RF_001. Corré RF_001 primero.")
}

feature_matrix <- read_csv(feat_mat_path, show_col_types = FALSE) |>
  arrange(rank)

message("  Variables totales en ranking: ", nrow(feature_matrix))

vars_top20 <- feature_matrix |> filter(rank <= 20) |> pull(variable)
vars_top40 <- feature_matrix |> filter(rank <= 40) |> pull(variable)
vars_top60 <- feature_matrix |> filter(rank <= 60) |> pull(variable)

message("  Top 20 - primera: ", vars_top20[1], "  |  última: ", vars_top20[20])
message("  Top 40 - primera: ", vars_top40[1], "  |  última: ", vars_top40[40])
message("  Top 60 - primera: ", vars_top60[1], "  |  última: ", vars_top60[60])

# ---- Mejores hiperparámetros de XGB_001 -------------------------------------
message("\n== Cargando mejores hiperparámetros de XGB_001 ==")

diag_path <- "02_outputs/models/XGBoost/XGB_001/diagnostics.rds"
if (!file.exists(diag_path)) {
  stop("No se encontró diagnostics.rds de XGB_001. Corré XGB_001 primero.")
}

diag001      <- readRDS(diag_path)
best_p       <- diag001$xgb_cv$bestTune

BEST_NROUNDS          <- best_p$nrounds
BEST_MAX_DEPTH        <- best_p$max_depth
BEST_ETA              <- best_p$eta
BEST_GAMMA            <- best_p$gamma
BEST_MIN_CHILD_WEIGHT <- best_p$min_child_weight
BEST_COLSAMPLE        <- best_p$colsample_bytree
BEST_SUBSAMPLE        <- best_p$subsample

message("  Hiperparámetros heredados de XGB_001:")
message("    nrounds          = ", BEST_NROUNDS)
message("    max_depth        = ", BEST_MAX_DEPTH)
message("    eta              = ", BEST_ETA)
message("    gamma            = ", BEST_GAMMA)
message("    min_child_weight = ", BEST_MIN_CHILD_WEIGHT)
message("    colsample_bytree = ", BEST_COLSAMPLE)
message("    subsample        = ", BEST_SUBSAMPLE)

# =============================================================================
# SECCIÓN 4: PREPROCESAMIENTO BASE
# =============================================================================
# XGBoost requiere entradas 100% numéricas. Misma lógica que XGB_001.
message("\n== Preprocesando variables ==")

preparar_X_xgb <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>
    mutate(zona = as.numeric(zona)) |>
    select(-any_of(c("ciudad", "dpto"))) |>
    mutate(across(where(is.character), as.numeric)) |>
    mutate(across(where(is.factor),   as.integer))
}

X_train_full <- preparar_X_xgb(train)
X_test_full  <- preparar_X_xgb(test)

cols_comunes  <- intersect(names(X_train_full), names(X_test_full))
X_train_full  <- X_train_full |> select(all_of(cols_comunes))
X_test_full   <- X_test_full  |> select(all_of(cols_comunes))

message("  Predictores disponibles: ", ncol(X_train_full))

# Función: intersectar subconjunto con columnas disponibles tras preprocesamiento
filtrar_cols <- function(X, vars) {
  vars_ok <- intersect(vars, names(X))
  if (length(vars_ok) < length(vars)) {
    message("  AVISO: ", length(vars) - length(vars_ok),
            " variable(s) del ranking no disponibles tras preprocesamiento.")
  }
  X |> select(all_of(vars_ok))
}

# Partición reproducible en 5 folds estratificados — compartida por los 3 modelos
set.seed(42)
folds <- caret::createFolds(y_train, k = 5, list = TRUE, returnTrain = TRUE)

# =============================================================================
# SECCIÓN 5: FUNCIÓN AUXILIAR — CV-5 CON HIPERPARÁMETROS FIJOS
# =============================================================================
# Corre CV-5 sobre el subconjunto de variables dado y devuelve:
#   - threshold óptimo OOF (max F1), F1 / Precision / Recall OOF, AUC-ROC OOF
#   - modelo final re-entrenado sobre todo train
#
# Parámetros:
#   X_tr   : data.frame numérico de predictores (train)
#   X_te   : data.frame numérico de predictores (test)
#   y_tr   : factor respuesta ("No"/"Yes")
#   label  : string identificador para los mensajes
entrenar_xgb_oof <- function(X_tr, X_te, y_tr, label) {
  message("\n========================================")
  message("Entrenando ", label, "  (", ncol(X_tr), " variables)")
  message("========================================")

  mat_tr <- as.matrix(X_tr)
  mat_te <- as.matrix(X_te)
  y_int  <- as.integer(y_tr == "Yes")

  params <- list(
    objective        = "binary:logistic",
    eval_metric      = "logloss",
    max_depth        = BEST_MAX_DEPTH,
    eta              = BEST_ETA,
    gamma            = BEST_GAMMA,
    min_child_weight = BEST_MIN_CHILD_WEIGHT,
    colsample_bytree = BEST_COLSAMPLE,
    subsample        = BEST_SUBSAMPLE,
    nthread          = N_THREADS
  )

  fold_met <- matrix(NA_real_, nrow = 5, ncol = 3,
                     dimnames = list(NULL, c("F1", "Precision", "Recall")))
  fold_oof <- vector("list", 5)

  set.seed(42)
  t0 <- Sys.time()

  for (f in seq_len(5)) {
    tr_idx  <- folds[[f]]
    val_idx <- setdiff(seq_len(nrow(mat_tr)), tr_idx)

    dtrain_f <- xgboost::xgb.DMatrix(mat_tr[tr_idx,  ], label = y_int[tr_idx])
    dval_f   <- xgboost::xgb.DMatrix(mat_tr[val_idx, ], label = y_int[val_idx])

    m_f <- xgboost::xgb.train(
      params  = params,
      data    = dtrain_f,
      nrounds = BEST_NROUNDS,
      verbose = 0
    )

    prob_val <- predict(m_f, dval_f)
    pred_val <- factor(ifelse(prob_val >= 0.5, "Yes", "No"), levels = niveles)
    obs_val  <- y_tr[val_idx]

    TP <- sum(pred_val == "Yes" & obs_val == "Yes")
    FP <- sum(pred_val == "Yes" & obs_val != "Yes")
    FN <- sum(pred_val == "No"  & obs_val == "Yes")
    fold_met[f, "F1"]        <- if ((2*TP+FP+FN) == 0) NA_real_ else 2*TP/(2*TP+FP+FN)
    fold_met[f, "Precision"] <- if ((TP+FP) == 0)       NA_real_ else TP/(TP+FP)
    fold_met[f, "Recall"]    <- if ((TP+FN) == 0)       NA_real_ else TP/(TP+FN)

    fold_oof[[f]] <- tibble(rowIndex = val_idx, Yes = prob_val, obs = obs_val)

    message(sprintf("  + Fold %d  F1(th=0.5) = %.4f", f, fold_met[f, "F1"]))
  }

  t1 <- Sys.time()
  message("  Tiempo CV-5: ",
          round(as.numeric(difftime(t1, t0, units = "mins")), 2), " min")

  avg_met <- colMeans(fold_met, na.rm = TRUE)
  message("  F1 CV-5 (th=0.5): ", round(avg_met["F1"], 4))

  # Predicciones OOF → threshold óptimo
  oof_df    <- bind_rows(fold_oof) |> arrange(rowIndex)
  probs_oof <- oof_df$Yes
  obs_oof   <- oof_df$obs

  th_search <- map_dfr(seq(0.05, 0.95, by = 0.01), function(th) {
    pred_th <- factor(ifelse(probs_oof >= th, "Yes", "No"), levels = niveles)
    TP <- sum(pred_th == "Yes" & obs_oof == "Yes")
    FP <- sum(pred_th == "Yes" & obs_oof != "Yes")
    FN <- sum(pred_th == "No"  & obs_oof == "Yes")
    f1   <- if ((2*TP+FP+FN) == 0) NA_real_ else 2*TP/(2*TP+FP+FN)
    prec <- if ((TP+FP) == 0)       NA_real_ else TP/(TP+FP)
    rec  <- if ((TP+FN) == 0)       NA_real_ else TP/(TP+FN)
    tibble(threshold = th, F1 = f1, Precision = prec, Recall = rec)
  })

  best_row <- th_search |> filter(!is.na(F1)) |>
    slice_max(F1, n = 1, with_ties = FALSE)
  best_th  <- best_row$threshold
  best_f1  <- best_row$F1
  best_pr  <- best_row$Precision
  best_rc  <- best_row$Recall

  roc_obj <- pROC::roc(
    response  = as.integer(obs_oof == "Yes"),
    predictor = probs_oof,
    quiet     = TRUE
  )
  auc_val <- as.numeric(pROC::auc(roc_obj))

  message("  Threshold óptimo OOF: ", best_th)
  message("  F1 OOF:               ", round(best_f1,  4))
  message("  Precision OOF:        ", round(best_pr,  4))
  message("  Recall OOF:           ", round(best_rc,  4))
  message("  AUC-ROC OOF:          ", round(auc_val,  4))

  # Modelo final sobre todo train
  message("  Re-entrenando modelo final sobre todo train...")
  set.seed(42)
  dtrain_full  <- xgboost::xgb.DMatrix(mat_tr, label = y_int)
  modelo_final <- xgboost::xgb.train(
    params  = params,
    data    = dtrain_full,
    nrounds = BEST_NROUNDS,
    verbose = 0
  )

  list(
    modelo     = modelo_final,
    mat_te     = mat_te,
    best_th    = best_th,
    f1         = best_f1,
    precision  = best_pr,
    recall     = best_rc,
    auc_roc    = auc_val,
    n_features = ncol(X_tr)
  )
}

# =============================================================================
# SECCIÓN 6: ENTRENAR MODELOS A (top20), B (top40), C (top60)
# =============================================================================

# -- Modelo A: top 20 --
X_train_20 <- filtrar_cols(X_train_full, vars_top20)
X_test_20  <- filtrar_cols(X_test_full,  vars_top20)
res_20     <- entrenar_xgb_oof(X_train_20, X_test_20, y_train, "Modelo A — Top 20")

# -- Modelo B: top 40 --
X_train_40 <- filtrar_cols(X_train_full, vars_top40)
X_test_40  <- filtrar_cols(X_test_full,  vars_top40)
res_40     <- entrenar_xgb_oof(X_train_40, X_test_40, y_train, "Modelo B — Top 40")

# -- Modelo C: top 60 --
X_train_60 <- filtrar_cols(X_train_full, vars_top60)
X_test_60  <- filtrar_cols(X_test_full,  vars_top60)
res_60     <- entrenar_xgb_oof(X_train_60, X_test_60, y_train, "Modelo C — Top 60")

# =============================================================================
# SECCIÓN 7: SUBMISSIONS
# =============================================================================
message("\n== Generando submissions ==")

generar_submission <- function(res, ids, tag) {
  dtest      <- xgboost::xgb.DMatrix(res$mat_te)
  probs_test <- predict(res$modelo, dtest)
  pred_test  <- as.integer(probs_test >= res$best_th)
  sub_name   <- sprintf("XGB_%s_nrounds%d_depth%d_eta%03.0f_th%03.0f.csv",
                        tag, BEST_NROUNDS, BEST_MAX_DEPTH,
                        BEST_ETA * 100, res$best_th * 100)
  sub_path   <- file.path(dir_subs, sub_name)
  write_csv(tibble(id = ids, pobre = pred_test), sub_path)
  message("  ", sub_path,
          "  | pobres predichos: ", sum(pred_test),
          " (", round(mean(pred_test) * 100, 1), "%)")
  sub_path
}

sub_20 <- generar_submission(res_20, ids, "top20")
sub_40 <- generar_submission(res_40, ids, "top40")
sub_60 <- generar_submission(res_60, ids, "top60")

# =============================================================================
# SECCIÓN 8: COMPARACIÓN FINAL DE MÉTRICAS
# =============================================================================
message("\n========================================")
message("COMPARACIÓN DE MODELOS — FEATURE SELECTION XGBoost")
message("========================================")

comparacion <- tibble(
  Modelo        = c("A — Top 20", "B — Top 40", "C — Top 60"),
  N_vars        = c(res_20$n_features, res_40$n_features, res_60$n_features),
  Threshold     = c(res_20$best_th,   res_40$best_th,    res_60$best_th),
  F1_OOF        = c(res_20$f1,        res_40$f1,         res_60$f1),
  Precision_OOF = c(res_20$precision, res_40$precision,  res_60$precision),
  Recall_OOF    = c(res_20$recall,    res_40$recall,     res_60$recall),
  AUC_ROC_OOF   = c(res_20$auc_roc,  res_40$auc_roc,    res_60$auc_roc)
)

# Agregar XGB_001 (todas las variables) como referencia
if (file.exists(reg_path)) {
  registry <- read_csv(reg_path, show_col_types = FALSE)
  xgb001   <- registry |> filter(model_id == "XGB_001")
  if (nrow(xgb001) > 0) {
    xgb001_row <- tibble(
      Modelo        = "XGB_001 (todas)",
      N_vars        = xgb001$n_features[1],
      Threshold     = xgb001$threshold[1],
      F1_OOF        = xgb001$cv_F1[1],
      Precision_OOF = xgb001$cv_Precision[1],
      Recall_OOF    = xgb001$cv_Recall[1],
      AUC_ROC_OOF   = xgb001$auc_roc[1]
    )
    comparacion <- bind_rows(xgb001_row, comparacion)
  }
}

# Formatear
comparacion_fmt <- comparacion |>
  mutate(across(where(is.double) & !Threshold, ~ round(., 4)),
         Threshold = round(Threshold, 2))

cat("\n")
cat(strrep("=", 80), "\n")
cat("  COMPARACIÓN XGBoost: F1 / Precision / Recall / AUC-ROC  (métricas OOF)\n")
cat("  Variables seleccionadas según ranking de importancia de RF_001\n")
cat(strrep("=", 80), "\n")
print(comparacion_fmt, n = Inf)
cat(strrep("-", 80), "\n")

best_model <- comparacion_fmt |> slice_max(F1_OOF, n = 1, with_ties = FALSE)
cat(sprintf("\n  Mejor F1 OOF: %s  →  F1 = %.4f | Precision = %.4f | Recall = %.4f | AUC = %.4f\n\n",
            best_model$Modelo,
            best_model$F1_OOF,
            best_model$Precision_OOF,
            best_model$Recall_OOF,
            best_model$AUC_ROC_OOF))

comp_path <- file.path(dir_model, "comparacion_features.csv")
write_csv(comparacion_fmt, comp_path)
message("  Tabla guardada: ", comp_path)

# =============================================================================
# SECCIÓN 9: REGISTRO EN model_registry.csv
# =============================================================================
message("\n== Actualizando model_registry.csv ==")

registrar_modelo <- function(tag, res) {
  mid <- paste0(MODEL_ID, "_", tag)
  tibble(
    model_id           = mid,
    fecha              = Sys.Date(),
    autor              = AUTOR,
    algoritmo          = "XGBoost",
    n_features         = res$n_features,
    imbalance_strategy = "none",
    cv_folds           = 5L,
    cv_F1              = round(res$f1,        4),
    cv_Precision       = round(res$precision, 4),
    cv_Recall          = round(res$recall,    4),
    auc_roc            = round(res$auc_roc,   4),
    kaggle_public_F1   = NA_real_,
    threshold          = res$best_th,
    notas              = paste0(
      "XGBoost. Feature selection por ranking permutación RF_001.",
      " Subconjunto: ", tag, " (", res$n_features, " vars).",
      " nrounds=", BEST_NROUNDS,
      " max_depth=", BEST_MAX_DEPTH,
      " eta=", BEST_ETA,
      " gamma=", BEST_GAMMA,
      " min_child_weight=", BEST_MIN_CHILD_WEIGHT,
      " colsample=", BEST_COLSAMPLE,
      " subsample=", BEST_SUBSAMPLE,
      ". th_oof=", res$best_th,
      " F1_oof=", round(res$f1, 4),
      " AUC_oof=", round(res$auc_roc, 4), "."
    ),
    nrounds          = BEST_NROUNDS,
    max_depth        = BEST_MAX_DEPTH,
    eta              = BEST_ETA,
    gamma            = BEST_GAMMA,
    min_child_weight = BEST_MIN_CHILD_WEIGHT,
    colsample_bytree = BEST_COLSAMPLE,
    subsample        = BEST_SUBSAMPLE
  )
}

filas_nuevas <- bind_rows(
  registrar_modelo("top20", res_20),
  registrar_modelo("top40", res_40),
  registrar_modelo("top60", res_60)
)

ids_nuevos <- filas_nuevas$model_id

if (file.exists(reg_path)) {
  registry <- read_csv(reg_path, show_col_types = FALSE) |>
    mutate(fecha = as.Date(fecha, origin = "1899-12-30"))
  registry <- registry |> filter(!model_id %in% ids_nuevos)
  registry <- bind_rows(registry, filas_nuevas)
} else {
  registry <- filas_nuevas
}

write_csv(registry, reg_path)
message("  Modelos registrados: ", paste(ids_nuevos, collapse = ", "))

# nolint end
