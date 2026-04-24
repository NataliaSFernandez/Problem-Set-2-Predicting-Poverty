#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/RandomForest/RF_002.R
#==============================================================================
#
# ALGORITMO: Random Forest — Feature Selection por importancia (RF_001)
#
# QUÉ HACE ESTE SCRIPT:
#   Entrena 3 modelos Random Forest con distintos subconjuntos de predictores,
#   seleccionados según el ranking de importancia por permutación calculado en
#   RF_001 (feature_matrix.csv). Los subconjuntos son:
#
#     Modelo A: top 20 variables (mayor importancia)
#     Modelo B: top 40 variables
#     Modelo C: top 60 variables
#
#   Para cada modelo se usan los mejores hiperparámetros de RF_001 sin re-tunear
#   (mtry fijo, splitrule = "gini", min.node.size = 5, num.trees = 150).
#   La evaluación es honesta: threshold optimizado sobre probabilidades OOB
#   (out-of-bag) nativas de ranger.
#
#   Al final se imprime una tabla comparativa de F1, Precision, Recall y AUC-ROC
#   para los tres modelos.
#
# HIPERPARÁMETROS (heredados de RF_001 — sin re-tuning):
#   mtry          = 20   (mejor valor de RF_001)
#   splitrule     = "gini"
#   min.node.size = 5
#   num.trees     = 150
#   importance    = "permutation"
#
# FLUJO DEL SCRIPT:
#   0. Paquetes
#   1. Configuración global
#   2. Cargar datos
#   3. Cargar ranking de importancia (feature_matrix.csv de RF_001)
#   4. Preprocesamiento y selección de features por subconjunto
#   5. Función auxiliar: entrenar un RF y extraer métricas OOB
#   6. Entrenar modelos A (top20), B (top40), C (top60)
#   7. Generar submissions para cada modelo
#   8. Comparación final de métricas (impresa en consola)
#
# INPUTS (relativos a la raíz del proyecto):
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#   02_outputs/models/RandomForest/RF_001/feature_matrix.csv
#
# OUTPUTS (relativos a la raíz del proyecto):
#   03_submissions/RF_mtry20_top20_*.csv
#   03_submissions/RF_mtry20_top40_*.csv
#   03_submissions/RF_mtry20_top60_*.csv
#   02_outputs/models/RandomForest/RF_002/comparacion_features.csv
#   02_outputs/model_registry.csv  (append)
#
# REPRODUCIBILIDAD:
#   Correr desde la raíz del proyecto. Semilla global: 42.
#==============================================================================
# nolint start

# =============================================================================
# SECCIÓN 0: PAQUETES
# =============================================================================
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,   # dplyr, ggplot2, purrr, tibble
  ranger,      # Random Forest en C++
  pROC,        # roc(), auc()
  fs           # dir_create()
)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
AUTOR    <- "Jonathan Melo"
MODEL_ID <- "RF_002"

set.seed(42)

dir_model <- file.path("02_outputs/models/RandomForest", MODEL_ID)
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model, recurse = TRUE)
fs::dir_create(dir_subs,  recurse = TRUE)

# Hiperparámetros fijos heredados de RF_001 (mejores valores del CV-5)
BEST_MTRY      <- 20L
BEST_SPLITRULE <- "gini"
BEST_MIN_NODE  <- 5L
NUM_TREES      <- 150L

# =============================================================================
# SECCIÓN 2: CARGAR DATOS
# =============================================================================
message("== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id

message("Train: ", nrow(train), " hogares | ", ncol(train), " columnas")
message("Test:  ", nrow(test),  " hogares | ", ncol(test),  " columnas")

# =============================================================================
# SECCIÓN 3: CARGAR RANKING DE IMPORTANCIA (RF_001)
# =============================================================================
# Leemos feature_matrix.csv generado por RF_001 con importancia por permutación.
# Está ordenado descendentemente por importancia; la columna 'rank' lo confirma.
message("\n== Cargando ranking de importancia (RF_001) ==")

feat_mat_path <- "02_outputs/models/RandomForest/RF_001/feature_matrix.csv"
if (!file.exists(feat_mat_path)) {
  stop("No se encontró feature_matrix.csv. Asegúrate de haber corrido RF_001 primero.")
}

feature_matrix <- read_csv(feat_mat_path, show_col_types = FALSE) |>
  arrange(rank)

message("  Variables totales en ranking: ", nrow(feature_matrix))

# Subconjuntos de variables por rango de importancia
vars_top20 <- feature_matrix |> filter(rank <= 20) |> pull(variable)
vars_top40 <- feature_matrix |> filter(rank <= 40) |> pull(variable)
vars_top60 <- feature_matrix |> filter(rank <= 60) |> pull(variable)

message("  Top 20 - primera: ", vars_top20[1], "  |  última: ", vars_top20[20])
message("  Top 40 - primera: ", vars_top40[1], "  |  última: ", vars_top40[40])
message("  Top 60 - primera: ", vars_top60[1], "  |  última: ", vars_top60[60])

# =============================================================================
# SECCIÓN 4: PREPROCESAMIENTO BASE
# =============================================================================
# Misma lógica que RF_001: excluir id, pobre, ciudad, dpto por alta cardinalidad.
# Convertir zona a numérico y resto de caracteres a factor.
message("\n== Preprocesando variables ==")

preparar_X_rf <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>
    mutate(zona = as.numeric(zona)) |>
    select(-any_of(c("ciudad", "dpto"))) |>
    mutate(across(where(is.character), as.factor))
}

X_train_full <- preparar_X_rf(train)
X_test_full  <- preparar_X_rf(test)

# -- Variables a excluir --------------------------------------
#Variables excluidas a mano ya que estas son interacciones que los arboles 
#por su naturaleza las pueden crear ellos mismos
#Se excluyen variables cuadráticas, logarítmicas, y de interacción, que no aportan

VARS_EXCLUIR <- c(
  # Cuadráticas de capital humano (Bloque 3)
  "educ_jefe_sq",
  "educ_prom_sq",
  # Logarítmicas (Bloque 9)
  "log_num_personas",
  "log_n_menores",
  # Cuadráticas generales (Bloque 9)
  "ratio_depend_sq",
  "tasa_ocup_sq",
  "hacinamiento_sq",
  "edad_jefe_sq",
  # Interacciones (Bloque 8)
  "educ_x_formal",
  "educ_x_ocup",
  "menores_x_desocup",
  "depend_x_informal",
  "rural_x_cta_propia",
  "educ_x_rural",
  "subsidiado_x_menores",
  "jefe_mayor_pension"
)

# -- Eliminar variables de VARS_EXCLUIR --------------------------------------
X_train_full <- X_train_full |> select(-any_of(VARS_EXCLUIR))
X_test_full  <- X_test_full  |> select(-any_of(VARS_EXCLUIR))

message("  Variables excluidas manualmente: ", length(VARS_EXCLUIR))
message("  Predictores tras exclusión: ", ncol(X_train))


# Alinear columnas train–test
cols_comunes   <- intersect(names(X_train_full), names(X_test_full))
X_train_full   <- X_train_full |> select(all_of(cols_comunes))
X_test_full    <- X_test_full  |> select(all_of(cols_comunes))

# Variable respuesta
niveles <- c("No", "Yes")
y_train <- factor(ifelse(train$pobre == 1, "Yes", "No"), levels = niveles)

message("  Predictores disponibles: ", ncol(X_train_full))

# Función: seleccionar columnas del subconjunto que existan en X_train_full
# (por si alguna variable del ranking fue excluida en preprocesamiento)
filtrar_cols <- function(X, vars) {
  vars_ok <- intersect(vars, names(X))
  if (length(vars_ok) < length(vars)) {
    message("  AVISO: ", length(vars) - length(vars_ok),
            " variables del ranking no están en X (probablemente excluidas).")
  }
  X |> select(all_of(vars_ok))
}

# =============================================================================
# SECCIÓN 5: FUNCIÓN AUXILIAR — ENTRENAR RF Y EXTRAER MÉTRICAS OOB
# =============================================================================
# Entrena un ranger con los hiperparámetros fijos y devuelve:
#   - modelo ranger (probability=TRUE, importance="permutation")
#   - threshold óptimo OOB (max F1)
#   - F1, Precision, Recall OOB con ese threshold
#   - AUC-ROC OOB
#   - probabilidades OOB para el threshold search
#
# Parámetros:
#   X_tr   : data.frame de predictores (train)
#   y_tr   : factor respuesta (niveles "No"/"Yes")
#   label  : string identificador para los mensajes
entrenar_rf_oob <- function(X_tr, y_tr, label) {
  message("\n========================================")
  message("Entrenando ", label, "  (", ncol(X_tr), " variables)")
  message("========================================")

  train_df <- bind_cols(X_tr, pobre = y_tr)

  set.seed(42)
  t0 <- Sys.time()

  modelo <- ranger::ranger(
    pobre ~ .,
    data          = train_df,
    num.trees     = NUM_TREES,
    mtry          = min(BEST_MTRY, ncol(X_tr)),  # mtry no puede superar p
    splitrule     = BEST_SPLITRULE,
    min.node.size = BEST_MIN_NODE,
    probability   = TRUE,
    importance    = "permutation",
    seed          = 42
  )

  t1 <- Sys.time()
  message("  Tiempo: ", round(as.numeric(difftime(t1, t0, units = "mins")), 2), " min")
  message("  OOB prediction error: ", round(modelo$prediction.error, 4))

  # Probabilidades OOB
  probs_oob <- modelo$predictions[, "Yes"]

  # Búsqueda del threshold que maximiza F1 sobre OOB
  th_grid <- seq(0.05, 0.95, by = 0.01)

  th_search <- map_dfr(th_grid, function(th) {
    pred_th <- factor(ifelse(probs_oob >= th, "Yes", "No"), levels = niveles)
    TP <- sum(pred_th == "Yes" & y_tr == "Yes")
    FP <- sum(pred_th == "Yes" & y_tr == "No")
    FN <- sum(pred_th == "No"  & y_tr == "Yes")
    f1   <- if ((2*TP + FP + FN) == 0) NA_real_ else 2*TP / (2*TP + FP + FN)
    prec <- if ((TP + FP) == 0)        NA_real_ else TP / (TP + FP)
    rec  <- if ((TP + FN) == 0)        NA_real_ else TP / (TP + FN)
    tibble(threshold = th, F1 = f1, Precision = prec, Recall = rec)
  })

  best_row <- th_search |> filter(!is.na(F1)) |>
    slice_max(F1, n = 1, with_ties = FALSE)

  best_th  <- best_row$threshold
  best_f1  <- best_row$F1
  best_pr  <- best_row$Precision
  best_rc  <- best_row$Recall

  # AUC-ROC OOB
  roc_obj <- pROC::roc(
    response  = as.integer(y_tr == "Yes"),
    predictor = probs_oob,
    quiet     = TRUE
  )
  auc_val <- as.numeric(pROC::auc(roc_obj))

  message("  Threshold óptimo OOB: ", best_th)
  message("  F1 OOB:               ", round(best_f1,  4))
  message("  Precision OOB:        ", round(best_pr,  4))
  message("  Recall OOB:           ", round(best_rc,  4))
  message("  AUC-ROC OOB:          ", round(auc_val,  4))

  list(
    modelo     = modelo,
    best_th    = best_th,
    f1         = best_f1,
    precision  = best_pr,
    recall     = best_rc,
    auc_roc    = auc_val,
    probs_oob  = probs_oob,
    th_search  = th_search,
    n_features = ncol(X_tr)
  )
}

# =============================================================================
# SECCIÓN 6: ENTRENAR MODELOS A (top20), B (top40), C (top60)
# =============================================================================

# -- Modelo A: top 20 --
X_train_20 <- filtrar_cols(X_train_full, vars_top20)
X_test_20  <- filtrar_cols(X_test_full,  vars_top20)
res_20     <- entrenar_rf_oob(X_train_20, y_train, "Modelo A — Top 20")

# -- Modelo B: top 40 --
X_train_40 <- filtrar_cols(X_train_full, vars_top40)
X_test_40  <- filtrar_cols(X_test_full,  vars_top40)
res_40     <- entrenar_rf_oob(X_train_40, y_train, "Modelo B — Top 40")

# -- Modelo C: top 60 --
X_train_60 <- filtrar_cols(X_train_full, vars_top60)
X_test_60  <- filtrar_cols(X_test_full,  vars_top60)
res_60     <- entrenar_rf_oob(X_train_60, y_train, "Modelo C — Top 60")

# =============================================================================
# SECCIÓN 7: SUBMISSIONS
# =============================================================================
message("\n== Generando submissions ==")

generar_submission <- function(modelo_res, X_te, ids, tag) {
  probs_test <- predict(modelo_res$modelo, data = X_te)$predictions[, "Yes"]
  pred_test  <- as.integer(probs_test >= modelo_res$best_th)
  sub_name   <- sprintf("RF_mtry%d_%s_th%03.0f.csv",
                        BEST_MTRY, tag, modelo_res$best_th * 100)
  sub_path   <- file.path(dir_subs, sub_name)
  write_csv(tibble(id = ids, pobre = pred_test), sub_path)
  message("  ", sub_path,
          "  | pobres predichos: ", sum(pred_test),
          " (", round(mean(pred_test) * 100, 1), "%)")
  sub_path
}

sub_20 <- generar_submission(res_20, X_test_20, ids, "top20")
sub_40 <- generar_submission(res_40, X_test_40, ids, "top40")
sub_60 <- generar_submission(res_60, X_test_60, ids, "top60")

# =============================================================================
# SECCIÓN 8: COMPARACIÓN FINAL DE MÉTRICAS
# =============================================================================
# Tabla con F1, Precision, Recall y AUC-ROC de los tres modelos,
# evaluados sobre predicciones OOB (estimación honesta, fuera de muestra).
# Incluye referencia de RF_001 (todas las variables) si está disponible.
message("\n========================================")
message("COMPARACIÓN DE MODELOS — FEATURE SELECTION")
message("========================================")

comparacion <- tibble(
  Modelo          = c("A — Top 20", "B — Top 40", "C — Top 60"),
  N_vars          = c(res_20$n_features, res_40$n_features, res_60$n_features),
  Threshold       = c(res_20$best_th,   res_40$best_th,    res_60$best_th),
  F1_OOB          = c(res_20$f1,        res_40$f1,         res_60$f1),
  Precision_OOB   = c(res_20$precision, res_40$precision,  res_60$precision),
  Recall_OOB      = c(res_20$recall,    res_40$recall,     res_60$recall),
  AUC_ROC_OOB     = c(res_20$auc_roc,   res_40$auc_roc,    res_60$auc_roc)
)

# Intentar agregar RF_001 como referencia si el registry está disponible
if (file.exists(reg_path)) {
  registry <- read_csv(reg_path, show_col_types = FALSE)
  rf001 <- registry |> filter(model_id == "RF_001")
  if (nrow(rf001) > 0) {
    rf001_row <- tibble(
      Modelo        = "RF_001 (todas)",
      N_vars        = rf001$n_features[1],
      Threshold     = rf001$threshold[1],
      F1_OOB        = rf001$cv_F1[1],
      Precision_OOB = rf001$cv_Precision[1],
      Recall_OOB    = rf001$cv_Recall[1],
      AUC_ROC_OOB   = rf001$auc_roc[1]
    )
    comparacion <- bind_rows(rf001_row, comparacion)
  }
}

# Formatear para impresión
comparacion_fmt <- comparacion |>
  mutate(across(where(is.double) & !Threshold, ~ round(., 4)),
         Threshold = round(Threshold, 2))

cat("\n")
cat(strrep("=", 80), "\n")
cat("  COMPARACIÓN: F1 / Precision / Recall / AUC-ROC  (métricas OOB — honestas)\n")
cat(strrep("=", 80), "\n")
print(comparacion_fmt, n = Inf)
cat(strrep("-", 80), "\n")

# Identificar el mejor modelo por F1
best_model <- comparacion_fmt |> slice_max(F1_OOB, n = 1, with_ties = FALSE)
cat(sprintf("\n  Mejor F1 OOB: %s  →  F1 = %.4f | Precision = %.4f | Recall = %.4f | AUC = %.4f\n\n",
            best_model$Modelo,
            best_model$F1_OOB,
            best_model$Precision_OOB,
            best_model$Recall_OOB,
            best_model$AUC_ROC_OOB))

# Guardar tabla de comparación en CSV
comp_path <- file.path(dir_model, "comparacion_features.csv")
write_csv(comparacion_fmt, comp_path)
message("  Tabla guardada: ", comp_path)

# =============================================================================
# SECCIÓN 9: REGISTRO EN model_registry.csv
# =============================================================================
# Registra los tres modelos como RF_002_top20, RF_002_top40, RF_002_top60.
message("\n== Actualizando model_registry.csv ==")

registrar_modelo <- function(tag, res, sub_path) {
  mid <- paste0(MODEL_ID, "_", tag)
  tibble(
    model_id           = mid,
    fecha              = Sys.Date(),
    autor              = AUTOR,
    algoritmo          = "RandomForest",
    n_features         = res$n_features,
    imbalance_strategy = "none",
    cv_folds           = NA_integer_,   # sin CV — evaluación OOB
    cv_F1              = round(res$f1,        4),
    cv_Precision       = round(res$precision, 4),
    cv_Recall          = round(res$recall,    4),
    auc_roc            = round(res$auc_roc,   4),
    kaggle_public_F1   = NA_real_,
    threshold          = res$best_th,
    notas              = paste0(
      "RF ranger. Feature selection por importancia RF_001.",
      " Subconjunto: ", tag, " (", res$n_features, " vars).",
      " mtry=", BEST_MTRY,
      " min_node_size=", BEST_MIN_NODE,
      " splitrule=gini num.trees=", NUM_TREES,
      ". Threshold optimizado sobre probs OOB (max F1).",
      " th=", res$best_th,
      " F1_OOB=", round(res$f1, 4),
      " AUC_OOB=", round(res$auc_roc, 4), "."
    ),
    cp              = NA_real_,
    maxdepth        = NA_real_,
    train_F1        = NA_real_,
    train_Precision = NA_real_,
    train_Recall    = NA_real_
  )
}

filas_nuevas <- bind_rows(
  registrar_modelo("top20", res_20, sub_20),
  registrar_modelo("top40", res_40, sub_40),
  registrar_modelo("top60", res_60, sub_60)
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
