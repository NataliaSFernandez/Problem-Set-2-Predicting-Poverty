#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/XGBoost/XGB_003.R
#==============================================================================
#
# ALGORITMO: XGBoost — Búsqueda aleatoria COMPLETA de hiperparámetros
#                     + Variables geográficas (ciudad / dpto)
#                     + Feature selection basado en importancia propia de XGBoost
#
# MOTIVACIÓN:
#   XGB_001 tuneó los HPs con una grilla reducida (8 combos) dejando algunos
#   parámetros fijos. XGB_002 heredó esos mismos HPs para comparar subconjuntos
#   de features, pero excluyó ciudad y dpto del set de predictores y dependía
#   del ranking de importancia del Random Forest (RF_001).
#
#   Este script es completamente autónomo — no depende de ningún otro modelo:
#
#   (a) BÚSQUEDA ALEATORIA SOBRE EL ESPACIO COMPLETO DE HIPERPARÁMETROS
#       En lugar de fijar algunos HPs de antemano, se muestrea aleatoriamente
#       el espacio de todos los HPs relevantes de XGBoost:
#
#         max_depth        : profundidad máxima del árbol (estructura)
#         eta              : tasa de aprendizaje (shrinkage)
#         gamma            : reducción mínima de pérdida para hacer una split
#         min_child_weight : peso mínimo acumulado requerido en un nodo hoja
#         colsample_bytree : fracción de columnas muestreadas por árbol
#         subsample        : fracción de filas muestreadas por iteración
#         lambda           : regularización L2 (ridge) sobre pesos de hoja
#         alpha            : regularización L1 (lasso) sobre pesos de hoja
#         nrounds          : determinado automáticamente via early stopping
#                            (early stopping sobre logloss de validación)
#
#       Para cada combinación aleatoria:
#         → CV-5 con early stopping → nrounds óptimo por fold
#         → Predicciones OOF → threshold óptimo (max F1 OOF)
#         → Registro de F1, Precision, Recall, AUC-ROC
#
#   (b) INCLUSIÓN DE VARIABLES GEOGRÁFICAS
#       ciudad y dpto se codifican como enteros ordinales (as.integer sobre
#       factor). XGBoost puede hallar particiones sobre esos códigos sin
#       necesidad de one-hot encoding (los árboles hacen splits binarios sobre
#       valores numéricos → equivalente a agrupar ciudades).
#
#   (c) FEATURE SELECTION BASADO EN IMPORTANCIA PROPIA DE XGBOOST
#       Tras encontrar los mejores HPs (Fase 1), se entrena un modelo
#       "de diagnóstico" sobre todas las variables para calcular la importancia
#       nativa de XGBoost (métrica: Gain = reducción promedio de pérdida
#       aportada por cada variable en todos los splits que la usan).
#       Los Top-20, Top-40 y Top-60 se derivan de ese ranking interno,
#       por lo que el feature selection es 100% propio de XGBoost.
#
# FLUJO EN TRES FASES:
#
#   FASE 1 — Búsqueda aleatoria de hiperparámetros (sobre todas las variables)
#     N_SEARCH combinaciones aleatorias evaluadas con CV-5 + early stopping.
#     Métrica de selección: F1 OOF. Se eligen los mejores HPs.
#
#   FASE 1.5 — Ranking de importancia XGBoost
#     Con los mejores HPs de Fase 1, se entrena un modelo sobre todos los
#     datos y se extraen las importancias (Gain) con xgb.importance().
#     Se derivan los subconjuntos Top-20, Top-40, Top-60.
#
#   FASE 2 — Feature selection con mejores HPs y ranking de Gain acumulado
#     Con los HPs óptimos de Fase 1, evaluar subconjuntos por umbral de Gain:
#       Baseline : TODAS las variables disponibles
#       gain60   : variables que acumulan >= 60% del Gain total
#       gain70   : variables que acumulan >= 70% del Gain total
#       gain80   : variables que acumulan >= 80% del Gain total
#       gain90   : variables que acumulan >= 90% del Gain total
#       gain95   : variables que acumulan >= 95% del Gain total
#       gain99   : variables que acumulan >= 99% del Gain total
#     Cada umbral corresponde a un número distinto de variables, determinado
#     endógenamente por la distribución de importancias del modelo.
#     Métrica: F1 OOF (CV-5, mismo esquema que Fase 1).
#
#   RESULTADO:
#     El mejor modelo global (HPs × subconjunto) genera el submission
#     y queda registrado en model_registry.csv.
#
# MÉTRICA DE OPTIMIZACIÓN: F1-score calculado sobre predicciones OOF (CV-5).
#   Sin data leakage: cada observación es predicha por modelos que no la
#   vieron. El threshold óptimo también se determina sobre OOF.
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#   02_outputs/model_registry.csv
#   (No depende de ningún modelo previo — script autónomo)
#
# OUTPUTS:
#   02_outputs/models/XGBoost/XGB_003/hp_search_results.csv
#   02_outputs/models/XGBoost/XGB_003/xgb_importance.csv      ← nuevo
#   02_outputs/models/XGBoost/XGB_003/feature_selection_results.csv
#   02_outputs/models/XGBoost/XGB_003/diagnostics.rds
#   03_submissions/XGB_003_<tag>.csv          ← submission del mejor modelo
#   02_outputs/model_registry.csv             ← append (solo el mejor modelo)
#
# NOTA DE TIEMPO:
#   Con N_SEARCH=30 y 164K filas de train, Fase 1 puede tardar ~20-50 min
#   dependiendo del hardware. Reducir N_SEARCH para pruebas rápidas.
#
# REPRODUCIBILIDAD:
#   Ejecutar desde la raíz del proyecto. Semilla global: 42.
#   Script completamente autónomo — no requiere ningún script previo.
#==============================================================================
# nolint start

# =============================================================================
# SECCIÓN 0: PAQUETES
# =============================================================================
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,   # dplyr, ggplot2, purrr, tibble, readr, stringr
  caret,       # createFolds() — partición estratificada reproducible
  xgboost,     # xgb.DMatrix(), xgb.train() — API nativa (compatible v3.x)
  pROC,        # roc(), auc() — cálculo de AUC-ROC sobre predicciones OOF
  fs           # dir_create() — creación segura de directorios
)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
AUTOR    <- "Jonathan Melo"
MODEL_ID <- "XGB_003"

set.seed(42)

# ---- Recursos de cómputo ----------------------------------------------------
# XGBoost es paralelizable por árbol. Se usan todos los núcleos menos uno
# para no saturar el sistema operativo.
N_THREADS <- max(1L, parallel::detectCores() - 1L)

# ---- Parámetros de búsqueda de hiperparámetros ------------------------------
# N_SEARCH : número de combinaciones aleatorias a evaluar en Fase 1.
#   - Más combinaciones → mejor búsqueda, mayor tiempo de cómputo.
#   - 30 es un buen balance para este problema; usar 50-100 para más rigor.
# MAX_NROUNDS : techo de iteraciones por fit; early stopping suele detenerse
#   mucho antes, especialmente con eta pequeño.
# EARLY_STOP  : rondas consecutivas sin mejora en logloss de validación → stop.
N_SEARCH    <- 30L
MAX_NROUNDS <- 500L
EARLY_STOP  <- 25L

# ---- Directorios de salida --------------------------------------------------
dir_model <- file.path("02_outputs/models/XGBoost", MODEL_ID)
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model, recurse = TRUE)
fs::dir_create(dir_subs,  recurse = TRUE)

# Niveles del factor respuesta (clase negativa primero → positiva)
niveles <- c("No", "Yes")

message("=======================================================")
message("  ", MODEL_ID, " — ", AUTOR)
message("  Núcleos XGBoost (nthread)  : ", N_THREADS)
message("  Combinaciones HP (N_SEARCH): ", N_SEARCH)
message("  Max rondas / Early stop    : ", MAX_NROUNDS, " / ", EARLY_STOP)
message("=======================================================")

# =============================================================================
# SECCIÓN 2: CARGAR DATOS
# =============================================================================
message("\n== SECCIÓN 2: Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id   # IDs del test para el submission

message("  Train: ", nrow(train), " filas x ", ncol(train), " columnas")
message("  Test : ", nrow(test),  " filas x ", ncol(test),  " columnas")

# Variable respuesta como factor (para caret/métricas) y como entero (xgboost)
y_factor <- factor(ifelse(train$pobre == 1, "Yes", "No"), levels = niveles)
y_int    <- as.integer(y_factor == "Yes")   # 1 = pobre, 0 = no pobre

# Resumen del balance de clases — útil para interpretar el threshold óptimo
n_pos <- sum(y_int)
n_neg <- length(y_int) - n_pos
message("  Balance: ", n_pos, " pobres (", round(100 * mean(y_int), 1), "%) | ",
        n_neg, " no-pobres (", round(100 * (1 - mean(y_int)), 1), "%)")

# =============================================================================
# SECCIÓN 3: (ELIMINADA — sin dependencia de RF_001)
# =============================================================================
# El ranking de importancia de variables se calcula en Sección 7.5 usando
# la importancia nativa de XGBoost (Gain) con los mejores hiperparámetros
# encontrados en Fase 1. Los subconjuntos Top-20, Top-40, Top-60 se
# derivan de ese ranking propio. No se requiere ningún modelo externo.
# =============================================================================
message("\n== SECCIÓN 3: Ranking de importancia — se calculará en Sección 7.5 (XGBoost Gain) ==")

# =============================================================================
# SECCIÓN 4: PREPROCESAMIENTO — INCLUSIÓN DE CIUDAD Y DPTO
# =============================================================================
# XGBoost requiere matrices 100% numéricas. Diferencia fundamental vs.
# XGB_001 / XGB_002:
#
#   ANTES: select(-any_of(c("ciudad", "dpto")))  → se descartaban
#   AHORA: as.integer(factor(.))                 → se conservan como entero
#
# Razonamiento para codificación ordinal (as.integer) vs. one-hot encoding:
#   - XGBoost usa árboles con splits binarios sobre valores numéricos.
#   - Un código entero 1:K permite al árbol descubrir agrupaciones de ciudades
#     con comportamiento similar (e.g., municipios rurales vs. urbanos),
#     sin el costo dimensional de 25+ columnas dummies.
#   - En la práctica, la diferencia de rendimiento entre ambos enfoques es
#     pequeña para XGBoost, pero la codificación ordinal es más eficiente.
# =============================================================================
message("\n== SECCIÓN 4: Preprocesando variables (ciudad y dpto INCLUIDAS) ==")

preparar_X_xgb <- function(df) {
  df |>
    # Paso 1: Eliminar identificador y variable respuesta
    select(-any_of(c("id", "pobre"))) |>
    # Paso 2: zona ya es numérica (0=rural/1=urbana); convertir por seguridad
    mutate(zona = as.numeric(zona)) |>
    # Paso 3: ciudad y dpto — factor → entero ordinal (DIFERENCIA vs. XGB_001/002)
    mutate(
      ciudad = as.integer(factor(ciudad)),   # 25 ciudades → códigos 1-25
      dpto   = as.integer(factor(dpto))      # 24 departamentos → códigos 1-24
    ) |>
    # Paso 4: Cualquier columna character residual → numérico
    mutate(across(where(is.character), as.numeric)) |>
    # Paso 5: Factores restantes → entero
    mutate(across(where(is.factor), as.integer))
}

X_train_full <- preparar_X_xgb(train)
X_test_full  <- preparar_X_xgb(test)

# Garantizar columnas comunes entre train y test (por si alguna variable
# categórica genera niveles distintos en train vs. test)
cols_comunes <- intersect(names(X_train_full), names(X_test_full))
X_train_full <- X_train_full |> select(all_of(cols_comunes))
X_test_full  <- X_test_full  |> select(all_of(cols_comunes))

message("  Predictores disponibles: ", ncol(X_train_full))
message("  ciudad incluida: ", "ciudad" %in% names(X_train_full),
        "  |  dpto incluida: ", "dpto" %in% names(X_train_full))

# Matriz de train completa (para Fase 1)
mat_train_full <- as.matrix(X_train_full)
mat_test_full  <- as.matrix(X_test_full)

# Función auxiliar: seleccionar subconjunto de features
#   vars    : vector de nombres de variables (desde ranking XGBoost Gain); NULL = todas
#   add_geo : si TRUE, agrega ciudad y dpto aunque no estén en vars
filtrar_cols_geo <- function(X_tr, X_te, vars = NULL, add_geo = TRUE) {
  if (is.null(vars)) {
    # "All" → todas las columnas disponibles (incluye ciudad/dpto si ya están)
    return(list(tr = as.matrix(X_tr), te = as.matrix(X_te),
                n = ncol(X_tr), cols = names(X_tr)))
  }
  vars_ok <- intersect(vars, names(X_tr))
  if (add_geo) {
    geo_vars <- intersect(c("ciudad", "dpto"), names(X_tr))
    vars_ok  <- unique(c(vars_ok, geo_vars))
  }
  n_miss <- length(vars) - length(intersect(vars, names(X_tr)))
  if (n_miss > 0) {
    message("  AVISO: ", n_miss, " variable(s) del ranking no disponibles.")
  }
  list(
    tr   = as.matrix(X_tr |> select(all_of(vars_ok))),
    te   = as.matrix(X_te |> select(all_of(vars_ok))),
    n    = length(vars_ok),
    cols = vars_ok
  )
}

# Los subconjuntos de features a evaluar en Fase 2 se definen en Sección 7.5,
# una vez calculada la importancia XGBoost (Gain acumulado). No se definen aquí
# porque el número de variables por umbral depende de la distribución real de
# importancias, que solo se conoce tras entrenar el modelo de diagnóstico.

# =============================================================================
# SECCIÓN 5: FOLDS ESTRATIFICADOS (compartidos por ambas fases)
# =============================================================================
# Se crean UNA SOLA VEZ y se reutilizan en Fase 1 y Fase 2.
# Esto garantiza que la comparación entre combinaciones de HPs y entre
# subconjuntos de features sea completamente justa (mismo split de datos).
set.seed(42)
folds <- caret::createFolds(y_factor, k = 5, list = TRUE, returnTrain = TRUE)

message("\n  Folds CV-5 estratificados creados.")
message("  Tamaños (train por fold): ",
        paste(sapply(folds, length), collapse = " / "))

# =============================================================================
# SECCIÓN 6: FUNCIONES AUXILIARES
# =============================================================================

# -----------------------------------------------------------------------------
# 6a. calcular_metricas_oof()
# Dadas las probabilidades OOF y los labels reales, busca el threshold en
# [0.05, 0.95] que maximiza F1. Devuelve una fila con threshold, F1, Precision
# y Recall óptimos, más el AUC-ROC.
# -----------------------------------------------------------------------------
calcular_metricas_oof <- function(probs_oof, obs_oof_factor, obs_oof_int) {
  # Búsqueda del threshold óptimo para F1
  best <- map_dfr(seq(0.05, 0.95, by = 0.01), function(th) {
    pred <- factor(ifelse(probs_oof >= th, "Yes", "No"), levels = niveles)
    TP   <- sum(pred == "Yes" & obs_oof_factor == "Yes")
    FP   <- sum(pred == "Yes" & obs_oof_factor == "No")
    FN   <- sum(pred == "No"  & obs_oof_factor == "Yes")
    f1   <- if ((2 * TP + FP + FN) == 0) NA_real_ else 2 * TP / (2 * TP + FP + FN)
    prec <- if ((TP + FP) == 0)           NA_real_ else TP / (TP + FP)
    rec  <- if ((TP + FN) == 0)           NA_real_ else TP / (TP + FN)
    tibble(threshold = th, F1 = f1, Precision = prec, Recall = rec)
  }) |>
    filter(!is.na(F1)) |>
    slice_max(F1, n = 1, with_ties = FALSE)

  # AUC-ROC
  roc_obj <- pROC::roc(response  = obs_oof_int,
                        predictor = probs_oof,
                        quiet     = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))

  list(
    threshold = best$threshold,
    F1        = best$F1,
    Precision = best$Precision,
    Recall    = best$Recall,
    AUC_ROC   = auc_val
  )
}

# -----------------------------------------------------------------------------
# 6b. cv5_xgb_oof()
# Ejecuta CV-5 manual con early stopping por fold sobre la matriz X_mat.
# Devuelve predicciones OOF (vector numérico, mismo orden que X_mat),
# el número de rondas por fold, y la mediana como nrounds recomendado
# para reentrenamiento sobre el dataset completo.
#
# Diseño:
#   - Early stopping sobre logloss de validación (métrica nativa de XGBoost)
#     → eficiente y estable. La optimización de F1 se hace ex-post sobre OOF.
#   - Se usa la MEDIANA de best_iteration a través de los folds como nrounds
#     para el modelo final, lo que es más robusto que el máximo o el promedio.
#
# Parámetros:
#   params   : lista de HPs de xgboost (sin nthread)
#   X_mat    : matrix numérica de predictores (train)
#   y_int    : vector entero 0/1 (variable respuesta)
#   y_factor : factor "No"/"Yes" (para log por fold, opcional)
#   folds    : lista de índices returnTrain de caret::createFolds()
#   max_nr   : máximo de rondas (techo para early stopping)
#   early_s  : rondas sin mejora → parar
#   nthread  : número de hilos a usar
#   verbose_f: si TRUE, imprime F1(th=0.5) por fold
# -----------------------------------------------------------------------------
cv5_xgb_oof <- function(params, X_mat, y_int, y_factor, folds,
                         max_nr, early_s, nthread, verbose_f = FALSE) {
  n           <- nrow(X_mat)
  oof_probs   <- numeric(n)          # contenedor de probs OOF
  nr_por_fold <- integer(length(folds))  # best_iteration por fold

  params_full <- c(params, list(nthread = nthread))

  for (f in seq_along(folds)) {
    tr_idx  <- folds[[f]]
    val_idx <- setdiff(seq_len(n), tr_idx)

    dtrain_f <- xgboost::xgb.DMatrix(X_mat[tr_idx,  ], label = y_int[tr_idx])
    dval_f   <- xgboost::xgb.DMatrix(X_mat[val_idx, ], label = y_int[val_idx])

    # Entrenar con early stopping sobre logloss del fold de validación
    m_f <- xgboost::xgb.train(
      params                = params_full,
      data                  = dtrain_f,
      nrounds               = max_nr,
      watchlist             = list(val = dval_f),
      early_stopping_rounds = early_s,
      verbose               = 0   # silencioso; solo reportamos por fold abajo
    )

    # En XGBoost >= 2.x, best_iteration ya no es un campo directo del modelo
    # sino un atributo del Booster. Se usa xgb.attr() como método robusto.
    # xgb.attr() devuelve string → as.integer(). Fallback: max_nr si es NULL.
    best_iter_raw      <- xgboost::xgb.attr(m_f, "best_iteration")
    nr_por_fold[f]     <- if (!is.null(best_iter_raw)) as.integer(best_iter_raw) else max_nr
    oof_probs[val_idx] <- predict(m_f, dval_f)

    if (verbose_f) {
      pred_val <- factor(ifelse(oof_probs[val_idx] >= 0.5, "Yes", "No"),
                         levels = niveles)
      obs_val  <- y_factor[val_idx]
      TP <- sum(pred_val == "Yes" & obs_val == "Yes")
      FP <- sum(pred_val == "Yes" & obs_val == "No")
      FN <- sum(pred_val == "No"  & obs_val == "Yes")
      f1_f <- if ((2 * TP + FP + FN) == 0) NA_real_ else 2 * TP / (2 * TP + FP + FN)
      message(sprintf("    Fold %d: best_iter=%3d  F1(th=0.5)=%.4f",
                      f, nr_por_fold[f], f1_f))
    }
  }

  list(
    oof_probs        = oof_probs,
    nrounds_per_fold = nr_por_fold,
    # Mediana como estimado robusto de los nrounds óptimos para el modelo final
    best_nrounds     = as.integer(ceiling(median(nr_por_fold)))
  )
}

# =============================================================================
# SECCIÓN 7: FASE 1 — BÚSQUEDA ALEATORIA DE HIPERPARÁMETROS
# =============================================================================
# La búsqueda aleatoria (Bergstra & Bengio, 2012) es más eficiente que la
# búsqueda en grilla para espacios de alta dimensión: con el mismo presupuesto
# de evaluaciones, cubre más regiones del espacio de HPs porque no desperdicia
# evaluaciones en combinaciones redundantes por alineación de grilla.
#
# Espacio de búsqueda:
#   max_depth        [3, 4, 5, 6, 8, 10]
#   eta              [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
#   gamma            [0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
#   min_child_weight [1, 3, 5, 8, 10, 15, 20]
#   colsample_bytree [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
#   subsample        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#   lambda (L2)      [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
#   alpha  (L1)      [0.0, 0.05, 0.1, 0.5, 1.0, 2.0]
#   nrounds          → early stopping (no se especifica a priori)
# =============================================================================
message("\n")
message(strrep("=", 70))
message("  FASE 1 — BÚSQUEDA ALEATORIA DE HIPERPARÁMETROS")
message("  N_SEARCH = ", N_SEARCH, " combos  |  MAX_NROUNDS = ", MAX_NROUNDS,
        "  |  EARLY_STOP = ", EARLY_STOP)
message("  Variables: TODAS (incluye ciudad y dpto)")
message(strrep("=", 70))

# Muestreo aleatorio del espacio de hiperparámetros
# set.seed garantiza reproducibilidad del orden de muestreo
set.seed(42)
hp_grid <- tibble(
  max_depth        = sample(c(3L, 4L, 5L, 6L, 8L, 10L),
                            N_SEARCH, replace = TRUE),
  eta              = sample(c(0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20),
                            N_SEARCH, replace = TRUE),
  gamma            = sample(c(0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0),
                            N_SEARCH, replace = TRUE),
  min_child_weight = sample(c(1L, 3L, 5L, 8L, 10L, 15L, 20L),
                            N_SEARCH, replace = TRUE),
  colsample_bytree = sample(c(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0),
                            N_SEARCH, replace = TRUE),
  subsample        = sample(c(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                            N_SEARCH, replace = TRUE),
  lambda           = sample(c(0.0, 0.1, 0.5, 1.0, 2.0, 5.0),
                            N_SEARCH, replace = TRUE),
  alpha            = sample(c(0.0, 0.05, 0.1, 0.5, 1.0, 2.0),
                            N_SEARCH, replace = TRUE)
)

message("\n  Grilla de búsqueda generada (primeras 3 filas a modo de ejemplo):")
print(head(hp_grid, 3))

# Contenedor de resultados
hp_results_list <- vector("list", N_SEARCH)
t_fase1 <- Sys.time()

for (i in seq_len(N_SEARCH)) {
  hp_i <- hp_grid[i, ]

  message(sprintf(
    "\n  [HP %2d/%d] depth=%2d  eta=%.3f  gamma=%.2f  mcw=%2d  "  ,
    i, N_SEARCH, hp_i$max_depth, hp_i$eta, hp_i$gamma,
    hp_i$min_child_weight), appendLF = FALSE)
  message(sprintf(
    "col=%.2f  sub=%.2f  L2=%.2f  L1=%.3f",
    hp_i$colsample_bytree, hp_i$subsample, hp_i$lambda, hp_i$alpha))

  params_i <- list(
    objective        = "binary:logistic",
    eval_metric      = "logloss",
    max_depth        = hp_i$max_depth,
    eta              = hp_i$eta,
    gamma            = hp_i$gamma,
    min_child_weight = hp_i$min_child_weight,
    colsample_bytree = hp_i$colsample_bytree,
    subsample        = hp_i$subsample,
    lambda           = hp_i$lambda,
    alpha            = hp_i$alpha
  )

  # CV-5 con early stopping para esta combinación de HPs
  cv_i <- tryCatch(
    cv5_xgb_oof(
      params    = params_i,
      X_mat     = mat_train_full,
      y_int     = y_int,
      y_factor  = y_factor,
      folds     = folds,
      max_nr    = MAX_NROUNDS,
      early_s   = EARLY_STOP,
      nthread   = N_THREADS,
      verbose_f = FALSE    # silencioso en búsqueda para reducir output
    ),
    error = function(e) {
      message("    !! ERROR en combo ", i, ": ", conditionMessage(e))
      NULL
    }
  )

  if (is.null(cv_i)) next

  # Métricas OOF para esta combinación
  met_i <- calcular_metricas_oof(cv_i$oof_probs, y_factor, y_int)

  hp_results_list[[i]] <- bind_cols(
    hp_i,
    tibble(
      best_nrounds = cv_i$best_nrounds,
      nrounds_sd   = round(sd(cv_i$nrounds_per_fold), 1),
      F1_OOF       = round(met_i$F1,        4),
      Precision    = round(met_i$Precision,  4),
      Recall       = round(met_i$Recall,     4),
      AUC_ROC      = round(met_i$AUC_ROC,    4),
      Threshold    = met_i$threshold
    )
  )

  message(sprintf("    → nrounds_mediana=%d (sd=%.1f)  F1_OOF=%.4f  "        ,
                  cv_i$best_nrounds, sd(cv_i$nrounds_per_fold), met_i$F1),
          appendLF = FALSE)
  message(sprintf("AUC=%.4f  th=%.2f", met_i$AUC_ROC, met_i$threshold))
}

t_fase1_end <- Sys.time()
message(sprintf(
  "\n  FASE 1 completada en %.1f minutos.",
  as.numeric(difftime(t_fase1_end, t_fase1, units = "mins"))
))

# Consolidar y ordenar resultados
hp_results <- bind_rows(hp_results_list) |> arrange(desc(F1_OOF))

# Guardar tabla completa de resultados de búsqueda
write_csv(hp_results, file.path(dir_model, "hp_search_results.csv"))
message("  Tabla de búsqueda HP guardada: hp_search_results.csv")

message("\n  Top-5 combinaciones por F1 OOF:")
print(
  hp_results |>
    select(max_depth:alpha, best_nrounds, F1_OOF, AUC_ROC, Threshold) |>
    head(5),
  n = 5
)

# Extraer la mejor combinación de HPs
best_hp <- hp_results |> slice_max(F1_OOF, n = 1, with_ties = FALSE)

# Variables globales para los mejores HPs (usadas en Fase 2 y en el registro)
BEST_MAX_DEPTH        <- best_hp$max_depth
BEST_ETA              <- best_hp$eta
BEST_GAMMA            <- best_hp$gamma
BEST_MIN_CHILD_WEIGHT <- best_hp$min_child_weight
BEST_COLSAMPLE        <- best_hp$colsample_bytree
BEST_SUBSAMPLE        <- best_hp$subsample
BEST_LAMBDA           <- best_hp$lambda
BEST_ALPHA            <- best_hp$alpha
BEST_NROUNDS_HP       <- best_hp$best_nrounds   # nrounds para "All features" (Fase 1)

# Lista de HPs para pasar a xgb.train / cv5_xgb_oof
best_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "logloss",
  max_depth        = BEST_MAX_DEPTH,
  eta              = BEST_ETA,
  gamma            = BEST_GAMMA,
  min_child_weight = BEST_MIN_CHILD_WEIGHT,
  colsample_bytree = BEST_COLSAMPLE,
  subsample        = BEST_SUBSAMPLE,
  lambda           = BEST_LAMBDA,
  alpha            = BEST_ALPHA
)

message("\n  ★ MEJORES HIPERPARÁMETROS (Fase 1):")
message("    max_depth        = ", BEST_MAX_DEPTH)
message("    eta              = ", BEST_ETA)
message("    gamma            = ", BEST_GAMMA)
message("    min_child_weight = ", BEST_MIN_CHILD_WEIGHT)
message("    colsample_bytree = ", BEST_COLSAMPLE)
message("    subsample        = ", BEST_SUBSAMPLE)
message("    lambda (L2)      = ", BEST_LAMBDA)
message("    alpha  (L1)      = ", BEST_ALPHA)
message("    nrounds (med.)   = ", BEST_NROUNDS_HP)
message("    F1_OOF           = ", best_hp$F1_OOF)

# =============================================================================
# SECCIÓN 7.5: RANKING DE IMPORTANCIA PROPIO DE XGBOOST (Fase 1.5)
# =============================================================================
# Con los mejores HPs de Fase 1, se entrena un modelo "de diagnóstico" sobre
# TODOS los datos de train. Luego se extraen las importancias nativas de
# XGBoost para derivar los subconjuntos de Fase 2.
#
# MÉTRICA: Gain (importancia por reducción de pérdida)
#   Gain = reducción promedio de logloss aportada por cada variable en todos
#   los splits que la utilizan, ponderada por el número de observaciones.
#   Es la métrica más interpretable de XGBoost: mide cuánto "mejora" el
#   modelo cada variable, no solo cuán frecuentemente aparece en los árboles.
#
# CRITERIO DE CORTE: Gain acumulado normalizado
#   En lugar de cortes arbitrarios (top-20, top-40, top-60), se usan umbrales
#   de Gain acumulado: se retienen las k* variables que explican al menos X%
#   del Gain total. Esto da un criterio continuo que se adapta a la
#   distribución real de importancias del modelo.
#
#   Umbrales evaluados: 60%, 70%, 80%, 90%, 95%, 99%  (+ "All" como base)
#
#   Ejemplo de lectura: si el umbral al 80% requiere 15 variables y al 90%
#   requiere 30 variables, sabemos que hay un salto importante entre esas
#   variables: la del rango 16-30 aportan poco Gain individual pero en
#   conjunto suman el 10% restante.
#
# NOTA SOBRE CIUDAD Y DPTO:
#   Al estar incluidas en el preprocesamiento, aparecen naturalmente en el
#   ranking según su Gain real. No se fuerza su inclusión: si tienen Gain > 0
#   entrarán en los subconjuntos donde el umbral las alcanza.
# =============================================================================
message("\n")
message(strrep("=", 70))
message("  SECCIÓN 7.5 — RANKING DE IMPORTANCIA XGBOOST (Gain acumulado)")
message("  Entrenando modelo de diagnóstico sobre todas las variables...")
message(strrep("=", 70))

# Entrenar modelo de diagnóstico: todos los datos, mejores HPs, nrounds de Fase 1
set.seed(42)
dtrain_diag <- xgboost::xgb.DMatrix(mat_train_full, label = y_int)
modelo_diag <- xgboost::xgb.train(
  params  = c(best_params, list(nthread = N_THREADS)),
  data    = dtrain_diag,
  nrounds = BEST_NROUNDS_HP,
  verbose = 0
)

# Extraer importancia por Gain y calcular Gain acumulado normalizado
# xgb.importance() devuelve: Feature, Gain, Cover, Frequency
xgb_imp <- xgboost::xgb.importance(model = modelo_diag) |>
  as_tibble() |>
  rename(variable = Feature) |>
  arrange(desc(Gain)) |>
  mutate(
    rank_gain   = row_number(),
    gain_norm   = Gain / sum(Gain),        # fracción del Gain total
    gain_cumul  = cumsum(gain_norm)        # Gain acumulado normalizado
  )

# Guardar tabla completa de importancia
write_csv(xgb_imp, file.path(dir_model, "xgb_importance.csv"))
message("  Importancia guardada: xgb_importance.csv")
message("  Variables con Gain > 0: ", nrow(xgb_imp), " de ",
        ncol(X_train_full), " totales")

# Mostrar top-15 y presencia de ciudad/dpto
message("  Top-15 variables por Gain:")
print(
  xgb_imp |>
    select(rank_gain, variable, Gain, gain_norm, gain_cumul) |>
    head(15) |>
    mutate(across(c(Gain, gain_norm, gain_cumul), ~ round(., 5))),
  n = 15
)

geo_rank <- xgb_imp |>
  filter(variable %in% c("ciudad", "dpto")) |>
  select(variable, rank_gain, Gain, gain_cumul)
if (nrow(geo_rank) > 0) {
  message("  Variables geográficas en el ranking XGBoost:")
  print(geo_rank |> mutate(Gain = round(Gain, 6), gain_cumul = round(gain_cumul, 4)))
} else {
  message("  AVISO: ciudad/dpto no usadas en ningún split (Gain = 0).")
}

# ── Construcción de los subconjuntos por umbral de Gain acumulado ──────────
# Para cada umbral X%, se retienen las primeras k* variables (ordenadas por
# Gain desc) tales que su Gain acumulado alcanza o supera X%.
# Esto da un subconjunto mínimo que captura X% de la capacidad predictiva
# total medida por Gain.
GAIN_THRESHOLDS <- c(0.60, 0.70, 0.80, 0.90, 0.95, 0.99)

# "All" = todos los predictores disponibles (incluidos los de Gain = 0)
feature_sets <- list(
  All = list(vars = NULL, label = "Modelo — Todas las variables (baseline)")
)

for (th in GAIN_THRESHOLDS) {
  # Índice de la primera variable cuyo gain_cumul supera el umbral
  idx_corte <- which(xgb_imp$gain_cumul >= th)[1]
  vars_th   <- xgb_imp$variable[seq_len(idx_corte)]
  key       <- paste0("gain", sprintf("%02d", round(th * 100)))
  feature_sets[[key]] <- list(
    vars  = vars_th,
    label = sprintf("Gain acum. >= %2.0f%%  (%d variables)", th * 100, length(vars_th))
  )
}

# Mostrar resumen de los umbrales
message("\n  Subconjuntos definidos por umbral de Gain acumulado:")
for (nm in names(feature_sets)) {
  fs <- feature_sets[[nm]]
  n_v <- if (is.null(fs$vars)) ncol(X_train_full) else length(fs$vars)
  message(sprintf("    %-12s  %3d vars  —  %s", nm, n_v, fs$label))
}

# =============================================================================
# SECCIÓN 8: FASE 2 — FEATURE SELECTION CON MEJORES HPs + GAIN ACUMULADO
# =============================================================================
# Con los HPs óptimos de Fase 1, se evalúan todos los subconjuntos definidos
# por umbral de Gain acumulado (Sección 7.5) más el baseline "All".
#
# Para cada subconjunto:
#   → CV-5 con early stopping (nrounds puede variar por subconjunto)
#   → Predicciones OOF → F1, Precision, Recall, AUC-ROC, threshold óptimo
#
# El resultado es una curva continua de "complejidad del modelo (N vars)
# vs. F1 OOF", donde el pico indica el umbral de Gain óptimo.
# =============================================================================
message("\n")
message(strrep("=", 70))
message("  FASE 2 — FEATURE SELECTION POR GAIN ACUMULADO (", length(feature_sets),
        " subconjuntos)")
message("  Método: Gain nativo de XGBoost (script 100% autónomo)")
message(strrep("=", 70))

feat_results_list <- list()

for (fs_name in names(feature_sets)) {
  fs_info <- feature_sets[[fs_name]]

  # Construir matrices para este subconjunto.
  # add_geo = FALSE: ciudad/dpto ya están incluidas si el ranking las posicionó
  # dentro del umbral. No se fuerza su presencia — aparecen por méritos propios.
  fs_data <- filtrar_cols_geo(
    X_tr    = X_train_full,
    X_te    = X_test_full,
    vars    = fs_info$vars,
    add_geo = FALSE
  )

  message(sprintf("\n  ── %s  (%d variables) ──",
                  fs_info$label, fs_data$n))

  cv_fs <- cv5_xgb_oof(
    params    = best_params,
    X_mat     = fs_data$tr,
    y_int     = y_int,
    y_factor  = y_factor,
    folds     = folds,
    max_nr    = MAX_NROUNDS,
    early_s   = EARLY_STOP,
    nthread   = N_THREADS,
    verbose_f = TRUE   # verbose en Fase 2 para monitorear progreso
  )

  met_fs <- calcular_metricas_oof(cv_fs$oof_probs, y_factor, y_int)

  message(sprintf(
    "  → nrounds_cv=%d  F1_OOF=%.4f  Prec=%.4f  Rec=%.4f  AUC=%.4f  th=%.2f",
    cv_fs$best_nrounds, met_fs$F1, met_fs$Precision,
    met_fs$Recall, met_fs$AUC_ROC, met_fs$threshold
  ))

  feat_results_list[[fs_name]] <- tibble(
    feature_set  = fs_name,
    label        = fs_info$label,
    n_features   = fs_data$n,
    nrounds_cv   = cv_fs$best_nrounds,
    F1_OOF       = round(met_fs$F1,        4),
    Precision    = round(met_fs$Precision,  4),
    Recall       = round(met_fs$Recall,     4),
    AUC_ROC      = round(met_fs$AUC_ROC,    4),
    Threshold    = met_fs$threshold
  )
}

feat_results <- bind_rows(feat_results_list) |> arrange(desc(F1_OOF))

# Guardar tabla de feature selection
write_csv(feat_results, file.path(dir_model, "feature_selection_results.csv"))
message("\n  Tabla de feature selection guardada: feature_selection_results.csv")
message("\n  Comparación de subconjuntos (Fase 2):")
print(feat_results |> select(feature_set, n_features, nrounds_cv,
                              F1_OOF, Precision, Recall, AUC_ROC, Threshold),
      n = Inf)

# =============================================================================
# SECCIÓN 9: SELECCIÓN DEL MEJOR MODELO GLOBAL
# =============================================================================
# El mejor modelo es el subconjunto de features (Fase 2) con mayor F1 OOF.
# Los HPs ya son los óptimos de Fase 1.
# =============================================================================
message("\n== SECCIÓN 9: Selección del mejor modelo global ==")

best_feat     <- feat_results |> slice_max(F1_OOF, n = 1, with_ties = FALSE)
BEST_FS_NAME  <- best_feat$feature_set
BEST_NROUNDS  <- best_feat$nrounds_cv
BEST_N_FEATS  <- best_feat$n_features
BEST_TH       <- best_feat$Threshold
BEST_F1       <- best_feat$F1_OOF
BEST_PREC     <- best_feat$Precision
BEST_REC      <- best_feat$Recall
BEST_AUC      <- best_feat$AUC_ROC

message("  Mejor subconjunto: ", BEST_FS_NAME,
        " (", BEST_N_FEATS, " variables)")
message("  F1 OOF del ganador: ", BEST_F1)
message("  nrounds del ganador: ", BEST_NROUNDS)
message("  Threshold óptimo:    ", BEST_TH)

# Reconstruir matrices del mejor subconjunto (para reentrenamiento y submission)
# add_geo = FALSE: las variables geográficas ya están en vars si corresponde
best_fs_data <- filtrar_cols_geo(
  X_tr    = X_train_full,
  X_te    = X_test_full,
  vars    = feature_sets[[BEST_FS_NAME]]$vars,
  add_geo = FALSE
)

# Re-entrenar modelo final sobre TODOS los datos de train con los mejores HPs
message("\n  Re-entrenando modelo final sobre train completo...")
set.seed(42)
dtrain_best  <- xgboost::xgb.DMatrix(best_fs_data$tr, label = y_int)
modelo_final <- xgboost::xgb.train(
  params  = c(best_params, list(nthread = N_THREADS)),
  data    = dtrain_best,
  nrounds = BEST_NROUNDS,   # nrounds determinados por CV, no por early stopping aquí
  verbose = 0
)
message("  Modelo final entrenado. nrounds = ", BEST_NROUNDS)

# =============================================================================
# SECCIÓN 10: SUBMISSION PARA KAGGLE
# =============================================================================
message("\n== SECCIÓN 10: Generando submission ==")

# Predecir sobre test con el threshold óptimo OOF
dtest_best  <- xgboost::xgb.DMatrix(best_fs_data$te)
probs_test  <- predict(modelo_final, dtest_best)
pred_test   <- as.integer(probs_test >= BEST_TH)

# Nombre de archivo informativo: incluye subconjunto, depth, eta, nrounds, threshold
sub_name <- sprintf(
  "XGB_003_%s_depth%d_eta%03.0f_nr%d_th%03.0f.csv",
  BEST_FS_NAME, BEST_MAX_DEPTH, BEST_ETA * 100, BEST_NROUNDS, BEST_TH * 100
)
sub_path <- file.path(dir_subs, sub_name)

write_csv(tibble(id = ids, pobre = pred_test), sub_path)

message("  Submission guardado: ", sub_path)
message("  Pobres predichos: ", sum(pred_test),
        " (", round(100 * mean(pred_test), 1), "%) de ", length(pred_test))

# =============================================================================
# SECCIÓN 11: GUARDAR DIAGNÓSTICOS COMPLETOS
# =============================================================================
message("\n== SECCIÓN 11: Guardando diagnostics.rds ==")

diagnostics <- list(
  model_id               = MODEL_ID,
  fecha                  = Sys.Date(),
  autor                  = AUTOR,
  # Resultados de Fase 1
  hp_search_results      = hp_results,
  best_hp_row            = best_hp,
  # Resultados de Fase 2
  feat_selection_results = feat_results,
  best_featureset        = BEST_FS_NAME,
  # Hiperparámetros finales
  best_params            = best_params,
  best_nrounds           = BEST_NROUNDS,
  # Métricas del mejor modelo
  best_threshold         = BEST_TH,
  f1_oof                 = BEST_F1,
  precision_oof          = BEST_PREC,
  recall_oof             = BEST_REC,
  auc_roc_oof            = BEST_AUC,
  n_features             = BEST_N_FEATS,
  feature_names          = best_fs_data$cols,
  # Modelo y submission
  modelo_final           = modelo_final,
  submission_path        = sub_path,
  # Metadatos de búsqueda
  n_search               = N_SEARCH,
  max_nrounds_search     = MAX_NROUNDS,
  early_stop             = EARLY_STOP
)

saveRDS(diagnostics, file.path(dir_model, "diagnostics.rds"))
message("  diagnostics.rds guardado en: ", dir_model)

# =============================================================================
# SECCIÓN 12: REGISTRO EN model_registry.csv
# =============================================================================
# Solo se registra el MEJOR modelo (HP × feature set ganador).
# Se evita registrar los N_SEARCH × 4 modelos intermedios para mantener
# el registry limpio y enfocado en modelos presentables.
# =============================================================================
message("\n== SECCIÓN 12: Actualizando model_registry.csv ==")

model_id_reg <- paste0(MODEL_ID, "_", BEST_FS_NAME)

new_entry <- tibble(
  model_id           = model_id_reg,
  fecha              = Sys.Date(),
  autor              = AUTOR,
  algoritmo          = "XGBoost",
  n_features         = BEST_N_FEATS,
  imbalance_strategy = "none",
  cv_folds           = 5L,
  cv_F1              = BEST_F1,
  cv_Precision       = BEST_PREC,
  cv_Recall          = BEST_REC,
  auc_roc            = BEST_AUC,
  kaggle_public_F1   = NA_real_,
  threshold          = BEST_TH,
  notas              = paste0(
    "XGBoost. Variables geog. incluidas (ciudad/dpto entero-codificadas). ",
    "Random search HP: N=", N_SEARCH, " combos, early_stop=", EARLY_STOP, ". ",
    "Mejor feature set: ", BEST_FS_NAME, " (", BEST_N_FEATS, " vars). ",
    "max_depth=", BEST_MAX_DEPTH,
    " eta=",              BEST_ETA,
    " gamma=",            BEST_GAMMA,
    " min_child_w=",      BEST_MIN_CHILD_WEIGHT,
    " colsample=",        BEST_COLSAMPLE,
    " subsample=",        BEST_SUBSAMPLE,
    " lambda(L2)=",       BEST_LAMBDA,
    " alpha(L1)=",        BEST_ALPHA,
    " nrounds=",          BEST_NROUNDS,
    ". OOF: th=",         BEST_TH,
    " F1=",               BEST_F1,
    " AUC=",              BEST_AUC, "."
  ),
  # Columnas estándar de HPs (compatibles con XGB_001 y XGB_002)
  nrounds          = BEST_NROUNDS,
  max_depth        = BEST_MAX_DEPTH,
  eta              = BEST_ETA,
  gamma            = BEST_GAMMA,
  min_child_weight = BEST_MIN_CHILD_WEIGHT,
  colsample_bytree = BEST_COLSAMPLE,
  subsample        = BEST_SUBSAMPLE,
  # Columnas nuevas: regularización (bind_rows rellenará con NA en modelos previos)
  lambda           = BEST_LAMBDA,
  alpha            = BEST_ALPHA
)

# Leer registry existente, eliminar entrada previa de este model_id (si existe)
# y agregar la nueva fila
if (file.exists(reg_path)) {
  registry <- read_csv(reg_path, show_col_types = FALSE) |>
    mutate(fecha = as.Date(fecha, origin = "1899-12-30")) |>
    filter(model_id != model_id_reg)
  registry <- bind_rows(registry, new_entry)
} else {
  registry <- new_entry
}

write_csv(registry, reg_path)
message("  Modelo registrado: ", model_id_reg)

# =============================================================================
# SECCIÓN 13: REPORTE FINAL EN CONSOLA
# =============================================================================
# Imprime:
#   (a) Resumen completo del mejor modelo XGB_003: HPs, features, métricas
#   (b) Top-5 combinaciones de HPs exploradas en Fase 1
#   (c) Comparación de subconjuntos de Fase 2
#   (d) Tabla comparativa con TODOS los modelos del model_registry
#   (e) Posición de XGB_003 en el ranking global
# =============================================================================

cat("\n\n")
cat(strrep("=", 90), "\n")
cat("  REPORTE FINAL — ", MODEL_ID, "\n", sep = "")
cat("  XGBoost: random search de HPs + ciudad/dpto + feature selection por Gain acumulado\n")
cat(strrep("=", 90), "\n")

# ── Mejor modelo ──────────────────────────────────────────────────────────────
cat("\n  ★ MEJOR MODELO SELECCIONADO ★\n")
cat(strrep("-", 70), "\n")
cat(sprintf("  Modelo ID      : %s\n", model_id_reg))
cat(sprintf("  Feature set    : %s  (%s)\n",
            BEST_FS_NAME, feature_sets[[BEST_FS_NAME]]$label))
cat(sprintf("  N° variables   : %d\n", BEST_N_FEATS))
cat(sprintf("  Búsqueda HP    : %d combinaciones aleatorias (random search)\n",
            N_SEARCH))

cat(strrep("-", 70), "\n")
cat("  HIPERPARÁMETROS OPTIMIZADOS (Fase 1 — CV-5 OOF, max F1):\n")
cat(sprintf("    max_depth        = %d\n",   BEST_MAX_DEPTH))
cat(sprintf("    eta              = %.4f\n", BEST_ETA))
cat(sprintf("    gamma            = %.4f\n", BEST_GAMMA))
cat(sprintf("    min_child_weight = %d\n",   BEST_MIN_CHILD_WEIGHT))
cat(sprintf("    colsample_bytree = %.4f\n", BEST_COLSAMPLE))
cat(sprintf("    subsample        = %.4f\n", BEST_SUBSAMPLE))
cat(sprintf("    lambda (L2 reg.) = %.4f\n", BEST_LAMBDA))
cat(sprintf("    alpha  (L1 reg.) = %.4f\n", BEST_ALPHA))
cat(sprintf("    nrounds          = %d  (mediana early stopping CV-5)\n",
            BEST_NROUNDS))

cat(strrep("-", 70), "\n")
cat("  MÉTRICAS OOF — CV-5, threshold óptimo (max F1):\n")
cat(sprintf("    F1        = %.4f\n", BEST_F1))
cat(sprintf("    Precision = %.4f\n", BEST_PREC))
cat(sprintf("    Recall    = %.4f\n", BEST_REC))
cat(sprintf("    AUC-ROC   = %.4f\n", BEST_AUC))
cat(sprintf("    Threshold = %.2f\n", BEST_TH))
cat(strrep("-", 70), "\n")
cat(sprintf("  Submission     : %s\n", sub_name))
cat(strrep("=", 90), "\n")

# ── Top-5 combinaciones HP exploradas ─────────────────────────────────────────
cat("\n  TOP-5 COMBINACIONES DE HIPERPARÁMETROS (de ", N_SEARCH, " exploradas):\n",
    sep = "")
cat(strrep("-", 90), "\n")
print(
  hp_results |>
    select(max_depth, eta, gamma, min_child_weight, colsample_bytree,
           subsample, lambda, alpha, best_nrounds, F1_OOF, AUC_ROC) |>
    head(5) |>
    mutate(across(where(is.double), ~ round(., 4))),
  n = 5
)
cat(strrep("-", 90), "\n")

# ── Comparación de feature sets (Fase 2) ──────────────────────────────────────
cat("\n  COMPARACIÓN DE SUBCONJUNTOS DE VARIABLES (Fase 2):\n")
cat(strrep("-", 90), "\n")
print(
  feat_results |>
    select(label, n_features, nrounds_cv, F1_OOF, Precision, Recall,
           AUC_ROC, Threshold) |>
    rename(Modelo = label, Vars = n_features, Nrounds = nrounds_cv),
  n = Inf
)
cat(strrep("-", 90), "\n")

# ── Comparación con model_registry ────────────────────────────────────────────
cat("\n  COMPARACIÓN GLOBAL — MODEL REGISTRY (ordenado por F1 OOF)\n")
cat(strrep("-", 90), "\n")

registry_view <- read_csv(reg_path, show_col_types = FALSE) |>
  mutate(fecha = as.Date(fecha, origin = "1899-12-30")) |>
  select(model_id, algoritmo, n_features, cv_F1, cv_Precision, cv_Recall,
         auc_roc, kaggle_public_F1, threshold) |>
  mutate(
    across(c(cv_F1, cv_Precision, cv_Recall, auc_roc), ~ round(., 4)),
    threshold = round(threshold, 2)
  ) |>
  arrange(desc(cv_F1))

print(registry_view, n = Inf)
cat(strrep("-", 90), "\n")

# Mejor modelo del registry por F1 OOF
best_reg <- registry_view |>
  filter(!is.na(cv_F1)) |>
  slice_max(cv_F1, n = 1, with_ties = FALSE)

cat(sprintf(
  "\n  ★ Mejor modelo en registry: %-25s  F1=%.4f  Prec=%.4f  Rec=%.4f  AUC=%.4f\n",
  best_reg$model_id, best_reg$cv_F1,
  best_reg$cv_Precision, best_reg$cv_Recall, best_reg$auc_roc
))

# Posición de XGB_003 en el ranking
pos_xgb003 <- which(registry_view$model_id == model_id_reg)
if (length(pos_xgb003) > 0 && !is.na(registry_view$cv_F1[pos_xgb003])) {
  n_con_f1 <- sum(!is.na(registry_view$cv_F1))
  cat(sprintf(
    "  XGB_003 ocupa la posición #%d de %d modelos con F1 reportado.\n\n",
    pos_xgb003, n_con_f1
  ))
}

cat(strrep("=", 90), "\n")
cat("  Script finalizado.\n")
cat(strrep("=", 90), "\n\n")

# nolint end
