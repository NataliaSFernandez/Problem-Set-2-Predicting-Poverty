#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/XGBoost/XGB_007.R
#==============================================================================
#
# ALGORITMO: XGBoost — Sin variables construidas + sin Bogotá en entrenamiento
#
# MOTIVACIÓN — POR QUÉ XGB_007 EXISTE:
#   XGB_006 excluye las 16 variables construidas del benchmark XGB_005.
#   XGB_007 agrega una segunda modificación: eliminar del training set todas
#   las observaciones de Bogotá D.C. (dpto == "11").
#
#   Razón para excluir Bogotá del entrenamiento:
#   ─────────────────────────────────────────────
#   El test set NO contiene hogares de Bogotá (los datos de la capital
#   no están disponibles para predicción). Entrenar con observaciones de un
#   dominio que no aparece en test puede introducir sesgo de distribución:
#   el modelo aprende patrones específicos de Bogotá (mayor formalidad,
#   menor pobreza, economía diferenciada) que no son generalizables al
#   conjunto de ciudades que SÍ hay que predecir.
#
#   Hipótesis: eliminar Bogotá produce un modelo entrenado en una distribución
#   más similar a la de test, reduciendo el sesgo de dominio y potencialmente
#   mejorando F1 en Kaggle (no necesariamente el F1 OOF, que se computa
#   sobre el training filtrado).
#
#   Hipótesis alternativa: Bogotá aporta varianza útil (patrones de hogares
#   no pobres bien definidos) que ayuda a calibrar el clasificador.
#   Eliminarla podría empeorar la estimación de la clase "No pobre".
#
#   Consecuencias operativas de filtrar Bogotá:
#     (a) El training set se reduce en ~N_bogota observaciones.
#     (b) La tasa de pobreza en train puede cambiar (Bogotá tiene pobreza
#         relativamente baja → su exclusión puede subir la prevalencia en train).
#     (c) La variable `bogota` queda constante 0 en train → se excluye de
#         los predictores para evitar features de varianza nula.
#     (d) Los folds CV se recomputan sobre el training filtrado.
#
#   NOTA sobre `bogota` en test: aunque no hay observaciones de Bogotá
#   en test, la variable `bogota` podría existir con valor 0. Al excluirla
#   de los predictores también en test se garantiza consistencia total.
#
# VARIABLES EXCLUIDAS RESPECTO A XGB_005 (17 en total):
#   Las mismas 16 que en XGB_006, más:
#    17. bogota  — constante 0 tras filtrar Bogotá de train (varianza nula)
#
#   CUADRÁTICAS DE CAPITAL HUMANO — Bloque 3 de 01_data_clean_and_merge.R:
#     1.  educ_jefe_sq       nivel_educ_jefe ^ 2
#     2.  educ_prom_sq       nivel_educ_prom ^ 2
#
#   LOGARÍTMICAS — Bloque 9 de 01_data_clean_and_merge.R:
#     3.  log_num_personas   log1p(num_personas)
#     4.  log_n_menores      log1p(n_menores_18)
#
#   CUADRÁTICAS GENERALES — Bloque 9 de 01_data_clean_and_merge.R:
#     5.  ratio_depend_sq    ratio_depend ^ 2
#     6.  tasa_ocup_sq       tasa_ocupacion ^ 2
#     7.  hacinamiento_sq    (num_personas / pmax(num_cuartos, 1)) ^ 2
#     8.  edad_jefe_sq       edad_jefe ^ 2
#
#   INTERACCIONES — Bloque 8 de 01_data_clean_and_merge.R:
#     9.  educ_x_formal      nivel_educ_jefe × prop_cotiza_pension
#    10.  educ_x_ocup        nivel_educ_jefe × tasa_ocupacion
#    11.  menores_x_desocup  n_menores_18    × (1 − tasa_ocupacion)
#    12.  depend_x_informal  ratio_depend    × prop_informal
#    13.  rural_x_cta_propia zona == 2       × prop_cta_propia
#    14.  educ_x_rural       nivel_educ_jefe × zona == 2
#    15.  subsidiado_x_menores prop_subsidiado × n_menores_18
#    16.  jefe_mayor_pension  edad_jefe > 55  × alguno_pension_jub
#
#   DUMMY CONSTANTE (varianza nula tras el filtro):
#    17. bogota              as.integer(dpto == "11") → 0 en todo el train
#
# DIFERENCIAS RESPECTO A XGB_005:
# ┌─────────────────────────────┬─────────────────────┬─────────────────────┐
# │ Aspecto                     │ XGB_005             │ XGB_007             │
# ├─────────────────────────────┼─────────────────────┼─────────────────────┤
# │ Variables construidas       │ Incluidas (16)      │ EXCLUIDAS (16)      │
# │ Variable `bogota`           │ Incluida            │ EXCLUIDA (const. 0) │
# │ N° predictores              │ 110                 │ 110 − 17 = 93       │
# │ Observaciones de Bogotá     │ Incluidas           │ EXCLUIDAS           │
# │ Grid HP (Fases 1 y 2)       │ Sin cambio          │ Sin cambio          │
# │ Técnicas de balanceo        │ Sin cambio          │ Sin cambio          │
# │ N_SEARCH, K_FOLDS           │ 30 / 5              │ 30 / 5              │
# │ MAX_NROUNDS / EARLY_STOP    │ 500 / 50            │ 500 / 50            │
# └─────────────────────────────┴─────────────────────┴─────────────────────┘
#
# DIFERENCIAS RESPECTO A XGB_006:
# ┌─────────────────────────────┬─────────────────────┬─────────────────────┐
# │ Aspecto                     │ XGB_006             │ XGB_007             │
# ├─────────────────────────────┼─────────────────────┼─────────────────────┤
# │ Observaciones de Bogotá     │ Incluidas           │ EXCLUIDAS           │
# │ Variable `bogota`           │ Incluida            │ EXCLUIDA            │
# │ N° predictores              │ ~94                 │ ~93                 │
# │ Todo lo demás               │ —                   │ Sin cambio          │
# └─────────────────────────────┴─────────────────────┴─────────────────────┘
#
# ESTRUCTURA (idéntica a XGB_005/XGB_006):
#
#   FASE 1 — Búsqueda aleatoria de HP
#     N_SEARCH combinaciones aleatorias × CV-5 + early stopping.
#     CV aplicado sobre el training SIN Bogotá.
#     Checkpoint: si hp_search_results.csv ya existe, se salta la fase.
#
#   FASE 2 — Comparación de técnicas de balanceo de clases
#     Con HP óptimos de Fase 1, se comparan 5 técnicas via CV-5 OOF.
#
#   FASE 3 — Modelo final
#     Entrenado sobre TODO el training filtrado (sin Bogotá).
#     Predice sobre test (que tampoco tiene Bogotá → distribución alineada).
#
# REFERENCIA XGB_005 (benchmark):
#   Ver 02_outputs/model_registry.csv para los valores actualizados.
#   Comparar XGB_007 vs XGB_006 aísla el efecto de excluir Bogotá.
#   Comparar XGB_007 vs XGB_005 mide el efecto combinado.
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#
# OUTPUTS:
#   02_outputs/models/XGBoost/XGB_007/hp_search_results.csv
#   02_outputs/models/XGBoost/XGB_007/cv_balance_summary.csv
#   02_outputs/models/XGBoost/XGB_007/balance_comparison.png
#   02_outputs/models/XGBoost/XGB_007/varimp.png
#   02_outputs/models/XGBoost/XGB_007/threshold.png
#   02_outputs/models/XGBoost/XGB_007/roc.png
#   02_outputs/models/XGBoost/XGB_007/prcurve.png
#   02_outputs/models/XGBoost/XGB_007/diagnostics.rds
#   03_submissions/XGB_007_*.csv
#   02_outputs/model_registry.csv  (append)
#
# TIEMPO ESTIMADO: ligeramente menor que XGB_006 (menos obs. en train)
#   Fase 1: ~20–35 min con 9 cores
#   Fase 2: ~6–10 min
#   Fase 3: <1 min
#
# REPRODUCIBILIDAD: semilla global 42. Correr desde raíz del proyecto.
#==============================================================================
# nolint start

# =============================================================================
# SECCIÓN 0: PAQUETES
# =============================================================================
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,   # dplyr, ggplot2, purrr, tibble, readr
  xgboost,     # xgb.DMatrix(), xgb.train(), xgb.importance()
  caret,       # createFolds(), downSample(), upSample()
  themis,      # smote()
  pROC,        # roc(), auc()
  fs           # dir_create()
)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
AUTOR    <- "Jonathan"
MODEL_ID <- "XGB_007"

set.seed(42)

N_THREADS <- max(1L, parallel::detectCores() - 1L)

N_SEARCH    <- 30L
MAX_NROUNDS <- 500L
EARLY_STOP  <- 50L
K_FOLDS     <- 5L
TECHNIQUES  <- c("baseline", "class_weights", "downsample", "upsample", "smote")
niveles     <- c("No", "Yes")

# -----------------------------------------------------------------------------
# VARIABLES CONSTRUIDAS QUE SE EXCLUYEN (16, igual que XGB_006)
# + variable `bogota` (constante 0 tras filtrar el training set)
# -----------------------------------------------------------------------------
# La variable `bogota` es as.integer(dpto == "11"). Al eliminar todos los
# hogares de Bogotá del training set, esta variable queda constante en 0
# (varianza nula). Una feature constante no aporta información predictiva
# y puede causar inestabilidad numérica en algunos algoritmos.
# Se excluye de ambos conjuntos (train y test) para consistencia.
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
  "jefe_mayor_pension",
  # Dummy constante tras filtrar Bogotá (varianza nula en train)
  "bogota"
)

# DPTO de Bogotá D.C. en la encuesta DANE
# En 01_data_clean_and_merge.R: dpto = Depto (factor), Bogotá = "11"
DPTO_BOGOTA <- "11"

dir_model <- file.path("02_outputs/models/XGBoost", MODEL_ID)
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model, recurse = TRUE)
fs::dir_create(dir_subs,  recurse = TRUE)

message(strrep("=", 65))
message("  ", MODEL_ID, "  —  ", AUTOR)
message("  Hipótesis 1: variables construidas son redundantes para XGBoost")
message("  Hipótesis 2: excluir Bogotá alinea train con la distribución de test")
message("  Variables excluidas: ", length(VARS_EXCLUIR),
        " (cuadráticas + logs + interacciones + bogota)")
message("  Bogotá en train     : EXCLUIDA (dpto == '", DPTO_BOGOTA, "')")
message("  Cores XGBoost (nthread) : ", N_THREADS)
message("  Búsqueda HP (N_SEARCH)  : ", N_SEARCH)
message("  Max rondas / Early stop : ", MAX_NROUNDS, " / ", EARLY_STOP)
message("  Técnicas de balanceo    : ", paste(TECHNIQUES, collapse = ", "))
message(strrep("=", 65))

# =============================================================================
# SECCIÓN 2: CARGAR DATOS Y FILTRAR BOGOTÁ DEL TRAINING SET
# =============================================================================
message("\n== Cargando datos ==")

train_raw <- readRDS("00_data/processed/train_final.rds")
test      <- readRDS("00_data/processed/test_final.rds")
ids       <- test$id

message("  Train (original, antes de filtrar): ",
        nrow(train_raw), " filas x ", ncol(train_raw), " columnas")
message("  Test                              : ",
        nrow(test), " filas x ", ncol(test), " columnas")

# ─────────────────────────────────────────────────────────────────────────────
# FILTRO: Eliminar observaciones de Bogotá D.C. del training set
# ─────────────────────────────────────────────────────────────────────────────
# La variable `dpto` en train_final.rds es un factor con niveles como
# "05", "08", "11", etc. (código DANE de departamento).
# Bogotá D.C. = "11". Se filtra con as.character() para comparación segura
# independientemente de los niveles del factor.
n_bogota <- sum(as.character(train_raw$dpto) == DPTO_BOGOTA, na.rm = TRUE)
message("\n  Hogares de Bogotá (dpto==", DPTO_BOGOTA, ") en train: ", n_bogota)

train <- train_raw |> filter(as.character(dpto) != DPTO_BOGOTA)

message("  Train filtrado (sin Bogotá): ", nrow(train), " filas",
        "  (reducción: ", n_bogota, " obs, ",
        round(100 * n_bogota / nrow(train_raw), 1), "%)")

y_factor <- factor(ifelse(train$pobre == 1, "Yes", "No"), levels = niveles)
y_int    <- as.integer(y_factor == "Yes")
n_pos    <- sum(y_int)
n_neg    <- length(y_int) - n_pos
message("  Balance (train filtrado): ", n_pos, " pobres (",
        round(100 * mean(y_int), 1), "%) | ",
        n_neg, " no-pobres (", round(100 * (1 - mean(y_int)), 1), "%)")

# Mostrar tasa de pobreza original vs filtrada para cuantificar el cambio
# en la distribución tras eliminar Bogotá (ciudad con baja pobreza relativa)
pob_original  <- mean(ifelse(train_raw$pobre == 1, 1, 0), na.rm = TRUE)
pob_filtrada  <- mean(y_int)
message("  Prevalencia pobreza — original: ",
        round(100 * pob_original, 2), "% | filtrada: ",
        round(100 * pob_filtrada, 2), "%",
        "  (delta: ", round(100 * (pob_filtrada - pob_original), 2), " pp)")

# =============================================================================
# SECCIÓN 3: PREPROCESAMIENTO
# =============================================================================
# Igual que XGB_006: ciudad y dpto como entero ordinal, se eliminan las 17
# variables de VARS_EXCLUIR (16 construidas + bogota).
message("\n== Preprocesando variables ==")
message("  Codificación: ciudad y dpto como entero ordinal")
message("  Exclusión: ", length(VARS_EXCLUIR), " variables (16 construidas + bogota)")

preparar_X_xgb <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>
    mutate(zona   = as.numeric(zona),
           ciudad = as.integer(factor(ciudad)),
           dpto   = as.integer(factor(dpto))) |>
    mutate(across(where(is.character), as.numeric)) |>
    mutate(across(where(is.factor),    as.integer))
}

X_train_full <- preparar_X_xgb(train)
X_test_full  <- preparar_X_xgb(test)

# Eliminar las 17 variables de ambos conjuntos.
# `bogota` se elimina de test aunque tenga valores 0 (no varianza nula en test
# si hubiera Bogotá, pero la eliminamos para mantener consistencia total entre
# train y test, y porque el modelo no fue entrenado con esta feature).
vars_encontradas_train <- intersect(VARS_EXCLUIR, names(X_train_full))
vars_encontradas_test  <- intersect(VARS_EXCLUIR, names(X_test_full))
message("  Variables eliminadas de train: ",
        length(vars_encontradas_train), " de ", length(VARS_EXCLUIR), " declaradas")
message("  Variables eliminadas de test : ",
        length(vars_encontradas_test),  " de ", length(VARS_EXCLUIR), " declaradas")

X_train_full <- X_train_full |> select(-any_of(VARS_EXCLUIR))
X_test_full  <- X_test_full  |> select(-any_of(VARS_EXCLUIR))

cols_comunes <- intersect(names(X_train_full), names(X_test_full))
X_train_full <- X_train_full |> select(all_of(cols_comunes))
X_test_full  <- X_test_full  |> select(all_of(cols_comunes))

P_TOTAL <- ncol(X_train_full)
message("  Predictores finales: ", P_TOTAL,
        "  (XGB_005: 110 | XGB_006: ~94 | XGB_007: ", P_TOTAL, ")")
message("  NAs en X_train_full: ", sum(is.na(X_train_full)))

mat_train_full <- as.matrix(X_train_full)
mat_test_full  <- as.matrix(X_test_full)

# =============================================================================
# SECCIÓN 4: FUNCIONES AUXILIARES
# =============================================================================
# Idénticas a XGB_005/XGB_006.

calcular_metricas_oof <- function(probs_oof, obs_factor, obs_int) {
  best <- map_dfr(seq(0.05, 0.95, by = 0.01), function(th) {
    pred <- factor(ifelse(probs_oof >= th, "Yes", "No"), levels = niveles)
    TP   <- sum(pred == "Yes" & obs_factor == "Yes")
    FP   <- sum(pred == "Yes" & obs_factor == "No")
    FN   <- sum(pred == "No"  & obs_factor == "Yes")
    f1   <- if ((2 * TP + FP + FN) == 0) NA_real_ else 2 * TP / (2 * TP + FP + FN)
    prec <- if ((TP + FP) == 0)           NA_real_ else TP / (TP + FP)
    rec  <- if ((TP + FN) == 0)           NA_real_ else TP / (TP + FN)
    tibble(threshold = th, F1 = f1, Precision = prec, Recall = rec)
  }) |>
    filter(!is.na(F1)) |>
    slice_max(F1, n = 1, with_ties = FALSE)

  roc_obj <- pROC::roc(response = obs_int, predictor = probs_oof, quiet = TRUE)
  list(threshold = best$threshold, F1 = best$F1,
       Precision = best$Precision, Recall = best$Recall,
       AUC_ROC   = as.numeric(pROC::auc(roc_obj)),
       roc_obj   = roc_obj)
}

entrenar_fold_xgb <- function(X_tr_mat, y_int_tr,
                               X_val_mat, y_int_val,
                               params, max_nr, early_s, nthread,
                               tech = "baseline", seed_val = 42) {
  X_use <- X_tr_mat
  y_use <- y_int_tr
  p_use <- params

  if (tech == "class_weights") {
    spw   <- sum(y_int_tr == 0) / sum(y_int_tr == 1)
    p_use <- c(params, list(scale_pos_weight = spw))

  } else if (tech == "downsample") {
    set.seed(seed_val)
    y_f   <- factor(ifelse(y_int_tr == 1, "Yes", "No"), levels = niveles)
    df    <- as_tibble(X_tr_mat)
    ds    <- caret::downSample(x = df, y = y_f, yname = "pobre")
    X_use <- as.matrix(ds |> select(-pobre))
    y_use <- as.integer(ds$pobre == "Yes")

  } else if (tech == "upsample") {
    set.seed(seed_val)
    y_f   <- factor(ifelse(y_int_tr == 1, "Yes", "No"), levels = niveles)
    df    <- as_tibble(X_tr_mat)
    us    <- caret::upSample(x = df, y = y_f, yname = "pobre")
    X_use <- as.matrix(us |> select(-pobre))
    y_use <- as.integer(us$pobre == "Yes")

  } else if (tech == "smote") {
    set.seed(seed_val)
    y_f    <- factor(ifelse(y_int_tr == 1, "Yes", "No"), levels = niveles)
    df_in  <- bind_cols(as_tibble(X_tr_mat), pobre = y_f)
    df_out <- themis::smote(df_in, var = "pobre", k = 5, over_ratio = 1)
    X_use  <- as.matrix(df_out |> select(-pobre))
    y_use  <- as.integer(df_out$pobre == "Yes")
  }

  p_full <- c(p_use, list(nthread = nthread))
  dtrain <- xgboost::xgb.DMatrix(X_use,     label = y_use)
  dval   <- xgboost::xgb.DMatrix(X_val_mat, label = y_int_val)

  mod <- xgboost::xgb.train(
    params                = p_full,
    data                  = dtrain,
    nrounds               = max_nr,
    watchlist             = list(val = dval),
    early_stopping_rounds = early_s,
    verbose               = 0
  )

  best_iter_raw <- xgboost::xgb.attr(mod, "best_iteration")
  best_nr <- if (!is.null(best_iter_raw)) as.integer(best_iter_raw) else max_nr

  list(probs_val = predict(mod, dval), best_nrounds = best_nr)
}

# =============================================================================
# SECCIÓN 5: FASE 1 — BÚSQUEDA ALEATORIA DE HIPERPARÁMETROS
# =============================================================================
# Los folds se crean sobre el training SIN Bogotá. Esto significa que el
# CV estima el desempeño del modelo en el dominio de ciudades que también
# están en el test set, sin contaminación de patrones de Bogotá.
set.seed(42)
folds <- caret::createFolds(y_factor, k = K_FOLDS, list = TRUE, returnTrain = TRUE)

hp_checkpoint <- file.path(dir_model, "hp_search_results.csv")

if (file.exists(hp_checkpoint)) {
  message("\n", strrep("=", 65))
  message("  FASE 1: cargando resultados guardados (checkpoint)")
  message(strrep("=", 65))
  hp_results <- read_csv(hp_checkpoint, show_col_types = FALSE)

} else {
  message("\n", strrep("=", 65))
  message("  FASE 1 — BÚSQUEDA ALEATORIA DE HIPERPARÁMETROS")
  message("  N_SEARCH=", N_SEARCH, " combos | ", P_TOTAL,
          " variables | train sin Bogotá (", nrow(train), " obs)")
  message("  Grid HP: idéntico a XGB_005/XGB_006 (anti-sobreajuste)")
  message(strrep("=", 65))

  # Grid HP idéntico al de XGB_005 y XGB_006 — sin cambios.
  set.seed(42)
  hp_grid <- tibble(
    max_depth         = sample(c(3L, 4L, 5L, 6L),
                               N_SEARCH, replace = TRUE),
    eta               = sample(c(0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20),
                               N_SEARCH, replace = TRUE),
    gamma             = sample(c(0.5, 1.0, 2.0, 3.0, 5.0),
                               N_SEARCH, replace = TRUE),
    min_child_weight  = sample(c(1L, 3L, 5L, 8L, 10L, 15L, 20L),
                               N_SEARCH, replace = TRUE),
    colsample_bytree  = sample(c(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0),
                               N_SEARCH, replace = TRUE),
    colsample_bylevel = sample(c(0.5, 0.7, 1.0),
                               N_SEARCH, replace = TRUE),
    subsample         = sample(c(0.6, 0.7, 0.8, 0.9),
                               N_SEARCH, replace = TRUE),
    lambda            = sample(c(1.0, 2.0, 5.0, 10.0),
                               N_SEARCH, replace = TRUE),
    alpha             = sample(c(0.0, 0.05, 0.1, 0.5, 1.0, 2.0),
                               N_SEARCH, replace = TRUE)
  )

  hp_results_list <- vector("list", N_SEARCH)
  t0_hp <- Sys.time()

  for (i in seq_len(N_SEARCH)) {
    hp_i <- hp_grid[i, ]
    message(sprintf(
      "\n  [HP %2d/%d] depth=%d eta=%.3f gam=%.1f mcw=%2d col_t=%.2f col_l=%.2f sub=%.2f L2=%.1f L1=%.3f",
      i, N_SEARCH,
      hp_i$max_depth, hp_i$eta, hp_i$gamma,
      hp_i$min_child_weight, hp_i$colsample_bytree, hp_i$colsample_bylevel,
      hp_i$subsample, hp_i$lambda, hp_i$alpha
    ))

    params_i <- list(
      objective         = "binary:logistic",
      eval_metric       = "logloss",
      max_depth         = hp_i$max_depth,
      eta               = hp_i$eta,
      gamma             = hp_i$gamma,
      min_child_weight  = hp_i$min_child_weight,
      colsample_bytree  = hp_i$colsample_bytree,
      colsample_bylevel = hp_i$colsample_bylevel,
      subsample         = hp_i$subsample,
      lambda            = hp_i$lambda,
      alpha             = hp_i$alpha,
      nthread           = N_THREADS
    )

    n_train    <- nrow(mat_train_full)
    oof_i      <- numeric(n_train)
    nr_folds_i <- integer(K_FOLDS)

    ok <- tryCatch({
      for (f in seq_along(folds)) {
        tr_idx  <- folds[[f]]
        val_idx <- setdiff(seq_len(n_train), tr_idx)
        dtrain_f <- xgboost::xgb.DMatrix(mat_train_full[tr_idx,  ],
                                          label = y_int[tr_idx])
        dval_f   <- xgboost::xgb.DMatrix(mat_train_full[val_idx, ],
                                          label = y_int[val_idx])
        m_f <- xgboost::xgb.train(
          params                = params_i,
          data                  = dtrain_f,
          nrounds               = MAX_NROUNDS,
          watchlist             = list(val = dval_f),
          early_stopping_rounds = EARLY_STOP,
          verbose               = 0
        )
        bir            <- xgboost::xgb.attr(m_f, "best_iteration")
        nr_folds_i[f]  <- if (!is.null(bir)) as.integer(bir) else MAX_NROUNDS
        oof_i[val_idx] <- predict(m_f, dval_f)
      }
      TRUE
    }, error = function(e) { message("    ERROR: ", conditionMessage(e)); FALSE })

    if (!ok) next

    met_i          <- calcular_metricas_oof(oof_i, y_factor, y_int)
    best_nrounds_i <- as.integer(ceiling(median(nr_folds_i)))

    message(sprintf("    → nrounds_med=%d (sd=%.1f)  F1_OOF=%.4f  AUC=%.4f  th=%.2f",
                    best_nrounds_i, sd(nr_folds_i),
                    met_i$F1, met_i$AUC_ROC, met_i$threshold))

    hp_results_list[[i]] <- bind_cols(
      hp_i,
      tibble(best_nrounds = best_nrounds_i,
             nrounds_sd   = round(sd(nr_folds_i), 1),
             F1_OOF       = round(met_i$F1,        4),
             Precision    = round(met_i$Precision,  4),
             Recall       = round(met_i$Recall,     4),
             AUC_ROC      = round(met_i$AUC_ROC,    4),
             Threshold    = met_i$threshold)
    )
  }

  t1_hp <- Sys.time()
  message(sprintf("\n  FASE 1 completada en %.1f minutos.",
                  as.numeric(difftime(t1_hp, t0_hp, units = "mins"))))

  hp_results <- bind_rows(hp_results_list) |> arrange(desc(F1_OOF))
  write_csv(hp_results, hp_checkpoint)
  message("  Guardado: hp_search_results.csv")
}

cat("\n-- Top-5 combos HP (F1 OOF) --\n")
print(hp_results |>
        select(max_depth:alpha, best_nrounds, nrounds_sd, F1_OOF, AUC_ROC, Threshold) |>
        head(5))

n_no_converge <- sum(hp_results$nrounds_sd == 0 & hp_results$best_nrounds >= MAX_NROUNDS - 5,
                     na.rm = TRUE)
message(sprintf("\n  Combos sin convergencia (nrounds_sd=0 y nrounds>=%d): %d/%d",
                MAX_NROUNDS - 5, n_no_converge, nrow(hp_results)))

best_hp <- hp_results |> slice_max(F1_OOF, n = 1, with_ties = FALSE)

best_params <- list(
  objective         = "binary:logistic",
  eval_metric       = "logloss",
  max_depth         = best_hp$max_depth,
  eta               = best_hp$eta,
  gamma             = best_hp$gamma,
  min_child_weight  = best_hp$min_child_weight,
  colsample_bytree  = best_hp$colsample_bytree,
  colsample_bylevel = best_hp$colsample_bylevel,
  subsample         = best_hp$subsample,
  lambda            = best_hp$lambda,
  alpha             = best_hp$alpha
)

message("\n  Mejor HP: depth=", best_hp$max_depth,
        " eta=", best_hp$eta,
        " gam=", best_hp$gamma,
        " mcw=", best_hp$min_child_weight,
        " col_t=", best_hp$colsample_bytree,
        " col_l=", best_hp$colsample_bylevel,
        " sub=", best_hp$subsample,
        " L2=", best_hp$lambda,
        " L1=", best_hp$alpha,
        " nrounds=", best_hp$best_nrounds,
        " | F1_OOF=", best_hp$F1_OOF)

# =============================================================================
# SECCIÓN 6: FASE 2 — COMPARACIÓN DE TÉCNICAS DE BALANCEO
# =============================================================================
message("\n", strrep("=", 65))
message("  FASE 2 — TÉCNICAS DE BALANCEO DE CLASES")
message("  HP fijo: depth=", best_hp$max_depth,
        " eta=", best_hp$eta,
        " gam=", best_hp$gamma,
        " nrounds<=", MAX_NROUNDS, " (early stopping=", EARLY_STOP, ")")
message("  Técnicas: ", paste(TECHNIQUES, collapse = ", "))
message("  NOTA: balance calculado sobre train sin Bogotá (", nrow(train), " obs)")
message(strrep("=", 65))

n_train <- nrow(mat_train_full)

bal_oof_list  <- vector("list", length(TECHNIQUES))
names(bal_oof_list) <- TECHNIQUES
bal_nr_list   <- vector("list", length(TECHNIQUES))
names(bal_nr_list)  <- TECHNIQUES
bal_fold_tbl  <- vector("list", length(TECHNIQUES))

t0_bal <- Sys.time()

for (tech in TECHNIQUES) {
  message("\n  -- Técnica: ", tech, " --")

  oof_tech <- numeric(n_train)
  nr_tech  <- integer(K_FOLDS)

  for (f in seq_along(folds)) {
    tr_idx  <- folds[[f]]
    val_idx <- setdiff(seq_len(n_train), tr_idx)
    seed_ft <- 42 * f + match(tech, TECHNIQUES) * 1000

    res_f <- tryCatch(
      entrenar_fold_xgb(
        X_tr_mat  = mat_train_full[tr_idx,  ],
        y_int_tr  = y_int[tr_idx],
        X_val_mat = mat_train_full[val_idx, ],
        y_int_val = y_int[val_idx],
        params    = best_params,
        max_nr    = MAX_NROUNDS,
        early_s   = EARLY_STOP,
        nthread   = N_THREADS,
        tech      = tech,
        seed_val  = seed_ft
      ),
      error = function(e) {
        message("    ERROR fold ", f, ": ", conditionMessage(e))
        NULL
      }
    )

    if (is.null(res_f)) next
    oof_tech[val_idx] <- res_f$probs_val
    nr_tech[f]        <- res_f$best_nrounds

    message(sprintf("    Fold %d: nrounds=%3d", f, res_f$best_nrounds))
  }

  met_tech <- calcular_metricas_oof(oof_tech, y_factor, y_int)
  nr_med   <- as.integer(ceiling(median(nr_tech)))

  message(sprintf(
    "    OOF → F1=%.4f  Precision=%.4f  Recall=%.4f  AUC=%.4f  th=%.2f  nrounds_med=%d",
    met_tech$F1, met_tech$Precision, met_tech$Recall,
    met_tech$AUC_ROC, met_tech$threshold, nr_med
  ))

  bal_oof_list[[tech]] <- oof_tech
  bal_nr_list[[tech]]  <- nr_med

  bal_fold_tbl[[tech]] <- tibble(
    technique    = tech,
    F1           = round(met_tech$F1,        4),
    Precision    = round(met_tech$Precision,  4),
    Recall       = round(met_tech$Recall,     4),
    AUC_ROC      = round(met_tech$AUC_ROC,    4),
    threshold    = met_tech$threshold,
    best_nrounds = nr_med
  )
}

t1_bal <- Sys.time()
message(sprintf("\n  FASE 2 completada en %.1f minutos.",
                as.numeric(difftime(t1_bal, t0_bal, units = "mins"))))

bal_summary <- bind_rows(bal_fold_tbl) |> arrange(desc(F1))

cat("\n-- Resultados por técnica (OOF global, train sin Bogotá) --\n")
print(bal_summary |> mutate(across(where(is.double), ~ round(., 4))))

cat("\n-- Referencia XGB_005 (benchmark — ver model_registry.csv) --\n")
cat("   NOTA: F1 OOF de XGB_007 no es directamente comparable con XGB_005/006\n")
cat("   porque se calcula sobre un training set diferente (sin Bogotá).\n")
cat("   La comparación definitiva es el Kaggle public F1.\n")

best_technique <- bal_summary |>
  slice_max(F1, n = 1, with_ties = FALSE) |>
  pull(technique)
best_bal_row   <- bal_summary |> filter(technique == best_technique)

message("\n  Mejor técnica: ", best_technique,
        "  (F1_OOF=", round(best_bal_row$F1, 4),
        " | th=", best_bal_row$threshold, ")")

write_csv(bal_summary, file.path(dir_model, "cv_balance_summary.csv"))
message("  Guardado: cv_balance_summary.csv")

# =============================================================================
# SECCIÓN 7: FASE 3 — MODELO FINAL
# =============================================================================
message("\n", strrep("=", 65))
message("  FASE 3 — MODELO FINAL")
message("  HP: depth=", best_hp$max_depth,
        " eta=", best_hp$eta,
        " gam=", best_hp$gamma,
        " L2=", best_hp$lambda)
message("  Técnica: ", best_technique,
        "  |  nrounds (mediana Fase 2): ", bal_nr_list[[best_technique]])
message("  Train: ", nrow(train), " obs (sin Bogotá)")
message(strrep("=", 65))

X_final_mat  <- mat_train_full
y_final_int  <- y_int
params_final <- best_params

if (best_technique == "class_weights") {
  spw_final    <- n_neg / n_pos
  params_final <- c(best_params, list(scale_pos_weight = spw_final))
  message("  scale_pos_weight = ", round(spw_final, 3))

} else if (best_technique == "downsample") {
  set.seed(42)
  y_f      <- factor(ifelse(y_int == 1, "Yes", "No"), levels = niveles)
  df_ds    <- as_tibble(mat_train_full)
  ds_final <- caret::downSample(x = df_ds, y = y_f, yname = "pobre")
  X_final_mat <- as.matrix(ds_final |> select(-pobre))
  y_final_int <- as.integer(ds_final$pobre == "Yes")
  message("  Down-sampled: ", nrow(X_final_mat), " obs | ",
          "Yes=", sum(y_final_int), " / No=", sum(y_final_int == 0))

} else if (best_technique == "upsample") {
  set.seed(42)
  y_f      <- factor(ifelse(y_int == 1, "Yes", "No"), levels = niveles)
  df_us    <- as_tibble(mat_train_full)
  us_final <- caret::upSample(x = df_us, y = y_f, yname = "pobre")
  X_final_mat <- as.matrix(us_final |> select(-pobre))
  y_final_int <- as.integer(us_final$pobre == "Yes")
  message("  Up-sampled: ", nrow(X_final_mat), " obs | ",
          "Yes=", sum(y_final_int), " / No=", sum(y_final_int == 0))

} else if (best_technique == "smote") {
  set.seed(42)
  y_f      <- factor(ifelse(y_int == 1, "Yes", "No"), levels = niveles)
  df_sm    <- bind_cols(as_tibble(mat_train_full), pobre = y_f)
  df_out   <- themis::smote(df_sm, var = "pobre", k = 5, over_ratio = 1)
  X_final_mat <- as.matrix(df_out |> select(-pobre))
  y_final_int <- as.integer(df_out$pobre == "Yes")
  message("  SMOTE: ", nrow(X_final_mat), " obs | ",
          "Yes=", sum(y_final_int), " / No=", sum(y_final_int == 0))
}

best_nrounds_final <- bal_nr_list[[best_technique]]
best_th            <- best_bal_row$threshold

params_final_full <- c(params_final, list(nthread = N_THREADS))
dtrain_final      <- xgboost::xgb.DMatrix(X_final_mat, label = y_final_int)

message("\nEntrenando modelo final (", nrow(X_final_mat), " obs | nrounds=",
        best_nrounds_final, " | features=", P_TOTAL, ")...")
t0_f <- proc.time()

set.seed(42)
xgb_final <- xgboost::xgb.train(
  params  = params_final_full,
  data    = dtrain_final,
  nrounds = best_nrounds_final,
  verbose = 0
)

t1_f <- proc.time()
message("  Modelo final entrenado en ",
        round((t1_f - t0_f)[["elapsed"]] / 60, 2), " min")
message("  Threshold OOF (mejor técnica): ", best_th)

varimp_xgb <- xgboost::xgb.importance(
  feature_names = colnames(mat_train_full),
  model         = xgb_final
)
write_csv(as_tibble(varimp_xgb), file.path(dir_model, "feature_importance.csv"))
message("  Guardado: feature_importance.csv (", nrow(varimp_xgb), " variables)")

oof_best  <- bal_oof_list[[best_technique]]
met_final <- calcular_metricas_oof(oof_best, y_factor, y_int)
roc_obj   <- met_final$roc_obj
auc_roc   <- met_final$AUC_ROC

message("  F1_OOF (train sin Bogotá) : ", round(met_final$F1,        4))
message("  Precision                 : ", round(met_final$Precision,  4))
message("  Recall                    : ", round(met_final$Recall,     4))
message("  AUC-ROC_OOF               : ", round(auc_roc,             4))
message("  RECORDATORIO: comparar con XGB_005/006 via Kaggle, no solo OOF.")

# =============================================================================
# SECCIÓN 8: GRÁFICOS DE DIAGNÓSTICO
# =============================================================================
message("\n== Generando gráficos ==")

# --- Gráfico 1: F1 OOF por técnica de balanceo ---
p_bal <- bal_summary |>
  mutate(is_best = technique == best_technique) |>
  ggplot(aes(x = reorder(technique, F1), y = F1, fill = is_best)) +
  geom_col(alpha = 0.85, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.4f", F1)), hjust = -0.1, size = 3.5) +
  scale_fill_manual(values = c("FALSE" = "#BDBDBD", "TRUE" = "#534AB7")) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(
    title    = paste0("XGBoost: F1 OOF por técnica de balanceo — ", MODEL_ID),
    subtitle = sprintf(
      "Mejor: %s | F1_OOF = %.4f | th = %.2f  (Sin vars. construidas, sin Bogotá)",
      best_technique, best_bal_row$F1, best_bal_row$threshold
    ),
    x        = "Técnica de balanceo",
    y        = "F1 (OOF global, train sin Bogotá)",
    caption  = "F1 calculado sobre predicciones OOF del training filtrado — estimación honesta."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "balance_comparison.png"),
       p_bal, width = 7, height = 5, dpi = 150)
message("  Guardado: balance_comparison.png")

# --- Gráfico 2: Importancia de variables (Top 20 por Gain) ---
top20_imp <- varimp_xgb |> head(20)

p_varimp <- ggplot(top20_imp, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#534AB7", alpha = 0.85) +
  coord_flip() +
  labs(
    title    = paste0("XGBoost: importancia de variables (Top 20, Gain) — ", MODEL_ID),
    subtitle = paste0("Técnica: ", best_technique,
                      " | depth=", best_hp$max_depth,
                      " | eta=", best_hp$eta,
                      " | gamma=", best_hp$gamma,
                      " | nrounds=", best_nrounds_final,
                      " | features=", P_TOTAL,
                      " | sin Bogotá"),
    x        = NULL,
    y        = "Gain (reducción media de logloss)"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "varimp.png"),
       p_varimp, width = 8, height = 6, dpi = 150)
message("  Guardado: varimp.png")

# --- Gráfico 3: Threshold vs métricas ---
th_full <- map_dfr(seq(0.05, 0.95, by = 0.01), function(th) {
  pred <- factor(ifelse(oof_best >= th, "Yes", "No"), levels = niveles)
  TP   <- sum(pred == "Yes" & y_factor == "Yes")
  FP   <- sum(pred == "Yes" & y_factor == "No")
  FN   <- sum(pred == "No"  & y_factor == "Yes")
  f1   <- if ((2*TP+FP+FN)==0) NA_real_ else 2*TP/(2*TP+FP+FN)
  prec <- if ((TP+FP)==0)       NA_real_ else TP/(TP+FP)
  rec  <- if ((TP+FN)==0)       NA_real_ else TP/(TP+FN)
  tibble(threshold = th, F1 = f1, Precision = prec, Recall = rec)
})

p_threshold <- th_full |>
  filter(!is.na(F1)) |>
  pivot_longer(cols = c(F1, Precision, Recall), names_to = "metrica") |>
  ggplot(aes(x = threshold, y = value, color = metrica)) +
  geom_line(linewidth = 0.9) +
  geom_vline(xintercept = best_th, linetype = "dashed", color = "#534AB7") +
  geom_vline(xintercept = 0.5,    linetype = "dotted",  color = "gray40") +
  annotate("text", x = best_th + 0.02, y = 0.08,
           label = sprintf("th = %.2f", best_th),
           hjust = 0, size = 3.2, color = "#534AB7") +
  scale_color_manual(
    values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3")
  ) +
  labs(
    title    = paste0("XGBoost: métricas por threshold (OOF) — ", MODEL_ID),
    subtitle = sprintf(
      "th=%.2f | técnica=%s | depth=%d | eta=%.3f | train sin Bogotá",
      best_th, best_technique, best_hp$max_depth, best_hp$eta
    ),
    x        = "Threshold de clasificación",
    y        = "Valor de la métrica",
    color    = NULL,
    caption  = "OOF = predicciones fuera de muestra (honesto). Discontinua = th óptimo."
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

ggsave(file.path(dir_model, "threshold.png"),
       p_threshold, width = 8, height = 5, dpi = 150)
message("  Guardado: threshold.png")

# --- Gráfico 4: Curva ROC ---
roc_df <- tibble(
  fpr = 1 - roc_obj$specificities,
  tpr = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.72, y = 0.65,
           label = "Clasificador aleatorio", size = 3, color = "gray50", angle = 35) +
  annotate("text", x = 0.55, y = 0.15,
           label = sprintf("AUC-ROC = %.4f\n(benchmark: XGB_005)", auc_roc),
           size = 3.5, color = "#534AB7") +
  scale_x_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  labs(
    title    = paste0("XGBoost: curva ROC — ", MODEL_ID),
    subtitle = sprintf(
      "depth=%d | eta=%.3f | gamma=%.1f | técnica: %s | OOF sin Bogotá",
      best_hp$max_depth, best_hp$eta, best_hp$gamma, best_technique
    ),
    x        = "Tasa de Falsos Positivos  (1 - Specificity)",
    y        = "Tasa de Verdaderos Positivos  (Recall)",
    caption  = "AUC calculado sobre predicciones OOF (train sin Bogotá) — honesto."
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc.png"),
       p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc.png")

# --- Gráfico 5: Curva Precision-Recall ---
n_pos_oof   <- sum(y_int)
n_neg_oof   <- length(y_int) - n_pos_oof
pr_curve_df <- tibble(
  TP        = roc_obj$sensitivities * n_pos_oof,
  FP        = (1 - roc_obj$specificities) * n_neg_oof,
  recall    = roc_obj$sensitivities,
  precision = TP / (TP + FP)
) |> filter(is.finite(precision), is.finite(recall))

prev_rate <- n_pos_oof / (n_pos_oof + n_neg_oof)
pt_opt    <- tibble(recall = met_final$Recall, precision = met_final$Precision)

p_prcurve <- ggplot(pr_curve_df, aes(x = recall, y = precision)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_hline(yintercept = prev_rate, linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.85, y = prev_rate + 0.015,
           label = sprintf("Clasificador aleatorio (%.0f%%)", prev_rate * 100),
           size = 3, color = "gray50") +
  geom_point(data = pt_opt, aes(x = recall, y = precision),
             color = "#E84855", size = 3) +
  annotate("text",
           x     = met_final$Recall - 0.1,
           y     = met_final$Precision + 0.015,
           label = sprintf("th=%.2f\nF1=%.4f", best_th, met_final$F1),
           size  = 3, color = "#E84855") +
  labs(
    title    = paste0("XGBoost: curva Precision-Recall — ", MODEL_ID),
    subtitle = sprintf(
      "AUC-ROC OOF: %.4f | depth=%d | gamma=%.1f | técnica: %s | sin Bogotá",
      auc_roc, best_hp$max_depth, best_hp$gamma, best_technique
    ),
    x        = "Recall  (fracción de pobres capturada)",
    y        = "Precision  (fracción de predichos pobres que lo son)",
    caption  = "OOF = predicciones fuera de muestra (train sin Bogotá). Punto rojo = threshold óptimo."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve.png"),
       p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve.png")

# Guardar diagnósticos completos
saveRDS(
  list(
    # Contexto y cambios
    model_id_benchmark = "XGB_005",
    cambios_vs_xgb005  = list(
      variables_excluidas = paste0(
        "17 variables: 16 construidas (igual que XGB_006) + bogota. ",
        "Cuadráticas: educ_jefe_sq, educ_prom_sq, ratio_depend_sq, ",
        "tasa_ocup_sq, hacinamiento_sq, edad_jefe_sq. ",
        "Logarítmicas: log_num_personas, log_n_menores. ",
        "Interacciones: educ_x_formal, educ_x_ocup, menores_x_desocup, ",
        "depend_x_informal, rural_x_cta_propia, educ_x_rural, ",
        "subsidiado_x_menores, jefe_mayor_pension. ",
        "Dummy constante: bogota (varianza nula tras filtro)."
      ),
      bogota_excluida    = paste0(
        "Sí. Filtro: dpto != '", DPTO_BOGOTA, "'. ",
        "Hogares eliminados: ", n_bogota, " (",
        round(100 * n_bogota / nrow(train_raw), 1), "% del train original). ",
        "Razón: Bogotá no está en test; entrenar en un dominio ausente en test ",
        "puede introducir sesgo de distribución."
      ),
      n_features         = paste0("110 (XGB_005) → ", P_TOTAL, " (XGB_007)"),
      n_train_obs        = paste0(nrow(train_raw), " (XGB_005) → ",
                                  nrow(train), " (XGB_007)"),
      comparacion_xgb006 = paste0(
        "XGB_006 tiene las mismas features excluidas pero usa Bogotá en train. ",
        "Diferencia XGB_007 vs XGB_006 aísla el efecto de excluir Bogotá. ",
        "Diferencia XGB_007 vs XGB_005 mide el efecto combinado."
      ),
      grid_hp            = "Sin cambio respecto a XGB_005"
    ),
    vars_excluidas       = VARS_EXCLUIR,
    n_bogota_excluidos   = n_bogota,
    pob_prevalencia_orig = pob_original,
    pob_prevalencia_filt = pob_filtrada,
    # Resultados
    hp_results           = hp_results,
    best_hp              = best_hp,
    best_params          = best_params,
    bal_summary          = bal_summary,
    best_technique       = best_technique,
    best_th              = best_th,
    best_nrounds         = best_nrounds_final,
    oof_best             = oof_best,
    xgb_final            = xgb_final,
    varimp_xgb           = varimp_xgb,
    roc_obj              = roc_obj,
    auc_roc              = auc_roc,
    f1_oof               = met_final$F1,
    precision_oof        = met_final$Precision,
    recall_oof           = met_final$Recall
  ),
  file.path(dir_model, "diagnostics.rds")
)
message("  Guardado: diagnostics.rds")

# =============================================================================
# SECCIÓN 9: SUBMISSION PARA KAGGLE
# =============================================================================
message("\n== Generando submission ==")

dtest_final <- xgboost::xgb.DMatrix(mat_test_full)
probs_test  <- predict(xgb_final, dtest_final)
pred_test   <- as.integer(probs_test >= best_th)

submission <- tibble(id = ids, pobre = pred_test)
sub_name   <- sprintf(
  "XGB_007_depth%d_gam%.1f_L2%.0f_eta%.3f_nrounds%d_%s_th%03.0f_sinBogota.csv",
  best_hp$max_depth, best_hp$gamma, best_hp$lambda,
  best_hp$eta, best_nrounds_final,
  best_technique, best_th * 100
)
sub_path <- file.path(dir_subs, sub_name)
write_csv(submission, sub_path)

message("  Guardado: ", sub_path)
message("  Pobres predichos: ", sum(pred_test),
        " (", round(mean(pred_test) * 100, 1), "%)")

# =============================================================================
# SECCIÓN 10: REGISTRO EN model_registry.csv
# =============================================================================
message("\n== Registro en model_registry.csv ==")

nueva_fila <- tibble(
  model_id           = MODEL_ID,
  fecha              = Sys.Date(),
  autor              = AUTOR,
  algoritmo          = "XGBoost",
  n_features         = P_TOTAL,
  imbalance_strategy = best_technique,
  cv_folds           = K_FOLDS,
  cv_F1              = round(met_final$F1,        4),
  cv_Precision       = round(met_final$Precision, 4),
  cv_Recall          = round(met_final$Recall,    4),
  auc_roc            = round(auc_roc,             4),
  kaggle_public_F1   = NA_real_,
  threshold          = best_th,
  notas              = paste0(
    "XGBoost sin vars construidas + sin Bogotá (successor XGB_005). ",
    "Excluidas 16 vars construidas (igual que XGB_006) + bogota (const. 0). ",
    "Bogotá excluida de train: dpto!='", DPTO_BOGOTA, "' (", n_bogota, " obs, ",
    round(100 * n_bogota / nrow(train_raw), 1), "% del train original). ",
    "Prevalencia pobreza: ", round(100 * pob_original, 2), "% → ",
    round(100 * pob_filtrada, 2), "% tras filtro. ",
    "N_features=", P_TOTAL, " | N_train=", nrow(train), ". ",
    "Grid HP idéntico a XGB_005. ",
    "Mejor HP: depth=", best_hp$max_depth,
    " eta=", best_hp$eta,
    " gam=", best_hp$gamma,
    " mcw=", best_hp$min_child_weight,
    " col_t=", best_hp$colsample_bytree,
    " col_l=", best_hp$colsample_bylevel,
    " sub=", best_hp$subsample,
    " L2=", best_hp$lambda,
    " L1=", best_hp$alpha,
    " F1_HP=", best_hp$F1_OOF, ". ",
    "Mejor balanceo: ", best_technique,
    " F1=", round(met_final$F1, 4),
    " nrounds=", best_nrounds_final,
    " th=", best_th,
    " AUC=", round(auc_roc, 4), "."
  ),
  cp              = NA_real_,
  maxdepth        = best_hp$max_depth,
  train_F1        = round(met_final$F1,        4),
  train_Precision = round(met_final$Precision, 4),
  train_Recall    = round(met_final$Recall,    4)
)

if (file.exists(reg_path)) {
  registry <- read_csv(reg_path, show_col_types = FALSE) |>
    mutate(fecha = as.Date(fecha, origin = "1899-12-30"))
  registry <- registry |> filter(model_id != MODEL_ID)
  registry <- bind_rows(registry, nueva_fila)
} else {
  registry <- nueva_fila
}
write_csv(registry, reg_path)
message("  Registro actualizado: ", reg_path)

cat("\n-- Fila registrada --\n")
print(nueva_fila)

cat("\n", strrep("=", 65), "\n")
cat("  ", MODEL_ID, " — RESUMEN FINAL\n")
cat(strrep("=", 65), "\n")
cat(sprintf("  F1_OOF (train sin Bogotá) : %.4f\n", met_final$F1))
cat(sprintf("  AUC-ROC_OOF               : %.4f\n", auc_roc))
cat(sprintf("  Threshold                 : %.2f\n", best_th))
cat(sprintf("  Técnica balanceo          : %s\n", best_technique))
cat(sprintf("  depth=%-2d  eta=%.3f  gamma=%.1f  lambda=%.0f  nrounds=%d\n",
            best_hp$max_depth, best_hp$eta, best_hp$gamma,
            best_hp$lambda, best_nrounds_final))
cat(sprintf("  N° features               : %d  (XGB_005: 110 | excluidas: %d)\n",
            P_TOTAL, length(VARS_EXCLUIR)))
cat(sprintf("  Bogotá en train           : EXCLUIDA (%d obs, %.1f%%)\n",
            n_bogota, 100 * n_bogota / nrow(train_raw)))
cat(sprintf("  Prevalencia pobreza train : %.2f%% → %.2f%% (delta: %.2f pp)\n",
            100 * pob_original, 100 * pob_filtrada,
            100 * (pob_filtrada - pob_original)))
cat(sprintf("  Submission                : %s\n", sub_path))
cat(strrep("=", 65), "\n\n")
cat("  → Subir a Kaggle y anotar kaggle_public_F1 en model_registry.csv\n")
cat("  → Comparar XGB_007 vs XGB_006 (via Kaggle) para aislar efecto Bogotá:\n")
cat("     XGB_007 >= XGB_006 → excluir Bogotá mejora generalización\n")
cat("     XGB_007 <  XGB_006 → Bogotá aportaba varianza útil al modelo\n")
cat("  → Comparar XGB_007 vs XGB_005 para medir efecto combinado\n\n")

# nolint end
