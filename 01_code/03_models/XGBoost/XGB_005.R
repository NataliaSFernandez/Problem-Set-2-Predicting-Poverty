#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/XGBoost/XGB_005.R
#==============================================================================
#
# ALGORITMO: XGBoost — Búsqueda aleatoria de HP (foco anti-sobreajuste)
#
# MOTIVACIÓN — POR QUÉ XGB_005 EXISTE:
#   XGB_004 fue el mejor modelo del equipo en F1 OOF (0.7383) pero mostró
#   sobreajuste claro: Kaggle public F1 = 0.72 → brecha de ~0.018 puntos.
#   Este script reenfoca la búsqueda de HP hacia configuraciones que
#   generalicen mejor, sin cambiar la arquitectura del modelo.
#
#   El diagnóstico del sobreajuste en XGB_004 arrojó seis posibles causas:
#
#   1. subsample = 1.0 en el mejor HP:
#      Sin submuestreo de filas por árbol → máxima correlación entre árboles
#      consecutivos → el ensemble memoriza el train más de lo necesario.
#      Se excluye 1.0 del grid (rango: 0.6–0.9).
#
#   2. max_depth = 8 en el mejor HP:
#      Árboles con 8 niveles crean particiones muy finas que capturan ruido
#      del train. Se limita el grid a {3, 4, 5, 6}.
#
#   3. gamma = 0, lambda = 0.1 — regularización casi nula:
#      El mejor combo de XGB_004 no penaliza splits ni pesos de hoja.
#      Se desplaza el grid hacia valores más altos de ambos parámetros:
#        gamma  : {0, 0.05, …, 2}  →  {0.5, 1, 2, 3, 5}
#        lambda : {0, 0.1, …, 5}   →  {1, 2, 5, 10}
#
#   4. EARLY_STOP = 25 demasiado corto:
#      14 de los 30 combos en XGB_004 alcanzaron nrounds_sd = 0
#      (todos los folds llegaron al máximo de 499 sin parar). Con una ventana
#      de solo 25 rondas el modelo puede seguir mejorando en train mientras
#      ya sobreajusta en val. Se sube a 50 rondas.
#
#   5. Sin colsample_bylevel:
#      XGB_004 solo regulaba columnas por árbol (colsample_bytree). Añadir
#      colsample_bylevel agrega una capa extra de aleatoriedad por nivel de
#      profundidad, especialmente útil cuando max_depth es alto.
#
#   6. (NO implementado) eval_metric "logloss" vs. objetivo F1:
#      El early stopping sigue usando logloss. Cambiar a "aucpr" podría
#      alinear mejor el criterio de parada con F1 en datos desbalanceados,
#      pero se omite en esta versión para no cambiar múltiples variables
#      a la vez y mantener comparabilidad directa con XGB_004.
#
# ESTRUCTURA (idéntica a XGB_004):
#
#   FASE 1 — Búsqueda aleatoria de hiperparámetros
#     N_SEARCH combinaciones aleatorias × CV-5 + early stopping.
#     Métrica: F1 OOF (max sobre threshold grid 0.05–0.95).
#     Variables: TODAS (110, ciudad/dpto como entero ordinal).
#     Checkpoint: si hp_search_results.csv ya existe, se salta la fase.
#
#   FASE 2 — Comparación de técnicas de balanceo de clases
#     Con HP óptimos de Fase 1, se comparan 5 técnicas via CV-5 OOF:
#       (a) baseline, (b) class_weights, (c) downsample,
#       (d) upsample,  (e) smote
#     Rebalanceo aplicado SOLO al fold de entrenamiento → sin data leakage.
#     Threshold óptimo determinado sobre OOF global.
#
#   FASE 3 — Modelo final
#     nrounds = mediana de best_iteration de los 5 folds de Fase 2.
#     Threshold = OOF de la Fase 2 (técnica ganadora).
#     Genera submission para Kaggle.
#
# TABLA COMPARATIVA DE PARÁMETROS: XGB_004 → XGB_005
# ┌──────────────────────┬───────────────────────────┬───────────────────────────┐
# │ Parámetro            │ XGB_004                   │ XGB_005                   │
# ├──────────────────────┼───────────────────────────┼───────────────────────────┤
# │ max_depth (grid)     │ {3,4,5,6,8,10}            │ {3,4,5,6}                 │
# │ subsample (grid)     │ {0.5,0.6,0.7,0.8,0.9,1.0}│ {0.6,0.7,0.8,0.9}        │
# │ gamma (grid)         │ {0,0.05,0.1,0.3,0.5,1,2} │ {0.5,1,2,3,5}            │
# │ lambda (grid)        │ {0,0.1,0.5,1,2,5}         │ {1,2,5,10}               │
# │ colsample_bylevel    │ no incluido               │ {0.5,0.7,1.0}            │
# │ EARLY_STOP           │ 25                        │ 50                        │
# │ eval_metric          │ logloss                   │ logloss (sin cambio)      │
# └──────────────────────┴───────────────────────────┴───────────────────────────┘
#
# Parámetros SIN CAMBIO respecto a XGB_004:
#   eta              : {0.01,0.03,0.05,0.08,0.10,0.15,0.20}
#   min_child_weight : {1,3,5,8,10,15,20}
#   colsample_bytree : {0.3,0.4,0.5,0.6,0.7,0.8,1.0}
#   alpha            : {0,0.05,0.1,0.5,1,2}
#   N_SEARCH         : 30
#   MAX_NROUNDS      : 500
#   K_FOLDS          : 5
#   Variables        : todas las 110 (ciudad/dpto entero ordinal)
#
# REFERENCIA XGB_004 (mejor resultado anterior):
#   depth=8  eta=0.08  gamma=0  mcw=10  col=0.3  sub=1.0
#   lambda=0.1  alpha=0  nrounds=316  técnica=baseline
#   F1_OOF=0.7383  Kaggle=0.72  brecha=0.018
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#
# OUTPUTS:
#   02_outputs/models/XGBoost/XGB_005/hp_search_results.csv
#   02_outputs/models/XGBoost/XGB_005/cv_balance_results.csv
#   02_outputs/models/XGBoost/XGB_005/cv_balance_summary.csv
#   02_outputs/models/XGBoost/XGB_005/balance_comparison.png
#   02_outputs/models/XGBoost/XGB_005/varimp.png
#   02_outputs/models/XGBoost/XGB_005/threshold.png
#   02_outputs/models/XGBoost/XGB_005/roc.png
#   02_outputs/models/XGBoost/XGB_005/prcurve.png
#   02_outputs/models/XGBoost/XGB_005/diagnostics.rds
#   03_submissions/XGB_005_*.csv
#   02_outputs/model_registry.csv  (append)
#
# TIEMPO ESTIMADO: ~35–55 min con 9 cores
#   Fase 1: N_SEARCH=30 × 5 folds + early stopping (~25–40 min)
#   Fase 2: 5 técnicas × 5 folds + early stopping (~8–12 min)
#   Fase 3: modelo final (~1 min)
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
MODEL_ID <- "XGB_005"

set.seed(42)

N_THREADS   <- max(1L, parallel::detectCores() - 1L)

# --- Búsqueda de HP ---
N_SEARCH    <- 30L    # igual que XGB_004; suficiente para explorar el espacio reducido

MAX_NROUNDS <- 500L   # techo de iteraciones (sin cambio respecto a XGB_004)

# EARLY_STOP: 25 → 50
# Razón: en XGB_004, 14 de 30 combos tenían nrounds_sd = 0 (todos los folds
# llegaron al límite de 499). Con solo 25 rondas de ventana el early stopping
# no detectaba el verdadero pico de validación en configs con eta bajo (≤0.05),
# lo que dejaba correr el modelo más allá del punto óptimo de F1.
EARLY_STOP  <- 50L

K_FOLDS     <- 5L
TECHNIQUES  <- c("baseline", "class_weights", "downsample", "upsample", "smote")
niveles     <- c("No", "Yes")

dir_model <- file.path("02_outputs/models/XGBoost", MODEL_ID)
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model, recurse = TRUE)
fs::dir_create(dir_subs,  recurse = TRUE)

message(strrep("=", 65))
message("  ", MODEL_ID, "  —  ", AUTOR)
message("  Successor de XGB_004 (F1_OOF=0.7383, Kaggle=0.72, brecha=0.018)")
message("  Enfoque: regularización más fuerte para reducir sobreajuste")
message("  Cores XGBoost (nthread) : ", N_THREADS)
message("  Búsqueda HP (N_SEARCH)  : ", N_SEARCH)
message("  Max rondas / Early stop : ", MAX_NROUNDS, " / ", EARLY_STOP,
        "  (XGB_004: 500 / 25)")
message("  Técnicas de balanceo    : ", paste(TECHNIQUES, collapse = ", "))
message(strrep("=", 65))

# =============================================================================
# SECCIÓN 2: CARGAR DATOS
# =============================================================================
message("\n== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id

message("  Train: ", nrow(train), " filas x ", ncol(train), " columnas")
message("  Test : ", nrow(test),  " filas x ", ncol(test),  " columnas")

y_factor <- factor(ifelse(train$pobre == 1, "Yes", "No"), levels = niveles)
y_int    <- as.integer(y_factor == "Yes")
n_pos    <- sum(y_int)
n_neg    <- length(y_int) - n_pos
message("  Balance: ", n_pos, " pobres (", round(100 * mean(y_int), 1), "%) | ",
        n_neg, " no-pobres (", round(100 * (1 - mean(y_int)), 1), "%)")

# =============================================================================
# SECCIÓN 3: PREPROCESAMIENTO
# =============================================================================
# Idéntico a XGB_004: ciudad y dpto se codifican como entero ordinal.
message("\n== Preprocesando variables (ciudad y dpto como entero ordinal) ==")

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

cols_comunes <- intersect(names(X_train_full), names(X_test_full))
X_train_full <- X_train_full |> select(all_of(cols_comunes))
X_test_full  <- X_test_full  |> select(all_of(cols_comunes))

P_TOTAL <- ncol(X_train_full)
message("  Predictores: ", P_TOTAL,
        "  (ciudad: ", "ciudad" %in% names(X_train_full),
        " | dpto: ", "dpto" %in% names(X_train_full), ")")
message("  NAs en X_train_full: ", sum(is.na(X_train_full)))

mat_train_full <- as.matrix(X_train_full)
mat_test_full  <- as.matrix(X_test_full)

# =============================================================================
# SECCIÓN 4: FUNCIONES AUXILIARES
# =============================================================================
# Idénticas a XGB_004.

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

# entrenar_fold_xgb: ahora recibe colsample_bylevel dentro de params
# (lo pasa como parte del list params; XGBoost lo aplica internamente).
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
# Folds estratificados: mismos que XGB_004 (semilla 42, createFolds).
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
  message("  N_SEARCH=", N_SEARCH, " combos | todas las ", P_TOTAL, " variables")
  message("  Grid ajustado vs. XGB_004 para reducir sobreajuste:")
  message("    max_depth     : {3,4,5,6}         (XGB_004: {3,4,5,6,8,10})")
  message("    subsample     : {0.6,0.7,0.8,0.9} (XGB_004: {0.5,...,1.0})")
  message("    gamma         : {0.5,1,2,3,5}     (XGB_004: {0,0.05,...,2})")
  message("    lambda        : {1,2,5,10}         (XGB_004: {0,0.1,...,5})")
  message("    colsample_bylevel: {0.5,0.7,1.0}  (XGB_004: no incluido)")
  message("    EARLY_STOP    : 50                 (XGB_004: 25)")
  message(strrep("=", 65))

  set.seed(42)
  hp_grid <- tibble(

    # max_depth: {3,4,5,6}
    # XGB_004 usaba hasta 10; el mejor HP resultó depth=8, que genera
    # particiones muy finas → sobreajuste. Se limita a 6 para controlar
    # complejidad estructural del árbol.
    max_depth = sample(c(3L, 4L, 5L, 6L),
                       N_SEARCH, replace = TRUE),

    # eta: sin cambio. Rango amplio para encontrar la tasa óptima de
    # aprendizaje independientemente de la regularización.
    eta = sample(c(0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20),
                 N_SEARCH, replace = TRUE),

    # gamma: {0.5, 1, 2, 3, 5}
    # XGB_004 usaba {0, 0.05, …, 2}; el mejor HP resultó gamma=0 (sin
    # penalización de splits). Desplazar el grid completo hacia valores
    # mayores obliga a explorar el espacio donde los splits solo se
    # realizan cuando generan ganancia suficiente → árboles más simples.
    gamma = sample(c(0.5, 1.0, 2.0, 3.0, 5.0),
                   N_SEARCH, replace = TRUE),

    # min_child_weight: sin cambio. Controla el mínimo de suma de pesos
    # en un nodo hoja; con datos desbalanceados valores altos (10-20) ya
    # eran eficaces en XGB_004.
    min_child_weight = sample(c(1L, 3L, 5L, 8L, 10L, 15L, 20L),
                              N_SEARCH, replace = TRUE),

    # colsample_bytree: sin cambio. 0.3 en XGB_004 ya era bajo (buena
    # regularización por columna); mantenemos el rango completo para
    # dejar que el search confirme ese nivel.
    colsample_bytree = sample(c(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0),
                              N_SEARCH, replace = TRUE),

    # colsample_bylevel: {0.5, 0.7, 1.0} — NUEVO en XGB_005
    # Submuestrea columnas por nivel de profundidad del árbol (adicional
    # al submuestreo por árbol de colsample_bytree). Proporciona
    # regularización extra especialmente efectiva cuando max_depth > 3,
    # porque las decisiones en niveles profundos ven menos variables.
    colsample_bylevel = sample(c(0.5, 0.7, 1.0),
                               N_SEARCH, replace = TRUE),

    # subsample: {0.6, 0.7, 0.8, 0.9}
    # XGB_004 permitía subsample=1.0; el mejor HP resultó precisamente
    # en 1.0 (sin submuestreo de filas → correlación máxima entre árboles).
    # Se excluye 1.0 del grid para forzar una inyección mínima de
    # aleatoriedad estocástica en cada árbol.
    subsample = sample(c(0.6, 0.7, 0.8, 0.9),
                       N_SEARCH, replace = TRUE),

    # lambda (L2): {1, 2, 5, 10}
    # XGB_004 usaba {0, 0.1, …, 5}; el mejor HP resultó lambda=0.1
    # (mínima penalización de pesos de hoja). Se elimina el rango bajo
    # y se desplaza hacia valores que penalizan pesos extremos en las hojas,
    # reduciendo la varianza del modelo.
    lambda = sample(c(1.0, 2.0, 5.0, 10.0),
                    N_SEARCH, replace = TRUE),

    # alpha (L1): sin cambio. La penalización L1 induce sparsity en los
    # pesos de hoja; el rango actual ya cubre desde 0 (sin efecto) hasta
    # valores moderados.
    alpha = sample(c(0.0, 0.05, 0.1, 0.5, 1.0, 2.0),
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
      eval_metric       = "logloss",   # sin cambio vs XGB_004
      max_depth         = hp_i$max_depth,
      eta               = hp_i$eta,
      gamma             = hp_i$gamma,
      min_child_weight  = hp_i$min_child_weight,
      colsample_bytree  = hp_i$colsample_bytree,
      colsample_bylevel = hp_i$colsample_bylevel,  # nuevo en XGB_005
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

# Selección del mejor HP
cat("\n-- Top-5 combos HP (F1 OOF) --\n")
print(hp_results |>
        select(max_depth:alpha, best_nrounds, nrounds_sd, F1_OOF, AUC_ROC, Threshold) |>
        head(5))

# Diagnóstico: ¿cuántos combos agotaron nrounds sin converger?
n_no_converge <- sum(hp_results$nrounds_sd == 0 & hp_results$best_nrounds >= MAX_NROUNDS - 5,
                     na.rm = TRUE)
message(sprintf("\n  Combos sin convergencia (nrounds_sd=0 y nrounds≥%d): %d/%d",
                MAX_NROUNDS - 5, n_no_converge, nrow(hp_results)))
if (n_no_converge > 5) {
  message("  AVISO: muchos combos no convergen — considerar subir MAX_NROUNDS en la siguiente iteración.")
}

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
        " nrounds≤", MAX_NROUNDS, " (early stopping=", EARLY_STOP, ")")
message("  Técnicas: ", paste(TECHNIQUES, collapse = ", "))
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

cat("\n-- Resultados por técnica (OOF global) --\n")
print(bal_summary |> mutate(across(where(is.double), ~ round(., 4))))

# Referencia XGB_004 para comparación directa
cat("\n-- Referencia XGB_004 (baseline) --\n")
cat("   F1=0.7383  Precision=0.6845  Recall=0.8014  AUC=0.9429  th=0.34  nrounds=316\n")

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
        best_nrounds_final, ")...")
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

# Importancia de variables (Gain)
varimp_xgb <- xgboost::xgb.importance(
  feature_names = colnames(mat_train_full),
  model         = xgb_final
)
write_csv(as_tibble(varimp_xgb), file.path(dir_model, "feature_importance.csv"))
message("  Guardado: feature_importance.csv (", nrow(varimp_xgb), " variables)")

# Métricas finales (OOF de la mejor técnica — honesto)
oof_best  <- bal_oof_list[[best_technique]]
met_final <- calcular_metricas_oof(oof_best, y_factor, y_int)
roc_obj   <- met_final$roc_obj
auc_roc   <- met_final$AUC_ROC

message("  F1_OOF:      ", round(met_final$F1,        4),
        "  (XGB_004 baseline: 0.7383)")
message("  Precision:   ", round(met_final$Precision,  4),
        "  (XGB_004: 0.6845)")
message("  Recall:      ", round(met_final$Recall,     4),
        "  (XGB_004: 0.8014)")
message("  AUC-ROC_OOF: ", round(auc_roc,             4),
        "  (XGB_004: 0.9429)")

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
    subtitle = sprintf("Mejor: %s | F1_OOF = %.4f | th = %.2f  (XGB_004 baseline: 0.7383)",
                       best_technique, best_bal_row$F1, best_bal_row$threshold),
    x        = "Técnica de balanceo",
    y        = "F1 (OOF global)",
    caption  = "F1 calculado sobre predicciones OOF — estimación honesta."
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
                      " | nrounds=", best_nrounds_final),
    x        = NULL,
    y        = "Gain (reducción media de logloss)"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "varimp.png"),
       p_varimp, width = 8, height = 6, dpi = 150)
message("  Guardado: varimp.png")

# --- Gráfico 3: Threshold vs métricas (OOF de la mejor técnica) ---
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
    subtitle = sprintf("th=%.2f | técnica=%s | depth=%d | eta=%.3f | gamma=%.1f",
                       best_th, best_technique,
                       best_hp$max_depth, best_hp$eta, best_hp$gamma),
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
           label = sprintf("AUC-ROC = %.4f\n(XGB_004: 0.9429)", auc_roc),
           size = 3.5, color = "#534AB7") +
  scale_x_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  labs(
    title    = paste0("XGBoost: curva ROC — ", MODEL_ID),
    subtitle = sprintf("depth=%d | eta=%.3f | gamma=%.1f | técnica: %s | OOF (honesto)",
                       best_hp$max_depth, best_hp$eta,
                       best_hp$gamma, best_technique),
    x        = "Tasa de Falsos Positivos  (1 - Specificity)",
    y        = "Tasa de Verdaderos Positivos  (Recall)",
    caption  = "AUC calculado sobre predicciones OOF — estimación honesta."
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
    subtitle = sprintf("AUC-ROC OOF: %.4f | depth=%d | gamma=%.1f | técnica: %s",
                       auc_roc, best_hp$max_depth,
                       best_hp$gamma, best_technique),
    x        = "Recall  (fracción de pobres capturada)",
    y        = "Precision  (fracción de predichos pobres que lo son)",
    caption  = "OOF = predicciones fuera de muestra — honesto. Punto rojo = threshold óptimo."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve.png"),
       p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve.png")

# Guardar diagnósticos completos
saveRDS(
  list(
    # Contexto
    model_id_anterior  = "XGB_004",
    f1_oof_xgb004      = 0.7383,
    kaggle_xgb004      = 0.72,
    brecha_xgb004      = 0.018,
    cambios_vs_xgb004  = list(
      max_depth_grid     = "reducido de {3,4,5,6,8,10} a {3,4,5,6}",
      subsample_grid     = "excluido 1.0, rango {0.6,0.7,0.8,0.9}",
      gamma_grid         = "desplazado a {0.5,1,2,3,5}",
      lambda_grid        = "desplazado a {1,2,5,10}",
      colsample_bylevel  = "nuevo parámetro {0.5,0.7,1.0}",
      early_stop         = "aumentado de 25 a 50 rondas"
    ),
    # Resultados
    hp_results     = hp_results,
    best_hp        = best_hp,
    best_params    = best_params,
    bal_summary    = bal_summary,
    best_technique = best_technique,
    best_th        = best_th,
    best_nrounds   = best_nrounds_final,
    oof_best       = oof_best,
    xgb_final      = xgb_final,
    varimp_xgb     = varimp_xgb,
    roc_obj        = roc_obj,
    auc_roc        = auc_roc,
    f1_oof         = met_final$F1,
    precision_oof  = met_final$Precision,
    recall_oof     = met_final$Recall
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
  "XGB_005_depth%d_gam%.1f_L2%.0f_eta%.3f_nrounds%d_%s_th%03.0f.csv",
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
    "XGBoost anti-overfit (successor XGB_004 F1_OOF=0.7383 Kaggle=0.72). ",
    "Cambios: max_depth<={3..6}; subsample in {0.6..0.9}; ",
    "gamma in {0.5..5}; lambda in {1..10}; +colsample_bylevel; early_stop=50. ",
    "Fase1: random search HP (N=", N_SEARCH, " combos) CV-5 OOF. ",
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
    "Fase2: ", length(TECHNIQUES), " tecnicas balanceo OOF global. ",
    "Mejor: ", best_technique,
    " F1_bal=", round(met_final$F1, 4), ". ",
    "nrounds=", best_nrounds_final,
    " th=", best_th,
    " AUC=", round(auc_roc, 4), ". ",
    "Todas las ", P_TOTAL, " vars (ciudad/dpto entero-ordinal)."
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
cat(sprintf("  F1_OOF          : %.4f  (XGB_004: 0.7383)\n", met_final$F1))
cat(sprintf("  AUC-ROC_OOF     : %.4f  (XGB_004: 0.9429)\n", auc_roc))
cat(sprintf("  Threshold       : %.2f\n", best_th))
cat(sprintf("  Técnica balanceo: %s\n", best_technique))
cat(sprintf("  depth=%-2d  eta=%.3f  gamma=%.1f  lambda=%.0f  nrounds=%d\n",
            best_hp$max_depth, best_hp$eta, best_hp$gamma,
            best_hp$lambda, best_nrounds_final))
cat(sprintf("  Submission      : %s\n", sub_path))
cat(strrep("=", 65), "\n\n")
cat("  → Subir a Kaggle y anotar kaggle_public_F1 en model_registry.csv\n")
cat("  → Comparar la brecha OOF vs Kaggle con la de XGB_004 (~0.018)\n\n")

# nolint end
