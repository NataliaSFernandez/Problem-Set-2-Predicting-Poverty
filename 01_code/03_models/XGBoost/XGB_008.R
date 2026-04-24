#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/XGBoost/XGB_008.R
#==============================================================================
#
# ALGORITMO: XGBoost — Dos especificaciones: índices vs componentes
#
# MOTIVACIÓN — POR QUÉ XGB_008 EXISTE:
#   XGB_007 (base) usa ~93 variables: el conjunto completo sin las 16 vars
#   construidas (cuadráticas + logs + interacciones) y sin Bogotá en train.
#   Dentro de ese conjunto coexisten dos tipos de variables:
#
#   COMPONENTES: variables granulares originales (alguno_*, dep_*, prop_*)
#   ÍNDICES:     variables compuestas que combinan linealmente componentes
#                (indice_formalidad, n_privaciones, costa_caribe, etc.)
#
#   Para LPM, mezclar índices y sus componentes genera rank-deficiency (alias
#   exactos). Para XGBoost ese problema no existe, pero SÍ hay un trade-off:
#
#   Especificación A (componentes, sin índices):
#     → Mayor granularidad; XGBoost puede descubrir qué combinaciones importan
#       sin estar guiado por las agregaciones preconstruidas.
#     → Más variables → mayor dimensionalidad → más sobreajuste potencial.
#
#   Especificación B (índices, sin componentes):
#     → Menor dimensionalidad; las agregaciones preconstruidas pueden
#       capturar señales más robustas y reducir ruido en variables sparse.
#     → Menos flexibilidad: si el índice es una mala ponderación, pierde
#       información respecto a sus componentes individuales.
#
#   XGB_008 evalúa ambas especificaciones de forma independiente (HP search
#   separado) y retiene el ganador como modelo final.
#
# DEFINICIÓN DE ÍNDICES Y COMPONENTES:
#
#   Variables base (heredadas de XGB_007 — excluidas de ambas specs):
#     17 vars: cuadráticas, logarítmicas, interacciones + bogota
#     (ver XGB_007.R para lista completa)
#
#   SPEC A — excluye ÍNDICES, conserva COMPONENTES:
#     Bloq.2: doble_ingreso          (= 1−sin_ocupados−un_solo_ocupado)
#     Bloq.3: brecha_educ            (= nivel_educ_max − nivel_educ_jefe)
#     Bloq.4: indice_formalidad      (= mean(prop_asalariado, prop_cotiza_pension,
#                                           prop_prima_serv, prop_sub_transp))
#             formal_estricto        (= prop_asalariado × prop_cotiza_pension)
#             vulnerabilidad_lab     (= prop_domestico + prop_fam_sin_rem + prop_jornalero)
#     Bloq.5: n_fuentes_no_lab       (= suma de 8 alguno_*)
#             tiene_ingreso_no_lab   (= flag binario de los 8 alguno_*)
#             solo_subsidio          (= combinación de alguno_*)
#             ingresos_activos       (= alguno_intereses | alguno_arriendos)
#     Bloq.6: proteccion_social      (= alguno_pension_jub | prop_cotiza_pension > 0)
#             desprotegido           (= prop_cotiza_pension == 0 & ...)
#             prop_contributivo      (= prop_afil_salud − prop_subsidiado)
#     Bloq.7: costa_caribe, region_pacifico, eje_cafetero  (= dummies regionales)
#     Bloq.10:n_privaciones          (= suma de 5 dep_*)
#             multi_privado          (= n_privaciones >= 2)
#             privado_severo         (= n_privaciones >= 3)
#
#   SPEC B — excluye COMPONENTES, conserva ÍNDICES:
#     Bloq.2: sin_ocupados, un_solo_ocupado
#     Bloq.3: nivel_educ_max, nivel_educ_jefe
#     Bloq.4: prop_asalariado, prop_cotiza_pension, prop_prima_serv, prop_sub_transp,
#             prop_domestico, prop_fam_sin_rem, prop_jornalero
#     Bloq.5: alguno_pension_jub, alguno_remesas, alguno_subsidio, alguno_transf_nac,
#             alguno_intereses, alguno_cesantias, alguno_agropec, alguno_arriendos
#     Bloq.6: prop_afil_salud, prop_subsidiado
#     Bloq.10:dep_educacion, dep_vivienda, dep_empleo_formal, dep_proteccion,
#             dep_dependencia
#
#   NOTA GEOGRÁFICA: ciudad y dpto se conservan en AMBAS especificaciones
#   (codificados como entero ordinal, igual que XGB_007). No se aplican las
#   exclusiones de LPM_003A/003B sobre zona/ciudad/dpto, que eran específicas
#   del contexto OLS (multicolinealidad exacta con dummies por factor).
#
# DIFERENCIAS RESPECTO A XGB_007:
# ┌───────────────────────────┬──────────────────┬──────────────────┬──────────────────┐
# │ Aspecto                   │ XGB_007          │ XGB_008 Spec A   │ XGB_008 Spec B   │
# ├───────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
# │ Vars construidas (16+bog) │ Excluidas        │ Excluidas        │ Excluidas        │
# │ Bogotá en train           │ Excluida         │ Excluida         │ Excluida         │
# │ Índices compuestos        │ Incluidos        │ EXCLUIDOS        │ Incluidos        │
# │ Componentes granulares    │ Incluidos        │ Incluidos        │ EXCLUIDOS        │
# │ Búsqueda HP               │ independiente    │ independiente    │ independiente    │
# │ Grid HP                   │ sin cambio       │ sin cambio       │ sin cambio       │
# └───────────────────────────┴──────────────────┴──────────────────┴──────────────────┘
#
# ESTRUCTURA:
#   Fase 1A — Búsqueda aleatoria de HP (Spec A)
#   Fase 2A — Técnicas de balanceo (Spec A)
#   Fase 1B — Búsqueda aleatoria de HP (Spec B)
#   Fase 2B — Técnicas de balanceo (Spec B)
#   Comparación — selección del ganador por F1 OOF
#   Fase 3  — Modelo final de ambas specs + submission ganadora como XGB_008
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#
# OUTPUTS:
#   02_outputs/models/XGBoost/XGB_008/especA/hp_search_results.csv
#   02_outputs/models/XGBoost/XGB_008/especA/cv_balance_summary.csv
#   02_outputs/models/XGBoost/XGB_008/especB/hp_search_results.csv
#   02_outputs/models/XGBoost/XGB_008/especB/cv_balance_summary.csv
#   02_outputs/models/XGBoost/XGB_008/comparison.csv
#   02_outputs/models/XGBoost/XGB_008/varimp.png
#   02_outputs/models/XGBoost/XGB_008/balance_comparison_A.png
#   02_outputs/models/XGBoost/XGB_008/balance_comparison_B.png
#   02_outputs/models/XGBoost/XGB_008/threshold.png
#   02_outputs/models/XGBoost/XGB_008/roc.png
#   02_outputs/models/XGBoost/XGB_008/prcurve.png
#   02_outputs/models/XGBoost/XGB_008/diagnostics.rds
#   03_submissions/XGB_008A_*.csv   (spec A)
#   03_submissions/XGB_008B_*.csv   (spec B)
#   03_submissions/XGB_008_*.csv    (ganador, copia del mejor)
#   02_outputs/model_registry.csv   (append — solo el ganador como XGB_008)
#
# TIEMPO ESTIMADO: ~70–110 min con 9 cores (doble HP search independiente)
#   Fase 1A: ~25–40 min | Fase 2A: ~8–12 min
#   Fase 1B: ~25–40 min | Fase 2B: ~8–12 min
#   Fase 3 + gráficos: ~3 min
#
# REPRODUCIBILIDAD: semilla global 42. Correr desde raíz del proyecto.
#==============================================================================
# nolint start

# =============================================================================
# SECCIÓN 0: PAQUETES
# =============================================================================
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,
  xgboost,
  caret,
  themis,
  pROC,
  fs
)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
AUTOR    <- "Jonathan"
MODEL_ID <- "XGB_008"

set.seed(42)

N_THREADS   <- max(1L, parallel::detectCores() - 1L)
N_SEARCH    <- 30L
MAX_NROUNDS <- 500L
EARLY_STOP  <- 50L
K_FOLDS     <- 5L
TECHNIQUES  <- c("baseline", "class_weights", "downsample", "upsample", "smote")
niveles     <- c("No", "Yes")
DPTO_BOGOTA <- "11"

dir_model  <- file.path("02_outputs/models/XGBoost", MODEL_ID)
dir_spec_A <- file.path(dir_model, "especA")
dir_spec_B <- file.path(dir_model, "especB")
dir_subs   <- "03_submissions"
reg_path   <- "02_outputs/model_registry.csv"

fs::dir_create(dir_spec_A, recurse = TRUE)
fs::dir_create(dir_spec_B, recurse = TRUE)
fs::dir_create(dir_subs,   recurse = TRUE)

# =============================================================================
# SECCIÓN 2: LISTAS DE EXCLUSIÓN POR ESPECIFICACIÓN
# =============================================================================

# Exclusiones base heredadas de XGB_007 (aplicadas a AMBAS specs)
VARS_EXCLUIR_BASE <- c(
  # Cuadráticas de capital humano (Bloque 3)
  "educ_jefe_sq", "educ_prom_sq",
  # Logarítmicas (Bloque 9)
  "log_num_personas", "log_n_menores",
  # Cuadráticas generales (Bloque 9)
  "ratio_depend_sq", "tasa_ocup_sq", "hacinamiento_sq", "edad_jefe_sq",
  # Interacciones (Bloque 8)
  "educ_x_formal", "educ_x_ocup", "menores_x_desocup", "depend_x_informal",
  "rural_x_cta_propia", "educ_x_rural", "subsidiado_x_menores", "jefe_mayor_pension",
  # Bogotá: constante 0 tras filtrar train (varianza nula)
  "bogota"
)

# Exclusiones adicionales para SPEC A: se eliminan los índices compuestos
# y se conservan sus componentes granulares.
VARS_EXCLUIR_A_EXTRA <- c(
  # Índice demográfico (Bloque 2)
  "doble_ingreso",           # = 1 - sin_ocupados - un_solo_ocupado
  # Índice educativo (Bloque 3)
  "brecha_educ",             # = nivel_educ_max - nivel_educ_jefe
  # Índices de calidad del empleo (Bloque 4)
  "indice_formalidad",       # = mean(prop_asalariado, cotiza, prima, sub_transp)
  "formal_estricto",         # = prop_asalariado * prop_cotiza_pension
  "vulnerabilidad_lab",      # = prop_domestico + prop_fam_sin_rem + prop_jornalero
  # Índices de diversificación de ingresos (Bloque 5)
  "n_fuentes_no_lab",        # = suma de los 8 alguno_*
  "tiene_ingreso_no_lab",    # = flag binario derivado de los 8 alguno_*
  "solo_subsidio",           # = combinación de alguno_subsidio y otros alguno_*
  "ingresos_activos",        # = alguno_intereses | alguno_arriendos
  # Índices de protección social (Bloque 6)
  "proteccion_social",       # = alguno_pension_jub | prop_cotiza_pension > 0
  "desprotegido",            # = prop_cotiza_pension==0 & alguno_pension_jub==0 & ...
  "prop_contributivo",       # = prop_afil_salud - prop_subsidiado
  # Dummies regionales compuestas (Bloque 7)
  # Cada una agrega varios dptos → son índices geográficos
  "costa_caribe",            # dptos 08, 13, 20, 23, 44, 47, 70
  "region_pacifico",         # dptos 19, 27, 52
  "eje_cafetero",            # dptos 17, 63, 66, 76
  # Índice multidimensional de privaciones (Bloque 10)
  "n_privaciones",           # = dep_educacion + dep_vivienda + ... + dep_dependencia
  "multi_privado",           # = n_privaciones >= 2  (Alkire-Foster k=2)
  "privado_severo"           # = n_privaciones >= 3
)

# Exclusiones adicionales para SPEC B: se eliminan los componentes granulares
# y se conservan los índices compuestos.
VARS_EXCLUIR_B_EXTRA <- c(
  # Componentes de doble_ingreso
  "sin_ocupados",
  "un_solo_ocupado",
  # Componentes de brecha_educ
  # (nivel_educ_prom NO es componente de ningún índice → permanece en Spec B)
  "nivel_educ_max",
  "nivel_educ_jefe",
  # Componentes de indice_formalidad
  "prop_asalariado",
  "prop_cotiza_pension",
  "prop_prima_serv",
  "prop_sub_transp",
  # Componentes de vulnerabilidad_lab
  "prop_domestico",
  "prop_fam_sin_rem",
  "prop_jornalero",
  # Componentes de n_fuentes_no_lab / tiene_ingreso_no_lab / solo_subsidio / ingresos_activos
  "alguno_pension_jub",
  "alguno_remesas",
  "alguno_subsidio",
  "alguno_transf_nac",
  "alguno_intereses",
  "alguno_cesantias",
  "alguno_agropec",
  "alguno_arriendos",
  # Componentes de prop_contributivo
  "prop_afil_salud",
  "prop_subsidiado",
  # Componentes de n_privaciones / multi_privado / privado_severo
  "dep_educacion",
  "dep_vivienda",
  "dep_empleo_formal",
  "dep_proteccion",
  "dep_dependencia"
)

message(strrep("=", 65))
message("  ", MODEL_ID, "  —  ", AUTOR)
message("  Dos especificaciones: A (componentes) vs B (índices)")
message("  Base XGB_007: sin vars construidas, sin Bogotá en train")
message("  Cores (nthread)    : ", N_THREADS)
message("  N_SEARCH / K_FOLDS : ", N_SEARCH, " / ", K_FOLDS)
message("  MAX_NROUNDS / EARLY: ", MAX_NROUNDS, " / ", EARLY_STOP)
message("  Excluidas base     : ", length(VARS_EXCLUIR_BASE), " vars")
message("  Excluidas extra A  : ", length(VARS_EXCLUIR_A_EXTRA), " vars (índices)")
message("  Excluidas extra B  : ", length(VARS_EXCLUIR_B_EXTRA), " vars (componentes)")
message(strrep("=", 65))

# =============================================================================
# SECCIÓN 3: CARGAR DATOS Y FILTRAR BOGOTÁ
# =============================================================================
message("\n== Cargando datos y filtrando Bogotá ==")

train_raw <- readRDS("00_data/processed/train_final.rds")
test      <- readRDS("00_data/processed/test_final.rds")
ids       <- test$id

n_bogota <- sum(as.character(train_raw$dpto) == DPTO_BOGOTA, na.rm = TRUE)
train    <- train_raw |> filter(as.character(dpto) != DPTO_BOGOTA)

message("  Train original: ", nrow(train_raw), " | Bogotá eliminada: ", n_bogota,
        " obs (", round(100 * n_bogota / nrow(train_raw), 1), "%)")
message("  Train filtrado: ", nrow(train), " obs")
message("  Test           : ", nrow(test),  " obs")

y_factor <- factor(ifelse(train$pobre == 1, "Yes", "No"), levels = niveles)
y_int    <- as.integer(y_factor == "Yes")
n_pos    <- sum(y_int)
n_neg    <- length(y_int) - n_pos
message("  Balance: ", n_pos, " pobres (", round(100 * mean(y_int), 1), "%) | ",
        n_neg, " no-pobres")

# =============================================================================
# SECCIÓN 4: PREPROCESAR Y CONSTRUIR MATRICES POR ESPECIFICACIÓN
# =============================================================================
message("\n== Construyendo matrices de features ==")

preparar_X_xgb <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>
    mutate(zona   = as.numeric(zona),
           ciudad = as.integer(factor(ciudad)),
           dpto   = as.integer(factor(dpto))) |>
    mutate(across(where(is.character), as.numeric)) |>
    mutate(across(where(is.factor),    as.integer))
}

X_train_raw <- preparar_X_xgb(train)
X_test_raw  <- preparar_X_xgb(test)

# Spec A: excluye base + índices compuestos → conserva componentes granulares
vars_exc_A   <- unique(c(VARS_EXCLUIR_BASE, VARS_EXCLUIR_A_EXTRA))
X_train_A    <- X_train_raw |> select(-any_of(vars_exc_A))
X_test_A     <- X_test_raw  |> select(-any_of(vars_exc_A))
cols_com_A   <- intersect(names(X_train_A), names(X_test_A))
X_train_A    <- X_train_A[, cols_com_A]
X_test_A     <- X_test_A[,  cols_com_A]
mat_train_A  <- as.matrix(X_train_A)
mat_test_A   <- as.matrix(X_test_A)
P_A          <- ncol(mat_train_A)

# Spec B: excluye base + componentes granulares → conserva índices compuestos
vars_exc_B   <- unique(c(VARS_EXCLUIR_BASE, VARS_EXCLUIR_B_EXTRA))
X_train_B    <- X_train_raw |> select(-any_of(vars_exc_B))
X_test_B     <- X_test_raw  |> select(-any_of(vars_exc_B))
cols_com_B   <- intersect(names(X_train_B), names(X_test_B))
X_train_B    <- X_train_B[, cols_com_B]
X_test_B     <- X_test_B[,  cols_com_B]
mat_train_B  <- as.matrix(X_train_B)
mat_test_B   <- as.matrix(X_test_B)
P_B          <- ncol(mat_train_B)

message("  Spec A (componentes, sin índices) : ", P_A, " predictores")
message("  Spec B (índices, sin componentes) : ", P_B, " predictores")
message("  NAs A train: ", sum(is.na(mat_train_A)),
        " | NAs B train: ", sum(is.na(mat_train_B)))

# Verificar que los conjuntos son disjuntos en las variables clave
vars_only_A <- setdiff(cols_com_A, cols_com_B)   # presentes en A pero no en B
vars_only_B <- setdiff(cols_com_B, cols_com_A)   # presentes en B pero no en A
vars_common  <- intersect(cols_com_A, cols_com_B) # presentes en ambas
message("  Variables solo en A (componentes) : ", length(vars_only_A))
message("  Variables solo en B (índices)     : ", length(vars_only_B))
message("  Variables compartidas (base)      : ", length(vars_common))

# =============================================================================
# SECCIÓN 5: FOLDS CV (compartidos por ambas specs — misma semilla)
# =============================================================================
# Los mismos folds garantizan que las diferencias en F1 OOF entre A y B se
# deban a las features, no a la partición de datos.
set.seed(42)
folds <- caret::createFolds(y_factor, k = K_FOLDS, list = TRUE, returnTrain = TRUE)
message("\n  Folds CV-5 generados (semilla 42, compartidos por Spec A y B)")

# =============================================================================
# SECCIÓN 6: FUNCIONES AUXILIARES
# =============================================================================

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

entrenar_fold_xgb <- function(X_tr_mat, y_int_tr, X_val_mat, y_int_val,
                               params, max_nr, early_s, nthread,
                               tech = "baseline", seed_val = 42) {
  X_use <- X_tr_mat
  y_use <- y_int_tr
  p_use <- params

  if (tech == "class_weights") {
    p_use <- c(params, list(scale_pos_weight = sum(y_int_tr == 0) / sum(y_int_tr == 1)))

  } else if (tech == "downsample") {
    set.seed(seed_val)
    y_f   <- factor(ifelse(y_int_tr == 1, "Yes", "No"), levels = niveles)
    ds    <- caret::downSample(x = as_tibble(X_tr_mat), y = y_f, yname = "pobre")
    X_use <- as.matrix(ds |> select(-pobre))
    y_use <- as.integer(ds$pobre == "Yes")

  } else if (tech == "upsample") {
    set.seed(seed_val)
    y_f   <- factor(ifelse(y_int_tr == 1, "Yes", "No"), levels = niveles)
    us    <- caret::upSample(x = as_tibble(X_tr_mat), y = y_f, yname = "pobre")
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

  dtrain <- xgboost::xgb.DMatrix(X_use,     label = y_use)
  dval   <- xgboost::xgb.DMatrix(X_val_mat, label = y_int_val)

  mod <- xgboost::xgb.train(
    params                = c(p_use, list(nthread = nthread)),
    data                  = dtrain,
    nrounds               = max_nr,
    watchlist             = list(val = dval),
    early_stopping_rounds = early_s,
    verbose               = 0
  )

  bir <- xgboost::xgb.attr(mod, "best_iteration")
  list(probs_val    = predict(mod, dval),
       best_nrounds = if (!is.null(bir)) as.integer(bir) else max_nr)
}

# =============================================================================
# SECCIÓN 7: FUNCIÓN PRINCIPAL — FASE 1 + FASE 2 POR ESPECIFICACIÓN
# =============================================================================
# Encapsula Phase 1 (HP search) + Phase 2 (balancing) para una spec dada.
# Retorna una lista con todo lo necesario para Phase 3 y la comparación final.

correr_spec <- function(spec_id, mat_train, mat_test,
                         dir_spec, p_total) {

  cat(strrep("=", 65), "\n")
  message("  ESPECIFICACIÓN ", spec_id,
          " (", if (spec_id == "A") "componentes, sin índices"
                else "índices, sin componentes",
          " | ", p_total, " features)")
  cat(strrep("=", 65), "\n")
  n_train <- nrow(mat_train)

  # ─── FASE 1: HP SEARCH ───────────────────────────────────────────────────
  hp_checkpoint <- file.path(dir_spec, "hp_search_results.csv")

  if (file.exists(hp_checkpoint)) {
    message("  [Spec ", spec_id, "] FASE 1: cargando checkpoint")
    hp_results <- read_csv(hp_checkpoint, show_col_types = FALSE)

  } else {
    message("  [Spec ", spec_id, "] FASE 1 — BÚSQUEDA ALEATORIA DE HP")
    message("  N_SEARCH=", N_SEARCH, " | features=", p_total,
            " | train=", n_train, " obs (sin Bogotá)")

    # Semilla fija por spec para reproducibilidad; ambas reciben el mismo
    # conjunto de combinaciones HP — la diferencia en F1 se atribuye
    # exclusivamente al conjunto de features, no al grid muestreado.
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
        "\n  [Spec %s | HP %2d/%d] d=%d eta=%.3f gam=%.1f mcw=%2d ct=%.2f cl=%.2f sub=%.2f L2=%.1f L1=%.3f",
        spec_id, i, N_SEARCH,
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

      oof_i      <- numeric(n_train)
      nr_folds_i <- integer(K_FOLDS)

      ok <- tryCatch({
        for (f in seq_along(folds)) {
          tr_idx  <- folds[[f]]
          val_idx <- setdiff(seq_len(n_train), tr_idx)
          dtrain_f <- xgboost::xgb.DMatrix(mat_train[tr_idx,  ],
                                            label = y_int[tr_idx])
          dval_f   <- xgboost::xgb.DMatrix(mat_train[val_idx, ],
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
    message(sprintf("\n  [Spec %s] FASE 1 completada en %.1f min.",
                    spec_id, as.numeric(difftime(t1_hp, t0_hp, units = "mins"))))

    hp_results <- bind_rows(hp_results_list) |> arrange(desc(F1_OOF))
    write_csv(hp_results, hp_checkpoint)
    message("  Guardado: ", hp_checkpoint)
  }

  cat(sprintf("\n-- [Spec %s] Top-5 HP (F1 OOF) --\n", spec_id))
  print(hp_results |>
          select(max_depth:alpha, best_nrounds, nrounds_sd, F1_OOF, AUC_ROC, Threshold) |>
          head(5))

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
  message(sprintf("  [Spec %s] Mejor HP: depth=%d eta=%.3f gam=%.1f mcw=%d nrounds=%d F1=%.4f",
                  spec_id, best_hp$max_depth, best_hp$eta, best_hp$gamma,
                  best_hp$min_child_weight, best_hp$best_nrounds, best_hp$F1_OOF))

  # ─── FASE 2: TÉCNICAS DE BALANCEO ────────────────────────────────────────
  message(sprintf("\n  [Spec %s] FASE 2 — TÉCNICAS DE BALANCEO", spec_id))

  bal_oof_list <- setNames(vector("list", length(TECHNIQUES)), TECHNIQUES)
  bal_nr_list  <- setNames(vector("list", length(TECHNIQUES)), TECHNIQUES)
  bal_fold_tbl <- vector("list", length(TECHNIQUES))
  t0_bal <- Sys.time()

  for (tech in TECHNIQUES) {
    message("  -- Spec ", spec_id, " | Técnica: ", tech, " --")
    oof_tech <- numeric(n_train)
    nr_tech  <- integer(K_FOLDS)

    for (f in seq_along(folds)) {
      tr_idx  <- folds[[f]]
      val_idx <- setdiff(seq_len(n_train), tr_idx)
      seed_ft <- 42 * f + match(tech, TECHNIQUES) * 1000

      res_f <- tryCatch(
        entrenar_fold_xgb(
          X_tr_mat  = mat_train[tr_idx,  ],
          y_int_tr  = y_int[tr_idx],
          X_val_mat = mat_train[val_idx, ],
          y_int_val = y_int[val_idx],
          params    = best_params,
          max_nr    = MAX_NROUNDS,
          early_s   = EARLY_STOP,
          nthread   = N_THREADS,
          tech      = tech,
          seed_val  = seed_ft
        ),
        error = function(e) { message("    ERROR fold ", f, ": ", conditionMessage(e)); NULL }
      )
      if (is.null(res_f)) next
      oof_tech[val_idx] <- res_f$probs_val
      nr_tech[f]        <- res_f$best_nrounds
      message(sprintf("    Fold %d: nrounds=%3d", f, res_f$best_nrounds))
    }

    met_tech <- calcular_metricas_oof(oof_tech, y_factor, y_int)
    nr_med   <- as.integer(ceiling(median(nr_tech)))
    message(sprintf("    OOF → F1=%.4f  Prec=%.4f  Rec=%.4f  AUC=%.4f  th=%.2f  nrounds=%d",
                    met_tech$F1, met_tech$Precision, met_tech$Recall,
                    met_tech$AUC_ROC, met_tech$threshold, nr_med))

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
  message(sprintf("  [Spec %s] FASE 2 completada en %.1f min.",
                  spec_id, as.numeric(difftime(t1_bal, t0_bal, units = "mins"))))

  bal_summary <- bind_rows(bal_fold_tbl) |> arrange(desc(F1))
  write_csv(bal_summary, file.path(dir_spec, "cv_balance_summary.csv"))
  message("  Guardado: cv_balance_summary.csv")

  cat(sprintf("\n-- [Spec %s] Resultados por técnica --\n", spec_id))
  print(bal_summary |> mutate(across(where(is.double), ~ round(., 4))))

  best_technique <- bal_summary |> slice_max(F1, n = 1, with_ties = FALSE) |> pull(technique)
  best_bal_row   <- bal_summary |> filter(technique == best_technique)
  oof_best       <- bal_oof_list[[best_technique]]
  met_final      <- calcular_metricas_oof(oof_best, y_factor, y_int)

  message(sprintf("  [Spec %s] Ganador balanceo: %s | F1_OOF=%.4f | th=%.2f",
                  spec_id, best_technique, best_bal_row$F1, best_bal_row$threshold))

  list(
    spec_id        = spec_id,
    hp_results     = hp_results,
    best_hp        = best_hp,
    best_params    = best_params,
    bal_summary    = bal_summary,
    bal_oof_list   = bal_oof_list,
    bal_nr_list    = bal_nr_list,
    best_technique = best_technique,
    best_bal_row   = best_bal_row,
    oof_best       = oof_best,
    met_final      = met_final,
    p_total        = p_total
  )
}

# =============================================================================
# SECCIÓN 8: EJECUTAR ESPECIFICACIÓN A
# =============================================================================
message("\n", strrep("#", 65))
message("  EJECUTANDO ESPECIFICACIÓN A: componentes granulares, sin índices")
message(strrep("#", 65))

res_A <- correr_spec(
  spec_id   = "A",
  mat_train = mat_train_A,
  mat_test  = mat_test_A,
  dir_spec  = dir_spec_A,
  p_total   = P_A
)

# =============================================================================
# SECCIÓN 9: EJECUTAR ESPECIFICACIÓN B
# =============================================================================
message("\n", strrep("#", 65))
message("  EJECUTANDO ESPECIFICACIÓN B: índices compuestos, sin componentes")
message(strrep("#", 65))

res_B <- correr_spec(
  spec_id   = "B",
  mat_train = mat_train_B,
  mat_test  = mat_test_B,
  dir_spec  = dir_spec_B,
  p_total   = P_B
)

# =============================================================================
# SECCIÓN 10: COMPARACIÓN Y SELECCIÓN DEL GANADOR
# =============================================================================
message("\n", strrep("=", 65))
message("  COMPARACIÓN FINAL: SPEC A vs SPEC B")
message(strrep("=", 65))

# NOTA: Los F1 OOF de A y B se calculan sobre el mismo training set filtrado
# (sin Bogotá) y con los mismos folds CV → comparación directa es válida.
comparison <- tibble(
  spec             = c("A", "B"),
  descripcion      = c("componentes (sin índices)", "índices (sin componentes)"),
  n_features       = c(P_A, P_B),
  best_hp_depth    = c(res_A$best_hp$max_depth,    res_B$best_hp$max_depth),
  best_hp_eta      = c(res_A$best_hp$eta,           res_B$best_hp$eta),
  best_hp_gamma    = c(res_A$best_hp$gamma,         res_B$best_hp$gamma),
  best_technique   = c(res_A$best_technique,        res_B$best_technique),
  best_nrounds     = c(res_A$bal_nr_list[[res_A$best_technique]],
                       res_B$bal_nr_list[[res_B$best_technique]]),
  F1_OOF           = c(round(res_A$met_final$F1,    4),
                       round(res_B$met_final$F1,    4)),
  Precision_OOF    = c(round(res_A$met_final$Precision, 4),
                       round(res_B$met_final$Precision, 4)),
  Recall_OOF       = c(round(res_A$met_final$Recall,    4),
                       round(res_B$met_final$Recall,    4)),
  AUC_ROC_OOF      = c(round(res_A$met_final$AUC_ROC,   4),
                       round(res_B$met_final$AUC_ROC,   4)),
  threshold        = c(res_A$met_final$threshold,   res_B$met_final$threshold)
)

write_csv(comparison, file.path(dir_model, "comparison.csv"))
cat("\n-- Tabla comparativa Spec A vs B --\n")
print(as.data.frame(comparison))

# Seleccionar el ganador por F1 OOF
WINNER_SPEC <- comparison |> slice_max(F1_OOF, n = 1, with_ties = FALSE) |> pull(spec)
res_winner  <- if (WINNER_SPEC == "A") res_A else res_B
mat_train_W <- if (WINNER_SPEC == "A") mat_train_A else mat_train_B
mat_test_W  <- if (WINNER_SPEC == "A") mat_test_A  else mat_test_B
P_W         <- if (WINNER_SPEC == "A") P_A else P_B

message("\n  *** GANADOR: Especificación ", WINNER_SPEC,
        " (F1_OOF = ", round(res_winner$met_final$F1, 4), ") ***")
message("  Delta F1 A vs B: ",
        round(res_A$met_final$F1 - res_B$met_final$F1, 4),
        "  (positivo → A mejor; negativo → B mejor)")

# =============================================================================
# SECCIÓN 11: FASE 3 — MODELO FINAL (AMBAS SPECS)
# =============================================================================
# Se entrena el modelo final para AMBAS specs: permite generar ambas
# submissions y verificar el resultado en Kaggle.

message("\n", strrep("=", 65))
message("  FASE 3 — MODELOS FINALES (A y B)")
message(strrep("=", 65))

entrenar_final <- function(res_spec, mat_train, n_pos_arg, n_neg_arg) {
  tech           <- res_spec$best_technique
  best_params    <- res_spec$best_params
  best_nrounds   <- res_spec$bal_nr_list[[tech]]
  best_bal_row   <- res_spec$best_bal_row
  params_final   <- best_params
  X_final_mat    <- mat_train
  y_final_int    <- y_int

  if (tech == "class_weights") {
    params_final <- c(best_params, list(scale_pos_weight = n_neg_arg / n_pos_arg))
    message("  scale_pos_weight = ", round(n_neg_arg / n_pos_arg, 3))

  } else if (tech == "downsample") {
    set.seed(42)
    y_f      <- factor(ifelse(y_int == 1, "Yes", "No"), levels = niveles)
    ds_f     <- caret::downSample(x = as_tibble(mat_train), y = y_f, yname = "pobre")
    X_final_mat  <- as.matrix(ds_f |> select(-pobre))
    y_final_int  <- as.integer(ds_f$pobre == "Yes")
    message("  Down-sampled: ", nrow(X_final_mat), " obs")

  } else if (tech == "upsample") {
    set.seed(42)
    y_f      <- factor(ifelse(y_int == 1, "Yes", "No"), levels = niveles)
    us_f     <- caret::upSample(x = as_tibble(mat_train), y = y_f, yname = "pobre")
    X_final_mat  <- as.matrix(us_f |> select(-pobre))
    y_final_int  <- as.integer(us_f$pobre == "Yes")
    message("  Up-sampled: ", nrow(X_final_mat), " obs")

  } else if (tech == "smote") {
    set.seed(42)
    y_f      <- factor(ifelse(y_int == 1, "Yes", "No"), levels = niveles)
    df_sm    <- bind_cols(as_tibble(mat_train), pobre = y_f)
    df_out   <- themis::smote(df_sm, var = "pobre", k = 5, over_ratio = 1)
    X_final_mat  <- as.matrix(df_out |> select(-pobre))
    y_final_int  <- as.integer(df_out$pobre == "Yes")
    message("  SMOTE: ", nrow(X_final_mat), " obs")
  }

  set.seed(42)
  dtrain_f <- xgboost::xgb.DMatrix(X_final_mat, label = y_final_int)
  xgb_f <- xgboost::xgb.train(
    params  = c(params_final, list(nthread = N_THREADS)),
    data    = dtrain_f,
    nrounds = best_nrounds,
    verbose = 0
  )
  list(model = xgb_f, nrounds = best_nrounds, threshold = best_bal_row$threshold)
}

message("\nEntrenando modelo final Spec A...")
fin_A <- entrenar_final(res_A, mat_train_A, n_pos, n_neg)
message("  Spec A: nrounds=", fin_A$nrounds, " | th=", fin_A$threshold)

message("\nEntrenando modelo final Spec B...")
fin_B <- entrenar_final(res_B, mat_train_B, n_pos, n_neg)
message("  Spec B: nrounds=", fin_B$nrounds, " | th=", fin_B$threshold)

# =============================================================================
# SECCIÓN 12: GRÁFICOS DE DIAGNÓSTICO (GANADOR)
# =============================================================================
message("\n== Generando gráficos del ganador (Spec ", WINNER_SPEC, ") ==")

res_W    <- res_winner
fin_W    <- if (WINNER_SPEC == "A") fin_A else fin_B
oof_best <- res_W$oof_best
met_W    <- res_W$met_final
best_th  <- fin_W$threshold
auc_roc  <- met_W$AUC_ROC
roc_obj  <- met_W$roc_obj

# Importancia de variables del ganador
varimp_W <- xgboost::xgb.importance(
  feature_names = if (WINNER_SPEC == "A") colnames(mat_train_A) else colnames(mat_train_B),
  model         = fin_W$model
)
write_csv(as_tibble(varimp_W), file.path(dir_model, "feature_importance.csv"))

# Gráfico 1: Comparación F1 A vs B (resumen de todas las técnicas)
comp_bal_both <- bind_rows(
  res_A$bal_summary |> mutate(spec = "A: componentes"),
  res_B$bal_summary |> mutate(spec = "B: índices")
)

p_comp_specs <- comp_bal_both |>
  ggplot(aes(x = reorder(paste(spec, technique), F1), y = F1,
             fill = spec)) +
  geom_col(alpha = 0.85, width = 0.7) +
  geom_text(aes(label = sprintf("%.4f", F1)), hjust = -0.1, size = 3) +
  coord_flip() +
  scale_fill_manual(values = c("A: componentes" = "#534AB7",
                                "B: índices"     = "#E84855")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title    = paste0("XGBoost: F1 OOF — Spec A (componentes) vs Spec B (índices) — ", MODEL_ID),
    subtitle = sprintf("Ganador: Spec %s (%s) | F1=%.4f | th=%.2f",
                       WINNER_SPEC, res_W$best_technique,
                       round(res_W$met_final$F1, 4), best_th),
    x        = "Técnica × Especificación",
    y        = "F1 (OOF global)",
    fill     = "Especificación",
    caption  = "Mismos folds CV-5 y misma semilla para ambas specs. Train sin Bogotá."
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top")

ggsave(file.path(dir_model, "comparison_specs.png"),
       p_comp_specs, width = 9, height = 7, dpi = 150)
message("  Guardado: comparison_specs.png")

# Gráfico 2: Importancia variables ganador (Top 20)
p_varimp <- varimp_W |> head(20) |>
  ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#534AB7", alpha = 0.85) +
  coord_flip() +
  labs(
    title    = paste0("XGBoost: importancia variables Top 20 (Gain) — ", MODEL_ID),
    subtitle = paste0("Ganador: Spec ", WINNER_SPEC, " | ", res_W$best_technique,
                      " | depth=", res_W$best_hp$max_depth,
                      " | eta=", res_W$best_hp$eta,
                      " | nrounds=", fin_W$nrounds,
                      " | features=", P_W),
    x = NULL, y = "Gain"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "varimp.png"),
       p_varimp, width = 8, height = 6, dpi = 150)
message("  Guardado: varimp.png")

# Gráfico 3: Threshold vs métricas (ganador)
th_full <- map_dfr(seq(0.05, 0.95, by = 0.01), function(th) {
  pred <- factor(ifelse(oof_best >= th, "Yes", "No"), levels = niveles)
  TP   <- sum(pred == "Yes" & y_factor == "Yes")
  FP   <- sum(pred == "Yes" & y_factor == "No")
  FN   <- sum(pred == "No"  & y_factor == "Yes")
  f1   <- if ((2*TP+FP+FN)==0) NA_real_ else 2*TP/(2*TP+FP+FN)
  tibble(threshold = th, F1 = f1,
         Precision = if ((TP+FP)==0) NA_real_ else TP/(TP+FP),
         Recall    = if ((TP+FN)==0) NA_real_ else TP/(TP+FN))
})

p_threshold <- th_full |>
  filter(!is.na(F1)) |>
  pivot_longer(c(F1, Precision, Recall), names_to = "metrica") |>
  ggplot(aes(x = threshold, y = value, color = metrica)) +
  geom_line(linewidth = 0.9) +
  geom_vline(xintercept = best_th, linetype = "dashed", color = "#534AB7") +
  geom_vline(xintercept = 0.5,    linetype = "dotted",  color = "gray40") +
  annotate("text", x = best_th + 0.02, y = 0.08,
           label = sprintf("th=%.2f", best_th), hjust = 0, size = 3.2, color = "#534AB7") +
  scale_color_manual(values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3")) +
  labs(title    = paste0("XGBoost: métricas por threshold — ", MODEL_ID, " (Spec ", WINNER_SPEC, ")"),
       subtitle = sprintf("th=%.2f | %s | depth=%d | eta=%.3f | sin Bogotá",
                          best_th, res_W$best_technique,
                          res_W$best_hp$max_depth, res_W$best_hp$eta),
       x = "Threshold", y = "Valor", color = NULL) +
  theme_minimal(base_size = 12) + theme(legend.position = "top")

ggsave(file.path(dir_model, "threshold.png"),
       p_threshold, width = 8, height = 5, dpi = 150)
message("  Guardado: threshold.png")

# Gráfico 4: ROC
roc_df <- tibble(fpr = 1 - roc_obj$specificities, tpr = roc_obj$sensitivities)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.55, y = 0.15,
           label = sprintf("AUC-ROC = %.4f\n(Spec %s — %s)",
                           auc_roc, WINNER_SPEC,
                           if (WINNER_SPEC == "A") "componentes" else "índices"),
           size = 3.5, color = "#534AB7") +
  scale_x_continuous(labels = function(x) paste0(round(x*100), "%")) +
  scale_y_continuous(labels = function(x) paste0(round(x*100), "%")) +
  labs(title    = paste0("XGBoost: curva ROC — ", MODEL_ID),
       subtitle = sprintf("Spec %s | %s | depth=%d | sin Bogotá",
                          WINNER_SPEC, res_W$best_technique, res_W$best_hp$max_depth),
       x = "Tasa de Falsos Positivos", y = "Tasa de Verdaderos Positivos",
       caption = "OOF (train sin Bogotá) — estimación honesta.") +
  coord_equal() + theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc.png"), p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc.png")

# Gráfico 5: Precision-Recall
n_pos_oof   <- sum(y_int)
n_neg_oof   <- length(y_int) - n_pos_oof
pr_curve_df <- tibble(
  TP        = roc_obj$sensitivities * n_pos_oof,
  FP        = (1 - roc_obj$specificities) * n_neg_oof,
  recall    = roc_obj$sensitivities,
  precision = TP / (TP + FP)
) |> filter(is.finite(precision), is.finite(recall))

prev_rate <- n_pos_oof / (n_pos_oof + n_neg_oof)

p_prcurve <- ggplot(pr_curve_df, aes(x = recall, y = precision)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_hline(yintercept = prev_rate, linetype = "dashed", color = "gray60") +
  geom_point(data = tibble(recall = met_W$Recall, precision = met_W$Precision),
             color = "#E84855", size = 3) +
  annotate("text", x = met_W$Recall - 0.1, y = met_W$Precision + 0.015,
           label = sprintf("th=%.2f\nF1=%.4f", best_th, met_W$F1),
           size = 3, color = "#E84855") +
  labs(title    = paste0("XGBoost: curva Precision-Recall — ", MODEL_ID),
       subtitle = sprintf("Spec %s | AUC-ROC: %.4f | %s | sin Bogotá",
                          WINNER_SPEC, auc_roc, res_W$best_technique),
       x = "Recall", y = "Precision",
       caption = "OOF sin Bogotá. Punto rojo = threshold óptimo.") +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve.png"), p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve.png")

# Diagnósticos completos
saveRDS(
  list(
    model_id_benchmark = "XGB_007",
    winner_spec        = WINNER_SPEC,
    comparison         = comparison,
    # Spec A
    res_A              = res_A,
    fin_A_model        = fin_A$model,
    fin_A_nrounds      = fin_A$nrounds,
    fin_A_threshold    = fin_A$threshold,
    # Spec B
    res_B              = res_B,
    fin_B_model        = fin_B$model,
    fin_B_nrounds      = fin_B$nrounds,
    fin_B_threshold    = fin_B$threshold,
    # Winner
    varimp_winner      = varimp_W,
    roc_obj            = roc_obj,
    auc_roc            = auc_roc,
    oof_best           = oof_best,
    # Config
    vars_excluir_base  = VARS_EXCLUIR_BASE,
    vars_excluir_A_extra = VARS_EXCLUIR_A_EXTRA,
    vars_excluir_B_extra = VARS_EXCLUIR_B_EXTRA,
    n_bogota_excluidos = n_bogota,
    p_A = P_A, p_B = P_B
  ),
  file.path(dir_model, "diagnostics.rds")
)
message("  Guardado: diagnostics.rds")

# =============================================================================
# SECCIÓN 13: SUBMISSIONS PARA KAGGLE
# =============================================================================
message("\n== Generando submissions ==")

# Submission Spec A
probs_A  <- predict(fin_A$model, xgboost::xgb.DMatrix(mat_test_A))
pred_A   <- as.integer(probs_A >= fin_A$threshold)
sub_A    <- tibble(id = ids, pobre = pred_A)
name_A   <- sprintf("XGB_008A_depth%d_gam%.1f_L2%.0f_eta%.3f_nr%d_%s_th%03.0f.csv",
                    res_A$best_hp$max_depth, res_A$best_hp$gamma, res_A$best_hp$lambda,
                    res_A$best_hp$eta, fin_A$nrounds, res_A$best_technique,
                    fin_A$threshold * 100)
path_A   <- file.path(dir_subs, name_A)
write_csv(sub_A, path_A)
message("  Spec A: ", path_A,
        "  | pobres=", sum(pred_A), " (", round(mean(pred_A)*100,1), "%)")

# Submission Spec B
probs_B  <- predict(fin_B$model, xgboost::xgb.DMatrix(mat_test_B))
pred_B   <- as.integer(probs_B >= fin_B$threshold)
sub_B    <- tibble(id = ids, pobre = pred_B)
name_B   <- sprintf("XGB_008B_depth%d_gam%.1f_L2%.0f_eta%.3f_nr%d_%s_th%03.0f.csv",
                    res_B$best_hp$max_depth, res_B$best_hp$gamma, res_B$best_hp$lambda,
                    res_B$best_hp$eta, fin_B$nrounds, res_B$best_technique,
                    fin_B$threshold * 100)
path_B   <- file.path(dir_subs, name_B)
write_csv(sub_B, path_B)
message("  Spec B: ", path_B,
        "  | pobres=", sum(pred_B), " (", round(mean(pred_B)*100,1), "%)")

# Submission del ganador (copia renombrada como XGB_008)
sub_winner <- if (WINNER_SPEC == "A") sub_A else sub_B
res_W_hp   <- res_winner$best_hp
fin_W_nr   <- fin_W$nrounds
name_W     <- sprintf("XGB_008_spec%s_depth%d_gam%.1f_L2%.0f_eta%.3f_nr%d_%s_th%03.0f.csv",
                      WINNER_SPEC,
                      res_W_hp$max_depth, res_W_hp$gamma, res_W_hp$lambda,
                      res_W_hp$eta, fin_W_nr, res_winner$best_technique,
                      fin_W$threshold * 100)
path_W     <- file.path(dir_subs, name_W)
write_csv(sub_winner, path_W)
message("  Ganador (XGB_008): ", path_W)

# =============================================================================
# SECCIÓN 14: REGISTRO EN model_registry.csv
# =============================================================================
message("\n== Registro en model_registry.csv ==")

nueva_fila <- tibble(
  model_id           = MODEL_ID,
  fecha              = Sys.Date(),
  autor              = AUTOR,
  algoritmo          = "XGBoost",
  n_features         = P_W,
  imbalance_strategy = res_winner$best_technique,
  cv_folds           = K_FOLDS,
  cv_F1              = round(met_W$F1,        4),
  cv_Precision       = round(met_W$Precision, 4),
  cv_Recall          = round(met_W$Recall,    4),
  auc_roc            = round(auc_roc,         4),
  kaggle_public_F1   = NA_real_,
  threshold          = fin_W$threshold,
  notas              = paste0(
    "XGBoost índices vs componentes. Base XGB_007 (sin vars construidas, sin Bogotá). ",
    "SpecA: ", P_A, " features (componentes, excluye ", length(VARS_EXCLUIR_A_EXTRA), " índices). ",
    "SpecB: ", P_B, " features (índices, excluye ", length(VARS_EXCLUIR_B_EXTRA), " componentes). ",
    "Ganador: Spec ", WINNER_SPEC, " (",
    if (WINNER_SPEC == "A") "componentes" else "índices", "). ",
    "F1_OOF_A=", round(res_A$met_final$F1, 4),
    " vs F1_OOF_B=", round(res_B$met_final$F1, 4),
    " (delta=", round(res_A$met_final$F1 - res_B$met_final$F1, 4), "). ",
    "Mejor HP ganador: depth=", res_W_hp$max_depth,
    " eta=", res_W_hp$eta, " gam=", res_W_hp$gamma,
    " mcw=", res_W_hp$min_child_weight,
    " col_t=", res_W_hp$colsample_bytree,
    " col_l=", res_W_hp$colsample_bylevel,
    " sub=", res_W_hp$subsample,
    " L2=", res_W_hp$lambda, " L1=", res_W_hp$alpha, ". ",
    "Técnica: ", res_winner$best_technique,
    " nrounds=", fin_W_nr,
    " th=", fin_W$threshold,
    " AUC=", round(auc_roc, 4), ". ",
    "Train sin Bogotá: ", nrow(train), " obs."
  ),
  cp              = NA_real_,
  maxdepth        = res_W_hp$max_depth,
  train_F1        = round(met_W$F1,        4),
  train_Precision = round(met_W$Precision, 4),
  train_Recall    = round(met_W$Recall,    4)
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

# =============================================================================
# RESUMEN FINAL
# =============================================================================
cat("\n", strrep("=", 65), "\n")
cat("  ", MODEL_ID, " — RESUMEN FINAL\n")
cat(strrep("=", 65), "\n")
cat(sprintf("  %-28s : Spec A=%d vars | Spec B=%d vars\n",
            "N° features", P_A, P_B))
cat(sprintf("  %-28s : Spec A=%.4f | Spec B=%.4f\n",
            "F1 OOF", res_A$met_final$F1, res_B$met_final$F1))
cat(sprintf("  %-28s : Spec A=%.4f | Spec B=%.4f\n",
            "AUC-ROC OOF",
            res_A$met_final$AUC_ROC, res_B$met_final$AUC_ROC))
cat(sprintf("  %-28s : %s (%s)\n",
            "GANADOR", WINNER_SPEC,
            if (WINNER_SPEC == "A") "componentes granulares, sin índices"
            else "índices compuestos, sin componentes"))
cat(sprintf("  %-28s : %.4f\n", "F1 OOF ganador", met_W$F1))
cat(sprintf("  %-28s : %.4f\n", "AUC-ROC OOF ganador", auc_roc))
cat(sprintf("  %-28s : %.2f\n", "Threshold", fin_W$threshold))
cat(sprintf("  %-28s : %s\n",   "Técnica balanceo", res_winner$best_technique))
cat(sprintf("  %-28s : depth=%d eta=%.3f gamma=%.1f nrounds=%d\n",
            "HP ganador",
            res_W_hp$max_depth, res_W_hp$eta, res_W_hp$gamma, fin_W_nr))
cat(sprintf("  %-28s : %s\n",   "Bogotá en train", "EXCLUIDA"))
cat(sprintf("  %-28s : %s\n",   "Submission ganador", path_W))
cat(sprintf("  %-28s : %s\n",   "Submission Spec A", path_A))
cat(sprintf("  %-28s : %s\n",   "Submission Spec B", path_B))
cat(strrep("=", 65), "\n\n")
cat("  → Subir a Kaggle AMBAS submissions (A y B) para verificar.\n")
cat(sprintf("  → Si Spec %s gana en Kaggle confirma hipótesis:\n", WINNER_SPEC))
if (WINNER_SPEC == "A") {
  cat("     XGBoost aprovecha mejor la granularidad de los componentes\n")
  cat("     que las agregaciones preconstruidas de los índices.\n")
} else {
  cat("     Los índices compuestos capturan señales más robustas que\n")
  cat("     sus componentes individuales para este problema.\n")
}
cat("  → Comparar con XGB_007 para medir el efecto neto de la decisión\n")
cat("    índices vs componentes sobre el F1 en test.\n\n")

# nolint end
