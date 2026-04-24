#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/RandomForest/RF_006.R
#==============================================================================
#
# ALGORITMO: Random Forest — HP tuning completo + Balanceo de clases
#
# QUÉ HACE ESTE SCRIPT:
#   FASE 1 – HP Tuning:
#     Busca la combinación óptima de mtry × min_node_size con CV-5 honesto,
#     maximizando F1 en el fold de validación. Usa todas las 110 variables.
#     (splitrule=gini y num.trees fijos; se pueden extender fácilmente.)
#
#   FASE 2 – Balanceo de clases:
#     Con los HP óptimos de la Fase 1, compara las 5 técnicas del cuaderno
#     del curso via CV-5 honesto:
#       (a) baseline      — sin corrección, threshold óptimo (PR-curve)
#       (b) class_weights — case.weights en ranger (peso = n_may/n_min)
#       (c) downsample    — caret::downSample en el fold de train
#       (d) upsample      — caret::upSample en el fold de train
#       (e) smote         — themis::smote en el fold de train (k=5)
#     En TODAS las técnicas, el rebalanceo se aplica SOLO al fold de
#     entrenamiento → sin data leakage desde el fold de validación.
#     El threshold se optimiza sobre las probabilidades predichas en el
#     fold de validación (PR-curve → max F1).
#
#   FASE 3 – Modelo final:
#     Entrena con mejor HP + mejor técnica sobre TODO el train.
#     Threshold: OOB (baseline/class_weights) o promedio CV (demás técnicas).
#     Genera submission para Kaggle.
#
# DISEÑO HONESTO (sin data leakage):
#   • Fold de validación nunca participa en: ranking de importancia,
#     rebalanceo, búsqueda de threshold de HP.
#   • SMOTE/down/upsample aplicados únicamente al fold de entrenamiento.
#   • Threshold final: OOB cuando el modelo se entrena en datos originales
#     (baseline, class_weights); promedio de threshold CV en los demás casos.
#
# GRILLA HP (Fase 1):
#   mtry          ∈ {20, 50, 80}
#   min_node_size ∈ {1,  3,  10}
#   splitrule     = "gini"  (fijo)
#   num.trees     = 150 (CV) / 500 (modelo final)
#   → 9 combos × 5 folds = 45 modelos
#
# TÉCNICAS (Fase 2): 5 técnicas × 5 folds = 25 modelos
#
# TIEMPO ESTIMADO: ~40-50 min con 9+ cores
#   Fase 1: ~25 min  |  Fase 2: ~15 min  |  Fase 3: ~3 min
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#
# OUTPUTS:
#   02_outputs/models/RandomForest/RF_006/cv_hp_results.csv
#   02_outputs/models/RandomForest/RF_006/cv_hp_summary.csv
#   02_outputs/models/RandomForest/RF_006/cv_balance_results.csv
#   02_outputs/models/RandomForest/RF_006/cv_balance_summary.csv
#   02_outputs/models/RandomForest/RF_006/hp_comparison.png
#   02_outputs/models/RandomForest/RF_006/balance_comparison.png
#   02_outputs/models/RandomForest/RF_006/varimp.png
#   02_outputs/models/RandomForest/RF_006/threshold.png
#   02_outputs/models/RandomForest/RF_006/diagnostics.rds
#   03_submissions/RF_006_*.csv
#   02_outputs/model_registry.csv  (append)
#
# REPRODUCIBILIDAD: semilla global 42. Correr desde raíz del proyecto.
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
  caret,       # downSample(), upSample()
  themis,      # smote()
  fs           # dir_create()
)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
AUTOR    <- "Jonathan"
MODEL_ID <- "RF_006"

set.seed(42)

dir_model <- file.path("02_outputs/models/RandomForest", MODEL_ID)
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model, recurse = TRUE)
fs::dir_create(dir_subs,  recurse = TRUE)

# ---------- Grilla de HP a tunear (Fase 1) ----------
HP_GRID <- expand.grid(
  mtry          = c(20L, 50L, 80L),
  min_node_size = c(1L,  3L,  10L),
  stringsAsFactors = FALSE
)

SPLITRULE       <- "gini"
NUM_TREES_CV    <- 150L   # árboles en CV (velocidad)
NUM_TREES_FINAL <- 500L   # árboles en modelo final (calidad)
K_FOLDS         <- 5L

# Técnicas de balanceo a comparar (Fase 2)
TECHNIQUES <- c("baseline", "class_weights", "downsample", "upsample", "smote")

n_cores <- max(1L, parallel::detectCores() - 1L)
message("Cores disponibles: ", n_cores)

# =============================================================================
# SECCIÓN 2: CARGAR DATOS
# =============================================================================
message("\n== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id

message("Train: ", nrow(train), " hogares | ", ncol(train), " columnas")
message("Test:  ", nrow(test),  " hogares | ", ncol(test),  " columnas")

pct_pobre <- round(mean(train$pobre) * 100, 1)
message("Balance: ", pct_pobre, "% pobres  /  ", round(100 - pct_pobre, 1), "% no pobres")

# =============================================================================
# SECCIÓN 3: PREPROCESAMIENTO
# =============================================================================
message("\n== Preprocesando variables ==")

preparar_X_rf <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>
    mutate(zona   = as.numeric(zona),
           ciudad = as.factor(ciudad),
           dpto   = as.factor(dpto)) |>
    mutate(across(where(is.character), as.factor))
}

X_train_raw <- preparar_X_rf(train)
X_test_raw  <- preparar_X_rf(test)

# Alinear niveles ciudad y dpto: unión train ∪ test
for (v in c("ciudad", "dpto")) {
  all_levels       <- union(levels(X_train_raw[[v]]), levels(X_test_raw[[v]]))
  X_train_raw[[v]] <- factor(X_train_raw[[v]], levels = all_levels)
  X_test_raw[[v]]  <- factor(X_test_raw[[v]],  levels = all_levels)
}

cols_comunes <- intersect(names(X_train_raw), names(X_test_raw))
X_train      <- X_train_raw |> select(all_of(cols_comunes))
X_test       <- X_test_raw  |> select(all_of(cols_comunes))

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
X_train <- X_train |> select(-any_of(VARS_EXCLUIR))
X_test <- X_test  |> select(-any_of(VARS_EXCLUIR))

message("  Variables excluidas manualmente: ", length(VARS_EXCLUIR))
message("  Predictores tras exclusión: ", ncol(X_train))

niveles <- c("No", "Yes")
y_train <- factor(ifelse(train$pobre == 1, "Yes", "No"), levels = niveles)

P_TOTAL <- ncol(X_train)
message("  Predictores disponibles: ", P_TOTAL, "  (incluye ciudad y dpto)")
message("  NAs en X_train: ", sum(is.na(X_train)))

# =============================================================================
# SECCIÓN 4: FUNCIONES AUXILIARES
# =============================================================================

f1_score <- function(probs, labels, th) {
  pred <- factor(ifelse(probs >= th, "Yes", "No"), levels = niveles)
  TP <- sum(pred == "Yes" & labels == "Yes")
  FP <- sum(pred == "Yes" & labels == "No")
  FN <- sum(pred == "No"  & labels == "Yes")
  if ((2 * TP + FP + FN) == 0) return(NA_real_)
  2 * TP / (2 * TP + FP + FN)
}

pr_rc <- function(probs, labels, th) {
  pred <- factor(ifelse(probs >= th, "Yes", "No"), levels = niveles)
  TP <- sum(pred == "Yes" & labels == "Yes")
  FP <- sum(pred == "Yes" & labels == "No")
  FN <- sum(pred == "No"  & labels == "Yes")
  list(
    precision = if ((TP + FP) == 0) NA_real_ else TP / (TP + FP),
    recall    = if ((TP + FN) == 0) NA_real_ else TP / (TP + FN)
  )
}

# smote_con_factores(): aplica themis::smote() a un df con columnas factor.
# Convierte factors a integer antes de SMOTE, los reconstruye después
# (redondeo + clamping al rango válido de niveles).
smote_con_factores <- function(df_in, var_y, k = 5, over_ratio = 1) {
  factor_cols <- setdiff(
    names(df_in)[sapply(df_in, is.factor)],
    var_y
  )
  factor_lvls <- lapply(factor_cols, function(col) levels(df_in[[col]]))
  names(factor_lvls) <- factor_cols

  df_num <- df_in |>
    mutate(across(all_of(factor_cols), as.integer))

  df_out <- themis::smote(df_num, var = var_y, k = k, over_ratio = over_ratio)

  for (col in factor_cols) {
    lvls     <- factor_lvls[[col]]
    int_vals <- as.integer(round(df_out[[col]]))
    int_vals <- pmax(1L, pmin(length(lvls), int_vals))
    df_out[[col]] <- factor(lvls[int_vals], levels = lvls)
  }
  df_out
}

# Busca threshold óptimo (max F1) sobre una curva PR con grilla de 0.01
buscar_th <- function(probs, labels, grid = seq(0.05, 0.95, by = 0.01)) {
  resultados <- map_dfr(grid, function(th) {
    tibble(threshold = th, F1 = f1_score(probs, labels, th))
  })
  resultados |> filter(!is.na(F1)) |> slice_max(F1, n = 1, with_ties = FALSE)
}

# Entrena un modelo ranger en el fold y devuelve F1 + threshold óptimos
# El threshold se optimiza sobre las predicciones en el fold de validación
# (criterio PR-curve max F1, consistente con el cuaderno del curso).
entrenar_fold <- function(X_tr, y_tr, X_val, y_val,
                          mtry_val, min_node_val, seed_val,
                          case_wts = NULL) {
  df_tr    <- bind_cols(X_tr, pobre = y_tr)
  mtry_eff <- min(mtry_val, ncol(X_tr))

  set.seed(seed_val)
  mod <- ranger::ranger(
    pobre ~ .,
    data                      = df_tr,
    num.trees                 = NUM_TREES_CV,
    mtry                      = mtry_eff,
    splitrule                 = SPLITRULE,
    min.node.size             = min_node_val,
    probability               = TRUE,
    importance                = "none",
    respect.unordered.factors = "order",
    case.weights              = case_wts,
    num.threads               = n_cores,
    seed                      = seed_val
  )

  probs_val <- predict(mod, data = X_val)$predictions[, "Yes"]
  best_row  <- buscar_th(probs_val, y_val)
  list(F1 = best_row$F1, th = best_row$threshold)
}

# =============================================================================
# SECCIÓN 5: FASE 1 — TUNING DE HIPERPARÁMETROS
# =============================================================================
# fold_ids se genera con set.seed(42) → deterministas; se reutilizan en Fase 2.
set.seed(42)
fold_ids <- sample(rep(seq_len(K_FOLDS), length.out = nrow(X_train)))

hp_checkpoint <- file.path(dir_model, "cv_hp_results.csv")

if (file.exists(hp_checkpoint)) {
  # --- CHECKPOINT: Fase 1 ya completada, cargar resultados del disco ---
  message("\n========================================")
  message("FASE 1: cargando resultados guardados (checkpoint)")
  message("========================================")
  hp_df      <- read_csv(hp_checkpoint, show_col_types = FALSE)
  hp_summary <- read_csv(file.path(dir_model, "cv_hp_summary.csv"),
                         show_col_types = FALSE)

} else {
  # --- Fase 1: ejecutar CV de HP ---
  message("\n========================================")
  message("FASE 1: Tuning de hiperparámetros")
  message("  Grid: ", nrow(HP_GRID), " combos (mtry x min_node_size) x ",
          K_FOLDS, " folds = ", nrow(HP_GRID) * K_FOLDS, " modelos")
  message("  Todas las ", P_TOTAL, " variables — num.trees=", NUM_TREES_CV,
          " — splitrule=", SPLITRULE)
  message("========================================")

  hp_resultados <- vector("list", K_FOLDS)
  t0_hp <- Sys.time()

  for (k in seq_len(K_FOLDS)) {
    message("\n-- HP Fold ", k, " / ", K_FOLDS, " --")

    idx_val   <- which(fold_ids == k)
    idx_train <- which(fold_ids != k)

    X_fold_tr  <- X_train[idx_train, ]
    X_fold_val <- X_train[idx_val, ]
    y_fold_tr  <- y_train[idx_train]
    y_fold_val <- y_train[idx_val]

    fold_hp_res <- map_dfr(seq_len(nrow(HP_GRID)), function(i) {
      hp     <- HP_GRID[i, ]
      seed_i <- 42 * k + i

      res <- entrenar_fold(
        X_tr       = X_fold_tr,  y_tr   = y_fold_tr,
        X_val      = X_fold_val, y_val  = y_fold_val,
        mtry_val   = hp$mtry,    min_node_val = hp$min_node_size,
        seed_val   = seed_i
      )

      tibble(
        fold          = k,
        mtry          = hp$mtry,
        min_node_size = hp$min_node_size,
        F1            = res$F1,
        th            = res$th
      )
    })

    message("  mtry | mns | F1_val:")
    for (i in seq_len(nrow(fold_hp_res))) {
      message("    ", formatC(fold_hp_res$mtry[i],          width = 3),
              " |  ", formatC(fold_hp_res$min_node_size[i], width = 2),
              " |  ", round(fold_hp_res$F1[i], 4))
    }

    hp_resultados[[k]] <- fold_hp_res
  }

  t1_hp <- Sys.time()
  message("\nFase 1 completada en ",
          round(as.numeric(difftime(t1_hp, t0_hp, units = "mins")), 1), " min")

  hp_df <- bind_rows(hp_resultados)

  hp_summary <- hp_df |>
    group_by(mtry, min_node_size) |>
    summarise(
      F1_mean = mean(F1, na.rm = TRUE),
      F1_sd   = sd(F1,   na.rm = TRUE),
      .groups = "drop"
    ) |>
    arrange(desc(F1_mean))

  write_csv(hp_df,      file.path(dir_model, "cv_hp_results.csv"))
  write_csv(hp_summary, file.path(dir_model, "cv_hp_summary.csv"))
  message("  Guardado: cv_hp_results.csv, cv_hp_summary.csv")
}

# Selección del mejor HP (mayor F1 promedio CV-5)
cat("\n-- HP Summary CV-5 --\n")
print(hp_summary |> mutate(across(where(is.double), ~ round(., 4))))

best_hp       <- hp_summary |> slice_max(F1_mean, n = 1, with_ties = FALSE)
BEST_MTRY     <- best_hp$mtry
BEST_MIN_NODE <- best_hp$min_node_size
BEST_F1_HP    <- best_hp$F1_mean

message("\n  Mejor HP → mtry=", BEST_MTRY, " | min_node_size=", BEST_MIN_NODE,
        " | F1 CV-5 = ", round(BEST_F1_HP, 4))

# =============================================================================
# SECCIÓN 6: FASE 2 — COMPARACIÓN DE TÉCNICAS DE BALANCEO
# =============================================================================
message("\n========================================")
message("FASE 2: Técnicas de balanceo de clases")
message("  HP fijo: mtry=", BEST_MTRY, " | min_node_size=", BEST_MIN_NODE)
message("  num.trees=", NUM_TREES_CV, " | K=", K_FOLDS, " folds")
message("  Técnicas: ", paste(TECHNIQUES, collapse = ", "))
message("========================================")

balance_resultados <- vector("list", K_FOLDS)
t0_bal <- Sys.time()

for (k in seq_len(K_FOLDS)) {
  message("\n-- Balance Fold ", k, " / ", K_FOLDS, " --")

  idx_val   <- which(fold_ids == k)
  idx_train <- which(fold_ids != k)

  X_fold_tr  <- X_train[idx_train, ]
  X_fold_val <- X_train[idx_val, ]
  y_fold_tr  <- y_train[idx_train]
  y_fold_val <- y_train[idx_val]

  fold_bal_res <- map_dfr(TECHNIQUES, function(tech) {
    seed_tech <- 42 * k + match(tech, TECHNIQUES) * 1000

    # Preparar datos de entrenamiento según técnica
    X_tr_use <- X_fold_tr
    y_tr_use <- y_fold_tr
    wts      <- NULL

    if (tech == "class_weights") {
      # Peso inversamente proporcional a la frecuencia de cada clase
      n_yes <- sum(y_fold_tr == "Yes")
      n_no  <- sum(y_fold_tr == "No")
      wts   <- ifelse(y_fold_tr == "Yes", n_no / n_yes, 1.0)

    } else if (tech == "downsample") {
      # Submuestreo: reducir mayoría hasta igualar minoría
      set.seed(seed_tech)
      ds       <- caret::downSample(x = as.data.frame(X_fold_tr),
                                    y = y_fold_tr, yname = "pobre")
      X_tr_use <- ds |> select(-pobre)
      y_tr_use <- ds$pobre

    } else if (tech == "upsample") {
      # Sobremuestreo: replicar minoría hasta igualar mayoría
      set.seed(seed_tech)
      us       <- caret::upSample(x = as.data.frame(X_fold_tr),
                                  y = y_fold_tr, yname = "pobre")
      X_tr_use <- us |> select(-pobre)
      y_tr_use <- us$pobre

    } else if (tech == "smote") {
      # SMOTE: sintetizar ejemplos de la clase minoritaria.
      # smote_con_factores() convierte factores a integer para SMOTE
      # y los reconstituye después (redondeo + clamping al rango válido).
      set.seed(seed_tech)
      df_in    <- bind_cols(as.data.frame(X_fold_tr), pobre = y_fold_tr)
      df_out   <- smote_con_factores(df_in, var_y = "pobre", k = 5, over_ratio = 1)
      X_tr_use <- df_out |> select(-pobre)
      y_tr_use <- df_out$pobre
    }
    # tech == "baseline": datos sin modificar, case_wts = NULL

    res <- entrenar_fold(
      X_tr       = X_tr_use,    y_tr   = y_tr_use,
      X_val      = X_fold_val,  y_val  = y_fold_val,
      mtry_val   = BEST_MTRY,   min_node_val = BEST_MIN_NODE,
      seed_val   = seed_tech,   case_wts     = wts
    )

    tibble(fold = k, technique = tech, F1 = res$F1, th = res$th)
  })

  message("  Técnica         | F1_val:")
  for (i in seq_len(nrow(fold_bal_res))) {
    message("    ", formatC(fold_bal_res$technique[i], width = 14, flag = "-"),
            " → ", round(fold_bal_res$F1[i], 4))
  }

  balance_resultados[[k]] <- fold_bal_res
}

t1_bal <- Sys.time()
message("\nFase 2 completada en ",
        round(as.numeric(difftime(t1_bal, t0_bal, units = "mins")), 1), " min")

# Selección de la mejor técnica (mayor F1 promedio CV-5)
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

cat("\n-- Resultados por técnica (CV-5) --\n")
print(bal_summary |> mutate(across(where(is.double), ~ round(., 4))))

best_technique <- bal_summary |>
  slice_max(F1_mean, n = 1, with_ties = FALSE) |>
  pull(technique)
best_f1_bal <- bal_summary |>
  filter(technique == best_technique) |>
  pull(F1_mean)

message("\n  Mejor técnica: ", best_technique,
        "  (F1 CV-5 = ", round(best_f1_bal, 4), ")")

write_csv(bal_df,      file.path(dir_model, "cv_balance_results.csv"))
write_csv(bal_summary, file.path(dir_model, "cv_balance_summary.csv"))
message("  Guardado: cv_balance_results.csv, cv_balance_summary.csv")

# =============================================================================
# SECCIÓN 7: FASE 3 — MODELO FINAL
# =============================================================================
message("\n========================================")
message("FASE 3: Modelo final")
message("  HP: mtry=", BEST_MTRY, " | min_node_size=", BEST_MIN_NODE,
        " | num.trees=", NUM_TREES_FINAL)
message("  Técnica de balanceo: ", best_technique)
message("========================================")

# Preparar el dataset de entrenamiento según la técnica óptima
X_final   <- X_train
y_final   <- y_train
wts_final <- NULL

if (best_technique == "class_weights") {
  n_yes_f   <- sum(y_train == "Yes")
  n_no_f    <- sum(y_train == "No")
  wts_final <- ifelse(y_train == "Yes", n_no_f / n_yes_f, 1.0)
  message("  Peso clase positiva (Yes/pobre): ", round(n_no_f / n_yes_f, 3))

} else if (best_technique == "downsample") {
  set.seed(42)
  ds_final <- caret::downSample(x = as.data.frame(X_train),
                                y = y_train, yname = "pobre")
  X_final  <- ds_final |> select(-pobre)
  y_final  <- ds_final$pobre
  message("  Down-sampled: ", nrow(X_final), " obs  | ",
          "Yes=", sum(y_final == "Yes"), " / No=", sum(y_final == "No"))

} else if (best_technique == "upsample") {
  set.seed(42)
  us_final <- caret::upSample(x = as.data.frame(X_train),
                               y = y_train, yname = "pobre")
  X_final  <- us_final |> select(-pobre)
  y_final  <- us_final$pobre
  message("  Up-sampled: ", nrow(X_final), " obs  | ",
          "Yes=", sum(y_final == "Yes"), " / No=", sum(y_final == "No"))

} else if (best_technique == "smote") {
  set.seed(42)
  df_smote_full <- bind_cols(as.data.frame(X_train), pobre = y_train)
  df_smote_out  <- smote_con_factores(df_smote_full, var_y = "pobre",
                                       k = 5, over_ratio = 1)
  X_final <- df_smote_out |> select(-pobre)
  y_final <- df_smote_out$pobre
  message("  SMOTE aplicado: ", nrow(X_final), " obs  | ",
          "Yes=", sum(y_final == "Yes"), " / No=", sum(y_final == "No"))
}

df_final_rf <- bind_cols(X_final, pobre = y_final)
mtry_final  <- min(BEST_MTRY, ncol(X_final))

message("\nEntrenando modelo final (", nrow(df_final_rf), " obs)...")
t0_f <- proc.time()

set.seed(42)
rf_final <- ranger::ranger(
  pobre ~ .,
  data                      = df_final_rf,
  num.trees                 = NUM_TREES_FINAL,
  mtry                      = mtry_final,
  splitrule                 = SPLITRULE,
  min.node.size             = BEST_MIN_NODE,
  probability               = TRUE,
  importance                = "permutation",
  respect.unordered.factors = "order",
  case.weights              = wts_final,
  num.threads               = n_cores,
  seed                      = 42
)

t1_f <- proc.time()
message("  Modelo final entrenado en ",
        round((t1_f - t0_f)[["elapsed"]] / 60, 2), " min")

# ---------- Threshold ----------
# OOB para técnicas que no alteran la distribución de entrenamiento;
# promedio del CV para técnicas con resampling (la distribución OOB del
# modelo no corresponde a la distribución real del problema).
if (best_technique %in% c("baseline", "class_weights")) {
  probs_oob <- rf_final$predictions[, "Yes"]
  best_row  <- buscar_th(probs_oob, y_train)   # y_train = distribución real
  best_th   <- best_row$threshold
  message("  Threshold desde OOB: ", best_th,
          "  | F1 OOB: ", round(best_row$F1, 4))
} else {
  best_th <- bal_summary |>
    filter(technique == best_technique) |>
    pull(th_mean) |>
    round(2)
  message("  Threshold desde CV promedio: ", best_th)
}

# ---------- Probabilidades para evaluación y curvas ROC/PR ----------
# Para baseline y class_weights: usar predicciones OOB (honesto).
# Para resampled (down/up/smote): OOB corresponde a distribución artificial,
# se usa predict() sobre X_train (in-sample, ligeramente optimista).
if (best_technique %in% c("baseline", "class_weights")) {
  probs_eval   <- rf_final$predictions[, "Yes"]   # OOB — honesto
  eval_label   <- "OOB"
} else {
  probs_eval   <- predict(rf_final, data = X_train)$predictions[, "Yes"]
  eval_label   <- "in-sample"
}

f1_eval  <- f1_score(probs_eval, y_train, best_th)
pr_eval  <- pr_rc(probs_eval,   y_train, best_th)

message("  F1 (", eval_label, ", th=", best_th, "): ", round(f1_eval, 4))
message("  Precision: ", round(pr_eval$precision, 4),
        "  | Recall: ", round(pr_eval$recall, 4))

roc_obj <- pROC::roc(
  response  = as.integer(y_train == "Yes"),
  predictor = probs_eval,
  quiet     = TRUE
)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC (", eval_label, "): ", round(auc_roc, 4))

# Importancia de variables
varimp_final   <- rf_final$variable.importance
feature_matrix <- data.frame(
  variable    = names(varimp_final),
  importancia = as.numeric(varimp_final)
) |>
  arrange(desc(importancia)) |>
  mutate(rank = row_number())

write_csv(feature_matrix, file.path(dir_model, "feature_matrix.csv"))
message("  Guardado: feature_matrix.csv (", nrow(feature_matrix), " variables)")

# =============================================================================
# SECCIÓN 8: GRÁFICOS DE DIAGNÓSTICO
# =============================================================================
message("\n== Generando gráficos ==")

# --- Gráfico 1: F1 CV-5 por combinación de HP ---
p_hp <- hp_summary |>
  mutate(
    combo   = paste0("mtry=", mtry, " / mns=", min_node_size),
    is_best = (mtry == BEST_MTRY & min_node_size == BEST_MIN_NODE)
  ) |>
  ggplot(aes(x = reorder(combo, F1_mean), y = F1_mean, fill = is_best)) +
  geom_col(alpha = 0.85, show.legend = FALSE) +
  geom_errorbar(aes(ymin = F1_mean - F1_sd, ymax = F1_mean + F1_sd),
                width = 0.3, color = "gray40") +
  scale_fill_manual(values = c("FALSE" = "#BDBDBD", "TRUE" = "#534AB7")) +
  coord_flip() +
  labs(
    title    = paste0("RF: F1 CV-5 por combinación de HP — ", MODEL_ID),
    subtitle = sprintf("Mejor: mtry=%d / mns=%d | F1 CV-5 = %.4f",
                       BEST_MTRY, BEST_MIN_NODE, BEST_F1_HP),
    x        = "Combinación (mtry / min_node_size)",
    y        = "F1 promedio CV-5",
    caption  = "Barras de error = ±1 SD entre folds. Azul = mejor combo."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "hp_comparison.png"),
       p_hp, width = 7, height = 6, dpi = 150)
message("  Guardado: hp_comparison.png")

# --- Gráfico 2: F1 CV-5 por técnica de balanceo ---
p_bal <- bal_summary |>
  mutate(is_best = technique == best_technique) |>
  ggplot(aes(x = reorder(technique, F1_mean), y = F1_mean, fill = is_best)) +
  geom_col(alpha = 0.85, show.legend = FALSE) +
  geom_errorbar(aes(ymin = F1_mean - F1_sd, ymax = F1_mean + F1_sd),
                width = 0.3, color = "gray40") +
  scale_fill_manual(values = c("FALSE" = "#BDBDBD", "TRUE" = "#534AB7")) +
  coord_flip() +
  labs(
    title    = paste0("RF: F1 CV-5 por técnica de balanceo — ", MODEL_ID),
    subtitle = sprintf("Mejor técnica: %s | F1 CV-5 = %.4f",
                       best_technique, best_f1_bal),
    x        = "Técnica de balanceo",
    y        = "F1 promedio CV-5",
    caption  = "Barras de error = ±1 SD entre folds. Azul = mejor técnica."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "balance_comparison.png"),
       p_bal, width = 7, height = 5, dpi = 150)
message("  Guardado: balance_comparison.png")

# --- Gráfico 3: Importancia de variables (Top 20) ---
top20 <- feature_matrix |> slice_head(n = 20)

p_varimp <- ggplot(top20, aes(x = reorder(variable, importancia), y = importancia)) +
  geom_col(fill = "#534AB7", alpha = 0.85) +
  coord_flip() +
  labs(
    title    = paste0("RF: importancia de variables (Top 20) — ", MODEL_ID),
    subtitle = paste0("Técnica: ", best_technique,
                      " | mtry=", BEST_MTRY, " | mns=", BEST_MIN_NODE),
    x        = NULL,
    y        = "Importancia por permutación"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "varimp.png"),
       p_varimp, width = 8, height = 6, dpi = 150)
message("  Guardado: varimp.png")

# --- Gráfico 4: Threshold vs métricas ---
th_full <- map_dfr(seq(0.05, 0.95, by = 0.01), function(th) {
  prc <- pr_rc(probs_eval, y_train, th)
  tibble(threshold = th,
         F1        = f1_score(probs_eval, y_train, th),
         Precision = prc$precision,
         Recall    = prc$recall)
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
    title    = paste0("RF: métricas por threshold — ", MODEL_ID),
    subtitle = sprintf("th=%.2f | técnica=%s | mtry=%d | mns=%d | eval=%s",
                       best_th, best_technique, BEST_MTRY, BEST_MIN_NODE, eval_label),
    x        = "Threshold de clasificación",
    y        = "Valor de la métrica",
    color    = NULL,
    caption  = "Línea discontinua = threshold seleccionado. Punteada = 0.5."
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

ggsave(file.path(dir_model, "threshold.png"),
       p_threshold, width = 8, height = 5, dpi = 150)
message("  Guardado: threshold.png")

# --- Gráfico 5: Curva ROC ---
roc_df <- tibble(
  fpr = 1 - roc_obj$specificities,
  tpr = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.72, y = 0.65,
           label = "Clasificador aleatorio",
           size = 3, color = "gray50", angle = 35) +
  annotate("text", x = 0.55, y = 0.15,
           label = sprintf("AUC-ROC = %.4f", auc_roc),
           size = 3.5, color = "#534AB7") +
  scale_x_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  labs(
    title    = paste0("RF: curva ROC — ", MODEL_ID),
    subtitle = sprintf("mtry=%d | mns=%d | técnica: %s | eval: %s",
                       BEST_MTRY, BEST_MIN_NODE, best_technique, eval_label),
    x        = "Tasa de Falsos Positivos  (1 - Specificity)",
    y        = "Tasa de Verdaderos Positivos  (Recall)",
    caption  = paste0("AUC calculado sobre predicciones ", eval_label, ".")
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc.png"),
       p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc.png")

# --- Gráfico 6: Curva Precision-Recall ---
n_pos <- sum(y_train == "Yes")
n_neg <- sum(y_train == "No")
pr_curve_df <- tibble(
  TP        = roc_obj$sensitivities * n_pos,
  FP        = (1 - roc_obj$specificities) * n_neg,
  recall    = roc_obj$sensitivities,
  precision = TP / (TP + FP)
) |> filter(is.finite(precision), is.finite(recall))

prev_rate <- n_pos / (n_pos + n_neg)
pt_opt    <- tibble(recall = pr_eval$recall, precision = pr_eval$precision)

p_prcurve <- ggplot(pr_curve_df, aes(x = recall, y = precision)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_hline(yintercept = prev_rate,
             linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.85, y = prev_rate + 0.015,
           label = sprintf("Clasificador aleatorio (%.0f%%)", prev_rate * 100),
           size = 3, color = "gray50") +
  geom_point(data = pt_opt, aes(x = recall, y = precision),
             color = "#E84855", size = 3) +
  annotate("text",
           x     = pr_eval$recall - 0.1,
           y     = pr_eval$precision + 0.015,
           label = sprintf("th=%.2f\nF1=%.4f", best_th, f1_eval),
           size  = 3, color = "#E84855") +
  labs(
    title    = paste0("RF: curva Precision-Recall — ", MODEL_ID),
    subtitle = sprintf("AUC-ROC: %.4f | mtry=%d | mns=%d | técnica: %s",
                       auc_roc, BEST_MTRY, BEST_MIN_NODE, best_technique),
    x        = "Recall  (fracción de pobres capturada)",
    y        = "Precision  (fracción de predichos pobres que lo son)",
    caption  = paste0("Curva basada en predicciones ", eval_label,
                      ". Punto rojo = threshold óptimo.")
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve.png"),
       p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve.png")

# Guardar diagnósticos completos
saveRDS(
  list(
    hp_df          = hp_df,
    hp_summary     = hp_summary,
    best_mtry      = BEST_MTRY,
    best_min_node  = BEST_MIN_NODE,
    best_f1_hp     = BEST_F1_HP,
    bal_df         = bal_df,
    bal_summary    = bal_summary,
    best_technique = best_technique,
    best_f1_bal    = best_f1_bal,
    rf_final       = rf_final,
    feature_matrix = feature_matrix,
    roc_obj        = roc_obj,
    auc_roc        = auc_roc,
    eval_label     = eval_label,
    best_th        = best_th,
    f1_eval        = f1_eval,
    precision_eval = pr_eval$precision,
    recall_eval    = pr_eval$recall
  ),
  file.path(dir_model, "diagnostics.rds")
)
message("  Guardado: diagnostics.rds")

# =============================================================================
# SECCIÓN 9: SUBMISSION PARA KAGGLE
# =============================================================================
message("\n========================================")
message("Generando submission para Kaggle")
message("========================================")

probs_test <- predict(rf_final, data = X_test)$predictions[, "Yes"]
pred_test  <- as.integer(probs_test >= best_th)

submission <- tibble(id = ids, pobre = pred_test)

sub_name <- sprintf(
  "RF_006_mtry%d_mns%d_ntrees%d_%s_th%03.0f.csv",
  BEST_MTRY, BEST_MIN_NODE, NUM_TREES_FINAL,
  best_technique, best_th * 100
)
sub_path <- file.path(dir_subs, sub_name)
write_csv(submission, sub_path)

message("  Guardado: ", sub_path)
message("  Threshold: ", best_th,
        "  | Pobres predichos: ", sum(pred_test),
        " (", round(mean(pred_test) * 100, 1), "%)")

# =============================================================================
# SECCIÓN 10: REGISTRO EN model_registry.csv
# =============================================================================
message("\n========================================")
message("Registro en model_registry.csv")
message("========================================")

nueva_fila <- tibble(
  model_id           = MODEL_ID,
  fecha              = Sys.Date(),
  autor              = AUTOR,
  algoritmo          = "RandomForest",
  n_features         = P_TOTAL,
  imbalance_strategy = best_technique,
  cv_folds           = K_FOLDS,
  cv_F1              = round(f1_eval,            4),
  cv_Precision       = round(pr_eval$precision, 4),
  cv_Recall          = round(pr_eval$recall,    4),
  auc_roc            = round(auc_roc,           4),
  kaggle_public_F1   = NA_real_,
  threshold          = best_th,
  notas              = paste0(
    "RF ranger. Fase1: HP tuning CV-5 (mtry x mns). ",
    "Mejor HP: mtry=", BEST_MTRY, " mns=", BEST_MIN_NODE,
    " F1_HP=", round(BEST_F1_HP, 4), ". ",
    "Fase2: comparacion ", length(TECHNIQUES), " tecnicas balanceo CV-5. ",
    "Mejor: ", best_technique, " F1_bal=", round(best_f1_bal, 4), ". ",
    "Modelo final: num.trees=", NUM_TREES_FINAL,
    " splitrule=", SPLITRULE,
    " th=", best_th, ". ",
    "Todas las ", P_TOTAL, " vars (incluye ciudad y dpto)."
  ),
  cp              = NA_real_,
  maxdepth        = NA_real_,
  train_F1        = round(f1_eval,              4),
  train_Precision = round(pr_eval$precision,    4),
  train_Recall    = round(pr_eval$recall,       4)
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

# nolint end
