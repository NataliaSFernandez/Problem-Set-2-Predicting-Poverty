#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/LPM/LPM_003.R
#==============================================================================
#
# ALGORITMO: Linear Probability Model (LPM) — Feature selection + Balanceo
#
# QUÉ HACE ESTE SCRIPT:
#
#   FASE 1 – Selección de variables por importancia RF_006:
#     Usa el ranking de importancia de permutación de RF_006 guardado en
#     02_outputs/models/RandomForest/RF_006/feature_matrix.csv.
#     Compara LPM entrenado con los subconjuntos top {20, 40, 60, todos}
#     mediante CV-5 honesto (threshold optimizado por fold sobre las
#     predicciones del fold de validación). Selecciona el subconjunto
#     que maximiza F1 promedio CV-5.
#
#   FASE 2 – Balanceo de clases:
#     Con los predictores óptimos de Fase 1, compara 5 técnicas via CV-5:
#       (a) baseline      — sin corrección, threshold óptimo
#       (b) class_weights — argumento weights en lm() (peso inversamente
#                           proporcional a la frecuencia de clase)
#       (c) downsample    — caret::downSample en el fold de entrenamiento
#       (d) upsample      — caret::upSample en el fold de entrenamiento
#       (e) smote         — themis::smote en el fold de entrenamiento (k=5)
#     El rebalanceo se aplica SOLO al fold de entrenamiento → sin data leakage.
#     El threshold se optimiza sobre el fold de validación (max F1).
#
#   FASE 3 – Modelo final:
#     Entrena con mejor subconjunto + mejor técnica sobre TODO el train.
#     Threshold: para baseline/class_weights se optimiza sobre predicciones
#     in-sample del modelo final; para técnicas con resampling se usa el
#     promedio de thresholds CV-5 (la distribución OOB no es representativa).
#     Genera submission para Kaggle y actualiza model_registry.csv.
#
# DISEÑO HONESTO (sin data leakage):
#   • El fold de validación no participa en: selección de variables,
#     rebalanceo ni búsqueda de threshold.
#   • Todos los folds son los mismos en Fase 1 y Fase 2 (generados una vez).
#   • SMOTE/down/upsample se aplican únicamente al fold de entrenamiento.
#
# MULTICOLINEALIDAD EN LPM:
#   Con el subconjunto "all" (110 variables), lm() detectará combinaciones
#   lineales perfectas (indice_formalidad = suma de sus 4 componentes, etc.)
#   y descartará silenciosamente las columnas redundantes (rank-deficient fit).
#   Con los subconjuntos top20/40/60, la colinealidad parcial es menor;
#   lm() emitirá warnings que se suprimen con suppressWarnings().
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#   02_outputs/models/RandomForest/RF_006/feature_matrix.csv
#
# OUTPUTS:
#   02_outputs/models/LPM/LPM_003/cv_features_results.csv
#   02_outputs/models/LPM/LPM_003/cv_features_summary.csv
#   02_outputs/models/LPM/LPM_003/cv_balance_results.csv
#   02_outputs/models/LPM/LPM_003/cv_balance_summary.csv
#   02_outputs/models/LPM/LPM_003/features_comparison.png
#   02_outputs/models/LPM/LPM_003/balance_comparison.png
#   02_outputs/models/LPM/LPM_003/threshold.png
#   02_outputs/models/LPM/LPM_003/roc.png
#   02_outputs/models/LPM/LPM_003/prcurve.png
#   02_outputs/models/LPM/LPM_003/coefs.png
#   02_outputs/models/LPM/LPM_003/diagnostics.rds
#   03_submissions/LPM_003_*.csv
#   02_outputs/model_registry.csv  (append, reemplaza fila LPM_003 si existe)
#
# REPRODUCIBILIDAD: semilla global 42. Correr desde la raíz del proyecto.
#==============================================================================
# nolint start

# =============================================================================
# SECCIÓN 0: PAQUETES
# =============================================================================
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,   # dplyr, ggplot2, purrr, tibble, readr
  pROC,        # roc(), auc()
  caret,       # downSample(), upSample()
  themis,      # smote()
  broom,       # tidy() para coeficientes del modelo final
  fs           # dir_create()
)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
AUTOR    <- "Jonathan"
MODEL_ID <- "LPM_003"

set.seed(42)

dir_model     <- file.path("02_outputs/models/LPM", MODEL_ID)
dir_subs      <- "03_submissions"
reg_path      <- "02_outputs/model_registry.csv"
feat_mat_path <- "02_outputs/models/RandomForest/RF_006/feature_matrix.csv"

fs::dir_create(dir_model, recurse = TRUE)
fs::dir_create(dir_subs,  recurse = TRUE)
fs::dir_create("02_outputs", recurse = TRUE)

K_FOLDS    <- 5L

# Subconjuntos a comparar: top 20, 40, 60 y todas las variables
# NA = usar todas las variables disponibles en la feature_matrix
N_TOPS <- c(20L, 40L, 60L, NA_integer_)

# Técnicas de balanceo a comparar (Fase 2)
TECHNIQUES <- c("baseline", "class_weights", "downsample", "upsample", "smote")

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

# y_train_num: vector 0/1 para lm() (necesita numérico, no factor)
y_train_num <- as.numeric(train$pobre)

# =============================================================================
# SECCIÓN 3: CARGAR FEATURE MATRIX Y DEFINIR SUBCONJUNTOS
# =============================================================================
message("\n== Cargando feature matrix de RF_006 ==")

feature_matrix_rf <- read_csv(feat_mat_path, show_col_types = FALSE) |>
  arrange(rank)

message("Variables en feature_matrix RF_006: ", nrow(feature_matrix_rf))

# Preprocesamiento: misma lógica que RF_006 para garantizar consistencia de tipos
# zona → numeric (1/2 en lugar del factor original)
# ciudad, dpto → factor (lm() los expandirá a dummies automáticamente)
# Otras character → factor
preparar_X_lpm <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>
    mutate(
      zona   = as.numeric(as.factor(zona)),
      ciudad = as.factor(ciudad),
      dpto   = as.factor(dpto)
    ) |>
    mutate(across(where(is.character), as.factor))
}

X_train_raw <- preparar_X_lpm(train)
X_test_raw  <- preparar_X_lpm(test)

# Alinear niveles de ciudad y dpto: unión train ∪ test
# Necesario para que predict() no devuelva NA en test al encontrar niveles nuevos
for (v in c("ciudad", "dpto")) {
  all_levels      <- union(levels(X_train_raw[[v]]), levels(X_test_raw[[v]]))
  X_train_raw[[v]] <- factor(X_train_raw[[v]], levels = all_levels)
  X_test_raw[[v]]  <- factor(X_test_raw[[v]],  levels = all_levels)
}

# Mantener SOLO las columnas que aparecen en la feature_matrix Y en ambos datasets
# Esto garantiza: (1) variables relevantes, (2) no habrá NAs en test por variables ausentes
cols_en_ambos <- intersect(names(X_train_raw), names(X_test_raw))
feat_order <- feature_matrix_rf |>
  filter(variable %in% cols_en_ambos) |>
  arrange(rank) |>
  pull(variable)

X_train_all <- X_train_raw[, feat_order, drop = FALSE]
X_test_all  <- X_test_raw[,  feat_order, drop = FALSE]

P_TOTAL <- ncol(X_train_all)
message("Predictores disponibles (feature_matrix ∩ train ∩ test): ", P_TOTAL)
message("NAs en X_train_all: ", sum(is.na(X_train_all)))
message("NAs en X_test_all:  ", sum(is.na(X_test_all)))

# Función auxiliar: crea el subconjunto de variables para top N
crear_subset <- function(n_top) {
  if (is.na(n_top)) {
    list(
      X_train = X_train_all,
      X_test  = X_test_all,
      n       = P_TOTAL,
      label   = "all"
    )
  } else {
    n_eff <- min(as.integer(n_top), P_TOTAL)
    vars  <- feat_order[seq_len(n_eff)]
    list(
      X_train = X_train_all[, vars, drop = FALSE],
      X_test  = X_test_all[,  vars, drop = FALSE],
      n       = n_eff,
      label   = paste0("top", n_eff)
    )
  }
}

subsets <- lapply(N_TOPS, crear_subset)
names(subsets) <- sapply(subsets, `[[`, "label")

cat("\nSubconjuntos a evaluar:\n")
for (nm in names(subsets)) {
  cat(sprintf("  %-6s: %d variables\n", nm, subsets[[nm]]$n))
}

# =============================================================================
# SECCIÓN 4: FUNCIONES AUXILIARES
# =============================================================================

# F1 score para predicciones numéricas 0/1 (no factores)
f1_score_lpm <- function(probs, y_num, th) {
  pred <- as.integer(probs >= th)
  TP   <- sum(pred == 1L & y_num == 1L)
  FP   <- sum(pred == 1L & y_num == 0L)
  FN   <- sum(pred == 0L & y_num == 1L)
  if ((2L * TP + FP + FN) == 0L) return(NA_real_)
  2L * TP / (2L * TP + FP + FN)
}

# Precision y Recall
pr_rc_lpm <- function(probs, y_num, th) {
  pred <- as.integer(probs >= th)
  TP   <- sum(pred == 1L & y_num == 1L)
  FP   <- sum(pred == 1L & y_num == 0L)
  FN   <- sum(pred == 0L & y_num == 1L)
  list(
    precision = if ((TP + FP) == 0L) NA_real_ else TP / (TP + FP),
    recall    = if ((TP + FN) == 0L) NA_real_ else TP / (TP + FN)
  )
}

# Busca el threshold que maximiza F1 sobre una grilla
buscar_th_lpm <- function(probs, y_num, grid = seq(0.05, 0.95, by = 0.01)) {
  res <- map_dfr(grid, function(th) {
    tibble(threshold = th, F1 = f1_score_lpm(probs, y_num, th))
  })
  res |> filter(!is.na(F1)) |> slice_max(F1, n = 1, with_ties = FALSE)
}

# Wrapper SMOTE para datos con columnas factor (mismo diseño que RF_006.R)
# Convierte factores a integer antes de SMOTE, los reconstituye después
smote_con_factores_lpm <- function(df_in, var_y, k = 5, over_ratio = 1) {
  factor_cols <- setdiff(names(df_in)[sapply(df_in, is.factor)], var_y)
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

# Entrena un LPM en un fold y devuelve F1 + threshold óptimos
# wts: vector de pesos (solo para class_weights; NULL en los demás casos)
entrenar_fold_lpm <- function(X_tr, y_num_tr, X_val, y_num_val, wts = NULL) {
  df_tr <- cbind(as.data.frame(X_tr), pobre = y_num_tr)
  # suppressWarnings: lm() puede advertir "rank-deficient fit" si hay colinealidad
  mod   <- suppressWarnings(lm(pobre ~ ., data = df_tr, weights = wts))
  probs <- suppressWarnings(predict(mod, newdata = as.data.frame(X_val)))
  probs <- pmax(pmin(probs, 1), 0)   # clampear a [0,1]
  best  <- buscar_th_lpm(probs, y_num_val)
  list(F1 = best$F1, th = best$threshold)
}

# =============================================================================
# SECCIÓN 5: FOLDS DE CV (compartidos por Fase 1 y Fase 2)
# =============================================================================
# Los folds se generan UNA SOLA VEZ para que la comparación entre subconjuntos
# y técnicas sea justa: cualquier diferencia en F1 es real, no varianza de folds.
set.seed(42)
fold_ids <- sample(rep(seq_len(K_FOLDS), length.out = nrow(X_train_all)))

message("\nFolds CV-5 generados (semilla 42). Mismos folds para Fase 1 y Fase 2.")

# =============================================================================
# SECCIÓN 6: FASE 1 — SELECCIÓN DE VARIABLES
# =============================================================================
feat_checkpoint <- file.path(dir_model, "cv_features_results.csv")

if (file.exists(feat_checkpoint)) {
  message("\n========================================")
  message("FASE 1: cargando resultados guardados (checkpoint)")
  message("========================================")
  feat_df      <- read_csv(feat_checkpoint,
                            show_col_types = FALSE)
  feat_summary <- read_csv(file.path(dir_model, "cv_features_summary.csv"),
                            show_col_types = FALSE)

} else {
  message("\n========================================")
  message("FASE 1: Selección de variables (feature selection)")
  message("  Subconjuntos: ", paste(names(subsets), collapse = ", "))
  message("  K = ", K_FOLDS, " folds")
  message("  Métrica: F1 promedio CV-5")
  message("========================================")

  feat_resultados <- vector("list", length(subsets))
  names(feat_resultados) <- names(subsets)

  t0_feat <- Sys.time()

  for (subset_name in names(subsets)) {
    sbs <- subsets[[subset_name]]
    message("\n-- Subconjunto: ", subset_name,
            " (", sbs$n, " variables) --")

    fold_res <- map_dfr(seq_len(K_FOLDS), function(k) {
      idx_val   <- which(fold_ids == k)
      idx_train <- which(fold_ids != k)

      res <- entrenar_fold_lpm(
        X_tr      = sbs$X_train[idx_train, , drop = FALSE],
        y_num_tr  = y_train_num[idx_train],
        X_val     = sbs$X_train[idx_val,   , drop = FALSE],
        y_num_val = y_train_num[idx_val]
      )
      message(sprintf("  Fold %d/%d | F1 = %.4f | th = %.2f",
                      k, K_FOLDS, res$F1, res$th))
      tibble(
        subset = subset_name,
        n_vars = sbs$n,
        fold   = k,
        F1     = res$F1,
        th     = res$th
      )
    })

    feat_resultados[[subset_name]] <- fold_res
  }

  t1_feat <- Sys.time()
  message("\nFase 1 completada en ",
          round(as.numeric(difftime(t1_feat, t0_feat, units = "mins")), 1), " min")

  feat_df <- bind_rows(feat_resultados)

  feat_summary <- feat_df |>
    group_by(subset, n_vars) |>
    summarise(
      F1_mean = mean(F1, na.rm = TRUE),
      F1_sd   = sd(F1,   na.rm = TRUE),
      th_mean = mean(th, na.rm = TRUE),
      .groups = "drop"
    ) |>
    arrange(desc(F1_mean))

  write_csv(feat_df,      file.path(dir_model, "cv_features_results.csv"))
  write_csv(feat_summary, file.path(dir_model, "cv_features_summary.csv"))
  message("  Guardado: cv_features_results.csv, cv_features_summary.csv")
}

cat("\n-- Resultados Fase 1: F1 por subconjunto (CV-5) --\n")
print(feat_summary |> mutate(across(where(is.double), ~ round(., 4))))

best_feat_row <- feat_summary |> slice_max(F1_mean, n = 1, with_ties = FALSE)
BEST_SUBSET   <- best_feat_row$subset
BEST_N        <- best_feat_row$n_vars
BEST_F1_FEAT  <- best_feat_row$F1_mean

message("\n  Mejor subconjunto: ", BEST_SUBSET,
        " (", BEST_N, " vars) | F1 CV-5 = ", round(BEST_F1_FEAT, 4))

# Seleccionar las matrices del mejor subconjunto
X_train_best <- subsets[[BEST_SUBSET]]$X_train
X_test_best  <- subsets[[BEST_SUBSET]]$X_test

# =============================================================================
# SECCIÓN 7: FASE 2 — COMPARACIÓN DE TÉCNICAS DE BALANCEO
# =============================================================================
message("\n========================================")
message("FASE 2: Técnicas de balanceo de clases")
message("  Subconjunto fijo: ", BEST_SUBSET, " (", BEST_N, " variables)")
message("  Técnicas: ", paste(TECHNIQUES, collapse = ", "))
message("  K = ", K_FOLDS, " folds")
message("========================================")

bal_checkpoint <- file.path(dir_model, "cv_balance_results.csv")

if (file.exists(bal_checkpoint)) {
  message("FASE 2: cargando resultados guardados (checkpoint)")
  bal_df      <- read_csv(bal_checkpoint,       show_col_types = FALSE)
  bal_summary <- read_csv(file.path(dir_model, "cv_balance_summary.csv"),
                           show_col_types = FALSE)

} else {
  balance_resultados <- vector("list", K_FOLDS)
  t0_bal <- Sys.time()

  for (k in seq_len(K_FOLDS)) {
    message("\n-- Balance Fold ", k, " / ", K_FOLDS, " --")

    idx_val   <- which(fold_ids == k)
    idx_train <- which(fold_ids != k)

    X_fold_tr  <- X_train_best[idx_train, , drop = FALSE]
    X_fold_val <- X_train_best[idx_val,   , drop = FALSE]
    y_fold_tr  <- y_train_num[idx_train]
    y_fold_val <- y_train_num[idx_val]

    fold_bal_res <- map_dfr(TECHNIQUES, function(tech) {
      seed_tech <- 42L * k + match(tech, TECHNIQUES) * 1000L

      X_tr_use <- X_fold_tr
      y_tr_use <- y_fold_tr
      wts      <- NULL

      if (tech == "class_weights") {
        # Peso inversamente proporcional a la frecuencia: da más importancia
        # a los hogares pobres (clase minoritaria ~20%) en la función de pérdida OLS
        n_pos <- sum(y_fold_tr == 1)
        n_neg <- sum(y_fold_tr == 0)
        wts   <- ifelse(y_fold_tr == 1, n_neg / n_pos, 1.0)

      } else if (tech == "downsample") {
        # Submuestreo: reducir la clase mayoritaria hasta igualar la minoritaria
        set.seed(seed_tech)
        y_fac    <- factor(ifelse(y_fold_tr == 1, "Yes", "No"),
                           levels = c("No", "Yes"))
        ds       <- caret::downSample(x = as.data.frame(X_fold_tr),
                                      y = y_fac, yname = "pobre")
        X_tr_use <- ds |> select(-pobre)
        y_tr_use <- as.numeric(ds$pobre == "Yes")

      } else if (tech == "upsample") {
        # Sobremuestreo: replicar la clase minoritaria hasta igualar la mayoritaria
        set.seed(seed_tech)
        y_fac    <- factor(ifelse(y_fold_tr == 1, "Yes", "No"),
                           levels = c("No", "Yes"))
        us       <- caret::upSample(x = as.data.frame(X_fold_tr),
                                    y = y_fac, yname = "pobre")
        X_tr_use <- us |> select(-pobre)
        y_tr_use <- as.numeric(us$pobre == "Yes")

      } else if (tech == "smote") {
        # SMOTE: sintetiza ejemplos artificiales de la clase minoritaria
        # smote_con_factores_lpm() maneja columnas factor (ciudad, dpto si presentes)
        set.seed(seed_tech)
        y_fac  <- factor(ifelse(y_fold_tr == 1, "Yes", "No"),
                         levels = c("No", "Yes"))
        df_in  <- bind_cols(as.data.frame(X_fold_tr), pobre = y_fac)
        df_out <- smote_con_factores_lpm(df_in, var_y = "pobre",
                                         k = 5L, over_ratio = 1)
        X_tr_use <- df_out |> select(-pobre)
        y_tr_use <- as.numeric(df_out$pobre == "Yes")
      }
      # tech == "baseline": X_tr_use = X_fold_tr, y_tr_use = y_fold_tr, wts = NULL

      res <- entrenar_fold_lpm(
        X_tr      = X_tr_use, y_num_tr  = y_tr_use,
        X_val     = X_fold_val, y_num_val = y_fold_val,
        wts       = wts
      )

      message(sprintf("    %-16s → F1 = %.4f | th = %.2f", tech, res$F1, res$th))
      tibble(fold = k, technique = tech, F1 = res$F1, th = res$th)
    })

    balance_resultados[[k]] <- fold_bal_res
  }

  t1_bal <- Sys.time()
  message("\nFase 2 completada en ",
          round(as.numeric(difftime(t1_bal, t0_bal, units = "mins")), 1), " min")

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

  write_csv(bal_df,      file.path(dir_model, "cv_balance_results.csv"))
  write_csv(bal_summary, file.path(dir_model, "cv_balance_summary.csv"))
  message("  Guardado: cv_balance_results.csv, cv_balance_summary.csv")
}

cat("\n-- Resultados Fase 2: F1 por técnica de balanceo (CV-5) --\n")
print(bal_summary |> mutate(across(where(is.double), ~ round(., 4))))

best_technique <- bal_summary |>
  slice_max(F1_mean, n = 1, with_ties = FALSE) |>
  pull(technique)
best_f1_bal <- bal_summary |>
  filter(technique == best_technique) |>
  pull(F1_mean)

message("\n  Mejor técnica: ", best_technique,
        "  (F1 CV-5 = ", round(best_f1_bal, 4), ")")

# =============================================================================
# SECCIÓN 8: FASE 3 — MODELO FINAL
# =============================================================================
message("\n========================================")
message("FASE 3: Modelo final")
message("  Subconjunto: ", BEST_SUBSET, " (", BEST_N, " variables)")
message("  Técnica de balanceo: ", best_technique)
message("========================================")

# Preparar el dataset de entrenamiento según la técnica óptima
X_final   <- X_train_best
y_final   <- y_train_num
wts_final <- NULL

if (best_technique == "class_weights") {
  n_pos_f   <- sum(y_train_num == 1)
  n_neg_f   <- sum(y_train_num == 0)
  wts_final <- ifelse(y_train_num == 1, n_neg_f / n_pos_f, 1.0)
  message("  Peso clase positiva (pobre=1): ", round(n_neg_f / n_pos_f, 3))

} else if (best_technique == "downsample") {
  set.seed(42)
  y_fac    <- factor(ifelse(y_train_num == 1, "Yes", "No"), levels = c("No", "Yes"))
  ds_final <- caret::downSample(x = as.data.frame(X_train_best),
                                y = y_fac, yname = "pobre")
  X_final  <- ds_final |> select(-pobre)
  y_final  <- as.numeric(ds_final$pobre == "Yes")
  message("  Down-sampled: ", nrow(X_final), " obs | ",
          "Pos=", sum(y_final == 1), " / Neg=", sum(y_final == 0))

} else if (best_technique == "upsample") {
  set.seed(42)
  y_fac    <- factor(ifelse(y_train_num == 1, "Yes", "No"), levels = c("No", "Yes"))
  us_final <- caret::upSample(x = as.data.frame(X_train_best),
                               y = y_fac, yname = "pobre")
  X_final  <- us_final |> select(-pobre)
  y_final  <- as.numeric(us_final$pobre == "Yes")
  message("  Up-sampled: ", nrow(X_final), " obs | ",
          "Pos=", sum(y_final == 1), " / Neg=", sum(y_final == 0))

} else if (best_technique == "smote") {
  set.seed(42)
  y_fac       <- factor(ifelse(y_train_num == 1, "Yes", "No"), levels = c("No", "Yes"))
  df_smote_in <- bind_cols(as.data.frame(X_train_best), pobre = y_fac)
  df_smote_out <- smote_con_factores_lpm(df_smote_in, var_y = "pobre",
                                          k = 5L, over_ratio = 1)
  X_final  <- df_smote_out |> select(-pobre)
  y_final  <- as.numeric(df_smote_out$pobre == "Yes")
  message("  SMOTE: ", nrow(X_final), " obs | ",
          "Pos=", sum(y_final == 1), " / Neg=", sum(y_final == 0))
}

df_final <- cbind(as.data.frame(X_final), pobre = y_final)

message("\nAjustando modelo final (", nrow(df_final), " observaciones)...")
t0_f <- proc.time()
set.seed(42)
model_final <- suppressWarnings(lm(pobre ~ ., data = df_final, weights = wts_final))
t1_f <- proc.time()
message("  Completado en ", round((t1_f - t0_f)[["elapsed"]], 1), " seg")

s <- summary(model_final)
cat(sprintf("\nR² = %.4f  |  R² ajustado = %.4f  |  Coeficientes = %d\n",
            s$r.squared, s$adj.r.squared, length(coef(model_final))))
cat(sprintf("Residual SE = %.4f\n", s$sigma))

# ─── Threshold ───────────────────────────────────────────────────────────────
# Para baseline y class_weights: los datos de entrenamiento no están rebalanceados,
# la distribución in-sample refleja la distribución real → optimizamos sobre todo
# el train (optimista pero consistente con el objetivo).
# Para técnicas de resampling: la distribución del modelo final no corresponde
# a la distribución real → usamos el promedio de thresholds del CV-5.
if (best_technique %in% c("baseline", "class_weights")) {
  probs_eval <- suppressWarnings(
    predict(model_final, newdata = as.data.frame(X_train_best))
  )
  probs_eval <- pmax(pmin(probs_eval, 1), 0)
  best_row   <- buscar_th_lpm(probs_eval, y_train_num)
  best_th    <- best_row$threshold
  eval_label <- "in-sample (train)"
} else {
  best_th <- round(
    bal_summary |> filter(technique == best_technique) |> pull(th_mean), 2
  )
  probs_eval <- suppressWarnings(
    predict(model_final, newdata = as.data.frame(X_train_best))
  )
  probs_eval <- pmax(pmin(probs_eval, 1), 0)
  eval_label <- "promedio th CV-5 (resampled)"
}

message("  Threshold: ", best_th, "  (", eval_label, ")")

f1_eval <- f1_score_lpm(probs_eval, y_train_num, best_th)
pr_eval <- pr_rc_lpm(probs_eval,   y_train_num, best_th)

message("  F1 (", eval_label, "): ", round(f1_eval,         4))
message("  Precision:             ", round(pr_eval$precision, 4))
message("  Recall:                ", round(pr_eval$recall,    4))

roc_obj <- pROC::roc(
  response  = y_train_num,
  predictor = probs_eval,
  quiet     = TRUE
)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC: ", round(auc_roc, 4))

# =============================================================================
# SECCIÓN 9: GRÁFICOS DE DIAGNÓSTICO
# =============================================================================
message("\n== Generando gráficos ==")

# --- Gráfico 1: F1 CV-5 por subconjunto de variables ---
p_feat <- feat_summary |>
  mutate(is_best = subset == BEST_SUBSET) |>
  ggplot(aes(x = reorder(subset, F1_mean), y = F1_mean, fill = is_best)) +
  geom_col(alpha = 0.85, show.legend = FALSE) +
  geom_errorbar(aes(ymin = F1_mean - F1_sd, ymax = F1_mean + F1_sd),
                width = 0.3, color = "gray40") +
  geom_text(aes(label = sprintf("%.4f", F1_mean)),
            hjust = -0.15, size = 3.5) +
  scale_fill_manual(values = c("FALSE" = "#BDBDBD", "TRUE" = "#534AB7")) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = paste0("LPM: F1 CV-5 por subconjunto de variables — ", MODEL_ID),
    subtitle = paste0("Mejor: ", BEST_SUBSET, " (", BEST_N, " vars) | F1 = ",
                      round(BEST_F1_FEAT, 4)),
    x        = "Subconjunto",
    y        = "F1 promedio CV-5",
    caption  = paste0("Barras de error = ±1 SD entre folds. Azul = mejor subconjunto.",
                      "\nVariables rankeadas por importancia de permutación RF_006.")
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "features_comparison.png"),
       p_feat, width = 7, height = 5, dpi = 150)
message("  Guardado: features_comparison.png")

# --- Gráfico 2: F1 CV-5 por técnica de balanceo ---
p_bal <- bal_summary |>
  mutate(is_best = technique == best_technique) |>
  ggplot(aes(x = reorder(technique, F1_mean), y = F1_mean, fill = is_best)) +
  geom_col(alpha = 0.85, show.legend = FALSE) +
  geom_errorbar(aes(ymin = F1_mean - F1_sd, ymax = F1_mean + F1_sd),
                width = 0.3, color = "gray40") +
  geom_text(aes(label = sprintf("%.4f", F1_mean)),
            hjust = -0.15, size = 3.5) +
  scale_fill_manual(values = c("FALSE" = "#BDBDBD", "TRUE" = "#534AB7")) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = paste0("LPM: F1 CV-5 por técnica de balanceo — ", MODEL_ID),
    subtitle = paste0("Subconjunto: ", BEST_SUBSET, " | Mejor técnica: ",
                      best_technique, " | F1 = ", round(best_f1_bal, 4)),
    x        = "Técnica de balanceo",
    y        = "F1 promedio CV-5",
    caption  = "Barras de error = ±1 SD entre folds. Azul = mejor técnica."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "balance_comparison.png"),
       p_bal, width = 7, height = 5, dpi = 150)
message("  Guardado: balance_comparison.png")

# --- Gráfico 3: Threshold vs F1 / Precision / Recall ---
th_full <- map_dfr(seq(0.05, 0.95, by = 0.01), function(th) {
  prc <- pr_rc_lpm(probs_eval, y_train_num, th)
  tibble(
    threshold = th,
    F1        = f1_score_lpm(probs_eval, y_train_num, th),
    Precision = prc$precision,
    Recall    = prc$recall
  )
})

p_threshold <- th_full |>
  filter(!is.na(F1)) |>
  pivot_longer(cols = c(F1, Precision, Recall), names_to = "metrica") |>
  ggplot(aes(x = threshold, y = value, color = metrica)) +
  geom_line(linewidth = 0.9) +
  geom_vline(xintercept = best_th, linetype = "dashed", color = "#534AB7") +
  geom_vline(xintercept = 0.5,    linetype = "dotted",  color = "gray40") +
  annotate("text", x = best_th + 0.02, y = 0.08,
           label = sprintf("th* = %.2f", best_th),
           hjust = 0, size = 3.2, color = "#534AB7") +
  scale_color_manual(
    values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3")
  ) +
  labs(
    title    = paste0("LPM: métricas por threshold — ", MODEL_ID),
    subtitle = sprintf("Subconjunto: %s | Técnica: %s | eval: %s",
                       BEST_SUBSET, best_technique, eval_label),
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

# --- Gráfico 4: Curva ROC ---
roc_df <- tibble(
  fpr = 1 - roc_obj$specificities,
  tpr = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.72, y = 0.65,
           label = "Clasificador aleatorio",
           size = 3, color = "gray50", angle = 35) +
  annotate("text", x = 0.55, y = 0.15,
           label = sprintf("AUC-ROC = %.4f", auc_roc),
           size = 3.5, color = "#534AB7") +
  scale_x_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  labs(
    title    = paste0("LPM: curva ROC — ", MODEL_ID),
    subtitle = sprintf("Subconjunto: %s | Técnica: %s | eval: %s",
                       BEST_SUBSET, best_technique, eval_label),
    x        = "Tasa de Falsos Positivos  (1 − Specificity)",
    y        = "Tasa de Verdaderos Positivos  (Recall)",
    caption  = "AUC calculado sobre predicciones in-sample del modelo final."
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc.png"),
       p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc.png")

# --- Gráfico 5: Curva Precision-Recall ---
n_pos        <- sum(y_train_num == 1)
n_neg        <- sum(y_train_num == 0)
prev_rate    <- n_pos / (n_pos + n_neg)
pr_curve_df  <- tibble(
  TP        = roc_obj$sensitivities * n_pos,
  FP        = (1 - roc_obj$specificities) * n_neg,
  recall    = roc_obj$sensitivities,
  precision = TP / (TP + FP)
) |> filter(is.finite(precision), is.finite(recall))

pt_opt <- tibble(recall = pr_eval$recall, precision = pr_eval$precision)

p_prcurve <- ggplot(pr_curve_df, aes(x = recall, y = precision)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_hline(yintercept = prev_rate, linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.85, y = prev_rate + 0.015,
           label = sprintf("Clasificador aleatorio (%.0f%%)", prev_rate * 100),
           size = 3, color = "gray50") +
  geom_point(data = pt_opt, aes(x = recall, y = precision),
             color = "#E84855", size = 3) +
  annotate("text",
           x     = pr_eval$recall - 0.1,
           y     = pr_eval$precision + 0.02,
           label = sprintf("th=%.2f\nF1=%.4f", best_th, f1_eval),
           size  = 3, color = "#E84855") +
  labs(
    title    = paste0("LPM: curva Precision-Recall — ", MODEL_ID),
    subtitle = sprintf("AUC-ROC: %.4f | Subconjunto: %s | Técnica: %s",
                       auc_roc, BEST_SUBSET, best_technique),
    x        = "Recall  (fracción de pobres capturada)",
    y        = "Precision  (fracción de predichos pobres que lo son)",
    caption  = "Punto rojo = threshold seleccionado."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve.png"),
       p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve.png")

# --- Gráfico 6: Top 20 coeficientes del modelo final ---
coefs_df <- broom::tidy(model_final) |>
  filter(term != "(Intercept)", !is.na(estimate)) |>
  mutate(abs_estimate = abs(estimate)) |>
  arrange(desc(abs_estimate))

cat(sprintf("\n━━━ Top 20 predictores por magnitud del efecto — %s ━━━\n", MODEL_ID))
print(coefs_df |> slice_head(n = 20) |>
        select(term, estimate, std.error, p.value), n = 20)

p_coefs <- coefs_df |>
  slice_head(n = 20) |>
  mutate(
    term      = fct_reorder(term, estimate),
    direccion = if_else(estimate > 0, "Aumenta P(pobre)", "Reduce P(pobre)")
  ) |>
  ggplot(aes(x = estimate, y = term, color = direccion)) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(xmin = estimate - 1.96 * std.error,
        xmax = estimate + 1.96 * std.error),
    width = 0.3, orientation = "y"
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  scale_color_manual(
    values = c("Aumenta P(pobre)" = "#E84855", "Reduce P(pobre)" = "#2196F3")
  ) +
  labs(
    title    = paste0("LPM: efectos marginales sobre P(pobre=1) — ", MODEL_ID),
    subtitle = paste0("Top 20 coeficientes | Subconjunto: ", BEST_SUBSET,
                      " | Técnica: ", best_technique),
    x        = "Efecto marginal (puntos porcentuales)",
    y        = NULL,
    color    = NULL,
    caption  = "Intervalos de confianza al 95%. Modelo ajustado sobre todo el training set."
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top")

ggsave(file.path(dir_model, "coefs.png"),
       p_coefs, width = 9, height = 6, dpi = 150)
message("  Guardado: coefs.png")

# --- Guardar diagnósticos completos ---
saveRDS(
  list(
    feat_df        = feat_df,
    feat_summary   = feat_summary,
    best_subset    = BEST_SUBSET,
    best_n         = BEST_N,
    best_f1_feat   = BEST_F1_FEAT,
    bal_df         = bal_df,
    bal_summary    = bal_summary,
    best_technique = best_technique,
    best_f1_bal    = best_f1_bal,
    model_final    = model_final,
    probs_eval     = probs_eval,
    roc_obj        = roc_obj,
    auc_roc        = auc_roc,
    best_th        = best_th,
    eval_label     = eval_label,
    f1_eval        = f1_eval,
    precision_eval = pr_eval$precision,
    recall_eval    = pr_eval$recall,
    fold_ids       = fold_ids,
    feat_order     = feat_order
  ),
  file.path(dir_model, "diagnostics.rds")
)
message("  Guardado: diagnostics.rds")

# =============================================================================
# SECCIÓN 10: SUBMISSION PARA KAGGLE
# =============================================================================
message("\n========================================")
message("Generando submission para Kaggle")
message("========================================")

probs_test <- suppressWarnings(
  predict(model_final, newdata = as.data.frame(X_test_best))
)
probs_test <- pmax(pmin(probs_test, 1), 0)
pred_test  <- as.integer(probs_test >= best_th)

submission <- tibble(id = ids, pobre = pred_test)

sub_name <- sprintf("LPM_003_%s_%s_th%03.0f.csv",
                    BEST_SUBSET, best_technique, best_th * 100)
sub_path <- file.path(dir_subs, sub_name)
write_csv(submission, sub_path)

message("  Guardado: ", sub_path)
message("  Threshold: ", best_th,
        " | Pobres predichos: ", sum(pred_test),
        " (", round(mean(pred_test) * 100, 1), "%)")
message("  → Subir este archivo a Kaggle y anotar el public F1 en model_registry.csv")

# =============================================================================
# SECCIÓN 11: REGISTRO EN model_registry.csv
# =============================================================================
message("\n========================================")
message("Registro en model_registry.csv")
message("========================================")

nueva_fila <- tibble(
  model_id           = MODEL_ID,
  fecha              = Sys.Date(),
  autor              = AUTOR,
  algoritmo          = "LPM",
  n_features         = BEST_N,
  imbalance_strategy = best_technique,
  cv_folds           = K_FOLDS,
  cv_F1              = round(best_f1_bal,       4),
  cv_Precision       = round(pr_eval$precision,  4),
  cv_Recall          = round(pr_eval$recall,     4),
  auc_roc            = round(auc_roc,            4),
  kaggle_public_F1   = NA_real_,
  threshold          = best_th,
  notas              = paste0(
    "LPM OLS. Fase1: feat selection CV-5 (top20/40/60/all desde RF_006). ",
    "Mejor subset: ", BEST_SUBSET, " (", BEST_N, " vars) F1_CV5=",
    round(BEST_F1_FEAT, 4), ". ",
    "Fase2: comparacion ", length(TECHNIQUES), " tecnicas balanceo CV-5. ",
    "Mejor: ", best_technique, " F1_bal=", round(best_f1_bal, 4), ". ",
    "th=", best_th, " AUC=", round(auc_roc, 4), ". ",
    "Importancia vars desde RF_006/feature_matrix.csv."
  ),
  cp              = NA_real_,
  maxdepth        = NA_real_,
  train_F1        = round(f1_eval,             4),
  train_Precision = round(pr_eval$precision,   4),
  train_Recall    = round(pr_eval$recall,      4)
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

cat("\n-- Fila registrada en model_registry.csv --\n")
print(nueva_fila)

# =============================================================================
# RESUMEN FINAL
# =============================================================================
cat("\n")
cat(strrep("═", 64), "\n")
cat("  ", MODEL_ID, " — RESUMEN FINAL\n")
cat(strrep("═", 64), "\n")
cat(sprintf("  Subconjunto óptimo : %-8s (%d variables)\n",  BEST_SUBSET, BEST_N))
cat(sprintf("  F1 Fase 1 (feat)   : %.4f\n", BEST_F1_FEAT))
cat(sprintf("  Técnica óptima     : %s\n", best_technique))
cat(sprintf("  F1 Fase 2 (bal)    : %.4f  [métrica CV-5 honesta]\n", best_f1_bal))
cat(sprintf("  Threshold          : %.2f\n", best_th))
cat(sprintf("  AUC-ROC            : %.4f\n", auc_roc))
cat(sprintf("  Submission         : %s\n",   sub_path))
cat(strrep("═", 64), "\n\n")
cat("  → Subir la submission a Kaggle y completar kaggle_public_F1\n")
cat("    en 02_outputs/model_registry.csv\n\n")

# nolint end
