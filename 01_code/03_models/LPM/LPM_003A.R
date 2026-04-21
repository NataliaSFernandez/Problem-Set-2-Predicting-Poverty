#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/LPM/LPM_003A.R
#==============================================================================
#
# ALGORITMO: LPM — Especificación A + Feature selection RF + Balanceo
#
# ESPECIFICACIÓN A (sin variables compuestas):
#   Elimina las variables que son combinaciones lineales exactas de otras
#   (los índices compuestos) y conserva sus componentes originales.
#   Variables excluidas: zona, bogota, doble_ingreso, brecha_educ,
#     indice_formalidad, vulnerabilidad_lab, n_fuentes_no_lab,
#     prop_contributivo, costa_caribe, region_pacifico, eje_cafetero,
#     n_privaciones, estrato_hog, estrato_bajo, estrato_x_zona.
#   El modelo conserva ciudad, dpto, dep_*, alguno_*, proporciones individuales.
#   → Máxima granularidad; más predictores; sin rank-deficient fit.
#
# QUÉ HACE ESTE SCRIPT:
#
#   FASE 1 – Selección de variables (feature selection):
#     Filtra el ranking de RF_006/feature_matrix.csv por la especificación A
#     (elimina las variables excluidas antes de seleccionar top N).
#     Compara LPM con subconjuntos top {20, 40, 60, todos} via CV-5.
#     Selecciona el subconjunto que maximiza F1 promedio CV-5.
#
#   FASE 2 – Balanceo de clases:
#     Con los predictores óptimos de Fase 1, compara 5 técnicas via CV-5:
#       (a) baseline, (b) class_weights, (c) downsample,
#       (d) upsample, (e) smote.
#     El rebalanceo se aplica SOLO al fold de entrenamiento.
#
#   FASE 3 – Modelo final:
#     Entrena con mejor subconjunto + mejor técnica sobre TODO el train.
#     Genera submission para Kaggle y actualiza model_registry.csv.
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#   02_outputs/models/RandomForest/RF_006/feature_matrix.csv
#
# OUTPUTS:
#   02_outputs/models/LPM/LPM_003A/cv_features_results.csv
#   02_outputs/models/LPM/LPM_003A/cv_features_summary.csv
#   02_outputs/models/LPM/LPM_003A/cv_balance_results.csv
#   02_outputs/models/LPM/LPM_003A/cv_balance_summary.csv
#   02_outputs/models/LPM/LPM_003A/features_comparison.png
#   02_outputs/models/LPM/LPM_003A/balance_comparison.png
#   02_outputs/models/LPM/LPM_003A/threshold.png
#   02_outputs/models/LPM/LPM_003A/roc.png
#   02_outputs/models/LPM/LPM_003A/prcurve.png
#   02_outputs/models/LPM/LPM_003A/coefs.png
#   02_outputs/models/LPM/LPM_003A/diagnostics.rds
#   03_submissions/LPM_003A_*.csv
#   02_outputs/model_registry.csv  (append)
#
# REPRODUCIBILIDAD: semilla global 42. Correr desde la raíz del proyecto.
#==============================================================================
# nolint start

# =============================================================================
# SECCIÓN 0: PAQUETES
# =============================================================================
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,
  pROC,
  caret,
  themis,
  broom,
  fs
)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
AUTOR    <- "Jonathan"
MODEL_ID <- "LPM_003A"

set.seed(42)

dir_model     <- file.path("02_outputs/models/LPM", MODEL_ID)
dir_subs      <- "03_submissions"
reg_path      <- "02_outputs/model_registry.csv"
feat_mat_path <- "02_outputs/models/RandomForest/RF_006/feature_matrix.csv"

fs::dir_create(dir_model, recurse = TRUE)
fs::dir_create(dir_subs,  recurse = TRUE)
fs::dir_create("02_outputs", recurse = TRUE)

K_FOLDS    <- 5L
N_TOPS     <- c(20L, 40L, 60L, NA_integer_)   # NA = usar todas las vars del pool A
TECHNIQUES <- c("baseline", "class_weights", "downsample", "upsample", "smote")

# =============================================================================
# SECCIÓN 2: VARIABLES EXCLUIDAS — ESPECIFICACIÓN A
# =============================================================================
# Se eliminan las variables que son combinaciones lineales exactas de otras.
# El modelo conserva los COMPONENTES y elimina los ÍNDICES COMPUESTOS.
# Fuente: alias() del LPM.R original. Ver comentarios detallados en LPM.R.

VARS_EXCLUIR_A <- c(
  "id", "pobre",
  "zona",             # ciudadRURAL en el factor ciudad ya captura rural/urbano
  "bogota",           # duplica ciudadBOGOTA y dpto11
  "doble_ingreso",    # = 1 - sin_ocupados - un_solo_ocupado
  "brecha_educ",      # = nivel_educ_max - nivel_educ_jefe
  "indice_formalidad",    # = prop_asalariado + prop_cotiza_pension + prop_prima_serv + prop_sub_transp
  "vulnerabilidad_lab",   # = prop_domestico + prop_fam_sin_rem + prop_jornalero
  "n_fuentes_no_lab",     # = suma de los 8 indicadores alguno_*
  "prop_contributivo",    # = prop_afil_salud + prop_subsidiado
  "costa_caribe",         # suma de dummies dpto del litoral Caribe
  "region_pacifico",      # suma de dummies dpto del litoral Pacífico
  "eje_cafetero",         # suma de dummies dpto del Eje Cafetero
  "n_privaciones",        # = dep_educacion + dep_vivienda + dep_empleo_formal + dep_proteccion + dep_dependencia
  "estrato_hog", "estrato_bajo", "estrato_x_zona"   # 100% NA en test
)

# =============================================================================
# SECCIÓN 3: CARGAR DATOS
# =============================================================================
message("\n== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id

message("Train: ", nrow(train), " hogares | ", ncol(train), " columnas")
message("Test:  ", nrow(test),  " hogares | ", ncol(test),  " columnas")
message("Balance: ", round(mean(train$pobre) * 100, 1), "% pobres")

y_train_num <- as.numeric(train$pobre)

# =============================================================================
# SECCIÓN 4: FEATURE MATRIX Y SUBCONJUNTOS (Especificación A)
# =============================================================================
message("\n== Cargando feature matrix de RF_006 y aplicando especificación A ==")

feature_matrix_rf <- read_csv(feat_mat_path, show_col_types = FALSE) |>
  arrange(rank)

# Preprocesamiento: tipos de columna coherentes con lm()
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

# Alinear niveles de ciudad y dpto (unión train ∪ test)
for (v in c("ciudad", "dpto")) {
  all_levels       <- union(levels(X_train_raw[[v]]), levels(X_test_raw[[v]]))
  X_train_raw[[v]] <- factor(X_train_raw[[v]], levels = all_levels)
  X_test_raw[[v]]  <- factor(X_test_raw[[v]],  levels = all_levels)
}

# Pool de variables para especificación A:
#   1. Presentes en feature_matrix RF_006 (variables con importancia medida)
#   2. Presentes en ambos datasets (train y test)
#   3. No en VARS_EXCLUIR_A
cols_en_ambos <- intersect(names(X_train_raw), names(X_test_raw))

feat_order_A <- feature_matrix_rf |>
  filter(
    variable %in% cols_en_ambos,
    !variable %in% VARS_EXCLUIR_A
  ) |>
  arrange(rank) |>
  pull(variable)

X_train_pool <- X_train_raw[, feat_order_A, drop = FALSE]
X_test_pool  <- X_test_raw[,  feat_order_A, drop = FALSE]

P_POOL <- length(feat_order_A)
message("Pool de variables disponibles (especificación A): ", P_POOL)
message("NAs en X_train_pool: ", sum(is.na(X_train_pool)))
message("NAs en X_test_pool:  ", sum(is.na(X_test_pool)))

cat("\nTop 10 variables del pool A (por importancia RF_006):\n")
print(head(feat_order_A, 10))

# Función: crea el subconjunto de variables para top N
crear_subset <- function(n_top) {
  if (is.na(n_top)) {
    list(X_train = X_train_pool, X_test = X_test_pool,
         n = P_POOL, label = "all")
  } else {
    n_eff <- min(as.integer(n_top), P_POOL)
    vars  <- feat_order_A[seq_len(n_eff)]
    list(
      X_train = X_train_pool[, vars, drop = FALSE],
      X_test  = X_test_pool[,  vars, drop = FALSE],
      n = n_eff, label = paste0("top", n_eff)
    )
  }
}

subsets <- lapply(N_TOPS, crear_subset)
names(subsets) <- sapply(subsets, `[[`, "label")

cat("\nSubconjuntos a evaluar (especificación A):\n")
for (nm in names(subsets)) {
  cat(sprintf("  %-6s: %d variables\n", nm, subsets[[nm]]$n))
}

# =============================================================================
# SECCIÓN 5: FUNCIONES AUXILIARES
# =============================================================================

f1_score_lpm <- function(probs, y_num, th) {
  pred <- as.integer(probs >= th)
  TP   <- sum(pred == 1L & y_num == 1L)
  FP   <- sum(pred == 1L & y_num == 0L)
  FN   <- sum(pred == 0L & y_num == 1L)
  if ((2L * TP + FP + FN) == 0L) return(NA_real_)
  2L * TP / (2L * TP + FP + FN)
}

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

buscar_th_lpm <- function(probs, y_num, grid = seq(0.05, 0.95, by = 0.01)) {
  res <- map_dfr(grid, function(th) {
    tibble(threshold = th, F1 = f1_score_lpm(probs, y_num, th))
  })
  res |> filter(!is.na(F1)) |> slice_max(F1, n = 1, with_ties = FALSE)
}

smote_con_factores_lpm <- function(df_in, var_y, k = 5, over_ratio = 1) {
  factor_cols <- setdiff(names(df_in)[sapply(df_in, is.factor)], var_y)
  factor_lvls <- lapply(factor_cols, function(col) levels(df_in[[col]]))
  names(factor_lvls) <- factor_cols

  df_num <- df_in |> mutate(across(all_of(factor_cols), as.integer))
  df_out <- themis::smote(df_num, var = var_y, k = k, over_ratio = over_ratio)

  for (col in factor_cols) {
    lvls     <- factor_lvls[[col]]
    int_vals <- pmax(1L, pmin(length(lvls), as.integer(round(df_out[[col]]))))
    df_out[[col]] <- factor(lvls[int_vals], levels = lvls)
  }
  df_out
}

entrenar_fold_lpm <- function(X_tr, y_num_tr, X_val, y_num_val, wts = NULL) {
  df_tr <- cbind(as.data.frame(X_tr), pobre = y_num_tr)
  mod   <- suppressWarnings(lm(pobre ~ ., data = df_tr, weights = wts))
  probs <- suppressWarnings(predict(mod, newdata = as.data.frame(X_val)))
  probs <- pmax(pmin(probs, 1), 0)
  best  <- buscar_th_lpm(probs, y_num_val)
  list(F1 = best$F1, th = best$threshold)
}

# =============================================================================
# SECCIÓN 6: FOLDS DE CV (compartidos por Fase 1 y Fase 2)
# =============================================================================
set.seed(42)
fold_ids <- sample(rep(seq_len(K_FOLDS), length.out = nrow(X_train_pool)))
message("\nFolds CV-5 generados (semilla 42). Mismos folds para Fase 1 y Fase 2.")

# =============================================================================
# SECCIÓN 7: FASE 1 — SELECCIÓN DE VARIABLES
# =============================================================================
feat_checkpoint <- file.path(dir_model, "cv_features_results.csv")

if (file.exists(feat_checkpoint)) {
  message("\n========================================")
  message("FASE 1: cargando resultados guardados (checkpoint)")
  message("========================================")
  feat_df      <- read_csv(feat_checkpoint, show_col_types = FALSE)
  feat_summary <- read_csv(file.path(dir_model, "cv_features_summary.csv"),
                            show_col_types = FALSE)

} else {
  message("\n========================================")
  message("FASE 1: Selección de variables — Especificación A")
  message("  Subconjuntos: ", paste(names(subsets), collapse = ", "))
  message("  K = ", K_FOLDS, " folds | Métrica: F1 promedio CV-5")
  message("========================================")

  feat_resultados <- vector("list", length(subsets))
  names(feat_resultados) <- names(subsets)
  t0_feat <- Sys.time()

  for (subset_name in names(subsets)) {
    sbs <- subsets[[subset_name]]
    message("\n-- Subconjunto: ", subset_name, " (", sbs$n, " variables) --")

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
      tibble(subset = subset_name, n_vars = sbs$n, fold = k,
             F1 = res$F1, th = res$th)
    })
    feat_resultados[[subset_name]] <- fold_res
  }

  t1_feat <- Sys.time()
  message("\nFase 1 completada en ",
          round(as.numeric(difftime(t1_feat, t0_feat, units = "mins")), 1), " min")

  feat_df <- bind_rows(feat_resultados)
  feat_summary <- feat_df |>
    group_by(subset, n_vars) |>
    summarise(F1_mean = mean(F1, na.rm = TRUE),
              F1_sd   = sd(F1,   na.rm = TRUE),
              th_mean = mean(th, na.rm = TRUE),
              .groups = "drop") |>
    arrange(desc(F1_mean))

  write_csv(feat_df,      file.path(dir_model, "cv_features_results.csv"))
  write_csv(feat_summary, file.path(dir_model, "cv_features_summary.csv"))
  message("  Guardado: cv_features_results.csv, cv_features_summary.csv")
}

cat("\n-- Resultados Fase 1: F1 por subconjunto (CV-5) — Especificación A --\n")
print(feat_summary |> mutate(across(where(is.double), ~ round(., 4))))

best_feat_row <- feat_summary |> slice_max(F1_mean, n = 1, with_ties = FALSE)
BEST_SUBSET   <- best_feat_row$subset
BEST_N        <- best_feat_row$n_vars
BEST_F1_FEAT  <- best_feat_row$F1_mean

message("\n  Mejor subconjunto: ", BEST_SUBSET,
        " (", BEST_N, " vars) | F1 CV-5 = ", round(BEST_F1_FEAT, 4))

X_train_best <- subsets[[BEST_SUBSET]]$X_train
X_test_best  <- subsets[[BEST_SUBSET]]$X_test

# =============================================================================
# SECCIÓN 8: FASE 2 — TÉCNICAS DE BALANCEO
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
  bal_df      <- read_csv(bal_checkpoint, show_col_types = FALSE)
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
        n_pos <- sum(y_fold_tr == 1)
        n_neg <- sum(y_fold_tr == 0)
        wts   <- ifelse(y_fold_tr == 1, n_neg / n_pos, 1.0)

      } else if (tech == "downsample") {
        set.seed(seed_tech)
        y_fac    <- factor(ifelse(y_fold_tr == 1, "Yes", "No"), levels = c("No", "Yes"))
        ds       <- caret::downSample(x = as.data.frame(X_fold_tr),
                                      y = y_fac, yname = "pobre")
        X_tr_use <- ds |> select(-pobre)
        y_tr_use <- as.numeric(ds$pobre == "Yes")

      } else if (tech == "upsample") {
        set.seed(seed_tech)
        y_fac    <- factor(ifelse(y_fold_tr == 1, "Yes", "No"), levels = c("No", "Yes"))
        us       <- caret::upSample(x = as.data.frame(X_fold_tr),
                                    y = y_fac, yname = "pobre")
        X_tr_use <- us |> select(-pobre)
        y_tr_use <- as.numeric(us$pobre == "Yes")

      } else if (tech == "smote") {
        set.seed(seed_tech)
        y_fac  <- factor(ifelse(y_fold_tr == 1, "Yes", "No"), levels = c("No", "Yes"))
        df_in  <- bind_cols(as.data.frame(X_fold_tr), pobre = y_fac)
        df_out <- smote_con_factores_lpm(df_in, var_y = "pobre", k = 5L, over_ratio = 1)
        X_tr_use <- df_out |> select(-pobre)
        y_tr_use <- as.numeric(df_out$pobre == "Yes")
      }

      res <- entrenar_fold_lpm(
        X_tr = X_tr_use, y_num_tr = y_tr_use,
        X_val = X_fold_val, y_num_val = y_fold_val,
        wts = wts
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
    summarise(F1_mean = mean(F1, na.rm = TRUE),
              F1_sd   = sd(F1,   na.rm = TRUE),
              th_mean = mean(th, na.rm = TRUE),
              .groups = "drop") |>
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
# SECCIÓN 9: FASE 3 — MODELO FINAL
# =============================================================================
message("\n========================================")
message("FASE 3: Modelo final")
message("  Especificación: A (componentes individuales, sin índices compuestos)")
message("  Subconjunto: ", BEST_SUBSET, " (", BEST_N, " variables)")
message("  Técnica de balanceo: ", best_technique)
message("========================================")

X_final   <- X_train_best
y_final   <- y_train_num
wts_final <- NULL

if (best_technique == "class_weights") {
  n_pos_f   <- sum(y_train_num == 1)
  n_neg_f   <- sum(y_train_num == 0)
  wts_final <- ifelse(y_train_num == 1, n_neg_f / n_pos_f, 1.0)
  message("  Peso clase positiva: ", round(n_neg_f / n_pos_f, 3))

} else if (best_technique == "downsample") {
  set.seed(42)
  y_fac    <- factor(ifelse(y_train_num == 1, "Yes", "No"), levels = c("No", "Yes"))
  ds_final <- caret::downSample(x = as.data.frame(X_train_best),
                                y = y_fac, yname = "pobre")
  X_final  <- ds_final |> select(-pobre)
  y_final  <- as.numeric(ds_final$pobre == "Yes")
  message("  Down-sampled: ", nrow(X_final), " obs | Pos=", sum(y_final == 1),
          " / Neg=", sum(y_final == 0))

} else if (best_technique == "upsample") {
  set.seed(42)
  y_fac    <- factor(ifelse(y_train_num == 1, "Yes", "No"), levels = c("No", "Yes"))
  us_final <- caret::upSample(x = as.data.frame(X_train_best),
                               y = y_fac, yname = "pobre")
  X_final  <- us_final |> select(-pobre)
  y_final  <- as.numeric(us_final$pobre == "Yes")
  message("  Up-sampled: ", nrow(X_final), " obs | Pos=", sum(y_final == 1),
          " / Neg=", sum(y_final == 0))

} else if (best_technique == "smote") {
  set.seed(42)
  y_fac        <- factor(ifelse(y_train_num == 1, "Yes", "No"), levels = c("No", "Yes"))
  df_smote_in  <- bind_cols(as.data.frame(X_train_best), pobre = y_fac)
  df_smote_out <- smote_con_factores_lpm(df_smote_in, var_y = "pobre",
                                          k = 5L, over_ratio = 1)
  X_final  <- df_smote_out |> select(-pobre)
  y_final  <- as.numeric(df_smote_out$pobre == "Yes")
  message("  SMOTE: ", nrow(X_final), " obs | Pos=", sum(y_final == 1),
          " / Neg=", sum(y_final == 0))
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

# Threshold
if (best_technique %in% c("baseline", "class_weights")) {
  probs_eval <- suppressWarnings(
    predict(model_final, newdata = as.data.frame(X_train_best))
  )
  probs_eval <- pmax(pmin(probs_eval, 1), 0)
  best_row   <- buscar_th_lpm(probs_eval, y_train_num)
  best_th    <- best_row$threshold
  eval_label <- "in-sample (train)"
} else {
  best_th    <- round(bal_summary |>
                        filter(technique == best_technique) |>
                        pull(th_mean), 2)
  probs_eval <- suppressWarnings(
    predict(model_final, newdata = as.data.frame(X_train_best))
  )
  probs_eval <- pmax(pmin(probs_eval, 1), 0)
  eval_label <- "promedio th CV-5 (resampled)"
}

message("  Threshold: ", best_th, "  (", eval_label, ")")

f1_eval <- f1_score_lpm(probs_eval, y_train_num, best_th)
pr_eval <- pr_rc_lpm(probs_eval,   y_train_num, best_th)

message("  F1: ", round(f1_eval, 4), " | Precision: ", round(pr_eval$precision, 4),
        " | Recall: ", round(pr_eval$recall, 4))

roc_obj <- pROC::roc(response = y_train_num, predictor = probs_eval, quiet = TRUE)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC: ", round(auc_roc, 4))

# =============================================================================
# SECCIÓN 10: GRÁFICOS
# =============================================================================
message("\n== Generando gráficos ==")

# Gráfico 1: F1 por subconjunto de variables
p_feat <- feat_summary |>
  mutate(is_best = subset == BEST_SUBSET) |>
  ggplot(aes(x = reorder(subset, F1_mean), y = F1_mean, fill = is_best)) +
  geom_col(alpha = 0.85, show.legend = FALSE) +
  geom_errorbar(aes(ymin = F1_mean - F1_sd, ymax = F1_mean + F1_sd),
                width = 0.3, color = "gray40") +
  geom_text(aes(label = sprintf("%.4f", F1_mean)), hjust = -0.15, size = 3.5) +
  scale_fill_manual(values = c("FALSE" = "#BDBDBD", "TRUE" = "#534AB7")) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = paste0("LPM: F1 CV-5 por subconjunto — ", MODEL_ID),
    subtitle = paste0("Esp. A (componentes). Mejor: ", BEST_SUBSET,
                      " (", BEST_N, " vars) | F1 = ", round(BEST_F1_FEAT, 4)),
    x = "Subconjunto", y = "F1 promedio CV-5",
    caption = "Barras ±1 SD. Variables rankeadas por importancia RF_006 tras excluir índices compuestos."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "features_comparison.png"),
       p_feat, width = 7, height = 5, dpi = 150)
message("  Guardado: features_comparison.png")

# Gráfico 2: F1 por técnica de balanceo
p_bal <- bal_summary |>
  mutate(is_best = technique == best_technique) |>
  ggplot(aes(x = reorder(technique, F1_mean), y = F1_mean, fill = is_best)) +
  geom_col(alpha = 0.85, show.legend = FALSE) +
  geom_errorbar(aes(ymin = F1_mean - F1_sd, ymax = F1_mean + F1_sd),
                width = 0.3, color = "gray40") +
  geom_text(aes(label = sprintf("%.4f", F1_mean)), hjust = -0.15, size = 3.5) +
  scale_fill_manual(values = c("FALSE" = "#BDBDBD", "TRUE" = "#534AB7")) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = paste0("LPM: F1 CV-5 por técnica de balanceo — ", MODEL_ID),
    subtitle = paste0("Subconjunto: ", BEST_SUBSET,
                      " | Mejor: ", best_technique,
                      " | F1 = ", round(best_f1_bal, 4)),
    x = "Técnica de balanceo", y = "F1 promedio CV-5",
    caption = "Barras ±1 SD. Azul = mejor técnica."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "balance_comparison.png"),
       p_bal, width = 7, height = 5, dpi = 150)
message("  Guardado: balance_comparison.png")

# Gráfico 3: Threshold vs métricas
th_full <- map_dfr(seq(0.05, 0.95, by = 0.01), function(th) {
  prc <- pr_rc_lpm(probs_eval, y_train_num, th)
  tibble(threshold = th,
         F1        = f1_score_lpm(probs_eval, y_train_num, th),
         Precision = prc$precision, Recall = prc$recall)
})

p_threshold <- th_full |>
  filter(!is.na(F1)) |>
  pivot_longer(cols = c(F1, Precision, Recall), names_to = "metrica") |>
  ggplot(aes(x = threshold, y = value, color = metrica)) +
  geom_line(linewidth = 0.9) +
  geom_vline(xintercept = best_th, linetype = "dashed", color = "#534AB7") +
  geom_vline(xintercept = 0.5,    linetype = "dotted",  color = "gray40") +
  annotate("text", x = best_th + 0.02, y = 0.08,
           label = sprintf("th* = %.2f", best_th), hjust = 0, size = 3.2, color = "#534AB7") +
  scale_color_manual(values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3")) +
  labs(title    = paste0("LPM: métricas por threshold — ", MODEL_ID),
       subtitle = sprintf("Subconjunto: %s | Técnica: %s | eval: %s",
                          BEST_SUBSET, best_technique, eval_label),
       x = "Threshold", y = "Valor de la métrica", color = NULL) +
  theme_minimal(base_size = 12) + theme(legend.position = "top")

ggsave(file.path(dir_model, "threshold.png"),
       p_threshold, width = 8, height = 5, dpi = 150)
message("  Guardado: threshold.png")

# Gráfico 4: Curva ROC
roc_df <- tibble(fpr = 1 - roc_obj$specificities, tpr = roc_obj$sensitivities)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.55, y = 0.15,
           label = sprintf("AUC-ROC = %.4f", auc_roc), size = 3.5, color = "#534AB7") +
  scale_x_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  labs(title    = paste0("LPM: curva ROC — ", MODEL_ID),
       subtitle = sprintf("Esp. A | Subconjunto: %s | Técnica: %s",
                          BEST_SUBSET, best_technique),
       x = "Tasa de Falsos Positivos", y = "Tasa de Verdaderos Positivos") +
  coord_equal() + theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc.png"), p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc.png")

# Gráfico 5: Curva Precision-Recall
n_pos       <- sum(y_train_num == 1)
n_neg       <- sum(y_train_num == 0)
prev_rate   <- n_pos / (n_pos + n_neg)
pr_curve_df <- tibble(
  TP        = roc_obj$sensitivities * n_pos,
  FP        = (1 - roc_obj$specificities) * n_neg,
  recall    = roc_obj$sensitivities,
  precision = TP / (TP + FP)
) |> filter(is.finite(precision), is.finite(recall))

p_prcurve <- ggplot(pr_curve_df, aes(x = recall, y = precision)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_hline(yintercept = prev_rate, linetype = "dashed", color = "gray60") +
  geom_point(data = tibble(recall = pr_eval$recall, precision = pr_eval$precision),
             color = "#E84855", size = 3) +
  annotate("text", x = pr_eval$recall - 0.1, y = pr_eval$precision + 0.02,
           label = sprintf("th=%.2f\nF1=%.4f", best_th, f1_eval),
           size = 3, color = "#E84855") +
  labs(title    = paste0("LPM: curva Precision-Recall — ", MODEL_ID),
       subtitle = sprintf("AUC-ROC: %.4f | Esp. A | %s | %s",
                          auc_roc, BEST_SUBSET, best_technique),
       x = "Recall", y = "Precision") +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve.png"), p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve.png")

# Gráfico 6: Top 20 coeficientes
coefs_df <- broom::tidy(model_final) |>
  filter(term != "(Intercept)", !is.na(estimate)) |>
  mutate(abs_estimate = abs(estimate)) |>
  arrange(desc(abs_estimate))

p_coefs <- coefs_df |>
  slice_head(n = 20) |>
  mutate(term      = fct_reorder(term, estimate),
         direccion = if_else(estimate > 0, "Aumenta P(pobre)", "Reduce P(pobre)")) |>
  ggplot(aes(x = estimate, y = term, color = direccion)) +
  geom_point(size = 3) +
  geom_errorbar(aes(xmin = estimate - 1.96 * std.error,
                    xmax = estimate + 1.96 * std.error),
                width = 0.3, orientation = "y") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  scale_color_manual(values = c("Aumenta P(pobre)" = "#E84855",
                                "Reduce P(pobre)"  = "#2196F3")) +
  labs(title    = paste0("LPM: efectos marginales — ", MODEL_ID),
       subtitle = paste0("Esp. A | Top 20 coeficientes | Subconjunto: ", BEST_SUBSET),
       x = "Efecto marginal (pp)", y = NULL, color = NULL,
       caption = "IC al 95%. Modelo ajustado sobre todo el training set.") +
  theme_minimal(base_size = 11) + theme(legend.position = "top")

ggsave(file.path(dir_model, "coefs.png"), p_coefs, width = 9, height = 6, dpi = 150)
message("  Guardado: coefs.png")

# Diagnósticos
saveRDS(
  list(especificacion  = "A",
       vars_excluidas  = VARS_EXCLUIR_A,
       feat_order      = feat_order_A,
       feat_df         = feat_df,
       feat_summary    = feat_summary,
       best_subset     = BEST_SUBSET,
       best_n          = BEST_N,
       best_f1_feat    = BEST_F1_FEAT,
       bal_df          = bal_df,
       bal_summary     = bal_summary,
       best_technique  = best_technique,
       best_f1_bal     = best_f1_bal,
       model_final     = model_final,
       probs_eval      = probs_eval,
       roc_obj         = roc_obj,
       auc_roc         = auc_roc,
       best_th         = best_th,
       eval_label      = eval_label,
       f1_eval         = f1_eval,
       precision_eval  = pr_eval$precision,
       recall_eval     = pr_eval$recall,
       fold_ids        = fold_ids),
  file.path(dir_model, "diagnostics.rds")
)
message("  Guardado: diagnostics.rds")

# =============================================================================
# SECCIÓN 11: SUBMISSION PARA KAGGLE
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
sub_name   <- sprintf("LPM_003A_%s_%s_th%03.0f.csv",
                      BEST_SUBSET, best_technique, best_th * 100)
sub_path   <- file.path(dir_subs, sub_name)
write_csv(submission, sub_path)

message("  Guardado: ", sub_path)
message("  Threshold: ", best_th, " | Pobres predichos: ", sum(pred_test),
        " (", round(mean(pred_test) * 100, 1), "%)")

# =============================================================================
# SECCIÓN 12: REGISTRO EN model_registry.csv
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
  cv_F1              = round(best_f1_bal,      4),
  cv_Precision       = round(pr_eval$precision, 4),
  cv_Recall          = round(pr_eval$recall,    4),
  auc_roc            = round(auc_roc,           4),
  kaggle_public_F1   = NA_real_,
  threshold          = best_th,
  notas              = paste0(
    "LPM Esp.A (componentes, sin indices compuestos). ",
    "Fase1: feat selection CV-5 (top20/40/60/all desde RF_006 filtrado). ",
    "Mejor subset: ", BEST_SUBSET, " (", BEST_N, " vars) F1_CV5=",
    round(BEST_F1_FEAT, 4), ". ",
    "Fase2: comparacion ", length(TECHNIQUES), " tecnicas balanceo CV-5. ",
    "Mejor: ", best_technique, " F1_bal=", round(best_f1_bal, 4), ". ",
    "th=", best_th, " AUC=", round(auc_roc, 4), "."
  ),
  cp              = NA_real_,
  maxdepth        = NA_real_,
  train_F1        = round(f1_eval,           4),
  train_Precision = round(pr_eval$precision, 4),
  train_Recall    = round(pr_eval$recall,    4)
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

cat("\n", strrep("═", 64), "\n")
cat("  ", MODEL_ID, " — RESUMEN FINAL (Especificación A)\n")
cat(strrep("═", 64), "\n")
cat(sprintf("  Especificación     : A — componentes individuales\n"))
cat(sprintf("  Pool de variables  : %d (de %d en RF_006, excluidos índices)\n",
            P_POOL, nrow(feature_matrix_rf)))
cat(sprintf("  Subconjunto óptimo : %-8s (%d variables)\n", BEST_SUBSET, BEST_N))
cat(sprintf("  F1 Fase 1 (feat)   : %.4f\n", BEST_F1_FEAT))
cat(sprintf("  Técnica óptima     : %s\n", best_technique))
cat(sprintf("  F1 Fase 2 (bal)    : %.4f  [F1 CV-5 honesto]\n", best_f1_bal))
cat(sprintf("  Threshold          : %.2f\n", best_th))
cat(sprintf("  AUC-ROC            : %.4f\n", auc_roc))
cat(sprintf("  Submission         : %s\n", sub_path))
cat(strrep("═", 64), "\n\n")

# nolint end
