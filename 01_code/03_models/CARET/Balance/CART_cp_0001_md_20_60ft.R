# nolint start
#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/CART_RF_top60_balance.R
#==============================================================================
# ALGORITMO : CART (rpart) — Top-60 vars de RF + Balanceo de clases
#
# QUÉ HACE ESTE SCRIPT:
#   compara 5 técnicas de balanceo
#   sobre el mejor CART encontrado (CART_RF_top60, cp=1e-04, th=0.25):
#     (a) baseline      — sin corrección, threshold óptimo CV-OOF
#     (b) class_weights — rpart parms=list(prior=c(w_no,w_si))
#     (c) downsample    — caret::downSample en el fold de train
#     (d) upsample      — caret::upSample en el fold de train
#     (e) smote         — themis::smote en el fold de train (k=5)
#
# HIPERPARÁMETROS FIJOS (mejores encontrados en CART_RF_features.R):
#   cp        = 1e-04
#   maxdepth  = 20
#   Variables = top-60 por importancia permutación RF_001
#
#   Rebalanceo aplicado SOLO al fold de entrenamiento.
#   Fold de validación siempre con distribución real (~20% pobres).
#   Threshold se optimiza sobre predicciones OOF del fold de validación.
#   Folds estratificados con caret::createFolds().
#
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#   02_outputs/models/RandomForest/RF_001/feature_matrix.csv
#
# OUTPUTS:
#   02_outputs/models/CARTs/CART_RF_top60_balance/cv_balance_results.csv
#   02_outputs/models/CARTs/CART_RF_top60_balance/cv_balance_summary.csv
#   02_outputs/models/CARTs/CART_RF_top60_balance/balance_comparison.png
#   02_outputs/models/CARTs/CART_RF_top60_balance/threshold.png
#   02_outputs/models/CARTs/CART_RF_top60_balance/roc.png
#   02_outputs/models/CARTs/CART_RF_top60_balance/prcurve.png
#   02_outputs/models/CARTs/CART_RF_top60_balance/diagnostics.rds
#   03_submissions/CART_RF_top60_balance_*.csv
#   02_outputs/model_registry.csv  (append)
#
#==============================================================================

# =============================================================================
# SECCIÓN 0: PAQUETES
# =============================================================================
library(tidyverse)
library(rpart)
library(caret)
library(pROC)
library(themis)
library(fs)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
AUTOR      <- "Natalia"
MODEL_BASE <- "Balance"
K_FOLDS    <- 5L
N_TOP      <- 60L
CP_FIXED   <- 1e-04
MAXDEPTH   <- 20L
TECHNIQUES <- c("baseline", "class_weights", "downsample", "upsample", "smote")

dir_model <- file.path("02_outputs/models/CARTs", MODEL_BASE)
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model, recurse = TRUE)
fs::dir_create(dir_subs,  recurse = TRUE)
fs::dir_create("02_outputs", recurse = TRUE)

set.seed(42)

# Función F1 manual (sin MLmetrics — evita problemas DLL Windows)
f1_manual <- function(obs, pred, positive = "Pobre") {
  TP <- sum(obs == positive & pred == positive)
  FP <- sum(obs != positive & pred == positive)
  FN <- sum(obs == positive & pred != positive)
  if ((2 * TP + FP + FN) == 0) return(NA_real_)
  2 * TP / (2 * TP + FP + FN)
}

buscar_th <- function(probs, y_bin, grid = seq(0.05, 0.95, by = 0.01)) {
  map_dfr(grid, function(th) {
    pred_c <- factor(if_else(probs >= th, "Pobre", "NoPobre"),
                     levels = c("NoPobre", "Pobre"))
    obs_c  <- factor(if_else(y_bin == 1, "Pobre", "NoPobre"),
                     levels = c("NoPobre", "Pobre"))
    tibble(threshold = th, F1 = f1_manual(obs_c, pred_c))
  }) |>
    filter(!is.na(F1)) |>
    slice_max(F1, n = 1, with_ties = FALSE)
}

# =============================================================================
# SECCIÓN 2: CARGAR DATOS
# =============================================================================
message("== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")

niveles <- c("0", "1")
y_train <- factor(train$pobre, levels = niveles)

message("  train: ", nrow(train), " x ", ncol(train))
pct_pobre <- round(mean(train$pobre) * 100, 1)
message("  Balance: ", pct_pobre, "% pobres / ", round(100 - pct_pobre, 1), "% no pobres")

# =============================================================================
# SECCIÓN 3: PREPROCESAMIENTO PARA CART
# =============================================================================
preparar_X_cart <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>
    mutate(zona   = as.numeric(as.character(zona)),
           ciudad = as.factor(ciudad),
           dpto   = as.factor(dpto)) |>
    mutate(across(where(is.character), as.factor))
}

X_full_train <- preparar_X_cart(train)
X_full_test  <- preparar_X_cart(test)

# Alinear niveles ciudad y dpto entre train y test
for (v in c("ciudad", "dpto")) {
  all_levels          <- union(levels(X_full_train[[v]]), levels(X_full_test[[v]]))
  X_full_train[[v]]   <- factor(X_full_train[[v]], levels = all_levels)
  X_full_test[[v]]    <- factor(X_full_test[[v]],  levels = all_levels)
}

cols_comunes <- intersect(names(X_full_train), names(X_full_test))
X_full_train <- X_full_train |> select(all_of(cols_comunes))
X_full_test  <- X_full_test  |> select(all_of(cols_comunes))

# =============================================================================
# SECCIÓN 4: VARIABLES DE RF (top-60)
# =============================================================================
message("\n== Cargando importancia RF (top-", N_TOP, ") ==")

rf_imp <- read_csv(
  "02_outputs/models/RandomForest/RF_006/feature_matrix.csv",
  show_col_types = FALSE
) |>
  filter(importancia > 0) |>
  arrange(rank) |>
  head(N_TOP)

VARS_SEL <- rf_imp$variable
# Mantener solo las que existen en X_full_train y X_full_test
VARS_SEL <- intersect(VARS_SEL, intersect(names(X_full_train), names(X_full_test)))

message("  Variables seleccionadas efectivamente: ", length(VARS_SEL))
cat("  Top 10:", paste(head(VARS_SEL, 10), collapse = ", "), "\n")

X_train_sel <- X_full_train |> select(all_of(VARS_SEL))
X_test_sel  <- X_full_test  |> select(all_of(VARS_SEL))

# =============================================================================
# SECCIÓN 5: SMOTE CON FACTORES
# =============================================================================
smote_con_factores <- function(df_in, var_y, k = 5, over_ratio = 1) {
  factor_cols <- setdiff(names(df_in)[sapply(df_in, is.factor)], var_y)
  factor_lvls <- lapply(factor_cols, function(col) levels(df_in[[col]]))
  names(factor_lvls) <- factor_cols

  df_num <- df_in |> mutate(across(all_of(factor_cols), as.integer))
  df_out <- themis::smote(df_num, var = var_y, k = k, over_ratio = over_ratio)

  for (col in factor_cols) {
    lvls     <- factor_lvls[[col]]
    int_vals <- as.integer(round(df_out[[col]]))
    int_vals <- pmax(1L, pmin(length(lvls), int_vals))
    df_out[[col]] <- factor(lvls[int_vals], levels = lvls)
  }
  df_out
}

# =============================================================================
# SECCIÓN 6: FOLDS ESTRATIFICADOS
# =============================================================================
set.seed(42)
fold_ids <- caret::createFolds(y = y_train, k = K_FOLDS, list = FALSE)

message("\nFolds estratificados:")
for (k in 1:K_FOLDS)
  message(sprintf("  Fold %d: %.1f%% pobres", k,
                  mean(train$pobre[fold_ids == k] == 1) * 100))

# =============================================================================
# SECCIÓN 7: CV-5 POR TÉCNICA DE BALANCEO
# =============================================================================
message("\n========================================")
message("CV-5: comparando ", length(TECHNIQUES), " técnicas de balanceo")
message("  cp=", CP_FIXED, " | maxdepth=", MAXDEPTH,
        " | variables: ", length(VARS_SEL), " (top-", N_TOP, ")")
message("========================================")

balance_resultados <- vector("list", K_FOLDS)
t0_bal <- Sys.time()

for (k in seq_len(K_FOLDS)) {
  message("\n-- Fold ", k, " / ", K_FOLDS, " --")

  idx_val   <- which(fold_ids == k)
  idx_train <- which(fold_ids != k)

  X_fold_tr  <- X_train_sel[idx_train, ]
  X_fold_val <- X_train_sel[idx_val, ]
  y_fold_tr  <- y_train[idx_train]
  y_fold_val <- y_train[idx_val]

  fold_bal_res <- map_dfr(TECHNIQUES, function(tech) {
    seed_tech <- 42 * k + match(tech, TECHNIQUES) * 1000

    # ----- Preparar datos según técnica -----
    X_tr_use <- X_fold_tr
    y_tr_use <- y_fold_tr
    parms_rpart <- NULL   # para class_weights

    if (tech == "class_weights") {
      # rpart acepta parms=list(prior=c(p_no, p_si)) para ajustar el peso
      # de cada clase en el cálculo de Gini. Pesos inversos a la frecuencia.
      n_yes <- sum(y_fold_tr == "1")
      n_no  <- sum(y_fold_tr == "0")
      total <- n_yes + n_no
      w_yes <- (n_no  / total)
      w_no  <- (n_yes / total)
      # prior debe sumar 1 y tener nombres que coincidan con los niveles de y
      parms_rpart <- list(prior = c("0" = w_no, "1" = w_yes))
      message(sprintf("    [class_weights] prior No=%.4f  Si=%.4f", w_no, w_yes))

    } else if (tech == "downsample") {
      set.seed(seed_tech)
      ds       <- caret::downSample(x = as.data.frame(X_fold_tr),
                                     y = y_fold_tr, yname = "pobre")
      X_tr_use <- ds |> select(-pobre)
      y_tr_use <- ds$pobre
      message(sprintf("    [downsample] %d obs | 1=%d / 0=%d",
                      nrow(X_tr_use), sum(y_tr_use == "1"), sum(y_tr_use == "0")))

    } else if (tech == "upsample") {
      set.seed(seed_tech)
      us       <- caret::upSample(x = as.data.frame(X_fold_tr),
                                   y = y_fold_tr, yname = "pobre")
      X_tr_use <- us |> select(-pobre)
      y_tr_use <- us$pobre
      message(sprintf("    [upsample] %d obs | 1=%d / 0=%d",
                      nrow(X_tr_use), sum(y_tr_use == "1"), sum(y_tr_use == "0")))

    } else if (tech == "smote") {
      set.seed(seed_tech)
      df_in    <- bind_cols(as.data.frame(X_fold_tr), pobre = y_fold_tr)
      df_out   <- smote_con_factores(df_in, var_y = "pobre", k = 5, over_ratio = 1)
      X_tr_use <- df_out |> select(-pobre)
      y_tr_use <- df_out$pobre
      message(sprintf("    [smote] %d obs | 1=%d / 0=%d",
                      nrow(X_tr_use), sum(y_tr_use == "1"), sum(y_tr_use == "0")))
    }
    # tech == "baseline": X_tr_use y y_tr_use sin modificar

    # Entrenar CART con los datos (posiblemente rebalanceados)
    train_fold <- bind_cols(X_tr_use, pobre = y_tr_use)

    set.seed(seed_tech)
    cart_k <- rpart(
      pobre ~ .,
      data    = train_fold,
      method  = "class",
      parms   = parms_rpart,   # NULL para baseline/down/up/smote
      control = rpart.control(
        cp        = CP_FIXED,
        maxdepth  = MAXDEPTH,
        minsplit  = 20,
        minbucket = 7
      )
    )

    # Predecir probabilidades en fold de validación (distribución real)
    prob_val <- predict(cart_k, newdata = X_fold_val, type = "prob")[, "1"]
    best_row <- buscar_th(prob_val, as.integer(as.character(y_fold_val)))

    tibble(fold = k, technique = tech,
           F1 = best_row$F1, th = best_row$threshold)
  })

  message("  Técnica         | F1_val | th:")
  for (i in seq_len(nrow(fold_bal_res))) {
    message(sprintf("    %-15s → %.4f | %.2f",
                    fold_bal_res$technique[i],
                    fold_bal_res$F1[i],
                    fold_bal_res$th[i]))
  }

  balance_resultados[[k]] <- fold_bal_res
}

t1_bal <- Sys.time()
message("\nCV completado en ",
        round(as.numeric(difftime(t1_bal, t0_bal, units = "mins")), 1), " min")

# =============================================================================
# SECCIÓN 8: SELECCIÓN DE MEJOR TÉCNICA
# =============================================================================
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

write_csv(bal_df,      file.path(dir_model, "cv_balance_results.csv"))
write_csv(bal_summary, file.path(dir_model, "cv_balance_summary.csv"))
message("  Guardado: cv_balance_results.csv, cv_balance_summary.csv")

best_technique <- bal_summary |>
  slice_max(F1_mean, n = 1, with_ties = FALSE) |>
  pull(technique)
best_f1_bal <- bal_summary |>
  filter(technique == best_technique) |>
  pull(F1_mean)
best_th_cv  <- bal_summary |>
  filter(technique == best_technique) |>
  pull(th_mean)

message("\n  Mejor técnica: ", best_technique,
        "  (F1 CV-5 = ", round(best_f1_bal, 4), ")",
        "  threshold promedio CV = ", round(best_th_cv, 2))

# =============================================================================
# SECCIÓN 9: MODELO FINAL
# =============================================================================
message("\n========================================")
message("Modelo final")
message("  Técnica: ", best_technique)
message("  cp=", CP_FIXED, " | maxdepth=", MAXDEPTH)
message("  Variables: ", length(VARS_SEL), " (top-", N_TOP, ")")
message("========================================")

X_final     <- X_train_sel
y_final     <- y_train
parms_final <- NULL

if (best_technique == "class_weights") {
  n_yes_f <- sum(y_train == "1")
  n_no_f  <- sum(y_train == "0")
  total_f <- n_yes_f + n_no_f
  w_yes_f <- n_no_f  / total_f
  w_no_f  <- n_yes_f / total_f
  parms_final <- list(prior = c("0" = w_no_f, "1" = w_yes_f))
  message(sprintf("  Prior ajustado: No=%.4f  Si=%.4f", w_no_f, w_yes_f))

} else if (best_technique == "downsample") {
  set.seed(42)
  ds_final <- caret::downSample(x = as.data.frame(X_train_sel),
                                 y = y_train, yname = "pobre")
  X_final  <- ds_final |> select(-pobre)
  y_final  <- ds_final$pobre
  message(sprintf("  Down-sampled: %d obs | 1=%d / 0=%d",
                  nrow(X_final), sum(y_final == "1"), sum(y_final == "0")))

} else if (best_technique == "upsample") {
  set.seed(42)
  us_final <- caret::upSample(x = as.data.frame(X_train_sel),
                               y = y_train, yname = "pobre")
  X_final  <- us_final |> select(-pobre)
  y_final  <- us_final$pobre
  message(sprintf("  Up-sampled: %d obs | 1=%d / 0=%d",
                  nrow(X_final), sum(y_final == "1"), sum(y_final == "0")))

} else if (best_technique == "smote") {
  set.seed(42)
  df_smote <- bind_cols(as.data.frame(X_train_sel), pobre = y_train)
  df_out   <- smote_con_factores(df_smote, var_y = "pobre", k = 5, over_ratio = 1)
  X_final  <- df_out |> select(-pobre)
  y_final  <- df_out$pobre
  message(sprintf("  SMOTE: %d obs | 1=%d / 0=%d",
                  nrow(X_final), sum(y_final == "1"), sum(y_final == "0")))
}

train_final_df <- bind_cols(X_final, pobre = y_final)

set.seed(42)
t0_f <- proc.time()
cart_final <- rpart(
  pobre ~ .,
  data    = train_final_df,
  method  = "class",
  parms   = parms_final,
  control = rpart.control(
    cp        = CP_FIXED,
    maxdepth  = MAXDEPTH,
    minsplit  = 20,
    minbucket = 7
  )
)
t1_f <- proc.time()
message("  Modelo final entrenado en ",
        round((t1_f - t0_f)[["elapsed"]], 1), " seg")
message("  Nodos terminales: ", sum(cart_final$frame$var == "<leaf>"))

# ---------- Threshold ----------
# CART con baseline/class_weights: usar predicciones in-sample (no tiene OOB).
# Con resampled: usar promedio de threshold CV (honesto).
# En todos los casos se usa el promedio CV como referencia honesta.
THRESHOLD <- round(best_th_cv, 2)
message("  Threshold (promedio CV-5): ", THRESHOLD)

# ---------- Probabilidades in-sample para curvas y métricas ----------
probs_train <- predict(cart_final, newdata = X_train_sel, type = "prob")[, "1"]

obs_train <- factor(if_else(as.integer(as.character(y_train)) == 1,
                             "Pobre", "NoPobre"),
                    levels = c("NoPobre", "Pobre"))

cm_train <- caret::confusionMatrix(
  factor(if_else(probs_train >= THRESHOLD, "Pobre", "NoPobre"),
         levels = c("NoPobre", "Pobre")),
  obs_train, positive = "Pobre"
)

train_F1        <- cm_train$byClass["F1"]
train_Precision <- cm_train$byClass["Precision"]
train_Recall    <- cm_train$byClass["Recall"]

cat("\n-- Métricas finales --\n")
cat(sprintf("  CV-5 F1 (honesto):     %.4f\n", best_f1_bal))
cat(sprintf("  Train F1 (optimista):  %.4f | Precision: %.4f | Recall: %.4f\n",
            train_F1, train_Precision, train_Recall))

FN <- cm_train$table["NoPobre", "Pobre"]
FP <- cm_train$table["Pobre",   "NoPobre"]
cat(sprintf("  Exclusion  (FN): %d pobres (%.1f%%)\n",
            FN, FN / sum(y_train == "1") * 100))
cat(sprintf("  Filtracion (FP): %d no-pobres (%.1f%%)\n",
            FP, FP / sum(y_train == "0") * 100))

roc_obj <- pROC::roc(
  response  = as.integer(as.character(y_train)),
  predictor = probs_train,
  quiet     = TRUE
)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC (in-sample): ", round(auc_roc, 4))

# =============================================================================
# SECCIÓN 10: GRÁFICOS
# =============================================================================
message("\n== Generando gráficos ==")

# --- Gráfico 1: F1 CV-5 por técnica ---
p_bal <- bal_summary |>
  mutate(is_best = technique == best_technique) |>
  ggplot(aes(x = reorder(technique, F1_mean), y = F1_mean, fill = is_best)) +
  geom_col(alpha = 0.85, show.legend = FALSE) +
  geom_errorbar(aes(ymin = F1_mean - F1_sd, ymax = F1_mean + F1_sd),
                width = 0.3, color = "gray40") +
  scale_fill_manual(values = c("FALSE" = "#BDBDBD", "TRUE" = "#1D9E75")) +
  coord_flip() +
  labs(
    title    = paste0("CART RF top-", N_TOP, ": F1 CV-5 por técnica de balanceo"),
    subtitle = sprintf("Mejor: %s | F1 CV-5 = %.4f | cp=%s | %d vars",
                       best_technique, best_f1_bal, CP_FIXED, length(VARS_SEL)),
    x        = "Técnica de balanceo",
    y        = "F1 promedio CV-5",
    caption  = "Barras de error = ±1 SD entre folds. Verde = mejor técnica."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "balance_comparison.png"),
       p_bal, width = 7, height = 5, dpi = 150)
message("  Guardado: balance_comparison.png")

# --- Gráfico 2: Threshold vs métricas (in-sample) ---
th_full <- map_dfr(seq(0.05, 0.95, by = 0.01), function(th) {
  pred_c <- factor(if_else(probs_train >= th, "Pobre", "NoPobre"),
                   levels = c("NoPobre", "Pobre"))
  cm_th  <- caret::confusionMatrix(pred_c, obs_train, positive = "Pobre")
  tibble(threshold = th,
         F1        = cm_th$byClass["F1"],
         Precision = cm_th$byClass["Precision"],
         Recall    = cm_th$byClass["Recall"])
})

th_label_x     <- if_else(THRESHOLD > 0.80, THRESHOLD - 0.04, THRESHOLD + 0.02)
th_label_hjust <- if_else(THRESHOLD > 0.80, 1, 0)

p_threshold <- th_full |>
  filter(!is.na(F1)) |>
  pivot_longer(c(F1, Precision, Recall), names_to = "metrica") |>
  ggplot(aes(threshold, value, color = metrica)) +
  geom_line(linewidth = 0.9) +
  geom_vline(xintercept = THRESHOLD, linetype = "dashed", color = "#1D9E75") +
  geom_vline(xintercept = 0.50,      linetype = "dotted", color = "gray40") +
  annotate("text", x = th_label_x, y = 0.08,
           label = sprintf("th = %.2f\n(CV promedio)", THRESHOLD),
           hjust = th_label_hjust, size = 3.2, color = "#085041") +
  annotate("text", x = 0.52, y = 0.20,
           label = "th = 0.5\n(default)",
           hjust = 0, size = 3.0, color = "gray50") +
  scale_color_manual(
    values = c(F1 = "#1D9E75", Precision = "#E84855", Recall = "#2196F3"),
    labels = c(F1 = "F1", Precision = "Precision", Recall = "Recall")
  ) +
  labs(
    title    = paste0("CART RF top-", N_TOP, ": métricas por threshold"),
    subtitle = sprintf("técnica=%s | cp=%s | %d vars | in-sample (optimista)",
                       best_technique, CP_FIXED, length(VARS_SEL)),
    x = "Threshold", y = "Metrica", color = NULL,
    caption  = "Discontinua = threshold CV-5 promedio. Punteada = 0.5."
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

ggsave(file.path(dir_model, "threshold.png"),
       p_threshold, width = 8, height = 5, dpi = 150)
message("  Guardado: threshold.png")

# --- Gráfico 3: ROC ---
roc_df <- tibble(fpr = 1 - roc_obj$specificities,
                 tpr = roc_obj$sensitivities)

p_roc <- ggplot(roc_df, aes(fpr, tpr)) +
  geom_line(color = "#1D9E75", linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.72, y = 0.65, label = "Clasificador aleatorio",
           size = 3, color = "gray50", angle = 35) +
  annotate("text", x = 0.55, y = 0.15,
           label = sprintf("AUC-ROC = %.4f", auc_roc),
           size = 3.5, color = "#1D9E75") +
  scale_x_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  labs(
    title    = paste0("CART RF top-", N_TOP, ": curva ROC"),
    subtitle = sprintf("técnica=%s | cp=%s | %d vars | in-sample",
                       best_technique, CP_FIXED, length(VARS_SEL)),
    x = "Tasa de Falsos Positivos (1 - Specificity)",
    y = "Tasa de Verdaderos Positivos (Recall)",
    caption  = "AUC calculado sobre predicciones in-sample (optimista)."
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc.png"), p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc.png")

# --- Gráfico 4: Precision-Recall ---
pr_df <- tibble(
  TP        = roc_obj$sensitivities * sum(y_train == "1"),
  FP        = (1 - roc_obj$specificities) * sum(y_train == "0"),
  recall    = roc_obj$sensitivities,
  precision = TP / (TP + FP)
) |> filter(is.finite(precision), is.finite(recall))

p_pr <- ggplot(pr_df, aes(recall, precision)) +
  geom_line(color = "#1D9E75", linewidth = 0.9) +
  geom_hline(yintercept = mean(train$pobre), linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.85, y = mean(train$pobre) + 0.015,
           label = sprintf("Azar (%.0f%%)", mean(train$pobre) * 100),
           size = 3, color = "gray50") +
  geom_point(aes(x = train_Recall, y = train_Precision),
             color = "#E84855", size = 3, inherit.aes = FALSE) +
  annotate("text",
           x = train_Recall - 0.10, y = train_Precision + 0.015,
           label = sprintf("th=%.2f\nF1=%.4f", THRESHOLD, train_F1),
           size = 3, color = "#E84855") +
  labs(
    title    = paste0("CART RF top-", N_TOP, ": curva PR"),
    subtitle = sprintf("AUC-ROC: %.4f | cp=%s | %d vars | técnica: %s",
                       auc_roc, CP_FIXED, length(VARS_SEL), best_technique),
    x = "Recall (fraccion de pobres capturada)",
    y = "Precision (fraccion de predichos pobres que lo son)",
    caption  = "Punto rojo = threshold seleccionado (in-sample)."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve.png"), p_pr, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve.png")

# =============================================================================
# SECCIÓN 11: SUBMISSION
# =============================================================================
message("\n========================================")
message("Generando submission para Kaggle")
message("========================================")

probs_test <- predict(cart_final, newdata = X_test_sel, type = "prob")[, "1"]
preds_test <- as.integer(probs_test >= THRESHOLD)

message("  Threshold: ", THRESHOLD)
message("  Pobres predichos: ", sum(preds_test),
        " (", round(mean(preds_test) * 100, 1), "%)")

sub_name <- sprintf("CART_RF_top%d_%s_cp%s_th%03.0f.csv",
                    N_TOP, best_technique,
                    gsub("\\.", "", as.character(CP_FIXED)),
                    THRESHOLD * 100)
sub_path <- file.path(dir_subs, sub_name)
write_csv(tibble(id = test$id, pobre = preds_test), sub_path)
message("  Guardado: ", sub_path)
message("  -> Subir a Kaggle y anotar kaggle_public_F1 en model_registry.csv")

# =============================================================================
# SECCIÓN 12: DIAGNÓSTICOS
# =============================================================================
saveRDS(
  list(
    bal_df         = bal_df,
    bal_summary    = bal_summary,
    best_technique = best_technique,
    best_f1_bal    = best_f1_bal,
    cart_final     = cart_final,
    vars_sel       = VARS_SEL,
    cp_fixed       = CP_FIXED,
    threshold      = THRESHOLD,
    probs_train    = probs_train,
    auc_roc        = auc_roc,
    train_F1       = train_F1,
    train_Precision = train_Precision,
    train_Recall   = train_Recall,
    cv_F1          = best_f1_bal,
    fold_ids       = fold_ids
  ),
  file.path(dir_model, "diagnostics.rds")
)
message("  Guardado: diagnostics.rds")

# =============================================================================
# SECCIÓN 13: REGISTRO EN model_registry.csv
# =============================================================================
message("\n========================================")
message("Registro en model_registry.csv")
message("========================================")

MODEL_ID <- paste0("CART_RF_top", N_TOP, "_", best_technique)

nueva_fila <- tibble(
  model_id           = MODEL_ID,
  fecha              = Sys.Date(),
  autor              = AUTOR,
  algoritmo          = "CART",
  n_features         = length(VARS_SEL),
  imbalance_strategy = best_technique,
  cv_folds           = K_FOLDS,
  cv_F1              = round(best_f1_bal,     4),
  cv_Precision       = NA_real_,
  cv_Recall          = NA_real_,
  auc_roc            = round(auc_roc,         4),
  kaggle_public_F1   = NA_real_,
  threshold          = THRESHOLD,
  notas              = paste0(
    "CART top-", N_TOP, " vars RF_001. cp=", CP_FIXED,
    ", maxdepth=", MAXDEPTH, ". ",
    "Balanceo: comparacion ", length(TECHNIQUES), " tecnicas CV-5. ",
    "Mejor: ", best_technique, " F1_CV5=", round(best_f1_bal, 4), ". ",
    "Threshold=promedio th CV-5 (", round(best_th_cv, 2), "). ",
    "class_weights via parms=list(prior) en rpart."
  ),
  cp                 = CP_FIXED,
  maxdepth           = MAXDEPTH,
  train_F1           = round(train_F1,        4),
  train_Precision    = round(train_Precision, 4),
  train_Recall       = round(train_Recall,    4)
)

este_id  <- MODEL_ID
registry <- if (file.exists(reg_path)) {
  existing <- read_csv(reg_path, show_col_types = FALSE)
  existing <- existing |> filter(model_id != este_id)
  bind_rows(existing, nueva_fila)
} else {
  nueva_fila
}
write_csv(registry, reg_path)
message("  Registro actualizado: ", reg_path)

cat("\n-- Fila registrada --\n")
print(nueva_fila)

# =============================================================================
# RESUMEN FINAL
# =============================================================================
message("\n========================================")
message("RESUMEN — CART RF top-", N_TOP, " + Balanceo")
message("========================================")
message("  cp = ", CP_FIXED, " | maxdepth = ", MAXDEPTH)
message("  Variables: ", length(VARS_SEL))
message("  Técnica ganadora: ", best_technique)
message("  F1 CV-5 (honesto):    ", round(best_f1_bal, 4))
message("  F1 train (optimista): ", round(train_F1,    4))
message("  AUC-ROC (in-sample):  ", round(auc_roc,     4))
message("  Threshold:            ", THRESHOLD)
message("  Submission: ", sub_path)
message("")

# nolint end