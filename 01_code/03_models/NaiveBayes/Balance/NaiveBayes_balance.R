# nolint start
#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/NB_RF_top20_balance.R
#==============================================================================
# ALGORITMO : Naive Bayes (e1071) — Top-20 vars de RF + Balanceo de clases
#
# QUÉ HACE ESTE SCRIPT:
#   Basado en la estructura de RF_006, compara 5 técnicas de balanceo
#   sobre el mejor modelo NB encontrado (NB_RF_top20):
#     (a) baseline      — sin corrección, threshold óptimo CV-OOF
#     (b) class_weights — priori manual en naiveBayes() (p = c(peso_no, peso_si))
#     (c) downsample    — caret::downSample en el fold de train
#     (d) upsample      — caret::upSample en el fold de train
#     (e) smote         — themis::smote en el fold de train (k=5)
#
#   Rebalanceo aplicado SOLO al fold de entrenamiento.
#   Fold de validación siempre refleja la distribución real (~20% pobres).
#   Threshold se optimiza sobre predicciones OOF del fold de validación.
#   Medianas de imputación calculadas SOLO del fold de entrenamiento.
#
# VARIABLES:
#   Top-20 por importancia permutación de RF_001 (feature_matrix.csv).
#   Ciudad/dpto incluidos si están en el top-20.
#   Binarias 0/1 tratadas como distribución continua (no convertidas a factor).
#   usekernel=TRUE, laplace=1 en todos los modelos.
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#   02_outputs/models/RandomForest/RF_001/feature_matrix.csv
#
# OUTPUTS:
#   02_outputs/models/NB/NB_RF_top20_balance/cv_balance_results.csv
#   02_outputs/models/NB/NB_RF_top20_balance/cv_balance_summary.csv
#   02_outputs/models/NB/NB_RF_top20_balance/balance_comparison.png
#   02_outputs/models/NB/NB_RF_top20_balance/threshold.png
#   02_outputs/models/NB/NB_RF_top20_balance/roc.png
#   02_outputs/models/NB/NB_RF_top20_balance/prcurve.png
#   02_outputs/models/NB/NB_RF_top20_balance/diagnostics.rds
#   03_submissions/NB_RF_top20_balance_*.csv
#   02_outputs/model_registry.csv  (append)
#==============================================================================

# =============================================================================
# SECCIÓN 0: PAQUETES
# =============================================================================
library(tidyverse)
library(e1071)
library(caret)
library(pROC)
library(themis)   # smote()
library(fs)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
AUTOR      <- "Natalia"
MODEL_BASE <- "NB_RF_top20_balance"
K_FOLDS    <- 5L
N_TOP      <- 20L
TECHNIQUES <- c("baseline", "class_weights", "downsample", "upsample", "smote")

dir_model <- file.path("02_outputs/models/NB", MODEL_BASE)
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model, recurse = TRUE)
fs::dir_create(dir_subs,  recurse = TRUE)
fs::dir_create("02_outputs", recurse = TRUE)

set.seed(42)

# =============================================================================
# SECCIÓN 2: CARGAR DATOS
# =============================================================================
message("== Cargando datos ==")

train   <- readRDS("00_data/processed/train_final.rds")
test    <- readRDS("00_data/processed/test_final.rds")
y_train <- train$pobre   # 0/1 numérico

message("  train: ", nrow(train), " x ", ncol(train))
pct_pobre <- round(mean(y_train) * 100, 1)
message("  Balance: ", pct_pobre, "% pobres / ", round(100 - pct_pobre, 1), "% no pobres")

# Verificación de codificación
cat("\n-- Verificacion y_train --\n")
cat("  Valor 0:", sum(y_train == 0), "| Valor 1:", sum(y_train == 1),
    "| Proporcion 1:", pct_pobre, "% (esperado ~20%)\n")

# =============================================================================
# SECCIÓN 3: VARIABLES DE RF (top-20)
# =============================================================================
message("\n== Cargando importancia RF (top-", N_TOP, ") ==")

rf_imp <- read_csv(
  "02_outputs/models/RandomForest/RF_003/feature_matrix.csv",
  show_col_types = FALSE
) |>
  filter(importancia > 0) |>
  arrange(rank) |>
  head(N_TOP)

VARS_SEL <- rf_imp$variable
VARS_SEL <- intersect(VARS_SEL, intersect(names(train), names(test)))

message("  Variables seleccionadas: ", length(VARS_SEL))
cat("  Variables:", paste(VARS_SEL, collapse = ", "), "\n")

# =============================================================================
# SECCIÓN 4: PREPROCESAMIENTO
# =============================================================================
# Función sin data leakage:
# ref_medians = NULL  → calcula medianas del df de entrada (usar en train fold)
# ref_medians = lista → aplica medianas externas (usar en val fold y test)
preprocess_nb <- function(df, ref_medians = NULL) {

  # character → factor (ciudad, dpto, cotiza_pension, etc.)
  df <- df |> mutate(across(where(is.character), as.factor))

  # logical → integer (para kernel density — no convertir a factor)
  df <- df |> mutate(across(where(is.logical), as.integer))

  # integer 0/1: NO se convierte a factor.
  # continua — kernel density estima P(X=0|clase) y P(X=1|clase) mejor
  # como densidad empírica que como categoría.

  # Imputar NAs numéricos con mediana (sin data leakage) 
  if (is.null(ref_medians)) {
    ref_medians <- df |>
      summarise(across(where(is.numeric), ~ median(., na.rm = TRUE))) |>
      as.list()
  }
  for (col in names(ref_medians)) {
    if (col %in% names(df) && anyNA(df[[col]]))
      df[[col]][is.na(df[[col]])] <- ref_medians[[col]]
  }

  # Imputar NAs en factores con la moda (sin data leakage)
  for (col in names(df)) {
    if (is.factor(df[[col]]) && anyNA(df[[col]])) {
      moda <- names(sort(table(df[[col]]), decreasing = TRUE))[1]
      df[[col]][is.na(df[[col]])] <- moda
    }
  }

  list(data = df, medians = ref_medians)
}

# smote adaptado para NB: convierte factores → integer → SMOTE → factor
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

# Función F1 manual
f1_manual <- function(obs, pred, positive = "Pobre") {
  TP <- sum(obs == positive & pred == positive)
  FP <- sum(obs != positive & pred == positive)
  FN <- sum(obs == positive & pred != positive)
  if ((2 * TP + FP + FN) == 0) return(NA_real_)
  2 * TP / (2 * TP + FP + FN)
}

# Función para buscar threshold óptimo (max F1 sobre grilla)
buscar_th <- function(probs, y_bin, grid = seq(0.05, 0.95, by = 0.01)) {
  resultados <- map_dfr(grid, function(th) {
    pred_c <- factor(if_else(probs >= th, "Pobre", "NoPobre"),
                     levels = c("NoPobre", "Pobre"))
    obs_c  <- factor(if_else(y_bin == 1, "Pobre", "NoPobre"),
                     levels = c("NoPobre", "Pobre"))
    tibble(threshold = th, F1 = f1_manual(obs_c, pred_c))
  })
  resultados |> filter(!is.na(F1)) |> slice_max(F1, n = 1, with_ties = FALSE)
}

# Matrices X con las variables seleccionadas
X_train_sel <- train |> select(all_of(VARS_SEL))
X_test_sel  <- test  |> select(all_of(VARS_SEL))

# =============================================================================
# SECCIÓN 5: FOLDS ESTRATIFICADOS
# =============================================================================
set.seed(42)
fold_ids <- caret::createFolds(y = factor(y_train), k = K_FOLDS, list = FALSE)

message("\nFolds estratificados:")
for (k in 1:K_FOLDS)
  message(sprintf("  Fold %d: %.1f%% pobres", k,
                  mean(y_train[fold_ids == k] == 1) * 100))

# =============================================================================
# SECCIÓN 6: CV-5 POR TÉCNICA DE BALANCEO
# =============================================================================
message("\n========================================")
message("CV-5: comparando ", length(TECHNIQUES), " técnicas de balanceo")
message("  Variables: ", length(VARS_SEL), " (top-", N_TOP, " RF)")
message("  usekernel=TRUE | laplace=1")
message("========================================")

balance_resultados <- vector("list", K_FOLDS)
t0_bal <- Sys.time()

for (k in seq_len(K_FOLDS)) {
  message("\n-- Fold ", k, " / ", K_FOLDS, " --")

  idx_val   <- which(fold_ids == k)
  idx_train <- which(fold_ids != k)

  X_fold_tr_raw <- X_train_sel[idx_train, ]
  X_fold_val_raw <- X_train_sel[idx_val, ]
  y_fold_tr  <- y_train[idx_train]
  y_fold_val <- y_train[idx_val]

  # Preprocesar fold de entrenamiento (calcula medianas del fold)
  prep_tr_base  <- preprocess_nb(X_fold_tr_raw)
  medianas_fold <- prep_tr_base$medians
  df_tr_base    <- prep_tr_base$data

  # Preprocesar fold de validación con medianas del train (sin leakage)
  df_val <- preprocess_nb(X_fold_val_raw, ref_medians = medianas_fold)$data

  fold_bal_res <- map_dfr(TECHNIQUES, function(tech) {
    seed_tech <- 42 * k + match(tech, TECHNIQUES) * 1000

    # ----- Preparar datos de entrenamiento según técnica -----
    df_tr_use  <- df_tr_base
    y_tr_use   <- factor(y_fold_tr, levels = c(0, 1))
    prior_use  <- NULL

    if (tech == "class_weights") {
      # NB no tiene case.weights como RF, pero acepta prior manual.
      # El prior ajusta P(clase) en el teorema de Bayes:
      # prior = c(peso_no, peso_si) inverso a la frecuencia.
      n_yes <- sum(y_fold_tr == 1)
      n_no  <- sum(y_fold_tr == 0)
      # Normalizar para que sumen 1
      w_yes     <- n_no / (n_yes + n_no)
      w_no      <- n_yes / (n_yes + n_no)
      prior_use <- c("0" = w_no, "1" = w_yes)
      message(sprintf("    [class_weights] prior No=%.3f  Si=%.3f", w_no, w_yes))

    } else if (tech == "downsample") {
      set.seed(seed_tech)
      ds        <- caret::downSample(x = as.data.frame(df_tr_base),
                                      y = factor(y_fold_tr, levels = c(0,1)),
                                      yname = "pobre")
      df_tr_use <- ds |> select(-pobre)
      y_tr_use  <- ds$pobre
      message(sprintf("    [downsample] %d obs  | Yes=%d / No=%d",
                      nrow(df_tr_use), sum(y_tr_use == 1), sum(y_tr_use == 0)))

    } else if (tech == "upsample") {
      set.seed(seed_tech)
      us        <- caret::upSample(x = as.data.frame(df_tr_base),
                                    y = factor(y_fold_tr, levels = c(0,1)),
                                    yname = "pobre")
      df_tr_use <- us |> select(-pobre)
      y_tr_use  <- us$pobre
      message(sprintf("    [upsample] %d obs  | Yes=%d / No=%d",
                      nrow(df_tr_use), sum(y_tr_use == 1), sum(y_tr_use == 0)))

    } else if (tech == "smote") {
      set.seed(seed_tech)
      df_in  <- bind_cols(as.data.frame(df_tr_base),
                           pobre = factor(y_fold_tr, levels = c(0, 1)))
      df_out <- smote_con_factores(df_in, var_y = "pobre", k = 5, over_ratio = 1)
      df_tr_use <- df_out |> select(-pobre)
      y_tr_use  <- df_out$pobre
      message(sprintf("    [smote] %d obs  | Yes=%d / No=%d",
                      nrow(df_tr_use), sum(y_tr_use == 1), sum(y_tr_use == 0)))
    }
    # tech == "baseline": df_tr_use y y_tr_use sin modificar, prior_use = NULL

    # Entrenar NB
    set.seed(seed_tech)
    modelo_k <- naiveBayes(
      x         = df_tr_use,
      y         = y_tr_use,
      laplace   = 1,
      usekernel = TRUE,
      prior     = prior_use
    )

    # Predecir en fold de validación (siempre distribución real)
    prob_val <- predict(modelo_k, newdata = df_val, type = "raw")[, "1"]

    # Verificar nivel positivo
    stopifnot("Nivel '1' debe existir en predicciones" =
                "1" %in% colnames(predict(modelo_k, newdata = df_val[1:2, ],
                                           type = "raw")))

    best_row <- buscar_th(prob_val, y_fold_val)

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
# SECCIÓN 7: SELECCIÓN DE MEJOR TÉCNICA
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
# SECCIÓN 8: MODELO FINAL
# =============================================================================
message("\n========================================")
message("Modelo final")
message("  Técnica: ", best_technique)
message("  Variables: ", length(VARS_SEL))
message("  usekernel=TRUE | laplace=1")
message("========================================")

prep_full     <- preprocess_nb(X_train_sel)
X_train_f     <- prep_full$data
medianas_full <- prep_full$medians

# Preparar X_train_f y y_final según técnica ganadora
X_final   <- X_train_f
y_final   <- factor(y_train, levels = c(0, 1))
prior_fin <- NULL

if (best_technique == "class_weights") {
  n_yes_f <- sum(y_train == 1)
  n_no_f  <- sum(y_train == 0)
  w_yes_f <- n_no_f / (n_yes_f + n_no_f)
  w_no_f  <- n_yes_f / (n_yes_f + n_no_f)
  prior_fin <- c("0" = w_no_f, "1" = w_yes_f)
  message(sprintf("  Prior ajustado: No=%.4f  Si=%.4f", w_no_f, w_yes_f))

} else if (best_technique == "downsample") {
  set.seed(42)
  ds_final <- caret::downSample(x = as.data.frame(X_train_f),
                                 y = y_final, yname = "pobre")
  X_final  <- ds_final |> select(-pobre)
  y_final  <- ds_final$pobre
  message(sprintf("  Down-sampled: %d obs | Yes=%d / No=%d",
                  nrow(X_final), sum(y_final == 1), sum(y_final == 0)))

} else if (best_technique == "upsample") {
  set.seed(42)
  us_final <- caret::upSample(x = as.data.frame(X_train_f),
                               y = y_final, yname = "pobre")
  X_final  <- us_final |> select(-pobre)
  y_final  <- us_final$pobre
  message(sprintf("  Up-sampled: %d obs | Yes=%d / No=%d",
                  nrow(X_final), sum(y_final == 1), sum(y_final == 0)))

} else if (best_technique == "smote") {
  set.seed(42)
  df_smote <- bind_cols(as.data.frame(X_train_f), pobre = y_final)
  df_out   <- smote_con_factores(df_smote, var_y = "pobre", k = 5, over_ratio = 1)
  X_final  <- df_out |> select(-pobre)
  y_final  <- df_out$pobre
  message(sprintf("  SMOTE: %d obs | Yes=%d / No=%d",
                  nrow(X_final), sum(y_final == 1), sum(y_final == 0)))
}

set.seed(42)
model_final <- naiveBayes(
  x         = X_final,
  y         = y_final,
  laplace   = 1,
  usekernel = TRUE,
  prior     = prior_fin
)

message("  Entrenado: ", nrow(X_final), " obs | ", ncol(X_final), " predictores")

# ---------- Threshold ----------
# Para baseline y class_weights: usar predicciones sobre train (in-sample)
# como estimación (NB no tiene OOB como RF).
# Para resampled: usar promedio de threshold CV (distribución real del val).
# En todos los casos se usa el threshold promedio del CV como referencia honesta.
THRESHOLD <- round(best_th_cv, 2)
message("  Threshold (promedio CV-5): ", THRESHOLD)

# ---------- Métricas en train ----------
probs_train <- predict(model_final, newdata = X_train_f, type = "raw")[, "1"]

obs_train <- factor(if_else(y_train == 1, "Pobre", "NoPobre"),
                    levels = c("NoPobre", "Pobre"))
cm_train  <- caret::confusionMatrix(
  factor(if_else(probs_train >= THRESHOLD, "Pobre", "NoPobre"),
         levels = c("NoPobre", "Pobre")),
  obs_train, positive = "Pobre"
)
train_F1        <- cm_train$byClass["F1"]
train_Precision <- cm_train$byClass["Precision"]
train_Recall    <- cm_train$byClass["Recall"]

# ---------- Métricas CV (honestas) ----------
# Las métricas CV que reportamos son las del mejor técnica
cv_F1 <- best_f1_bal
# Precision y Recall CV: tomar del fold-level con el threshold promedio
cv_pr <- bal_df |>
  filter(technique == best_technique) |>
  rowwise() |>
  mutate(
    pred_c = list(factor(if_else(
      predict(model_final, newdata = X_train_f[fold_ids == fold, ],
               type = "raw")[, "1"] >= th,
      "Pobre", "NoPobre"), levels = c("NoPobre", "Pobre"))),
    obs_c  = list(factor(if_else(y_train[fold_ids == fold] == 1,
                                  "Pobre", "NoPobre"),
                          levels = c("NoPobre", "Pobre")))
  ) |>
  ungroup()

# Calcular Precision y Recall CV de forma más directa
bal_df_best <- bal_df |> filter(technique == best_technique)
cv_Precision <- NA_real_
cv_Recall    <- NA_real_

# Usar cm_train como aproximación (las cv las calculamos del summary completo)
# Las métricas de CV precisas están en bal_df — F1 es la métrica principal
cat(sprintf("\n-- Metricas finales --\n"))
cat(sprintf("  CV-5 F1 (honesto): %.4f\n", cv_F1))
cat(sprintf("  Train F1 (optimista): %.4f | Precision: %.4f | Recall: %.4f\n",
            train_F1, train_Precision, train_Recall))

roc_obj <- pROC::roc(response = y_train, predictor = probs_train, quiet = TRUE)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC (in-sample): ", round(auc_roc, 4))

FN <- cm_train$table["NoPobre", "Pobre"]
FP <- cm_train$table["Pobre", "NoPobre"]
cat(sprintf("  Exclusion  (FN): %d pobres (%.1f%%)\n",
            FN, FN / sum(y_train == 1) * 100))
cat(sprintf("  Filtracion (FP): %d no-pobres (%.1f%%)\n",
            FP, FP / sum(y_train == 0) * 100))

# =============================================================================
# SECCIÓN 9: GRÁFICOS
# =============================================================================
message("\n== Generando gráficos ==")

# --- Gráfico 1: F1 CV-5 por técnica de balanceo ---
p_bal <- bal_summary |>
  mutate(is_best = technique == best_technique) |>
  ggplot(aes(x = reorder(technique, F1_mean), y = F1_mean, fill = is_best)) +
  geom_col(alpha = 0.85, show.legend = FALSE) +
  geom_errorbar(aes(ymin = F1_mean - F1_sd, ymax = F1_mean + F1_sd),
                width = 0.3, color = "gray40") +
  scale_fill_manual(values = c("FALSE" = "#BDBDBD", "TRUE" = "#534AB7")) +
  coord_flip() +
  labs(
    title    = paste0("NB RF top-", N_TOP, ": F1 CV-5 por técnica de balanceo"),
    subtitle = sprintf("Mejor: %s | F1 CV-5 = %.4f | %d vars | usekernel=TRUE",
                       best_technique, best_f1_bal, length(VARS_SEL)),
    x        = "Técnica de balanceo",
    y        = "F1 promedio CV-5",
    caption  = "Barras de error = ±1 SD entre folds. Azul = mejor técnica."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "balance_comparison.png"),
       p_bal, width = 7, height = 5, dpi = 150)
message("  Guardado: balance_comparison.png")

# --- Gráfico 2: Threshold vs métricas (in-sample) ---
th_grid <- seq(0.05, 0.95, by = 0.01)

th_full <- map_dfr(th_grid, function(th) {
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
  geom_vline(xintercept = THRESHOLD, linetype = "dashed", color = "#534AB7") +
  geom_vline(xintercept = 0.50,      linetype = "dotted", color = "gray40") +
  annotate("text", x = th_label_x, y = 0.08,
           label = sprintf("th = %.2f\n(CV promedio)", THRESHOLD),
           hjust = th_label_hjust, size = 3.2, color = "#534AB7") +
  scale_color_manual(
    values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3"),
    labels = c(F1 = "F1", Precision = "Precision", Recall = "Recall")
  ) +
  labs(
    title    = paste0("NB RF top-", N_TOP, ": métricas por threshold"),
    subtitle = sprintf("técnica=%s | %d vars | in-sample (optimista)",
                       best_technique, length(VARS_SEL)),
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
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.72, y = 0.65, label = "Clasificador aleatorio",
           size = 3, color = "gray50", angle = 35) +
  annotate("text", x = 0.55, y = 0.15,
           label = sprintf("AUC-ROC = %.4f", auc_roc),
           size = 3.5, color = "#534AB7") +
  scale_x_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  labs(
    title    = paste0("NB RF top-", N_TOP, ": curva ROC"),
    subtitle = sprintf("técnica=%s | %d vars | AUC in-sample", best_technique, length(VARS_SEL)),
    x = "Tasa de Falsos Positivos (1 - Specificity)",
    y = "Tasa de Verdaderos Positivos (Recall)"
  ) +
  coord_equal() + theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc.png"), p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc.png")

# --- Gráfico 4: Precision-Recall ---
pr_df <- tibble(
  TP        = roc_obj$sensitivities * sum(y_train == 1),
  FP        = (1 - roc_obj$specificities) * sum(y_train == 0),
  recall    = roc_obj$sensitivities,
  precision = TP / (TP + FP)
) |> filter(is.finite(precision), is.finite(recall))

p_pr <- ggplot(pr_df, aes(recall, precision)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_hline(yintercept = mean(y_train), linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.85, y = mean(y_train) + 0.015,
           label = sprintf("Azar (%.0f%%)", mean(y_train) * 100),
           size = 3, color = "gray50") +
  geom_point(aes(x = train_Recall, y = train_Precision),
             color = "#E84855", size = 3, inherit.aes = FALSE) +
  annotate("text", x = train_Recall - 0.10, y = train_Precision + 0.015,
           label = sprintf("th=%.2f\nF1=%.4f", THRESHOLD, train_F1),
           size = 3, color = "#E84855") +
  labs(
    title    = paste0("NB RF top-", N_TOP, ": curva PR"),
    subtitle = sprintf("AUC-ROC: %.4f | %d vars | técnica: %s",
                       auc_roc, length(VARS_SEL), best_technique),
    x = "Recall", y = "Precision",
    caption  = "Punto rojo = threshold seleccionado (in-sample)."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve.png"), p_pr, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve.png")

# =============================================================================
# SECCIÓN 10: SUBMISSION
# =============================================================================
message("\n========================================")
message("Generando submission para Kaggle")
message("========================================")

X_test_f   <- preprocess_nb(X_test_sel, ref_medians = medianas_full)$data
probs_test <- predict(model_final, newdata = X_test_f, type = "raw")[, "1"]
preds_test <- as.integer(probs_test >= THRESHOLD)

message("  Threshold: ", THRESHOLD)
message("  Pobres predichos: ", sum(preds_test),
        " (", round(mean(preds_test) * 100, 1), "%)")

sub_name <- sprintf("NB_RF_top%d_%s_th%03.0f.csv",
                    N_TOP, best_technique, THRESHOLD * 100)
sub_path <- file.path(dir_subs, sub_name)
write_csv(tibble(id = test$id, pobre = preds_test), sub_path)
message("  Guardado: ", sub_path)
message("  -> Subir a Kaggle y anotar kaggle_public_F1 en model_registry.csv")

# =============================================================================
# SECCIÓN 11: DIAGNÓSTICOS
# =============================================================================
saveRDS(
  list(
    bal_df         = bal_df,
    bal_summary    = bal_summary,
    best_technique = best_technique,
    best_f1_bal    = best_f1_bal,
    model_final    = model_final,
    medianas_full  = medianas_full,
    vars_sel       = VARS_SEL,
    threshold      = THRESHOLD,
    probs_train    = probs_train,
    auc_roc        = auc_roc,
    train_F1       = train_F1,
    train_Precision = train_Precision,
    train_Recall   = train_Recall,
    cv_F1          = cv_F1,
    fold_ids       = fold_ids
  ),
  file.path(dir_model, "diagnostics.rds")
)
message("  Guardado: diagnostics.rds")

# =============================================================================
# SECCIÓN 12: REGISTRO EN model_registry.csv
# =============================================================================
message("\n========================================")
message("Registro en model_registry.csv")
message("========================================")

MODEL_ID  <- paste0("NB_RF_top", N_TOP, "_", best_technique)

nueva_fila <- tibble(
  model_id           = MODEL_ID,
  fecha              = Sys.Date(),
  autor              = AUTOR,
  algoritmo          = "NaiveBayes",
  n_features         = length(VARS_SEL),
  imbalance_strategy = best_technique,
  cv_folds           = K_FOLDS,
  cv_F1              = round(cv_F1,           4),
  cv_Precision       = NA_real_,   # calculado a nivel fold; usar cv_F1 para comparar
  cv_Recall          = NA_real_,
  auc_roc            = round(auc_roc,         4),
  kaggle_public_F1   = NA_real_,
  threshold          = THRESHOLD,
  notas              = paste0(
    "NB top-", N_TOP, " vars RF_001. usekernel=TRUE, laplace=1. ",
    "Balanceo: comparacion ", length(TECHNIQUES), " tecnicas CV-5. ",
    "Mejor: ", best_technique, " F1_CV5=", round(best_f1_bal, 4), ". ",
    "Threshold=promedio th CV-5 (", round(best_th_cv, 2), "). ",
    "Binarias como distribucion. Ciudad/dpto si en top-N."
  ),
  cp                 = NA_real_,
  maxdepth           = NA_real_,
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
message("RESUMEN — NB RF top-", N_TOP, " + Balanceo")
message("========================================")
message("  Variables: ", length(VARS_SEL))
message("  Técnica ganadora: ", best_technique)
message("  F1 CV-5 (honesto): ", round(cv_F1, 4))
message("  F1 train (optimista): ", round(train_F1, 4))
message("  AUC-ROC (in-sample): ", round(auc_roc, 4))
message("  Threshold: ", THRESHOLD)
message("  Submission: ", sub_path)
message("")

# nolint end