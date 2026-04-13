# nolint start
#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/CART_RF_features.R
#==============================================================================
# ALGORITMO : CART (rpart) con variables seleccionadas por Random Forest
#
# MOTIVACION:
#   Comparar si reducir las variables de CART a las más importantes de RF
#   mejora el F1 respecto al baseline con todas las variables.
#   CART puede manejar variables correlacionadas, pero con menos variables
#   el árbol puede generalizar mejor y entrenarse más rápido.
#
# ESPECIFICACIONES (3 cortes):
#   CART_RF_top20 — top 20 variables de RF
#   CART_RF_top40 — top 40 variables de RF
#   CART_RF_top60 — top 60 variables de RF
#
# ESTRATEGIA DE CV (dos niveles):
#   CV EXTERIOR (K=5): genera predicciones OOF para optimizar threshold
#   CV INTERIOR (K=3): dentro de cada fold exterior, busca el mejor cp
#   Esto garantiza que cp Y threshold se eligen sin data leakage entre sí.
#
#   Grilla de cp: c(0.0001, 0.0005, 0.001, 0.005)
#   Grilla de threshold: seq(0.05, 0.95, by=0.01)
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#   02_outputs/models/RandomForest/RF_001/feature_matrix.csv
#
# OUTPUTS:
#   03_submissions/CART_RF_top{20,40,60}.csv
#   02_outputs/models/CARTs/CART_RF_top{20,40,60}/{threshold,roc,prcurve}.png
#   02_outputs/models/CARTs/CART_RF_top{20,40,60}/diagnostics.rds
#   02_outputs/model_registry.csv  (append)
#
#==============================================================================

# -- 0. Paquetes -------------------------------------------------------------
library(tidyverse)
library(rpart)
library(caret)
library(pROC)
library(fs)

dir_outputs_cart <- "02_outputs/models/CARTs/Features"
dir_submissions  <- "03_submissions"
registry_path    <- "02_outputs/model_registry.csv"

fs::dir_create(dir_outputs_cart, recurse = TRUE)
fs::dir_create(dir_submissions,  recurse = TRUE)
fs::dir_create("02_outputs",     recurse = TRUE)

# -- 1. Configuración --------------------------------------------------------
AUTOR      <- "Natalia"
K_EXT      <- 5                              # folds exteriores (threshold OOF)
K_INT      <- 3                              # folds interiores (tuning cp)
CP_GRID    <- c(0.0001, 0.0005, 0.001, 0.005)
MAXDEPTH   <- 20

# Función F1 
f1_manual <- function(obs, pred, positive = "1") {
  TP <- sum(obs == positive & pred == positive)
  FP <- sum(obs != positive & pred == positive)
  FN <- sum(obs == positive & pred != positive)
  if ((2 * TP + FP + FN) == 0) return(NA_real_)
  2 * TP / (2 * TP + FP + FN)
}

# -- 2. Carga de datos -------------------------------------------------------
message("== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")

niveles <- c("0", "1")
y_train <- factor(train$pobre, levels = niveles)

message("  train: ", nrow(train), " x ", ncol(train))
message("  Prevalencia pobre: ", round(mean(train$pobre == 1) * 100, 1), "%")

# -- 3. Preprocesamiento -----------------------------------------------------
preparar_X_cart <- function(df) {
  df |>
    select(-any_of(c("id", "pobre", "ciudad", "dpto"))) |>
    mutate(zona = as.numeric(as.character(zona))) |>
    mutate(across(where(is.character), as.factor))
}

X_full_train <- preparar_X_cart(train)
X_full_test  <- preparar_X_cart(test)

cols_comunes <- intersect(names(X_full_train), names(X_full_test))
X_full_train <- X_full_train |> select(all_of(cols_comunes))
X_full_test  <- X_full_test  |> select(all_of(cols_comunes))

# -- 4. Variables de RF ------------------------------------------------------
message("\n== Cargando importancia de RF ==")

rf_imp <- read_csv(
  "02_outputs/models/RandomForest/RF_001/feature_matrix.csv",
  show_col_types = FALSE
) |>
  filter(importancia > 0) |>
  filter(!variable %in% c("ciudad", "dpto")) |>
  arrange(rank)

message("  Variables con importancia > 0: ", nrow(rf_imp))

CUTS <- list(
  CART_RF_top20 = rf_imp |> head(20) |> pull(variable),
  CART_RF_top40 = rf_imp |> head(40) |> pull(variable),
  CART_RF_top60 = rf_imp |> head(60) |> pull(variable)
)

cat("\n-- Top 10 variables (RF) --\n")
print(rf_imp |> head(10) |> select(rank, variable, importancia))

# -- 5. Folds exteriores estratificados (compartidos entre los 3 modelos) ----
set.seed(42)
fold_ext <- caret::createFolds(y = y_train, k = K_EXT, list = FALSE)

message("\nFolds exteriores estratificados (semilla 42):")
for (k in 1:K_EXT) {
  prop_k <- mean(train$pobre[fold_ext == k] == 1)
  message(sprintf("  Fold %d: %.1f%% pobres", k, prop_k * 100))
}

# =============================================================================
# FUNCIÓN run_cart_rf()
# =============================================================================
run_cart_rf <- function(model_id, vars_sel) {

  message("\n", strrep("=", 62))
  message("  INICIANDO: ", model_id, " (", length(vars_sel), " variables)")

  dir_model <- file.path(dir_outputs_cart, model_id)
  fs::dir_create(dir_model, recurse = TRUE)

  # -- Seleccionar columnas disponibles en ambos datasets -------------------
  vars_ok      <- intersect(vars_sel, names(X_full_train))
  vars_ok      <- intersect(vars_ok,  names(X_full_test))
  vars_missing <- setdiff(vars_sel, vars_ok)

  if (length(vars_missing) > 0)
    message("  Vars no encontradas: ", paste(vars_missing, collapse = ", "))

  message("  Variables efectivas: ", length(vars_ok))

  X_train_sel <- X_full_train |> select(all_of(vars_ok))
  X_test_sel  <- X_full_test  |> select(all_of(vars_ok))

  train_sel <- bind_cols(X_train_sel, pobre = y_train)

  # ── CV ANIDADO ─────────────────────────────────────────────────────────────
  # Nivel exterior: genera OOF probs para threshold
  # Nivel interior: dentro de cada fold exterior, elige cp óptimo con CV-3
  # ──────────────────────────────────────────────────────────────────────────
  oof_probs <- rep(NA_real_, nrow(train_sel))
  cp_por_fold <- rep(NA_real_, K_EXT)

  message("\n--- CV anidado (", K_EXT, " x ", K_INT, ") ---")

  for (k_out in seq_len(K_EXT)) {

    idx_val   <- which(fold_ext == k_out)
    idx_train <- which(fold_ext != k_out)

    X_tr <- train_sel[idx_train, ]
    X_vl <- train_sel[idx_val, ]

    # ── CV INTERIOR: buscar mejor cp en el fold de entrenamiento ─────────
    # Para cada cp, hacemos K_INT folds dentro del fold de entrenamiento
    set.seed(42 + k_out)
    fold_int <- caret::createFolds(
      y = X_tr$pobre, k = K_INT, list = FALSE
    )

    cp_f1 <- map_dfr(CP_GRID, function(cp_val) {
      f1s <- map_dbl(seq_len(K_INT), function(k_in) {
        idx_in_val   <- which(fold_int == k_in)
        idx_in_train <- which(fold_int != k_in)

        m <- rpart(
          pobre ~ .,
          data    = X_tr[idx_in_train, ],
          method  = "class",
          control = rpart.control(cp = cp_val, maxdepth = MAXDEPTH,
                                   minsplit = 20, minbucket = 7)
        )
        pred_in <- as.character(
          predict(m, newdata = X_tr[idx_in_val, ], type = "class")
        )
        obs_in <- as.character(X_tr$pobre[idx_in_val])
        f1_manual(obs_in, pred_in, positive = "1")
      })
      tibble(cp = cp_val, f1_mean = mean(f1s, na.rm = TRUE))
    })

    best_cp_fold <- cp_f1 |>
      slice_max(f1_mean, n = 1, with_ties = FALSE) |>
      pull(cp)

    cp_por_fold[k_out] <- best_cp_fold

    # ── FOLD EXTERIOR: entrenar con mejor cp, predecir en validación ─────
    cart_k <- rpart(
      pobre ~ .,
      data    = X_tr,
      method  = "class",
      control = rpart.control(cp = best_cp_fold, maxdepth = MAXDEPTH,
                               minsplit = 20, minbucket = 7)
    )

    prob_k <- predict(cart_k, newdata = X_vl, type = "prob")
    oof_probs[idx_val] <- prob_k[, "1"]

    message(sprintf(
      "  Fold %d/%d: best_cp=%.4f | P(pobre) en [%.3f, %.3f]",
      k_out, K_EXT, best_cp_fold,
      min(prob_k[, "1"], na.rm = TRUE),
      max(prob_k[, "1"], na.rm = TRUE)
    ))
  }

  # cp final: el más frecuente entre los K_EXT folds
  # (si hay empate, el menor — árboles más profundos generalizan mejor aquí)
  cp_tabla <- sort(table(cp_por_fold), decreasing = TRUE)
  CP_FINAL <- as.numeric(names(cp_tabla)[1])

  cat(sprintf("\n  cp por fold: %s\n", paste(cp_por_fold, collapse = ", ")))
  message("  cp final (más frecuente): ", CP_FINAL)

  if (any(is.na(oof_probs))) stop("NAs en oof_probs — revisar loop CV")

  cat(sprintf("\n--- Diagnostico OOF — %s ---\n", model_id))
  print(summary(oof_probs))

  # ── THRESHOLD ÓPTIMO (sobre predicciones OOF) ──────────────────────────
  th_grid <- seq(0.05, 0.95, by = 0.01)

  th_results <- map_dfr(th_grid, function(th) {
    pred_c <- factor(if_else(oof_probs >= th, "Pobre", "NoPobre"),
                     levels = c("NoPobre", "Pobre"))
    obs_c  <- factor(if_else(as.integer(as.character(y_train)) == 1,
                              "Pobre", "NoPobre"),
                     levels = c("NoPobre", "Pobre"))
    cm <- confusionMatrix(pred_c, obs_c, positive = "Pobre")
    tibble(threshold = th,
           F1        = cm$byClass["F1"],
           Precision = cm$byClass["Precision"],
           Recall    = cm$byClass["Recall"])
  })

  best_th   <- th_results |> filter(!is.na(F1)) |>
    slice_max(F1, n = 1, with_ties = FALSE)
  THRESHOLD <- best_th$threshold

  cat(sprintf("\n--- Threshold optimo — %s ---\n", model_id))
  print(best_th)

  # Métricas CV con threshold óptimo
  cm_cv <- confusionMatrix(
    factor(if_else(oof_probs >= THRESHOLD, "Pobre", "NoPobre"),
           levels = c("NoPobre", "Pobre")),
    factor(if_else(as.integer(as.character(y_train)) == 1, "Pobre", "NoPobre"),
           levels = c("NoPobre", "Pobre")),
    positive = "Pobre"
  )

  cv_F1        <- cm_cv$byClass["F1"]
  cv_Precision <- cm_cv$byClass["Precision"]
  cv_Recall    <- cm_cv$byClass["Recall"]

  FN <- cm_cv$table["NoPobre", "Pobre"]
  FP <- cm_cv$table["Pobre",   "NoPobre"]

  cat(sprintf("  CV F1: %.4f | Precision: %.4f | Recall: %.4f\n",
              cv_F1, cv_Precision, cv_Recall))
  cat(sprintf("  Exclusion  (FN): %d pobres (%.1f%%)\n",
              FN, FN / sum(y_train == "1") * 100))
  cat(sprintf("  Filtracion (FP): %d no-pobres (%.1f%%)\n",
              FP, FP / sum(y_train == "0") * 100))

  roc_obj <- pROC::roc(
    response  = as.integer(as.character(y_train)),
    predictor = oof_probs,
    quiet     = TRUE
  )
  auc_roc <- as.numeric(pROC::auc(roc_obj))
  message("  AUC-ROC (OOF): ", round(auc_roc, 4))

  # ── MODELO FINAL (todo el train, cp elegido por CV) ──────────────────────
  message("\n--- Modelo final con cp = ", CP_FINAL, " ---")

  cart_final <- rpart(
    pobre ~ .,
    data    = train_sel,
    method  = "class",
    control = rpart.control(cp = CP_FINAL, maxdepth = MAXDEPTH,
                             minsplit = 20, minbucket = 7)
  )

  message("  Nodos terminales: ", sum(cart_final$frame$var == "<leaf>"))

  # Métricas en train (en muestra — optimistas)
  probs_tr  <- predict(cart_final, newdata = X_train_sel, type = "prob")[, "1"]
  obs_train <- factor(if_else(as.integer(as.character(y_train)) == 1,
                               "Pobre", "NoPobre"),
                       levels = c("NoPobre", "Pobre"))
  cm_train <- confusionMatrix(
    factor(if_else(probs_tr >= THRESHOLD, "Pobre", "NoPobre"),
           levels = c("NoPobre", "Pobre")),
    obs_train, positive = "Pobre"
  )
  train_F1        <- cm_train$byClass["F1"]
  train_Precision <- cm_train$byClass["Precision"]
  train_Recall    <- cm_train$byClass["Recall"]

  cat(sprintf("  Train F1: %.4f | Precision: %.4f | Recall: %.4f  (optimistas)\n",
              train_F1, train_Precision, train_Recall))

  # ── GRÁFICOS ──────────────────────────────────────────────────────────────
  th_label_x     <- if_else(THRESHOLD > 0.80, THRESHOLD - 0.04, THRESHOLD + 0.02)
  th_label_hjust <- if_else(THRESHOLD > 0.80, 1, 0)

  p_threshold <- th_results |>
    filter(!is.na(F1)) |>
    pivot_longer(c(F1, Precision, Recall), names_to = "metrica") |>
    ggplot(aes(threshold, value, color = metrica)) +
    geom_line(linewidth = 0.9) +
    geom_vline(xintercept = THRESHOLD, linetype = "dashed", color = "gray30") +
    geom_vline(xintercept = 0.50,      linetype = "dotted", color = "gray60") +
    annotate("text", x = th_label_x, y = 0.08,
             label = sprintf("th* = %.2f\n(max F1)", THRESHOLD),
             hjust = th_label_hjust, size = 3.2, color = "gray25") +
    annotate("text", x = 0.52, y = 0.20,
             label = "th = 0.5\n(default)",
             hjust = 0, size = 3.0, color = "gray50") +
    scale_color_manual(
      values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3"),
      labels = c(F1        = "F1 (balance)",
                 Precision = "Precision (menos filtracion)",
                 Recall    = "Recall (menos exclusion)")
    ) +
    labs(
      title    = paste0("CART vars RF: threshold — ", model_id),
      subtitle = sprintf("cp=%s | %d vars | CV %dx%d anidado | th*=%.2f",
                         CP_FINAL, length(vars_ok), K_EXT, K_INT, THRESHOLD),
      x = "Threshold", y = "Metrica", color = NULL,
      caption  = "Discontinua = threshold optimo CV-OOF. Punteada = default 0.5."
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top")

  ggsave(file.path(dir_model, "threshold.png"),
         p_threshold, width = 8, height = 5, dpi = 150)

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
      title    = paste0("CART vars RF: ROC — ", model_id),
      subtitle = sprintf("cp=%s | %d vars | CV-5 OOF", CP_FINAL, length(vars_ok)),
      x = "Tasa de Falsos Positivos (1 - Specificity)",
      y = "Tasa de Verdaderos Positivos (Recall)"
    ) +
    coord_equal() +
    theme_minimal(base_size = 12)

  ggsave(file.path(dir_model, "roc.png"), p_roc, width = 6, height = 6, dpi = 150)

  pr_df <- tibble(
    TP        = roc_obj$sensitivities * sum(y_train == "1"),
    FP        = (1 - roc_obj$specificities) * sum(y_train == "0"),
    recall    = roc_obj$sensitivities,
    precision = TP / (TP + FP)
  ) |> filter(is.finite(precision), is.finite(recall))

  p_pr <- ggplot(pr_df, aes(recall, precision)) +
    geom_line(color = "#534AB7", linewidth = 0.9) +
    geom_hline(yintercept = mean(train$pobre), linetype = "dashed", color = "gray60") +
    annotate("text", x = 0.85, y = mean(train$pobre) + 0.015,
             label = sprintf("Azar (%.0f%%)", mean(train$pobre) * 100),
             size = 3, color = "gray50") +
    labs(
      title    = paste0("CART vars RF: PR — ", model_id),
      subtitle = sprintf("AUC-ROC: %.4f | cp=%s | %d vars | CV-5 OOF",
                         auc_roc, CP_FINAL, length(vars_ok)),
      x = "Recall", y = "Precision"
    ) +
    theme_minimal(base_size = 12)

  ggsave(file.path(dir_model, "prcurve.png"), p_pr, width = 6, height = 5, dpi = 150)

  message("  Graficos guardados en: ", dir_model)

  # ── SUBMISSION ────────────────────────────────────────────────────────────
  probs_test <- predict(cart_final, newdata = X_test_sel, type = "prob")[, "1"]
  preds_test <- as.integer(probs_test >= THRESHOLD)

  cat(sprintf("  %d pobres predichos (%.1f%%) | threshold = %.2f\n",
              sum(preds_test), mean(preds_test) * 100, THRESHOLD))

  sub_file <- file.path(dir_submissions, paste0(model_id, ".csv"))
  write_csv(tibble(id = test$id, pobre = preds_test), sub_file)
  message("  Submission: ", sub_file)
  message("  -> Subir a Kaggle y anotar kaggle_public_F1 en model_registry.csv")

  # ── DIAGNÓSTICOS ──────────────────────────────────────────────────────────
  saveRDS(
    list(model_id = model_id, vars_usadas = vars_ok,
         n_features = length(vars_ok), cp_final = CP_FINAL,
         cp_por_fold = cp_por_fold, fold_ext = fold_ext,
         oof_probs = oof_probs, threshold = THRESHOLD,
         th_results = th_results,
         cm_cv = cm_cv, cm_train = cm_train,
         cv_F1 = cv_F1, cv_Precision = cv_Precision, cv_Recall = cv_Recall,
         train_F1 = train_F1, train_Precision = train_Precision,
         train_Recall = train_Recall, auc_roc = auc_roc),
    file.path(dir_model, "diagnostics.rds")
  )

  # ── REGISTRY ──────────────────────────────────────────────────────────────
  nueva_fila <- tibble(
    model_id           = model_id,
    fecha              = Sys.Date(),
    autor              = AUTOR,
    algoritmo          = "CART",
    n_features         = length(vars_ok),
    imbalance_strategy = "none",
    cv_folds           = K_EXT,
    cv_F1              = round(cv_F1,           4),
    cv_Precision       = round(cv_Precision,    4),
    cv_Recall          = round(cv_Recall,       4),
    auc_roc            = round(auc_roc,         4),
    kaggle_public_F1   = NA_real_,
    threshold          = THRESHOLD,
    notas              = paste0(
      "CART top-", length(vars_ok), " vars RF_001. ",
      "CV anidado ", K_EXT, "x", K_INT, ". ",
      "cp elegido por CV-3 interno=", CP_FINAL, ". ",
      "Threshold CV-5 OOF=", THRESHOLD, "."
    ),
    cp                 = CP_FINAL,
    maxdepth           = MAXDEPTH,
    train_F1           = round(train_F1,        4),
    train_Precision    = round(train_Precision, 4),
    train_Recall       = round(train_Recall,    4)
  )

  este_id  <- model_id
  registry <- if (file.exists(registry_path)) {
    existing <- read_csv(registry_path, show_col_types = FALSE)
    existing <- existing |> filter(model_id != este_id)
    bind_rows(existing, nueva_fila)
  } else {
    nueva_fila
  }
  write_csv(registry, registry_path)
  message("  Registry actualizado.")

  message(strrep("=", 62))

  tibble(
    model_id   = model_id,
    n_features = length(vars_ok),
    cp_final   = CP_FINAL,
    cv_F1      = round(cv_F1,    4),
    auc_roc    = round(auc_roc,  4),
    threshold  = THRESHOLD,
    train_F1   = round(train_F1, 4)
  )
}

# =============================================================================
# EJECUTAR LOS TRES CORTES
# =============================================================================
message("\n", strrep("*", 62))
message("  CART_RF_top20")
message(strrep("*", 62))
result_20 <- run_cart_rf("CART_RF_top20", CUTS$CART_RF_top20)

message("\n", strrep("*", 62))
message("  CART_RF_top40")
message(strrep("*", 62))
result_40 <- run_cart_rf("CART_RF_top40", CUTS$CART_RF_top40)

message("\n", strrep("*", 62))
message("  CART_RF_top60")
message(strrep("*", 62))
result_60 <- run_cart_rf("CART_RF_top60", CUTS$CART_RF_top60)

# =============================================================================
# COMPARACIÓN FINAL
# =============================================================================
comparacion <- bind_rows(result_20, result_40, result_60)

cat("\n", strrep("=", 62), "\n")
cat("  COMPARACION FINAL: CART con variables de RF\n")
cat(strrep("=", 62), "\n\n")
print(comparacion)

ganador <- comparacion |> slice_max(cv_F1, n = 1, with_ties = FALSE)
cat(sprintf(
  "\nMejor corte: %s\n  F1 CV = %.4f | AUC = %.4f | %d vars | cp = %.4f | th = %.2f\n",
  ganador$model_id, ganador$cv_F1, ganador$auc_roc,
  ganador$n_features, ganador$cp_final, ganador$threshold
))
# nolint end