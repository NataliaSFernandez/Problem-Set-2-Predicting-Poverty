#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/CART_cp0001_threshold_opt.R
#==============================================================================
# ALGORITMO : CART — rpart
# OBJETIVO  : Usar cp=0.0001 (mejor cp del tuning anterior) + optimizar
#             threshold con CV-5 OOF (igual que LPM y NB)
#
# DIFERENCIA respecto al script baseline:
#   - cp fijado en 0.0001 (encontrado en el tuning con CV-5)
#   - threshold NO es 0.5 sino el óptimo según CV-5 OOF
#   - registry incluye tanto cv_F1 (honesto) como train_F1 (en muestra)
#
# OUTPUTS:
#   03_submissions/CART_cp0001_thopt.csv
#   02_outputs/models/CARTs/threshold_cp0001.png
#   02_outputs/models/CARTs/roc_cp0001.png
#   02_outputs/models/CARTs/prcurve_cp0001.png
#   02_outputs/model_registry.csv  (append)
#==============================================================================
# nolint start

# -- 0. Paquetes -------------------------------------------------------------
library(tidyverse)
library(rpart)
library(rpart.plot)
library(caret)
library(pROC)
library(fs)

# -- 1. Configuración --------------------------------------------------------
AUTOR    <- "Natalia"
MODEL_ID <- "CART_cp0001_thopt"
CP_OPT   <- 0.0001     # mejor cp encontrado en el tuning anterior
K_FOLDS  <- 5

dir_model <- "02_outputs/models/CARTs"
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model,    recurse = TRUE)
fs::dir_create(dir_subs,     recurse = TRUE)
fs::dir_create("02_outputs", recurse = TRUE)

# -- 2. Carga de datos -------------------------------------------------------
message("== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id

message("  train: ", nrow(train), " x ", ncol(train))
message("  Prevalencia pobre: ", round(mean(train$pobre == 1) * 100, 1), "%")

# -- 3. Preprocesamiento -----------------------------------------------------
message("\n== Preprocesando ==")

preparar_X_cart <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>
    mutate(zona = as.numeric(zona)) |>
    select(-any_of(c("ciudad", "dpto"))) |>
    mutate(across(where(is.character), as.factor))
}

X_train <- preparar_X_cart(train)
X_test  <- preparar_X_cart(test)

cols_comunes <- intersect(names(X_train), names(X_test))
X_train <- X_train |> select(all_of(cols_comunes))
X_test  <- X_test  |> select(all_of(cols_comunes))

niveles <- c("0", "1")
y_train <- factor(train$pobre, levels = niveles)
train_cart <- bind_cols(X_train, pobre = y_train)

message("  Predictores: ", ncol(X_train))
message("  y_train: ", sum(y_train == "1"), " pobres | ",
        sum(y_train == "0"), " no pobres")

# =============================================================================
# PASO 1 — ENTRENAR ÁRBOL FINAL CON cp = 0.0001
# =============================================================================
message("\n== Entrenando árbol con cp = ", CP_OPT, " ==")

set.seed(42)
t0 <- Sys.time()

cart_final <- rpart(
  pobre ~ .,
  data    = train_cart,
  method  = "class",
  control = rpart.control(
    cp        = CP_OPT,
    maxdepth  = 20,
    minsplit  = 20,
    minbucket = 7
  )
)

message("  Tiempo: ",
        round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), " seg")
message("  Nodos terminales: ", sum(cart_final$frame$var == "<leaf>"))

# =============================================================================
# PASO 2 — CV-5 OOF PARA THRESHOLD
# =============================================================================
# Se entrena el MISMO modelo (cp=0.0001) en cada fold para obtener
# predicciones OOF honestas sobre las que optimizar el threshold.
message("\n== CV-5 OOF para threshold ==")

set.seed(42)
fold_ids  <- sample(rep(1:K_FOLDS, length.out = nrow(train_cart)))
oof_probs <- rep(NA_real_, nrow(train_cart))

for (k in seq_len(K_FOLDS)) {

  idx_val   <- which(fold_ids == k)
  idx_train <- which(fold_ids != k)

  cart_k <- rpart(
    pobre ~ .,
    data    = train_cart[idx_train, ],
    method  = "class",
    control = rpart.control(cp = CP_OPT, maxdepth = 20,
                             minsplit = 20, minbucket = 7)
  )

  prob_k <- predict(cart_k, newdata = train_cart[idx_val, ], type = "prob")
  oof_probs[idx_val] <- prob_k[, "1"]

  message(sprintf("  Fold %d/%d completado", k, K_FOLDS))
}

cat("\n-- Distribución predicciones OOF --\n")
print(summary(oof_probs))

# Barrido de thresholds sobre predicciones OOF
th_grid <- seq(0.05, 0.95, by = 0.01)

th_results <- map_dfr(th_grid, function(th) {
  pred_th <- factor(if_else(oof_probs >= th, "Pobre", "NoPobre"),
                    levels = c("NoPobre", "Pobre"))
  obs_th  <- factor(if_else(as.integer(as.character(y_train)) == 1,
                             "Pobre", "NoPobre"),
                    levels = c("NoPobre", "Pobre"))
  cm <- confusionMatrix(pred_th, obs_th, positive = "Pobre")
  tibble(threshold = th,
         F1        = cm$byClass["F1"],
         Precision = cm$byClass["Precision"],
         Recall    = cm$byClass["Recall"])
})

best_th <- th_results |>
  filter(!is.na(F1)) |>
  slice_max(F1, n = 1, with_ties = FALSE)

THRESHOLD <- best_th$threshold

cat("\n-- Threshold óptimo (CV-5 OOF) --\n")
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

cat("\n-- Métricas CV-OOF --\n")
cat("  F1:        ", round(cv_F1,        4), "\n")
cat("  Precision: ", round(cv_Precision, 4), "\n")
cat("  Recall:    ", round(cv_Recall,    4), "\n")

FN <- cm_cv$table["NoPobre", "Pobre"]
FP <- cm_cv$table["Pobre",   "NoPobre"]
cat(sprintf("  Exclusión  (FN): %d pobres excluidos (%.1f%%)\n",
            FN, FN / sum(y_train == "1") * 100))
cat(sprintf("  Filtración (FP): %d no-pobres con beneficio (%.1f%%)\n",
            FP, FP / sum(y_train == "0") * 100))

# AUC-ROC sobre OOF
roc_obj <- pROC::roc(response  = as.integer(as.character(y_train)),
                      predictor = oof_probs, quiet = TRUE)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC (OOF): ", round(auc_roc, 4))

# Métricas en train (en muestra — para registry, son optimistas)
pred_train <- factor(
  as.character(predict(cart_final, newdata = X_train, type = "class")),
  levels = niveles
)
cm_train <- confusionMatrix(pred_train, y_train, positive = "1")

train_F1        <- cm_train$byClass["F1"]
train_Precision <- cm_train$byClass["Precision"]
train_Recall    <- cm_train$byClass["Recall"]

cat("\n-- Métricas en train (en muestra — optimistas) --\n")
cat("  F1:        ", round(train_F1,        4), "\n")
cat("  Precision: ", round(train_Precision, 4), "\n")
cat("  Recall:    ", round(train_Recall,    4), "\n")

# =============================================================================
# PASO 3 — GRÁFICOS
# =============================================================================
message("\n== Generando gráficos ==")

# Anotación del threshold — moverla si está muy a la derecha
th_label_x    <- if_else(THRESHOLD > 0.80, THRESHOLD - 0.04, THRESHOLD + 0.02)
th_label_hjust <- if_else(THRESHOLD > 0.80, 1, 0)

# ── Threshold ────────────────────────────────────────────────────────────────
p_threshold <- th_results |>
  filter(!is.na(F1)) |>
  pivot_longer(c(F1, Precision, Recall), names_to = "metrica") |>
  ggplot(aes(threshold, value, color = metrica)) +
  geom_line(linewidth = 0.9) +
  geom_vline(xintercept = THRESHOLD,
             linetype = "dashed", color = "gray30") +
  geom_vline(xintercept = 0.5,
             linetype = "dotted", color = "gray60") +
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
    title    = paste0("CART: metricas por threshold — ", MODEL_ID),
    subtitle = sprintf("cp = %s | %d-fold CV OOF | n = %d hogares",
                       CP_OPT, K_FOLDS, nrow(train_cart)),
    x = "Threshold de clasificacion", y = "Metrica", color = NULL,
    caption  = "Linea discontinua = threshold optimo CV-OOF. Punteada = default 0.5."
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

ggsave(file.path(dir_model, "threshold_cp0001.png"),
       p_threshold, width = 8, height = 5, dpi = 150)
message("  Guardado: threshold_cp0001.png")

# ── ROC ──────────────────────────────────────────────────────────────────────
roc_df <- tibble(fpr = 1 - roc_obj$specificities,
                 tpr = roc_obj$sensitivities)

p_roc <- ggplot(roc_df, aes(fpr, tpr)) +
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
    title    = paste0("CART: curva ROC — ", MODEL_ID),
    subtitle = sprintf("cp = %s | Predicciones CV-5 out-of-fold", CP_OPT),
    x = "Tasa de Falsos Positivos (1 - Specificity)",
    y = "Tasa de Verdaderos Positivos (Recall)"
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc_cp0001.png"),
       p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc_cp0001.png")

# ── Precision-Recall ─────────────────────────────────────────────────────────
pr_df <- tibble(
  TP        = roc_obj$sensitivities        * sum(y_train == "1"),
  FP        = (1 - roc_obj$specificities) * sum(y_train == "0"),
  recall    = roc_obj$sensitivities,
  precision = TP / (TP + FP)
) |> filter(is.finite(precision), is.finite(recall))

p_prcurve <- ggplot(pr_df, aes(recall, precision)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_hline(yintercept = mean(train$pobre),
             linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.85, y = mean(train$pobre) + 0.015,
           label = sprintf("Azar (%.0f%%)", mean(train$pobre) * 100),
           size = 3, color = "gray50") +
  labs(
    title    = paste0("CART: curva PR — ", MODEL_ID),
    subtitle = sprintf("AUC-ROC: %.4f | cp = %s | CV-5 OOF", auc_roc, CP_OPT),
    x = "Recall (fraccion de pobres capturada)",
    y = "Precision (fraccion de predichos pobres que lo son)"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve_cp0001.png"),
       p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve_cp0001.png")

# =============================================================================
# PASO 4 — SUBMISSION
# =============================================================================
message("\n== Generando submission ==")

prob_test  <- predict(cart_final, newdata = X_test, type = "prob")[, "1"]
preds_test <- as.integer(prob_test >= THRESHOLD)

cat(sprintf("  %d hogares predichos como pobres (%.1f%%) | threshold = %.2f\n",
            sum(preds_test), mean(preds_test) * 100, THRESHOLD))

sub_file <- file.path(dir_subs, "CART_cp0001_thopt.csv")
write_csv(tibble(id = ids, pobre = preds_test), sub_file)
message("  Guardado: ", sub_file)
message("  -> Subir a Kaggle y anotar public F1 en model_registry.csv")

# =============================================================================
# PASO 5 — REGISTRO EN model_registry.csv
# =============================================================================
message("\n== Actualizando model_registry.csv ==")

nueva_fila <- tibble(
  model_id         = MODEL_ID,
  fecha            = Sys.Date(),
  autor            = AUTOR,
  algoritmo        = "CART",
  n_features       = ncol(X_train),
  imbalance_strategy = "none",
  cv_folds         = K_FOLDS,
  cv_F1            = round(cv_F1,        4),
  cv_Precision     = round(cv_Precision, 4),
  cv_Recall        = round(cv_Recall,    4),
  auc_roc          = round(auc_roc,      4),
  kaggle_public_F1 = 0,67,
  threshold        = THRESHOLD,
  notas            = paste0("CART cp=", CP_OPT,
                            " (mejor del tuning CV-5). ",
                            "Threshold optimo CV-5 OOF=", THRESHOLD, "."),
  cp               = CP_OPT,
  maxdepth         = 20,
  train_F1         = round(train_F1,        4),
  train_Precision  = round(train_Precision, 4),
  train_Recall     = round(train_Recall,    4)
)


registry <- if (file.exists(reg_path)) {
  existing <- read_csv(reg_path, show_col_types = FALSE)
  existing <- existing |> filter(model_id != MODEL_ID)
  bind_rows(existing, nueva_fila)
} else {
  nueva_fila
}

write_csv(registry, reg_path)
message("  Registro actualizado: ", reg_path)

cat("\n-- Fila registrada --\n")
print(nueva_fila)

# =============================================================================
# RESUMEN
# =============================================================================
message("\n== RESUMEN — ", MODEL_ID, " ==")
message("  cp:             ", CP_OPT)
message("  Threshold OOF:  ", THRESHOLD, "  (vs 0.5 default)")
message("  F1 CV-OOF:      ", round(cv_F1,    4), "  (honesto)")
message("  F1 train:       ", round(train_F1, 4), "  (optimista)")
message("  AUC-ROC OOF:    ", round(auc_roc,  4))
message("  Submission:     ", sub_file)

# nolint end

