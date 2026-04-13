#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/CART_baseline.R
#==============================================================================
# ALGORITMO : CART — Classification and Regression Tree (rpart)
# AUTOR     : Natalia
#
# DESCRIPCIÓN:
#   Baseline de CART con todas las variables y cp = 0.001 (sin tuning de
#   hiperparámetros). El threshold de clasificación se mantiene en 0.5
#   (default). Este script sirve como punto de referencia para comparar
#   con versiones tuneadas (cp óptimo, threshold OOF, class weights).
#
# ESTRATEGIA DE VARIABLES:
#   Se usan TODAS las variables del train_final. CART puede manejar
#   variables redundantes y correlacionadas sin problema — simplemente
#   elige en cada partición la variable que más reduce el Gini, e ignora
#   las que no aportan. No hay penalización por incluir derivadas.
#
# HIPERPARÁMETROS (fijos en este script):
#   cp = 0.001     → poda moderada, árbol razonablemente profundo
#   maxdepth = 20  → prácticamente sin límite de profundidad
#   threshold = 0.5 → default, sin optimización CV-OOF
#   (Ver CART_cp0001_threshold_opt.R para versión con threshold óptimo)
#
# PREPROCESAMIENTO:
#   rpart acepta NAs nativamente (surrogate splits).
#   Solo se requiere:
#     - zona ("1"/"2" chr) → numérico
#     - ciudad y dpto (alta cardinalidad, 24+ niveles) → eliminar
#       Las variables costa_caribe, bogota, eje_cafetero ya capturan región
#     - demás character → factor
#
# OUTPUTS:
#   03_submissions/CART_baseline.csv
#   02_outputs/models/CARTs/CART_baseline.pdf   (árbol podado para visualizar)
#   02_outputs/models/CARTs/threshold_baseline.png
#   02_outputs/models/CARTs/roc_baseline.png
#   02_outputs/models/CARTs/prcurve_baseline.png
#   02_outputs/model_registry.csv               (append)
#
# COLUMNAS DE REGISTRY:
#   model_id, fecha, autor, algoritmo, n_features, imbalance_strategy,
#   cv_folds, cv_F1, cv_Precision, cv_Recall, auc_roc, kaggle_public_F1,
#   threshold, notas, cp, maxdepth, train_F1, train_Precision, train_Recall
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
MODEL_ID <- "CART_baseline_cp001"
CP       <- 0.001
MAXDEPTH <- 20
THRESHOLD <- 0.5   # default — sin optimización CV-OOF

dir_model <- "02_outputs/models/CARTs/CART_001/CART_baseline"
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model,    recurse = TRUE)
fs::dir_create(dir_subs,     recurse = TRUE)
fs::dir_create("02_outputs", recurse = TRUE)
fs::dir_create("00_data/processed", recurse = TRUE)

# -- 2. Carga de datos -------------------------------------------------------
message("== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id

message("  train: ", nrow(train), " x ", ncol(train))
message("  test:  ", nrow(test),  " x ", ncol(test))
message("  Prevalencia pobre: ",
        round(mean(train$pobre == 1) * 100, 1), "%")

# -- 3. Preprocesamiento -----------------------------------------------------
message("\n== Preprocesando variables ==")

preparar_X_cart <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>
    mutate(zona = as.numeric(zona)) |>
    select(-any_of(c("ciudad", "dpto"))) |>
    mutate(across(where(is.character), as.factor))
}

X_train <- preparar_X_cart(train)
X_test  <- preparar_X_cart(test)

niveles <- c("0", "1")
y_train <- factor(train$pobre, levels = niveles)

cols_comunes <- intersect(names(X_train), names(X_test))
solo_train   <- setdiff(names(X_train), names(X_test))
solo_test    <- setdiff(names(X_test),  names(X_train))

if (length(solo_train) > 0)
  message("  Solo en train (se eliminan): ", paste(solo_train, collapse = ", "))
if (length(solo_test) > 0)
  message("  Solo en test  (se eliminan): ", paste(solo_test,  collapse = ", "))

X_train <- X_train |> select(all_of(cols_comunes))
X_test  <- X_test  |> select(all_of(cols_comunes))

message("  Predictores finales: ", ncol(X_train))
message("  NAs en X_train (rpart los maneja nativamente): ", sum(is.na(X_train)))
message("  y_train: ", sum(y_train == "1"), " pobres (1) | ",
        sum(y_train == "0"), " no pobres (0)")

train_cart <- bind_cols(X_train, pobre = y_train)

# =============================================================================
# PASO 1 — ENTRENAR ÁRBOL BASELINE
# =============================================================================
message("\n========================================")
message("PASO 1 — Entrenando CART baseline")
message("========================================")
message("  cp = ", CP, "  |  maxdepth = ", MAXDEPTH)
message("  threshold = ", THRESHOLD, " (default, sin optimización CV-OOF)")

set.seed(42)
t0 <- Sys.time()

cart_v1 <- rpart(
  pobre ~ .,
  data    = train_cart,
  method  = "class",
  control = rpart.control(
    cp        = CP,
    maxdepth  = MAXDEPTH,
    minsplit  = 20,
    minbucket = 7
  )
)

t1 <- Sys.time()
message("  Tiempo: ",
        round(as.numeric(difftime(t1, t0, units = "secs")), 1), " seg")
message("  Nodos terminales: ", sum(cart_v1$frame$var == "<leaf>"))
message("  Variables usadas: ",
        length(unique(cart_v1$frame$var[cart_v1$frame$var != "<leaf>"])))

cp_optimo_sugerido <- cart_v1$cptable[
  which.min(cart_v1$cptable[, "xerror"]), "CP"]
message("  cp con menor error CV interno: ", round(cp_optimo_sugerido, 6))

# Visualización — copia podada para que sea legible
cat("\n-- Árbol guardado en PDF --\n")
pdf(file.path(dir_model, "CART_baseline.pdf"), width = 12, height = 10)
rpart.plot(
  prune(cart_v1, cp = 0.01),
  type          = 4,
  extra         = 104,
  under         = TRUE,
  fallen.leaves = FALSE,
  main          = "CART baseline — primeros niveles (cp=0.01 para visualizacion)",
  tweak         = 0.85
)
dev.off()
message("  Árbol guardado: ", file.path(dir_model, "CART_baseline.pdf"))

# -- Importancia de variables ------------------------------------------------
cat("\n-- Importancia de variables (top 25) --\n")
imp_df <- tibble(
  variable    = names(cart_v1$variable.importance),
  importancia = cart_v1$variable.importance
) |>
  arrange(desc(importancia)) |>
  mutate(
    pct      = importancia / sum(importancia),
    pct_acum = cumsum(pct),
    pct_fmt  = paste0(round(pct      * 100, 1), "%"),
    acum_fmt = paste0(round(pct_acum * 100, 1), "%")
  )

imp_df |>
  head(25) |>
  select(variable, pct_fmt, acum_fmt) |>
  print(n = 25)

vars_top10 <- imp_df |> head(10)               |> pull(variable)
vars_top20 <- imp_df |> head(20)               |> pull(variable)
vars_top90 <- imp_df |> filter(pct_acum<=0.90) |> pull(variable)

message("\n  Top 10 variables:")
cat("  ", paste(vars_top10, collapse = ", "), "\n")
message("  Variables con 90% de importancia: ", length(vars_top90))

saveRDS(imp_df,     "00_data/processed/cart_importancia.rds")
saveRDS(vars_top10, "00_data/processed/cart_vars_top10.rds")
saveRDS(vars_top20, "00_data/processed/cart_vars_top20.rds")
saveRDS(vars_top90, "00_data/processed/cart_vars_top90.rds")

# -- Evaluación en train (threshold = 0.5) -----------------------------------
probs_train <- predict(cart_v1, newdata = X_train, type = "prob")[, "1"]

pred_train_v1 <- factor(
  as.character(predict(cart_v1, newdata = X_train, type = "class")),
  levels = niveles
)
cm_train <- confusionMatrix(pred_train_v1, y_train, positive = "1")

TP <- cm_train$table["1", "1"]
FN <- cm_train$table["0", "1"]
FP <- cm_train$table["1", "0"]
TN <- cm_train$table["0", "0"]

train_F1        <- cm_train$byClass["F1"]
train_Precision <- cm_train$byClass["Precision"]
train_Recall    <- cm_train$byClass["Recall"]

cat("\n-- Matriz de confusión (train, threshold = 0.5) --\n")
print(cm_train$table)

cat("\n-- Métricas (train — OPTIMISTAS, el real lo da Kaggle) --\n")
cat("  F1:              ", round(train_F1,        4), "\n")
cat("  Precision:       ", round(train_Precision, 4), "\n")
cat("  Recall:          ", round(train_Recall,    4), "\n")
cat("  Accuracy:        ", round(cm_train$overall["Accuracy"], 4), "\n")
cat("  Exclusión  (FN): ", round(FN / (TP + FN) * 100, 1),
    "% de pobres clasificados como no pobres\n")
cat("  Filtración (FP): ", round(FP / (FP + TN) * 100, 1),
    "% de no pobres clasificados como pobres\n")

# =============================================================================
# PASO 2 — GRÁFICOS
# =============================================================================
message("\n========================================")
message("PASO 2 — Generando gráficos")
message("========================================")

# AUC-ROC sobre train (en muestra — optimista)
roc_obj <- pROC::roc(
  response  = as.integer(as.character(y_train)),
  predictor = probs_train,
  quiet     = TRUE
)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC (train, en muestra): ", round(auc_roc, 4))

# ── Threshold ────────────────────────────────────────────────────────────────
th_grid <- seq(0.05, 0.95, by = 0.01)

th_results <- map_dfr(th_grid, function(th) {
  pred_th <- factor(if_else(probs_train >= th, "Pobre", "NoPobre"),
                    levels = c("NoPobre", "Pobre"))
  obs_th  <- factor(if_else(as.integer(as.character(y_train)) == 1,
                             "Pobre", "NoPobre"),
                    levels = c("NoPobre", "Pobre"))
  cm_th <- confusionMatrix(pred_th, obs_th, positive = "Pobre")
  tibble(threshold = th,
         F1        = cm_th$byClass["F1"],
         Precision = cm_th$byClass["Precision"],
         Recall    = cm_th$byClass["Recall"])
})

th_max_f1 <- th_results |>
  filter(!is.na(F1)) |>
  slice_max(F1, n = 1, with_ties = FALSE) |>
  pull(threshold)

th_label_x     <- if_else(th_max_f1 > 0.80, th_max_f1 - 0.04, th_max_f1 + 0.02)
th_label_hjust <- if_else(th_max_f1 > 0.80, 1, 0)

p_threshold <- th_results |>
  filter(!is.na(F1)) |>
  pivot_longer(c(F1, Precision, Recall), names_to = "metrica") |>
  ggplot(aes(threshold, value, color = metrica)) +
  geom_line(linewidth = 0.9) +
  geom_vline(xintercept = THRESHOLD,
             linetype = "dashed", color = "gray30") +
  geom_vline(xintercept = th_max_f1,
             linetype = "dotted", color = "#534AB7", alpha = 0.7) +
  annotate("text", x = THRESHOLD + 0.02, y = 0.08,
           label = sprintf("th = %.1f\n(submission)", THRESHOLD),
           hjust = 0, size = 3.2, color = "gray25") +
  annotate("text", x = th_label_x, y = 0.22,
           label = sprintf("th = %.2f\n(max F1 train)", th_max_f1),
           hjust = th_label_hjust, size = 3.0, color = "#534AB7") +
  scale_color_manual(
    values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3"),
    labels = c(F1        = "F1 (balance)",
               Precision = "Precision (menos filtracion)",
               Recall    = "Recall (menos exclusion)")
  ) +
  labs(
    title    = paste0("CART: metricas por threshold — ", MODEL_ID),
    subtitle = paste0("cp = ", CP, " | calculado sobre train (en muestra — optimista)"),
    x = "Threshold de clasificacion", y = "Metrica", color = NULL,
    caption  = "Discontinua = threshold usado en submission (0.5). Punteada = max F1 en train."
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

ggsave(file.path(dir_model, "threshold_baseline.png"),
       p_threshold, width = 8, height = 5, dpi = 150)
message("  Guardado: threshold_baseline.png")

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
    subtitle = paste0("cp = ", CP, " | Probabilidades sobre train (en muestra)"),
    x = "Tasa de Falsos Positivos (1 - Specificity)",
    y = "Tasa de Verdaderos Positivos (Recall)",
    caption  = "AUC en muestra es optimista. Leer junto con la curva PR."
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc_baseline.png"),
       p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc_baseline.png")

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
    subtitle = sprintf("AUC-ROC: %.4f | cp = %s | train (en muestra)", auc_roc, CP),
    x = "Recall (fraccion de pobres capturada)",
    y = "Precision (fraccion de predichos pobres que lo son)",
    caption  = "La curva por encima de la linea punteada indica mejor que el azar."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve_baseline.png"),
       p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve_baseline.png")

# =============================================================================
# PASO 3 — SUBMISSION
# =============================================================================
message("\n========================================")
message("PASO 3 — Submission")
message("========================================")

pred_test_v1  <- predict(cart_v1, newdata = X_test, type = "class")
submission_v1 <- tibble(id = ids,
                         pobre = as.integer(as.character(pred_test_v1) == "1"))
sub_file <- file.path(dir_subs, "CART_baseline.csv")
write_csv(submission_v1, sub_file)

message("  Guardado: ", sub_file)
message("  Pobres predichos: ", sum(submission_v1$pobre),
        " (", round(mean(submission_v1$pobre) * 100, 1), "%)")
message("  -> Subir a Kaggle y anotar kaggle_public_F1 en model_registry.csv")

# =============================================================================
# PASO 4 — REGISTRY
# =============================================================================
message("\n========================================")
message("PASO 4 — Registro en model_registry.csv")
message("========================================")

# cv_F1, cv_Precision, cv_Recall = NA porque este baseline no usa CV-OOF
# Para versión con CV-OOF ver CART_cp0001_threshold_opt.R

nueva_fila <- tibble(
  model_id           = MODEL_ID,
  fecha              = Sys.Date(),
  autor              = AUTOR,
  algoritmo          = "CART",
  n_features         = ncol(X_train),
  imbalance_strategy = "none",
  cv_folds           = NA_integer_,
  cv_F1              = NA_real_,
  cv_Precision       = NA_real_,
  cv_Recall          = NA_real_,
  auc_roc            = round(auc_roc,         4),
  kaggle_public_F1   = 0.62,
  threshold          = THRESHOLD,
  notas              = paste0("Baseline CART. cp=", CP,
                              ", maxdepth=", MAXDEPTH,
                              ". threshold=0.5 (default, sin CV-OOF). ",
                              "Top vars: ", paste(vars_top10[1:3], collapse=", ")),
  cp                 = CP,
  maxdepth           = MAXDEPTH,
  train_F1           = round(train_F1,        4),
  train_Precision    = round(train_Precision, 4),
  train_Recall       = round(train_Recall,    4)
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
message("\n========================================")
message("RESUMEN — ", MODEL_ID)
message("========================================")
message("  cp = ", CP, " | maxdepth = ", MAXDEPTH, " | threshold = ", THRESHOLD)
message("  F1 train:    ", round(train_F1,        4), "  (optimista)")
message("  AUC-ROC:     ", round(auc_roc,         4), "  (en muestra)")
message("  Submission:  ", sub_file)
message("")
message("  Gráficos: ", dir_model)
message("    threshold_baseline.png")
message("    roc_baseline.png")
message("    prcurve_baseline.png")
message("    CART_baseline.pdf")

# nolint end