#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/Cart_cp_0001_maxdepth_20.R
#==============================================================================
# ALGORITMO : CART — Classification and Regression Tree (rpart)
#
# PASOS:
#   Paso 1 — Baseline sin tuning
#   Paso 2 — Gráficos: threshold, ROC, Precision-Recall
#   Paso 3 — Submission para Kaggle
#   Paso 4 — Registro en model_registry.csv
#
# THRESHOLD:
#   Se usa threshold = 0.5 (default). Las probabilidades vienen de
#   predict(type="prob") sobre el train completo.
#   Los gráficos muestran cómo varía F1/Precision/Recall con distintos
#   thresholds, pero la submission usa 0.5.
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#
# OUTPUTS:
#   03_submissions/CART_cp_0.001_maxdepth_20.csv
#   02_outputs/models/CARTs/CART_baseline.pdf
#   02_outputs/models/CARTs/threshold.png
#   02_outputs/models/CARTs/roc.png
#   02_outputs/models/CARTs/prcurve.png
#   02_outputs/models/CARTs/diagnostics.rds
#   02_outputs/model_registry.csv   (append)
#==============================================================================
# nolint start

# -- 0. Paquetes -------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,
  rpart,       # árbol de decisión
  rpart.plot,  # visualización del árbol
  caret,       # confusionMatrix
  pROC,        # roc(), auc()
  fs           # dir_create
)

# -- 1. Carga de datos -------------------------------------------------------
message("== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id

AUTOR    <- "Natalia"
MODEL_ID <- "CART_cp0001_maxdepth20"
THRESHOLD <- 0.5   # threshold por defecto — sin optimización CV

# Rutas de outputs
dir_model <- "02_outputs/models/CARTs"
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model,    recurse = TRUE)
fs::dir_create(dir_subs,     recurse = TRUE)
fs::dir_create("02_outputs", recurse = TRUE)

message("  train: ", nrow(train), " x ", ncol(train))
message("  test:  ", nrow(test),  " x ", ncol(test))
message("  Prevalencia pobre: ",
        round(mean(train$pobre == 1) * 100, 1), "%")

# -- 2. Preprocesamiento mínimo para rpart -----------------------------------
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

# Alinear columnas train/test por si difieren
cols_comunes <- intersect(names(X_train), names(X_test))
X_train <- X_train |> select(all_of(cols_comunes))
X_test  <- X_test  |> select(all_of(cols_comunes))

niveles <- c("0", "1")
y_train <- factor(train$pobre, levels = niveles)

message("  y_train: ", sum(y_train == "1"), " pobres (1) | ",
        sum(y_train == "0"), " no pobres (0)")
message("  Predictores: ", ncol(X_train))

train_cart <- bind_cols(X_train, pobre = y_train)

# =============================================================================
# PASO 1 — BASELINE SIN TUNING
# =============================================================================
message("\n========================================")
message("PASO 1 — Baseline CART (sin tuning)")
message("========================================")
message("  cp = 0.001  |  maxdepth = 20")

set.seed(42)
t0 <- Sys.time()

cart_v1 <- rpart(
  pobre ~ .,
  data    = train_cart,
  method  = "class",
  control = rpart.control(
    cp        = 0.001,
    maxdepth  = 20,
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

# Visualización del árbol — copia podada para que sea legible en el plot
cat("\n-- Árbol baseline (guardado en PDF) --\n")
pdf(file.path(dir_model, "CART_baseline.pdf"), width = 12, height = 10)
cart_plot <- prune(cart_v1, cp = 0.01)
rpart.plot(
  cart_plot,
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

# Guardar para reutilizar en NB y Logit
fs::dir_create("00_data/processed", recurse = TRUE)
saveRDS(imp_df,     "00_data/processed/cart_importancia.rds")
saveRDS(vars_top10, "00_data/processed/cart_vars_top10.rds")
saveRDS(vars_top20, "00_data/processed/cart_vars_top20.rds")
saveRDS(vars_top90, "00_data/processed/cart_vars_top90.rds")

# -- Evaluación en train con threshold = 0.5 ---------------------------------
# Probabilidades del modelo completo sobre train (para diagnóstico y gráficos)
probs_train <- predict(cart_v1, newdata = X_train, type = "prob")[, "1"]

pred_train <- factor(
  as.character(predict(cart_v1, newdata = X_train, type = "class")),
  levels = niveles
)
cm_train <- confusionMatrix(pred_train, y_train, positive = "1")

cat("\n-- Matriz de confusión (train, threshold = 0.5) --\n")
print(cm_train$table)

TP <- cm_train$table["1", "1"]
FN <- cm_train$table["0", "1"]
FP <- cm_train$table["1", "0"]
TN <- cm_train$table["0", "0"]

train_F1        <- cm_train$byClass["F1"]
train_Precision <- cm_train$byClass["Precision"]
train_Recall    <- cm_train$byClass["Recall"]

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
# PASO 2 — GRÁFICOS: THRESHOLD, ROC, PRECISION-RECALL
# =============================================================================
# Los gráficos se construyen sobre las probabilidades del modelo en train.
# NOTA: estas son probabilidades "en muestra" (el modelo vio estos datos),
# así que las métricas serán optimistas. Son útiles para visualizar el
# comportamiento del modelo, no para comparar con otros algoritmos.
message("\n========================================")
message("PASO 2 — Generando gráficos")
message("========================================")

# AUC-ROC sobre train (en muestra)
roc_obj <- pROC::roc(
  response  = as.integer(as.character(y_train)),
  predictor = probs_train,
  quiet     = TRUE
)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC (train, en muestra): ", round(auc_roc, 4))

# ── Gráfico 1: Cómo varía F1, Precision y Recall con el threshold ───────────
# Muestra el trade-off entre precisión y cobertura para distintos puntos de corte.
th_grid <- seq(0.05, 0.95, by = 0.01)

th_results <- map_dfr(th_grid, function(th) {
  pred_th <- factor(
    if_else(probs_train >= th, "Pobre", "NoPobre"),
    levels = c("NoPobre", "Pobre")
  )
  obs_th <- factor(
    if_else(as.integer(as.character(y_train)) == 1, "Pobre", "NoPobre"),
    levels = c("NoPobre", "Pobre")
  )
  cm_th <- confusionMatrix(pred_th, obs_th, positive = "Pobre")
  tibble(
    threshold = th,
    F1        = cm_th$byClass["F1"],
    Precision = cm_th$byClass["Precision"],
    Recall    = cm_th$byClass["Recall"]
  )
})

# Threshold con mejor F1 en train (solo referencia visual, no se usa en submission)
th_max_f1 <- th_results |> filter(!is.na(F1)) |>
  slice_max(F1, n = 1, with_ties = FALSE) |>
  pull(threshold)

p_threshold <- th_results |>
  filter(!is.na(F1)) |>
  pivot_longer(cols = c(F1, Precision, Recall), names_to = "metrica") |>
  ggplot(aes(x = threshold, y = value, color = metrica)) +
  geom_line(linewidth = 0.9) +
  # Línea del threshold usado en la submission
  geom_vline(xintercept = THRESHOLD,
             linetype = "dashed", color = "gray30") +
  # Línea del threshold con mayor F1 en train (solo referencia)
  geom_vline(xintercept = th_max_f1,
             linetype = "dotted", color = "#534AB7", alpha = 0.7) +
  annotate("text", x = THRESHOLD + 0.02, y = 0.08,
           label = sprintf("th = %.1f\n(submission)", THRESHOLD),
           hjust = 0, size = 3.2, color = "gray25") +
  annotate("text", x = th_max_f1 - 0.02, y = 0.25,
           label = sprintf("th = %.2f\n(max F1 train)", th_max_f1),
           hjust = 1, size = 3.0, color = "#534AB7") +
  scale_color_manual(
    values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3"),
    labels = c(
      F1        = "F1 (balance)",
      Precision = "Precision (menos filtracion)",
      Recall    = "Recall (menos exclusion)"
    )
  ) +
  labs(
    title    = paste0("CART: metricas por threshold — ", MODEL_ID),
    subtitle = "Calculado sobre train completo (en muestra — optimista)",
    x        = "Threshold de clasificacion",
    y        = "Valor de la metrica",
    color    = NULL,
    caption  = "Linea discontinua = threshold usado en la submission (0.5). Punteada = max F1 en train."
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

ggsave(file.path(dir_model, "threshold.png"),
       p_threshold, width = 8, height = 5, dpi = 150)
message("  Guardado: threshold.png")

# ── Gráfico 2: Curva ROC ────────────────────────────────────────────────────
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
            scale_x_continuous(labels = function(x) paste0(round(x * 100), "%"))+
            scale_y_continuous(labels = function(x) paste0(round(x * 100), "%"))
  labs(
    title    = paste0("CART: curva ROC — ", MODEL_ID),
    subtitle = "Probabilidades del modelo sobre train completo (en muestra)",
    x        = "Tasa de Falsos Positivos  (1 - Specificity)",
    y        = "Tasa de Verdaderos Positivos  (Recall)",
    caption  = "AUC en muestra es optimista. Leer junto con la curva PR."
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc.png"),
       p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc.png")

# ── Gráfico 3: Curva Precision-Recall ───────────────────────────────────────
pr_df <- tibble(
  TP        = roc_obj$sensitivities        * sum(y_train == "1"),
  FP        = (1 - roc_obj$specificities) * sum(y_train == "0"),
  recall    = roc_obj$sensitivities,
  precision = TP / (TP + FP)
) |>
  filter(is.finite(precision), is.finite(recall))

p_prcurve <- ggplot(pr_df, aes(x = recall, y = precision)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_hline(yintercept = mean(train$pobre),
             linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.85, y = mean(train$pobre) + 0.015,
           label = sprintf("Clasificador aleatorio (%.0f%%)",
                           mean(train$pobre) * 100),
           size = 3, color = "gray50") +
  labs(
    title    = paste0("CART: curva Precision-Recall — ", MODEL_ID),
    subtitle = sprintf("AUC-ROC: %.4f | Probabilidades sobre train (en muestra)", auc_roc),
    x        = "Recall  (fraccion de pobres capturada)",
    y        = "Precision  (fraccion de predichos pobres que lo son)",
    caption  = "La curva por encima de la linea punteada indica mejor que el azar."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve.png"),
       p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve.png")

# =============================================================================
# PASO 3 — SUBMISSION PARA KAGGLE
# =============================================================================
message("\n========================================")
message("PASO 3 — Submission para Kaggle")
message("========================================")

# Usar predict type="class" directamente (equivalente a threshold=0.5)
pred_test  <- predict(cart_v1, newdata = X_test, type = "class")
preds_test <- as.integer(as.character(pred_test) == "1")

cat(sprintf("  %d hogares predichos como pobres (%.1f%%) | threshold = %.1f\n",
            sum(preds_test), mean(preds_test) * 100, THRESHOLD))

submission <- tibble(id = ids, pobre = preds_test)
sub_file   <- file.path(dir_subs, "CART_cp_0.001_maxdepth_20.csv")
write_csv(submission, sub_file)
message("  Guardado: ", sub_file)
message("  → Subir a Kaggle y anotar el public F1 en model_registry.csv")

# =============================================================================
# PASO 4 — DIAGNÓSTICOS Y REGISTRO EN model_registry.csv
# =============================================================================
message("\n========================================")
message("PASO 4 — Diagnosticos y registro")
message("========================================")

# Guardar objeto de diagnóstico completo
saveRDS(
  list(
    model_id        = MODEL_ID,
    autor           = AUTOR,
    fecha           = Sys.Date(),
    cp              = 0.001,
    maxdepth        = 20,
    threshold       = THRESHOLD,
    model_final     = cart_v1,
    probs_train     = probs_train,
    actuals_train   = y_train,
    cm_train        = cm_train,
    train_F1        = train_F1,
    train_Precision = train_Precision,
    train_Recall    = train_Recall,
    auc_roc         = auc_roc,
    th_results      = th_results,
    imp_df          = imp_df,
    predictores     = names(X_train),
    n_features      = ncol(X_train)
  ),
  file.path(dir_model, "diagnostics.rds")
)
message("  Guardado: diagnostics.rds")

# ── Registro en model_registry.csv ──────────────────────────────────────────
# Una fila por corrida. Si el archivo no existe lo crea; si ya existe, agrega.
# Si ya hay una fila con este MODEL_ID la sobreescribe para evitar duplicados.

nueva_fila <- tibble(
  model_id           = MODEL_ID,
  fecha              = Sys.Date(),
  autor              = AUTOR,
  algoritmo          = "CART",
  cp                 = 0.001,
  maxdepth           = 20,
  n_features         = ncol(X_train),
  imbalance_strategy = "none",
  cv_folds           = NA_integer_,     # sin CV en este baseline
  threshold          = THRESHOLD,
  train_F1           = round(train_F1,        4),
  train_Precision    = round(train_Precision, 4),
  train_Recall       = round(train_Recall,    4),
  auc_roc            = round(auc_roc,         4),
  kaggle_public_F1   = NA_real_,        # <<< llenar manualmente tras Kaggle
  notas              = paste0(
    "CART baseline. cp=0.001, maxdepth=20. threshold=0.5 (default). ",
    "Top vars: ", paste(vars_top10[1:3], collapse = ", ")
  )
)

if (file.exists(reg_path)) {
  registry <- read_csv(reg_path, show_col_types = FALSE)
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
message("\n========================================")
message("RESUMEN — ", MODEL_ID)
message("========================================")
message("  Submission:     ", sub_file)
message("  Threshold:      ", THRESHOLD, "  (default, sin optimizacion CV)")
message("  F1 en train:    ", round(train_F1, 4),
        "  (OPTIMISTA — el real lo da Kaggle)")
message("  AUC-ROC train:  ", round(auc_roc, 4))
message("  Precision train:", round(train_Precision, 4))
message("  Recall train:   ", round(train_Recall,    4))
message("")
message("  Graficos en: ", dir_model)
message("    threshold.png  — F1/Precision/Recall vs threshold")
message("    roc.png        — curva ROC con AUC")
message("    prcurve.png    — curva Precision-Recall")
message("    CART_baseline.pdf — visualizacion del arbol")
message("")
message("  Importancia de variables en 00_data/processed/:")
message("    cart_importancia.rds / cart_vars_top10/20/90.rds")
message("")

# nolint end