#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/Cart_cp_0001_maxdepth_20.R
#==============================================================================
# ALGORITMO : CART — Classification and Regression Tree (rpart)
# PASOS:
#   Paso 2 — Tuning cp con CV-5           
# Mismo procesamiento que baseline, pero con tuning de cp usando validación cruzada.
# HIPERPARÁMETROS:
#   cp (complexity parameter): penaliza la complejidad del árbol.
#      Valor pequeño = árbol profundo = más flexibilidad, riesgo de overfitting.
#      Valor grande  = árbol simple  = más generalizable, posible underfitting.
#      Rango a explorar: 0.0001, 0.0005, 0.001, 0.005
#   maxdepth: profundidad máxima del árbol (20 = prácticamente sin límite).
#==============================================================================
# nolint start
# -- 0. Paquetes -------------------------------------------------------------
library(tidyverse)
library(rpart)
library(rpart.plot)
library(caret)
library(pROC)
library(fs)
# -- 1. Carga de datos -------------------------------------------------------
message("== Cargando datos ==")
 
train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id

AUTOR    <- "Natalia"
MODEL_ID <- "CART_cv5_F1opt"

# Rutas de outputs
dir_model <- "02_outputs/models/CARTs/Baseline/CART_cv5_F1opt"
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model,    recurse = TRUE)
fs::dir_create(dir_subs,     recurse = TRUE)
fs::dir_create("02_outputs", recurse = TRUE)
 
# -- 2. Preprocesamiento mínimo para rpart -----------------------------------
message("\n== Preprocesando variables ==")
 
preparar_X_cart <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>
 
    # zona: "1"/"2" (character) → numérico
    mutate(zona = as.numeric(zona)) |>
 
    # ciudad y dpto: character de alta cardinalidad
    # Con 24+ niveles, rpart crea ramas con muy pocas obs → no generaliza
    # Las variables costa_caribe, bogota, eje_cafetero ya capturan la región
    select(-any_of(c("ciudad", "dpto"))) |>
    mutate(across(where(is.character), as.factor))
}
 
X_train <- preparar_X_cart(train)
X_test  <- preparar_X_cart(test)
niveles <- c("0", "1")
y_train <- factor(train$pobre, levels = niveles)

# Alinear columnas train/test
cols_comunes <- intersect(names(X_train), names(X_test))
X_train <- X_train |> select(all_of(cols_comunes))
X_test  <- X_test  |> select(all_of(cols_comunes))

# Unir X e y para rpart (usa sintaxis de fórmula pobre ~ .)
train_cart <- bind_cols(X_train, pobre = y_train)

# =============================================================================
# PASO 2 — TUNING CON CV-5 + GRILLA DE cp
# =============================================================================
message("\n========================================")
message("PASO 2 — Tuning CART con CV-5")
message("========================================")
message("  Grilla: 4 valores de cp")
message("  Tiempo estimado: 3-8 minutos")
 
# F1 calculado manualmente — sin dependencia de MLmetrics
# TP / (TP + 0.5*(FP+FN)) es equivalente a 2*TP / (2*TP + FP + FN)
f1_manual <- function(obs, pred, positive = "1") {
  TP <- sum(obs == positive & pred == positive)
  FP <- sum(obs != positive & pred == positive)
  FN <- sum(obs == positive & pred != positive)
  if ((2 * TP + FP + FN) == 0) return(NA_real_)
  2 * TP / (2 * TP + FP + FN)
}

f1_summary <- function(data, lev = NULL, model = NULL) {
  f1 <- tryCatch({
    obs  <- as.character(data$obs)
    pred <- as.character(data$pred)
    f1_manual(obs, pred, positive = "1")
  }, error = function(e) {
    NA_real_
  })
  c(F1 = f1)
}
 
ctrl_cv <- trainControl(
  method          = "cv",
  number          = 5,
  summaryFunction = f1_summary,
  classProbs      = FALSE,
  verboseIter     = TRUE,
  savePredictions = "final"
)
 
# Grilla: 4 valores de cp que cubren el rango relevante
tune_grid <- expand.grid(
  cp = c(0.0001, 0.0005, 0.001, 0.005)
)
 
set.seed(42)
t0 <- Sys.time()
 
# Se pasa X_train e y_train directamente (interfaz x/y) para evitar
# ambigüedades con la fórmula cuando pobre es 0/1 numérico
cart_cv <- train(
  x         = X_train,
  y         = y_train,
  method    = "rpart",
  trControl = ctrl_cv,
  tuneGrid  = tune_grid,
  metric    = "F1"
)
 
t1 <- Sys.time()
message("  Tiempo CV: ",
        round(as.numeric(difftime(t1, t0, units = "mins")), 1), " min")
 
cat("\n-- Resultados por cp (CV-5) --\n")
print(cart_cv$results |>
        arrange(desc(F1)) |>
        mutate(across(where(is.numeric), ~ round(., 4))))
 
best_cp    <- cart_cv$bestTune$cp
best_f1_cv <- max(cart_cv$results$F1, na.rm = TRUE)
message("  Mejor cp según CV-5: ", best_cp)
message("  Mejor F1 en CV-5:    ", round(best_f1_cv, 4))
 
# Re-entrenar con el mejor cp sobre todos los datos de train
cart_v2 <- rpart(
  pobre ~ .,
  data    = train_cart,
  method  = "class",
  control = rpart.control(
    cp        = best_cp,
    maxdepth  = 20,
    minsplit  = 20,
    minbucket = 7
  )
)
 
message("  Nodos terminales del mejor árbol: ",
        sum(cart_v2$frame$var == "<leaf>"))

# Evaluación en train
pred_train_v2 <- factor(
  as.character(predict(cart_v2, newdata = X_train, type = "class")),
  levels = niveles
)
cm_v2 <- confusionMatrix(pred_train_v2, y_train, positive = "1")
 
cat("\n-- Métricas v2 (train) --\n")
cat("  F1:        ", round(cm_v2$byClass["F1"],        4), "\n")
cat("  Precision: ", round(cm_v2$byClass["Precision"], 4), "\n")
cat("  Recall:    ", round(cm_v2$byClass["Recall"],    4), "\n")
 
# Submission v2
pred_test_v2  <- predict(cart_v2, newdata = X_test, type = "class")
submission_v2 <- tibble(id = ids,
                         pobre = as.integer(as.character(pred_test_v2) == "1"))
write_csv(submission_v2, file.path(dir_subs, "CART_cp_0.0001_F1opt.csv"))
message("\n  Guardado: ", file.path(dir_subs, "CART_cp_0.0001_F1opt.csv"))
message("  Pobres predichos: ", sum(submission_v2$pobre),
        " (", round(mean(submission_v2$pobre) * 100, 1), "%)")

# =============================================================================
# GRÁFICOS: THRESHOLD, ROC, PRECISION-RECALL
# =============================================================================
# Los gráficos se construyen sobre las probabilidades del modelo final
# (cart_v2, re-entrenado con el mejor cp) aplicadas sobre train completo.
# Son métricas "en muestra" — útiles para visualizar comportamiento del modelo.
message("\n========================================")
message("Generando gráficos")
message("========================================")

# Probabilidades del modelo final sobre train (para construir curvas)
probs_train <- predict(cart_v2, newdata = X_train, type = "prob")[, "1"]

# AUC-ROC sobre train
roc_obj <- pROC::roc(
  response  = as.integer(as.character(y_train)),
  predictor = probs_train,
  quiet     = TRUE
)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC (train, en muestra): ", round(auc_roc, 4))

# ── Gráfico 1: Threshold ────────────────────────────────────────────────────
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

th_max_f1 <- th_results |>
  filter(!is.na(F1)) |>
  slice_max(F1, n = 1, with_ties = FALSE) |>
  pull(threshold)

p_threshold <- th_results |>
  filter(!is.na(F1)) |>
  pivot_longer(cols = c(F1, Precision, Recall), names_to = "metrica") |>
  ggplot(aes(x = threshold, y = value, color = metrica)) +
  geom_line(linewidth = 0.9) +
  geom_vline(xintercept = 0.5,
             linetype = "dashed", color = "gray30") +
  geom_vline(xintercept = th_max_f1,
             linetype = "dotted", color = "#534AB7", alpha = 0.7) +
  annotate("text", x = 0.5 + 0.02, y = 0.08,
           label = "th = 0.5\n(submission)",
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
    subtitle = paste0("cp optimo = ", best_cp,
                      " | CV-5 F1 = ", round(best_f1_cv, 4),
                      " | calculado sobre train (en muestra)"),
    x        = "Threshold de clasificacion",
    y        = "Valor de la metrica",
    color    = NULL,
    caption  = "Linea discontinua = threshold usado en la submission (0.5). Punteada = max F1 en train."
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

ggsave(file.path(dir_model, "threshold_cv5.png"),
       p_threshold, width = 8, height = 5, dpi = 150)
message("  Guardado: threshold_cv5.png")

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
  scale_x_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  labs(
    title    = paste0("CART: curva ROC — ", MODEL_ID),
    subtitle = paste0("cp optimo = ", best_cp,
                      " | Probabilidades sobre train (en muestra)"),
    x        = "Tasa de Falsos Positivos  (1 - Specificity)",
    y        = "Tasa de Verdaderos Positivos  (Recall)",
    caption  = "AUC en muestra es optimista. Leer junto con la curva PR."
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc_cv5.png"),
       p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc_cv5.png")

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
    subtitle = sprintf("AUC-ROC: %.4f | cp optimo = %s | train (en muestra)",
                       auc_roc, best_cp),
    x        = "Recall  (fraccion de pobres capturada)",
    y        = "Precision  (fraccion de predichos pobres que lo son)",
    caption  = "La curva por encima de la linea punteada indica mejor que el azar."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve_cv5.png"),
       p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve_cv5.png")

# =============================================================================
# REGISTRO EN model_registry.csv
# =============================================================================

message("\n========================================")
message("Registro en model_registry.csv")
message("========================================")

train_F1        <- cm_v2$byClass["F1"]
train_Precision <- cm_v2$byClass["Precision"]
train_Recall    <- cm_v2$byClass["Recall"]

nueva_fila <- tibble(
  model_id           = MODEL_ID,
  fecha              = Sys.Date(),
  autor              = AUTOR,
  algoritmo          = "CART",
  cp                 = best_cp,
  maxdepth           = 20,
  n_features         = ncol(X_train),
  imbalance_strategy = "none",
  cv_folds           = 5L,
  cv_F1              = round(best_f1_cv,   4),
  train_F1           = round(train_F1,     4),
  train_Precision    = round(train_Precision, 4),
  train_Recall       = round(train_Recall, 4),
  auc_roc            = round(auc_roc,      4),
  threshold          = 0.5,
  kaggle_public_F1   = 0.66,
  notas              = paste0(
    "CART tuneado con CV-5. Mejor cp=", best_cp,
    ". F1 CV-5=", round(best_f1_cv, 4), ". threshold=0.5 (default)."
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

cat("\n-- Fila registrada --\n")
print(nueva_fila)

# nolint end