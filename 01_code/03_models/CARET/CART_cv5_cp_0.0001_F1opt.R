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
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,
  rpart,        # árbol de decisión
  rpart.plot,   # visualización del árbol
  caret,        # confusionMatrix, trainControl, train
  MLmetrics,    # F1_Score
  fs            # dir_create
)
# -- 1. Carga de datos -------------------------------------------------------
message("== Cargando datos ==")
 
train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id
 
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
 
# Función de métrica F1 para caret
# data$obs y data$pred son factores con niveles c("0","1")
# positive = "1" porque 1 = pobre (la clase que queremos detectar)
f1_summary <- function(data, lev = NULL, model = NULL) {
  f1 <- tryCatch({
    # Convertir a character para evitar problemas con niveles de factor
    obs  <- as.character(data$obs)
    pred <- as.character(data$pred)
    MLmetrics::F1_Score(y_true = obs, y_pred = pred, positive = "1")
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
 
best_cp <- cart_cv$bestTune$cp
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
write_csv(submission_v2, "03_submissions/CART_cp_0.0001_F1opt.csv")
message("\n  Guardado: 03_submissions/CART_cp_0.0001_F1opt.csv")
message("  Pobres predichos: ", sum(submission_v2$pobre),
        " (", round(mean(submission_v2$pobre) * 100, 1), "%)")
 