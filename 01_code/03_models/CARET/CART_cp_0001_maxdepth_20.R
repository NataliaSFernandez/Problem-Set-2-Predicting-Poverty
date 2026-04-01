#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/Cart_cp_0001_maxdepth_20.R
#==============================================================================
# ALGORITMO : CART — Classification and Regression Tree (rpart)
# PASOS:
#   Paso 1 — Baseline sin tuning          → submissions/cart_v1.csv

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
 
message("  train: ", nrow(train), " x ", ncol(train))
message("  test:  ", nrow(test),  " x ", ncol(test))
message("  Prevalencia pobre: ",
        round(mean(train$pobre == 1) * 100, 1), "%")
# -- 2. Preprocesamiento mínimo para rpart -----------------------------------
# rpart NO necesita:
#   - eliminar variables redundantes (el árbol las ignora sólo)
#   - escalar variables (no usa distancias)
#   - convertir binarias a dummy (las maneja nativamente)
#   - imputar NAs (los maneja internamente con surrogate splits)
#
# rpart SÍ necesita:
#   - variables numéricas o factor (no character suelto)
#   - eliminar id (identificador, no predictor)
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
 
    # cotiza_pension: ya es character ("cotiza"/"no_cotiza"/"pensionado")
    # rpart lo convierte automáticamente a factor → OK, no hace falta tocarlo
 
    # Cualquier otro character → factor para que rpart lo entienda
    mutate(across(where(is.character), as.factor))
}
 
X_train <- preparar_X_cart(train)
X_test  <- preparar_X_cart(test)
niveles <- c("0", "1")
y_train <- factor(train$pobre, levels = niveles)
 
message("  y_train: ", sum(y_train == "1"), " pobres (1) | ",
        sum(y_train == "0"), " no pobres (0)")
# Unir X e y para rpart (usa sintaxis de fórmula pobre ~ .)
train_cart <- bind_cols(X_train, pobre = y_train)
 
# =============================================================================
# PASO 1 — BASELINE SIN TUNING
# =============================================================================
message("\n========================================")
message("PASO 1 — Baseline CART (sin tuning)")
message("========================================")
message("  Todas las variables incluidas")
message("  cp = 0.001  |  maxdepth = 20")
 
set.seed(42)
t0 <- Sys.time()
 
cart_v1 <- rpart(
  pobre ~ .,
  data    = train_cart,
  method  = "class",
  control = rpart.control(
    cp        = 0.001,  # poda moderada — árbol razonablemente profundo
    maxdepth  = 20,     # sin límite práctico de profundidad
    minsplit  = 20,     # mín obs para intentar una partición
    minbucket = 7       # mín obs en una hoja terminal
  )
)
 
t1 <- Sys.time()
message("  Tiempo: ",
        round(as.numeric(difftime(t1, t0, units = "secs")), 1), " seg")
message("  Nodos terminales: ", sum(cart_v1$frame$var == "<leaf>"))
message("  Variables usadas: ",
        length(unique(cart_v1$frame$var[cart_v1$frame$var != "<leaf>"])))
 
# Cp óptimo sugerido por el propio árbol (para referencia en paso 2)
cp_optimo_sugerido <- cart_v1$cptable[
  which.min(cart_v1$cptable[, "xerror"]), "CP"]
message("  cp con menor error CV interno: ", round(cp_optimo_sugerido, 6))
 
# Visualización (primeros 4 niveles — árbol completo sería ilegible)
cat("\n-- Árbol baseline (primeros 4 niveles) --\n")
pdf("02_outputs/models/CARTs/CART_baseline.pdf", width = 12, height = 10)
rpart.plot(
  cart_v1,
  type          = 4,
  extra         = 104,   # muestra % de clase + n obs
  under         = TRUE,
  fallen.leaves = FALSE,
  main          = "CART baseline — pobre / no pobre",
  tweak         = 0.85,
)
dev.off()

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
    pct_fmt  = scales::percent(pct,      accuracy = 0.1),
    acum_fmt = scales::percent(pct_acum, accuracy = 0.1)
  )
 
imp_df |>
  head(25) |>
  select(variable, pct_fmt, acum_fmt) |>
  print(n = 25)
 
# Sets de variables para usar en otros modelos
vars_top10 <- imp_df |> head(10)          |> pull(variable)
vars_top20 <- imp_df |> head(20)          |> pull(variable)
vars_top90 <- imp_df |>
  filter(pct_acum <= 0.90)               |> pull(variable)
 
message("\n  Top 10 variables:")
cat("  ", paste(vars_top10, collapse = ", "), "\n")
message("  Variables con 90% de importancia: ", length(vars_top90))

# -- Evaluación en train -----------------------------------------------------
pred_train_v1 <- factor(
  as.character(predict(cart_v1, newdata = X_train, type = "class")),
  levels = niveles
)
cm_v1 <- confusionMatrix(pred_train_v1, y_train, positive = "1")
 
cat("\n-- Matriz de confusión (train) --\n")
print(cm_v1$table)
 
TP_v1 <- cm_v1$table["1", "1"]
FN_v1 <- cm_v1$table["0", "1"]
FP_v1 <- cm_v1$table["1", "0"]
TN_v1 <- cm_v1$table["0", "0"]
 
cat("\n-- Métricas (train) --\n")
cat("  F1:              ", round(cm_v1$byClass["F1"],        4), "\n")
cat("  Precision:       ", round(cm_v1$byClass["Precision"], 4), "\n")
cat("  Recall:          ", round(cm_v1$byClass["Recall"],    4), "\n")
cat("  Accuracy:        ", round(cm_v1$overall["Accuracy"],  4), "\n")
cat("  Exclusión  (FN): ", round(FN_v1 / (TP_v1 + FN_v1) * 100, 1),
    "% de pobres clasificados como no pobres\n")
cat("  Filtración (FP): ", round(FP_v1 / (FP_v1 + TN_v1) * 100, 1),
    "% de no pobres clasificados como pobres\n")
 
# -- Submission cp 0.001 y maxdepth 20 -----------------------------------------------------------
pred_test_v1  <- predict(cart_v1, newdata = X_test, type = "class")
submission_v1 <- tibble(id = ids,
                         pobre = as.integer(as.character(pred_test_v1) == "1"))
write_csv(submission_v1, "03_submissions/CART_cp_0.001_maxdepth_20.csv")
message("\n  Guardado: 03_submissions/CART_cp_0.001_maxdepth_20.csv")
message("  Pobres predichos: ", sum(submission_v1$pobre),
        " (", round(mean(submission_v1$pobre) * 100, 1), "%)")

