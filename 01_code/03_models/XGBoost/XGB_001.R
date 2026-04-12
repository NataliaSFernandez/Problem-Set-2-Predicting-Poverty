#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/XGBoost/XGB_001.R
#==============================================================================
#
# ALGORITMO: XGBoost — Extreme Gradient Boosting (xgbTree, via caret)
#
# QUÉ HACE ESTE SCRIPT:
#   Entrena un modelo XGBoost de clasificación binaria. XGBoost es un
#   algoritmo de boosting que construye árboles de forma secuencial: cada
#   árbol nuevo se enfoca en corregir los errores (residuos) de los árboles
#   anteriores, minimizando una función de pérdida regularizada.
#
#   Diferencia con Random Forest:
#     - RF entrena árboles en paralelo sobre bootstraps independientes.
#     - XGBoost entrena secuencialmente; cada árbol se construye sobre los
#       gradientes del error acumulado. Esto lo hace potencialmente más
#       preciso, pero también más propenso a overfitting sin regularización.
#
#   XGBoost añade regularización explícita (gamma, lambda) sobre los pesos
#   de las hojas y la complejidad del árbol, lo que lo diferencia del GBM
#   clásico y lo hace más robusto.
#
# HIPERPARÁMETROS TUNADOS CON CV-5 (notebook Lecture10, Tree Methods):
#
#   nrounds          : número de árboles (iteraciones de boosting).
#                      Más árboles = más capacidad, pero mayor riesgo de
#                      overfitting si el learning rate es alto.
#                      Valores explorados: c(100, 200).
#                      (El notebook usa c(100, 250); se reduce 250 → 200
#                       para acotar el tiempo de cómputo.)
#
#   max_depth        : profundidad máxima de cada árbol.
#                      Árboles profundos capturan interacciones complejas
#                      pero pueden memorizar el ruido del training set.
#                      Valores explorados: c(3, 6).
#                      (El notebook usa c(2, 4); se ajustan los puntos
#                       medios para cubrir un rango similar sin agregar
#                       combinaciones extra.)
#
#   eta              : learning rate (shrinkage). Escala la contribución de
#                      cada árbol. Valores pequeños generalizan mejor pero
#                      requieren más árboles.
#                      Valores explorados: c(0.05, 0.10).
#                      (El notebook usa c(0.01, 0.05); se eleva el mínimo
#                       para que nrounds=100 sea suficiente para converger.)
#
#   gamma            : pérdida mínima necesaria para realizar una división
#                      en un nodo. Actúa como regularización de la estructura
#                      del árbol.
#                      Fijo en 0 (sin penalización por split). El notebook
#                      usa c(0, 1); se fija en el valor conservador para
#                      reducir la grilla.
#
#   min_child_weight : mínimo de observaciones en un nodo hoja.
#                      Previene divisiones sobre pocos datos. Fijo en 10
#                      (razonable para 164k obs). El notebook usa c(10, 25).
#
#   colsample_bytree : fracción de predictores por árbol.
#                      Fijo en 0.66 (submuestra moderada). El notebook
#                      usa c(0.33, 0.66).
#
#   subsample        : fracción de observaciones por árbol.
#                      Fijo en 0.80 (poco agresivo). El notebook
#                      usa c(0.4, 0.8).
#
# DECISIÓN DE DISEÑO DE LA GRILLA:
#   El notebook usa 2^7 = 128 combinaciones × 5 folds = 640 fits.
#   Con 164k observaciones eso toma 4-8 horas. Para mantener el tiempo
#   total bajo ~1 hora se tunean los 3 hiperparámetros de mayor impacto
#   (nrounds, max_depth, eta) y se fijan los 4 restantes en valores
#   razonables para el tamaño del dataset.
#   Resultado: 2×2×2 = 8 combinaciones × 5 folds = 40 fits.
#   Si el tiempo disponible lo permite, se puede descomentar la grilla
#   completa del notebook al final de la Sección 6.
#
# PARALELISMO:
#   xgboost usa OpenMP internamente para paralelizar la construcción de
#   cada árbol (nthread). Se detectan automáticamente los núcleos disponibles.
#
# FLUJO DEL SCRIPT:
#   0. Paquetes
#   1. Configuración global (AUTOR, MODEL_ID, rutas, directorios)
#   2. Cargar datos
#   3. Preprocesamiento (todo numérico — requisito de XGBoost)
#   4. Funciones auxiliares (F1, Precision, Recall)
#   5. trainControl (CV-5, métrica F1)
#   6. Grilla de hiperparámetros (8 combinaciones)
#   7. Entrenamiento con CV
#   8. Resultados del CV y mejores hiperparámetros
#   9. Probabilidades del modelo final e importancia de variables
#   10. Gráficos de diagnóstico (varimp.png, threshold.png, roc.png,
#       prcurve.png)
#   11. Submission CSV (threshold = 0.5)
#   12. Registro en model_registry.csv
#
# INPUTS (relativos a la raíz del proyecto):
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#
# OUTPUTS (relativos a la raíz del proyecto):
#   03_submissions/submission_XGB_001.csv          ← subir a Kaggle
#   02_outputs/models/XGBoost/XGB_001/varimp.png
#   02_outputs/models/XGBoost/XGB_001/threshold.png
#   02_outputs/models/XGBoost/XGB_001/roc.png
#   02_outputs/models/XGBoost/XGB_001/prcurve.png
#   02_outputs/models/XGBoost/XGB_001/diagnostics.rds
#   02_outputs/model_registry.csv                  ← tabla maestra (append)
#
# REPRODUCIBILIDAD:
#   Correr desde la raíz del proyecto. Semilla global: 42.
#   [ANTES de correr]  Verificar que AUTOR esté correcto.
#   [DESPUÉS de correr] Subir el CSV a Kaggle y anotar el public F1
#                       en model_registry.csv (columna kaggle_public_F1).
#==============================================================================
# nolint start

# =============================================================================
# SECCIÓN 0: PAQUETES
# =============================================================================
# pacman::p_load() instala el paquete si no está instalado y lo carga.
# Garantiza que el script sea autocontenido.
#
# Produce: entorno con todos los paquetes disponibles.
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,  # dplyr (manipulación), ggplot2 (gráficos), purrr (map_dfr), tibble
  caret,      # train(), trainControl(), confusionMatrix(), varImp()
  xgboost,    # XGBoost; usado internamente por caret (method = "xgbTree")
  pROC,       # roc(), auc() — curva ROC y AUC
  fs          # dir_create() — crear carpetas de forma multiplataforma
)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
# Define las constantes de identificación, la semilla global y las rutas de
# salida. Crear los directorios aquí evita errores al guardar archivos más tarde.
#
# Produce: variables AUTOR, MODEL_ID, rutas y carpetas de output creadas.
AUTOR    <- "Jonathan Melo"
MODEL_ID <- "XGB_001"

# Semilla global — todos los scripts del equipo usan 42 para comparabilidad
set.seed(42)

# Número de núcleos para parallelismo interno de xgboost (OpenMP)
N_THREADS <- max(1L, parallel::detectCores() - 1L)

dir_model <- file.path("02_outputs/models/XGBoost", MODEL_ID)
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model,    recurse = TRUE)
fs::dir_create(dir_subs,     recurse = TRUE)
fs::dir_create("02_outputs", recurse = TRUE)

message("  Núcleos disponibles para XGBoost (nthread): ", N_THREADS)

# =============================================================================
# SECCIÓN 2: CARGAR DATOS
# =============================================================================
# Cargamos los .rds porque son más rápidos que los .csv y preservan los tipos
# de variables (factores, enteros, dobles) exactamente como los dejó el
# script de preprocesamiento.
#
# Produce: data frames train (164,960 filas) y test (sin 'pobre'), vector ids.
message("== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id  # identificadores del test para construir el submission

message("Train: ", nrow(train), " hogares | ", ncol(train), " columnas")
message("Test:  ", nrow(test),  " hogares | ", ncol(test),  " columnas")
message("Balance: ", round(mean(train$pobre) * 100, 1), "% pobres  /  ",
        round(mean(1 - train$pobre) * 100, 1), "% no pobres")
# Con ~20% pobres el dataset está desbalanceado (ratio 1:4). Un clasificador
# trivial que predice siempre "No pobre" alcanza accuracy = 80% pero F1 = 0.
# Por eso usamos F1 como métrica de tuning.

# =============================================================================
# SECCIÓN 3: PREPROCESAMIENTO
# =============================================================================
# XGBoost requiere que TODAS las entradas sean numéricas: no acepta factores
# ni caracteres. La estrategia es:
#   1. Quitar id y pobre (target e identificador sin valor predictivo).
#   2. Convertir zona de character a numérico.
#   3. Excluir ciudad y dpto (alta cardinalidad, 24+ niveles), misma decisión
#      que en los scripts CART y RF.
#   4. Convertir cualquier columna character restante a numérico.
#      Las variables binarias codificadas como "0"/"1" en character pasan
#      directamente a 0 y 1 numérico.
#   5. Convertir factores a entero (su código subyacente).
#   6. Imputar NAs con la mediana de cada columna. Aunque xgboost puede
#      manejar NAs internamente, caret los rechaza antes de llamar a xgb.
#
# Produce: X_train y X_test completamente numéricos y sin NAs, y_train factor.
message("\n== Preprocesando variables ==")

preparar_X_xgb <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>              # quitar target e ID
    mutate(zona = as.numeric(zona)) |>                 # "1"/"2" (chr) → numérico
    select(-any_of(c("ciudad", "dpto"))) |>            # alta cardinalidad → excluir
    mutate(across(where(is.character), as.numeric)) |> # "0"/"1" chr → numérico
    mutate(across(where(is.factor),   as.integer)) |>  # factor → código entero
    mutate(across(everything(),                        # imputar NAs con mediana
                  ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
}

X_train <- preparar_X_xgb(train)
X_test  <- preparar_X_xgb(test)

# Alinear columnas: garantiza que train y test tengan las mismas variables
# en el mismo orden (evita errores en predict())
cols_comunes <- intersect(names(X_train), names(X_test))
X_train      <- X_train |> select(all_of(cols_comunes))
X_test       <- X_test  |> select(all_of(cols_comunes))

# Convertir a matrix: xgboost trabaja internamente con matrices densas.
# Hacerlo aquí evita conversiones implícitas en cada fold del CV.
X_train_mat <- as.matrix(X_train)
X_test_mat  <- as.matrix(X_test)

# Variable respuesta: factor con niveles válidos para caret.
# caret requiere identificadores de R válidos como niveles (no "0"/"1").
# "No" = clase negativa (no pobre), "Yes" = clase positiva (pobre).
niveles <- c("No", "Yes")
y_train <- factor(ifelse(train$pobre == 1, "Yes", "No"), levels = niveles)

message("  Predictores: ", ncol(X_train))
message("  NAs restantes en X_train: ", sum(is.na(X_train)))

# =============================================================================
# SECCIÓN 4: FUNCIONES AUXILIARES
# =============================================================================
# prf_summary() es el summaryFunction que caret llama al final de cada fold.
# Retorna F1, Precision y Recall simultáneamente para que las tres métricas
# queden guardadas en xgb_cv$results y podamos registrarlas.
# No depende de MLmetrics (que puede tener comportamiento inestable con NAs).
# Positivo = "Yes" (pobre = 1).
#
# Produce: función disponible en el entorno para ser pasada a trainControl().
prf_summary <- function(data, lev = NULL, model = NULL) {
  obs  <- as.character(data$obs)
  pred <- as.character(data$pred)
  TP <- sum(obs == "Yes" & pred == "Yes")
  FP <- sum(obs != "Yes" & pred == "Yes")
  FN <- sum(obs == "Yes" & pred != "Yes")
  precision <- if ((TP + FP) == 0) NA_real_ else TP / (TP + FP)
  recall    <- if ((TP + FN) == 0) NA_real_ else TP / (TP + FN)
  f1        <- if ((2 * TP + FP + FN) == 0) NA_real_ else 2 * TP / (2 * TP + FP + FN)
  c(F1 = f1, Precision = precision, Recall = recall)
}

# =============================================================================
# SECCIÓN 5: trainControl
# =============================================================================
# Define la estrategia de validación: CV de 5 folds, optimizando F1.
# classProbs = TRUE es necesario para que, después del CV, podamos llamar
# predict(xgb_cv, newdata = X_train_mat, type = "prob") y obtener
# probabilidades continuas P(pobre=1|X) para las curvas de diagnóstico.
# savePredictions = "final" guarda las predicciones OOF de la combinación
# óptima de hiperparámetros.
#
# Produce: objeto ctrl_cv pasado a train().
ctrl_cv <- trainControl(
  method          = "cv",
  number          = 5,
  summaryFunction = prf_summary,   # retorna F1, Precision, Recall por fold
  classProbs      = TRUE,          # necesario para predict(..., type = "prob")
  verboseIter     = TRUE,          # imprime el progreso de cada fold en consola
  savePredictions = "final"
)

# =============================================================================
# SECCIÓN 6: GRILLA DE HIPERPARÁMETROS
# =============================================================================
# La grilla del notebook (Lecture10) tiene 2^7 = 128 combinaciones × 5 folds
# = 640 fits, lo que toma 4-8 horas con 164k observaciones.
#
# Para mantener el tiempo total bajo ~1 hora se tunean los 3 hiperparámetros
# de mayor impacto sobre el bias-varianza:
#   nrounds   → controla cuántos árboles suma el ensamble (más árboles = más
#               capacidad pero también más tiempo de cómputo por fit)
#   max_depth → controla la complejidad de cada árbol
#   eta       → controla cuánto "aprende" cada árbol nuevo
#
# Los 4 hiperparámetros restantes se fijan en valores razonables para el
# tamaño del dataset (tomados del rango del notebook):
#   gamma            = 0    (sin penalización por split; valor conservador)
#   min_child_weight = 10   (hoja mínima de 10 obs; extremo bajo del notebook)
#   colsample_bytree = 0.66 (2/3 de columnas por árbol; extremo alto del notebook)
#   subsample        = 0.80 (80% de filas por árbol; extremo alto del notebook)
#
# Resultado: 2×2×2 = 8 combinaciones × 5 folds = 40 fits (~30-50 min).
#
# Para usar la grilla completa del notebook (128 combinaciones, ~4-8 horas),
# descomentar el bloque alternativo al final de esta sección.
#
# Produce: data frame tune_grid con todas las combinaciones a evaluar.
tune_grid <- expand.grid(
  nrounds          = c(100, 200),   # iteraciones; 200 en vez de 250 para acotar tiempo
  max_depth        = c(3, 6),       # profundidad del árbol
  eta              = c(0.05, 0.10), # learning rate; se eleva el mínimo (0.01 → 0.05)
                                    # para que nrounds=100 sea suficiente para converger
  gamma            = 0,             # sin penalización por split (fijo)
  min_child_weight = 10,            # hoja mínima de 10 observaciones (fijo)
  colsample_bytree = 0.66,          # 2/3 de columnas por árbol (fijo)
  subsample        = 0.80           # 80% de filas por árbol (fijo)
)

# -- Grilla completa del notebook (descomentar si el tiempo no es restricción) --
# tune_grid <- expand.grid(
#   nrounds          = c(100, 250),
#   max_depth        = c(2, 4),
#   eta              = c(0.01, 0.05),
#   gamma            = c(0, 1),
#   min_child_weight = c(10, 25),
#   colsample_bytree = c(0.33, 0.66),
#   subsample        = c(0.4, 0.8)
# )
# -------------------------------------------------------------------------------

message("\n== Grilla XGBoost ==")
message("  Combinaciones: ", nrow(tune_grid),
        " (reducida de 128 para tiempo < 1 hora)")
message("  Fits totales:  ", nrow(tune_grid) * 5, " (grilla × 5 folds)")
message("  Núcleos usados por xgboost (nthread): ", N_THREADS)
message("  Tiempo estimado: 30-50 min con 164k observaciones")

# =============================================================================
# SECCIÓN 7: ENTRENAMIENTO CON CV
# =============================================================================
# caret::train() con method = "xgbTree":
#   1. Divide train en 5 folds.
#   2. Para cada una de las 8 combinaciones de hiperparámetros:
#      - Entrena XGBoost en 4 folds.
#      - Evalúa F1 en el fold restante.
#      - Promedia F1 sobre los 5 folds.
#   3. Selecciona la combinación con mayor F1 promedio.
#   4. Re-entrena automáticamente el modelo final con esa combinación
#      sobre TODOS los datos de train.
#
# nthread se pasa como argumento adicional a xgboost para paralelizar
# la construcción de cada árbol (nivel de CPU, no de folds).
#
# Produce: objeto xgb_cv con resultados de CV, bestTune y finalModel.
message("\n========================================")
message("Entrenando XGBoost con CV-5")
message("========================================")

set.seed(42)
t0 <- Sys.time()

xgb_cv <- train(
  x         = X_train_mat,   # xgboost requiere matrix, no data frame
  y         = y_train,
  method    = "xgbTree",     # usa el paquete xgboost internamente
  trControl = ctrl_cv,
  tuneGrid  = tune_grid,
  metric    = "F1",          # caret selecciona el modelo que maximiza F1
  verbose   = FALSE,         # suprime la salida de xgboost por iteración
  nthread   = N_THREADS      # paralelismo interno de xgboost (OpenMP)
)

t1 <- Sys.time()
message("  Tiempo CV: ",
        round(as.numeric(difftime(t1, t0, units = "mins")), 1), " min")

# =============================================================================
# SECCIÓN 8: RESULTADOS DEL CV
# =============================================================================
# xgb_cv$results contiene el F1, Precision y Recall promedio de los 5 folds
# para cada combinación de hiperparámetros.
# xgb_cv$bestTune indica la combinación óptima según F1.
#
# Produce: tabla impresa en consola (ordenada por F1 desc), variables best_*
#          para el registro y para etiquetar los gráficos.
cat("\n-- Resultados por combinación (CV-5, ordenados por F1 desc) --\n")
print(xgb_cv$results |>
        arrange(desc(F1)) |>
        mutate(across(where(is.numeric), ~ round(., 4))))

best_nrounds          <- xgb_cv$bestTune$nrounds
best_max_depth        <- xgb_cv$bestTune$max_depth
best_eta              <- xgb_cv$bestTune$eta
best_gamma            <- xgb_cv$bestTune$gamma
best_min_child_weight <- xgb_cv$bestTune$min_child_weight
best_colsample        <- xgb_cv$bestTune$colsample_bytree
best_subsample        <- xgb_cv$bestTune$subsample
best_f1_cv            <- max(xgb_cv$results$F1, na.rm = TRUE)

# Extraer Precision y Recall de la fila con los mejores hiperparámetros
best_row <- xgb_cv$results |>
  filter(nrounds          == best_nrounds,
         max_depth        == best_max_depth,
         eta              == best_eta,
         gamma            == best_gamma,
         min_child_weight == best_min_child_weight,
         colsample_bytree == best_colsample,
         subsample        == best_subsample)

best_prec_cv   <- best_row$Precision
best_recall_cv <- best_row$Recall

message("\n  Mejor combinación:")
message("    nrounds          = ", best_nrounds)
message("    max_depth        = ", best_max_depth)
message("    eta              = ", best_eta)
message("    gamma            = ", best_gamma)
message("    min_child_weight = ", best_min_child_weight)
message("    colsample_bytree = ", best_colsample)
message("    subsample        = ", best_subsample)
message("  Mejor F1 CV-5:    ", round(best_f1_cv,     4))
message("  Precision CV-5:   ", round(best_prec_cv,   4))
message("  Recall CV-5:      ", round(best_recall_cv, 4))

# =============================================================================
# SECCIÓN 9: PROBABILIDADES DEL MODELO FINAL, THRESHOLD OOF E IMPORTANCIAS
# =============================================================================
# caret re-entrenó automáticamente el modelo final (xgb_cv$finalModel) con
# los mejores hiperparámetros sobre todos los datos de train. A diferencia
# del script de RF, no es necesario re-entrenar manualmente porque caret
# ya activa las probabilidades a través de classProbs = TRUE.
#
# Flujo de esta sección:
#   1. Extraer predicciones OOF de xgb_cv$pred y buscar el threshold óptimo.
#      OOF = out-of-fold: cada observación fue predicha por el modelo que NO
#      la vio en ese fold → estimación honesta fuera de muestra, análoga a
#      las OOF del CV manual del LPM.
#   2. Obtener probabilidades del modelo final sobre train (en muestra).
#   3. Aplicar el threshold OOF al modelo final para métricas en muestra.
#   4. Calcular importancias de variables.
#
# Produce: best_th (threshold óptimo OOF), probs_train (probabilidades en
#          muestra), pred_train (clase con best_th), cm_final, métricas
#          train_*, varimp_xgb.
message("\n========================================")
message("Evaluando modelo final y optimizando threshold (OOF)")
message("========================================")

# ---------------------------------------------------------------------------
# 9.1  OPTIMIZACIÓN DEL THRESHOLD SOBRE PREDICCIONES OOF
# ---------------------------------------------------------------------------
# xgb_cv$pred (disponible por savePredictions = "final") contiene las
# predicciones out-of-fold para la combinación óptima de hiperparámetros:
# cada fila es una observación de train predicha por el modelo que NO la vio
# durante ese fold. Filtramos las filas de los mejores hiperparámetros y
# ordenamos por rowIndex para garantizar que el orden coincide con y_train.

oof_preds <- xgb_cv$pred |>
  filter(nrounds          == best_nrounds,
         max_depth        == best_max_depth,
         eta              == best_eta,
         gamma            == best_gamma,
         min_child_weight == best_min_child_weight,
         colsample_bytree == best_colsample,
         subsample        == best_subsample) |>
  arrange(rowIndex)

probs_oof <- oof_preds$Yes   # P(pobre=1|X) fuera de muestra para cada hogar

# Búsqueda del threshold que maximiza F1 sobre las predicciones OOF
th_grid <- seq(0.05, 0.95, by = 0.01)

th_oof_search <- map_dfr(th_grid, function(th) {
  pred_th <- factor(ifelse(probs_oof >= th, "Yes", "No"), levels = niveles)
  TP <- sum(pred_th == "Yes" & oof_preds$obs == "Yes")
  FP <- sum(pred_th == "Yes" & oof_preds$obs != "Yes")
  FN <- sum(pred_th == "No"  & oof_preds$obs == "Yes")
  f1   <- if ((2 * TP + FP + FN) == 0) NA_real_ else 2 * TP / (2 * TP + FP + FN)
  prec <- if ((TP + FP) == 0)          NA_real_ else TP / (TP + FP)
  rec  <- if ((TP + FN) == 0)          NA_real_ else TP / (TP + FN)
  tibble(threshold = th, F1 = f1, Precision = prec, Recall = rec)
})

best_th_row <- th_oof_search |>
  filter(!is.na(F1)) |>
  slice_max(F1, n = 1, with_ties = FALSE)

best_th     <- best_th_row$threshold
best_oof_f1 <- best_th_row$F1
best_oof_pr <- best_th_row$Precision
best_oof_rc <- best_th_row$Recall

message("  Threshold óptimo (OOF, max F1): ", best_th)
message("  F1 OOF:        ", round(best_oof_f1, 4))
message("  Precision OOF: ", round(best_oof_pr, 4))
message("  Recall OOF:    ", round(best_oof_rc, 4))

# ---------------------------------------------------------------------------
# 9.2  PROBABILIDADES DEL MODELO FINAL (en muestra — para gráficos)
# ---------------------------------------------------------------------------
# predict con type = "prob" disponible porque classProbs = TRUE en trainControl
probs_train <- predict(xgb_cv,
                       newdata = X_train_mat,
                       type    = "prob")[, "Yes"]

# Clase predicha sobre train con el threshold óptimo OOF
pred_train <- factor(
  ifelse(probs_train >= best_th, "Yes", "No"),
  levels = niveles
)

cm_final        <- confusionMatrix(pred_train, y_train, positive = "Yes")
train_F1        <- cm_final$byClass["F1"]
train_Precision <- cm_final$byClass["Precision"]
train_Recall    <- cm_final$byClass["Recall"]

cat("\n-- Métricas modelo final (train, en muestra, threshold =", best_th, ") --\n")
cat("  F1:        ", round(train_F1,        4), "\n")
cat("  Precision: ", round(train_Precision, 4), "\n")
cat("  Recall:    ", round(train_Recall,    4), "\n")
# NOTA: estas métricas son en muestra (optimistas). Las referencias confiables
# son cv_F1 (Sección 8) y las métricas OOF del threshold (arriba).

# Importancia de variables por Gain (reducción de la función de pérdida
# logística atribuida a cada predictor al usarse como criterio de split).
# varImp() normaliza a [0, 100] para comparabilidad.
varimp_xgb <- varImp(xgb_cv, scale = TRUE)

# =============================================================================
# SECCIÓN 10: GRÁFICOS DE DIAGNÓSTICO
# =============================================================================
# Construimos cuatro gráficos con los resultados del modelo final:
#
#   varimp.png    : Top 20 predictores por importancia (Gain de XGBoost).
#                   Indica qué variables contribuyen más a reducir la pérdida.
#   threshold.png : F1, Precision y Recall en función del threshold, calculado
#                   sobre predicciones OOF (fuera de muestra, honesto).
#                   Marca el threshold óptimo OOF (el que se usa en submission)
#                   y 0.5 como referencia.
#   roc.png       : curva ROC con el AUC. Evalúa la capacidad discriminante
#                   del modelo independiente del threshold.
#   prcurve.png   : curva Precision-Recall. Más informativa que ROC en
#                   datasets desbalanceados (clase minoritaria = pobre).
#
# Produce: varimp.png, threshold.png, roc.png, prcurve.png en dir_model/.
message("\n========================================")
message("Generando gráficos de diagnóstico")
message("========================================")

# -- Gráfico 1: Importancia de variables (Gain) --------------------------------
top20 <- varimp_xgb$importance |>
  rownames_to_column("variable") |>
  rename(importancia = Overall) |>
  arrange(desc(importancia)) |>
  slice_head(n = 20)    # nos quedamos con los 20 más importantes

p_varimp <- ggplot(top20, aes(x = reorder(variable, importancia),
                               y = importancia)) +
  geom_col(fill = "#534AB7", alpha = 0.85) +
  coord_flip() +
  labs(
    title    = paste0("XGBoost: importancia de variables (Gain) — ", MODEL_ID),
    subtitle = paste0("Top 20 predictores | nrounds = ", best_nrounds,
                      " | max_depth = ", best_max_depth,
                      " | eta = ", best_eta),
    x        = NULL,
    y        = "Importancia (Gain, normalizado a 100)"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "varimp.png"),
       p_varimp, width = 8, height = 6, dpi = 150)
message("  Guardado: varimp.png")

# AUC-ROC sobre train (en muestra — métrica optimista, sólo para diagnóstico)
roc_obj <- pROC::roc(
  response  = as.integer(y_train == "Yes"),
  predictor = probs_train,
  quiet     = TRUE
)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC (train, en muestra): ", round(auc_roc, 4))

# -- Gráfico 2: Threshold vs F1 / Precision / Recall (sobre predicciones OOF) --
# Usa th_oof_search ya calculado en la Sección 9 con probabilidades OOF.
# OOF es honesto (fuera de muestra): cada observación es predicha sólo por
# los modelos de los folds que no la incluyeron en el entrenamiento.
th_results <- th_oof_search   # renombramos para mantener compatibilidad

p_threshold <- th_results |>
  filter(!is.na(F1)) |>
  pivot_longer(cols = c(F1, Precision, Recall), names_to = "metrica") |>
  ggplot(aes(x = threshold, y = value, color = metrica)) +
  geom_line(linewidth = 0.9) +
  geom_vline(xintercept = best_th,
             linetype = "dashed", color = "#534AB7") +
  geom_vline(xintercept = 0.5,
             linetype = "dotted", color = "gray40") +
  annotate("text", x = best_th + 0.02, y = 0.08,
           label = sprintf("th = %.2f\n(óptimo OOF)", best_th),
           hjust = 0, size = 3.2, color = "#534AB7") +
  annotate("text", x = 0.52, y = 0.25,
           label = "th = 0.5", hjust = 0, size = 3.0, color = "gray40") +
  scale_color_manual(
    values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3"),
    labels = c(F1        = "F1 (balance)",
               Precision = "Precision (menos filtración)",
               Recall    = "Recall (menos exclusión)")
  ) +
  labs(
    title    = paste0("XGBoost: métricas por threshold (OOF) — ", MODEL_ID),
    subtitle = sprintf("Threshold óptimo = %.2f | F1 OOF = %.4f | nrounds = %d | eta = %.2f",
                       best_th, best_oof_f1, best_nrounds, best_eta),
    x        = "Threshold de clasificación",
    y        = "Valor de la métrica",
    color    = NULL,
    caption  = "Línea discontinua = threshold óptimo OOF (submission). Punteada = 0.5."
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

ggsave(file.path(dir_model, "threshold.png"),
       p_threshold, width = 8, height = 5, dpi = 150)
message("  Guardado: threshold.png")

# -- Gráfico 3: Curva ROC -------------------------------------------------------
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
    title    = paste0("XGBoost: curva ROC — ", MODEL_ID),
    subtitle = paste0("nrounds = ", best_nrounds,
                      " | eta = ", best_eta,
                      " | Probabilidades sobre train (en muestra)"),
    x        = "Tasa de Falsos Positivos  (1 - Specificity)",
    y        = "Tasa de Verdaderos Positivos  (Recall)",
    caption  = "AUC en muestra es optimista. Leer junto con la curva PR."
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc.png"),
       p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc.png")

# -- Gráfico 4: Curva Precision-Recall -----------------------------------------
# Derivada de la curva ROC: para cada threshold implícito, calcula Precision
# y Recall. Más informativa que la ROC en presencia de desbalance porque
# Precision depende de los falsos positivos (que son abundantes).
pr_df <- tibble(
  TP        = roc_obj$sensitivities        * sum(y_train == "Yes"),
  FP        = (1 - roc_obj$specificities) * sum(y_train == "No"),
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
    title    = paste0("XGBoost: curva Precision-Recall — ", MODEL_ID),
    subtitle = sprintf("AUC-ROC: %.4f | nrounds = %d | eta = %.2f | train (en muestra)",
                       auc_roc, best_nrounds, best_eta),
    x        = "Recall  (fracción de pobres capturada)",
    y        = "Precision  (fracción de predichos pobres que lo son)",
    caption  = "La curva por encima de la línea punteada indica mejor que el azar."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve.png"),
       p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve.png")

# Guardar todos los objetos de diagnóstico en un .rds para inspección posterior
# sin necesidad de re-entrenar el modelo.
saveRDS(
  list(xgb_cv     = xgb_cv,       # resultados de CV + finalModel de caret
       roc_obj    = roc_obj,       # objeto pROC para reconstruir la curva ROC
       th_results = th_results,    # data frame threshold vs métricas
       varimp     = varimp_xgb),   # importancias de variables (Gain)
  file.path(dir_model, "diagnostics.rds")
)
message("  Guardado: diagnostics.rds")

# =============================================================================
# SECCIÓN 11: SUBMISSION
# =============================================================================
# Aplica el modelo final al test set y genera el CSV en el formato de Kaggle:
# dos columnas — id (identificador del hogar) y pobre (0/1).
# Se usa best_th (threshold óptimo OOF), validado sobre predicciones fuera
# de muestra. No hay data leakage: el threshold fue elegido sobre folds que
# el modelo final nunca vio.
#
# Produce: submission_XGB_001.csv listo para subir a Kaggle.
message("\n========================================")
message("Generando submission para Kaggle")
message("========================================")

probs_test <- predict(xgb_cv,
                      newdata = X_test_mat,
                      type    = "prob")[, "Yes"]
pred_test  <- as.integer(probs_test >= best_th)

submission <- tibble(id = ids, pobre = pred_test)
sub_path   <- file.path(dir_subs, paste0("submission_", MODEL_ID, ".csv"))
write_csv(submission, sub_path)

message("  Guardado: ", sub_path)
message("  Threshold usado: ", best_th, " (óptimo OOF)")
message("  Pobres predichos: ", sum(submission$pobre),
        " (", round(mean(submission$pobre) * 100, 1), "%)")

# =============================================================================
# SECCIÓN 12: REGISTRO EN model_registry.csv
# =============================================================================
# Agrega (o actualiza si ya existe) una fila con los resultados de este modelo
# en la tabla maestra del equipo. La columna kaggle_public_F1 se deja en NA
# hasta que se suba el CSV a Kaggle y se anote el resultado manualmente.
#
# Produce: model_registry.csv actualizado con la fila de XGB_001.
message("\n========================================")
message("Registro en model_registry.csv")
message("========================================")

nueva_fila <- tibble(
  model_id           = MODEL_ID,
  fecha              = Sys.Date(),
  autor              = AUTOR,
  algoritmo          = "XGBoost",
  n_features         = ncol(X_train),
  imbalance_strategy = "none",
  cv_folds           = 5L,
  cv_F1              = round(best_f1_cv,     4),
  cv_Precision       = round(best_prec_cv,   4),
  cv_Recall          = round(best_recall_cv, 4),
  auc_roc            = round(auc_roc,        4),
  kaggle_public_F1   = NA_real_,       # llenar manualmente después de subir
  threshold          = best_th,        # threshold óptimo OOF (no 0.5)
  notas              = paste0(
    "XGBoost grilla reducida (8 combos). nrounds=", best_nrounds,
    " max_depth=", best_max_depth,
    " eta=", best_eta,
    " gamma=", best_gamma,
    " min_child_weight=", best_min_child_weight,
    " colsample=", best_colsample,
    " subsample=", best_subsample,
    ". F1 CV-5=", round(best_f1_cv, 4),
    ". threshold_oof=", best_th,
    " F1_oof=", round(best_oof_f1, 4), "."
  ),
  # columnas específicas de XGBoost (NA en modelos de otro algoritmo)
  nrounds            = best_nrounds,
  max_depth          = best_max_depth,
  eta                = best_eta,
  gamma              = best_gamma,
  min_child_weight   = best_min_child_weight,
  colsample_bytree   = best_colsample,
  subsample          = best_subsample,
  train_F1           = round(train_F1,        4),
  train_Precision    = round(train_Precision, 4),
  train_Recall       = round(train_Recall,    4)
)

# Si el registry ya existe: quitar la fila anterior de este MODEL_ID (si la hay)
# y agregar la nueva. Así cada corrida actualiza el registro sin duplicados.
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
