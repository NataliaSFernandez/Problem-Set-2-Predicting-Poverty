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
#                      Fijo en 0.5 (penalización moderada por split). El
#                      notebook usa c(0, 1); se fija en el punto medio para
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
#
# COMPATIBILIDAD — xgboost vs. caret:
#   caret::train() con method = "xgbTree" NO es compatible con xgboost >= 3.0.0.
#   La causa: xgboost 3.x cambió su objeto modelo interno a clases ALTREP
#   (Alternative Representation de R). caret intenta asignar
#   modelFit$xNames <- colnames(x) después del ajuste, operación que las
#   clases ALTREP no soportan ("ALTLIST classes must provide a Set_elt
#   method [class: XGBAltrepPointerClass, pkg: xgboost]").
#   caret está en modo mantenimiento y no tiene previsto corregirlo
#   (ver issue topepo/caret#1412).
#
#   Intentos fallidos de downgrade a xgboost 1.7.11.1:
#     - xgboost no publicó rama 2.x en CRAN (el salto fue 1.7.x → 3.x).
#     - xgboost 1.7.x no tiene binario prebuilt para R 4.5/arm64; requiere
#       compilación desde fuente.
#     - La compilación falla en macOS con Xcode 26 beta (SDK MacOSX26.x):
#       el SDK elige headers de distinta ubicación y clang++ no encuentra
#       <vector> aunque los headers existan.
#     Conclusión: el downgrade no es viable en este entorno.
#
#   Solución definitiva (Secciones 5 y 7):
#     Se elimina caret del flujo de entrenamiento. La Sección 7 implementa
#     un CV-5 manual usando xgboost::xgb.train() directamente, construyendo
#     al final un objeto xgb_cv con la misma estructura que habría dejado
#     caret (campos results, bestTune, pred, finalModel) para que las
#     secciones 8–12 no requieran cambios. La importancia de variables
#     (Sección 9) y las predicciones sobre test (Sección 11) usan también
#     la API nativa. caret sigue cargado sólo para caret::createFolds().
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
  caret,      # createFolds() para el CV manual de la Sección 7
  xgboost,    # xgb.train() / xgb.DMatrix() — API nativa (compatible con v3.x)
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
AUTOR    <- "Jonathan"
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
#
# Los datos ya vienen limpios del script de preprocesamiento: no hay NAs.
#
# Produce: X_train y X_test completamente numéricos, y_train factor.
message("\n== Preprocesando variables ==")

preparar_X_xgb <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>              # quitar target e ID
    mutate(zona = as.numeric(zona)) |>                 # "1"/"2" (chr) → numérico
    select(-any_of(c("ciudad", "dpto"))) |>            # alta cardinalidad → excluir
    mutate(across(where(is.character), as.numeric)) |> # "0"/"1" chr → numérico
    mutate(across(where(is.factor),   as.integer))     # factor → código entero
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
# SECCIÓN 5: trainControl (no aplica — CV implementado con API nativa)
# =============================================================================
# caret::train() + trainControl() no se usan porque caret es incompatible con
# xgboost >= 3.x (ver sección COMPATIBILIDAD en el encabezado). El CV se
# implementa directamente en la Sección 7 con xgboost::xgb.train() y
# caret::createFolds(). La lógica equivalente (5 folds, F1 como métrica,
# predicciones OOF) se reproduce manualmente.

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
#   gamma            = 0.5  (penalización moderada por split; punto medio)
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
  gamma            = 0.5,           # penalización moderada por split (fijo)
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
# SECCIÓN 7: ENTRENAMIENTO CON CV (API nativa xgboost)
# =============================================================================
# Implementación manual del mismo CV-5 que habría hecho caret::train():
#   1. caret::createFolds() genera los índices de los 5 folds.
#   2. Para cada combinación de la grilla × cada fold:
#      - xgb.DMatrix() empaqueta el subconjunto de entrenamiento y validación.
#      - xgb.train() ajusta el modelo sobre los 4 folds de entrenamiento.
#      - predict() devuelve P(pobre=1|X) para el fold de validación.
#      - Se calculan F1, Precision y Recall con threshold = 0.5.
#      - Se guardan las predicciones OOF (out-of-fold) para la búsqueda de
#        threshold óptimo en la Sección 9.
#   3. Se promedia F1 sobre los 5 folds por combinación.
#   4. Se selecciona la combinación con mayor F1 promedio.
#   5. Se re-entrena el modelo final sobre TODOS los datos de train con
#      la combinación óptima.
#
# Al finalizar se construye la lista xgb_cv con los campos que esperan
# las secciones 8–12: results, bestTune, pred (OOF del mejor combo),
# finalModel. Esto permite reutilizar todo el código downstream sin cambios.
#
# nthread paraleliza la construcción de cada árbol (OpenMP, no los folds).
#
# Produce: lista xgb_cv con resultados de CV, bestTune, OOF preds y
#          finalModel (objeto xgboost::xgb.Booster).
message("\n========================================")
message("Entrenando XGBoost con CV-5")
message("========================================")

set.seed(42)
t0 <- Sys.time()

# Partición reproducible en 5 folds estratificados por y_train
folds        <- caret::createFolds(y_train, k = 5, list = TRUE,
                                   returnTrain = TRUE)
n_combos     <- nrow(tune_grid)
cv_rows      <- vector("list", n_combos)   # métricas promedio por combo
oof_by_combo <- vector("list", n_combos)   # predicciones OOF por combo

for (i in seq_len(n_combos)) {
  p        <- tune_grid[i, ]
  fold_met <- matrix(NA_real_, nrow = 5, ncol = 3,
                     dimnames = list(NULL, c("F1", "Precision", "Recall")))
  fold_oof <- vector("list", 5)

  for (f in seq_len(5)) {
    tr_idx  <- folds[[f]]
    val_idx <- setdiff(seq_len(nrow(X_train_mat)), tr_idx)

    dtrain_f <- xgboost::xgb.DMatrix(
      X_train_mat[tr_idx,  ],
      label = as.integer(y_train[tr_idx]  == "Yes"))
    dval_f <- xgboost::xgb.DMatrix(
      X_train_mat[val_idx, ],
      label = as.integer(y_train[val_idx] == "Yes"))

    m_f <- xgboost::xgb.train(
      params = list(
        objective        = "binary:logistic",
        eval_metric      = "logloss",
        max_depth        = p$max_depth,
        eta              = p$eta,
        gamma            = p$gamma,
        min_child_weight = p$min_child_weight,
        colsample_bytree = p$colsample_bytree,
        subsample        = p$subsample,
        nthread          = N_THREADS
      ),
      data    = dtrain_f,
      nrounds = p$nrounds,
      verbose = 0
    )

    message(sprintf(
      "+ Fold%d: eta=%.2f, max_depth=%d, gamma=%.1f, colsample_bytree=%.2f, min_child_weight=%d, subsample=%.1f, nrounds=%d",
      f, p$eta, p$max_depth, p$gamma, p$colsample_bytree,
      p$min_child_weight, p$subsample, p$nrounds))

    prob_val <- predict(m_f, dval_f)
    pred_val <- factor(ifelse(prob_val >= 0.5, "Yes", "No"), levels = niveles)
    obs_val  <- y_train[val_idx]

    TP <- sum(pred_val == "Yes" & obs_val == "Yes")
    FP <- sum(pred_val == "Yes" & obs_val != "Yes")
    FN <- sum(pred_val == "No"  & obs_val == "Yes")
    fold_met[f, "F1"]        <- if ((2*TP+FP+FN) == 0) NA_real_ else 2*TP/(2*TP+FP+FN)
    fold_met[f, "Precision"] <- if ((TP+FP) == 0)       NA_real_ else TP/(TP+FP)
    fold_met[f, "Recall"]    <- if ((TP+FN) == 0)       NA_real_ else TP/(TP+FN)

    fold_oof[[f]] <- tibble(
      rowIndex         = val_idx,
      Yes              = prob_val,
      No               = 1 - prob_val,
      obs              = obs_val,
      nrounds          = p$nrounds,
      max_depth        = p$max_depth,
      eta              = p$eta,
      gamma            = p$gamma,
      min_child_weight = p$min_child_weight,
      colsample_bytree = p$colsample_bytree,
      subsample        = p$subsample
    )
  }

  avg_met    <- colMeans(fold_met, na.rm = TRUE)
  cv_rows[[i]] <- tibble(
    nrounds          = p$nrounds,
    max_depth        = p$max_depth,
    eta              = p$eta,
    gamma            = p$gamma,
    min_child_weight = p$min_child_weight,
    colsample_bytree = p$colsample_bytree,
    subsample        = p$subsample,
    F1               = avg_met["F1"],
    Precision        = avg_met["Precision"],
    Recall           = avg_met["Recall"]
  )
  oof_by_combo[[i]] <- bind_rows(fold_oof)
}

# Mejor combinación según F1 promedio CV-5
cv_results_df <- bind_rows(cv_rows)
best_idx      <- which.max(cv_results_df$F1)
best_params   <- tune_grid[best_idx, ]

# Modelo final: re-entrenamiento sobre TODOS los datos de train
dtrain_full <- xgboost::xgb.DMatrix(
  X_train_mat, label = as.integer(y_train == "Yes"))

set.seed(42)
final_model <- xgboost::xgb.train(
  params = list(
    objective        = "binary:logistic",
    eval_metric      = "logloss",
    max_depth        = best_params$max_depth,
    eta              = best_params$eta,
    gamma            = best_params$gamma,
    min_child_weight = best_params$min_child_weight,
    colsample_bytree = best_params$colsample_bytree,
    subsample        = best_params$subsample,
    nthread          = N_THREADS
  ),
  data    = dtrain_full,
  nrounds = best_params$nrounds,
  verbose = 0
)

# Estructura compatible con secciones 8–12
# (mismos campos que habría dejado caret::train())
xgb_cv <- list(
  results    = cv_results_df,          # métricas CV por combinación
  bestTune   = best_params,            # mejor combo (data.frame 1 fila)
  pred       = oof_by_combo[[best_idx]], # predicciones OOF del mejor combo
  finalModel = final_model             # modelo entrenado en todo train
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
#   2. Calcular importancias de variables.
#
# Produce: best_th (threshold óptimo OOF), métricas OOF, varimp_xgb.
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

# Importancia de variables por Gain (reducción de la función de pérdida
# logística atribuida a cada predictor al usarse como criterio de split).
# xgb.importance() devuelve un data.table con columnas Feature y Gain.
# Se convierte a data.frame con rownames = Feature y columna Overall
# normalizada a [0, 100], replicando la estructura que devuelve varImp().
imp_raw    <- xgboost::xgb.importance(model = final_model)
imp_df     <- data.frame(
  Overall   = imp_raw$Gain / max(imp_raw$Gain) * 100,
  row.names = imp_raw$Feature
)
varimp_xgb <- list(importance = imp_df)

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

# AUC-ROC sobre predicciones OOF del CV (honesto — fuera de muestra).
# probs_oof fue extraído en Sección 9 de xgb_cv$pred filtrado por best params.
roc_obj <- pROC::roc(
  response  = as.integer(oof_preds$obs == "Yes"),
  predictor = probs_oof,
  quiet     = TRUE
)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC (OOF, CV-5, honesto): ", round(auc_roc, 4))

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
                      " | Probabilidades OOF — CV-5 (fuera de muestra)"),
    x        = "Tasa de Falsos Positivos  (1 - Specificity)",
    y        = "Tasa de Verdaderos Positivos  (Recall)",
    caption  = "AUC calculado sobre predicciones OOF del CV: estimación honesta."
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc.png"),
       p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc.png")

# -- Gráfico 4: Curva Precision-Recall -----------------------------------------
# Derivada de la curva ROC OOF: para cada threshold implícito calcula Precision
# y Recall sobre las mismas predicciones fuera de muestra.
n_pos_oof <- sum(oof_preds$obs == "Yes")
n_neg_oof <- sum(oof_preds$obs != "Yes")
pr_df <- tibble(
  TP        = roc_obj$sensitivities       * n_pos_oof,
  FP        = (1 - roc_obj$specificities) * n_neg_oof,
  recall    = roc_obj$sensitivities,
  precision = TP / (TP + FP)
) |>
  filter(is.finite(precision), is.finite(recall))

prev_oof <- n_pos_oof / (n_pos_oof + n_neg_oof)
p_prcurve <- ggplot(pr_df, aes(x = recall, y = precision)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_hline(yintercept = prev_oof,
             linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.85, y = prev_oof + 0.015,
           label = sprintf("Clasificador aleatorio (%.0f%%)", prev_oof * 100),
           size = 3, color = "gray50") +
  labs(
    title    = paste0("XGBoost: curva Precision-Recall — ", MODEL_ID),
    subtitle = sprintf("AUC-ROC OOF: %.4f | nrounds = %d | eta = %.2f | OOF — CV-5",
                       auc_roc, best_nrounds, best_eta),
    x        = "Recall  (fracción de pobres capturada)",
    y        = "Precision  (fracción de predichos pobres que lo son)",
    caption  = "Curva calculada sobre predicciones OOF del CV: estimación honesta."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve.png"),
       p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve.png")

# Guardar todos los objetos de diagnóstico en un .rds para inspección posterior
# sin necesidad de re-entrenar el modelo.
saveRDS(
  list(xgb_cv     = xgb_cv,       # resultados de CV + finalModel de caret
       roc_obj    = roc_obj,       # objeto pROC — OOF (honesto)
       th_results = th_results,    # data frame threshold vs métricas OOF
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

dtest      <- xgboost::xgb.DMatrix(X_test_mat)
probs_test <- predict(final_model, dtest)   # devuelve P(pobre=1|X) directamente
pred_test  <- as.integer(probs_test >= best_th)

submission <- tibble(id = ids, pobre = pred_test)
sub_name   <- sprintf("XGB_nrounds%d_depth%d_eta%03.0f_th%03.0f.csv",
                      best_nrounds, best_max_depth,
                      best_eta * 100, best_th * 100)
sub_path   <- file.path(dir_subs, sub_name)
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
  subsample          = best_subsample
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
