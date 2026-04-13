#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/RandomForest/RF_001.R
#==============================================================================
#
# ALGORITMO: Random Forest (ranger, via caret)
#
# QUÉ HACE ESTE SCRIPT:
#   Entrena un Random Forest de clasificación usando ranger (implementación
#   eficiente de RF en C++) orquestado por caret. El modelo construye 500
#   árboles de decisión, cada uno sobre un bootstrap del conjunto de entrenamiento.
#   En cada nodo, selecciona aleatoriamente mtry predictores candidatos antes
#   de buscar la mejor división. La predicción final es la media de probabilidades
#   de los 500 árboles.
#
#   Ventaja sobre CART: la aleatorización doble (bootstrap + subespacio de
#   variables) decorrelaciona los árboles, reduciendo la varianza sin aumentar
#   el sesgo y mejorando la generalización fuera de muestra.
#
# HIPERPARÁMETROS TUNADOS CON CV-5 (notebook Lecture10, Tree Methods):
#
#   mtry          : número de predictores candidatos seleccionados al azar en
#                   cada nodo. Controla el trade-off bias-varianza:
#                   - mtry pequeño → árboles más diferentes entre sí (menor
#                     correlación), menor varianza del ensamble, pero cada
#                     árbol es peor individualmente (más sesgo).
#                   - mtry grande → árboles más similares entre sí (mayor
#                     correlación), más varianza del ensamble.
#                   El notebook usa c(1, 2, 3) para 3 predictores.
#                   Aquí: c(20, 30, 50) — valores altos porque el preprocesamiento
#                   produce ~108 predictores y queremos explorar más variables
#                   por nodo (sqrt(108) ≈ 10 es conservador para este dataset).
#
#   splitrule     : criterio de división de cada nodo.
#                   "gini" para clasificación (índice de Gini).
#                   El notebook usa "variance" porque predice una variable
#                   continua (log de precio). Aquí la tarea es binaria.
#
#   min.node.size : mínimo de observaciones que debe tener un nodo terminal.
#                   Un valor bajo permite árboles profundos (más flexibles
#                   pero con riesgo de overfitting). Valores: c(1, 3, 5),
#                   tomados directamente del notebook.
#
# PARÁMETROS FIJOS:
#   num.trees  = 150  (reducido de 500 para controlar costo computacional;
#                      suficiente para estabilidad de la media de probabilidades
#                      con 108 predictores y 164k observaciones)
#   importance = "permutation"  (activado SOLO en el modelo final, NO en CV;
#                                mide cuánto cae el OOB accuracy al permutar
#                                aleatoriamente cada predictor; en CV añade ~2×
#                                el tiempo de cada fit sin mejorar el tuning)
#
# FLUJO DEL SCRIPT:
#   0. Paquetes
#   1. Configuración global (AUTOR, MODEL_ID, rutas, directorios)
#   2. Cargar datos
#   3. Preprocesamiento
#   4. Funciones auxiliares (F1, Precision, Recall)
#   5. trainControl (CV-5, métrica F1)
#   6. Grilla de hiperparámetros (9 combinaciones)
#   7. Entrenamiento con CV
#   8. Resultados del CV y mejores hiperparámetros
#   9. Modelo final (re-entrenado sobre todo train con probability = TRUE)
#   10. Importancia de variables — gráfico Top 20 (varimp.png)
#   11. Gráficos de diagnóstico (threshold.png, roc.png, prcurve.png)
#   12. Submission CSV (threshold óptimo sobre predicciones OOF del CV)
#   13. Registro en model_registry.csv
#
# INPUTS (relativos a la raíz del proyecto):
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#
# OUTPUTS (relativos a la raíz del proyecto):
#   03_submissions/submission_RF_001.csv                   ← subir a Kaggle
#   02_outputs/models/RandomForest/RF_001/varimp.png
#   02_outputs/models/RandomForest/RF_001/feature_matrix.csv  ← todas las vars
#   02_outputs/models/RandomForest/RF_001/threshold.png
#   02_outputs/models/RandomForest/RF_001/roc.png
#   02_outputs/models/RandomForest/RF_001/prcurve.png
#   02_outputs/models/RandomForest/RF_001/diagnostics.rds
#   02_outputs/model_registry.csv                         ← tabla maestra (append)
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
  tidyverse,   # dplyr (manipulación), ggplot2 (gráficos), purrr (map_dfr), tibble
  caret,       # train(), trainControl(), confusionMatrix(), varImp()
  ranger,      # Random Forest en C++; usado internamente por caret (method = "ranger")
  pROC,        # roc(), auc() — curva ROC y AUC
  fs,          # dir_create() — crear carpetas de forma multiplataforma
  doParallel,  # backend paralelo para foreach (usado por caret en CV)
  parallel     # detectCores() — detecta el número de cores disponibles
)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
# Define las constantes de identificación, la semilla global y las rutas de
# salida. Crear los directorios aquí evita errores al guardar archivos más tarde.
#
# Produce: variables AUTOR, MODEL_ID, rutas y carpetas de output creadas.
AUTOR    <- "Jonathan Melo"
MODEL_ID <- "RF_001"

# Semilla global — todos los scripts del equipo usan 42 para comparabilidad
set.seed(42)

dir_model <- file.path("02_outputs/models/RandomForest", MODEL_ID)
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model,    recurse = TRUE)
fs::dir_create(dir_subs,     recurse = TRUE)
fs::dir_create("02_outputs", recurse = TRUE)

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
# ranger puede manejar factores nativamente, a diferencia de glmnet o xgboost.
# Se excluyen ciudad y dpto por su alta cardinalidad (24+ niveles): incluirlos
# fragmentaría los árboles en nodos con muy pocas observaciones sin mejorar
# la generalización. La misma decisión se aplica en el script CART.
#
# Produce: X_train y X_test alineados (mismas columnas), y_train como factor.
message("\n== Preprocesando variables ==")

preparar_X_rf <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>        # quitar target e ID
    mutate(zona = as.numeric(zona)) |>           # "1"/"2" (chr) → numérico
    select(-any_of(c("ciudad", "dpto"))) |>      # alta cardinalidad → excluir
    mutate(across(where(is.character), as.factor))  # resto de chr → factor
}

X_train <- preparar_X_rf(train)
X_test  <- preparar_X_rf(test)

# Alinear columnas: garantiza que train y test tengan exactamente las mismas
# variables en el mismo orden (evita errores en predict())
cols_comunes <- intersect(names(X_train), names(X_test))
X_train      <- X_train |> select(all_of(cols_comunes))
X_test       <- X_test  |> select(all_of(cols_comunes))

# Variable respuesta: factor con niveles válidos para caret.
# caret requiere identificadores de R válidos como niveles (no "0"/"1").
niveles <- c("No", "Yes")
y_train <- factor(ifelse(train$pobre == 1, "Yes", "No"), levels = niveles)

# train_rf combina X e y para usar la interfaz de fórmula en el modelo final
train_rf <- bind_cols(X_train, pobre = y_train)

message("  Predictores: ", ncol(X_train))
message("  NAs en X_train: ", sum(is.na(X_train)))

# =============================================================================
# SECCIÓN 4: FUNCIONES AUXILIARES
# =============================================================================
# prf_summary() es el summaryFunction que caret llama al final de cada fold.
# Retorna F1, Precision y Recall simultáneamente para que las tres métricas
# queden guardadas en rf_cv$results y podamos registrarlas.
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
# savePredictions = "final" guarda las predicciones OOF de la mejor
# combinación de hiperparámetros (útil si se quiere analizar errores por fold).
#
# Produce: objeto ctrl_cv pasado a train().
ctrl_cv <- trainControl(
  method          = "cv",
  number          = 5,
  summaryFunction = prf_summary,   # retorna F1, Precision, Recall por fold
  classProbs      = TRUE,          # necesario para guardar P(Yes) en rf_cv$pred → threshold OOF
  verboseIter     = TRUE,          # imprime el progreso de cada fold en consola
  savePredictions = "final"        # guarda columnas rowIndex, obs, pred, Yes, No
)

# =============================================================================
# SECCIÓN 6: GRILLA DE HIPERPARÁMETROS
# =============================================================================
# Estructura tomada del notebook Lecture10 (Tree Methods):
#   expand.grid(mtry, splitrule, min.node.size)
# Los valores de mtry se escalan a ~100 predictores reales.
# Resultado: 3 × 1 × 3 = 9 combinaciones × 5 folds = 45 fits totales.
#
# Produce: data frame tune_grid con todas las combinaciones a evaluar.
tune_grid <- expand.grid(
  mtry          = c(20, 30, 50),    # candidatos por nodo; valores altos explorados intencionalmente
  splitrule     = "gini",           # criterio de clasificación (Gini impurity)
  min.node.size = c(1, 3, 5)        # profundidad del árbol controlada por nodo mínimo
)

message("\n== Grilla RF ==")
message("  Combinaciones: ", nrow(tune_grid), " (3 mtry × 3 min.node.size)")
message("  Fits totales:  ", nrow(tune_grid) * 5, " (grilla × 5 folds)")
message("  Tiempo estimado: 2.5-3.5 h con 164k obs, 150 árboles, sin importance en CV")

# =============================================================================
# SECCIÓN 7: ENTRENAMIENTO CON CV
# =============================================================================
# caret::train() con method = "ranger":
#   1. Divide train en 5 folds.
#   2. Para cada una de las 9 combinaciones de hiperparámetros:
#      - Entrena ranger en 4 folds.
#      - Evalúa F1 en el fold restante.
#      - Promedia F1 sobre los 5 folds.
#   3. Selecciona la combinación con mayor F1 promedio.
#   4. Re-entrena automáticamente el modelo final con esa combinación
#      sobre TODOS los datos de train.
#
# Produce: objeto rf_cv con resultados de CV, bestTune y finalModel.
message("\n========================================")
message("Entrenando RF con CV-5")
message("========================================")

# Paralelización: cada uno de los 45 fits del CV corre en un core separado.
# Limitamos ranger a num.threads = 1 dentro de cada fit para evitar que los
# procesos hijo compitan entre sí por los cores (oversubscription).
# Con N cores disponibles: 45 fits / N cores ≈ ceil(45/N) rondas en paralelo.
n_cores <- max(1L, parallel::detectCores() - 1L)
cl <- makeCluster(n_cores)
registerDoParallel(cl)
message("  Cores registrados para CV paralelo: ", n_cores)

set.seed(42)
t0 <- Sys.time()

rf_cv <- train(
  x           = X_train,
  y           = y_train,
  method      = "ranger",       # usa el paquete ranger internamente
  trControl   = ctrl_cv,
  tuneGrid    = tune_grid,
  metric      = "F1",           # caret selecciona el modelo que maximiza F1
  num.trees   = 150,            # 150 árboles — más rápido que 500, suficiente para CV
  num.threads = 1L              # 1 thread por fit: la paralalización la hace doParallel
  # importance NO se activa aquí: calcularla en cada uno de los 45 fits
  # duplica el tiempo de entrenamiento. Se calcula sólo en el modelo final.
)

t1 <- Sys.time()
stopCluster(cl)   # liberar workers al terminar el CV
message("  Tiempo CV: ",
        round(as.numeric(difftime(t1, t0, units = "mins")), 1), " min")

# =============================================================================
# SECCIÓN 8: RESULTADOS DEL CV
# =============================================================================
# rf_cv$results contiene el F1, Precision y Recall promedio de los 5 folds
# para cada combinación de hiperparámetros.
# rf_cv$bestTune indica la combinación óptima según F1.
#
# Produce: tabla impresa en consola, variables best_* para el registro.
cat("\n-- Resultados por combinación (CV-5, ordenados por F1 desc) --\n")
print(rf_cv$results |>
        arrange(desc(F1)) |>
        mutate(across(where(is.numeric), ~ round(., 4))))

best_mtry      <- rf_cv$bestTune$mtry
best_splitrule <- rf_cv$bestTune$splitrule
best_min_node  <- rf_cv$bestTune$min.node.size
best_f1_cv     <- max(rf_cv$results$F1, na.rm = TRUE)

# Extraer Precision y Recall de la fila con los mejores hiperparámetros
best_row       <- rf_cv$results |>
                    filter(mtry == best_mtry, min.node.size == best_min_node)
best_prec_cv   <- best_row$Precision
best_recall_cv <- best_row$Recall

message("\n  Mejor mtry:          ", best_mtry)
message("  Mejor min.node.size: ", best_min_node)
message("  Mejor F1 CV-5:       ", round(best_f1_cv,     4))
message("  Precision CV-5:      ", round(best_prec_cv,   4))
message("  Recall CV-5:         ", round(best_recall_cv, 4))

# =============================================================================
# SECCIÓN 8b: PREDICCIONES OOF DEL CV — PARA CURVAS ROC Y PR
# =============================================================================
# classProbs=TRUE en trainControl activó la columna "Yes" (P̂(pobre=1|X))
# en rf_cv$pred. Filtramos la mejor combinación de hiperparámetros para
# obtener las predicciones OOF (out-of-fold): cada observación fue predicha
# exactamente una vez, por el fold en que NO participó del entrenamiento.
# Estas predicciones se usarán en la Sección 11 para construir las curvas
# ROC y Precision-Recall honestas (fuera de muestra).
#
# Nota: el threshold óptimo se calcula sobre predicciones OOB del modelo final
# en la Sección 9 (estimación nativa de ranger, bootstrap-based).
#
# Produce: probs_oof y labels_oof (vectores alineados con y_train) para ROC/PR.
message("\n== Extrayendo predicciones OOF del CV para curvas ROC/PR ==")

oof_preds  <- rf_cv$pred |>
  filter(mtry          == best_mtry,
         min.node.size == best_min_node) |>
  arrange(rowIndex)   # garantiza orden original de observaciones

probs_oof  <- oof_preds$Yes
labels_oof <- oof_preds$obs
message("  Predicciones OOF extraídas: ", length(probs_oof), " observaciones")

# =============================================================================
# SECCIÓN 9: MODELO FINAL
# =============================================================================
# Aunque caret ya guardó un finalModel, lo re-entrenamos con ranger() directo
# para activar probability = TRUE y importance = "permutation". Estas opciones
# no están disponibles desde la interfaz de caret pero son necesarias para:
#   - probability=TRUE  → predicciones OOB continuas P̂(pobre=1|X) y probs en test
#   - importance="permutation" → importancia por permutación para análisis de features
#
# El threshold se optimiza sobre predicciones OOB (out-of-bag): cada árbol
# predice sólo las observaciones que NO usó para entrenar → estimación honesta
# nativa de ranger sin necesidad de un CV adicional.
# Las curvas ROC y PR (Sección 11) usan predicciones OOF del CV (Sección 8b).
#
# Produce: rf_final (objeto ranger con OOB probs), probs_oob, best_th (OOB),
#          best_oob_f1/pr/rc, th_oob_search, probs_train, cm_final, métricas train_*.
message("\n========================================")
message("Entrenando modelo final sobre todos los datos de train")
message("========================================")

set.seed(42)
rf_final <- ranger::ranger(
  pobre ~ .,
  data          = train_rf,          # todos los datos de entrenamiento
  num.trees     = 150,               # mismo número que CV para consistencia
  mtry          = best_mtry,
  splitrule     = best_splitrule,
  min.node.size = best_min_node,
  probability   = TRUE,              # activa predicciones de probabilidad [0,1]
  importance    = "permutation",     # sólo aquí (no en CV) → añade ~50% del tiempo
  seed          = 42
)

# Probabilidades OOB: cada árbol predice sólo las observaciones que NO usó
# para entrenar → estimación honesta fuera de muestra, nativa de ranger.
probs_oob <- rf_final$predictions[, "Yes"]

# Búsqueda del threshold que maximiza F1 sobre probabilidades OOB
th_grid <- seq(0.05, 0.95, by = 0.01)

th_oob_search <- map_dfr(th_grid, function(th) {
  pred_th <- factor(ifelse(probs_oob >= th, "Yes", "No"), levels = niveles)
  TP <- sum(pred_th == "Yes" & y_train == "Yes")
  FP <- sum(pred_th == "Yes" & y_train == "No")
  FN <- sum(pred_th == "No"  & y_train == "Yes")
  f1   <- if ((2*TP + FP + FN) == 0) NA_real_ else 2*TP / (2*TP + FP + FN)
  prec <- if ((TP + FP) == 0)        NA_real_ else TP / (TP + FP)
  rec  <- if ((TP + FN) == 0)        NA_real_ else TP / (TP + FN)
  tibble(threshold = th, F1 = f1, Precision = prec, Recall = rec)
})

best_th_row  <- th_oob_search |> filter(!is.na(F1)) |>
                  slice_max(F1, n = 1, with_ties = FALSE)
best_th      <- best_th_row$threshold
best_oob_f1  <- best_th_row$F1
best_oob_pr  <- best_th_row$Precision
best_oob_rc  <- best_th_row$Recall

message("  Threshold óptimo (OOB, max F1): ", best_th)
message("  F1 OOB:        ", round(best_oob_f1, 4))
message("  Precision OOB: ", round(best_oob_pr, 4))
message("  Recall OOB:    ", round(best_oob_rc, 4))

# Métricas en train con threshold óptimo OOB (en muestra — sólo diagnóstico)
probs_train <- predict(rf_final, data = X_train)$predictions[, "Yes"]
pred_train  <- factor(ifelse(probs_train >= best_th, "Yes", "No"), levels = niveles)

cm_final        <- confusionMatrix(pred_train, y_train, positive = "Yes")
train_F1        <- cm_final$byClass["F1"]
train_Precision <- cm_final$byClass["Precision"]
train_Recall    <- cm_final$byClass["Recall"]

cat("\n-- Métricas modelo final (train, en muestra, threshold = ", best_th, ") --\n")
cat("  F1:        ", round(train_F1,        4), "\n")
cat("  Precision: ", round(train_Precision, 4), "\n")
cat("  Recall:    ", round(train_Recall,    4), "\n")
# NOTA: estas métricas son en muestra (optimistas). La referencia confiable
# es el F1 OOB con threshold óptimo calculado arriba.

# =============================================================================
# SECCIÓN 10: IMPORTANCIA DE VARIABLES
# =============================================================================
# ranger calculó importance="permutation" en rf_final (Sección 9). varImp()
# extrae esas importancias: para cada predictor, permuta sus valores en OOB
# y mide cuánto cae la accuracy respecto al original. Un predictor importante
# produce una caída grande al permutarlo.
#
# Produce:
#   varimp.png           — gráfico Top 20 de predictores
#   feature_matrix.csv   — tabla completa de todas las variables con su importancia,
#                          ordenada descendentemente; útil para feature selection
#                          en iteraciones futuras (eliminar las de importancia ~0).
message("\n========================================")
message("Importancia de variables (permutation)")
message("========================================")

# rf_final$variable.importance es un named numeric vector con la importancia por
# permutación de cada predictor. Lo extraemos directamente del objeto ranger.
varimp_rf <- rf_final$variable.importance   # named vector: nombre → drop OOB accuracy

# Tabla completa de importancias — base para feature selection
feature_matrix <- data.frame(
  variable    = names(varimp_rf),
  importancia = as.numeric(varimp_rf)
) |>
  arrange(desc(importancia)) |>
  mutate(rank = row_number())

feat_mat_path <- file.path(dir_model, "feature_matrix.csv")
write_csv(feature_matrix, feat_mat_path)
message("  Guardado: feature_matrix.csv  (",
        nrow(feature_matrix), " variables, importancia por permutación)")

# Top 20 para el gráfico
top20 <- feature_matrix |>
  slice_head(n = 20)    # las 20 más importantes

p_varimp <- ggplot(top20, aes(x = reorder(variable, importancia),
                               y = importancia)) +
  geom_col(fill = "#534AB7", alpha = 0.85) +
  coord_flip() +
  labs(
    title    = paste0("RF: importancia de variables (permutation) — ", MODEL_ID),
    subtitle = paste0("Top 20 predictores | mtry = ", best_mtry,
                      " | min.node.size = ", best_min_node,
                      " | num.trees = 150"),
    x        = NULL,
    y        = "Importancia por permutación"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "varimp.png"),
       p_varimp, width = 8, height = 6, dpi = 150)
message("  Guardado: varimp.png")

# =============================================================================
# SECCIÓN 11: GRÁFICOS DE DIAGNÓSTICO
# =============================================================================
# Construimos tres gráficos con las probabilidades del modelo final sobre
# el train completo (métricas en muestra — optimistas, sólo para diagnóstico):
#
#   threshold.png : F1, Precision y Recall en función del threshold.
#                   Muestra dónde está el máximo F1 en train y dónde
#                   quedará la submission (threshold = 0.5).
#   roc.png       : curva ROC con el AUC. Evalúa la capacidad discriminante
#                   del modelo independiente del threshold.
#   prcurve.png   : curva Precision-Recall. Más informativa que ROC en
#                   datasets desbalanceados: muestra si el modelo es útil
#                   para detectar la clase minoritaria (pobre = 1).
#
# Produce: threshold.png, roc.png, prcurve.png en dir_model/.
message("\n========================================")
message("Generando gráficos de diagnóstico")
message("========================================")

# AUC-ROC sobre predicciones OOF del CV (honesto — fuera de muestra).
# probs_oof y labels_oof fueron extraídos en Sección 8b.
roc_obj <- pROC::roc(
  response  = as.integer(labels_oof == "Yes"),
  predictor = probs_oof,
  quiet     = TRUE
)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC (OOF, CV-5, honesto): ", round(auc_roc, 4))

# -- Gráfico 1: Threshold vs F1 / Precision / Recall (sobre predicciones OOB) --
# Usa th_oob_search calculado en Sección 9 con probabilidades OOB del modelo final.
# OOB es honesto (fuera de muestra): cada árbol predice sólo las observaciones
# que no usó durante su entrenamiento — mecanismo nativo de ranger.
th_results <- th_oob_search   # alias para compatibilidad con saveRDS posterior

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
           label = sprintf("th = %.2f\n(óptimo OOB)", best_th),
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
    title    = paste0("RF: métricas por threshold (OOB) — ", MODEL_ID),
    subtitle = sprintf("Threshold óptimo = %.2f | F1 OOB = %.4f | mtry = %d | min.node.size = %d",
                       best_th, best_oob_f1, best_mtry, best_min_node),
    x        = "Threshold de clasificación",
    y        = "Valor de la métrica",
    color    = NULL,
    caption  = "Línea discontinua = threshold óptimo OOB (submission). Punteada = 0.5."
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

ggsave(file.path(dir_model, "threshold.png"),
       p_threshold, width = 8, height = 5, dpi = 150)
message("  Guardado: threshold.png")

# -- Gráfico 2: Curva ROC ------------------------------------------------------
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
    title    = paste0("RF: curva ROC — ", MODEL_ID),
    subtitle = paste0("mtry = ", best_mtry,
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

# -- Gráfico 3: Curva Precision-Recall ----------------------------------------
# Derivada de la curva ROC OOF: para cada threshold implícito calcula Precision
# y Recall sobre las mismas predicciones fuera de muestra. Más informativa que
# la ROC con desbalance porque Precision depende de los FP, que son abundantes.
n_pos <- sum(labels_oof == "Yes")
n_neg <- sum(labels_oof == "No")
pr_df <- tibble(
  TP        = roc_obj$sensitivities       * n_pos,
  FP        = (1 - roc_obj$specificities) * n_neg,
  recall    = roc_obj$sensitivities,
  precision = TP / (TP + FP)
) |>
  filter(is.finite(precision), is.finite(recall))

prev_oof <- n_pos / (n_pos + n_neg)   # prevalencia real en OOF
p_prcurve <- ggplot(pr_df, aes(x = recall, y = precision)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_hline(yintercept = prev_oof,
             linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.85, y = prev_oof + 0.015,
           label = sprintf("Clasificador aleatorio (%.0f%%)", prev_oof * 100),
           size = 3, color = "gray50") +
  labs(
    title    = paste0("RF: curva Precision-Recall — ", MODEL_ID),
    subtitle = sprintf("AUC-ROC OOF: %.4f | mtry = %d | OOF — CV-5 (fuera de muestra)",
                       auc_roc, best_mtry),
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
  list(rf_cv           = rf_cv,          # resultados de CV + finalModel de caret
       rf_final        = rf_final,       # modelo final con probability=TRUE e importance
       roc_obj         = roc_obj,        # objeto pROC — OOF (honesto)
       th_results      = th_results,     # data frame threshold vs métricas OOB
       varimp          = varimp_rf,      # named vector: variable → importancia permutación
       feature_matrix  = feature_matrix, # tabla completa de importancias
       best_th         = best_th,        # threshold óptimo OOB
       best_oob_f1     = best_oob_f1),   # F1 OOB con ese threshold
  file.path(dir_model, "diagnostics.rds")
)
message("  Guardado: diagnostics.rds")

# =============================================================================
# SECCIÓN 12: SUBMISSION
# =============================================================================
# Aplica el modelo final al test set y genera el CSV en el formato de Kaggle:
# dos columnas — id (identificador del hogar) y pobre (0/1).
# Se usa threshold = 0.5 (criterio de Bayes bajo pérdida 0-1 simétrica).
#
# Produce: submission_RF_001.csv listo para subir a Kaggle.
message("\n========================================")
message("Generando submission para Kaggle")
message("========================================")

probs_test <- predict(rf_final, data = X_test)$predictions[, "Yes"]
pred_test  <- as.integer(probs_test >= best_th)   # threshold óptimo OOF

submission <- tibble(id = ids, pobre = pred_test)
sub_name   <- sprintf("RF_mtry%d_minnodesize%d_ntrees150_th%03.0f.csv",
                      best_mtry, best_min_node, best_th * 100)
sub_path   <- file.path(dir_subs, sub_name)
write_csv(submission, sub_path)

message("  Guardado: ", sub_path)
message("  Threshold usado: ", best_th, " (óptimo OOB, Sección 9)")
message("  Pobres predichos: ", sum(submission$pobre),
        " (", round(mean(submission$pobre) * 100, 1), "%)")

# =============================================================================
# SECCIÓN 13: REGISTRO EN model_registry.csv
# =============================================================================
# Agrega (o actualiza si ya existe) una fila con los resultados de este modelo
# en la tabla maestra del equipo. La columna kaggle_public_F1 se deja en NA
# hasta que se suba el CSV a Kaggle y se anote el resultado manualmente.
#
# Produce: model_registry.csv actualizado con la fila de RF_001.
message("\n========================================")
message("Registro en model_registry.csv")
message("========================================")

nueva_fila <- tibble(
  model_id           = MODEL_ID,
  fecha              = Sys.Date(),
  autor              = AUTOR,
  algoritmo          = "RandomForest",
  n_features         = ncol(X_train),
  imbalance_strategy = "none",
  cv_folds           = 5L,
  # cv_F1/Precision/Recall: calculados sobre predicciones OOB del modelo final
  # con threshold óptimo OOB (estimación nativa de ranger, honesta).
  # auc_roc calculado sobre predicciones OOF del CV (también honesto).
  cv_F1              = round(best_oob_f1, 4),
  cv_Precision       = round(best_oob_pr, 4),
  cv_Recall          = round(best_oob_rc, 4),
  auc_roc            = round(auc_roc,     4),   # OOF, honesto
  kaggle_public_F1   = NA_real_,       # llenar manualmente después de subir
  threshold          = best_th,        # threshold óptimo OOB
  notas              = paste0(
    "RF ranger. mtry=", best_mtry,
    " min_node_size=", best_min_node,
    " splitrule=gini num.trees=150.",
    " Threshold optimizado sobre probs OOB (max F1).",
    " Curvas ROC/PR sobre OOF CV-5.",
    " th=", best_th,
    " F1_OOB=", round(best_oob_f1, 4),
    " AUC_OOF=", round(auc_roc, 4),
    " (CV-5 F1 th=0.5: ", round(best_f1_cv, 4), ")."
  ),
  cp             = NA_real_,
  maxdepth       = NA_real_,
  train_F1       = round(train_F1,        4),
  train_Precision= round(train_Precision, 4),
  train_Recall   = round(train_Recall,    4)
)

# Si el registry ya existe: quitar la fila anterior de este MODEL_ID (si la hay)
# y agregar la nueva. Así cada corrida actualiza el registro sin duplicados.
if (file.exists(reg_path)) {
  registry <- read_csv(reg_path, show_col_types = FALSE) |>
    mutate(fecha = as.Date(fecha, origin = "1899-12-30"))  # serial Excel → Date
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
