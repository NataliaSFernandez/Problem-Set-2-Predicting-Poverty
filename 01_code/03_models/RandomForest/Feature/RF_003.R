#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/RandomForest/RF_003.R
#==============================================================================
#
# ALGORITMO: Random Forest (ranger, via caret)
#
# DIFERENCIA CON RF_001:
#   RF_001 excluía ciudad y dpto por "alta cardinalidad". RF_003 las incluye
#   como factores tratados con respect.unordered.factors = "order".
#
#   La justificación metodológica es la siguiente: ciudad y dpto no son
#   variables Bernoulli independientes — son realizaciones de una misma
#   distribución multinomial subyacente que refleja efectos geográficos
#   (infraestructura, mercado laboral local, transferencias regionales, etc.).
#   Convertirlas a dummies (one-hot encoding) introduce 100+ columnas binarias
#   que (a) fragmentan el espacio de splits, (b) pierden la estructura de
#   distribución común y (c) no aportan nada que la codificación ordinal no
#   haga mejor.
#
#   La opción respect.unordered.factors = "order" en ranger implementa lo
#   siguiente en cada nodo:
#     1. Ordena los k niveles del factor por su media de la variable respuesta
#        en ese nodo (tasa local de pobreza).
#     2. Evalúa k-1 cortes binarios sobre ese ordenamiento.
#     3. Selecciona el corte con menor impureza Gini.
#   Esto es O(k log k) por nodo — eficiente incluso con 300+ municipios — y
#   captura la señal geográfica sin presuponer que las ciudades son
#   distribuciones independientes entre sí.
#
# HIPERPARÁMETROS TUNADOS CON CV-5 (mismos que RF_001):
#
#   mtry          : c(20, 30, 50)
#   splitrule     : "gini"
#   min.node.size : c(1, 3, 5)
#
# PARÁMETROS FIJOS:
#   num.trees = 150
#   importance = "permutation"  (solo en modelo final)
#
# FLUJO DEL SCRIPT:
#   0. Paquetes
#   1. Configuración global (AUTOR, MODEL_ID, rutas, directorios)
#   2. Cargar datos
#   3. Preprocesamiento (ciudad y dpto incluidas como factores; niveles alineados)
#   4. Funciones auxiliares (F1, Precision, Recall)
#   5. trainControl (CV-5, métrica F1)
#   6. Grilla de hiperparámetros (9 combinaciones)
#   7. Entrenamiento con CV
#   8. Resultados del CV y mejores hiperparámetros
#   8b. Predicciones OOF del CV para curvas ROC y PR
#   9. Modelo final (probability=TRUE, importance="permutation")
#   10. Importancia de variables — gráfico Top 20 y feature_matrix.csv
#   11. Gráficos de diagnóstico (threshold.png, roc.png, prcurve.png)
#   12. Submission CSV
#   13. Registro en model_registry.csv
#
# INPUTS (relativos a la raíz del proyecto):
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#
# OUTPUTS (relativos a la raíz del proyecto):
#   03_submissions/RF_mtry{M}_minnodesize{N}_ntrees150_th{T}.csv
#   02_outputs/models/RandomForest/RF_003/varimp.png
#   02_outputs/models/RandomForest/RF_003/feature_matrix.csv
#   02_outputs/models/RandomForest/RF_003/threshold.png
#   02_outputs/models/RandomForest/RF_003/roc.png
#   02_outputs/models/RandomForest/RF_003/prcurve.png
#   02_outputs/models/RandomForest/RF_003/diagnostics.rds
#   02_outputs/model_registry.csv
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
AUTOR    <- "Jonathan Melo"
MODEL_ID <- "RF_003"

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
message("== Cargando datos ==")

train <- readRDS("00_data/processed/train_final.rds")
test  <- readRDS("00_data/processed/test_final.rds")
ids   <- test$id

message("Train: ", nrow(train), " hogares | ", ncol(train), " columnas")
message("Test:  ", nrow(test),  " hogares | ", ncol(test),  " columnas")
message("Balance: ", round(mean(train$pobre) * 100, 1), "% pobres  /  ",
        round(mean(1 - train$pobre) * 100, 1), "% no pobres")

# =============================================================================
# SECCIÓN 3: PREPROCESAMIENTO
# =============================================================================
# A diferencia de RF_001, ciudad y dpto se conservan como factores.
# ranger los maneja nativamente con respect.unordered.factors = "order":
# en cada nodo ordena los k niveles del factor por su media de la variable
# respuesta y evalúa k-1 cortes binarios — captura la señal geográfica sin
# convertir a dummies ni tratar cada ciudad como una Bernoulli independiente.
#
# Los niveles se alinean entre train y test (unión de niveles) para que
# ranger no encuentre niveles desconocidos durante predict(). Municipios que
# aparezcan solo en test quedarán como NA; el árbol los enrutará por la rama
# mayoritaria del nodo, comportamiento conservador y transparente.
#
# Produce: X_train y X_test alineados (mismas columnas), y_train como factor.
message("\n== Preprocesando variables ==")

preparar_X_rf <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>         # quitar target e ID
    mutate(zona   = as.numeric(zona),             # "1"/"2" (chr) → numérico
           ciudad = as.factor(ciudad),            # factor: municipio (alta cardinalidad)
           dpto   = as.factor(dpto)) |>           # factor: departamento
    mutate(across(where(is.character), as.factor))
}

X_train_raw <- preparar_X_rf(train)
X_test_raw  <- preparar_X_rf(test)

# Alinear niveles de ciudad y dpto: unión train ∪ test.
# Sin esto, niveles presentes en test pero ausentes en train quedan fuera
# del factor y ranger los trata como NA durante predicción.
for (v in c("ciudad", "dpto")) {
  if (v %in% names(X_train_raw) && v %in% names(X_test_raw)) {
    all_levels       <- union(levels(X_train_raw[[v]]), levels(X_test_raw[[v]]))
    X_train_raw[[v]] <- factor(X_train_raw[[v]], levels = all_levels)
    X_test_raw[[v]]  <- factor(X_test_raw[[v]],  levels = all_levels)
  }
}
# -- Variables a excluir --------------------------------------
#Variables excluidas a mano ya que estas son interacciones que los arboles 
#por su naturaleza las pueden crear ellos mismos
#Se excluyen variables cuadráticas, logarítmicas, y de interacción, que no aportan

VARS_EXCLUIR <- c(
  # Cuadráticas de capital humano (Bloque 3)
  "educ_jefe_sq",
  "educ_prom_sq",
  # Logarítmicas (Bloque 9)
  "log_num_personas",
  "log_n_menores",
  # Cuadráticas generales (Bloque 9)
  "ratio_depend_sq",
  "tasa_ocup_sq",
  "hacinamiento_sq",
  "edad_jefe_sq",
  # Interacciones (Bloque 8)
  "educ_x_formal",
  "educ_x_ocup",
  "menores_x_desocup",
  "depend_x_informal",
  "rural_x_cta_propia",
  "educ_x_rural",
  "subsidiado_x_menores",
  "jefe_mayor_pension"
)

# -- Eliminar variables de VARS_EXCLUIR --------------------------------------
X_train <- X_train |> select(-any_of(VARS_EXCLUIR))
X_test  <- X_test  |> select(-any_of(VARS_EXCLUIR))

message("  Variables excluidas manualmente: ", length(VARS_EXCLUIR))
message("  Predictores tras exclusión: ", ncol(X_train))


# Alinear columnas: mismas variables en el mismo orden para train y test
cols_comunes <- intersect(names(X_train_raw), names(X_test_raw))
X_train      <- X_train_raw |> select(all_of(cols_comunes))
X_test       <- X_test_raw  |> select(all_of(cols_comunes))

# Variable respuesta: factor con niveles válidos para caret
niveles <- c("No", "Yes")
y_train <- factor(ifelse(train$pobre == 1, "Yes", "No"), levels = niveles)

# train_rf combina X e y para la interfaz de fórmula del modelo final
train_rf <- bind_cols(X_train, pobre = y_train)

message("  Predictores: ", ncol(X_train),
        "  (incluye ciudad y dpto como factores)")
message("  Niveles ciudad (train ∪ test): ", nlevels(X_train$ciudad))
message("  Niveles dpto   (train ∪ test): ", nlevels(X_train$dpto))
message("  NAs en X_train: ", sum(is.na(X_train)))

# =============================================================================
# SECCIÓN 4: FUNCIONES AUXILIARES
# =============================================================================
# prf_summary() es el summaryFunction que caret llama al final de cada fold.
# Retorna F1, Precision y Recall simultáneamente.
# Positivo = "Yes" (pobre = 1).
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
# CV de 5 folds, optimizando F1.
# savePredictions = "final" guarda las predicciones OOF para curvas ROC/PR.
ctrl_cv <- trainControl(
  method          = "cv",
  number          = 5,
  summaryFunction = prf_summary,
  classProbs      = TRUE,          # guarda P(Yes) en rf_cv$pred → OOF para ROC/PR
  verboseIter     = TRUE,
  savePredictions = "final"        # columnas: rowIndex, obs, pred, Yes, No
)

# =============================================================================
# SECCIÓN 6: GRILLA DE HIPERPARÁMETROS
# =============================================================================
# Mismos valores que RF_001 para comparabilidad directa.
# Con ~110 predictores (108 + ciudad + dpto), mtry = c(20, 30, 50) sigue
# siendo razonable (sqrt(110) ≈ 10 es conservador, 50 es agresivo).
# Resultado: 3 × 1 × 3 = 9 combinaciones × 5 folds = 45 fits totales.
tune_grid <- expand.grid(
  mtry          = c(20, 30, 50),
  splitrule     = "gini",
  min.node.size = c(1, 3, 5)
)

message("\n== Grilla RF ==")
message("  Combinaciones: ", nrow(tune_grid), " (3 mtry × 3 min.node.size)")
message("  Fits totales:  ", nrow(tune_grid) * 5, " (grilla × 5 folds)")
message("  Tiempo estimado: 3-4 h (ciudad/dpto añaden costo de ordenamiento en cada split)")

# =============================================================================
# SECCIÓN 7: ENTRENAMIENTO CON CV
# =============================================================================
# respect.unordered.factors = "order" se pasa a ranger() a través de los ...
# de caret::train(). En cada fit del CV, ranger ordena los k niveles de cada
# factor por su media de respuesta en el nodo antes de buscar el corte óptimo.
message("\n========================================")
message("Entrenando RF con CV-5  [ciudad+dpto como factores, order encoding]")
message("========================================")

n_cores <- max(1L, parallel::detectCores() - 1L)
cl <- makeCluster(n_cores)
registerDoParallel(cl)
message("  Cores registrados para CV paralelo: ", n_cores)

set.seed(42)
t0 <- Sys.time()

rf_cv <- train(
  x           = X_train,
  y           = y_train,
  method      = "ranger",
  trControl   = ctrl_cv,
  tuneGrid    = tune_grid,
  metric      = "F1",
  num.trees   = 150,
  num.threads = 1L,
  respect.unordered.factors = "order"   # ← ciudad y dpto como distribución, no dummies
)

t1 <- Sys.time()
stopCluster(cl)
message("  Tiempo CV: ",
        round(as.numeric(difftime(t1, t0, units = "mins")), 1), " min")

# =============================================================================
# SECCIÓN 8: RESULTADOS DEL CV
# =============================================================================
cat("\n-- Resultados por combinación (CV-5, ordenados por F1 desc) --\n")
print(rf_cv$results |>
        arrange(desc(F1)) |>
        mutate(across(where(is.numeric), ~ round(., 4))))

best_mtry      <- rf_cv$bestTune$mtry
best_splitrule <- rf_cv$bestTune$splitrule
best_min_node  <- rf_cv$bestTune$min.node.size
best_f1_cv     <- max(rf_cv$results$F1, na.rm = TRUE)

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
# classProbs=TRUE activó la columna "Yes" en rf_cv$pred. Filtramos la mejor
# combinación para obtener predicciones OOF (cada observación predicha una
# vez, por el fold en que no participó del entrenamiento).
message("\n== Extrayendo predicciones OOF del CV para curvas ROC/PR ==")

oof_preds  <- rf_cv$pred |>
  filter(mtry          == best_mtry,
         min.node.size == best_min_node) |>
  arrange(rowIndex)

probs_oof  <- oof_preds$Yes
labels_oof <- oof_preds$obs
message("  Predicciones OOF extraídas: ", length(probs_oof), " observaciones")

# =============================================================================
# SECCIÓN 9: MODELO FINAL
# =============================================================================
# Re-entrenamos con ranger() directo para probability=TRUE e importance="permutation".
# Se pasa la misma opción respect.unordered.factors = "order" para que el
# modelo final trate los factores igual que en el CV — fundamental para que
# las predicciones OOB sean comparables con las OOF del CV.
#
# El threshold se optimiza sobre predicciones OOB (estimación nativa de ranger).
message("\n========================================")
message("Entrenando modelo final sobre todos los datos de train")
message("========================================")

set.seed(42)
rf_final <- ranger::ranger(
  pobre ~ .,
  data                      = train_rf,
  num.trees                 = 150,
  mtry                      = best_mtry,
  splitrule                 = best_splitrule,
  min.node.size             = best_min_node,
  probability               = TRUE,
  importance                = "permutation",
  respect.unordered.factors = "order",   # mismo tratamiento que en CV
  seed                      = 42
)

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

# =============================================================================
# SECCIÓN 10: IMPORTANCIA DE VARIABLES
# =============================================================================
# rf_final$variable.importance contiene la importancia por permutación.
# ciudad y dpto aparecerán en el ranking — su posición indica cuánta señal
# geográfica capturan más allá de las otras variables del hogar.
message("\n========================================")
message("Importancia de variables (permutation)")
message("========================================")

varimp_rf <- rf_final$variable.importance

feature_matrix <- data.frame(
  variable    = names(varimp_rf),
  importancia = as.numeric(varimp_rf)
) |>
  arrange(desc(importancia)) |>
  mutate(rank = row_number())

feat_mat_path <- file.path(dir_model, "feature_matrix.csv")
write_csv(feature_matrix, feat_mat_path)
message("  Guardado: feature_matrix.csv  (",
        nrow(feature_matrix), " variables, incluyendo ciudad y dpto)")

top20 <- feature_matrix |> slice_head(n = 20)

p_varimp <- ggplot(top20, aes(x = reorder(variable, importancia),
                               y = importancia)) +
  geom_col(fill = "#534AB7", alpha = 0.85) +
  coord_flip() +
  labs(
    title    = paste0("RF: importancia de variables (permutation) — ", MODEL_ID),
    subtitle = paste0("Top 20 predictores | mtry = ", best_mtry,
                      " | min.node.size = ", best_min_node,
                      " | num.trees = 150 | ciudad+dpto incluidas"),
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
message("\n========================================")
message("Generando gráficos de diagnóstico")
message("========================================")

# AUC-ROC sobre predicciones OOF del CV (honesto — fuera de muestra)
roc_obj <- pROC::roc(
  response  = as.integer(labels_oof == "Yes"),
  predictor = probs_oof,
  quiet     = TRUE
)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC (OOF, CV-5, honesto): ", round(auc_roc, 4))

# -- Gráfico 1: Threshold vs F1 / Precision / Recall (OOB) --
th_results <- th_oob_search

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
    subtitle = sprintf(
      "Threshold óptimo = %.2f | F1 OOB = %.4f | mtry = %d | ciudad+dpto incluidas",
      best_th, best_oob_f1, best_mtry),
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

# -- Gráfico 2: Curva ROC --
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
                      " | Probabilidades OOF — CV-5 | ciudad+dpto incluidas"),
    x        = "Tasa de Falsos Positivos  (1 - Specificity)",
    y        = "Tasa de Verdaderos Positivos  (Recall)",
    caption  = "AUC calculado sobre predicciones OOF del CV: estimación honesta."
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc.png"),
       p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc.png")

# -- Gráfico 3: Curva Precision-Recall --
n_pos    <- sum(labels_oof == "Yes")
n_neg    <- sum(labels_oof == "No")
pr_df <- tibble(
  TP        = roc_obj$sensitivities       * n_pos,
  FP        = (1 - roc_obj$specificities) * n_neg,
  recall    = roc_obj$sensitivities,
  precision = TP / (TP + FP)
) |>
  filter(is.finite(precision), is.finite(recall))

prev_oof <- n_pos / (n_pos + n_neg)
p_prcurve <- ggplot(pr_df, aes(x = recall, y = precision)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_hline(yintercept = prev_oof,
             linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.85, y = prev_oof + 0.015,
           label = sprintf("Clasificador aleatorio (%.0f%%)", prev_oof * 100),
           size = 3, color = "gray50") +
  labs(
    title    = paste0("RF: curva Precision-Recall — ", MODEL_ID),
    subtitle = sprintf("AUC-ROC OOF: %.4f | mtry = %d | ciudad+dpto incluidas",
                       auc_roc, best_mtry),
    x        = "Recall  (fracción de pobres capturada)",
    y        = "Precision  (fracción de predichos pobres que lo son)",
    caption  = "Curva calculada sobre predicciones OOF del CV: estimación honesta."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve.png"),
       p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve.png")

# Guardar diagnósticos
saveRDS(
  list(rf_cv          = rf_cv,
       rf_final       = rf_final,
       roc_obj        = roc_obj,
       th_results     = th_results,
       varimp         = varimp_rf,
       feature_matrix = feature_matrix,
       best_th        = best_th,
       best_oob_f1    = best_oob_f1),
  file.path(dir_model, "diagnostics.rds")
)
message("  Guardado: diagnostics.rds")

# =============================================================================
# SECCIÓN 12: SUBMISSION
# =============================================================================
message("\n========================================")
message("Generando submission para Kaggle")
message("========================================")

probs_test <- predict(rf_final, data = X_test)$predictions[, "Yes"]
pred_test  <- as.integer(probs_test >= best_th)

submission <- tibble(id = ids, pobre = pred_test)
sub_name   <- sprintf("RF_mtry%d_minnodesize%d_ntrees150_th%03.0f.csv",
                      best_mtry, best_min_node, best_th * 100)
sub_path   <- file.path(dir_subs, sub_name)
write_csv(submission, sub_path)

message("  Guardado: ", sub_path)
message("  Threshold usado: ", best_th, " (óptimo OOB)")
message("  Pobres predichos: ", sum(submission$pobre),
        " (", round(mean(submission$pobre) * 100, 1), "%)")

# =============================================================================
# SECCIÓN 13: REGISTRO EN model_registry.csv
# =============================================================================
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
  cv_F1              = round(best_oob_f1, 4),
  cv_Precision       = round(best_oob_pr, 4),
  cv_Recall          = round(best_oob_rc, 4),
  auc_roc            = round(auc_roc,     4),
  kaggle_public_F1   = NA_real_,
  threshold          = best_th,
  notas              = paste0(
    "RF ranger. mtry=", best_mtry,
    " min_node_size=", best_min_node,
    " splitrule=gini num.trees=150.",
    " ciudad+dpto incluidas como factor (respect.unordered.factors=order).",
    " Threshold OOB (max F1). Curvas ROC/PR sobre OOF CV-5.",
    " th=", best_th,
    " F1_OOB=", round(best_oob_f1, 4),
    " AUC_OOF=", round(auc_roc, 4),
    " (CV-5 F1 th=0.5: ", round(best_f1_cv, 4), ")."
  ),
  cp              = NA_real_,
  maxdepth        = NA_real_,
  train_F1        = round(train_F1,        4),
  train_Precision = round(train_Precision, 4),
  train_Recall    = round(train_Recall,    4)
)

if (file.exists(reg_path)) {
  registry <- read_csv(reg_path, show_col_types = FALSE) |>
    mutate(fecha = as.Date(fecha, origin = "1899-12-30"))
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
