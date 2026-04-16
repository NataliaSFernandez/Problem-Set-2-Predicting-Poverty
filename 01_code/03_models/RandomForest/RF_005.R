#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/RandomForest/RF_005.R
#==============================================================================
#
# ALGORITMO: Random Forest — Feature Selection con CV honesto por importancia
#
# QUÉ HACE ESTE SCRIPT:
#   Tuea el número óptimo de variables (n_top) usando un CV-5 completamente
#   honesto: dentro de cada fold, el ranking de importancia se recalcula
#   exclusivamente con los datos de entrenamiento de ese fold (sin ver el fold
#   de validación). Esto evita el sesgo de RF_002/RF_004, donde el ranking se
#   calcula una vez sobre todo el train y luego se usa para evaluar subconjuntos.
#
# POR QUÉ RF_002/RF_004 SON OPTIMISTAS:
#   En RF_004 el ranking proviene de RF_003 (entrenado en todo el train).
#   Al evaluar top-20 sobre OOB del modelo final, la importancia ya "vio" las
#   mismas observaciones que se usan en la evaluación → el ranking está adaptado
#   al dataset completo, introduciendo un sesgo de selección.
#
# DISEÑO DE ESTE SCRIPT:
#   Para cada uno de los 5 folds externos:
#     1. Modelo de importancia (train del fold, ~131k obs, importance=permutation)
#        → ranking honesto de variables para ese fold
#     2. Para cada n_top ∈ {20, 40, 60, 80, 110}:
#        - Entrenar RF con top-n_top variables del ranking del fold
#        - Evaluar F1 (threshold óptimo) en el fold de validación (~33k obs)
#   Promediar F1 a través de los 5 folds → seleccionar n_top óptimo.
#   Modelo final:
#     - Entrenar modelo de importancia sobre todo el train → ranking final
#     - Entrenar RF con n_top óptimo → threshold sobre OOB → submission
#
# HIPERPARÁMETROS (fijos, heredados de RF_003):
#   mtry          = 50
#   splitrule     = "gini"
#   min.node.size = 3
#   num.trees     = 150
#   respect.unordered.factors = "order"  (ciudad y dpto como factores)
#
# VALORES TUNEADOS:
#   n_top ∈ {20, 40, 60, 80, 110 (todas)}   — 5 × 5 folds = 25 evaluaciones
#
# FLUJO DEL SCRIPT:
#   0. Paquetes
#   1. Configuración global
#   2. Cargar datos
#   3. Preprocesamiento (ciudad y dpto como factores; niveles alineados)
#   4. Funciones auxiliares (F1, threshold search)
#   5. CV-5 con importancia recalculada por fold
#   6. Selección de n_top óptimo (mejor F1 CV)
#   7. Modelo final (importancia + selección + OOB threshold)
#   8. Gráficos de diagnóstico
#   9. Submission CSV
#   10. Registro en model_registry.csv
#
# INPUTS:
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#
# OUTPUTS:
#   03_submissions/RF_mtry50_minnodesize3_ntrees150_top{N}_cv5imp_th{T}.csv
#   02_outputs/models/RandomForest/RF_005/cv_results.csv
#   02_outputs/models/RandomForest/RF_005/feature_matrix.csv
#   02_outputs/models/RandomForest/RF_005/varimp.png
#   02_outputs/models/RandomForest/RF_005/threshold.png
#   02_outputs/models/RandomForest/RF_005/diagnostics.rds
#   02_outputs/model_registry.csv  (append)
#
# TIEMPO ESTIMADO: ~14 min con 9 cores
#   5 folds × (1.2 min importance + 5 × 0.1 min selección) + 2.7 min final
#
# REPRODUCIBILIDAD:
#   Correr desde la raíz del proyecto. Semilla global: 42.
#==============================================================================
# nolint start

# =============================================================================
# SECCIÓN 0: PAQUETES
# =============================================================================
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,   # dplyr, ggplot2, purrr, tibble
  ranger,      # Random Forest en C++
  pROC,        # roc(), auc()
  fs           # dir_create()
)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
AUTOR    <- "Jonathan Melo"
MODEL_ID <- "RF_005"

set.seed(42)

dir_model <- file.path("02_outputs/models/RandomForest", MODEL_ID)
dir_subs  <- "03_submissions"
reg_path  <- "02_outputs/model_registry.csv"

fs::dir_create(dir_model, recurse = TRUE)
fs::dir_create(dir_subs,  recurse = TRUE)

# Hiperparámetros fijos heredados de RF_003
BEST_MTRY      <- 50L
BEST_SPLITRULE <- "gini"
BEST_MIN_NODE  <- 3L
NUM_TREES      <- 150L
K_FOLDS        <- 5L

# Número de cores para paralelización intra-modelo
n_cores <- max(1L, parallel::detectCores() - 1L)
message("Cores disponibles: ", n_cores)

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
# Misma lógica que RF_003: ciudad y dpto como factores con niveles alineados.
message("\n== Preprocesando variables ==")

preparar_X_rf <- function(df) {
  df |>
    select(-any_of(c("id", "pobre"))) |>
    mutate(zona   = as.numeric(zona),
           ciudad = as.factor(ciudad),
           dpto   = as.factor(dpto)) |>
    mutate(across(where(is.character), as.factor))
}

X_train_raw <- preparar_X_rf(train)
X_test_raw  <- preparar_X_rf(test)

# Alinear niveles ciudad y dpto: unión train ∪ test
for (v in c("ciudad", "dpto")) {
  all_levels       <- union(levels(X_train_raw[[v]]), levels(X_test_raw[[v]]))
  X_train_raw[[v]] <- factor(X_train_raw[[v]], levels = all_levels)
  X_test_raw[[v]]  <- factor(X_test_raw[[v]],  levels = all_levels)
}

cols_comunes <- intersect(names(X_train_raw), names(X_test_raw))
X_train      <- X_train_raw |> select(all_of(cols_comunes))
X_test       <- X_test_raw  |> select(all_of(cols_comunes))

niveles <- c("No", "Yes")
y_train <- factor(ifelse(train$pobre == 1, "Yes", "No"), levels = niveles)

P_TOTAL <- ncol(X_train)  # número total de predictores (incluye ciudad y dpto)

message("  Predictores disponibles: ", P_TOTAL,
        "  (incluye ciudad y dpto)")
message("  NAs en X_train: ", sum(is.na(X_train)))

# Valores de n_top a evaluar: incluye "todas" (P_TOTAL) como referencia
TOP_N_VALS <- c(20L, 40L, 60L, 80L, P_TOTAL)

# =============================================================================
# SECCIÓN 4: FUNCIONES AUXILIARES
# =============================================================================

# f1_score(): calcula F1 dado un vector de probabilidades, etiquetas y threshold
f1_score <- function(probs, labels, th) {
  pred <- factor(ifelse(probs >= th, "Yes", "No"), levels = niveles)
  TP <- sum(pred == "Yes" & labels == "Yes")
  FP <- sum(pred == "Yes" & labels == "No")
  FN <- sum(pred == "No"  & labels == "Yes")
  if ((2*TP + FP + FN) == 0) return(NA_real_)
  2*TP / (2*TP + FP + FN)
}

# pr_rc(): calcula Precision y Recall
pr_rc <- function(probs, labels, th) {
  pred <- factor(ifelse(probs >= th, "Yes", "No"), levels = niveles)
  TP <- sum(pred == "Yes" & labels == "Yes")
  FP <- sum(pred == "Yes" & labels == "No")
  FN <- sum(pred == "No"  & labels == "Yes")
  list(
    precision = if ((TP + FP) == 0) NA_real_ else TP / (TP + FP),
    recall    = if ((TP + FN) == 0) NA_real_ else TP / (TP + FN)
  )
}

# buscar_th(): encuentra el threshold óptimo (max F1) en un vector de probs
buscar_th <- function(probs, labels, grid = seq(0.05, 0.95, by = 0.01)) {
  resultados <- map_dfr(grid, function(th) {
    tibble(threshold = th, F1 = f1_score(probs, labels, th))
  })
  resultados |> filter(!is.na(F1)) |> slice_max(F1, n = 1, with_ties = FALSE)
}

# =============================================================================
# SECCIÓN 5: CV-5 CON IMPORTANCIA RECALCULADA POR FOLD
# =============================================================================
# Para cada fold k:
#   (a) Modelo de importancia (train_k, ~131k obs, permutation, probability=TRUE)
#       → ranking de variables honesto para ese fold (sin ver val_k)
#   (b) Para cada n_top en TOP_N_VALS:
#       - Seleccionar top-n_top vars del ranking del fold
#       - Entrenar RF en train_k con esas vars (probability=TRUE, sin importance)
#       - Predecir en val_k → buscar threshold óptimo → registrar F1
#
# Paralelización: num.threads = n_cores dentro de cada modelo ranger
# (los folds son secuenciales porque cada fold necesita su propio ranking).
message("\n========================================")
message("CV-5: importancia recalculada por fold")
message("========================================")

# Crear índices de fold reproducibles
set.seed(42)
fold_ids <- sample(rep(seq_len(K_FOLDS), length.out = nrow(X_train)))

# Almacenamiento de resultados por fold × n_top
cv_resultados <- vector("list", K_FOLDS)

t0_cv <- Sys.time()

for (k in seq_len(K_FOLDS)) {

  message("\n-- Fold ", k, " / ", K_FOLDS, " --")

  idx_val   <- which(fold_ids == k)
  idx_train <- which(fold_ids != k)

  X_fold_tr  <- X_train[idx_train, ]
  X_fold_val <- X_train[idx_val, ]
  y_fold_tr  <- y_train[idx_train]
  y_fold_val <- y_train[idx_val]

  df_fold_tr <- bind_cols(X_fold_tr, pobre = y_fold_tr)

  # (a) Modelo de importancia — todos los predictores, sólo necesitamos el ranking
  message("  Entrenando modelo de importancia (", nrow(df_fold_tr), " obs)...")
  t0_imp <- proc.time()

  set.seed(42 * k)
  imp_model <- ranger::ranger(
    pobre ~ .,
    data                      = df_fold_tr,
    num.trees                 = NUM_TREES,
    mtry                      = BEST_MTRY,
    splitrule                 = BEST_SPLITRULE,
    min.node.size             = BEST_MIN_NODE,
    probability               = TRUE,
    importance                = "permutation",
    respect.unordered.factors = "order",
    num.threads               = n_cores,  # todos los cores → rápido
    seed                      = 42 * k
  )

  t1_imp <- proc.time()
  message("  Importancia calculada en ",
          round((t1_imp - t0_imp)[["elapsed"]] / 60, 2), " min")

  # Ranking honesto para este fold
  fold_ranking <- sort(imp_model$variable.importance, decreasing = TRUE)
  message("  Variable #1 en este fold: ", names(fold_ranking)[1])

  # (b) Para cada n_top: entrenar modelo de selección y evaluar en val
  fold_res <- map_dfr(TOP_N_VALS, function(n_top) {

    vars_top  <- names(fold_ranking)[seq_len(min(n_top, length(fold_ranking)))]
    X_tr_sub  <- X_fold_tr  |> select(any_of(vars_top))
    X_val_sub <- X_fold_val |> select(any_of(vars_top))
    df_tr_sub <- bind_cols(X_tr_sub, pobre = y_fold_tr)
    mtry_eff  <- min(BEST_MTRY, ncol(X_tr_sub))

    set.seed(42 * k + n_top)
    sel_model <- ranger::ranger(
      pobre ~ .,
      data                      = df_tr_sub,
      num.trees                 = NUM_TREES,
      mtry                      = mtry_eff,
      splitrule                 = BEST_SPLITRULE,
      min.node.size             = BEST_MIN_NODE,
      probability               = TRUE,
      importance                = "none",   # no necesario para selección
      respect.unordered.factors = "order",
      num.threads               = n_cores,
      seed                      = 42 * k + n_top
    )

    probs_val <- predict(sel_model, data = X_val_sub)$predictions[, "Yes"]
    best_row  <- buscar_th(probs_val, y_fold_val)

    tibble(
      fold  = k,
      n_top = n_top,
      F1    = best_row$F1,
      th    = best_row$threshold
    )
  })

  message("  n_top | F1_val:")
  for (i in seq_len(nrow(fold_res))) {
    message("    ", fold_res$n_top[i], "\t→  ", round(fold_res$F1[i], 4))
  }

  cv_resultados[[k]] <- fold_res
}

t1_cv <- Sys.time()
message("\nCV completado en ",
        round(as.numeric(difftime(t1_cv, t0_cv, units = "mins")), 1), " min")

# =============================================================================
# SECCIÓN 6: SELECCIÓN DE n_top ÓPTIMO
# =============================================================================
cv_df <- bind_rows(cv_resultados)

# Promediar F1 a través de los 5 folds para cada n_top
cv_summary <- cv_df |>
  group_by(n_top) |>
  summarise(
    F1_mean = mean(F1, na.rm = TRUE),
    F1_sd   = sd(F1,   na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(desc(F1_mean))

cat("\n-- Resultados CV-5 por n_top --\n")
print(cv_summary |> mutate(across(where(is.double), ~ round(., 4))))

best_n_top  <- cv_summary |> slice_max(F1_mean, n = 1, with_ties = FALSE) |> pull(n_top)
best_f1_cv  <- cv_summary |> filter(n_top == best_n_top) |> pull(F1_mean)

message("\n  n_top óptimo: ", best_n_top,
        "  (F1 CV-5 = ", round(best_f1_cv, 4), ")")

# Guardar resultados del CV
cv_path <- file.path(dir_model, "cv_results.csv")
write_csv(cv_df |> arrange(fold, n_top), cv_path)
write_csv(cv_summary, file.path(dir_model, "cv_summary.csv"))
message("  Guardado: cv_results.csv y cv_summary.csv")

# =============================================================================
# SECCIÓN 7: MODELO FINAL
# =============================================================================
# Paso 1: entrenar modelo de importancia sobre TODO el train → ranking final
# Paso 2: entrenar modelo de selección con best_n_top vars → threshold OOB
message("\n========================================")
message("Modelo final (n_top = ", best_n_top, ")")
message("========================================")

# 7a — Modelo de importancia final
message("Entrenando modelo de importancia final (164k obs)...")
t0_f <- proc.time()

df_full_rf <- bind_cols(X_train, pobre = y_train)

set.seed(42)
imp_final <- ranger::ranger(
  pobre ~ .,
  data                      = df_full_rf,
  num.trees                 = NUM_TREES,
  mtry                      = BEST_MTRY,
  splitrule                 = BEST_SPLITRULE,
  min.node.size             = BEST_MIN_NODE,
  probability               = TRUE,
  importance                = "permutation",
  respect.unordered.factors = "order",
  num.threads               = n_cores,
  seed                      = 42
)

t1_f <- proc.time()
message("  Importancia final en ",
        round((t1_f - t0_f)[["elapsed"]] / 60, 2), " min")

# Ranking final y feature_matrix
varimp_final   <- imp_final$variable.importance
feature_matrix <- data.frame(
  variable    = names(varimp_final),
  importancia = as.numeric(varimp_final)
) |>
  arrange(desc(importancia)) |>
  mutate(rank = row_number())

write_csv(feature_matrix, file.path(dir_model, "feature_matrix.csv"))
message("  Guardado: feature_matrix.csv (", nrow(feature_matrix), " variables)")

# Variables seleccionadas con el n_top óptimo
vars_final <- feature_matrix |>
  slice_head(n = best_n_top) |>
  pull(variable)

X_train_sel <- X_train |> select(any_of(vars_final))
X_test_sel  <- X_test  |> select(any_of(vars_final))
df_final_rf <- bind_cols(X_train_sel, pobre = y_train)
mtry_final  <- min(BEST_MTRY, ncol(X_train_sel))

message("  Variables en modelo final: ", ncol(X_train_sel),
        "  | mtry efectivo: ", mtry_final)
message("  ¿ciudad en subset? ", "ciudad" %in% vars_final,
        "  | ¿dpto? ", "dpto" %in% vars_final)

# 7b — Modelo de selección final
message("\nEntrenando modelo de selección final...")
t0_s <- proc.time()

set.seed(42)
rf_final <- ranger::ranger(
  pobre ~ .,
  data                      = df_final_rf,
  num.trees                 = NUM_TREES,
  mtry                      = mtry_final,
  splitrule                 = BEST_SPLITRULE,
  min.node.size             = BEST_MIN_NODE,
  probability               = TRUE,
  importance                = "none",
  respect.unordered.factors = "order",
  num.threads               = n_cores,
  seed                      = 42
)

t1_s <- proc.time()
message("  Modelo final entrenado en ",
        round((t1_s - t0_s)[["elapsed"]] / 60, 2), " min")

# Threshold óptimo sobre predicciones OOB del modelo de selección
probs_oob <- rf_final$predictions[, "Yes"]
best_row  <- buscar_th(probs_oob, y_train)
best_th   <- best_row$threshold
best_f1   <- best_row$F1
best_pr_rc <- pr_rc(probs_oob, y_train, best_th)

message("  Threshold óptimo OOB: ", best_th)
message("  F1 OOB:               ", round(best_f1,            4))
message("  Precision OOB:        ", round(best_pr_rc$precision, 4))
message("  Recall OOB:           ", round(best_pr_rc$recall,   4))

# AUC-ROC OOB
roc_obj <- pROC::roc(
  response  = as.integer(y_train == "Yes"),
  predictor = probs_oob,
  quiet     = TRUE
)
auc_roc <- as.numeric(pROC::auc(roc_obj))
message("  AUC-ROC OOB:          ", round(auc_roc, 4))

# =============================================================================
# SECCIÓN 8: GRÁFICOS DE DIAGNÓSTICO
# =============================================================================
message("\n========================================")
message("Generando gráficos de diagnóstico")
message("========================================")

# -- Gráfico 1: F1 por n_top (CV-5) --
p_cv <- cv_summary |>
  mutate(n_top_label = ifelse(n_top == P_TOTAL, "todas", as.character(n_top))) |>
  ggplot(aes(x = factor(n_top), y = F1_mean)) +
  geom_col(aes(fill = n_top == best_n_top), alpha = 0.85, show.legend = FALSE) +
  geom_errorbar(aes(ymin = F1_mean - F1_sd, ymax = F1_mean + F1_sd),
                width = 0.25, color = "gray40") +
  scale_fill_manual(values = c("FALSE" = "#BDBDBD", "TRUE" = "#534AB7")) +
  scale_x_discrete(labels = function(x) ifelse(x == as.character(P_TOTAL),
                                                "todas", x)) +
  labs(
    title    = paste0("RF: selección de n_top por CV-5 — ", MODEL_ID),
    subtitle = sprintf(
      "n_top óptimo = %d | F1 CV-5 = %.4f | importancia recalculada por fold",
      best_n_top, best_f1_cv),
    x        = "Número de variables (top-N por importancia)",
    y        = "F1 promedio CV-5",
    caption  = "Barras de error = ±1 SD entre folds. Azul = n_top óptimo."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "cv_ntop.png"),
       p_cv, width = 7, height = 5, dpi = 150)
message("  Guardado: cv_ntop.png")

# -- Gráfico 2: Importancia de variables (Top 20) --
top20 <- feature_matrix |> slice_head(n = 20)

p_varimp <- ggplot(top20, aes(x = reorder(variable, importancia), y = importancia)) +
  geom_col(fill = "#534AB7", alpha = 0.85) +
  coord_flip() +
  labs(
    title    = paste0("RF: importancia de variables (final) — ", MODEL_ID),
    subtitle = paste0("Top 20 | modelo final sobre todo el train | n_top óptimo = ",
                      best_n_top),
    x        = NULL,
    y        = "Importancia por permutación"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "varimp.png"),
       p_varimp, width = 8, height = 6, dpi = 150)
message("  Guardado: varimp.png")

# -- Gráfico 3: Threshold vs métricas (OOB) --
th_search <- buscar_th(probs_oob, y_train,
                        grid = seq(0.05, 0.95, by = 0.01)) # reutilizamos buscar_th
th_full <- map_dfr(seq(0.05, 0.95, by = 0.01), function(th) {
  prc <- pr_rc(probs_oob, y_train, th)
  tibble(threshold = th,
         F1        = f1_score(probs_oob, y_train, th),
         Precision = prc$precision,
         Recall    = prc$recall)
})

p_threshold <- th_full |>
  filter(!is.na(F1)) |>
  pivot_longer(cols = c(F1, Precision, Recall), names_to = "metrica") |>
  ggplot(aes(x = threshold, y = value, color = metrica)) +
  geom_line(linewidth = 0.9) +
  geom_vline(xintercept = best_th, linetype = "dashed", color = "#534AB7") +
  geom_vline(xintercept = 0.5,    linetype = "dotted", color = "gray40") +
  annotate("text", x = best_th + 0.02, y = 0.08,
           label = sprintf("th = %.2f\n(óptimo OOB)", best_th),
           hjust = 0, size = 3.2, color = "#534AB7") +
  scale_color_manual(
    values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3")
  ) +
  labs(
    title    = paste0("RF: métricas por threshold (OOB) — ", MODEL_ID),
    subtitle = sprintf(
      "Threshold óptimo = %.2f | F1 OOB = %.4f | n_top = %d",
      best_th, best_f1, best_n_top),
    x        = "Threshold de clasificación",
    y        = "Valor de la métrica",
    color    = NULL,
    caption  = "Línea discontinua = threshold óptimo OOB."
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

ggsave(file.path(dir_model, "threshold.png"),
       p_threshold, width = 8, height = 5, dpi = 150)
message("  Guardado: threshold.png")

# -- Gráfico 4: Curva ROC --
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
    subtitle = sprintf("n_top=%d | th=%.2f | predicciones OOB (honesto)",
                       best_n_top, best_th),
    x        = "Tasa de Falsos Positivos  (1 - Specificity)",
    y        = "Tasa de Verdaderos Positivos  (Recall)",
    caption  = "AUC calculado sobre predicciones OOB del modelo final: estimación honesta."
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "roc.png"),
       p_roc, width = 6, height = 6, dpi = 150)
message("  Guardado: roc.png")

# -- Gráfico 5: Curva Precision-Recall --
n_pos    <- sum(y_train == "Yes")
n_neg    <- sum(y_train == "No")
pr_curve_df <- tibble(
  TP        = roc_obj$sensitivities * n_pos,
  FP        = (1 - roc_obj$specificities) * n_neg,
  recall    = roc_obj$sensitivities,
  precision = TP / (TP + FP)
) |> filter(is.finite(precision), is.finite(recall))

prev_rate <- n_pos / (n_pos + n_neg)
pt_opt    <- tibble(recall = best_pr_rc$recall, precision = best_pr_rc$precision)

p_prcurve <- ggplot(pr_curve_df, aes(x = recall, y = precision)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  geom_hline(yintercept = prev_rate,
             linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.85, y = prev_rate + 0.015,
           label = sprintf("Clasificador aleatorio (%.0f%%)", prev_rate * 100),
           size = 3, color = "gray50") +
  geom_point(data = pt_opt, aes(x = recall, y = precision),
             color = "#E84855", size = 3) +
  annotate("text",
           x     = best_pr_rc$recall - 0.1,
           y     = best_pr_rc$precision + 0.015,
           label = sprintf("th=%.2f\nF1=%.4f", best_th, best_f1),
           size  = 3, color = "#E84855") +
  labs(
    title    = paste0("RF: curva Precision-Recall — ", MODEL_ID),
    subtitle = sprintf("AUC-ROC OOB: %.4f | n_top=%d | imbalance: none",
                       auc_roc, best_n_top),
    x        = "Recall  (fracción de pobres capturada)",
    y        = "Precision  (fracción de predichos pobres que lo son)",
    caption  = "Curva basada en predicciones OOB — honesta. Punto rojo = threshold óptimo."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(dir_model, "prcurve.png"),
       p_prcurve, width = 6, height = 5, dpi = 150)
message("  Guardado: prcurve.png")

# Guardar diagnósticos
saveRDS(
  list(cv_df          = cv_df,
       cv_summary     = cv_summary,
       best_n_top     = best_n_top,
       imp_final      = imp_final,
       rf_final       = rf_final,
       feature_matrix = feature_matrix,
       roc_obj        = roc_obj,
       auc_roc        = auc_roc,
       best_th        = best_th,
       best_oob_f1    = best_f1,
       precision_oob  = best_pr_rc$precision,
       recall_oob     = best_pr_rc$recall),
  file.path(dir_model, "diagnostics.rds")
)
message("  Guardado: diagnostics.rds")

# =============================================================================
# SECCIÓN 9: SUBMISSION
# =============================================================================
message("\n========================================")
message("Generando submission para Kaggle")
message("========================================")

probs_test <- predict(rf_final, data = X_test_sel)$predictions[, "Yes"]
pred_test  <- as.integer(probs_test >= best_th)

submission <- tibble(id = ids, pobre = pred_test)
sub_name   <- sprintf(
  "RF_mtry%d_minnodesize%d_ntrees%d_top%d_cv5imp_th%03.0f.csv",
  mtry_final, BEST_MIN_NODE, NUM_TREES, best_n_top, best_th * 100
)
sub_path   <- file.path(dir_subs, sub_name)
write_csv(submission, sub_path)

message("  Guardado: ", sub_path)
message("  Threshold: ", best_th, "  | Pobres predichos: ", sum(pred_test),
        " (", round(mean(pred_test) * 100, 1), "%)")

# =============================================================================
# SECCIÓN 10: REGISTRO EN model_registry.csv
# =============================================================================
message("\n========================================")
message("Registro en model_registry.csv")
message("========================================")

nueva_fila <- tibble(
  model_id           = MODEL_ID,
  fecha              = Sys.Date(),
  autor              = AUTOR,
  algoritmo          = "RandomForest",
  n_features         = ncol(X_train_sel),
  imbalance_strategy = "none",
  cv_folds           = K_FOLDS,
  cv_F1              = round(best_f1,                4),
  cv_Precision       = round(best_pr_rc$precision,   4),
  cv_Recall          = round(best_pr_rc$recall,      4),
  auc_roc            = round(auc_roc,                4),
  kaggle_public_F1   = NA_real_,
  threshold          = best_th,
  notas              = paste0(
    "RF ranger. Feature selection con CV-5 honesto: importance recalculada por fold.",
    " n_top óptimo=", best_n_top,
    " (evaluado sobre {20,40,60,80,", P_TOTAL, "}).",
    " mtry=", mtry_final,
    " min_node_size=", BEST_MIN_NODE,
    " splitrule=gini num.trees=", NUM_TREES,
    " respect.unordered.factors=order.",
    " ciudad en subset: ", "ciudad" %in% vars_final,
    ". Threshold OOB (max F1).",
    " th=", best_th,
    " F1_OOB=", round(best_f1, 4),
    " F1_CV5=", round(best_f1_cv, 4),
    " AUC_OOB=", round(auc_roc, 4), "."
  ),
  cp              = NA_real_,
  maxdepth        = NA_real_,
  train_F1        = round(best_f1,              4),
  train_Precision = round(best_pr_rc$precision, 4),
  train_Recall    = round(best_pr_rc$recall,    4)
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
