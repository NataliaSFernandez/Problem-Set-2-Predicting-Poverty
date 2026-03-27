#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/LPM/LPM.R
#==============================================================================
#
# ALGORITMO  : Linear Probability Model (LPM)
#
# QUÉ HACE ESTE SCRIPT:
#   Estima una regresión OLS donde la variable dependiente es binaria (pobre =
#   1 / no pobre = 0). El modelo aprende una función lineal de las características
#   del hogar para predecir la probabilidad de ser pobre. Cada coeficiente β_j
#   es directamente interpretable como: "un incremento de una unidad en X_j
#   cambia la probabilidad de pobreza en β_j puntos porcentuales, manteniendo
#   lo demás constante". Esto hace al LPM el modelo más interpretable del
#   repertorio del problem set.
#
# LIMITACIÓN CONOCIDA:
#   El LPM puede predecir probabilidades fuera de [0,1]. En la práctica, esto
#   afecta a una fracción de observaciones extremas pero no invalida el modelo
#   para clasificación (sólo necesitamos comparar el score continuo con un
#   threshold).
#
# FLUJO DEL SCRIPT:
#   0. Configuración y paquetes
#   1. Identificación del modelo (rellenar antes de correr)
#   2. Cargar datos
#   3. Preparar X y Y
#   4. Cross-validation (k-fold manual, 5 folds)
#   5. Optimizar threshold sobre predicciones out-of-fold
#   6. Modelo final sobre todo el train
#   7. Análisis de resultados (coeficientes, matriz de confusión, curva PR)
#   8. Generar submission para Kaggle
#   9. Guardar diagnósticos
#  10. Registrar en model_registry.csv
#  11. Sugerencias de mejora (comentadas, no ejecutar)
#
# INPUTS (relativos a la raíz del proyecto):
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#
# OUTPUTS (relativos a la raíz del proyecto):
#   03_submissions/submission_[MODEL_ID].csv        (para subir a Kaggle)
#   02_outputs/models/LPM/[MODEL_ID]/coefs.png
#   02_outputs/models/LPM/[MODEL_ID]/threshold.png
#   02_outputs/models/LPM/[MODEL_ID]/prcurve.png
#   02_outputs/models/LPM/[MODEL_ID]/diagnostics.rds
#   02_outputs/model_registry.csv                   ← tabla maestra (append)
#
# LÓGICA DE ESTE SCRIPT:
#   Está diseñado para ejecutarse entero de principio a fin. Sin embargo, hay
#   tres momentos de intervención manual:
#
#   [ANTES de correr]
#     Sección 1 → editar MODEL_ID, AUTOR y NOTAS.
#     Cada corrida distinta debe tener un MODEL_ID nuevo (LPM_001, LPM_002, …).
#
#   [PAUSA OPCIONAL a mitad del script]
#     Sección 5 calcula y guarda el threshold que maximiza F1 en CV.
#     Sección 6 te permite usarlo tal cual o cambiarlo manualmente.
#     Default → THRESHOLD <- THRESHOLD_OPT  (el óptimo de CV)
#     Si quieres explorar otro valor, edita esa línea y corre desde la Sección 6
#     hacia abajo.
#
#   [DESPUÉS de correr]
#     Subir el CSV generado a Kaggle y anotar el public F1 manualmente en
#     02_outputs/model_registry.csv (columna kaggle_public_F1).
#     Default → kaggle_public_F1 = NA  (se llena a mano post-Kaggle)
#
#   La Sección 12 está completamente comentada: son ideas para la próxima
#   iteración, no se ejecuta.
#
# REPRODUCIBILIDAD:
#   Correr desde la raíz del proyecto. Semilla global: 42.
#==============================================================================


# =============================================================================
# SECCIÓN 0: PAQUETES Y CONFIGURACIÓN GLOBAL
# =============================================================================
# pacman::p_load() instala el paquete si no está instalado, luego lo carga.
# Usarlo en lugar de install.packages() + library() garantiza que el script
# sea autocontenido: cualquier persona puede ejecutarlo sin configuración previa.

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,  # dplyr (manipulación), ggplot2 (gráficos), readr (CSV), tibble
  caret,      # confusionMatrix(): calcula F1, precision, recall con una sola llamada
  pROC,       # roc(): construye la curva ROC para calcular AUC como diagnóstico
  fs          # dir_create(): crea carpetas de forma multiplataforma (Mac/Linux/Windows)
)

# Semilla global — TODOS los scripts del equipo deben usar 42
# Garantiza que los folds de CV sean idénticos si alguien corre el script de nuevo
set.seed(42)

# ---------- Rutas (relativas a la raíz del proyecto) -------------------------
# Si lanzas el script con source() desde otro directorio, ajusta estas rutas.
# Si usas RStudio Projects, la raíz es automáticamente la carpeta del proyecto.

dir_processed   <- "00_data/processed"       # donde están train_final.rds y test_final.rds
dir_outputs_lpm <- "02_outputs/models/LPM"   # carpeta base del algoritmo LPM
dir_submissions <- "03_submissions"           # CSVs listos para subir a Kaggle
registry_path   <- "02_outputs/model_registry.csv"  # tabla maestra de todos los modelos


# =============================================================================
# SECCIÓN 1: IDENTIFICACIÓN DEL MODELO
# =============================================================================
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  >>>  MODIFICAR ESTOS TRES CAMPOS ANTES DE CORRER EL SCRIPT  <<<         │
# │                                                                          │
# │  MODEL_ID: identificador único. Convención: LPM_001, LPM_002, ...        │
# │            Cambiar en cada corrida distinta para tener historial.        │
# │  AUTOR:    nombre del autor (para el registro compartido del equipo).    │
# │  NOTAS:    descripción breve de qué cambió respecto a la corrida         │
# │            anterior. Ej: "threshold ajustado a 0.35 con pesos".          │
# └──────────────────────────────────────────────────────────────────────────┘

MODEL_ID <- "LPM_001"                                   # <<< MODIFICAR
AUTOR    <- "Jonathan"                                       # <<< MODIFICAR
NOTAS    <- "Baseline LPM — todas las variables, threshold óptimo CV"  # <<< MODIFICAR

# Carpeta de outputs de esta corrida específica (subcarpeta por MODEL_ID)
dir_model <- file.path(dir_outputs_lpm, MODEL_ID)

# Crear directorios si no existen (no falla si ya existen)
fs::dir_create(dir_model,       recurse = TRUE)
fs::dir_create(dir_submissions, recurse = TRUE)
# garantiza que 02_outputs/ exista para el registry
fs::dir_create("02_outputs", recurse = TRUE)


# =============================================================================
# SECCIÓN 2: CARGAR DATOS
# =============================================================================
# Cargamos los .rds porque son más rápidos que los .csv y preservan los tipos
# de variables (factores, enteros, dobles) exactamente como los dejó script 01.
#
# train_final: 164,960 hogares, 115 columnas (114 predictores + 'pobre')
# test_final:  sin 'pobre' — es lo que debemos predecir para Kaggle

train <- readRDS(file.path(dir_processed, "train_final.rds"))
test  <- readRDS(file.path(dir_processed, "test_final.rds"))

# Información diagnóstica básica
message("━━━ Datos cargados ━━━")
message("Train: ", nrow(train), " hogares | ", ncol(train), " columnas")
message("Test:  ", nrow(test),  " hogares | ", ncol(test),  " columnas")
message("Balance: ", round(mean(train$pobre) * 100, 1), "% pobres  /  ",
        round(mean(1 - train$pobre) * 100, 1), "% no pobres")
# Con ~20% pobres el dataset está desbalanceado (ratio 1:4).
# Esto importa: un modelo que clasifica TODO como no-pobre tiene accuracy=80%
# pero F1=0. Por eso usamos F1 como métrica, no accuracy.


# =============================================================================
# SECCIÓN 3: PREPARAR LAS MATRICES X y Y
# =============================================================================
# Usamos TODAS las variables disponibles en los datos finales.
# Se excluye únicamente 'id' (identificador sin valor predictivo) y 'pobre'
# (la variable a predecir, que no está en test).
#
# Las variables del dataset final son (construidas en 01_data_clean_and_merge.R):
#
# ► Del hogar: zona, ciudad, dpto, num_cuartos, num_cuartos_dorm, tipo_tenencia,
#   arriendo_pagado, arriendo_estimado, valor_alojamiento, num_personas,
#   num_personas_ug.
#
# ► Demográficas: n_menores_18, n_adultos_may, edad_prom, edad_jefe,
#   prop_mujeres, ratio_depend, mujer_jefa, hogar_monoparental,
#   prop_menores, prop_adultos_may_p, menores_por_ocupado.
#
# ► Educación: nivel_educ_max, nivel_educ_jefe, nivel_educ_prom,
#   educ_jefe_sq, educ_prom_sq, educ_media_o_mas, educ_superior_jefe,
#   brecha_educ.
#
# ► Mercado laboral: n_ocupados, n_desempleados, n_inactivos, tasa_ocupacion,
#   tasa_desempleo, prop_asalariado, prop_cta_propia, prop_informal,
#   prop_patron, prop_domestico, prop_fam_sin_rem, prop_jornalero,
#   prop_microempresa, prop_gran_empresa, prop_subempleo, prop_2do_empleo,
#   horas_prom, tasa_empleo_total, antiguedad_prom, antiguedad_alta.
#
# ► Salud y seguridad social: prop_afil_salud, prop_subsidiado,
#   prop_cotiza_pension, prop_pensionado, prop_prima_serv, prop_sub_alim,
#   prop_sub_transp, prop_contributivo, proteccion_social, desprotegido.
#
# ► Ingresos no laborales: alguno_pension_jub, alguno_remesas, alguno_subsidio,
#   alguno_transf_nac, alguno_intereses, alguno_cesantias, alguno_agropec,
#   alguno_arriendos.
#
# ► Vivienda: hacinamiento, hacinamiento_dorm, cuartos_pc, tenencia_insegura,
#   vivienda_propia_pag.
#
# ► Índices compuestos: indice_formalidad, formal_estricto, vulnerabilidad_lab,
#   n_fuentes_no_lab, tiene_ingreso_no_lab, solo_subsidio, ingresos_activos.
#
# ► IPM (Alkire-Foster): dep_educacion, dep_vivienda, dep_empleo_formal,
#   dep_proteccion, dep_dependencia, n_privaciones, multi_privado,
#   privado_severo.
#
# ► Región: costa_caribe, region_pacifico, bogota, eje_cafetero,
#   rural_costa_caribe.
#
# ► Interacciones: educ_x_formal, educ_x_ocup, menores_x_desocup,
#   depend_x_informal, rural_x_cta_propia, educ_x_rural,
#   subsidiado_x_menores, jefe_mayor_pension, estrato_x_zona,
#   sin_ocupados, un_solo_ocupado, doble_ingreso.
#
# ► Transformaciones no lineales: log_num_personas, log_n_menores,
#   ratio_depend_sq, tasa_ocup_sq, hacinamiento_sq, edad_jefe_sq.
#
# ► Estrato: estrato_hog, estrato_bajo.

# Columnas a excluir: identificador puro (sin valor predictivo)
# 'pobre' no está en test, así que select() la ignora automáticamente en X_test
VARS_EXCLUIR <- c("id", "pobre")

X_train <- train |> select(-any_of(VARS_EXCLUIR))
y_train <- train$pobre          # vector numérico 0/1 — lm() necesita numérico
X_test  <- test  |> select(-any_of(VARS_EXCLUIR))

# Verificar dimensiones y tipos
message("\n━━━ Matrices preparadas ━━━")
message("Predictores: ", ncol(X_train))
message("NAs en X_train: ", sum(is.na(X_train)), " (esperamos 0 — datos ya limpios)")
message("NAs en X_test:  ", sum(is.na(X_test)))

# Diagnóstico de tipos de columna: lm() convierte factores a dummies automáticamente
tipos <- X_train |> summarise(across(everything(), ~ class(.)[1]))
cat("\nTipos de columnas:\n")
print(table(unlist(tipos)))
# Si hay columnas de tipo 'factor' con muchos niveles (ciudad, dpto),
# lm() creará una dummy por nivel — esto aumenta la dimensión pero es válido.


# =============================================================================
# SECCIÓN 4: CROSS-VALIDATION SIMPLE (k-fold manual)
# =============================================================================
#
# ¿POR QUÉ HACEMOS CV?
# Si entrenamos el modelo en todos los datos y medimos el error en esos
# mismos datos, estaremos sobreestimando el rendimiento (overfitting).
# La CV divide los datos en k partes: usamos k-1 para entrenar y 1 para
# validar, rotando las partes. Así obtenemos predicciones "out-of-fold" (OOF) 
# para cada hogar, sin que el modelo haya visto ese hogar durante el entrenamiento. 
# Estas predicciones OOF son la mejor estimación del rendimiento real del modelo 
# en datos nuevos.
#
# IMPLEMENTACIÓN:
#   Usamos un loop manual (no caret::train) para que el proceso sea
#   completamente transparente. En cada iteración:
#   a) Asignamos ~80% de los hogares al entrenamiento y ~20% a validación
#   b) Ajustamos lm() en el subconjunto de entrenamiento
#   c) Predecimos sobre el subconjunto de validación
#   d) Guardamos las probabilidades predichas out-of-fold (OOF)
#
#   AL FINAL: tenemos probabilidades OOF para los 164,960 hogares.
#   Estas probabilidades se usan en la sección 5 para elegir el threshold.

K_FOLDS <- 5  # 5 folds → cada fold valida sobre ~32,992 hogares

# Asignar cada fila a uno de los K_FOLDS folds de forma aleatoria
# rep(1:5, length.out = n) crea un vector 1,2,3,4,5,1,2,3,... del tamaño exacto
# del train. Luego sample() lo permuta aleatoriamente.
fold_ids <- sample(rep(1:K_FOLDS, length.out = nrow(X_train)))

# Contenedor para las predicciones out-of-fold
# Inicializamos con NA para detectar fácilmente si algún fold no se completó
oof_probs <- rep(NA_real_, nrow(X_train))

message("\n━━━ Cross-Validation (", K_FOLDS, " folds) ━━━")

for (k in seq_len(K_FOLDS)) {

  # Índices del fold de validación (20%) y de entrenamiento (80%)
  idx_val   <- which(fold_ids == k)
  idx_train <- which(fold_ids != k)

  # Construir el data.frame de entrenamiento uniendo X y Y
  # lm() necesita que los predictores y el outcome estén en el mismo data.frame
  df_tr <- cbind(X_train[idx_train, ], pobre = y_train[idx_train])
  X_vl  <- X_train[idx_val, ]

  # Ajustar el LPM: OLS con pobre ~ . (el punto significa "todos los predictores")
  modelo_k <- lm(pobre ~ ., data = df_tr)

  # Predecir probabilidades en el fold de validación
  # type = "response" no existe en lm(); predict.lm() devuelve valores
  # continuos directamente. Pueden ser < 0 o > 1 (limitación conocida del LPM).
  probs_k <- predict(modelo_k, newdata = X_vl)

  # Guardar en el vector out-of-fold en las posiciones correctas
  oof_probs[idx_val] <- probs_k

  message(sprintf("  Fold %d/%d: rango predicciones [%.3f, %.3f]",
                  k, K_FOLDS, min(probs_k, na.rm=TRUE), max(probs_k, na.rm=TRUE)))
}

# Verificar que ningún valor quedó en NA (todos los folds se completaron)
if (any(is.na(oof_probs))) stop("Hay NAs en oof_probs. Revisar el loop de CV.")

# Diagnóstico de predicciones fuera del rango [0,1]
cat("\n━━━ Diagnóstico de predicciones OOF ━━━\n")
print(summary(oof_probs))
cat(sprintf("Predicciones < 0:  %d  (%.1f%% del total)\n",
            sum(oof_probs < 0), mean(oof_probs < 0) * 100))
cat(sprintf("Predicciones > 1:  %d  (%.1f%% del total)\n",
            sum(oof_probs > 1), mean(oof_probs > 1) * 100))

# Clampear a [0,1] para que puedan usarse como probabilidades en el threshold
# Esto no cambia la clasificación para la mayoría de observaciones, sólo
# "recorta" los extremos más allá del rango válido de una probabilidad
oof_probs_cl <- pmax(pmin(oof_probs, 1), 0)


# =============================================================================
# SECCIÓN 5: OPTIMIZACIÓN DEL THRESHOLD
# =============================================================================
#
# ¿QUÉ ES EL THRESHOLD?
#   El modelo predice un score continuo (probabilidad estimada). Para
#   clasificar en Pobre/NoPobre necesitamos un punto de corte: si la
#   probabilidad >= threshold → clasificamos como Pobre.
#
# ¿POR QUÉ NO USAR SIEMPRE 0.5?
#   Con clases desbalanceadas (20% pobres), el modelo tiende a predecir
#   probabilidades bajas para todos los hogares. El threshold que maximiza
#   F1 suele ser < 0.5 para capturar más verdaderos positivos (pobres reales).
#
# PROCEDIMIENTO:
#   Para cada threshold candidato (de 0.05 a 0.95 en pasos de 0.01),
#   aplicamos ese threshold a las predicciones OOF y calculamos F1.
#   Elegimos el threshold con el F1 más alto.
#
# IMPORTANTE: esto se hace sobre las predicciones OOF, no sobre las predicciones
#   del modelo final, para evitar sobreajuste al threshold.

th_grid <- seq(0.05, 0.95, by = 0.01)

# Para cada threshold, calcular F1, Precision y Recall
th_results <- map_dfr(th_grid, function(th) {

  # Clasificar con este threshold
  pred_class <- factor(
    if_else(oof_probs_cl >= th, "Pobre", "NoPobre"),
    levels = c("NoPobre", "Pobre")   # NoPobre = referencia, Pobre = positivo
  )
  # Etiqueta real en el mismo formato factor
  obs_class <- factor(
    if_else(y_train == 1, "Pobre", "NoPobre"),
    levels = c("NoPobre", "Pobre")
  )

  # confusionMatrix calcula automáticamente TP, FP, TN, FN y las métricas
  cm <- confusionMatrix(pred_class, obs_class, positive = "Pobre")

  tibble(
    threshold = th,
    F1        = cm$byClass["F1"],
    Precision = cm$byClass["Precision"],   # TP / (TP + FP) — cuántos predichos pobres son pobres
    Recall    = cm$byClass["Recall"]       # TP / (TP + FN) — cuántos pobres reales capturo
  )
})

# Fila con el threshold que maximiza F1
best_th_row <- th_results |>
  filter(!is.na(F1)) |>
  slice_max(F1, n = 1, with_ties = FALSE)   # si empatan, queda el primero

THRESHOLD_OPT <- best_th_row$threshold

cat("\n━━━ Threshold óptimo sobre predicciones CV ━━━\n")
print(best_th_row)

# Gráfico: cómo cambian F1, Precision y Recall con el threshold
# Útil para la slide de "policy implications":
#   - Moverse a la izquierda (umbral más bajo) = mayor recall = menos exclusión
#   - Moverse a la derecha (umbral más alto)   = mayor precision = menos filtración
p_threshold <- th_results |>
  filter(!is.na(F1)) |>
  pivot_longer(cols = c(F1, Precision, Recall), names_to = "metrica") |>
  ggplot(aes(x = threshold, y = value, color = metrica)) +
  geom_line(linewidth = 0.9) +
  # Línea vertical: threshold óptimo para F1
  geom_vline(xintercept = THRESHOLD_OPT,
             linetype = "dashed", color = "gray30") +
  # Línea vertical: threshold naive = 0.5 (para comparación)
  geom_vline(xintercept = 0.50,
             linetype = "dotted", color = "gray60") +
  annotate("text", x = THRESHOLD_OPT + 0.02, y = 0.08,
           label = sprintf("th* = %.2f\n(max F1)", THRESHOLD_OPT),
           hjust = 0, size = 3.2, color = "gray25") +
  annotate("text", x = 0.50 + 0.02, y = 0.20,
           label = "th = 0.5\n(naive)",
           hjust = 0, size = 3.2, color = "gray50") +
  scale_color_manual(
    values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3"),
    labels = c(
      F1        = "F1 (balance)",
      Precision = "Precision (↑ = menos filtración)",
      Recall    = "Recall (↑ = menos exclusión)"
    )
  ) +
  labs(
    title    = "LPM: métricas de clasificación por threshold",
    subtitle = sprintf("%d-fold CV out-of-fold | n = %d hogares", K_FOLDS, nrow(X_train)),
    x        = "Threshold de clasificación",
    y        = "Valor de la métrica",
    color    = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

ggsave(
  file.path(dir_model, "threshold.png"),
  p_threshold, width = 8, height = 5, dpi = 150
)
message("Gráfico guardado: ", MODEL_ID, "/threshold.png")


# =============================================================================
# SECCIÓN 6: ELECCIÓN DEL THRESHOLD PARA SUBMISSION
# =============================================================================
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  >>>  DECIDIR AQUÍ QUÉ THRESHOLD USAR  <<<                              │
# │                                                                          │
# │  THRESHOLD_OPT : threshold que maximizó F1 en CV → mejor para Kaggle    │
# │  0.50          : threshold naive → útil para comparación de baseline    │
# │                                                                          │
# │  Para cambiar a threshold fijo:  THRESHOLD <- 0.50                      │
# │  Para usar el óptimo de CV:      THRESHOLD <- THRESHOLD_OPT             │
# └──────────────────────────────────────────────────────────────────────────┘

THRESHOLD <- THRESHOLD_OPT   # <<< MODIFICAR si quieres explorar otro valor

cat(sprintf("\nThreshold elegido para submission: %.2f\n", THRESHOLD))
cat(sprintf("(threshold óptimo CV fue: %.2f)\n", THRESHOLD_OPT))


# =============================================================================
# SECCIÓN 7: MODELO FINAL (ajustado en todo el training set)
# =============================================================================
# La CV nos dio el F1 esperado en datos nuevos. Ahora ajustamos el modelo sobre
# TODOS los 164,960 hogares para maximizar la información usada en predicción.
# Este modelo final es el que usamos para predecir en test.
#
# Nota: el threshold elegido en la sección 6 ya fue validado sobre datos que
# el modelo final no vio (los OOF de CV). No hay data leakage.

message("\n━━━ Ajustando modelo final (todo el training set) ━━━")

df_train_full <- cbind(X_train, pobre = y_train)
model_final   <- lm(pobre ~ ., data = df_train_full)

# Resumen compacto del modelo
s <- summary(model_final)
cat("\n━━━ Bondad de ajuste del modelo final ━━━\n")
cat(sprintf("R²:           %.4f  (fracción de varianza de 'pobre' explicada por X)\n",
            s$r.squared))
cat(sprintf("R² ajustado:  %.4f  (penalizado por número de predictores)\n",
            s$adj.r.squared))
cat(sprintf("Coeficientes: %d  (incluye dummies de factores)\n",
            length(coef(model_final))))
cat(sprintf("Residual SE:  %.4f\n", s$sigma))
# Nota: el R² del LPM tiende a ser bajo (~0.3–0.4) porque la variable
# dependiente es binaria y la varianza teórica está acotada. Esto no indica
# mal ajuste — comparar con otros modelos para contextualizar.


# =============================================================================
# SECCIÓN 8: ANÁLISIS DE RESULTADOS
# =============================================================================
# Esta sección produce los insumos para las diapositivas del problem set.
# Corre independientemente de la submission; puede re-ejecutarse sin re-entrenar
# si primero se carga el .rds de diagnósticos (sección 9).

# ─── 8.1  Coeficientes más importantes ──────────────────────────────────────
# En el LPM, β_j = efecto marginal de X_j sobre P(pobre).
# Un β_j = 0.10 en 'prop_subsidiado' significa: los hogares donde toda la
# afiliación de salud es subsidiada tienen 10 pp más de probabilidad de ser
# pobres que un hogar sin afiliación subsidiada, manteniendo todo lo demás igual.
# Ordenamos por magnitud (|β|) para identificar los predictores más influyentes.

coefs_df <- broom::tidy(model_final) |>
  filter(term != "(Intercept)") |>
  mutate(
    abs_estimate = abs(estimate),
    significativo = p.value < 0.05   # señal de significancia estadística
  ) |>
  arrange(desc(abs_estimate))

cat("\n━━━ Top 20 predictores por magnitud del efecto ━━━\n")
print(
  coefs_df |>
    slice_head(n = 20) |>
    select(term, estimate, std.error, p.value, significativo),
  n = 20
)

# Gráfico de coeficientes (dot-plot con intervalos de confianza al 95%)
# Los IC se calculan como β ± 1.96 * SE (distribución asintótica normal)
p_coefs <- coefs_df |>
  slice_head(n = 20) |>
  mutate(
    term      = fct_reorder(term, estimate),
    direccion = if_else(estimate > 0,
                        "Aumenta P(pobre)",
                        "Reduce P(pobre)")
  ) |>
  ggplot(aes(x = estimate, y = term, color = direccion)) +
  geom_point(size = 3) +
  # Barra horizontal = intervalo de confianza al 95%
  geom_errorbarh(
    aes(xmin = estimate - 1.96 * std.error,
        xmax = estimate + 1.96 * std.error),
    height = 0.3
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  scale_color_manual(
    values = c("Aumenta P(pobre)" = "#E84855", "Reduce P(pobre)" = "#2196F3")
  ) +
  labs(
    title    = "LPM: efectos marginales sobre P(pobre = 1)",
    subtitle = "Top 20 predictores | Intervalos de confianza al 95%",
    x        = "Efecto marginal (puntos porcentuales)",
    y        = NULL,
    color    = NULL,
    caption  = "Coeficientes del modelo ajustado sobre todo el training set."
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top")

ggsave(
  file.path(dir_model, "coefs.png"),
  p_coefs, width = 9, height = 6, dpi = 150
)
message("Gráfico guardado: ", MODEL_ID, "/coefs.png")

# ─── 8.2  Confusion Matrix (predicciones OOF con threshold elegido) ──────────
# La confusion matrix descompone los errores del modelo en cuatro tipos:
#
#  Verdadero Positivo (TP):  hogar POBRE  → predigo POBRE    ✓  (programa correcto)
#  Falso Negativo    (FN):  hogar POBRE  → predigo NO POBRE ✗  (EXCLUSIÓN del programa)
#  Falso Positivo    (FP):  hogar NO POBRE → predigo POBRE  ✗  (FILTRACIÓN al programa)
#  Verdadero Negativo (TN):  hogar NO POBRE → predigo NO POBRE ✓ (correcto)
#
#  F1   = 2 × Precision × Recall / (Precision + Recall)  → métrica de Kaggle
#  Precision = TP / (TP + FP)  → de los que predigo pobres, qué fracción lo son
#  Recall    = TP / (TP + FN)  → de los pobres reales, qué fracción capturo

oof_clase <- factor(
  if_else(oof_probs_cl >= THRESHOLD, "Pobre", "NoPobre"),
  levels = c("NoPobre", "Pobre")
)
obs_clase <- factor(
  if_else(y_train == 1, "Pobre", "NoPobre"),
  levels = c("NoPobre", "Pobre")
)
cm_cv <- confusionMatrix(oof_clase, obs_clase, positive = "Pobre")

cat(sprintf("\n━━━ Confusion Matrix (CV out-of-fold, threshold = %.2f) ━━━\n", THRESHOLD))
print(cm_cv)

# Extraer métricas clave
cv_F1        <- cm_cv$byClass["F1"]
cv_Precision <- cm_cv$byClass["Precision"]
cv_Recall    <- cm_cv$byClass["Recall"]

cat("\n━━━ Resumen de métricas CV ━━━\n")
cat(sprintf("F1:             %.4f  (métrica de Kaggle)\n",         cv_F1))
cat(sprintf("Precision:      %.4f  (¿qué tan 'limpia' es la lista?)\n", cv_Precision))
cat(sprintf("Recall:         %.4f  (¿qué fracción de pobres captura?)\n", cv_Recall))

# Costos de política
FN <- cm_cv$table["NoPobre", "Pobre"]    # pobres clasificados como no-pobres
FP <- cm_cv$table["Pobre", "NoPobre"]    # no-pobres clasificados como pobres

cat("\n━━━ Costos de política ━━━\n")
cat(sprintf("Falsos negativos (exclusión):  %d hogares pobres excluidos del programa\n",  FN))
cat(sprintf("Falsos positivos (filtración): %d hogares no-pobres con beneficio indebido\n", FP))
cat(sprintf("Tasa exclusión:   %.1f%% de los pobres reales\n", FN / sum(y_train == 1) * 100))
cat(sprintf("Tasa filtración:  %.1f%% de los no-pobres reales\n", FP / sum(y_train == 0) * 100))

# ─── 8.3  Curva Precision-Recall ─────────────────────────────────────────────
# Con clases desbalanceadas, la curva ROC puede ser engañosamente optimista
# (muchos TN "inflan" la specificity). La curva Precision-Recall es más
# informativa: muestra el trade-off entre encontrar pobres (recall) y
# "contaminar" la lista con no-pobres (1 - precision).
# Un clasificador aleatorio en un dataset con 20% de positivos tiene AUC-PR ≈ 0.20.

roc_obj <- pROC::roc(
  response  = y_train,          # etiqueta real (0/1)
  predictor = oof_probs_cl,      # probabilidad predicha
  quiet     = TRUE
)
auc_roc <- as.numeric(pROC::auc(roc_obj))

# Construir datos para la curva PR manualmente desde los umbrales del objeto ROC
pr_df <- tibble(
  threshold   = roc_obj$thresholds,
  recall      = roc_obj$sensitivities,
  # Precision desde TP y FP:  TP = sens * n_pos,  FP = (1-spec) * n_neg
  n_pos = sum(y_train == 1),
  n_neg = sum(y_train == 0),
  TP    = roc_obj$sensitivities * n_pos,
  FP    = (1 - roc_obj$specificities) * n_neg,
  precision   = TP / (TP + FP)
) |>
  filter(is.finite(precision), is.finite(recall))

p_prcurve <- ggplot(pr_df, aes(x = recall, y = precision)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  # Línea horizontal = rendimiento de un clasificador aleatorio
  geom_hline(yintercept = mean(y_train),
             linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.85, y = mean(y_train) + 0.015,
           label = sprintf("Clasificador aleatorio (%.0f%%)", mean(y_train) * 100),
           size = 3, color = "gray50") +
  labs(
    title    = "LPM: curva Precision-Recall",
    subtitle = sprintf("AUC-ROC: %.4f | Predicciones CV out-of-fold",
                       auc_roc),
    x        = "Recall  (fracción de pobres capturada)",
    y        = "Precision  (fracción de predichos pobres que lo son)",
    caption  = "Mejor que el clasificador aleatorio → la curva está sobre la línea de puntos."
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(dir_model, "prcurve.png"),
  p_prcurve, width = 6, height = 5, dpi = 150
)
message("Gráfico guardado: ", MODEL_ID, "/prcurve.png")

# ─── 8.4  Curva ROC ───────────────────────────────────────────────────────────
# La curva ROC grafica True Positive Rate (Recall) vs. False Positive Rate
# (1 - Specificity) para todos los thresholds posibles.
# AUC-ROC = probabilidad de que el modelo le asigne mayor score a un hogar
# pobre que a uno no-pobre elegidos al azar. AUC = 0.5 → azar; AUC = 1 → perfecto.
# Con clases desbalanceadas conviene leer esta curva junto con la PR (8.3):
# la ROC tiende a verse más optimista porque los TN abundantes "inflan" el eje X.

roc_df <- tibble(
  fpr       = 1 - roc_obj$specificities,   # False Positive Rate
  tpr       = roc_obj$sensitivities         # True Positive Rate (Recall)
)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "#534AB7", linewidth = 0.9) +
  # Diagonal = clasificador aleatorio (AUC = 0.5)
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.72, y = 0.65,
           label = "Clasificador aleatorio",
           size = 3, color = "gray50", angle = 35) +
  annotate("text", x = 0.55, y = 0.15,
           label = sprintf("AUC-ROC = %.4f", auc_roc),
           size = 3.5, color = "#534AB7") +
  scale_x_continuous(labels = scales::percent_format()) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title    = "LPM: curva ROC",
    subtitle = "Predicciones CV out-of-fold",
    x        = "Tasa de Falsos Positivos  (1 − Specificity)",
    y        = "Tasa de Verdaderos Positivos  (Recall)",
    caption  = "Leer junto con la curva PR: la ROC puede ser optimista con clases desbalanceadas."
  ) +
  coord_equal() +
  theme_minimal(base_size = 12)

ggsave(
  file.path(dir_model, "roc.png"),
  p_roc, width = 6, height = 6, dpi = 150
)
message("Gráfico guardado: ", MODEL_ID, "/roc.png")


# =============================================================================
# SECCIÓN 9: PREDICCIÓN EN TEST Y SUBMISSION KAGGLE
# =============================================================================
# Aplicamos el modelo final (ajustado en todo el train) sobre los datos de test
# y clasificamos con el threshold elegido en la sección 6.

probs_test <- predict(model_final, newdata = X_test)
probs_test <- pmax(pmin(probs_test, 1), 0)       # clampear a [0,1]
preds_test <- as.integer(probs_test >= THRESHOLD)

cat(sprintf(
  "\n━━━ Predicciones en test ━━━\n%d hogares predichos como pobres (%.1f%%) | threshold = %.2f\n",
  sum(preds_test), mean(preds_test) * 100, THRESHOLD
))

# El CSV debe tener exactamente dos columnas: id y pobre
submission <- tibble(
  id    = test$id,
  pobre = preds_test
)

submission_file <- file.path(
  dir_submissions,
  paste0("submission_", MODEL_ID, ".csv")
)
write_csv(submission, submission_file)
message("Submission guardada: ", submission_file)
message("→ Subir este archivo a Kaggle y anotar el public F1 en model_registry.csv")


# =============================================================================
# SECCIÓN 10: GUARDAR DIAGNÓSTICOS
# =============================================================================
# Guardamos todo en un .rds para poder recargar sin re-entrenar.
# Útil para: comparar modelos, hacer gráficos adicionales, debugging.

diagnostics <- list(
  # Identificación
  model_id        = MODEL_ID,
  autor           = AUTOR,
  notas           = NOTAS,
  fecha           = Sys.Date(),
  # Modelos
  model_final     = model_final,   # objeto lm() completo
  # Predicciones CV (para análisis posterior)
  oof_probs_raw   = oof_probs,     # sin clampear — útil para análisis de extremos
  oof_probs_cl    = oof_probs_cl,  # clampeadas a [0,1]
  oof_actuals     = y_train,
  fold_ids        = fold_ids,      # qué fila fue a qué fold — para reproducir exactamente
  # Optimización de threshold
  th_results      = th_results,
  threshold_opt   = THRESHOLD_OPT,
  threshold_usado = THRESHOLD,
  # Métricas
  cm_cv           = cm_cv,
  cv_F1           = cv_F1,
  cv_Precision    = cv_Precision,
  cv_Recall       = cv_Recall,
  auc_roc         = auc_roc,
  # Variables
  predictores     = names(X_train),
  n_features      = ncol(X_train)
)

diag_file <- file.path(dir_model, "diagnostics.rds")
saveRDS(diagnostics, diag_file)
message("Diagnósticos guardados: ", MODEL_ID, "/diagnostics.rds")


# =============================================================================
# SECCIÓN 11: REGISTRAR EN MODEL REGISTRY
# =============================================================================
# model_registry.csv es la tabla maestra del equipo: una fila por corrida.
# Si el archivo no existe, lo crea. Si ya existe, agrega la fila al final.
# Permite comparar todos los modelos del equipo en un solo lugar.
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  >>>  DESPUÉS DE SUBIR LA SUBMISSION A KAGGLE  <<<                      │
# │  Editar manualmente model_registry.csv y completar kaggle_public_F1     │
# │  con el puntaje que aparece en el public leaderboard de Kaggle.         │
# └──────────────────────────────────────────────────────────────────────────┘

nueva_fila <- tibble(
  model_id           = MODEL_ID,
  fecha              = as.character(Sys.Date()),
  autor              = AUTOR,
  algoritmo          = "LPM",
  n_features         = ncol(X_train),
  imbalance_strategy = "none",
  cv_folds           = K_FOLDS,
  cv_F1              = round(cv_F1,        4),
  cv_Precision       = round(cv_Precision, 4),
  cv_Recall          = round(cv_Recall,    4),
  auc_roc            = round(auc_roc,      4),
  kaggle_public_F1   = NA_real_,   # <<< llenar manualmente después de Kaggle
  threshold          = THRESHOLD,
  notas              = NOTAS
)

# Si el registro ya existe, leer y agregar; si no, crear desde cero
if (file.exists(registry_path)) {
  registry <- read_csv(registry_path, show_col_types = FALSE)
  registry <- bind_rows(registry, nueva_fila)
} else {
  registry <- nueva_fila
}
write_csv(registry, registry_path)
message("Registro actualizado: ", registry_path)

cat("\n━━━ Fila registrada en model_registry.csv ━━━\n")
print(nueva_fila)

message("\n═══════════════════════════════════════════════")
message("  LPM ", MODEL_ID, " completado ✓")
message("═══════════════════════════════════════════════\n")


# =============================================================================
# SECCIÓN 12: IDEAS PARA MEJORAR EL F1 (no ejecutar — anotar para iterar)
# =============================================================================
# Para cada experimento: incrementar MODEL_ID, actualizar NOTAS, correr el
# script entero, subir a Kaggle y registrar el public F1.

# ── IDEA 1: Ajustar el threshold hacia un valor de política ──────────────────
# El F1 equilibra precision y recall con igual peso (β=1 en F_β). Pero en
# política social, excluir a un pobre (FN) puede ser más costoso que filtrar
# a un no-pobre (FP). Para dar más peso al recall:
#   Buscar el threshold que maximiza F_2 (que pondera recall el doble que precision):
#   F2 = 5 * Precision * Recall / (4 * Precision + Recall)
#   Sustituir en la búsqueda de th_results:
#     F2 = (1 + 4) * (Precision * Recall) / (4 * Precision + Recall)

# ── IDEA 2: Pesos por clase ──────────────────────────────────────────────────
# lm() acepta un argumento `weights`. Dar más peso a la clase minoritaria
# penaliza más los errores sobre hogares pobres → aumenta recall.
#   n_pos   <- sum(y_train == 1)
#   n_neg   <- sum(y_train == 0)
#   w_train <- if_else(y_train == 1, n_neg / n_pos, 1.0)
#   lm(pobre ~ ., data = df_train_full, weights = w_train)
# Impacto esperado: recall ↑, precision ↓. Si el F1 mejora, mantener.

# ── IDEA 3: Selección de variables por valor-p o magnitud ────────────────────
# Hay ~114 variables + dummies de factores. Muchas pueden ser ruidosas.
# Alternativa 1: retener solo los predictores con p-value < 0.05 en el modelo final.
# Alternativa 2: retener solo los predictores con |beta| > 0.03 (umbral pragmático).
# Un modelo más parsimonioso puede generalizar mejor (menos overfitting).
# Implementación:
#   vars_seleccionadas <- coefs_df |> filter(abs_estimate > 0.03) |> pull(term)
#   # Ajustar con solo esas variables y comparar F1 CV

# ── IDEA 4: Interacciones adicionales ────────────────────────────────────────
# Variables que podrían capturar efectos heterogéneos no modelados:
#   zona × nivel_educ_jefe       (retornos a educación diferenciados por zona)
#   n_menores_18 × sin_ocupados  (máxima carga sin ingreso laboral)
#   estrato_hog × prop_informal  (estrato bajo + informal = doble riesgo)
#   edad_jefe × nivel_educ_jefe  (ciclo de vida con mayor o menor capital humano)
# Crear las interacciones en un nuevo tibble y usarlo como X_train alternativo.

# ── IDEA 5: Estratificar los folds de CV ──────────────────────────────────────
# Con 20% de pobres, algunos folds por azar tendrán menos/más pobres.
# Estratificar garantiza que cada fold tenga el mismo ratio de clases.
#   folds_estratificados <- caret::createFolds(
#     y    = factor(y_train),
#     k    = K_FOLDS,
#     list = TRUE      # devuelve una lista con los índices de validación por fold
#   )
#   # Reemplazar el loop usando folds_estratificados[[k]] en lugar de which(fold_ids == k)

# ── IDEA 6: Comparar LPM vs. Logit ──────────────────────────────────────────
# La diferencia fundamental: Logit modela P(Y=1|X) = σ(Xβ) donde σ es la
# función sigmoide → las predicciones siempre están en [0,1].
# El LPM estima la misma cantidad pero sin la restricción [0,1].
# Si el LPM produce muchas predicciones fuera de [0,1], el Logit puede
# ser más estable. Para comparar con el mismo set de variables:
#   modelo_logit <- glm(pobre ~ ., data = df_train_full, family = binomial)
#   probs_logit  <- predict(modelo_logit, newdata = X_test, type = "response")
