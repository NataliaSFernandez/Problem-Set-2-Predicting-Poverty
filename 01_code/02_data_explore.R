
#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script 02: Análisis Exploratorio de Datos (EDA)
#==============================================================================
# OBJETIVO: Explorar los datos limpios para entender la estructura del problema,
#           identificar variables informativas y guiar el modelado.
#
# INPUT:  00_data/processed/train_final.rds
#         00_data/processed/test_final.rds
#
# OUTPUT: Gráficos y tablas que responden las 3 preguntas de Fase 1:
#   1. ¿Qué variables deberían predecir pobreza según razonamiento económico?
#   2. ¿Qué patrones en los datos confirman o desafían esas hipótesis?
#   3. ¿Qué tan severo es el desbalance de clases y qué implica para el modelado?
#
# AUTOR: Daniel Solano
# FECHA: 2026-03-14
#==============================================================================

# ==============================================================================
# BLOQUE 0: PAQUETES Y DATOS
# ==============================================================================
# Cargamos los paquetes que necesitamos:
# - tidyverse: para manipular datos y hacer gráficos con ggplot2
# - corrplot: para visualizar la matriz de correlaciones
# - skimr: para un resumen rápido de todas las variables

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,   # manipulación de datos + ggplot2
  corrplot,    # visualización de correlaciones
  skimr,       # resumen rápido de dataframes
  scales       # formateo de ejes en gráficos
)

# Cargamos los datos limpios que generó Natalia en el script 01
dir_processed <- "00_data/processed"
train <- readRDS(file.path(dir_processed, "train_final.rds"))
test  <- readRDS(file.path(dir_processed, "test_final.rds"))

# Convertimos 'pobre' a factor con etiquetas legibles para los gráficos
# 0 = No pobre, 1 = Pobre
train <- train %>%
  mutate(pobre_f = factor(pobre, levels = c(0, 1), labels = c("No pobre", "Pobre")))

cat("Train:", nrow(train), "hogares |", ncol(train), "columnas\n")
cat("Test: ", nrow(test),  "hogares |", ncol(test),  "columnas\n")


# ==============================================================================
# BLOQUE 1: DESBALANCE DE CLASES
# ==============================================================================
# Pregunta 3 del plan: ¿Qué tan severo es el desbalance y qué implica?
#
# ¿Por qué importa? Si el 80% de hogares NO son pobres, un modelo que
# simplemente diga "nadie es pobre" tendría 80% de accuracy. Pero no
# identificaría a ningún hogar pobre — su recall sería 0.
# El F1 score (que usa Kaggle) penaliza esto porque necesita TANTO
# precisión COMO recall. Así que tenemos que prestarle atención especial
# a la clase minoritaria (pobres).

cat("\n========================================\n")
cat("BLOQUE 1: DESBALANCE DE CLASES\n")
cat("========================================\n")

tabla_pobre <- train %>%
  count(pobre_f) %>%
  mutate(
    porcentaje = round(n / sum(n) * 100, 1),
    etiqueta   = paste0(format(n, big.mark = ","), " (", porcentaje, "%)")
  )

print(tabla_pobre)
cat("\nRatio No pobre : Pobre =",
    round(tabla_pobre$n[1] / tabla_pobre$n[2], 1), ": 1\n")

# Gráfico de barras del desbalance
ggplot(tabla_pobre, aes(x = pobre_f, y = n, fill = pobre_f)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = etiqueta), vjust = -0.5, size = 4) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.15))) +
  scale_fill_manual(values = c("No pobre" = "#2E86AB", "Pobre" = "#E84855")) +
  labs(
    title    = "Distribución de la variable objetivo",
    subtitle = "Desbalance 4:1 — la clase 'Pobre' es minoritaria",
    x = NULL, y = "Número de hogares"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("01_desbalance_clases.png", width = 7, height = 5, dpi = 150)


# ==============================================================================
# BLOQUE 2: RESUMEN DESCRIPTIVO GENERAL
# ==============================================================================
# Aquí vemos un panorama rápido de todas las variables: medias, medianas,
# valores faltantes, distribución. Esto nos ayuda a detectar problemas
# (variables con demasiados NAs, distribuciones extrañas, etc.)

cat("\n========================================\n")
cat("BLOQUE 2: RESUMEN DESCRIPTIVO GENERAL\n")
cat("========================================\n")

# Seleccionamos solo las variables numéricas (excluyendo id y pobre_f)
vars_numericas <- train %>%
  select(where(is.numeric), -pobre) %>%
  names()

cat("Variables numéricas disponibles:", length(vars_numericas), "\n\n")

# Resumen rápido con skimr
skim_result <- train %>%
  select(all_of(vars_numericas), pobre) %>%
  skim()

print(skim_result)

# Variables con muchos NAs (más del 10%)
vars_na <- train %>%
  summarise(across(all_of(vars_numericas), ~ mean(is.na(.)) * 100)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "pct_na") %>%
  filter(pct_na > 10) %>%
  arrange(desc(pct_na))

cat("\n-- Variables con más de 10% de NAs --\n")
print(vars_na)
cat("\nNota: arriendo_pagado tiene 96.6% NA porque solo aplica a hogares que arriendan.\n")
cat("horas_prom y prop_cta_propia tienen ~13.5% NA (hogares sin nadie en PET).\n")


# ==============================================================================
# BLOQUE 3: DIFERENCIAS ENTRE POBRES Y NO POBRES
# ==============================================================================
# Pregunta 1 y 2: ¿Qué variables predicen pobreza y qué muestran los datos?
#
# La intuición económica nos dice que la pobreza se asocia con:
# - Menor educación (del jefe y del hogar en general)
# - Menor tasa de ocupación / mayor desempleo
# - Mayor informalidad laboral
# - Más dependientes (menores, adultos mayores) por persona en edad productiva
# - Menor acceso a seguridad social (pensión, salud contributiva)
# - Hogares más grandes con hacinamiento
#
# Vamos a verificar si los datos confirman esto.

cat("\n========================================\n")
cat("BLOQUE 3: COMPARACIÓN POBRES vs NO POBRES\n")
cat("========================================\n")

# Seleccionamos variables clave para la comparación
vars_comparar <- c(
  # Vivienda
  "num_cuartos", "num_cuartos_dorm", "tipo_tenencia",
  # Demografía
  "num_personas", "n_menores_18", "n_adultos_may",
  "edad_jefe", "prop_mujeres", "ratio_depend",
  # Educación
  "nivel_educ_max", "nivel_educ_jefe", "nivel_educ_prom",
  # Mercado laboral
  "n_ocupados", "tasa_ocupacion", "tasa_desempleo",
  "prop_asalariado", "prop_informal",
  # Seguridad social
  "prop_afil_salud", "prop_subsidiado", "prop_cotiza_pension",
  # Otras
  "prop_prima_serv", "alguno_subsidio", "alguno_remesas",
  "horas_prom"
)

# Tabla de medias por grupo
tabla_medias <- train %>%
  group_by(pobre_f) %>%
  summarise(
    across(all_of(vars_comparar), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  pivot_longer(-pobre_f, names_to = "variable", values_to = "media") %>%
  pivot_wider(names_from = pobre_f, values_from = media) %>%
  mutate(
    diferencia = Pobre - `No pobre`,
    ratio      = round(Pobre / `No pobre`, 2)
  ) %>%
  arrange(desc(abs(diferencia)))

cat("\n-- Medias por grupo (ordenadas por diferencia absoluta) --\n")
print(tabla_medias, n = 30)


# ==============================================================================
# BLOQUE 4: CORRELACIONES CON LA VARIABLE OBJETIVO
# ==============================================================================
# Calculamos la correlación de Pearson de cada variable numérica con 'pobre'.
# Valores positivos = a más de esa variable, más probabilidad de ser pobre.
# Valores negativos = a más de esa variable, menos probabilidad de ser pobre.

cat("\n========================================\n")
cat("BLOQUE 4: CORRELACIONES CON 'pobre'\n")
cat("========================================\n")

# Calculamos correlación solo con variables sin demasiados NAs
vars_para_corr <- train %>%
  select(all_of(vars_numericas)) %>%
  select(where(~ mean(is.na(.)) < 0.15)) %>%
  names()

correlaciones <- train %>%
  select(pobre, all_of(vars_para_corr)) %>%
  cor(use = "pairwise.complete.obs") %>%
  as.data.frame() %>%
  rownames_to_column("variable") %>%
  select(variable, corr_pobre = pobre) %>%
  filter(variable != "pobre") %>%
  arrange(desc(abs(corr_pobre)))

cat("\n-- Top 20 variables más correlacionadas con pobreza --\n")
print(head(correlaciones, 20))

# Gráfico de barras de correlaciones (top 20)
top_corr <- head(correlaciones, 20)

ggplot(top_corr, aes(x = reorder(variable, corr_pobre), y = corr_pobre,
                      fill = corr_pobre > 0)) +
  geom_col(width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#E84855", "FALSE" = "#2E86AB"),
                    labels = c("Reduce pobreza", "Aumenta pobreza")) +
  labs(
    title    = "Correlación de cada variable con la pobreza",
    subtitle = "Top 20 variables más asociadas (correlación de Pearson)",
    x = NULL, y = "Correlación con 'pobre'",
    fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("02_correlaciones_pobre.png", width = 8, height = 7, dpi = 150)


# ==============================================================================
# BLOQUE 5: VISUALIZACIONES CLAVE
# ==============================================================================
# Gráficos de las variables más informativas comparando distribución
# entre hogares pobres y no pobres.

cat("\n========================================\n")
cat("BLOQUE 5: VISUALIZACIONES CLAVE\n")
cat("========================================\n")

# --- 5.1 Nivel educativo del jefe del hogar ---
# Hipótesis: mayor educación del jefe = mayor capacidad de generar
# ingreso = menor probabilidad de pobreza.

ggplot(train, aes(x = factor(nivel_educ_jefe), fill = pobre_f)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent) +
  scale_fill_manual(values = c("No pobre" = "#2E86AB", "Pobre" = "#E84855")) +
  labs(
    title    = "Proporción de pobreza por nivel educativo del jefe de hogar",
    subtitle = "1=Ninguno, 2=Preescolar, 3=Primaria, 4=Secundaria, 5=Media, 6=Superior",
    x = "Nivel educativo del jefe", y = "Proporción",
    fill = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

ggsave("03_educ_jefe_vs_pobre.png", width = 8, height = 5, dpi = 150)


# --- 5.2 Ratio de dependencia ---
# Hipótesis: hogares con más dependientes (menores + adultos mayores)
# por persona en edad productiva tienen menos ingresos per cápita.

ggplot(train, aes(x = pobre_f, y = ratio_depend, fill = pobre_f)) +
  geom_boxplot(outlier.alpha = 0.1) +
  scale_fill_manual(values = c("No pobre" = "#2E86AB", "Pobre" = "#E84855")) +
  labs(
    title = "Ratio de dependencia por condición de pobreza",
    subtitle = "(Menores de 15 + Mayores de 65) / Personas en edad productiva",
    x = NULL, y = "Ratio de dependencia",
    fill = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("04_ratio_depend_vs_pobre.png", width = 7, height = 5, dpi = 150)


# --- 5.3 Proporción de cotizantes a pensión ---
# Hipótesis: hogares donde nadie cotiza a pensión tienden a estar en
# empleos informales con ingresos más bajos e inestables.

ggplot(train, aes(x = pobre_f, y = prop_cotiza_pension, fill = pobre_f)) +
  geom_boxplot(outlier.alpha = 0.1) +
  scale_fill_manual(values = c("No pobre" = "#2E86AB", "Pobre" = "#E84855")) +
  labs(
    title = "Proporción de miembros que cotizan pensión",
    subtitle = "Proxy de formalidad laboral del hogar",
    x = NULL, y = "Proporción que cotiza pensión",
    fill = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("05_pension_vs_pobre.png", width = 7, height = 5, dpi = 150)


# --- 5.4 Tasa de ocupación del hogar ---
# Hipótesis: a mayor proporción de personas ocupadas,
# mayor ingreso total y menor probabilidad de pobreza.

ggplot(train, aes(x = pobre_f, y = tasa_ocupacion, fill = pobre_f)) +
  geom_boxplot(outlier.alpha = 0.1) +
  scale_fill_manual(values = c("No pobre" = "#2E86AB", "Pobre" = "#E84855")) +
  labs(
    title = "Tasa de ocupación del hogar por condición de pobreza",
    subtitle = "Proporción de personas en edad de trabajar que están ocupadas",
    x = NULL, y = "Tasa de ocupación",
    fill = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("06_tasa_ocup_vs_pobre.png", width = 7, height = 5, dpi = 150)


# --- 5.5 Proporción con régimen subsidiado de salud ---
# Hipótesis: mayor proporción de subsidiados = mayor vulnerabilidad.

ggplot(train %>% filter(!is.na(prop_subsidiado)),
       aes(x = pobre_f, y = prop_subsidiado, fill = pobre_f)) +
  geom_boxplot(outlier.alpha = 0.1) +
  scale_fill_manual(values = c("No pobre" = "#2E86AB", "Pobre" = "#E84855")) +
  labs(
    title = "Proporción en régimen subsidiado de salud",
    subtitle = "Mayor proporción de subsidiados = mayor vulnerabilidad económica",
    x = NULL, y = "Proporción en régimen subsidiado",
    fill = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("07_subsidiado_vs_pobre.png", width = 7, height = 5, dpi = 150)


# --- 5.6 Número de menores de 18 años ---
# Hipótesis: más menores = más dependientes sin ingreso = mayor pobreza.

ggplot(train, aes(x = factor(n_menores_18), fill = pobre_f)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent) +
  scale_fill_manual(values = c("No pobre" = "#2E86AB", "Pobre" = "#E84855")) +
  labs(
    title = "Proporción de pobreza según número de menores en el hogar",
    x = "Número de menores de 18 años", y = "Proporción",
    fill = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

ggsave("08_menores_vs_pobre.png", width = 8, height = 5, dpi = 150)


# ==============================================================================
# BLOQUE 6: MATRIZ DE CORRELACIONES ENTRE PREDICTORES
# ==============================================================================
# Detectar multicolinealidad: si dos variables están muy correlacionadas
# entre sí, incluir ambas en un modelo lineal puede ser redundante.

cat("\n========================================\n")
cat("BLOQUE 6: MATRIZ DE CORRELACIONES\n")
cat("========================================\n")

vars_matriz <- c(
  "nivel_educ_jefe", "nivel_educ_max", "nivel_educ_prom",
  "ratio_depend", "n_menores_18", "num_personas",
  "tasa_ocupacion", "tasa_desempleo", "prop_informal",
  "prop_cotiza_pension", "prop_subsidiado",
  "prop_afil_salud", "prop_asalariado",
  "edad_jefe", "num_cuartos_dorm", "pobre"
)

mat_corr <- train %>%
  select(all_of(vars_matriz)) %>%
  cor(use = "pairwise.complete.obs")

png("09_matriz_correlaciones.png", width = 900, height = 800, res = 120)
corrplot(mat_corr,
         method  = "color",
         type    = "lower",
         tl.col  = "black",
         tl.cex  = 0.8,
         addCoef.col = "black",
         number.cex  = 0.6,
         title   = "Matriz de correlaciones entre predictores clave",
         mar     = c(0, 0, 2, 0))
dev.off()


# ==============================================================================
# BLOQUE 7: RESUMEN Y CONCLUSIONES
# ==============================================================================

cat("\n========================================\n")
cat("BLOQUE 7: RESUMEN DE HALLAZGOS\n")
cat("========================================\n")

cat("
PREGUNTA 1: ¿Qué variables deberían predecir pobreza?
------------------------------------------------------
Desde razonamiento económico, las variables más relevantes son:
- Educación del jefe del hogar (capital humano -> capacidad de ingreso)
- Tasa de ocupación del hogar (más ocupados -> más fuentes de ingreso)
- Informalidad laboral (empleo informal -> ingresos bajos e inestables)
- Ratio de dependencia (más dependientes -> menos ingreso per cápita)
- Acceso a seguridad social (pensión, salud contributiva -> empleo formal)
- Número de menores (costo de crianza sin ingreso adicional)
- Zona (urbano/rural -> acceso a mercados laborales, servicios)

PREGUNTA 2: ¿Qué confirman o desafían los datos?
-------------------------------------------------
Revisa la tabla de medias del Bloque 3 y los gráficos del Bloque 5.
Las variables que mostraron mayor separación entre grupos fueron:
[Completa esto después de correr el script y ver los resultados]

PREGUNTA 3: ¿Qué tan severo es el desbalance y qué implica?
------------------------------------------------------------
- El desbalance es 4:1 (80% no pobres, 20% pobres)
- Esto implica que:
  * Accuracy NO es buena métrica (80% sin hacer nada)
  * Usar F1 score es apropiado porque penaliza modelos que ignoran
    la clase minoritaria
  * Estrategias a considerar en el modelado:
    - Ajustar el threshold de clasificación (no usar 0.5 por defecto)
    - Upsampling/downsampling en el entrenamiento
    - Usar metric='Sens' o summaryFunction=twoClassSummary en caret
    - Usar class weights en algoritmos que lo soporten
")

cat("\n¡EDA completo! Los gráficos se guardaron en la carpeta de trabajo.\n")
cat("Archivos generados:\n")
cat("  01_desbalance_clases.png\n")
cat("  02_correlaciones_pobre.png\n")
cat("  03_educ_jefe_vs_pobre.png\n")
cat("  04_ratio_depend_vs_pobre.png\n")
cat("  05_pension_vs_pobre.png\n")
cat("  06_tasa_ocup_vs_pobre.png\n")
cat("  07_subsidiado_vs_pobre.png\n")
cat("  08_menores_vs_pobre.png\n")
cat("  09_matriz_correlaciones.png\n")