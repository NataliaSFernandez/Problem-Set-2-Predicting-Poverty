#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script 02: Analisis Exploratorio de Datos (EDA)
#==============================================================================
# OBJETIVO: Explorar los datos limpios para entender la estructura del problema,
#           identificar variables informativas y guiar el modelado.
#
# INPUT:  00_data/processed/train_final.rds
#
# OUTPUT: 02_outputs/figures/data_explore/
#         - 9 graficos PNG para slides
#         - resultados_eda.txt con tablas de metricas
#
# Group 04
#==============================================================================

# ==============================================================================
# BLOQUE 0: PAQUETES Y DATOS
# ==============================================================================
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,
  corrplot,
  skimr,
  scales
)

# Carpeta de salida
dir_figures <- "02_outputs/figures/data_explore"
if (!dir.exists(dir_figures)) dir.create(dir_figures, recursive = TRUE)

# Cargar datos
train <- readRDS("00_data/processed/train_final.rds")

train <- train |>
  mutate(pobre_f = factor(pobre, levels = c(0, 1), labels = c("No pobre", "Pobre")))

cat("Train:", nrow(train), "hogares |", ncol(train), "columnas\n")

# Paleta consistente para todos los graficos
COL_NO <- "#2E86AB"
COL_SI <- "#E84855"

# Tema base para todos los graficos (fondo blanco, limpio)
tema_base <- theme_minimal(base_size = 13) +
  theme(
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", size = 14),
    plot.subtitle    = element_text(color = "grey40", size = 11)
  )


# ==============================================================================
# GRAFICO 1: DESBALANCE DE CLASES
# ==============================================================================
# Contexto: el 80% de hogares NO son pobres. Esto justifica el uso de F1
# en vez de accuracy, y la necesidad de optimizar el threshold.

cat("\n== Grafico 1: Desbalance de clases ==\n")

tabla_pobre <- train |>
  count(pobre_f) |>
  mutate(
    porcentaje = round(n / sum(n) * 100, 1),
    etiqueta   = paste0(format(n, big.mark = ","), " (", porcentaje, "%)")
  )

print(tabla_pobre)
cat("Ratio No pobre : Pobre =",
    round(tabla_pobre$n[1] / tabla_pobre$n[2], 1), ": 1\n")

ggplot(tabla_pobre, aes(x = pobre_f, y = n, fill = pobre_f)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = etiqueta), vjust = -0.5, size = 4.5) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.15))) +
  scale_fill_manual(values = c("No pobre" = COL_NO, "Pobre" = COL_SI)) +
  labs(
    title    = "Distribucion de la variable objetivo",
    subtitle = "Desbalance 4:1 entre hogares no pobres y pobres",
    x = NULL, y = "Numero de hogares"
  ) +
  tema_base +
  theme(legend.position = "none")

ggsave(file.path(dir_figures, "01_desbalance_clases.png"),
       width = 7, height = 5, dpi = 200)


# ==============================================================================
# GRAFICO 2: POBREZA POR ZONA (URBANO vs RURAL)
# ==============================================================================
# La pobreza rural es estructuralmente mayor por menor acceso a mercados
# laborales formales, servicios publicos y educacion.

cat("\n== Grafico 2: Pobreza por zona ==\n")

zona_pobre <- train |>
  mutate(zona_lab = ifelse(zona == 1, "Urbano", "Rural")) |>
  group_by(zona_lab) |>
  summarise(
    n = n(),
    pobres = sum(pobre),
    tasa = mean(pobre) * 100,
    .groups = "drop"
  )

print(zona_pobre)

train |>
  mutate(zona_lab = factor(ifelse(zona == 1, "Urbano", "Rural"),
                            levels = c("Urbano", "Rural"))) |>
  ggplot(aes(x = zona_lab, fill = pobre_f)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent) +
  scale_fill_manual(values = c("No pobre" = COL_NO, "Pobre" = COL_SI)) +
  labs(
    title    = "Tasa de pobreza por zona",
    subtitle = "La pobreza rural duplica la urbana por menor acceso a empleo formal",
    x = NULL, y = "Proporcion", fill = NULL
  ) +
  tema_base +
  theme(legend.position = "bottom")

ggsave(file.path(dir_figures, "02_pobreza_zona.png"),
       width = 7, height = 5, dpi = 200)


# ==============================================================================
# GRAFICO 3: POBREZA POR DEPARTAMENTO
# ==============================================================================
# La pobreza tiene un componente geografico fuerte en Colombia.
# Departamentos como Choco y La Guajira tienen tasas mucho mayores
# que Bogota o Antioquia.

cat("\n== Grafico 3: Pobreza por departamento ==\n")

dpto_pobre <- train |>
  group_by(dpto) |>
  summarise(
    n = n(),
    tasa = mean(pobre) * 100,
    .groups = "drop"
  ) |>
  arrange(desc(tasa))

print(dpto_pobre)

dpto_pobre |>
  mutate(dpto = fct_reorder(as.character(dpto), tasa)) |>
  ggplot(aes(x = dpto, y = tasa / 100, fill = tasa)) +
  geom_col(width = 0.7) +
  coord_flip() +
  scale_y_continuous(labels = percent) +
  scale_fill_gradient(low = COL_NO, high = COL_SI, guide = "none") +
  labs(
    title    = "Tasa de pobreza por departamento",
    subtitle = "Heterogeneidad geografica: la pobreza varia hasta 5x entre departamentos",
    x = NULL, y = "Tasa de pobreza"
  ) +
  tema_base

ggsave(file.path(dir_figures, "03_pobreza_dpto.png"),
       width = 8, height = 7, dpi = 200)


# ==============================================================================
# GRAFICO 4: EDUCACION DEL JEFE vs POBREZA
# ==============================================================================
# Capital humano es el determinante mas fuerte de ingreso y de pobreza.
# Mayor educacion del jefe = mayor capacidad de generar ingreso = menor pobreza.
# Escala: 1=Ninguno, 2=Preescolar, 3=Primaria, 4=Secundaria, 5=Media, 6=Superior

cat("\n== Grafico 4: Educacion del jefe vs pobreza ==\n")

educ_labels <- c("1\nNinguno", "2\nPreescolar", "3\nPrimaria",
                  "4\nSecundaria", "5\nMedia", "6\nSuperior")

train |>
  filter(nivel_educ_jefe %in% 1:6) |>
  ggplot(aes(x = factor(nivel_educ_jefe), fill = pobre_f)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent) +
  scale_x_discrete(labels = educ_labels) +
  scale_fill_manual(values = c("No pobre" = COL_NO, "Pobre" = COL_SI)) +
  labs(
    title    = "Pobreza segun nivel educativo del jefe de hogar",
    subtitle = "Cada nivel adicional de educacion reduce la probabilidad de pobreza",
    x = "Nivel educativo del jefe", y = "Proporcion", fill = NULL
  ) +
  tema_base +
  theme(legend.position = "bottom")

ggsave(file.path(dir_figures, "04_educ_jefe_vs_pobre.png"),
       width = 8, height = 5, dpi = 200)


# ==============================================================================
# GRAFICO 5: FORMALIDAD LABORAL vs POBREZA
# ==============================================================================
# La informalidad atrapa a los hogares en ingresos bajos e inestables,
# sin acceso a seguridad social. prop_contributivo mide la proporcion
# de miembros en regimen contributivo de salud (proxy de empleo formal).

cat("\n== Grafico 5: Formalidad laboral vs pobreza ==\n")

ggplot(train, aes(x = pobre_f, y = prop_contributivo, fill = pobre_f)) +
  geom_boxplot(outlier.alpha = 0.1, width = 0.5) +
  scale_fill_manual(values = c("No pobre" = COL_NO, "Pobre" = COL_SI)) +
  labs(
    title    = "Proporcion en regimen contributivo de salud",
    subtitle = "Los hogares pobres casi no tienen miembros con empleo formal",
    x = NULL, y = "Proporcion en regimen contributivo", fill = NULL
  ) +
  tema_base +
  theme(legend.position = "none")

ggsave(file.path(dir_figures, "05_formalidad_vs_pobre.png"),
       width = 7, height = 5, dpi = 200)


# ==============================================================================
# GRAFICO 6: CARGA DE DEPENDIENTES vs POBREZA
# ==============================================================================
# Mas dependientes (menores) por trabajador = menos ingreso per capita.
# Es una trampa de pobreza: hogares pobres tienen mas hijos,
# lo que reduce su ingreso per capita y los mantiene pobres.

cat("\n== Grafico 6: Carga de dependientes ==\n")

ggplot(train, aes(x = pobre_f, y = menores_por_ocupado, fill = pobre_f)) +
  geom_boxplot(outlier.alpha = 0.1, width = 0.5) +
  scale_fill_manual(values = c("No pobre" = COL_NO, "Pobre" = COL_SI)) +
  coord_cartesian(ylim = c(0, 5)) +
  labs(
    title    = "Menores por persona ocupada en el hogar",
    subtitle = "Los hogares pobres sostienen mas dependientes por cada trabajador",
    x = NULL, y = "Menores de 18 / Ocupados", fill = NULL
  ) +
  tema_base +
  theme(legend.position = "none")

ggsave(file.path(dir_figures, "06_menores_por_ocupado.png"),
       width = 7, height = 5, dpi = 200)


# ==============================================================================
# GRAFICO 7: HACINAMIENTO vs POBREZA
# ==============================================================================
# Condiciones de vivienda como proxy de bienestar y acumulacion de activos.
# Menos cuartos per capita = mayor hacinamiento = mayor pobreza.

cat("\n== Grafico 7: Hacinamiento ==\n")

ggplot(train, aes(x = pobre_f, y = cuartos_pc, fill = pobre_f)) +
  geom_boxplot(outlier.alpha = 0.1, width = 0.5) +
  scale_fill_manual(values = c("No pobre" = COL_NO, "Pobre" = COL_SI)) +
  coord_cartesian(ylim = c(0, 4)) +
  labs(
    title    = "Cuartos per capita por condicion de pobreza",
    subtitle = "Los hogares pobres viven en condiciones de mayor hacinamiento",
    x = NULL, y = "Cuartos per capita", fill = NULL
  ) +
  tema_base +
  theme(legend.position = "none")

ggsave(file.path(dir_figures, "07_hacinamiento.png"),
       width = 7, height = 5, dpi = 200)


# ==============================================================================
# GRAFICO 8: TOP 15 CORRELACIONES CON POBREZA
# ==============================================================================
# Las variables mas correlacionadas con pobreza guian la seleccion de
# predictores para el modelado.

cat("\n== Grafico 8: Top 15 correlaciones ==\n")

vars_numericas <- train |>
  select(where(is.numeric), -pobre) |>
  select(where(~ mean(is.na(.)) < 0.15)) |>
  names()

correlaciones <- train |>
  select(pobre, all_of(vars_numericas)) |>
  cor(use = "pairwise.complete.obs") |>
  as.data.frame() |>
  rownames_to_column("variable") |>
  select(variable, corr_pobre = pobre) |>
  filter(variable != "pobre") |>
  arrange(desc(abs(corr_pobre)))

cat("Top 15 correlaciones:\n")
print(head(correlaciones, 15))

top_corr <- head(correlaciones, 15)

ggplot(top_corr, aes(x = reorder(variable, corr_pobre), y = corr_pobre,
                      fill = corr_pobre > 0)) +
  geom_col(width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = COL_SI, "FALSE" = COL_NO),
                    labels = c("Reduce pobreza", "Aumenta pobreza")) +
  labs(
    title    = "Correlacion de cada variable con pobreza",
    subtitle = "Top 15 predictores por correlacion de Pearson",
    x = NULL, y = "Correlacion con pobre", fill = NULL
  ) +
  tema_base +
  theme(legend.position = "bottom")

ggsave(file.path(dir_figures, "08_correlaciones_pobre.png"),
       width = 8, height = 6, dpi = 200)


# ==============================================================================
# GRAFICO 9: MATRIZ DE CORRELACIONES ENTRE PREDICTORES
# ==============================================================================
# Detectar multicolinealidad y relaciones entre los predictores clave
# que usan los modelos finales.

cat("\n== Grafico 9: Matriz de correlaciones ==\n")

vars_matriz <- c(
  "horas_prom", "arriendo_estimado", "tasa_empleo_total",
  "prop_contributivo", "menores_por_ocupado",
  "prop_cotiza_pension", "prop_subsidiado",
  "nivel_educ_prom", "edad_prom", "hacinamiento",
  "cuartos_pc", "prop_asalariado", "prop_informal",
  "n_ocupados", "num_personas", "pobre"
)

mat_corr <- train |>
  select(any_of(vars_matriz)) |>
  cor(use = "pairwise.complete.obs")

png(file.path(dir_figures, "09_matriz_correlaciones.png"),
    width = 1000, height = 900, res = 120, bg = "white")
corrplot(mat_corr,
         method  = "color",
         type    = "lower",
         tl.col  = "black",
         tl.cex  = 0.75,
         addCoef.col = "black",
         number.cex  = 0.55,
         title   = "Correlaciones entre predictores clave",
         mar     = c(0, 0, 2, 0))
dev.off()


# ==============================================================================
# GUARDAR TABLAS EN ARCHIVO DE TEXTO
# ==============================================================================

cat("\n== Guardando tablas en resultados_eda.txt ==\n")

# Tabla de medias por grupo
vars_comparar <- c(
  "num_cuartos", "cuartos_pc", "hacinamiento",
  "num_personas", "n_menores_18", "menores_por_ocupado",
  "edad_jefe", "prop_mujeres", "ratio_depend",
  "nivel_educ_max", "nivel_educ_jefe", "nivel_educ_prom",
  "n_ocupados", "tasa_ocupacion", "tasa_empleo_total", "tasa_desempleo",
  "prop_asalariado", "prop_informal", "prop_contributivo",
  "prop_subsidiado", "prop_cotiza_pension",
  "horas_prom", "antiguedad_prom",
  "alguno_subsidio", "alguno_remesas"
)

tabla_medias <- train |>
  group_by(pobre_f) |>
  summarise(
    across(any_of(vars_comparar), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  ) |>
  pivot_longer(-pobre_f, names_to = "variable", values_to = "media") |>
  pivot_wider(names_from = pobre_f, values_from = media) |>
  mutate(
    diferencia = Pobre - `No pobre`,
    ratio      = round(Pobre / `No pobre`, 2)
  ) |>
  arrange(desc(abs(diferencia)))

sink(file.path(dir_figures, "resultados_eda.txt"))
cat("=== RESUMEN DESCRIPTIVO GENERAL ===\n\n")
cat("Hogares:", nrow(train), "\n")
cat("Predictores:", ncol(train) - 2, "(excluyendo id y pobre)\n")
cat("Prevalencia pobreza:", round(mean(train$pobre) * 100, 1), "%\n")
cat("Ratio No pobre : Pobre =",
    round(tabla_pobre$n[1] / tabla_pobre$n[2], 1), ": 1\n\n")

cat("\n=== POBREZA POR ZONA ===\n\n")
print(zona_pobre)

cat("\n\n=== POBREZA POR DEPARTAMENTO ===\n\n")
print(dpto_pobre)

cat("\n\n=== MEDIAS POR GRUPO (POBRES vs NO POBRES) ===\n\n")
print(tabla_medias, n = 30)

cat("\n\n=== TOP 15 CORRELACIONES CON POBREZA ===\n\n")
print(head(correlaciones, 15))
sink()


# ==============================================================================
# RESUMEN
# ==============================================================================

cat("\n========================================\n")
cat("EDA completo\n")
cat("========================================\n")
cat("Graficos guardados en:", dir_figures, "\n")
cat("  01_desbalance_clases.png\n")
cat("  02_pobreza_zona.png\n")
cat("  03_pobreza_dpto.png\n")
cat("  04_educ_jefe_vs_pobre.png\n")
cat("  05_formalidad_vs_pobre.png\n")
cat("  06_menores_por_ocupado.png\n")
cat("  07_hacinamiento.png\n")
cat("  08_correlaciones_pobre.png\n")
cat("  09_matriz_correlaciones.png\n")
cat("  resultados_eda.txt\n")
