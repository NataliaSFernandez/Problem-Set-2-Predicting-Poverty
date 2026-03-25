#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script 01: Data Cleaning and Merging
#==============================================================================
# OBJETIVO: Limpiar y preparar los datos para el análisis exploratorio y modelado.
#
# INPUTS:  00_data/raw/test_hogares.csv
#          00_data/raw/train_hogares.csv
#          00_data/raw/test_personas.csv
#          00_data/raw/train_personas.csv
#
# OUTPUTS: 00_data/processed/train_final.rds / .csv
#          00_data/processed/test_final.rds  / .csv
#
# METODOLOGÍA:
#    1.  Carga de datos
#    2.  Renombrado legible
#    3.  Diagnóstico de NAs en los 4 datasets crudos
#    4.  Imputación de NAs estructurales en personas (replace_na(0))
#    5.  Recodificación binarias 1=Si/2=No --> 1/0 (moda para 9=NS)
#    6.  Limpieza básica (Inf --> NA, tipos)
#    7.  Agregados personas --> hogar
#    8.  Merge
#    9.  Selección y alineación train/test
#   10.  Guardado
#   11.  Variables adicionales basadas en literatura académica
#        (hacinamiento, IPM, interacciones, formalidad, región, etc.)
#==============================================================================
# nolint start
# -- 0. Paquetes -------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,
  tidyr,
  dplyr,
  readr,
  janitor,
  skimr,
  fs
)

# -- 1. Rutas ----------------------------------------------------------------
dir_raw       <- "00_data/raw"
dir_processed <- "00_data/processed"
dir_create(dir_processed, recurse = TRUE)

# -- 2. Carga de datos -------------------------------------------------------
message("== Cargando datos ==")

train_hog <- read_csv(file.path(dir_raw, "train_hogares.csv"),  show_col_types = FALSE)
test_hog  <- read_csv(file.path(dir_raw, "test_hogares.csv"),   show_col_types = FALSE)
train_per <- read_csv(unz(file.path(dir_raw, "train_personas.zip"),
                          "train_personas/train_personas.csv"), show_col_types = FALSE)
test_per  <- read_csv(file.path(dir_raw, "test_personas.csv"),  show_col_types = FALSE)

message("  train_hogares:  ", nrow(train_hog), " filas | ", ncol(train_hog), " cols")
message("  test_hogares:   ", nrow(test_hog),  " filas | ", ncol(test_hog),  " cols")
message("  train_personas: ", nrow(train_per), " filas | ", ncol(train_per), " cols")
message("  test_personas:  ", nrow(test_per),  " filas | ", ncol(test_per),  " cols")


# -- 3. Renombrado a nombres legibles ----------------------------------------
message("\n== Renombrando variables ==")

renombrar_hogares <- function(df) {
  df |> rename(
    zona              = Clase,          # 1=Cabecera, 2=Resto
    ciudad            = Dominio,
    num_cuartos       = P5000,          # Cuartos totales (sin cocina/banos/garaje)
    num_cuartos_dorm  = P5010,          # Cuartos para dormir
    tipo_tenencia     = P5090,          # 1=propia pagada, 2=pagando, 3=arriendo,
                                        # 4=usufructo, 5=posesion sin titulo, 6=otra
    arriendo_pagado   = P5100,          # Valor arriendo mensual pagado (COP)
    arriendo_estimado = P5130,          # Valor arriendo estimado (COP)
    valor_alojamiento = P5140,          # Valor alojamiento hotel/pension (COP)
    num_personas      = Nper,           # Total personas en el hogar
    num_personas_ug   = Npersug,        # Personas en la unidad de gasto
    linea_indigencia  = Li,             # NO predictor: define outcome
    linea_pobreza     = Lp,             # NO predictor: define outcome
    factor_exp        = Fex_c,          # Factor de expansion nacional
    dpto              = Depto,
    factor_exp_dpto   = Fex_dpto        # Factor de expansion departamental
  )
}

renombrar_personas <- function(df) {
  mapa <- c(
    orden              = "Orden",
    zona               = "Clase",
    ciudad             = "Dominio",
    dpto               = "Depto",
    factor_exp         = "Fex_c",
    factor_exp_dpto    = "Fex_dpto",
    sexo               = "P6020",       # 1=Hombre, 2=Mujer --> recodificar
    edad               = "P6040",
    parentesco         = "P6050",       # 1=Jefe, 2=Conyuge, 3=Hijo...
    afil_salud         = "P6090",       # 1=Si, 2=No, 9=NS
    tipo_afil_salud    = "P6100",       # 1=Contributivo, 2=Especial, 3=Subsidiado
    nivel_educ         = "P6210",       # 1=Ninguno...6=Superior, 9=NS
    ultimo_grado_educ  = "P6210s1",
    actividad_ppal     = "P6240",       # 1=Trabajo, 2=Tiene emp, 3=Busco,
                                        # 4=Estudio, 5=Hogar, 6=Incap.
    ocupacion_ciuo     = "Oficio",      # Codigo CIUO-88
    antiguedad_meses   = "P6426",
    posicion_ocup      = "P6430",       # 1=Emp.privado, 2=Emp.publico,
                                        # 3=Domestico, 4=Cta propia, 5=Patron,
                                        # 6=Fam sin rem, 7=Sin rem no fam,
                                        # 8=Jornalero, 9=Otro
    recibe_horas_ext   = "P6510",       # 1=Si, 2=No, 9=NS
    recibe_primas      = "P6545",       # 1=Si, 2=No, 9=NS
    recibe_bonif       = "P6580",       # 1=Si, 2=No, 9=NS
    sub_especie_alim   = "P6585s1",     # 1=Si, 2=No, 9=NS
    sub_especie_transp = "P6585s2",
    sub_especie_fam    = "P6585s3",
    sub_especie_educ   = "P6585s4",
    especie_alimentos  = "P6590",       # 1=Si, 2=No, 9=NS
    especie_vivienda   = "P6600",
    especie_transporte = "P6610",
    especie_otros      = "P6620",
    recibio_prima_serv = "P6630s1",     # 1=Si, 2=No
    recibio_prima_nav  = "P6630s2",
    recibio_prima_vac  = "P6630s3",
    recibio_viaticos   = "P6630s4",
    recibio_bonif_anu  = "P6630s6",
    horas_trabajadas   = "P6800",
    tamano_empresa     = "P6870",       # 1=Solo...9=101+ personas
    cotiza_pension     = "P6920",       # 1=Si, 2=No, 3=Ya pensionado
    tiene_2do_empleo   = "P7040",       # 1=Si, 2=No
    horas_2do_emp      = "P7045",       # NA>90% --> eliminar en paso 9
    ingreso_2do_emp    = "P7050",       # NA>90% --> eliminar en paso 9
    quiere_mas_horas   = "P7090",       # 1=Si, 2=No
    dilig_mas_horas    = "P7110",       # NA>90% --> eliminar en paso 9
    disp_mas_horas     = "P7120",       # NA>90% --> eliminar en paso 9
    dilig_cambiar      = "P7150",       # 1=Si, 2=No
    puede_nuevo_emp    = "P7160",       # 1=Si, 2=No
    busca_1era_vez     = "P7310",       # NA>90% --> eliminar en paso 9
    posicion_ant       = "P7350",       # NA>90% --> eliminar en paso 9
    recibe_dom_otro_hog = "P7422",      # NA>90% --> eliminar en paso 9
    recibe_agropec      = "P7472",      # 1=Si, 2=No
    recibe_arr_pension  = "P7495",      # 1=Si, 2=No
    recibe_pension_jub  = "P7500s2",    # 1=Si, 2=No, 9=NS
    recibe_pension_alim = "P7500s3",    # 1=Si, 2=No, 9=NS
    recibe_otros_nlab   = "P7505",      # 1=Si, 2=No
    recibe_transf_nac   = "P7510s1",    # 1=Si, 2=No, 9=NS
    recibe_remesas      = "P7510s2",
    recibe_subsidio     = "P7510s3",
    recibe_intereses    = "P7510s5",
    recibe_cesantias    = "P7510s6",
    recibe_otros_ing    = "P7510s7",
    pet                 = "Pet",        # Ya 0/1 segun PDF
    ocupado             = "Oc",         # Ya 0/1
    desempleado         = "Des",        # Ya 0/1
    inactivo            = "Ina",        # Ya 0/1
    estrato            = "Estrato1",    # Estrato socioeconómico (1-6, 9=No aplica)
    recibe_arriendos   = "P7500s1"      # 1=Si, 2=No (ingreso por arrendamientos)
  )
  mapa_valido <- mapa[mapa %in% names(df)]
  rename(df, any_of(mapa_valido))
}

train_hog <- renombrar_hogares(train_hog)
test_hog  <- renombrar_hogares(test_hog)
train_per <- renombrar_personas(train_per)
test_per  <- renombrar_personas(test_per)


# ==========================================================================
# -- 4. DIAGNÓSTICO DE NAs -------------------------------------------------
#  Se corre DESPUÉS del renombrado (nombres legibles) y ANTES de cualquier
#  imputación, para reflejar el estado real del dato crudo.
# ==========================================================================
message("\n== Diagnóstico de NAs (post-renombrado, pre-imputación) ==")

calcular_nas <- function(df, nombre) {
  df |>
    summarise(across(everything(), ~ mean(is.na(.)))) |>
    pivot_longer(everything(), names_to = "variable", values_to = "prop_na") |>
    filter(prop_na > 0) |>
    arrange(desc(prop_na)) |>
    mutate(
      dataset  = nombre,
      pct_na   = scales::percent(prop_na, accuracy = 0.1),
      decision = case_when(
        prop_na >= 0.90 ~ "ELIMINAR (varianza ~0 al agregar)",
        prop_na >= 0.50 ~ "replace_na(0) - NA estructural por diseno",
        prop_na >= 0.10 ~ "recodificar_si_no() imputa con moda",
        TRUE            ~ "sin accion"
      )
    ) |>
    select(dataset, variable, pct_na, decision)
}

nas_train_hog <- calcular_nas(train_hog, "train_hogares")
nas_test_hog  <- calcular_nas(test_hog,  "test_hogares")
nas_train_per <- calcular_nas(train_per, "train_personas")
nas_test_per  <- calcular_nas(test_per,  "test_personas")

cat("\n--- NAs train_hogares ---\n");        print(nas_train_hog, n = Inf)
cat("\n--- NAs test_hogares ---\n");         print(nas_test_hog,  n = Inf)
cat("\n--- NAs train_personas (top 30) ---\n"); print(nas_train_per, n = 30)
cat("\n--- NAs test_personas ---\n");        print(nas_test_per,  n = Inf)

# Resumen agregado por dataset
cat("\n--- Resumen: variables con NA por dataset y decision ---\n")
bind_rows(nas_train_hog, nas_test_hog, nas_train_per, nas_test_per) |>
  count(dataset, decision) |>
  pivot_wider(names_from = decision, values_from = n, values_fill = 0) |>
  print()


# ==========================================================================
# -- 5. IMPUTACIÓN DE NAs ESTRUCTURALES EN PERSONAS -----------------------
#
#  NA estructural: la pregunta NO APLICA a esa persona.
#  Se corre ANTES de recodificar_si_no() para que la moda se calcule
#  sobre datos ya completos.
#
#  Lógica por grupo:
#
#  A) ~55% NA: solo para OCUPADOS
#     Inactivos/desocupados no responden preguntas de empleo.
#     Para ellos la respuesta correcta es 0, no un dato faltante.
#
#  B) ~79-90% NA: solo para ASALARIADOS / FORMALES
#     Preguntas sobre prestaciones y beneficios laborales.
#     Si no es asalariado formal = 0.
#
#  C) ~83-93% NA: ingresos NO LABORALES
#     Solo se preguntan a quienes responden "Sí" a la pregunta filtro.
#     Si no llegaron a la pregunta = no tienen ese ingreso = 0.
#
#  D) >90% NA: casi cero varianza al agregar al hogar --> ELIMINAR
# ==========================================================================
message("\n== Imputando NAs estructurales ==")

# A) Solo para ocupados (~55% NA)
# cotiza_pension se excluye aqui: su NA se maneja en recodificar_personas
# con case_when (TRUE -> "no_cotiza"), no con replace_na(0)
vars_na0_ocupados <- c(
  "ocupacion_ciuo", "antiguedad_meses",
  "posicion_ocup", "horas_trabajadas", "tamano_empresa",
  "tiene_2do_empleo", "quiere_mas_horas", "recibe_agropec","tipo_afil_salud"
)

# B) Solo para asalariados/formales (~79-90% NA)
vars_na0_formales <- c(
  "recibe_horas_ext", "recibe_primas", "recibe_bonif",
  "sub_especie_alim", "sub_especie_transp", "sub_especie_fam", "sub_especie_educ",
  "especie_alimentos", "especie_vivienda", "especie_transporte", "especie_otros",
  "recibio_prima_serv", "recibio_prima_nav", "recibio_prima_vac",
  "recibio_viaticos", "recibio_bonif_anu"
)

# C) Solo para quienes tienen ingresos no laborales (~83-93% NA)
# NOTA: recibe_pension_jub y recibe_pension_alim tienen ~92% NA en test pero
# se conservan porque en train tienen informacion util y el umbral exacto
# depende del dataset. Se imputan con 0 (estructural: no llego a la pregunta).
vars_na0_nolaboral <- c(
  "recibe_pension_jub", "recibe_pension_alim",
  "recibe_transf_nac", "recibe_remesas", "recibe_subsidio",
  "recibe_intereses", "recibe_cesantias", "recibe_otros_ing",
  "recibe_arriendos"   # P7500s1: ingreso por arrendamientos (NA estructural)
)

# D) >90% NA en AMBOS train y test: varianza ~0 al agregar --> ELIMINAR
# desempleado y ocupado NO se incluyen aqui: son flags DANE 0/1 validos,
# su NA indica persona que no completo la encuesta, no pregunta que no aplica.
vars_eliminar_alta_na <- c(
  "horas_2do_emp", "ingreso_2do_emp",
  "dilig_mas_horas", "disp_mas_horas",
  "busca_1era_vez", "posicion_ant",
  "recibe_dom_otro_hog"
)

imputar_na_estructural <- function(df) {
  df |> mutate(across(
    any_of(c(vars_na0_ocupados, vars_na0_formales, vars_na0_nolaboral)),
    ~ replace_na(., 0)
  ))
}

train_per <- imputar_na_estructural(train_per)
test_per  <- imputar_na_estructural(test_per)

# Imputacion estructural en HOGARES:
# arriendo_pagado (96% NA):   solo aplica a arrendatarios (~30%). Duenos = 0.
# valor_alojamiento (61% NA): solo aplica a hoteles/pensiones. Resto = 0.
# arriendo_estimado (39% NA): valor imputado para propietarios. Arrendatarios = 0.
# Los tres son estructurales: NA no es ignorancia, es "no aplica al tipo de tenencia".
train_hog <- train_hog |>
  mutate(across(any_of(c("arriendo_pagado", "valor_alojamiento", "arriendo_estimado")),
                ~ replace_na(., 0)))
test_hog <- test_hog |>
  mutate(across(any_of(c("arriendo_pagado", "valor_alojamiento", "arriendo_estimado")),
                ~ replace_na(., 0)))

message("  Imputados con 0: ",
        length(c(vars_na0_ocupados, vars_na0_formales, vars_na0_nolaboral)),
        " variables")
message("  Marcadas para eliminar (NA>90%): ", length(vars_eliminar_alta_na))


# -- 6. Recodificación de variables binarias ---------------------------------
#  Convención original DANE: 1=Si | 2=No | 9=NS/NI
#  Nueva convención:         1=Si | 0=No | moda (para 9=NS)
#
#  Casos especiales:
#  - sexo (P6020):           1=Hombre/2=Mujer --> mujer: 1=Mujer / 0=Hombre
#  - cotiza_pension (P6920): 1=Si/2=No/3=
#  - Pet, Oc, Des, Ina:      ya son 1/0 segun PDF, no tocar
message("\n== Recodificando variables binarias ==")

recodificar_si_no <- function(x) {
  x_bin <- case_when(
    x == 1 ~ 1L,
    x == 2 ~ 0L,
    TRUE   ~ NA_integer_
  )
  moda <- as.integer(names(which.max(table(x_bin, useNA = "no"))))
  if_else(is.na(x_bin), moda, x_bin)
}

recodificar_personas <- function(df) {
  df |> mutate(

    # Sexo: nueva columna 0/1 (sexo original se elimina en paso 9)
    mujer              = if_else(sexo == 2, 1L, if_else(sexo == 1, 0L, NA_integer_)),

    # Salud (~18% NA --> moda)
    afil_salud         = recodificar_si_no(afil_salud),

    # Ingresos complementarios (ya con 0 por imputacion estructural)
    recibe_horas_ext   = recodificar_si_no(recibe_horas_ext),
    recibe_primas      = recodificar_si_no(recibe_primas),
    recibe_bonif       = recodificar_si_no(recibe_bonif),

    # Subsidios en especie
    sub_especie_alim   = recodificar_si_no(sub_especie_alim),
    sub_especie_transp = recodificar_si_no(sub_especie_transp),
    sub_especie_fam    = recodificar_si_no(sub_especie_fam),
    sub_especie_educ   = recodificar_si_no(sub_especie_educ),

    # Pago en especie
    especie_alimentos  = recodificar_si_no(especie_alimentos),
    especie_vivienda   = recodificar_si_no(especie_vivienda),
    especie_transporte = recodificar_si_no(especie_transporte),
    especie_otros      = recodificar_si_no(especie_otros),

    # Prestaciones anuales
    recibio_prima_serv = recodificar_si_no(recibio_prima_serv),
    recibio_prima_nav  = recodificar_si_no(recibio_prima_nav),
    recibio_prima_vac  = recodificar_si_no(recibio_prima_vac),
    recibio_viaticos   = recodificar_si_no(recibio_viaticos),
    recibio_bonif_anu  = recodificar_si_no(recibio_bonif_anu),

    # Cotizacion pension: se conserva como factor de 3 niveles (NO binaria)
    # 1=Cotiza activamente (formal estable)
    # 2=No cotiza          (informal o desocupado)
    # 3=Ya pensionado      (ingreso garantizado -- perfil opuesto al 2)
    # Colapsarlos en 0/1 perderia informacion: pensionado != informal
    cotiza_pension = case_when(
      cotiza_pension == 1 ~ "cotiza",
      cotiza_pension == 2 ~ "no_cotiza",
      cotiza_pension == 3 ~ "pensionado",
      TRUE                ~ "no_cotiza"    # NS/NI -> conservador
    ),

    # Segundo empleo
    tiene_2do_empleo   = recodificar_si_no(tiene_2do_empleo),

    # Subempleo
    quiere_mas_horas   = recodificar_si_no(quiere_mas_horas),
    dilig_cambiar      = recodificar_si_no(dilig_cambiar),
    puede_nuevo_emp    = recodificar_si_no(puede_nuevo_emp),

    # Otras fuentes
    recibe_agropec      = recodificar_si_no(recibe_agropec),
    recibe_arr_pension  = recodificar_si_no(recibe_arr_pension),

    # Ingresos no laborales (ya con 0 por imputacion estructural)
    recibe_pension_jub  = recodificar_si_no(recibe_pension_jub),
    recibe_pension_alim = recodificar_si_no(recibe_pension_alim),
    recibe_otros_nlab   = recodificar_si_no(recibe_otros_nlab),

    # Transferencias
    recibe_transf_nac   = recodificar_si_no(recibe_transf_nac),
    recibe_remesas      = recodificar_si_no(recibe_remesas),
    recibe_subsidio     = recodificar_si_no(recibe_subsidio),
    recibe_intereses    = recodificar_si_no(recibe_intereses),
    recibe_cesantias    = recodificar_si_no(recibe_cesantias),
    recibe_otros_ing    = recodificar_si_no(recibe_otros_ing)

    # Pet, Oc, Des, Ina: ya son 1/0, no se tocan
  ) |>
  # recibe_arriendos puede no existir en test_per (P7500s1 ausente)
  mutate(across(any_of("recibe_arriendos"), recodificar_si_no))
}

train_per <- recodificar_personas(train_per)
test_per  <- recodificar_personas(test_per)


# -- 7. Limpieza básica ------------------------------------------------------
message("\n== Limpieza básica ==")

limpiar_tipos <- function(df) {
  df |>
    mutate(across(where(is.numeric), ~ if_else(is.infinite(.), NA_real_, .))) |>
    mutate(across(any_of(c("dpto", "ciudad")), factor))
}

train_hog <- limpiar_tipos(train_hog)
test_hog  <- limpiar_tipos(test_hog)
train_per <- limpiar_tipos(train_per)
test_per  <- limpiar_tipos(test_per)


# -- 8. Agregados personas --> hogar -----------------------------------------
#  Las vars_eliminar_alta_na no se incluyen en summarise (no se agregan).
message("\n== Agregando personas al nivel hogar ==")

agregar_personas <- function(df) {
  # Columnas presentes en train_per pero ausentes en test_per
  if (!"estrato"          %in% names(df)) df <- mutate(df, estrato          = NA_real_)
  if (!"recibe_arriendos" %in% names(df)) df <- mutate(df, recibe_arriendos = NA_real_)
  df |>
    group_by(id) |>
    summarise(

      # -- Demografía
      n_menores_18        = sum(edad < 18,  na.rm = TRUE),
      n_adultos_may       = sum(edad >= 65, na.rm = TRUE),
      edad_prom           = mean(edad, na.rm = TRUE),
      edad_jefe           = first(edad[parentesco == 1]),
      prop_mujeres        = mean(mujer, na.rm = TRUE),
      ratio_depend        = (sum(edad < 15, na.rm = TRUE) +
                             sum(edad >= 65, na.rm = TRUE)) /
                            pmax(sum(edad >= 15 & edad < 65, na.rm = TRUE), 1),

      # -- Educación
      nivel_educ_max      = suppressWarnings(max(nivel_educ, na.rm = TRUE)),
      nivel_educ_jefe     = first(nivel_educ[parentesco == 1]),
      nivel_educ_prom     = mean(nivel_educ, na.rm = TRUE),

      # -- Mercado laboral (Pet/Oc/Des/Ina ya son 0/1)
      n_ocupados          = sum(ocupado     == 1, na.rm = TRUE),
      n_desempleados      = sum(desempleado == 1, na.rm = TRUE),
      n_inactivos         = sum(inactivo    == 1, na.rm = TRUE),
      tasa_ocupacion      = sum(ocupado == 1, na.rm = TRUE) /
                            pmax(sum(pet == 1, na.rm = TRUE), 1),
      tasa_desempleo      = sum(desempleado == 1, na.rm = TRUE) /
                            pmax(sum(ocupado == 1,     na.rm = TRUE) +
                                 sum(desempleado == 1, na.rm = TRUE), 1),

      # -- Posición ocupacional
      prop_asalariado     = mean(posicion_ocup %in% c(1, 2),       na.rm = TRUE),
      prop_cta_propia     = mean(posicion_ocup == 4,                na.rm = TRUE),
      prop_informal       = mean(posicion_ocup %in% c(4, 6, 7, 8), na.rm = TRUE),

      # -- Salud
      prop_afil_salud     = mean(afil_salud == 1,      na.rm = TRUE),
      prop_subsidiado     = mean(tipo_afil_salud == 3, na.rm = TRUE),

      # -- Seguridad social y prestaciones
      # cotiza_pension es factor: "cotiza" / "no_cotiza" / "pensionado"
      prop_cotiza_pension = mean(cotiza_pension == "cotiza",    na.rm = TRUE),
      prop_pensionado     = mean(cotiza_pension == "pensionado", na.rm = TRUE),
      prop_prima_serv     = mean(recibio_prima_serv == 1, na.rm = TRUE),

      # -- Subsidios en especie (proxy formalidad)
      prop_sub_alim       = mean(sub_especie_alim   == 1, na.rm = TRUE),
      prop_sub_transp     = mean(sub_especie_transp == 1, na.rm = TRUE),

      # -- Subempleo
      prop_subempleo      = mean(quiere_mas_horas == 1, na.rm = TRUE),

      # -- Segundo empleo
      prop_2do_empleo     = mean(tiene_2do_empleo == 1, na.rm = TRUE),

      # -- Ingresos no laborales (presencia en el hogar)
      alguno_pension_jub  = as.integer(any(recibe_pension_jub == 1, na.rm = TRUE)),
      alguno_remesas      = as.integer(any(recibe_remesas     == 1, na.rm = TRUE)),
      alguno_subsidio     = as.integer(any(recibe_subsidio    == 1, na.rm = TRUE)),
      alguno_transf_nac   = as.integer(any(recibe_transf_nac  == 1, na.rm = TRUE)),
      alguno_intereses    = as.integer(any(recibe_intereses   == 1, na.rm = TRUE)),
      alguno_cesantias    = as.integer(any(recibe_cesantias   == 1, na.rm = TRUE)),
      alguno_agropec      = as.integer(any(recibe_agropec    == 1, na.rm = TRUE)),

      # -- Horas trabajadas
      horas_prom          = mean(horas_trabajadas, na.rm = TRUE),

      # -----------------------------------------------------------------------
      # Variables adicionales basadas en literatura académica
      # -----------------------------------------------------------------------

      # -- Estrato socioeconómico del hogar (moda; 9=No aplica --> NA)
      # Referencia: DANE IPM Colombia; Profamilia (2015)
      # El estrato determina tarifas de servicios públicos y acceso a subsidios.
      # Se incluye como control a pesar de sus posibles errores.
      estrato_hog      = {
        vals <- estrato[!is.na(estrato) & estrato != 9]
        if (length(vals) == 0) NA_real_
        else as.numeric(names(which.max(table(vals))))
      },
      estrato_bajo     = as.integer({
        vals <- estrato[!is.na(estrato) & estrato != 9]
        if (length(vals) == 0) NA_integer_
        else as.integer(as.numeric(names(which.max(table(vals)))) <= 2)
      }),
      
      # -- Sexo del jefe de hogar
      # Referencia: Buvinic & Gupta (1997); Medeiros & Costa (2008)
      # Los hogares con jefa mujer enfrentan mayor vulnerabilidad a la pobreza
      # por brechas salariales de género y mayor carga de trabajo de cuidado.
      mujer_jefa       = as.integer(any(parentesco == 1 & mujer == 1, na.rm = TRUE)),

      # Hogar monoparental: jefa mujer con al menos un menor de 18
      # Referencia: Buvinic & Gupta (1997) — alta vulnerabilidad estructural
      hogar_monoparental   = as.integer(mujer_jefa == 1 & n_menores_18 > 0),

      # -- Posiciones ocupacionales adicionales
      # Referencia: ILO (2018) "Women and men in the informal economy"
      # El trabajo doméstico, jornalero y familiar sin remuneración son las
      # categorías laborales más precarias e informales en Colombia.
      prop_patron      = mean(posicion_ocup == 5, na.rm = TRUE),  # empleador/patrón
      prop_domestico   = mean(posicion_ocup == 3, na.rm = TRUE),  # trabajador doméstico
      prop_fam_sin_rem = mean(posicion_ocup == 6, na.rm = TRUE),  # familiar sin remuneración
      prop_jornalero   = mean(posicion_ocup == 8, na.rm = TRUE),  # jornalero/peón (rural)
    

      # -- Tamaño de empresa (proxy de formalidad)
      # Referencia: Levy (2008) "Good intentions, bad outcomes"
      # Las microempresas (<5 trabajadores) concentran el empleo informal en
      # América Latina y pagan salarios sistemáticamente más bajos (es proxy de informalidad laboral y precariedad salarial).
      prop_microempresa = mean(tamano_empresa <= 2, na.rm = TRUE),  # 1-3 personas
      prop_gran_empresa = mean(tamano_empresa >= 6, na.rm = TRUE),  # 20+ personas

      # -- Antigüedad laboral (estabilidad del empleo)
      # Referencia: Topel (1991) "Specific capital, mobility, and wages"
      # Mayor antigüedad refleja estabilidad laboral y salario acumulado.
      # Baja antigüedad indica inestabilidad o empleo reciente precario.
      antiguedad_prom  = mean(antiguedad_meses, na.rm = TRUE),
      antiguedad_alta  = as.integer(any(antiguedad_meses > 24, na.rm = TRUE)),  # >2 años

      # -- Ingreso por arrendamientos (posesión de activos)
      # Referencia: Ellis (2000) "Rural Livelihoods and Diversity"
      # El ingreso por arrendamientos refleja posesión de activos.
      alguno_arriendos = as.integer(any(recibe_arriendos == 1, na.rm = TRUE)),

      .groups = "drop"
    ) |>
    mutate(across(where(is.numeric), ~ if_else(is.infinite(.), NA_real_, .)))
}

vars_per_train <- agregar_personas(train_per)
vars_per_test  <- agregar_personas(test_per)

message("  Variables agregadas desde personas: ", ncol(vars_per_train) - 1)


# -- 9. Merge ----------------------------------------------------------------
message("\n== Merge hogar + personas ==")

train_full <- left_join(train_hog, vars_per_train, by = "id")
test_full  <- left_join(test_hog,  vars_per_test,  by = "id")

sin_match_train <- sum(is.na(train_full$n_ocupados))
sin_match_test  <- sum(is.na(test_full$n_ocupados))
if (sin_match_train > 0) message("  Sin match en train: ", sin_match_train, " hogares")
if (sin_match_test  > 0) message("  Sin match en test:  ", sin_match_test,  " hogares")

message("  train_full: ", nrow(train_full), " x ", ncol(train_full))
message("  test_full:  ", nrow(test_full),  " x ", ncol(test_full))

train_full <- train_full |> rename(pobre = Pobre)


# -- 9b. Variables adicionales basadas en literatura académica ---------------
#  Se aplican DESPUÉS del merge, sobre el dataset completo hogar+personas.
#  Dependen de las variables ya disponibles en train_full/test_full,
#  incluyendo las nuevas agregaciones del paso 8 (estrato_hog, mujer_jefa,
#  prop_patron, prop_domestico, prop_fam_sin_rem, prop_jornalero,
#  prop_microempresa, prop_gran_empresa, antiguedad_prom, antiguedad_alta,
#  alguno_arriendos).
#
#  Organización en 10 bloques temáticos:
#    1.  Calidad de vivienda y hacinamiento
#    2.  Composición demográfica y carga económica
#    3.  Capital humano (efectos no lineales y umbrales)
#    4.  Calidad del empleo e índice de formalidad
#    5.  Diversificación y resiliencia de ingresos
#    6.  Protección social formal e informal
#    7.  Clasificación regional
#    8.  Términos de interacción
#    9.  Transformaciones no lineales
#   10.  Índice de privaciones múltiples (inspirado en el IPM)
# Referencia general: Alkire & Foster (2011); DANE IPM Colombia (2022)
message("\n== Construyendo variables adicionales basadas en literatura ==")

construir_nuevas_variables <- function(df) {
  df |>
    mutate(

      # ========================================================================
      # BLOQUE 1: CALIDAD DE VIVIENDA Y HACINAMIENTO
      # Referencia: Ferreira & Ravallion (2008); DANE IPM Colombia (2022)
      # El hacinamiento es uno de los indicadores más robustos de privación
      # material. El IPM Colombia define hacinamiento crítico como >3 personas
      # por cuarto. La inseguridad en la tenencia es un factor de vulnerabilidad
      # patrimonial que impide la acumulación de activos.
      # ========================================================================

      # Personas por cuarto total (hacinamiento general)
      hacinamiento         = num_personas / pmax(num_cuartos, 1),

      # Personas por cuarto para dormir (hacinamiento nocturno — más directo)
      hacinamiento_dorm    = num_personas / pmax(num_cuartos_dorm, 1),

      # Cuartos per cápita (proxy positivo de espacio y calidad de vivienda)
      cuartos_pc           = num_cuartos / pmax(num_personas, 1),

      # Tenencia insegura: arriendo, usufructo, posesión sin título u otra
      tenencia_insegura    = as.integer(tipo_tenencia %in% c(3, 4, 5, 6)),

      # Propiedad totalmente pagada (máxima estabilidad patrimonial)
      vivienda_propia_pag  = as.integer(tipo_tenencia == 1),

      # ========================================================================
      # BLOQUE 2: COMPOSICIÓN DEMOGRÁFICA Y CARGA ECONÓMICA
      # Referencia: Lanjouw & Ravallion (1995); Buvinic & Gupta (1997)
      # El número de dependientes por trabajador determina directamente el
      # ingreso per cápita disponible. Los hogares con muchos menores y pocos
      # ocupados son estructuralmente vulnerables a la pobreza.
      # ========================================================================

      # Proporción del hogar que son menores de 18 años
      prop_menores         = n_menores_18 / pmax(num_personas, 1),

      # Proporción del hogar que son adultos mayores (>=65)
      prop_adultos_may_p   = n_adultos_may / pmax(num_personas, 1),

      # Menores por ocupado: cuántos niños financia cada trabajador
      menores_por_ocupado  = n_menores_18 / pmax(n_ocupados, 1),

      # Hogar sin ningún ocupado (ingreso laboral nulo — vulnerabilidad extrema)
      sin_ocupados         = as.integer(n_ocupados == 0),

      # Hogar con un único ocupado (riesgo de ingreso único — alta vulnerabilidad)
      un_solo_ocupado      = as.integer(n_ocupados == 1),

      # Hogar con dos o más ocupados (diversificación de ingreso laboral)
      doble_ingreso        = as.integer(n_ocupados >= 2),

      # ========================================================================
      # BLOQUE 3: CAPITAL HUMANO — EFECTOS NO LINEALES Y UMBRALES
      # Referencia: Mincer (1974); Card (1999) "The causal effect of education
      # on earnings". Los retornos a la educación no son lineales: existen
      # "sheepskin effects" (saltos discontinuos) en bachillerato y universidad.
      # El IPM Colombia define privación educativa del jefe como <= primaria.
      # ========================================================================

      # Término cuadrático de educación del jefe (retornos crecientes)
      educ_jefe_sq         = nivel_educ_jefe^2,

      # Término cuadrático de educación promedio del hogar
      educ_prom_sq         = nivel_educ_prom^2,

      # Jefe con educación media o superior (bachillerato completo o más)
      educ_media_o_mas     = as.integer(nivel_educ_jefe >= 5),

      # Jefe con educación superior (técnico, tecnológico o universitario)
      educ_superior_jefe   = as.integer(nivel_educ_jefe >= 6),

      # Brecha educativa intergeneracional: ¿alguien del hogar superó al jefe?
      # Un valor positivo indica movilidad educativa ascendente intra-hogar.
      brecha_educ          = nivel_educ_max - nivel_educ_jefe,

      # ========================================================================
      # BLOQUE 4: CALIDAD DEL EMPLEO E ÍNDICE DE FORMALIDAD
      # Referencia: Fields (2011); ILO (2018); Levy (2008)
      # La formalidad laboral es el principal canal de protección contra la
      # pobreza en América Latina. Combina: contrato salarial, cotización a
      # pensión, prima de servicios y acceso a subsidios de transporte.
      # ========================================================================

      # Índice compuesto de formalidad: promedio de 4 indicadores clave
      indice_formalidad    = rowMeans(
        cbind(prop_asalariado, prop_cotiza_pension,
              prop_prima_serv, prop_sub_transp),
        na.rm = TRUE
      ),

      # Formalidad estricta: salarial Y cotizando pensión simultáneamente
      formal_estricto      = prop_asalariado * prop_cotiza_pension,

      # Vulnerabilidad laboral extrema: doméstico + familiar sin rem + jornalero
      vulnerabilidad_lab   = prop_domestico + prop_fam_sin_rem + prop_jornalero,

      # Tasa de empleo sobre total del hogar (no solo sobre PET)
      tasa_empleo_total    = n_ocupados / pmax(num_personas, 1),

      # ========================================================================
      # BLOQUE 5: DIVERSIFICACIÓN Y RESILIENCIA DE INGRESOS
      # Referencia: Ellis (2000); McKenzie & Rapoport (2011); Morduch (1995)
      # "Income smoothing and consumption smoothing". Los hogares con múltiples
      # fuentes de ingreso son más resilientes a choques transitorios de pobreza.
      # ========================================================================

      # Número total de fuentes de ingreso no laboral presentes en el hogar
      n_fuentes_no_lab     = alguno_pension_jub + alguno_remesas   +
                             alguno_subsidio    + alguno_transf_nac +
                             alguno_intereses   + alguno_cesantias  +
                             alguno_arriendos   + alguno_agropec,

      # Tiene al menos una fuente de ingreso no laboral
      tiene_ingreso_no_lab = as.integer(
        alguno_pension_jub == 1 | alguno_remesas    == 1 |
        alguno_subsidio    == 1 | alguno_transf_nac == 1 |
        alguno_intereses   == 1 | alguno_cesantias  == 1 |
        alguno_arriendos   == 1 | alguno_agropec    == 1
      ),

      # Depende exclusivamente de subsidios estatales (fuente más precaria)
      solo_subsidio        = as.integer(
        alguno_subsidio    == 1 &
        alguno_pension_jub == 0 &
        alguno_remesas     == 0 &
        alguno_intereses   == 0 &
        alguno_arriendos   == 0
      ),

      # Tiene ingresos de activos: intereses o arrendamientos (capital acumulado)
      ingresos_activos     = as.integer(
        alguno_intereses == 1 | alguno_arriendos == 1
      ),

      # ========================================================================
      # BLOQUE 6: PROTECCIÓN SOCIAL FORMAL E INFORMAL
      # Referencia: Holzmann & Jørgensen (2001) "Social Risk Management";
      # Barrientos (2011) "Social Protection and Poverty".
      # La pensión contributiva y el régimen subsidiado tienen efectos muy
      # distintos: la primera protege contra la pobreza, la segunda la señala.
      # ========================================================================

      # Protección social amplia: cotiza a pensión O ya recibe jubilación
      proteccion_social    = as.integer(
        alguno_pension_jub == 1 | prop_cotiza_pension > 0
      ),

      # Desprotegido: alta informalidad, sin cotización ni jubilación previa
      desprotegido         = as.integer(
        prop_cotiza_pension == 0 &
        alguno_pension_jub  == 0 &
        prop_informal > 0.5
      ),

      # Proporción en salud contributiva (no subsidiada): mejor calidad de afil.
      prop_contributivo    = prop_afil_salud - prop_subsidiado,

      # ========================================================================
      # BLOQUE 7: CLASIFICACIÓN REGIONAL
      # Referencia: Galvis & Meisel (2010); Bonet (2006) "Desequilibrios
      # regionales en Colombia". Las regiones costeras y pacífica presentan
      # pobreza estructuralmente más alta que el interior, con mecanismos
      # causales distintos (exclusión histórica, menor capital humano, etc.).
      # ========================================================================

      # Costa Caribe: históricamente las tasas de pobreza más altas de Colombia
      costa_caribe         = as.integer(
        dpto %in% c("08", "13", "20", "23", "44", "47", "70")
      ),

      # Región Pacífica (excluye Cali — Valle del Cauca)
      region_pacifico      = as.integer(dpto %in% c("19", "27", "52")),

      # Bogotá D.C.: menor pobreza por concentración de empleo formal
      bogota               = as.integer(dpto == "11"),

      # Eje Cafetero y Valle del Cauca: economías diversificadas y más formales
      eje_cafetero         = as.integer(dpto %in% c("17", "63", "66", "76")),

      # Intersección rural x Costa Caribe: pobreza rural costera estructural
      rural_costa_caribe   = as.integer(
        zona == 2 & dpto %in% c("08", "13", "20", "23", "44", "47", "70")
      ),

      # ========================================================================
      # BLOQUE 8: TÉRMINOS DE INTERACCIÓN
      # Referencia: Attanasio & Székely (2001); Bourguignon & Chakravarty (2003)
      # "The measurement of multidimensional poverty". Las interacciones capturan
      # la complementariedad entre dimensiones: la pobreza es peor cuando
      # múltiples factores de riesgo se combinan simultáneamente.
      # ========================================================================

      # Educación x Formalidad: hogar con capital humano Y empleo formal
      # El más protegido: Mincer (1974) + Fields (2011)
      educ_x_formal        = nivel_educ_jefe * prop_cotiza_pension,

      # Educación x Tasa de ocupación: capital humano convertido efectivamente
      # en empleo (complementariedad entre oferta y demanda laboral)
      educ_x_ocup          = nivel_educ_jefe * tasa_ocupacion,

      # Menores x Desocupación: carga de menores sin ingreso laboral
      # Captura la pobreza infantil dinámica (Lanjouw & Ravallion 1995)
      menores_x_desocup    = n_menores_18 * (1 - tasa_ocupacion),

      # Dependencia x Informalidad: doble penalización estructural
      # (muchos dependientes + trabajo precario sin seguridad de ingresos)
      depend_x_informal    = ratio_depend * prop_informal,

      # Rural x Cuenta propia: autoempleo rural distinto del urbano
      # En zonas rurales puede reflejar subsistencia más que elección
      rural_x_cta_propia   = as.numeric(zona == 2) * prop_cta_propia,

      # Educación x Rural: menores retornos a la educación en zona rural
      # por menor oferta de empleos cualificados fuera de ciudades
      educ_x_rural         = nivel_educ_jefe * as.numeric(zona == 2),

      # Subsidiado x Menores: salud subsidiada Y muchos menores
      # La combinación más fuerte de señales de pobreza infantil en el hogar
      subsidiado_x_menores = prop_subsidiado * n_menores_18,

      # Jefe mayor con jubilación: adulto mayor con pensión (ciclo de vida
      # avanzado con protección formal — habitualmente no pobre)
      jefe_mayor_pension   = as.numeric(edad_jefe > 55) * alguno_pension_jub,

      # Estrato x Zona: el estrato tiene distinto significado en lo urbano
      # y rural (en rural el estrato 1 es casi universal)
      estrato_x_zona       = estrato_hog * as.numeric(zona == 2),

      # ========================================================================
      # BLOQUE 9: TRANSFORMACIONES NO LINEALES
      # Referencia: Manning (2002) "The logged dependent variable"; Deaton
      # (1997) "The Analysis of Household Surveys". Las transformaciones log
      # y cuadráticas capturan relaciones no lineales y reducen el peso de
      # outliers extremos en variables de conteo y tasas.
      # ========================================================================

      # Logaritmo del número de personas (impacto marginal decreciente del tamaño)
      log_num_personas     = log1p(num_personas),

      # Logaritmo de menores de 18 (mismo argumento que el anterior)
      log_n_menores        = log1p(n_menores_18),

      # Cuadrado del ratio de dependencia (relación convexa con pobreza)
      ratio_depend_sq      = ratio_depend^2,

      # Cuadrado de la tasa de ocupación (relación convexa con bienestar)
      tasa_ocup_sq         = tasa_ocupacion^2,

      # Cuadrado del hacinamiento (penalización adicional en hacinamiento extremo)
      hacinamiento_sq      = (num_personas / pmax(num_cuartos, 1))^2,

      # Cuadrado de la edad del jefe (ciclo de vida en U invertida)
      edad_jefe_sq         = edad_jefe^2,

      # ========================================================================
      # BLOQUE 10: ÍNDICE DE PRIVACIONES MÚLTIPLES (inspirado en el IPM)
      # Referencia: Alkire & Foster (2011); DANE IPM Colombia (2022).
      # El enfoque de conteo de Alkire-Foster identifica como "pobre
      # multidimensional" al hogar que acumula privaciones en al menos
      # k dimensiones. Se usan 5 dimensiones con umbral k=2.
      # ========================================================================

      # Privación 1 — Educación: jefe del hogar con máximo básica primaria
      dep_educacion        = as.integer(nivel_educ_jefe <= 3),

      # Privación 2 — Vivienda: hacinamiento crítico (>3 personas por cuarto)
      dep_vivienda         = as.integer(
        num_personas / pmax(num_cuartos, 1) > 3
      ),

      # Privación 3 — Empleo formal: algún ocupado pero ninguno cotiza ni
      # es asalariado (todos en informalidad total)
      dep_empleo_formal    = as.integer(
        prop_asalariado == 0 & prop_cotiza_pension == 0 & n_ocupados > 0
      ),

      # Privación 4 — Protección social: salud mayoritariamente subsidiada
      # y nadie recibe jubilación o pensión contributiva
      dep_proteccion       = as.integer(
        prop_subsidiado > 0.5 & alguno_pension_jub == 0
      ),

      # Privación 5 — Carga demográfica: más dependientes que personas en
      # edad de trabajar (ratio de dependencia > 1)
      dep_dependencia      = as.integer(ratio_depend > 1)

    ) |>
    mutate(
      # Número total de privaciones acumuladas (0 a 5)
      n_privaciones        = dep_educacion     + dep_vivienda      +
                             dep_empleo_formal + dep_proteccion    +
                             dep_dependencia,

      # Hogar multiprivado: >=2 privaciones simultáneas (umbral Alkire-Foster)
      multi_privado        = as.integer(n_privaciones >= 2),

      # Pobreza multidimensional severa: >=3 privaciones simultáneas
      privado_severo       = as.integer(n_privaciones >= 3)
    )
}

train_full <- construir_nuevas_variables(train_full)
test_full  <- construir_nuevas_variables(test_full)

message("  Variables adicionales construidas: OK")


# -- 10. Selección y alineación de columnas ----------------------------------
message("\n== Selección de columnas ==")

vars_excluir_ambos <- c(
  "linea_pobreza", "linea_indigencia",   # data leakage: definen el outcome
  "factor_exp", "factor_exp_dpto",       # ponderadores muestrales
  "sexo"                                 # reemplazada por 'mujer' (0/1)
)

vars_excluir_solo_train <- c(
  "Ingtotug", "Ingtotugarr", "Ingpcug",  # ingreso total = base para calcular Pobre
  "Indigente", "Npobres", "Nindigentes", # derivadas del outcome
  vars_eliminar_alta_na                  # NA>90% en personas: varianza ~0
)

train_model <- train_full |>
  select(-any_of(vars_excluir_ambos), -any_of(vars_excluir_solo_train))

test_model  <- test_full  |>
  select(-any_of(vars_excluir_ambos), -any_of("pobre"))

# Alinear: solo columnas comunes (pobre se excluye del chequeo)
cols_train    <- setdiff(names(train_model), "pobre")
solo_en_train <- setdiff(cols_train, names(test_model))
solo_en_test  <- setdiff(names(test_model), cols_train)

if (length(solo_en_train) > 0) {
  message("  Aún en train pero no en test (se eliminan): ",
          paste(solo_en_train, collapse = ", "))
  train_model <- select(train_model, -any_of(solo_en_train))
}
if (length(solo_en_test) > 0) {
  message("  En test pero no en train (se eliminan): ",
          paste(solo_en_test, collapse = ", "))
  test_model <- select(test_model, -any_of(solo_en_test))
}

cols_finales <- setdiff(names(train_model), "pobre")
test_model   <- select(test_model, all_of(cols_finales))

message("  Predictores finales: ", length(cols_finales))
cat("\n-- Lista de predictores --\n")
print(cols_finales)

stopifnot("pobre debe estar en train_model" = "pobre" %in% names(train_model))
message("  OK: 'pobre' presente en train_model")

# -- 11. Guardado ------------------------------------------------------------
message("\n== Guardando ==")

saveRDS(train_model, file.path(dir_processed, "train_final.rds"))
saveRDS(test_model,  file.path(dir_processed, "test_final.rds"))

# CSV locales (no se suben a GitHub)
write_csv(train_model, file.path(dir_processed, "train_final.csv"))
write_csv(test_model,  file.path(dir_processed, "test_final.csv"))

n_pobres    <- sum(as.character(train_model$pobre) == "Pobre",    na.rm = TRUE)
n_no_pobres <- sum(as.character(train_model$pobre) == "No_pobre", na.rm = TRUE)
prevalencia <- round(n_pobres / (n_pobres + n_no_pobres) * 100, 1)
message("  train_final: ", nrow(train_model), " x ", ncol(train_model),
        "  (", n_pobres, " pobres / ", n_no_pobres, " no pobres | ",
        prevalencia, "% prevalencia)")
message("  test_final:  ", nrow(test_model), " x ", ncol(test_model))

# -- Uso en scripts posteriores ---------------------------------------------
# train <- readRDS("00_data/processed/train_final.rds")
# test  <- readRDS("00_data/processed/test_final.rds")
# nolint end

