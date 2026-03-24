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
#==============================================================================

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
    inactivo            = "Ina"         # Ya 0/1
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
  "recibe_intereses", "recibe_cesantias", "recibe_otros_ing"
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
  )
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

      # -- Horas trabajadas
      horas_prom          = mean(horas_trabajadas, na.rm = TRUE),

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
