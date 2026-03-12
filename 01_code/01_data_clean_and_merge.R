
#==============================================================================
#PROBLEM SET 2: PREDICTING POVERTY
#Script 01: Data Cleaning and Merging
#==============================================================================
#OBJETIVO: Limpiar y preparar los datos para el análisis exploratorio y modelado predictivo.

#INPUTS:  00_data/raw/test_hogares.csv
#         00_data/raw/train_hogares.csv
#         00_data/raw/test_personas.csv
#         00_data/raw/train_personas.csv
#OUTPUTS: 
#  - 

#METODOLOGÍA:
#    1. Carga de datos
#    2. Renombrado legible 
#    3. Recodificacion binarias 1=Si/2=No --> 1/0 (moda para 9=NS)
#    4. Limpieza basica
#    5. Agregados personas --> hogar
#    6. Merge
#    7. Variable objetivo (solo train)

#
#  Output: data/processed/train_final.rds
#          data/processed/test_final.rds
   
#==============================================================================

# ==============================================================================
# PASO 1: CARGAR DATOS CRUDOS
# ==============================================================================
# -- 0. Paquetes ------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,   # manipulacion de datos
  janitor,     # clean_names, tabyl
  readr,
  skimr,       # resumen rapido de dataframes
  fs           # manejo de rutas
)

# -- 1. Rutas ---------------------------------------------------------------
dir_raw       <- "00_data/raw"
dir_processed <- "00_data/processed"
dir_create(dir_processed, recurse = TRUE)

# -- 2. Carga de datos ------------------------------------------------------
message("== Cargando datos ==")

train_hog <- read_csv(file.path(dir_raw, "train_hogares.csv"),  show_col_types = FALSE)
test_hog  <- read_csv(file.path(dir_raw, "test_hogares.csv"),   show_col_types = FALSE)
train_per <- read_csv(unz(file.path(dir_raw, "train_personas.zip"),"train_personas/train_personas.csv"))
test_per  <- read_csv(file.path(dir_raw, "test_personas.csv"),  show_col_types = FALSE)

message("  train_hogares:  ", nrow(train_hog), " filas | ", ncol(train_hog), " cols")
message("  test_hogares:   ", nrow(test_hog),  " filas | ", ncol(test_hog),  " cols")
message("  train_personas: ", nrow(train_per), " filas | ", ncol(train_per), " cols")
message("  test_personas:  ", nrow(test_per),  " filas | ", ncol(test_per),  " cols")

# -- 3. Renombrado a nombres legibles --------------------------------------
#  Las columnas de train que no estan en test se ignoran aqui;
message("\n== Renombrando variables ==")
# ---- 3.1 Hogares ----------------------------------------------------------

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
    linea_indigencia  = Li,             # NO usar como predictor: define outcome
    linea_pobreza     = Lp,             # NO usar como predictor: define outcome
    factor_exp        = Fex_c,          # Factor de expansion nacional
    dpto              = Depto,
    factor_exp_dpto   = Fex_dpto        # Factor de expansion departamental
  )
}

# ---- 3.2 Personas ---------------------------------------------------------
renombrar_personas <- function(df) {
  mapa <- c(
    # -- Identificacion
    orden              = "Orden",
    zona               = "Clase",       # 1=Cabecera, 2=Resto
    ciudad             = "Dominio",
    dpto               = "Depto",
    factor_exp         = "Fex_c",
    factor_exp_dpto    = "Fex_dpto",

    # -- Sociodemografia
    sexo               = "P6020",       # 1=Hombre, 2=Mujer --> recodificar
    edad               = "P6040",
    parentesco         = "P6050",       # 1=Jefe, 2=Conyuge, 3=Hijo...
    afil_salud         = "P6090",       # 1=Si, 2=No, 9=NS --> recodificar
    tipo_afil_salud    = "P6100",       # 1=Contributivo, 2=Especial, 3=Subsidiado

    # -- Educacion
    nivel_educ         = "P6210",       # 1=Ninguno...9=Postgrado (ver PDF p.32)
                                        # OJO: en PDF hay 7 categorias, no 9.
                                        # a=Ninguno b=Preescolar c=Primaria
                                        # d=Secundaria e=Media f=Superior g=NS
    ultimo_grado_educ  = "P6210s1",

    # -- Empleo
    actividad_ppal     = "P6240",       # 1=Trabajo, 2=Tiene emp sin trabajar,
                                        # 3=Busco, 4=Estudio, 5=Hogar, 6=Incap.
    ocupacion_ciuo     = "Oficio",      # Codigo CIUO-88
    antiguedad_meses   = "P6426",       # Meses en el empleo actual
    posicion_ocup      = "P6430",       # 1=Emp.privado, 2=Emp.publico,
                                        # 3=Domestico, 4=Cta propia, 5=Patron,
                                        # 6=Fam sin rem, 7=Sin rem no fam,
                                        # 8=Jornalero, 9=Otro

    # -- Ingresos complementarios (solo indicador Si/No, sin valores en COP)
    recibe_horas_ext   = "P6510",       # 1=Si, 2=No, 9=NS --> recodificar
    recibe_primas      = "P6545",       # 1=Si, 2=No, 9=NS --> recodificar
    recibe_bonif       = "P6580",       # 1=Si, 2=No, 9=NS --> recodificar

    # -- Subsidios en especie (solo indicador Si/No)
    sub_especie_alim   = "P6585s1",     # 1=Si, 2=No, 9=NS --> recodificar
    sub_especie_transp = "P6585s2",     # 1=Si, 2=No, 9=NS --> recodificar
    sub_especie_fam    = "P6585s3",     # 1=Si, 2=No, 9=NS --> recodificar
    sub_especie_educ   = "P6585s4",     # 1=Si, 2=No, 9=NS --> recodificar

    # -- Pago en especie (alimentos, vivienda, transporte, otros)
    especie_alimentos  = "P6590",       # 1=Si, 2=No, 9=NS --> recodificar
    especie_vivienda   = "P6600",       # 1=Si, 2=No, 9=NS --> recodificar
    especie_transporte = "P6610",       # 1=Si, 2=No, 9=NS --> recodificar
    especie_otros      = "P6620",       # 1=Si, 2=No, 9=NS --> recodificar

    # -- Prestaciones anuales recibidas (todos 1=Si, 2=No --> recodificar)
    recibio_prima_serv = "P6630s1",     # Prima de servicios
    recibio_prima_nav  = "P6630s2",     # Prima de navidad
    recibio_prima_vac  = "P6630s3",     # Prima de vacaciones
    recibio_viaticos   = "P6630s4",     # Viaticos permanentes
    recibio_bonif_anu  = "P6630s6",     # Bonificaciones anuales

    # -- Caracteristicas empleo
    horas_trabajadas   = "P6800",       # Horas/semana empleo principal
    tamano_empresa     = "P6870",       # 1=Solo...9=101+ personas

    # -- Cotizacion pension (P6920 en PDF es 'cotizando a pension')
    # 1=Si, 2=No, 3=Ya pensionado --> recodificar especial
    cotiza_pension     = "P6920",

    # -- Segundo empleo
    tiene_2do_empleo   = "P7040",       # 1=Si, 2=No --> recodificar
    horas_2do_emp      = "P7045",
    ingreso_2do_emp    = "P7050",

    # -- Subempleo (todos 1=Si, 2=No --> recodificar)
    quiere_mas_horas   = "P7090",       # Quiere trabajar mas horas
    dilig_mas_horas    = "P7110",       # Hizo diligencias para trabajar mas horas
    disp_mas_horas     = "P7120",       # Estaba disponible para mas horas
    dilig_cambiar      = "P7150",       # Hizo diligencias para cambiar de trabajo
    puede_nuevo_emp    = "P7160",       # Puede empezar nuevo empleo antes de 1 mes
    busca_1era_vez     = "P7310",       # 1=Primera vez, 2=Trabajo antes

    # -- Desocupados
    posicion_ant       = "P7350",       # Posicion ocupacional ultimo trabajo

    # -- Otras fuentes de ingreso (indicadores Si/No)
    recibe_dom_otro_hog = "P7422",      # Ingreso domestico en otro hogar: 1=Si, 2=No
    recibe_agropec      = "P7472",      # Ingreso trabajo agropecuario: 1=Si, 2=No
    recibe_arr_pension  = "P7495",      # Arriendos y/o pensiones: 1=Si, 2=No

    # -- Ingresos no laborales (indicadores Si/No; sin montos en test)
    recibe_pension_jub  = "P7500s2",    # Pension/jubilacion: 1=Si, 2=No, 9=NS
    recibe_pension_alim = "P7500s3",    # Pension alimenticia: 1=Si, 2=No, 9=NS
    recibe_otros_nlab   = "P7505",      # Otros no laborales: 1=Si, 2=No

    # -- Transferencias e ingresos varios (indicadores Si/No)
    recibe_transf_nac   = "P7510s1",    # Dinero otros hogares en pais: 1=Si, 2=No, 9=NS
    recibe_remesas      = "P7510s2",    # Remesas exterior: 1=Si, 2=No, 9=NS
    recibe_subsidio     = "P7510s3",    # Ayudas instituciones: 1=Si, 2=No, 9=NS
    recibe_intereses    = "P7510s5",    # Intereses/dividendos/CDT: 1=Si, 2=No, 9=NS
    recibe_cesantias    = "P7510s6",    # Cesantias: 1=Si, 2=No, 9=NS
    recibe_otros_ing    = "P7510s7",    # Otras fuentes: 1=Si, 2=No, 9=NS

    # -- Flags laborales DANE (YA son 1=Si / 0=No segun PDF V128-V131, NO recodificar)
    pet                 = "Pet",        # Poblacion en edad de trabajar
    ocupado             = "Oc",
    desempleado         = "Des",
    inactivo            = "Ina"
  )

  mapa_valido <- mapa[mapa %in% names(df)]
  rename(df, any_of(mapa_valido))
}

train_hog <- renombrar_hogares(train_hog)
test_hog  <- renombrar_hogares(test_hog)
train_per <- renombrar_personas(train_per)
test_per  <- renombrar_personas(test_per)


# -- 4. Recodificacion de variables binarias -------------------------------
#  #  Convencion original DANE:  1=Si  |  2=No  |  9=NS/NI
#  Nueva convencion:          1=Si  |  0=No  |  moda (para 9=NS)
#  Casos especiales:
#  - sexo (P6020): 1=Hombre/2=Mujer --> mujer: 1=Mujer/0=Hombre
#  - cotiza_pension (P6920): 1=Si/2=No/3=Ya pensionado --> 1/0/0
#  - busca_1era_vez (P7310): 1=Primera vez/2=Trabajo antes --> 1/0 (distinto significado)
#  - Pet, Oc, Des, Ina: ya son 1/0 segun PDF, no tocar

message("\n== Recodificando variables binarias ==")

recodificar_si_no <- function(x) {
  x_bin <- case_when(
    x == 1 ~ 1L,
    x == 2 ~ 0L,
    TRUE   ~ NA_integer_          # 9=NS/NI u otros -> NA temporal
  )
  # Imputar con la moda (valor mas frecuente entre 0 y 1)
  moda <- as.integer(names(which.max(table(x_bin, useNA = "no"))))
  if_else(is.na(x_bin), moda, x_bin)
}

recodificar_personas <- function(df) {
  df |> mutate(

    # Sexo: nueva columna 0/1 (se elimina 'sexo' original en paso 8)
    mujer              = if_else(sexo == 2, 1L, if_else(sexo == 1, 0L, NA_integer_)),

    # Afiliacion salud
    afil_salud         = recodificar_si_no(afil_salud),

    # Ingresos complementarios
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

    # Cotizacion pension: 1=Si / 2=No / 3=Ya pensionado -> 1 / 0 / 0
    cotiza_pension     = case_when(
      cotiza_pension == 1 ~ 1L,
      cotiza_pension == 2 ~ 0L,
      cotiza_pension == 3 ~ 0L,   # ya pensionado = no cotiza actualmente
      TRUE                ~ 0L    # NS/NI -> conservador: asumir que no cotiza
    ),

    # Segundo empleo
    tiene_2do_empleo   = recodificar_si_no(tiene_2do_empleo),

    # Subempleo
    quiere_mas_horas   = recodificar_si_no(quiere_mas_horas),
    dilig_mas_horas    = recodificar_si_no(dilig_mas_horas),
    disp_mas_horas     = recodificar_si_no(disp_mas_horas),
    dilig_cambiar      = recodificar_si_no(dilig_cambiar),
    puede_nuevo_emp    = recodificar_si_no(puede_nuevo_emp),

    # busca_1era_vez: 1=Primera vez / 2=Trabajo antes -> 1=Primera vez / 0=Trabajo antes
    busca_1era_vez     = recodificar_si_no(busca_1era_vez),

    # Otras fuentes de ingreso
    recibe_dom_otro_hog = recodificar_si_no(recibe_dom_otro_hog),
    recibe_agropec      = recodificar_si_no(recibe_agropec),
    recibe_arr_pension  = recodificar_si_no(recibe_arr_pension),

    # Ingresos no laborales
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

    # Pet, Oc, Des, Ina: ya son 1/0 
  )
}

train_per <- recodificar_personas(train_per)
test_per  <- recodificar_personas(test_per)


# -- 5. Limpieza basica -----------------------------------------------------
message("\n== Limpieza basica ==")