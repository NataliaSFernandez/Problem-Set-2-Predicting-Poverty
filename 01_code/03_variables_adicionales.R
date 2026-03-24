
#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script 03: Variables adicionales basadas en literatura académica
#==============================================================================
# OBJETIVO: Construir nuevas variables predictoras con respaldo en la literatura
#   académica de caracterización y predicción de pobreza, a partir de las
#   variables ya disponibles en los datos procesados y de re-agregaciones
#   adicionales desde las personas.
#
# INPUTS:  00_data/processed/train_final.rds
#          00_data/processed/test_final.rds
#          00_data/raw/train_personas.zip  (re-agregaciones adicionales)
#          00_data/raw/test_personas.csv   (re-agregaciones adicionales)
#
# OUTPUTS: 00_data/processed/train_features.rds
#          00_data/processed/test_features.rds
#==============================================================================

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, readr, fs)

# ==============================================================================
# PASO 1: CARGAR DATOS PROCESADOS
# ==============================================================================

dir_raw       <- "00_data/raw"
dir_processed <- "00_data/processed"
dir_create(dir_processed, recurse = TRUE)

message("== Cargando datos procesados ==")
train <- readRDS(file.path(dir_processed, "train_final.rds"))
test  <- readRDS(file.path(dir_processed, "test_final.rds"))

message("  train: ", nrow(train), " x ", ncol(train))
message("  test:  ", nrow(test),  " x ", ncol(test))

# ==============================================================================
# PASO 2: CARGAR PERSONAS CRUDOS (para nuevas agregaciones)
# ==============================================================================
# Variables no capturadas en el script 01 pero relevantes según la literatura:
#   - Estrato socioeconómico (Estrato1): proxy directo de nivel de vida en Colombia
#   - Sexo del jefe de hogar (mujer_jefa): factor de vulnerabilidad estructural
#   - Posiciones ocupacionales adicionales: patrón, jornalero, doméstico,
#     familiar sin remuneración
#   - Tamaño de empresa: proxy de formalidad del empleo
#   - Antigüedad laboral: estabilidad del empleo
#   - Subempleo por ingresos y por competencias (adicionales al de horas)
#   - Ingreso agropecuario y de arrendamientos

message("\n== Cargando datos crudos de personas ==")

train_per <- read_csv(
  unz(file.path(dir_raw, "train_personas.zip"), "train_personas/train_personas.csv"),
  show_col_types = FALSE
)
test_per <- read_csv(
  file.path(dir_raw, "test_personas.csv"),
  show_col_types = FALSE
)

# Función auxiliar: recodificar 1=Sí / 2=No / 9=NS -> 1 / 0 con moda para NA
recodificar_si_no <- function(x) {
  x_bin <- case_when(x == 1 ~ 1L, x == 2 ~ 0L, TRUE ~ NA_integer_)
  moda  <- as.integer(names(which.max(table(x_bin, useNA = "no"))))
  if_else(is.na(x_bin), moda, x_bin)
}

# ==============================================================================
# PASO 3: NUEVAS AGREGACIONES DESDE PERSONAS CRUDOS
# ==============================================================================

message("\n== Agregando nuevas variables desde personas ==")

agregar_nuevas_vars_per <- function(df) {

  df <- df |>
    mutate(
      # Sexo del individuo (para identificar si el jefe es mujer)
      mujer         = if_else(P6020 == 2, 1L, if_else(P6020 == 1, 0L, NA_integer_)),
      parentesco    = P6050,

      # Estrato socioeconómico: 9 = No aplica -> NA
      estrato       = if_else(Estrato1 %in% c(9), NA_real_, as.numeric(Estrato1)),

      # Posición ocupacional
      posicion_ocup = P6430,

      # Tamaño de empresa (P6870): 1=1 persona, 2=2-3, ..., 9=más de 100
      tamano_emp    = if_else(is.na(P6870), NA_real_, as.numeric(P6870)),

      # Antigüedad en el empleo actual (meses)
      antiguedad    = as.numeric(P6426),

      # Subempleo por competencias (P7110) y por ingresos (P7120)
      sub_comp      = recodificar_si_no(P7110),
      sub_ing       = recodificar_si_no(P7120),

      # Ingreso agropecuario (P7472) e ingreso por arriendo (P7500s1)
      recibe_agropec   = recodificar_si_no(P7472),
      recibe_arriendos = recodificar_si_no(P7500s1)
    )

  df |>
    group_by(id) |>
    summarise(

      # -----------------------------------------------------------------------
      # A. Estrato socioeconómico del hogar
      # Referencia: DANE IPM Colombia; Profamilia (2015)
      # El estrato determina tarifas de servicios públicos y acceso a subsidios.
      # Es el indicador socioeconómico más usado en política social colombiana.
      # Se toma la moda (todos los miembros del hogar deben tener el mismo).
      # -----------------------------------------------------------------------
      estrato_hog = {
        vals <- estrato[!is.na(estrato)]
        if (length(vals) == 0) NA_real_
        else as.numeric(names(which.max(table(vals))))
      },
      estrato_bajo = as.integer({
        vals <- estrato[!is.na(estrato)]
        if (length(vals) == 0) NA_integer_
        else as.integer(as.numeric(names(which.max(table(vals)))) <= 2)
      }),

      # -----------------------------------------------------------------------
      # B. Sexo del jefe de hogar
      # Referencia: Buvinic & Gupta (1997); Medeiros & Costa (2008)
      # Los hogares con jefa mujer enfrentan mayor vulnerabilidad a la pobreza
      # por brechas salariales de género y mayor carga de trabajo de cuidado.
      # -----------------------------------------------------------------------
      mujer_jefa = as.integer(any(parentesco == 1 & mujer == 1, na.rm = TRUE)),

      # -----------------------------------------------------------------------
      # C. Posiciones ocupacionales adicionales
      # Referencia: ILO (2018) "Women and men in the informal economy"
      # El trabajo doméstico, jornalero y familiar sin remuneración son las
      # categorías laborales más precarias e informales en Colombia.
      # -----------------------------------------------------------------------
      prop_patron      = mean(posicion_ocup == 5, na.rm = TRUE),  # empleador/patrón
      prop_domestico   = mean(posicion_ocup == 3, na.rm = TRUE),  # trabajador doméstico
      prop_fam_sin_rem = mean(posicion_ocup == 6, na.rm = TRUE),  # familiar sin remuneración
      prop_jornalero   = mean(posicion_ocup == 8, na.rm = TRUE),  # jornalero/peón (rural)

      # -----------------------------------------------------------------------
      # D. Tamaño de empresa
      # Referencia: Levy (2008) "Good intentions, bad outcomes"
      # Las microempresas (<5 trabajadores) concentran el empleo informal en
      # América Latina y pagan salarios sistemáticamente más bajos.
      # -----------------------------------------------------------------------
      prop_microempresa = mean(tamano_emp <= 2, na.rm = TRUE),   # 1-3 personas
      prop_gran_empresa = mean(tamano_emp >= 6, na.rm = TRUE),   # 20+ personas

      # -----------------------------------------------------------------------
      # E. Antigüedad laboral promedio
      # Referencia: Topel (1991) "Specific capital, mobility, and wages"
      # Mayor antigüedad refleja estabilidad laboral y salario acumulado.
      # Baja antigüedad indica inestabilidad o empleo reciente precario.
      # -----------------------------------------------------------------------
      antiguedad_prom = mean(antiguedad, na.rm = TRUE),
      antiguedad_alta = as.integer(any(antiguedad > 24, na.rm = TRUE)),  # >2 años

      # -----------------------------------------------------------------------
      # F. Subempleo por ingresos y competencias
      # Referencia: ILO (2007) "Decent Work Indicators"
      # El subempleo por ingresos identifica "trabajadores pobres": personas
      # que trabajan tiempo completo pero no alcanzan ingresos suficientes.
      # -----------------------------------------------------------------------
      prop_sub_ingresos     = mean(sub_ing  == 1, na.rm = TRUE),
      prop_sub_competencias = mean(sub_comp == 1, na.rm = TRUE),

      # -----------------------------------------------------------------------
      # G. Diversificación de ingresos: agropecuario y de activos
      # Referencia: Ellis (2000) "Rural Livelihoods and Diversity"
      # El ingreso agropecuario es típico de hogares rurales pobres.
      # El ingreso por arrendamientos refleja posesión de activos.
      # -----------------------------------------------------------------------
      alguno_agropec   = as.integer(any(recibe_agropec   == 1, na.rm = TRUE)),
      alguno_arriendos = as.integer(any(recibe_arriendos == 1, na.rm = TRUE)),

      .groups = "drop"
    ) |>
    mutate(across(where(is.numeric), ~ if_else(is.infinite(.), NA_real_, .)))
}

vars_nuevas_train <- agregar_nuevas_vars_per(train_per)
vars_nuevas_test  <- agregar_nuevas_vars_per(test_per)

message("  Nuevas vars agregadas desde personas: ", ncol(vars_nuevas_train) - 1)

rm(train_per, test_per)
gc()

# ==============================================================================
# PASO 4: MERGE CON DATOS PROCESADOS
# ==============================================================================

message("\n== Merge con datos procesados ==")
train <- left_join(train, vars_nuevas_train, by = "id")
test  <- left_join(test,  vars_nuevas_test,  by = "id")

# ==============================================================================
# PASO 5: NUEVAS VARIABLES DERIVADAS DE VARIABLES YA DISPONIBLES
# ==============================================================================

message("\n== Construyendo nuevas variables derivadas ==")

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

      # Hacinamiento crítico: >3 personas por cuarto (umbral IPM Colombia)
      hacinamiento_critico = as.integer(num_personas / pmax(num_cuartos, 1) > 3),

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

      # Proporción del hogar que son adultos mayores (≥65)
      prop_adultos_may_p   = n_adultos_may / pmax(num_personas, 1),

      # Menores por ocupado: cuántos niños financia cada trabajador
      menores_por_ocupado  = n_menores_18 / pmax(n_ocupados, 1),

      # Hogar sin ningún ocupado (ingreso laboral nulo — vulnerabilidad extrema)
      sin_ocupados         = as.integer(n_ocupados == 0),

      # Hogar con un único ocupado (riesgo de ingreso único — alta vulnerabilidad)
      un_solo_ocupado      = as.integer(n_ocupados == 1),

      # Hogar con dos o más ocupados (diversificación de ingreso laboral)
      doble_ingreso        = as.integer(n_ocupados >= 2),

      # Hogar monoparental: jefa mujer con al menos un menor de 18
      # Referencia: Buvinic & Gupta (1997) — alta vulnerabilidad estructural
      hogar_monoparental   = as.integer(mujer_jefa == 1 & n_menores_18 > 0),

      # ========================================================================
      # BLOQUE 3: CAPITAL HUMANO — EFECTOS NO LINEALES Y UMBRALES
      # Referencia: Mincer (1974); Card (1999) "The causal effect of education
      # on earnings". Los retornos a la educación son no lineales: existen
      # "sheepskin effects" (saltos discontinuos) en bachillerato y universidad.
      # El IPM Colombia define privación educativa del jefe como ≤ primaria.
      # ========================================================================

      # Término cuadrático de educación del jefe (retornos crecientes)
      educ_jefe_sq         = nivel_educ_jefe^2,

      # Término cuadrático de educación promedio del hogar
      educ_prom_sq         = nivel_educ_prom^2,

      # Privación educativa del jefe: máximo básica primaria (umbral IPM)
      educ_insuf_jefe      = as.integer(nivel_educ_jefe <= 3),

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

      # Subempleo total: promedio de tres dimensiones (horas, ingresos, competencias)
      subempleo_total      = rowMeans(
        cbind(prop_subempleo, prop_sub_ingresos, prop_sub_competencias),
        na.rm = TRUE
      ),

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

      # Intersección rural × Costa Caribe: pobreza rural costera estructural
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

      # Educación × Formalidad: hogar con capital humano Y empleo formal
      # El más protegido: Mincer (1974) + Fields (2011)
      educ_x_formal        = nivel_educ_jefe * prop_cotiza_pension,

      # Educación × Tasa de ocupación: capital humano convertido efectivamente
      # en empleo (complementariedad entre oferta y demanda laboral)
      educ_x_ocup          = nivel_educ_jefe * tasa_ocupacion,

      # Menores × Desocupación: carga de menores sin ingreso laboral
      # Captura la pobreza infantil dinámica (Lanjouw & Ravallion 1995)
      menores_x_desocup    = n_menores_18 * (1 - tasa_ocupacion),

      # Dependencia × Informalidad: doble penalización estructural
      # (muchos dependientes + trabajo precario sin seguridad de ingresos)
      depend_x_informal    = ratio_depend * prop_informal,

      # Rural × Cuenta propia: autoempleo rural distinto del urbano
      # En zonas rurales puede reflejar subsistencia más que elección
      rural_x_cta_propia   = as.numeric(zona == 2) * prop_cta_propia,

      # Educación × Rural: menores retornos a la educación en zona rural
      # por menor oferta de empleos cualificados fuera de ciudades
      educ_x_rural         = nivel_educ_jefe * as.numeric(zona == 2),

      # Subsidiado × Menores: salud subsidiada Y muchos menores
      # La combinación más fuerte de señales de pobreza infantil en el hogar
      subsidiado_x_menores = prop_subsidiado * n_menores_18,

      # Jefe mayor con jubilación: adulto mayor con pensión (ciclo de vida
      # avanzado con protección formal — habitualmente no pobre)
      jefe_mayor_pension   = as.numeric(edad_jefe > 55) * alguno_pension_jub,

      # Estrato × Zona: el estrato tiene distinto significado en lo urbano
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
      n_privaciones        = dep_educacion  + dep_vivienda      +
                             dep_empleo_formal + dep_proteccion  +
                             dep_dependencia,

      # Hogar multiprivado: ≥2 privaciones simultáneas (umbral Alkire-Foster)
      multi_privado        = as.integer(n_privaciones >= 2),

      # Pobreza multidimensional severa: ≥3 privaciones simultáneas
      privado_severo       = as.integer(n_privaciones >= 3)
    )
}

train <- construir_nuevas_variables(train)
test  <- construir_nuevas_variables(test)

# ==============================================================================
# PASO 6: DIAGNÓSTICO DE LAS NUEVAS VARIABLES
# ==============================================================================

message("\n== Diagnóstico de nuevas variables ==")

vars_nuevas <- c(
  # Bloque 1: Vivienda
  "hacinamiento", "hacinamiento_dorm", "hacinamiento_critico",
  "cuartos_pc", "tenencia_insegura", "vivienda_propia_pag",
  # Bloque 2: Demografía
  "prop_menores", "prop_adultos_may_p", "menores_por_ocupado",
  "sin_ocupados", "un_solo_ocupado", "doble_ingreso", "hogar_monoparental",
  # Bloque 3: Capital humano
  "educ_jefe_sq", "educ_prom_sq", "educ_insuf_jefe",
  "educ_media_o_mas", "educ_superior_jefe", "brecha_educ",
  # Bloque 4: Calidad empleo
  "indice_formalidad", "formal_estricto", "vulnerabilidad_lab",
  "tasa_empleo_total", "subempleo_total",
  # Bloque 5: Diversificación ingresos
  "n_fuentes_no_lab", "tiene_ingreso_no_lab", "solo_subsidio", "ingresos_activos",
  # Bloque 6: Protección social
  "proteccion_social", "desprotegido", "prop_contributivo",
  # Bloque 7: Región
  "costa_caribe", "region_pacifico", "bogota", "eje_cafetero", "rural_costa_caribe",
  # Bloque 8: Interacciones
  "educ_x_formal", "educ_x_ocup", "menores_x_desocup",
  "depend_x_informal", "rural_x_cta_propia", "educ_x_rural",
  "subsidiado_x_menores", "jefe_mayor_pension", "estrato_x_zona",
  # Bloque 9: No lineales
  "log_num_personas", "log_n_menores", "ratio_depend_sq",
  "tasa_ocup_sq", "hacinamiento_sq", "edad_jefe_sq",
  # Bloque 10: IPM
  "dep_educacion", "dep_vivienda", "dep_empleo_formal",
  "dep_proteccion", "dep_dependencia",
  "n_privaciones", "multi_privado", "privado_severo",
  # Desde personas crudos
  "estrato_hog", "estrato_bajo", "mujer_jefa",
  "prop_patron", "prop_domestico", "prop_fam_sin_rem", "prop_jornalero",
  "prop_microempresa", "prop_gran_empresa",
  "antiguedad_prom", "antiguedad_alta",
  "prop_sub_ingresos", "prop_sub_competencias",
  "alguno_agropec", "alguno_arriendos"
)

cat("\n-- Resumen de nuevas variables en train --\n")
for (v in vars_nuevas) {
  if (v %in% names(train)) {
    x <- train[[v]]
    cat(sprintf("  %-30s | media = %6.3f | NA = %.1f%%\n",
                v, mean(x, na.rm = TRUE), 100 * mean(is.na(x))))
  } else {
    cat(sprintf("  %-30s | *** NO GENERADA ***\n", v))
  }
}

# Correlación de las nuevas variables con la variable objetivo (solo train)
if ("pobre" %in% names(train)) {
  cat("\n-- Correlación de nuevas variables con 'pobre' (train) --\n")
  cors <- sapply(vars_nuevas, function(v) {
    if (v %in% names(train)) {
      tryCatch(
        cor(as.numeric(train[[v]]), train$pobre, use = "complete.obs"),
        error = function(e) NA_real_
      )
    } else {
      NA_real_
    }
  })
  cors_df <- data.frame(variable = names(cors), correlacion = round(cors, 4)) |>
    filter(!is.na(correlacion)) |>
    arrange(desc(abs(correlacion)))
  print(cors_df, row.names = FALSE)
}

# ==============================================================================
# PASO 7: GUARDAR DATOS ENRIQUECIDOS
# ==============================================================================

message("\n== Guardando datos enriquecidos ==")

saveRDS(train, file.path(dir_processed, "train_features.rds"))
saveRDS(test,  file.path(dir_processed, "test_features.rds"))

write_csv(train, file.path(dir_processed, "train_features.csv"))
write_csv(test,  file.path(dir_processed, "test_features.csv"))

n_orig <- ncol(readRDS(file.path(dir_processed, "train_final.rds")))
message("  train_features: ", nrow(train), " x ", ncol(train),
        "  (+", ncol(train) - n_orig, " nuevas variables)")
message("  test_features:  ", nrow(test),  " x ", ncol(test))

cat("\n== Script 03 completado ==\n")
cat("Outputs guardados en 00_data/processed/:\n")
cat("  train_features.rds / train_features.csv\n")
cat("  test_features.rds  / test_features.csv\n\n")

# Uso en el script de modelado:
# train <- readRDS("00_data/processed/train_features.rds")
# test  <- readRDS("00_data/processed/test_features.rds")
