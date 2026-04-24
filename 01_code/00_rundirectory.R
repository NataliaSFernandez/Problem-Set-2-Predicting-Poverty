################################################################################
# Master script
#
# Running this file reproduces all results in the repository.
#
# To reproduce all results, run from R: source("01_code/00_rundirectory.R")
# or from command line: Rscript 01_code/00_rundirectory.R
#
# Authors:
# - Natalia Suescun
# - Daniela Solano
# - Jonathan Melo
################################################################################

# Detectar directorio raíz del proyecto
if (basename(getwd()) == "01_code") {
  setwd("..")
}

cat("
================================================================================
PROBLEM SET 1: PREDICCIÓN DE INGRESOS LABORALES
Pipeline de Ejecución Completo
================================================================================
Directorio de trabajo:", getwd(), "
Tiempo estimado: 20-30 minutos
================================================================================
\n")

# Tiempo de inicio
start_time <- Sys.time()

# Función auxiliar para ejecutar scripts Python
run_python <- function(script_path, description) {
  cat("\n================================================================\n")
  cat("EJECUTANDO:", description, "\n")
  cat("================================================================\n")
  step_start <- Sys.time()
  
  result <- system(paste("python", script_path), intern = FALSE)
  
  step_end <- Sys.time()
  elapsed <- as.numeric(difftime(step_end, step_start, units = "secs"))
  
  if (result == 0) {
    cat(sprintf("\n✓ Completado en %.1f segundos\n", elapsed))
  } else {
    stop(paste("ERROR en", script_path, "- Código de salida:", result))
  }
}

# Función auxiliar para ejecutar scripts R
run_r_script <- function(script_path, description) {
  cat("\n================================================================\n")
  cat("EJECUTANDO:", description, "\n")
  cat("================================================================\n")
  step_start <- Sys.time()
  
  tryCatch({
    source(script_path, local = new.env())
    step_end <- Sys.time()
    elapsed <- as.numeric(difftime(step_end, step_start, units = "secs"))
    cat(sprintf("\n✓ Completado en %.1f segundos\n", elapsed))
  }, error = function(e) {
    cat("\n✗ ERROR:", e$message, "\n")
    stop(paste("Ejecución falló en", script_path))
  })
}

# ================================================================
# PASO 0: OBTENCIÓN Y PREPARACIÓN DE DATOS
# ================================================================

run_python("01_code/00_data_scrapper.py", 
           "PASO 0.1: Web Scraping GEIH 2018")

run_python("01_code/00_data_cleaning.py", 
           "PASO 0.2: Limpieza y Preparación de Datos")

run_python("01_code/00_descriptive_stats.py", 
           "PASO 0.3: Estadísticas Descriptivas")

# ================================================================
# PASO 1: SECCIÓN 1 - PERFIL EDAD-INGRESO
# ================================================================

run_r_script("01_code/01_section1_age_income.R", 
             "PASO 1: Sección 1 - Perfil Edad-Ingreso")

# ================================================================
# PASO 2: SECCIÓN 2 - BRECHA DE GÉNERO
# ================================================================

run_r_script("01_code/02_section2_gendergap_income.R", 
             "PASO 2: Sección 2 - Brecha Salarial de Género")

# ================================================================
# PASO 3: SECCIÓN 3 - PREDICCIÓN DE INGRESOS
# ================================================================

run_r_script("01_code/03_section3_income_prediction.R", 
             "PASO 3: Sección 3 - Predicción de Ingresos")

# ================================================================
# RESUMEN FINAL
# ================================================================

end_time <- Sys.time()
total_elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("
================================================================================
PIPELINE COMPLETADO EXITOSAMENTE
================================================================================
Tiempo total de ejecución:", round(total_elapsed / 60, 1), "minutos
Iniciado:", format(start_time, "%Y-%m-%d %H:%M:%S"),"
Finalizado:", format(end_time, "%Y-%m-%d %H:%M:%S"),"
================================================================================

Todos los resultados han sido generados en:
- 00_data/cleaned/
- 02_output/figures/
- 02_output/tables/
- 02_output/05_diagnostic_analysis/

Para verificar los resultados, revisar los archivos PNG y TEX en 02_output/.
================================================================================
\n")
