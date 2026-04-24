################################################################################
# Master script
#
# Running this file reproduces all results in the repository.
#
# To reproduce all results, run from R: source("01_code/00_rundirectory.R")
# or from command line: Rscript 01_code/00_rundirectory.R
#
# Authors: Group 04
# - Natalia Suescun
# - Daniela Solano
# - Jonathan Melo
################################################################################

# Detectar directorio raiz del proyecto
if (basename(getwd()) == "01_code") {
  setwd("..")
}

cat("
================================================================================
PROBLEM SET 2: PREDICTING POVERTY
Pipeline de Ejecucion Completo
================================================================================
Directorio de trabajo:", getwd(), "
Tiempo estimado: 4-6 horas
================================================================================
\n")

# Tiempo de inicio
start_time <- Sys.time()

# Funcion auxiliar para ejecutar scripts R
run_r_script <- function(script_path, description) {
  cat("\n================================================================\n")
  cat("EJECUTANDO:", description, "\n")
  cat("Script:", script_path, "\n")
  cat("================================================================\n")
  step_start <- Sys.time()

  tryCatch({
    source(script_path, local = new.env())
    step_end <- Sys.time()
    elapsed <- as.numeric(difftime(step_end, step_start, units = "mins"))
    cat(sprintf("\nCompletado en %.1f minutos\n", elapsed))
  }, error = function(e) {
    cat("\nERROR:", e$message, "\n")
    stop(paste("Ejecucion fallo en", script_path))
  })
}

# ================================================================
# PASO 1: LIMPIEZA Y MERGE DE DATOS
# ================================================================
# Toma los CSV crudos de Kaggle (00_data/raw/) y genera
# train_final.rds y test_final.rds en 00_data/processed/
# con variables renombradas, recodificadas, agregadas a nivel
# hogar, y variables adicionales basadas en literatura.

run_r_script("01_code/01_data_clean_and_merge.R",
             "PASO 1: Limpieza, merge y construccion de variables")

# ================================================================
# PASO 2: ANALISIS EXPLORATORIO (EDA)
# ================================================================
# Genera graficos descriptivos, tablas de correlacion y
# resumen de hallazgos sobre la estructura del problema.

run_r_script("01_code/02_data_explore.R",
             "PASO 2: Analisis exploratorio de datos")

# ================================================================
# PASO 3: MODELOS DE CLASIFICACION
# ================================================================
# Se entrenan 6 algoritmos distintos. Cada script carga los datos
# desde 00_data/processed/, entrena con CV-5, optimiza threshold
# sobre predicciones OOF, y guarda metricas en 02_outputs/ y
# submissions en 03_submissions/.

# -- 3.1 LPM (Modelo de Probabilidad Lineal) --------------------------------
# Regresion OLS con variable dependiente binaria.
# Incluye feature selection (top 20/40/60 de RF) y balanceo
# (5 tecnicas). Mejor: class_weights con 98 vars.

run_r_script("01_code/03_models/LPM/Balance/LPM_003A.R",
             "PASO 3.1: LPM - Mejor modelo (class_weights, 98 vars)")

# -- 3.2 Elastic Net (Regresion logistica penalizada) ------------------------
# glmnet binomial con alpha/lambda optimizados por ROC.
# Threshold optimizado sobre OOF. Lasso (alpha=1) selecciona
# variables automaticamente. Mejor: baseline con 93 vars.

run_r_script("01_code/03_models/ElasticNet/Baseline/ElasticNet.R",
             "PASO 3.2: Elastic Net - Mejor modelo (baseline, 93 vars)")

# -- 3.3 Naive Bayes ---------------------------------------------------------
# Clasificador probabilistico con supuesto de independencia
# condicional. Incluye feature selection con top-20 de RF.
# Mejor: top-20 variables.

run_r_script("01_code/03_models/NaiveBayes/Features/NaiveBayes_feature.R",
             "PASO 3.3: Naive Bayes - Mejor modelo (top-20 vars RF)")

# -- 3.4 CART (Arbol de decision) --------------------------------------------
# Arbol de clasificacion con tuning de cp via CV anidado.
# Feature selection con top-60 de RF. Threshold optimizado OOF.

run_r_script("01_code/03_models/CARET/Features/CART_cv5_feat.R",
             "PASO 3.4: CART - Mejor modelo (top-60 vars RF)")

# -- 3.5 Random Forest -------------------------------------------------------
# Ensemble de 500 arboles (ranger). HP tuning de mtry y
# min.node.size via CV-5. Comparacion de 5 tecnicas de balanceo.
# Mejor: baseline con todas las variables.

run_r_script("01_code/03_models/RandomForest/RF_006.R",
             "PASO 3.5: Random Forest - Mejor modelo (baseline, 110 vars)")

# -- 3.6 XGBoost (Gradient Boosting) -----------------------------------------
# Boosting secuencial con arboles. HP tuning de depth, eta,
# gamma, min_child_weight, colsample, subsample, L1, L2.
# Mejor: spec A (componentes, 75 vars) sin Bogota.

run_r_script("01_code/03_models/XGBoost/run_r_script("01_code/03_models/XGBoost/XGB_007.R",
             "PASO 3.6: XGBoost - Mejor modelo (93 vars, sin Bogota)")XGB_008.R"

# ================================================================
# RESUMEN FINAL
# ================================================================

end_time <- Sys.time()
total_elapsed <- as.numeric(difftime(end_time, start_time, units = "mins"))

cat("
================================================================================
PIPELINE COMPLETADO
================================================================================
Tiempo total de ejecucion:", round(total_elapsed, 1), "minutos
Iniciado:", format(start_time, "%Y-%m-%d %H:%M:%S"),"
Finalizado:", format(end_time, "%Y-%m-%d %H:%M:%S"),"
================================================================================

Todos los resultados han sido generados en:
- 00_data/processed/    -- datos limpios (train_final.rds, test_final.rds)
- 02_outputs/models/    -- metricas y diagnosticos por modelo
- 02_outputs/model_registry.csv -- tabla resumen de todos los modelos
- 03_submissions/       -- predicciones para Kaggle

================================================================================
\n")
