
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
#    2. Limpieza individual de cada base
#    3. Feature engineering a nivel persona (agregado al hogar)
#    4. Merge hogar + personas (train y test)
#    5. Feature engineering a nivel hogar
#    6. Alineacion de columnas train/test
#    7. Guardado de bases limpias
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
  skimr,       # diagnosticos rapidos
  fs           # manejo de rutas
)

# -- 1. Rutas ---------------------------------------------------------------
dir_raw       <- "data/raw"
dir_processed <- "data/processed"
dir_create(dir_processed, recurse = TRUE)

# -- 2. Carga de datos ------------------------------------------------------
message("== Cargando datos ==")

train_hog <- read_csv(file.path(dir_raw, "train_hogares.csv"),  show_col_types = FALSE)
test_hog  <- read_csv(file.path(dir_raw, "test_hogares.csv"),   show_col_types = FALSE)
train_per <- read_csv(file.path(unz(file.path(dir_raw, "train_personas.zip"), "train_personas.csv")))
test_per  <- read_csv(file.path(dir_raw, "test_personas.csv"),  show_col_types = FALSE)

message("  train_hogares:  ", nrow(train_hog), " filas | ", ncol(train_hog), " cols")
message("  test_hogares:   ", nrow(test_hog),  " filas | ", ncol(test_hog),  " cols")
message("  train_personas: ", nrow(train_per), " filas | ", ncol(train_per), " cols")
message("  test_personas:  ", nrow(test_per),  " filas | ", ncol(test_per),  " cols")

