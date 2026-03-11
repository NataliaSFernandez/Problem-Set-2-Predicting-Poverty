# 00_download_data.R

if (!requireNamespace("here", quietly = TRUE)) install.packages("here")
library(here)

raw_dir <- here("00_data", "raw")

if (!dir.exists(raw_dir)) {
  dir.create(raw_dir, recursive = TRUE)
}

zip_path <- file.path(raw_dir, "uniandes-bdml-2026-10-ps-2.zip")

# Descargar competencia de Kaggle
cmd_download <- paste(
  "kaggle competitions download -c uniandes-bdml-2026-10-ps-2",
  "-p", shQuote(raw_dir)
)

system(cmd_download)

# Descomprimir
if (file.exists(zip_path)) {
  unzip(zip_path, exdir = raw_dir)
  message("Datos descargados y descomprimidos en: ", raw_dir)
} else {
  stop("No se encontró el archivo zip descargado. Revisa autenticación de Kaggle o acceso a la competencia.")
}