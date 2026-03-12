# -- Uso en scripts posteriores --------------------------------------------
dir_processed <- "00_data/processed"

train <- readRDS(file.path(dir_processed, "train_final.rds"))
test  <- readRDS(file.path(dir_processed, "test_final.rds"))