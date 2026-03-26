#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 03_modelos/naive_bayes.R
#==============================================================================
# ALGORITMO : Naive Bayes
# PASO      : 1 — Baseline sin tuning
# OUTPUT    : submissions/NaiveBayes_default.csv
#
# SUPUESTO CLAVE:
#   Naive Bayes asume independencia condicional entre predictores dado Y.
#   En la práctica este supuesto se viola, pero el clasificador sigue siendo
#   competitivo con variables débilmente correlacionadas.
#   Para variables continuas asume distribución gaussiana por defecto.
#
# PREPROCESAMIENTO NECESARIO:
#   e1071::naiveBayes() no acepta NAs, characters ni factores de texto.
#==============================================================================