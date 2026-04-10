#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/Cart_cp_0001_maxdepth_20.R
#==============================================================================
# ALGORITMO : CART — Classification and Regression Tree (rpart)
# PASOS:
#Paso 3 — Desbalance con class weights → submissions/cart_v3.csv
# Mismo procesamiento que baseline, pero con tuning de cp usando validación cruzada.
# HIPERPARÁMETROS:
#   cp (complexity parameter): penaliza la complejidad del árbol.
#      Valor pequeño = árbol profundo = más flexibilidad, riesgo de overfitting.
#      Valor grande  = árbol simple  = más generalizable, posible underfitting.
#      Rango a explorar: 0.0001, 0.0005, 0.001, 0.005
#   maxdepth: profundidad máxima del árbol (20 = prácticamente sin límite).
#==============================================================================
# nolint start

