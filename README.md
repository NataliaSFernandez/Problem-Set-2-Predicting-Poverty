# Problem Set 2: Predicción de la Pobreza

**Big Data and Machine Learning para Economía Aplicada (MECA 4107)**
Universidad de los Andes | 2026-10

## Equipo 04

| Integrante |
|------------|
| Natalia Suescún |
| Daniela Solano |
| Jonathan Melo |

## Objetivo

Predecir si un hogar se encuentra en situación de pobreza en Colombia, utilizando modelos de clasificación entrenados con datos de la GEIH (Gran Encuesta Integrada de Hogares) del DANE. Se evalúan seis algoritmos distintos, se compara su desempeño predictivo (F1 score) y se discuten las implicaciones para la focalización de política pública.

## Cómo Reproducir los Resultados

### Requisitos

- R >= 4.5
- Paquetes: tidyverse, caret, glmnet, ranger, xgboost, e1071, pROC, MLmetrics, themis, corrplot, skimr, scales, janitor, fs

### Pasos

1. Clonar el repositorio:
```bash
git clone https://github.com/NataliaSFernandez/Problem-Set-2-Predicting-Poverty.git
cd Problem-Set-2-Predicting-Poverty
```

2. Ejecutar el pipeline completo:
```bash
Rscript 01_code/00_rundirectory.R
```

Esto ejecuta todos los scripts de forma secuencial: limpieza de datos, análisis exploratorio y los seis modelos de clasificación. Tiempo total aproximado: 80 minutos.

3. Todos los resultados (figuras, métricas, submissions) se regeneran en `02_outputs/` y `03_submissions/`.

## Estructura del Repositorio

```
Problem-Set-2-Predicting-Poverty/
|
|-- 00_data/
|   |-- raw/                          # Datos crudos de Kaggle
|   |   |-- train_hogares.csv
|   |   |-- test_hogares.csv
|   |   |-- train_personas.zip
|   |   |-- test_personas.csv
|   |   |-- DiccionarioDatos.md
|   |-- processed/                    # Datos limpios (generados por el pipeline)
|       |-- train_final.rds
|       |-- test_final.rds
|
|-- 01_code/
|   |-- 00_rundirectory.R             # Script maestro: ejecuta todo el pipeline
|   |-- 01_data_clean_and_merge.R     # Limpieza, recodificación, agregación
|   |-- 02_data_explore.R             # Análisis exploratorio de datos
|   |-- 03_models/
|       |-- LPM/                      # Modelo de Probabilidad Lineal
|       |   |-- Baseline/LPM.R
|       |   |-- Feature/LPM_003.R
|       |   |-- Balance/LPM_003A.R    # Mejor LPM (class_weights, 81 vars)
|       |-- ElasticNet/               # Regresión logística con penalización L1/L2
|       |   |-- Baseline/ElasticNet.R # Mejor EN (baseline, 91 vars)
|       |   |-- Features/ElasticNet_features.R
|       |   |-- Balance/ElasticNet_balance.R
|       |-- NaiveBayes/               # Clasificador probabilístico
|       |   |-- Baseline/NaiveBayes_default.R
|       |   |-- Features/NaiveBayes_feature.R  # Mejor NB (top-20 vars RF)
|       |   |-- Balance/NaiveBayes_balance.R
|       |-- CARET/                    # Árboles de decisión (CART)
|       |   |-- Baseline/
|       |   |-- Features/CART_cv5_feat.R  # Mejor CART (top-60 vars RF)
|       |   |-- Balance/
|       |-- RandomForest/             # Ensamble de árboles de decisión
|       |   |-- RF_001.R              # Baseline + ranking de importancia
|       |   |-- RF_006.R              # Mejor RF (HP tuning + balance, 92 vars)
|       |-- XGBoost/                  # Gradient Boosting
|           |-- XGB_001.R
|           |-- XGB_008.R            # Mejor XGB (75 vars, sin Bogotá)
|
|-- 02_outputs/
|   |-- figures/data_explore/         # Figuras del EDA
|   |-- models/                       # Métricas, diagnósticos y gráficos por modelo
|   |   |-- LPM/
|   |   |-- EN/
|   |   |-- NB/
|   |   |-- CARTs/
|   |   |-- RandomForest/
|   |   |-- XGBoost/
|   |-- model_registry.csv            # Tabla resumen de todos los modelos
|
|-- 03_submissions/                   # Archivos CSV de predicciones para Kaggle
|
|-- renv.lock                         # Ambiente de R para reproducibilidad
```

## Descripción del Pipeline

El script maestro `00_rundirectory.R` ejecuta los siguientes pasos:

| Paso | Script | Descripción | Tiempo |
|------|--------|-------------|--------|
| 1 | `01_data_clean_and_merge.R` | Carga datos crudos, renombra variables, recodifica binarias DANE, imputa NAs estructurales, agrega datos de personas a nivel hogar, construye variables adicionales basadas en literatura económica. Resultado: `train_final.rds` (154,393 hogares, 110 variables) y `test_final.rds`. | 1 min |
| 2 | `02_data_explore.R` | Análisis de desbalance de clases (80/20), tasas de pobreza por zona y departamento, comparaciones bivariadas, correlaciones. Resultado: 9 figuras PNG + resumen en texto. | <1 min |
| 3.1 | `LPM_003A.R` | Modelo de Probabilidad Lineal con class_weights, 81 variables. Incluye selección de variables y comparación de 5 técnicas de balanceo. | 2 min |
| 3.2 | `ElasticNet.R` | Elastic Net (glmnet, grilla alpha/lambda, optimización de threshold sobre OOF). | 9 min |
| 3.3 | `NaiveBayes_feature.R` | Naive Bayes con selección de variables RF (top-20/40/60). | 3 min |
| 3.4 | `CART_cv5_feat.R` | CART con CV anidado (5x3) y selección de variables RF. | 21 min |
| 3.5 | `RF_006.R` | Random Forest (ranger, tuning de hiperparámetros, comparación de 5 técnicas de balanceo). | 33 min |
| 3.6 | `XGB_008.R` | XGBoost (búsqueda aleatoria de hiperparámetros, dos especificaciones: componentes vs índices). | 10 min |
| | **Total** | | **80 min** |

## Comparación de Modelos (Deck 2)

Mejor modelo de cada algoritmo, ordenados por F1 score:

| Ranking | Modelo | Algoritmo | Variables | F1 CV-5 | Precisión | Recall | AUC-ROC | Threshold |
|---------|--------|-----------|-----------|---------|-----------|--------|---------|-----------|
| 1 | XGB_008 | XGBoost | 75 | 0.739 | 0.698 | 0.785 | 0.941 | 0.37 |
| 2 | RF_006 | Random Forest | 92 | 0.722 | 0.669 | 0.783 | 0.931 | 0.50 |
| 3 | EN_001A | Elastic Net | 91 | 0.698 | 0.652 | 0.751 | 0.921 | 0.36 |
| 4 | LPM_003A | LPM | 81 | 0.685 | 0.626 | 0.757 | 0.914 | 0.61 |
| 5 | CART_RF_top60 | CART | 60 | 0.674 | 0.671 | 0.678 | 0.887 | 0.29 |
| 6 | NB_RF_top20 | Naive Bayes | 20 | 0.562 | 0.421 | 0.842 | 0.861 | 0.95 |

## Mapeo: Scripts a Figuras de las Slides

### Deck 1 (Best Model Deep Dive): XGB_008

| Slide | Figura/Tabla | Fuente |
|-------|-------------|--------|
| Resumen del mejor modelo | Especificación + F1 Kaggle | `02_outputs/model_registry.csv` |
| Datos: desbalance de clases | Gráfico de barras 80/20 | `02_outputs/figures/data_explore/01_desbalance_clases.png` |
| Datos: pobreza por zona | Urbano vs rural | `02_outputs/figures/data_explore/02_pobreza_zona.png` |
| Datos: correlaciones clave | Top 15 correlaciones con pobreza | `02_outputs/figures/data_explore/08_correlaciones_pobre.png` |
| Mejor modelo: importancia de variables | Gráfico de importancia | `02_outputs/models/XGBoost/XGB_008/varimp.png` |
| Mejor modelo: tuning de HP | Comparación de configuraciones | `02_outputs/models/XGBoost/XGB_008/comparison_specs.png` |
| Mejor modelo: threshold | Curva F1 vs threshold | `02_outputs/models/XGBoost/XGB_008/threshold.png` |
| Mejor modelo: ROC | Curva ROC | `02_outputs/models/XGBoost/XGB_008/roc.png` |
| Implicaciones de política | Trade-off precisión-recall | `02_outputs/models/XGBoost/XGB_008/prcurve.png` |

### Deck 2 (Comparación de Algoritmos)

| Slide | Figura/Tabla | Fuente |
|-------|-------------|--------|
| Resumen del mejor modelo | Especificación + F1 Kaggle | `02_outputs/model_registry.csv` |
| Datos: educación vs pobreza | Barras apiladas por nivel educativo | `02_outputs/figures/data_explore/04_educ_jefe_vs_pobre.png` |
| Datos: formalidad vs pobreza | Boxplot régimen contributivo | `02_outputs/figures/data_explore/05_formalidad_vs_pobre.png` |
| Comparación: tabla resumen | F1/Precisión/Recall de los 6 modelos | `02_outputs/model_registry.csv` |
| Comparación: XGB vs RF | ROC, threshold, importancia de variables | `02_outputs/models/XGBoost/XGB_008/`, `02_outputs/models/RandomForest/RF_006/` |
| Conclusión: trade-off de política | Interpretabilidad vs rendimiento | Síntesis de todos los modelos |

## Decisiones de Diseño

1. **Bogotá excluida del entrenamiento**: Bogotá está presente solo en train, no en test. Incluirla crearía un desfase de niveles al momento de predecir. Se eliminaron 10,567 observaciones (6.4%).

2. **Feature engineering basado en literatura económica**: Se construyeron más de 40 variables adicionales, incluyendo ratios de dependencia, índices de formalidad laboral, medidas de hacinamiento y términos de interacción.

3. **Optimización de threshold**: Todos los modelos optimizan el umbral de clasificación sobre predicciones out-of-fold para maximizar F1, en lugar de usar el valor por defecto de 0.5.

4. **Desbalance de clases**: Se evaluaron cinco técnicas de balanceo (baseline, class_weights, downsample, upsample, SMOTE) de forma consistente en todos los algoritmos. El baseline ganó en la mayoría de los casos; solo LPM se benefició de class_weights.

5. **Selección de variables**: El ranking de importancia por permutación del RF_001 se usó para crear subconjuntos (top-20/40/60) evaluados en todos los algoritmos.

## Competencia Kaggle

- Competencia: Poverty Prediction Challenge (MECA 4107)
- Mejor F1 público: XGB_008
