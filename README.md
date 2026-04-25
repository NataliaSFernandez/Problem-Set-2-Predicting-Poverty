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
- Los paquetes necesarios se instalan automáticamente al ejecutar el pipeline.

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

Esto ejecuta todos los scripts de forma secuencial: limpieza de datos, análisis exploratorio y los seis modelos de clasificación. Tiempo total aproximado: 60 minutos.

3. Todos los resultados (figuras, métricas, submissions) se regeneran en `02_outputs/` y `03_submissions/`.

## Estructura del Repositorio

```
Problem-Set-2-Predicting-Poverty/
|
|-- 00_data/
|   |-- raw/                          # Datos crudos de Kaggle
|   |-- processed/                    # Datos limpios (generados por el pipeline)
|
|-- 01_code/
|   |-- 00_rundirectory.R             # Script maestro: ejecuta todo el pipeline
|   |-- 01_data_clean_and_merge.R     # Limpieza, recodificación, agregación
|   |-- 02_data_explore.R             # Análisis exploratorio de datos
|   |-- 03_models/
|       |-- LPM/Balance/LPM_003A.R           # Mejor LPM (class_weights, 81 vars)
|       |-- ElasticNet/Baseline/ElasticNet.R  # Mejor EN (baseline, 91 vars)
|       |-- NaiveBayes/Features/NaiveBayes_feature.R  # Mejor NB (top-20 vars RF)
|       |-- CARET/Features/CART_cv5_feat.R    # Mejor CART (top-40 vars RF)
|       |-- RandomForest/RF_006.R             # Mejor RF (HP tuning + balance, 92 vars)
|       |-- XGBoost/XGB_007.R                 # Mejor XGB (92 vars, sin Bogotá)
|
|-- 02_outputs/
|   |-- figures/data_explore/         # Figuras del EDA
|   |-- models/                       # Métricas, diagnósticos y gráficos por modelo
|   |-- model_registry.csv            # Tabla resumen de todos los modelos
|
|-- 03_submissions/                   # Archivos CSV de predicciones para Kaggle
```

## Descripción del Pipeline

| Paso | Script | Descripción | Tiempo |
|------|--------|-------------|--------|
| 1 | `01_data_clean_and_merge.R` | Limpieza, recodificación, agregación de personas a hogar, feature engineering, exclusión de Bogotá. Resultado: 154,393 hogares, 110 variables. | 1 min |
| 2 | `02_data_explore.R` | Desbalance de clases (80/20), pobreza por zona y departamento, correlaciones. 9 figuras PNG. | <1 min |
| 3.1 | `LPM_003A.R` | LPM con class_weights, 81 variables. | 1 min |
| 3.2 | `ElasticNet.R` | Elastic Net (glmnet, grilla alpha/lambda, threshold optimizado). | 9 min |
| 3.3 | `NaiveBayes_feature.R` | Naive Bayes con selección de variables RF (top-20/40/60). | 3 min |
| 3.4 | `CART_cv5_feat.R` | CART con CV anidado (5x3) y selección de variables RF. | 21 min |
| 3.5 | `RF_006.R` | Random Forest (ranger, HP tuning, 5 técnicas de balanceo). | 22 min |
| 3.6 | `XGB_007.R` | XGBoost (búsqueda aleatoria de HP, sin Bogotá). | 4 min |
| | **Total** | | **~60 min** |

## Comparación de Modelos

| Ranking | Modelo | Algoritmo | Variables | F1 CV-5 | Precisión | Recall | AUC-ROC | Kaggle F1 |
|---------|--------|-----------|-----------|---------|-----------|--------|---------|-----------|
| 1 | XGB_007 | XGBoost | 93 | 0.741 | 0.706 | 0.779 | 0.942 | 0.73 |
| 2 | RF_006 | Random Forest | 93 | 0.721 | 0.669 | 0.783 | 0.931 | 0.71 |
| 3 | EN_001A | Elastic Net | 93 | 0.695 | 0.644 | 0.756 | 0.923 | 0.69 |
| 4 | LPM_003A | LPM | 98 | 0.683 | 0.617 | 0.766 | 0.916 | 0.68 |
| 5 | CART_RF_top60 | CART | 60 | 0.677 | 0.644 | 0.713 | 0.897 | 0.68 |
| 6 | NB_RF_top20 | Naive Bayes | 20 | 0.619 | 0.513 | 0.778 | 0.876 | 0.61 |

## Mapeo: Scripts a Figuras de las Slides

### Deck 1 (Best Model Deep Dive): XGB_007 — 18 slides

| Sección | Slides | Contenido | Fuente |
|---------|--------|-----------|--------|
| 01 Datos | 3-6 | Datos crudos, evidencia descriptiva, variables clave con razonamiento económico | `02_outputs/figures/data_explore/` |
| 02 Entrenamiento | 7-8 | Proceso de selección en 4 fases | Documentado en scripts de cada modelo |
| 03 Visión General | 9-10 | Especificación XGB_007, métricas F1/AUC/Precisión/Recall | `02_outputs/model_registry.csv` |
| 04 Importancia | 11-12 | Top 20 variables, interpretación económica | `02_outputs/models/XGBoost/XGB_007/varimp.png` |
| 05 Diagnóstico | 13-15 | Confusion matrix, errores FN/FP, threshold 0.38 | `02_outputs/models/XGBoost/XGB_007/threshold.png`, `roc.png` |
| 06 Política | 16-18 | Fortalezas, limitaciones, recomendación, conclusiones | Síntesis de todos los modelos |

### Deck 2 (Comparación de Algoritmos) — 21 slides

| Sección | Slides | Contenido | Fuente |
|---------|--------|-----------|--------|
| 01 Datos | 3-7 | Datos crudos, evidencia descriptiva, razonamiento económico, proceso de selección | `02_outputs/figures/data_explore/` |
| 02 Visión General | 8-9 | Especificación XGB_007 | `02_outputs/model_registry.csv` |
| 03 Resumen | 10-11 | Tabla comparativa 6 algoritmos + gráfico F1 | `02_outputs/model_registry.csv` |
| 04 Por Qué | 12-13 | 5 factores: no linealidades, dimensionalidad, desbalance, correlación, geografía | Síntesis |
| 05 XGB vs RF | 14-16 | Confusion matrix comparativa, métricas, 4 razones | `02_outputs/models/XGBoost/XGB_007/`, `02_outputs/models/RandomForest/RF_006/` |
| 06 XGB vs EN | 17-19 | Confusion matrix comparativa, métricas, ventajas de cada uno | `02_outputs/models/XGBoost/XGB_007/`, `02_outputs/models/EN/` |
| 07 Conclusión | 20-21 | Trade-off interpretabilidad vs rendimiento, recomendación | Síntesis |

## Decisiones de Diseño

1. **Bogotá excluida del entrenamiento**: presente solo en train, no en test. 10,567 observaciones eliminadas (6.4%).
2. **Feature engineering**: más de 40 variables construidas basadas en literatura económica (Mincer, Becker, Fields, Levy, World Bank).
3. **Optimización de threshold**: todos los modelos optimizan el umbral sobre predicciones out-of-fold para maximizar F1.
4. **Desbalance de clases**: 5 técnicas evaluadas consistentemente. Baseline ganó en la mayoría; solo LPM se benefició de class_weights.
5. **Selección de variables**: ranking de importancia por permutación del RF, subconjuntos top-20/40/60 evaluados en todos los algoritmos.

## Competencia Kaggle

- Mejor F1 público: XGB_007 (0.73)
- GitHub: https://github.com/NataliaSFernandez/Problem-Set-2-Predicting-Poverty
