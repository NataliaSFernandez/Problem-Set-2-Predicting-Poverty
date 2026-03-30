#==============================================================================
# PROBLEM SET 2: PREDICTING POVERTY
# Script: 01_code/03_models/LPM/LPM.R
#==============================================================================
#
# ALGORITMO  : Linear Probability Model (LPM)
#
# QUÉ HACE ESTE SCRIPT:
#   Estima una regresión OLS donde la variable dependiente es binaria (pobre =
#   1 / no pobre = 0). El modelo aprende una función lineal de las características
#   del hogar para predecir la probabilidad de ser pobre. Cada coeficiente β_j
#   es directamente interpretable como: "un incremento de una unidad en X_j
#   cambia la probabilidad de pobreza en β_j puntos porcentuales, manteniendo
#   lo demás constante". Esto hace al LPM el modelo más interpretable del
#   repertorio del problem set.
#
# LIMITACIÓN CONOCIDA:
#   El LPM puede predecir probabilidades fuera de [0,1]. En la práctica, esto
#   afecta a una fracción de observaciones extremas pero no invalida el modelo
#   para clasificación (sólo necesitamos comparar el score continuo con un
#   threshold).
#
# MULTICOLINEALIDAD DETECTADA:
#   Al correr el modelo con todas las variables, alias() reveló que 13 variables
#   son combinaciones lineales exactas de otras ya presentes en X. R descarta
#   esos coeficientes silenciosamente (rank-deficient fit), lo que hace los
#   resultados difíciles de interpretar. Para corregirlo, este script corre
#   DOS especificaciones alternativas:
#
#   LPM_002A — Sin variables compuestas: se eliminan los 13 derivados y se
#              conservan sus componentes originales (ciudad, dpto, dep_*,
#              alguno_*, etc.). Máxima granularidad, modelo más grande.
#
#   LPM_002B — Con variables compuestas: se eliminan los componentes y se
#              conservan los índices agregados (zona, bogota, n_privaciones,
#              indice_formalidad, etc.). Modelo más parsimonioso.
#
#   Al final, el script compara F1-CV, Precision, Recall y AUC-ROC de ambos
#   para decidir cuál especificación conviene subir a Kaggle.
#
#   Las 13 relaciones de dependencia lineal perfecta son:
#     ciudadRURAL    = f(zona)
#     dpto11         = f(ciudadBOGOTA)
#     doble_ingreso  = 1 − sin_ocupados − un_solo_ocupado
#     brecha_educ    = nivel_educ_max − nivel_educ_jefe
#     indice_formalidad  = prop_asalariado + prop_cotiza_pension +
#                          prop_prima_serv + prop_sub_transp
#     vulnerabilidad_lab = prop_domestico + prop_fam_sin_rem + prop_jornalero
#     n_fuentes_no_lab   = suma de los 8 indicadores alguno_*
#     prop_contributivo  = prop_afil_salud + prop_subsidiado
#     costa_caribe       = suma de dummies dpto del litoral Caribe
#     region_pacifico    = suma de dummies dpto del litoral Pacífico
#     bogota             = ciudadBOGOTA
#     eje_cafetero       = suma de dummies dpto del Eje Cafetero
#     n_privaciones      = dep_educacion + dep_vivienda + dep_empleo_formal +
#                          dep_proteccion + dep_dependencia
#
# FLUJO DEL SCRIPT:
#   0. Configuración y paquetes
#   1. Identificación de los dos modelos (rellenar AUTOR antes de correr)
#   2. Cargar datos
#   3. Listas de variables a excluir (Opción A y Opción B)
#   4. Generar folds de CV (una sola vez para que la comparación sea justa)
#   5. Función run_lpm() — pipeline completo reutilizable
#   6. Ejecutar modelo A y modelo B
#   7. Comparación final de métricas
#   8. Ideas para la próxima iteración (comentadas, no ejecutar)
#
# INPUTS (relativos a la raíz del proyecto):
#   00_data/processed/train_final.rds
#   00_data/processed/test_final.rds
#
# OUTPUTS (relativos a la raíz del proyecto):
#   03_submissions/submission_LPM_002A.csv        (para subir a Kaggle)
#   03_submissions/submission_LPM_002B.csv        (para subir a Kaggle)
#   02_outputs/models/LPM/LPM_002A/coefs.png
#   02_outputs/models/LPM/LPM_002A/threshold.png
#   02_outputs/models/LPM/LPM_002A/prcurve.png
#   02_outputs/models/LPM/LPM_002A/roc.png
#   02_outputs/models/LPM/LPM_002A/diagnostics.rds
#   02_outputs/models/LPM/LPM_002B/  (mismos archivos)
#   02_outputs/model_registry.csv                 ← tabla maestra (append)
#
# LÓGICA DE ESTE SCRIPT:
#   El pipeline completo (CV → threshold → modelo final → gráficos →
#   submission → registro) está encapsulado en la función run_lpm(). Eso
#   permite correr dos especificaciones distintas sin duplicar código.
#   Los folds de CV se generan UNA SOLA VEZ antes de llamar a run_lpm(),
#   garantizando que ambos modelos sean evaluados sobre exactamente los
#   mismos subconjuntos de datos y que la comparación de F1 sea justa.
#
#   [ANTES de correr]
#     Sección 1 → editar AUTOR.
#     Los MODEL_ID están fijos: LPM_002A y LPM_002B.
#
#   [DESPUÉS de correr]
#     Subir ambos CSVs a Kaggle y anotar el public F1 manualmente en
#     02_outputs/model_registry.csv (columna kaggle_public_F1).
#
# REPRODUCIBILIDAD:
#   Correr desde la raíz del proyecto. Semilla global: 42.
#==============================================================================


# =============================================================================
# SECCIÓN 0: PAQUETES Y CONFIGURACIÓN GLOBAL
# =============================================================================
# pacman::p_load() instala el paquete si no está instalado, luego lo carga.
# Usarlo en lugar de install.packages() + library() garantiza que el script
# sea autocontenido: cualquier persona puede ejecutarlo sin configuración previa.

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,  # dplyr (manipulación), ggplot2 (gráficos), readr (CSV), tibble
  caret,      # confusionMatrix(): calcula F1, precision, recall con una sola llamada
  pROC,       # roc(): construye la curva ROC para calcular AUC como diagnóstico
  fs          # dir_create(): crea carpetas de forma multiplataforma (Mac/Linux/Windows)
)

# Semilla global — TODOS los scripts del equipo deben usar 42
# Garantiza que los folds de CV sean idénticos si alguien corre el script de nuevo
set.seed(42)

# ---------- Rutas (relativas a la raíz del proyecto) -------------------------
# Si lanzas el script con source() desde otro directorio, ajusta estas rutas.
# Si usas RStudio Projects, la raíz es automáticamente la carpeta del proyecto.

dir_processed   <- "00_data/processed"       # donde están train_final.rds y test_final.rds
dir_outputs_lpm <- "02_outputs/models/LPM"   # carpeta base del algoritmo LPM
dir_submissions <- "03_submissions"           # CSVs listos para subir a Kaggle
registry_path   <- "02_outputs/model_registry.csv"  # tabla maestra de todos los modelos

# Crear directorios si no existen (no falla si ya existen)
fs::dir_create(dir_outputs_lpm, recurse = TRUE)
fs::dir_create(dir_submissions, recurse = TRUE)
# garantiza que 02_outputs/ exista para el registry
fs::dir_create("02_outputs", recurse = TRUE)


# =============================================================================
# SECCIÓN 1: IDENTIFICACIÓN DE LOS DOS MODELOS
# =============================================================================
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  >>>  MODIFICAR ESTOS CAMPOS ANTES DE CORRER EL SCRIPT  <<<              │
# │                                                                          │
# │  AUTOR:   nombre del autor (para el registro compartido del equipo).     │
# │  K_FOLDS: número de folds de CV (no cambiar salvo razón justificada).   │
# │                                                                          │
# │  Los MODEL_ID están fijos: LPM_002A y LPM_002B.                          │
# │  No modificarlos — identifican unívocamente las dos especificaciones.   │
# └──────────────────────────────────────────────────────────────────────────┘

AUTOR   <- "Jonathan"  # <<< MODIFICAR
K_FOLDS <- 5           # 5 folds → cada fold valida sobre ~32,992 hogares


# =============================================================================
# SECCIÓN 2: CARGAR DATOS
# =============================================================================
# Cargamos los .rds porque son más rápidos que los .csv y preservan los tipos
# de variables (factores, enteros, dobles) exactamente como los dejó script 01.
#
# train_final: 164,960 hogares, 115 columnas (114 predictores + 'pobre')
# test_final:  sin 'pobre' — es lo que debemos predecir para Kaggle

train <- readRDS(file.path(dir_processed, "train_final.rds"))
test  <- readRDS(file.path(dir_processed, "test_final.rds"))

# Información diagnóstica básica
message("━━━ Datos cargados ━━━")
message("Train: ", nrow(train), " hogares | ", ncol(train), " columnas")
message("Test:  ", nrow(test),  " hogares | ", ncol(test),  " columnas")
message("Balance: ", round(mean(train$pobre) * 100, 1), "% pobres  /  ",
        round(mean(1 - train$pobre) * 100, 1), "% no pobres")
# Con ~20% pobres el dataset está desbalanceado (ratio 1:4).
# Esto importa: un modelo que clasifica TODO como no-pobre tiene accuracy=80%
# pero F1=0. Por eso usamos F1 como métrica, no accuracy.

# y_train se define aquí porque es compartido por ambas especificaciones:
# ambos modelos predicen la misma variable sobre el mismo conjunto de hogares.
y_train <- train$pobre   # vector numérico 0/1 — lm() necesita numérico


# =============================================================================
# SECCIÓN 3: LISTAS DE EXCLUSIÓN DE VARIABLES
# =============================================================================
#
# ¿POR QUÉ DOS LISTAS DISTINTAS?
#   alias() reveló que 13 variables son combinaciones lineales exactas de otras.
#   Hay dos formas simétricas de resolver la multicolinealidad:
#
#   Opción A: eliminar las variables del lado izquierdo del alias (los derivados/
#             compuestos). El modelo conserva los componentes originales con la
#             máxima granularidad, pero tiene más predictores.
#
#   Opción B: eliminar las variables del lado derecho del alias (los componentes).
#             El modelo conserva los índices agregados. Es más parsimonioso y
#             los coeficientes son más fáciles de interpretar, pero pierde la
#             heterogeneidad dentro de cada grupo.
#
# En ambas opciones el punto de partida es el mismo:
#   'id'    → identificador sin valor predictivo.
#   'pobre' → variable a predecir; no existe en test, así que select() la
#             ignora automáticamente en X_test si se excluye aquí.
#
# ---------- OPCIÓN A: conservar componentes, eliminar derivados ---------------
#
# Se eliminan las 13 variables que alias() identificó como combinaciones
# lineales de otras. El modelo queda con:
#   • factor ciudad  (captura la distinción rural/urbano Y la ciudad específica)
#   • factor dpto    (captura todos los departamentos con sus propias dummies)
#   • sin_ocupados, un_solo_ocupado (los dos componentes de doble_ingreso)
#   • nivel_educ_max, nivel_educ_jefe (los dos términos de brecha_educ)
#   • prop_asalariado, prop_cotiza_pension, prop_prima_serv, prop_sub_transp
#     (los cuatro componentes de indice_formalidad)
#   • prop_domestico, prop_fam_sin_rem, prop_jornalero
#     (los tres componentes de vulnerabilidad_lab)
#   • los ocho alguno_* (componentes de n_fuentes_no_lab)
#   • prop_afil_salud, prop_subsidiado (componentes de prop_contributivo)
#   • dep_educacion, dep_vivienda, dep_empleo_formal, dep_proteccion,
#     dep_dependencia (los cinco componentes de n_privaciones)

VARS_EXCLUIR_A <- c(
  "id", "pobre",
  # zona es redundante: el nivel RURAL del factor ciudad (ciudadRURAL) ya
  # captura la misma información que zona. Al eliminar zona, lm() no genera
  # la combinación lineal que causaba el alias.
  "zona",
  # bogota es un indicador binario que duplica exactamente a ciudadBOGOTA
  # (nivel del factor ciudad) y a dpto11 (nivel del factor dpto). Con el
  # factor ciudad y dpto ya en el modelo, bogota es completamente redundante.
  "bogota",
  # doble_ingreso fue construido como 1 - sin_ocupados - un_solo_ocupado.
  # Al eliminarla, conservamos sus dos componentes originales del mercado laboral.
  "doble_ingreso",
  # brecha_educ fue construida como nivel_educ_max - nivel_educ_jefe.
  # Al eliminarla, conservamos ambos niveles de educación por separado.
  "brecha_educ",
  # indice_formalidad es la suma de cuatro proporciones laborales que ya
  # están en el modelo como variables independientes. Eliminarla evita que
  # esa suma lineal cause el alias sin perder ninguna información.
  "indice_formalidad",
  # vulnerabilidad_lab es la suma de tres proporciones de ocupación precaria.
  # Sus componentes (prop_domestico, prop_fam_sin_rem, prop_jornalero)
  # permanecen en el modelo.
  "vulnerabilidad_lab",
  # n_fuentes_no_lab cuenta cuántos de los 8 indicadores alguno_* son TRUE.
  # Al eliminarla, los 8 indicadores individuales permanecen, lo que permite
  # que cada fuente de ingreso no laboral tenga su propio efecto.
  "n_fuentes_no_lab",
  # prop_contributivo = prop_afil_salud + prop_subsidiado. Eliminarla conserva
  # la distinción entre ambos tipos de afiliación al sistema de salud.
  "prop_contributivo",
  # Las tres dummies regionales son sumas exactas de subconjuntos de dummies
  # del factor dpto, que ya está en el modelo. Eliminarlas es la única forma
  # de resolver el alias sin perder las dummies departamentales.
  "costa_caribe",
  "region_pacifico",
  "eje_cafetero",
  # n_privaciones es la suma de los 5 indicadores dep_* de privación del IPM.
  # Al eliminarla, cada dimensión de privación tiene su propio coeficiente.
  "n_privaciones",
  # estrato_hog, estrato_bajo y estrato_x_zona son NA en el 100% de las filas
  # de test (estrato no está disponible en test_personas.csv). Excluirlas evita
  # que predict() devuelva NA para todo el test.
  "estrato_hog", "estrato_bajo", "estrato_x_zona"
)

# ---------- OPCIÓN B: conservar índices compuestos, eliminar componentes ------
#
# Se eliminan los componentes granulares y se conservan los índices agregados.
# El modelo queda con:
#   • zona           (rural/urbano — captura la distinción geográfica clave)
#   • bogota         (indicador binario de Bogotá)
#   • costa_caribe, region_pacifico, eje_cafetero (dummies regionales)
#   • doble_ingreso  (indicador de hogar con dos o más ocupados)
#   • brecha_educ    (brecha de educación entre el máximo del hogar y el jefe)
#   • indice_formalidad, vulnerabilidad_lab (índices del mercado laboral)
#   • n_fuentes_no_lab (conteo de fuentes de ingreso no laboral)
#   • prop_contributivo (fracción con seguridad social contributiva)
#   • n_privaciones  (conteo de dimensiones IPM con privación)
#
# NOTA: al eliminar 'ciudad' y 'dpto' (factores con muchos niveles) y
# reemplazarlos por zona + dummies regionales + bogota, el modelo pierde
# la heterogeneidad intra-región pero gana parsimonia. Algunos departamentos
# no cubiertos por las dummies regionales quedan sin identificación geográfica
# propia — esto es una limitación conocida de esta especificación.

VARS_EXCLUIR_B <- c(
  "id", "pobre",
  # ciudad tiene muchos niveles (una dummy por ciudad). Su nivel RURAL es
  # redundante con zona; su nivel BOGOTA es redundante con bogota. Al eliminar
  # ciudad, zona y bogota cubren la distinción rural/urbano y la capital.
  "ciudad",
  # dpto tiene ~33 niveles (una dummy por departamento). Las dummies regionales
  # (costa_caribe, region_pacifico, eje_cafetero) y bogota ya capturan la
  # variación geográfica de forma más agregada. Al eliminar dpto, resolvemos
  # los alias con costa_caribe, region_pacifico, eje_cafetero y bogota.
  "dpto",
  # sin_ocupados y un_solo_ocupado son los dos componentes de doble_ingreso.
  # doble_ingreso = 1 - sin_ocupados - un_solo_ocupado, así que basta con
  # conservar doble_ingreso y eliminar sus dos componentes.
  "sin_ocupados",
  "un_solo_ocupado",
  # nivel_educ_max y nivel_educ_jefe son los dos sumandos de brecha_educ.
  # Nota: nivel_educ_prom permanece en el modelo porque no entra en el alias.
  "nivel_educ_max",
  "nivel_educ_jefe",
  # Los cuatro componentes de indice_formalidad. Al eliminarlos, conservamos
  # el índice que los resume en una sola dimensión de formalidad laboral.
  "prop_asalariado",
  "prop_cotiza_pension",
  "prop_prima_serv",
  "prop_sub_transp",
  # Los tres componentes de vulnerabilidad_lab. Al eliminarlos, conservamos
  # el índice que los agrega en una sola dimensión de precariedad laboral.
  "prop_domestico",
  "prop_fam_sin_rem",
  "prop_jornalero",
  # Los ocho indicadores binarios de ingresos no laborales que suman n_fuentes_no_lab.
  # Al eliminarlos, n_fuentes_no_lab captura cuántas fuentes tiene el hogar
  # sin distinguir cuál fuente específica tiene.
  "alguno_pension_jub",
  "alguno_remesas",
  "alguno_subsidio",
  "alguno_transf_nac",
  "alguno_intereses",
  "alguno_cesantias",
  "alguno_agropec",
  "alguno_arriendos",
  # Los dos componentes de prop_contributivo. Al eliminarlos, prop_contributivo
  # captura la fracción del hogar con seguridad social contributiva sin
  # distinguir entre subsidiados y afiliados directos.
  "prop_afil_salud",
  "prop_subsidiado",
  # Los cinco indicadores de privación del IPM. Al eliminarlos, n_privaciones
  # captura cuántas dimensiones tiene el hogar sin el efecto de cada dimensión.
  "dep_educacion",
  "dep_vivienda",
  "dep_empleo_formal",
  "dep_proteccion",
  "dep_dependencia",
  # estrato_hog, estrato_bajo y estrato_x_zona son NA en el 100% de las filas
  # de test (estrato no está disponible en test_personas.csv). Excluirlas evita
  # que predict() devuelva NA para todo el test.
  "estrato_hog", "estrato_bajo", "estrato_x_zona"
)


# =============================================================================
# SECCIÓN 4: GENERAR FOLDS DE CV (una sola vez)
# =============================================================================
#
# ¿POR QUÉ GENERAR LOS FOLDS AQUÍ Y NO DENTRO DE run_lpm()?
#   Si cada llamada a run_lpm() generara sus propios folds, los dos modelos
#   serían evaluados sobre particiones distintas del train, y una diferencia
#   en F1 podría deberse simplemente a que un modelo tuvo "suerte" con sus
#   folds (más hogares pobres en validación, etc.) y no a su mejor capacidad
#   predictiva.
#
#   Al generar fold_ids aquí, ambos modelos comparten exactamente los mismos
#   subconjuntos de entrenamiento y validación en cada fold. Cualquier
#   diferencia en F1 refleja diferencias reales en la especificación del modelo,
#   no varianza de muestreo de los folds.
#
# IMPLEMENTACIÓN:
#   rep(1:5, length.out = n) crea un vector 1,2,3,4,5,1,2,3,... del tamaño
#   exacto del train. Luego sample() lo permuta aleatoriamente con semilla 42.

fold_ids <- sample(rep(1:K_FOLDS, length.out = nrow(train)))

message("\nFolds de CV generados (semilla 42) — mismos folds para LPM_002A y LPM_002B.")


# =============================================================================
# SECCIÓN 5: FUNCIÓN run_lpm()
# =============================================================================
#
# ¿POR QUÉ UNA FUNCIÓN?
#   Sin la función, el pipeline (secciones 3-11 del script original) tendría
#   que duplicarse íntegro para la segunda especificación. Eso crearía ~400
#   líneas de código casi idénticas, difíciles de mantener: si se corrige un
#   bug o se agrega un gráfico, habría que hacerlo dos veces.
#
#   La función recibe los tres elementos que varían entre especificaciones
#   (MODEL_ID, NOTAS, VARS_EXCLUIR) y accede al resto de las variables
#   del entorno global (train, test, y_train, fold_ids, K_FOLDS, AUTOR,
#   las rutas de directorios). En R, las funciones buscan variables no
#   declaradas como parámetros en el entorno desde donde fueron definidas
#   (lexical scoping), así que no es necesario pasarlas explícitamente.
#
# PARÁMETROS:
#   model_id    : string identificador único (ej. "LPM_002A")
#   notas       : descripción breve de la especificación (para el registry)
#   vars_excluir: vector de nombres de columnas a excluir de X
#
# RETORNA:
#   Un tibble con una fila y las métricas clave del modelo (F1, Precision,
#   Recall, AUC-ROC, threshold, n_features). Este tibble es usado en la
#   Sección 7 para construir la tabla comparativa.

run_lpm <- function(model_id, notas, vars_excluir) {

  message("\n", strrep("═", 62))
  message("  INICIANDO MODELO: ", model_id)
  message(strrep("═", 62))

  # Carpeta de outputs de esta corrida específica (subcarpeta por MODEL_ID)
  dir_model <- file.path(dir_outputs_lpm, model_id)
  fs::dir_create(dir_model, recurse = TRUE)


  # ---------------------------------------------------------------------------
  # PASO 1: PREPARAR LAS MATRICES X y Y
  # ---------------------------------------------------------------------------
  # Usamos any_of() en lugar de all_of() para que select() no falle si alguna
  # variable de VARS_EXCLUIR no existe en el dataset (p. ej. si en el futuro
  # se llama a run_lpm() con una lista de exclusión que contiene variables que
  # ya no están en los datos).
  #
  # 'pobre' no está en test, así que select(-any_of(vars_excluir)) la ignora
  # automáticamente en X_test — no hay que tratarla por separado.

  X_train <- train |> select(-any_of(vars_excluir))
  X_test  <- test  |> select(-any_of(vars_excluir))

  message("\n━━━ Matrices preparadas (", model_id, ") ━━━")
  message("Predictores: ", ncol(X_train))
  message("NAs en X_train: ", sum(is.na(X_train)),
          " (esperamos 0 — datos ya limpios)")
  message("NAs en X_test:  ", sum(is.na(X_test)))

  # Diagnóstico de tipos de columna: lm() convierte factores a dummies automáticamente
  tipos <- X_train |> summarise(across(everything(), ~ class(.)[1]))
  cat("\nTipos de columnas:\n")
  print(table(unlist(tipos)))


  # ---------------------------------------------------------------------------
  # PASO 2: CROSS-VALIDATION (k-fold manual)
  # ---------------------------------------------------------------------------
  #
  # ¿POR QUÉ HACEMOS CV?
  # Si entrenamos el modelo en todos los datos y medimos el error en esos
  # mismos datos, estaremos sobreestimando el rendimiento (overfitting).
  # La CV divide los datos en k partes: usamos k-1 para entrenar y 1 para
  # validar, rotando las partes. Así obtenemos predicciones "out-of-fold" (OOF)
  # para cada hogar, sin que el modelo haya visto ese hogar durante el
  # entrenamiento. Estas predicciones OOF son la mejor estimación del
  # rendimiento real del modelo en datos nuevos.
  #
  # NOTA SOBRE suppressWarnings():
  #   Con la multicolinealidad resuelta, lm() no debería emitir el warning de
  #   rank-deficient fit. suppressWarnings() se mantiene como salvaguarda por
  #   si algún fold (por azar de muestreo) genera una combinación lineal
  #   inesperada entre variables que no son alias perfectos en el dataset
  #   completo pero sí en un subconjunto del 80%.

  # Contenedor para las predicciones out-of-fold
  # Inicializamos con NA para detectar fácilmente si algún fold no se completó
  oof_probs <- rep(NA_real_, nrow(X_train))

  message("\n━━━ Cross-Validation (", K_FOLDS, " folds) — ", model_id, " ━━━")

  for (k in seq_len(K_FOLDS)) {

    # Índices del fold de validación (20%) y de entrenamiento (80%)
    idx_val   <- which(fold_ids == k)
    idx_train <- which(fold_ids != k)

    # Construir el data.frame de entrenamiento uniendo X y Y
    # lm() necesita que los predictores y el outcome estén en el mismo data.frame
    df_tr <- cbind(X_train[idx_train, ], pobre = y_train[idx_train])
    X_vl  <- X_train[idx_val, ]

    # Ajustar el LPM: OLS con pobre ~ . (el punto significa "todos los predictores")
    modelo_k <- suppressWarnings(lm(pobre ~ ., data = df_tr))

    # Predecir probabilidades en el fold de validación
    # type = "response" no existe en lm(); predict.lm() devuelve valores
    # continuos directamente. Pueden ser < 0 o > 1 (limitación conocida del LPM).
    probs_k <- suppressWarnings(predict(modelo_k, newdata = X_vl))

    # Guardar en el vector out-of-fold en las posiciones correctas
    oof_probs[idx_val] <- probs_k

    message(sprintf("  Fold %d/%d: rango predicciones [%.3f, %.3f]",
                    k, K_FOLDS,
                    min(probs_k, na.rm = TRUE),
                    max(probs_k, na.rm = TRUE)))
  }

  # Verificar que ningún valor quedó en NA (todos los folds se completaron)
  if (any(is.na(oof_probs))) stop("Hay NAs en oof_probs. Revisar el loop de CV.")

  # Diagnóstico de predicciones fuera del rango [0,1]
  cat(sprintf("\n━━━ Diagnóstico de predicciones OOF — %s ━━━\n", model_id))
  print(summary(oof_probs))
  cat(sprintf("Predicciones < 0:  %d  (%.1f%% del total)\n",
              sum(oof_probs < 0), mean(oof_probs < 0) * 100))
  cat(sprintf("Predicciones > 1:  %d  (%.1f%% del total)\n",
              sum(oof_probs > 1), mean(oof_probs > 1) * 100))

  # Clampear a [0,1] para que puedan usarse como probabilidades en el threshold
  # Esto no cambia la clasificación para la mayoría de observaciones, sólo
  # "recorta" los extremos más allá del rango válido de una probabilidad
  oof_probs_cl <- pmax(pmin(oof_probs, 1), 0)


  # ---------------------------------------------------------------------------
  # PASO 3: OPTIMIZACIÓN DEL THRESHOLD
  # ---------------------------------------------------------------------------
  #
  # ¿QUÉ ES EL THRESHOLD?
  #   El modelo predice un score continuo (probabilidad estimada). Para
  #   clasificar en Pobre/NoPobre necesitamos un punto de corte: si la
  #   probabilidad >= threshold → clasificamos como Pobre.
  #
  # ¿POR QUÉ NO USAR SIEMPRE 0.5?
  #   Con clases desbalanceadas (20% pobres), el modelo tiende a predecir
  #   probabilidades bajas para todos los hogares. El threshold que maximiza
  #   F1 suele ser < 0.5 para capturar más verdaderos positivos (pobres reales).
  #
  # PROCEDIMIENTO:
  #   Para cada threshold candidato (de 0.05 a 0.95 en pasos de 0.01),
  #   aplicamos ese threshold a las predicciones OOF y calculamos F1.
  #   Elegimos el threshold con el F1 más alto.
  #
  # IMPORTANTE: esto se hace sobre las predicciones OOF, no sobre las predicciones
  #   del modelo final, para evitar sobreajuste al threshold.

  th_grid <- seq(0.05, 0.95, by = 0.01)

  # Para cada threshold, calcular F1, Precision y Recall
  th_results <- map_dfr(th_grid, function(th) {

    # Clasificar con este threshold
    pred_class <- factor(
      if_else(oof_probs_cl >= th, "Pobre", "NoPobre"),
      levels = c("NoPobre", "Pobre")   # NoPobre = referencia, Pobre = positivo
    )
    # Etiqueta real en el mismo formato factor
    obs_class <- factor(
      if_else(y_train == 1, "Pobre", "NoPobre"),
      levels = c("NoPobre", "Pobre")
    )

    # confusionMatrix calcula automáticamente TP, FP, TN, FN y las métricas
    cm <- confusionMatrix(pred_class, obs_class, positive = "Pobre")

    tibble(
      threshold = th,
      F1        = cm$byClass["F1"],
      Precision = cm$byClass["Precision"],   # TP / (TP + FP) — cuántos predichos pobres son pobres
      Recall    = cm$byClass["Recall"]       # TP / (TP + FN) — cuántos pobres reales capturo
    )
  })

  # Fila con el threshold que maximiza F1
  best_th_row <- th_results |>
    filter(!is.na(F1)) |>
    slice_max(F1, n = 1, with_ties = FALSE)   # si empatan, queda el primero

  THRESHOLD_OPT <- best_th_row$threshold
  THRESHOLD     <- THRESHOLD_OPT   # usar el óptimo de CV

  cat(sprintf("\n━━━ Threshold óptimo sobre predicciones CV — %s ━━━\n", model_id))
  print(best_th_row)

  # Gráfico: cómo cambian F1, Precision y Recall con el threshold
  # Útil para la slide de "policy implications":
  #   - Moverse a la izquierda (umbral más bajo) = mayor recall = menos exclusión
  #   - Moverse a la derecha (umbral más alto)   = mayor precision = menos filtración
  p_threshold <- th_results |>
    filter(!is.na(F1)) |>
    pivot_longer(cols = c(F1, Precision, Recall), names_to = "metrica") |>
    ggplot(aes(x = threshold, y = value, color = metrica)) +
    geom_line(linewidth = 0.9) +
    # Línea vertical: threshold óptimo para F1
    geom_vline(xintercept = THRESHOLD_OPT,
               linetype = "dashed", color = "gray30") +
    # Línea vertical: threshold naive = 0.5 (para comparación)
    geom_vline(xintercept = 0.50,
               linetype = "dotted", color = "gray60") +
    annotate("text", x = THRESHOLD_OPT + 0.02, y = 0.08,
             label = sprintf("th* = %.2f\n(max F1)", THRESHOLD_OPT),
             hjust = 0, size = 3.2, color = "gray25") +
    annotate("text", x = 0.50 + 0.02, y = 0.20,
             label = "th = 0.5\n(naive)",
             hjust = 0, size = 3.2, color = "gray50") +
    scale_color_manual(
      values = c(F1 = "#534AB7", Precision = "#E84855", Recall = "#2196F3"),
      labels = c(
        F1        = "F1 (balance)",
        Precision = "Precision (↑ = menos filtración)",
        Recall    = "Recall (↑ = menos exclusión)"
      )
    ) +
    labs(
      title    = paste0("LPM: métricas de clasificación por threshold — ", model_id),
      subtitle = sprintf("%d-fold CV out-of-fold | n = %d hogares", K_FOLDS, nrow(X_train)),
      x        = "Threshold de clasificación",
      y        = "Valor de la métrica",
      color    = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top")

  ggsave(
    file.path(dir_model, "threshold.png"),
    p_threshold, width = 8, height = 5, dpi = 150
  )
  message("Gráfico guardado: ", model_id, "/threshold.png")


  # ---------------------------------------------------------------------------
  # PASO 4: MODELO FINAL (ajustado en todo el training set)
  # ---------------------------------------------------------------------------
  # La CV nos dio el F1 esperado en datos nuevos. Ahora ajustamos el modelo sobre
  # TODOS los 164,960 hogares para maximizar la información usada en predicción.
  # Este modelo final es el que usamos para predecir en test.
  #
  # Nota: el threshold elegido ya fue validado sobre datos que el modelo final
  # no vio (los OOF de CV). No hay data leakage.

  message("\n━━━ Ajustando modelo final (todo el training set) — ", model_id, " ━━━")

  df_train_full <- cbind(X_train, pobre = y_train)
  model_final   <- suppressWarnings(lm(pobre ~ ., data = df_train_full))

  # Resumen compacto del modelo
  s <- summary(model_final)
  cat(sprintf("\n━━━ Bondad de ajuste del modelo final — %s ━━━\n", model_id))
  cat(sprintf("R²:           %.4f  (fracción de varianza de 'pobre' explicada por X)\n",
              s$r.squared))
  cat(sprintf("R² ajustado:  %.4f  (penalizado por número de predictores)\n",
              s$adj.r.squared))
  cat(sprintf("Coeficientes: %d  (incluye dummies de factores)\n",
              length(coef(model_final))))
  cat(sprintf("Residual SE:  %.4f\n", s$sigma))
  # Nota: el R² del LPM tiende a ser bajo (~0.3–0.4) porque la variable
  # dependiente es binaria y la varianza teórica está acotada. Esto no indica
  # mal ajuste — comparar con otros modelos para contextualizar.


  # ---------------------------------------------------------------------------
  # PASO 5: ANÁLISIS DE RESULTADOS
  # ---------------------------------------------------------------------------

  # ─── 5.1  Coeficientes más importantes ─────────────────────────────────────
  # En el LPM, β_j = efecto marginal de X_j sobre P(pobre).
  # Un β_j = 0.10 en 'prop_subsidiado' significa: los hogares donde toda la
  # afiliación de salud es subsidiada tienen 10 pp más de probabilidad de ser
  # pobres que un hogar sin afiliación subsidiada, manteniendo todo lo demás igual.
  # Ordenamos por magnitud (|β|) para identificar los predictores más influyentes.

  coefs_df <- broom::tidy(model_final) |>
    filter(term != "(Intercept)") |>
    mutate(
      abs_estimate = abs(estimate),
      significativo = p.value < 0.05   # señal de significancia estadística
    ) |>
    arrange(desc(abs_estimate))

  cat(sprintf("\n━━━ Top 20 predictores por magnitud del efecto — %s ━━━\n", model_id))
  print(
    coefs_df |>
      slice_head(n = 20) |>
      select(term, estimate, std.error, p.value, significativo),
    n = 20
  )

  # Gráfico de coeficientes (dot-plot con intervalos de confianza al 95%)
  # Los IC se calculan como β ± 1.96 * SE (distribución asintótica normal)
  p_coefs <- coefs_df |>
    slice_head(n = 20) |>
    mutate(
      term      = fct_reorder(term, estimate),
      direccion = if_else(estimate > 0,
                          "Aumenta P(pobre)",
                          "Reduce P(pobre)")
    ) |>
    ggplot(aes(x = estimate, y = term, color = direccion)) +
    geom_point(size = 3) +
    # Barra horizontal = intervalo de confianza al 95%
    geom_errorbar(
      aes(xmin = estimate - 1.96 * std.error,
          xmax = estimate + 1.96 * std.error),
      width = 0.3,
      orientation = "y"
    ) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    scale_color_manual(
      values = c("Aumenta P(pobre)" = "#E84855", "Reduce P(pobre)" = "#2196F3")
    ) +
    labs(
      title    = paste0("LPM: efectos marginales sobre P(pobre = 1) — ", model_id),
      subtitle = "Top 20 predictores | Intervalos de confianza al 95%",
      x        = "Efecto marginal (puntos porcentuales)",
      y        = NULL,
      color    = NULL,
      caption  = "Coeficientes del modelo ajustado sobre todo el training set."
    ) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "top")

  ggsave(
    file.path(dir_model, "coefs.png"),
    p_coefs, width = 9, height = 6, dpi = 150
  )
  message("Gráfico guardado: ", model_id, "/coefs.png")

  # ─── 5.2  Confusion Matrix (predicciones OOF con threshold elegido) ─────────
  # La confusion matrix descompone los errores del modelo en cuatro tipos:
  #
  #  Verdadero Positivo (TP):  hogar POBRE  → predigo POBRE    ✓  (programa correcto)
  #  Falso Negativo    (FN):  hogar POBRE  → predigo NO POBRE ✗  (EXCLUSIÓN del programa)
  #  Falso Positivo    (FP):  hogar NO POBRE → predigo POBRE  ✗  (FILTRACIÓN al programa)
  #  Verdadero Negativo (TN):  hogar NO POBRE → predigo NO POBRE ✓ (correcto)
  #
  #  F1   = 2 × Precision × Recall / (Precision + Recall)  → métrica de Kaggle
  #  Precision = TP / (TP + FP)  → de los que predigo pobres, qué fracción lo son
  #  Recall    = TP / (TP + FN)  → de los pobres reales, qué fracción capturo

  oof_clase <- factor(
    if_else(oof_probs_cl >= THRESHOLD, "Pobre", "NoPobre"),
    levels = c("NoPobre", "Pobre")
  )
  obs_clase <- factor(
    if_else(y_train == 1, "Pobre", "NoPobre"),
    levels = c("NoPobre", "Pobre")
  )
  cm_cv <- confusionMatrix(oof_clase, obs_clase, positive = "Pobre")

  cat(sprintf("\n━━━ Confusion Matrix (CV out-of-fold, threshold = %.2f) — %s ━━━\n",
              THRESHOLD, model_id))
  print(cm_cv)

  # Extraer métricas clave
  cv_F1        <- cm_cv$byClass["F1"]
  cv_Precision <- cm_cv$byClass["Precision"]
  cv_Recall    <- cm_cv$byClass["Recall"]

  cat(sprintf("\n━━━ Resumen de métricas CV — %s ━━━\n", model_id))
  cat(sprintf("F1:             %.4f  (métrica de Kaggle)\n",         cv_F1))
  cat(sprintf("Precision:      %.4f  (¿qué tan 'limpia' es la lista?)\n", cv_Precision))
  cat(sprintf("Recall:         %.4f  (¿qué fracción de pobres captura?)\n", cv_Recall))

  # Costos de política
  FN <- cm_cv$table["NoPobre", "Pobre"]    # pobres clasificados como no-pobres
  FP <- cm_cv$table["Pobre", "NoPobre"]    # no-pobres clasificados como pobres

  cat("\n━━━ Costos de política ━━━\n")
  cat(sprintf("Falsos negativos (exclusión):  %d hogares pobres excluidos del programa\n",  FN))
  cat(sprintf("Falsos positivos (filtración): %d hogares no-pobres con beneficio indebido\n", FP))
  cat(sprintf("Tasa exclusión:   %.1f%% de los pobres reales\n", FN / sum(y_train == 1) * 100))
  cat(sprintf("Tasa filtración:  %.1f%% de los no-pobres reales\n", FP / sum(y_train == 0) * 100))

  # ─── 5.3  Curva Precision-Recall ────────────────────────────────────────────
  # Con clases desbalanceadas, la curva ROC puede ser engañosamente optimista
  # (muchos TN "inflan" la specificity). La curva Precision-Recall es más
  # informativa: muestra el trade-off entre encontrar pobres (recall) y
  # "contaminar" la lista con no-pobres (1 - precision).
  # Un clasificador aleatorio en un dataset con 20% de positivos tiene AUC-PR ≈ 0.20.

  roc_obj <- pROC::roc(
    response  = y_train,          # etiqueta real (0/1)
    predictor = oof_probs_cl,      # probabilidad predicha
    quiet     = TRUE
  )
  auc_roc <- as.numeric(pROC::auc(roc_obj))

  # Construir datos para la curva PR manualmente desde los umbrales del objeto ROC
  pr_df <- tibble(
    threshold   = roc_obj$thresholds,
    recall      = roc_obj$sensitivities,
    # Precision desde TP y FP:  TP = sens * n_pos,  FP = (1-spec) * n_neg
    n_pos = sum(y_train == 1),
    n_neg = sum(y_train == 0),
    TP    = roc_obj$sensitivities * n_pos,
    FP    = (1 - roc_obj$specificities) * n_neg,
    precision   = TP / (TP + FP)
  ) |>
    filter(is.finite(precision), is.finite(recall))

  p_prcurve <- ggplot(pr_df, aes(x = recall, y = precision)) +
    geom_line(color = "#534AB7", linewidth = 0.9) +
    # Línea horizontal = rendimiento de un clasificador aleatorio
    geom_hline(yintercept = mean(y_train),
               linetype = "dashed", color = "gray60") +
    annotate("text", x = 0.85, y = mean(y_train) + 0.015,
             label = sprintf("Clasificador aleatorio (%.0f%%)", mean(y_train) * 100),
             size = 3, color = "gray50") +
    labs(
      title    = paste0("LPM: curva Precision-Recall — ", model_id),
      subtitle = sprintf("AUC-ROC: %.4f | Predicciones CV out-of-fold", auc_roc),
      x        = "Recall  (fracción de pobres capturada)",
      y        = "Precision  (fracción de predichos pobres que lo son)",
      caption  = "Mejor que el clasificador aleatorio → la curva está sobre la línea de puntos."
    ) +
    theme_minimal(base_size = 12)

  ggsave(
    file.path(dir_model, "prcurve.png"),
    p_prcurve, width = 6, height = 5, dpi = 150
  )
  message("Gráfico guardado: ", model_id, "/prcurve.png")

  # ─── 5.4  Curva ROC ──────────────────────────────────────────────────────────
  # La curva ROC grafica True Positive Rate (Recall) vs. False Positive Rate
  # (1 - Specificity) para todos los thresholds posibles.
  # AUC-ROC = probabilidad de que el modelo le asigne mayor score a un hogar
  # pobre que a uno no-pobre elegidos al azar. AUC = 0.5 → azar; AUC = 1 → perfecto.
  # Con clases desbalanceadas conviene leer esta curva junto con la PR (5.3):
  # la ROC tiende a verse más optimista porque los TN abundantes "inflan" el eje X.

  roc_df <- tibble(
    fpr = 1 - roc_obj$specificities,   # False Positive Rate
    tpr = roc_obj$sensitivities         # True Positive Rate (Recall)
  )

  p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
    geom_line(color = "#534AB7", linewidth = 0.9) +
    # Diagonal = clasificador aleatorio (AUC = 0.5)
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "gray60") +
    annotate("text", x = 0.72, y = 0.65,
             label = "Clasificador aleatorio",
             size = 3, color = "gray50", angle = 35) +
    annotate("text", x = 0.55, y = 0.15,
             label = sprintf("AUC-ROC = %.4f", auc_roc),
             size = 3.5, color = "#534AB7") +
    scale_x_continuous(labels = scales::percent_format()) +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(
      title    = paste0("LPM: curva ROC — ", model_id),
      subtitle = "Predicciones CV out-of-fold",
      x        = "Tasa de Falsos Positivos  (1 − Specificity)",
      y        = "Tasa de Verdaderos Positivos  (Recall)",
      caption  = "Leer junto con la curva PR: la ROC puede ser optimista con clases desbalanceadas."
    ) +
    coord_equal() +
    theme_minimal(base_size = 12)

  ggsave(
    file.path(dir_model, "roc.png"),
    p_roc, width = 6, height = 6, dpi = 150
  )
  message("Gráfico guardado: ", model_id, "/roc.png")


  # ---------------------------------------------------------------------------
  # PASO 6: PREDICCIÓN EN TEST Y SUBMISSION KAGGLE
  # ---------------------------------------------------------------------------
  # Aplicamos el modelo final (ajustado en todo el train) sobre los datos de test
  # y clasificamos con el threshold elegido (óptimo de CV).

  probs_test <- suppressWarnings(predict(model_final, newdata = X_test))
  probs_test <- pmax(pmin(probs_test, 1), 0)       # clampear a [0,1]
  preds_test <- as.integer(probs_test >= THRESHOLD)

  cat(sprintf(
    "\n━━━ Predicciones en test — %s ━━━\n%d hogares predichos como pobres (%.1f%%) | threshold = %.2f\n",
    model_id, sum(preds_test), mean(preds_test) * 100, THRESHOLD
  ))

  # El CSV debe tener exactamente dos columnas: id y pobre
  submission <- tibble(
    id    = test$id,
    pobre = preds_test
  )

  submission_file <- file.path(
    dir_submissions,
    paste0("submission_", model_id, ".csv")
  )
  write_csv(submission, submission_file)
  message("Submission guardada: ", submission_file)
  message("→ Subir este archivo a Kaggle y anotar el public F1 en model_registry.csv")


  # ---------------------------------------------------------------------------
  # PASO 7: GUARDAR DIAGNÓSTICOS
  # ---------------------------------------------------------------------------
  # Guardamos todo en un .rds para poder recargar sin re-entrenar.
  # Útil para: comparar modelos, hacer gráficos adicionales, debugging.

  diagnostics <- list(
    # Identificación
    model_id        = model_id,
    autor           = AUTOR,
    notas           = notas,
    fecha           = Sys.Date(),
    vars_excluir    = vars_excluir,    # qué variables se excluyeron — para reproducir
    # Modelos
    model_final     = model_final,     # objeto lm() completo
    # Predicciones CV (para análisis posterior)
    oof_probs_raw   = oof_probs,       # sin clampear — útil para análisis de extremos
    oof_probs_cl    = oof_probs_cl,    # clampeadas a [0,1]
    oof_actuals     = y_train,
    fold_ids        = fold_ids,        # qué fila fue a qué fold — para reproducir exactamente
    # Optimización de threshold
    th_results      = th_results,
    threshold_opt   = THRESHOLD_OPT,
    threshold_usado = THRESHOLD,
    # Métricas
    cm_cv           = cm_cv,
    cv_F1           = cv_F1,
    cv_Precision    = cv_Precision,
    cv_Recall       = cv_Recall,
    auc_roc         = auc_roc,
    # Variables
    predictores     = names(X_train),
    n_features      = ncol(X_train)
  )

  diag_file <- file.path(dir_model, "diagnostics.rds")
  saveRDS(diagnostics, diag_file)
  message("Diagnósticos guardados: ", model_id, "/diagnostics.rds")


  # ---------------------------------------------------------------------------
  # PASO 8: REGISTRAR EN MODEL REGISTRY
  # ---------------------------------------------------------------------------
  # model_registry.csv es la tabla maestra del equipo: una fila por corrida.
  # Si el archivo no existe, lo crea. Si ya existe, agrega la fila al final.
  # Permite comparar todos los modelos del equipo en un solo lugar.
  #
  # ┌──────────────────────────────────────────────────────────────────────────┐
  # │  >>>  DESPUÉS DE SUBIR LA SUBMISSION A KAGGLE  <<<                      │
  # │  Editar manualmente model_registry.csv y completar kaggle_public_F1     │
  # │  con el puntaje que aparece en el public leaderboard de Kaggle.         │
  # └──────────────────────────────────────────────────────────────────────────┘

  nueva_fila <- tibble(
    model_id           = model_id,
    fecha              = Sys.Date(),
    autor              = AUTOR,
    algoritmo          = "LPM",
    n_features         = ncol(X_train),
    imbalance_strategy = "none",
    cv_folds           = K_FOLDS,
    cv_F1              = round(cv_F1,        4),
    cv_Precision       = round(cv_Precision, 4),
    cv_Recall          = round(cv_Recall,    4),
    auc_roc            = round(auc_roc,      4),
    kaggle_public_F1   = NA_real_,   # <<< llenar manualmente después de Kaggle
    threshold          = THRESHOLD,
    notas              = notas
  )

  # Si el registro ya existe, leer y agregar; si no, crear desde cero
  if (file.exists(registry_path)) {
    registry <- read_csv(registry_path, show_col_types = FALSE)
    registry <- bind_rows(registry, nueva_fila)
  } else {
    registry <- nueva_fila
  }
  write_csv(registry, registry_path)
  message("Registro actualizado: ", registry_path)

  cat("\n━━━ Fila registrada en model_registry.csv ━━━\n")
  print(nueva_fila)

  message("\n", strrep("═", 62))
  message("  ", model_id, " completado ✓")
  message(strrep("═", 62), "\n")

  # Retornar métricas clave para la tabla comparativa de la Sección 7
  tibble(
    model_id         = model_id,
    especificacion   = if_else(grepl("A$", model_id), "componentes", "indices"),
    n_features       = ncol(X_train),
    cv_F1            = round(cv_F1,        4),
    cv_Precision     = round(cv_Precision, 4),
    cv_Recall        = round(cv_Recall,    4),
    auc_roc          = round(auc_roc,      4),
    threshold        = THRESHOLD
  )
}


# =============================================================================
# SECCIÓN 6: EJECUTAR AMBOS MODELOS
# =============================================================================
# Cada llamada a run_lpm() corre el pipeline completo (CV → threshold →
# modelo final → gráficos → submission → registro) para una especificación.
#
# IMPORTANTE: los folds de CV fueron generados en la Sección 4 y son los
# mismos para ambas llamadas. Cualquier diferencia en las métricas CV se debe
# a la especificación de variables, no al muestreo de los folds.

result_A <- run_lpm(
  model_id     = "LPM_002A",
  notas        = "Sin índices compuestos — se conservan componentes: ciudad, dpto, dep_*, alguno_*, prop. individuales",
  vars_excluir = VARS_EXCLUIR_A
)

result_B <- run_lpm(
  model_id     = "LPM_002B",
  notas        = "Con índices compuestos — se eliminan componentes: zona, bogota, dummies regionales, n_privaciones, etc.",
  vars_excluir = VARS_EXCLUIR_B
)


# =============================================================================
# SECCIÓN 7: COMPARACIÓN FINAL DE MODELOS
# =============================================================================
#
# ¿QUÉ COMPARAMOS?
#   F1-CV es la métrica principal (es la que evalúa Kaggle). AUC-ROC es
#   diagnóstica: un AUC mayor indica mejor separación entre pobres y no-pobres
#   independientemente del threshold.
#
#   n_features muestra el costo en complejidad: un modelo con más predictores
#   es más difícil de interpretar y puede sobreajustar más si los predictores
#   adicionales son ruidosos.
#
# CRITERIO DE DECISIÓN:
#   1. Si la diferencia en F1-CV es >= 0.005 → el modelo con mayor F1 gana.
#   2. Si la diferencia es < 0.005 → los dos modelos son prácticamente iguales
#      en términos predictivos; preferir el más parsimonioso (LPM_002B, menos
#      predictores) porque es más interpretable y generaliza mejor.
#   3. Siempre verificar con el public F1 de Kaggle: el CV puede favorecer a
#      un modelo que el leaderboard contradice (sobreajuste al threshold).

comparacion <- bind_rows(result_A, result_B)

cat("\n")
cat(strrep("═", 62), "\n")
cat("  COMPARACIÓN FINAL: LPM_002A vs LPM_002B\n")
cat(strrep("═", 62), "\n\n")
print(comparacion, n = Inf)

# Identificar el ganador por F1-CV
ganador <- comparacion |> slice_max(cv_F1, n = 1, with_ties = FALSE)

cat(sprintf(
  "\n► Mejor modelo por F1-CV: %s  (F1 = %.4f, AUC-ROC = %.4f, n_features = %d)\n",
  ganador$model_id, ganador$cv_F1, ganador$auc_roc, ganador$n_features
))

diferencia_F1 <- abs(comparacion$cv_F1[1] - comparacion$cv_F1[2])
cat(sprintf("► Diferencia absoluta en F1-CV: %.4f\n", diferencia_F1))

if (diferencia_F1 < 0.005) {
  mas_parsimonioso <- comparacion |> slice_min(n_features, n = 1, with_ties = FALSE)
  cat("► Diferencia < 0.005: los modelos son prácticamente equivalentes.\n")
  cat(sprintf("  Se recomienda preferir %s (%d features, más parsimonioso).\n",
              mas_parsimonioso$model_id, mas_parsimonioso$n_features))
}

cat("\n► Próximos pasos:\n")
cat("  1. Subir ambas submissions a Kaggle y registrar el public F1 en\n")
cat("     02_outputs/model_registry.csv (columna kaggle_public_F1).\n")
cat("  2. Si el public F1 confirma al ganador del CV, usar esa especificación\n")
cat("     como base para la siguiente iteración (ver Sección 8).\n")
cat("  3. Si el CV y Kaggle discrepan, sospechar sobreajuste al threshold;\n")
cat("     probar con threshold fijo (ej. 0.50) o folds estratificados.\n\n")


# =============================================================================
# SECCIÓN 8: IDEAS PARA MEJORAR EL F1 (no ejecutar — anotar para iterar)
# =============================================================================
# Para cada experimento: incrementar MODEL_ID, actualizar NOTAS, correr el
# script entero, subir a Kaggle y registrar el public F1.

# ── IDEA 1: Ajustar el threshold hacia un valor de política ──────────────────
# El F1 equilibra precision y recall con igual peso (β=1 en F_β). Pero en
# política social, excluir a un pobre (FN) puede ser más costoso que filtrar
# a un no-pobre (FP). Para dar más peso al recall:
#   Buscar el threshold que maximiza F_2 (que pondera recall el doble que precision):
#   F2 = 5 * Precision * Recall / (4 * Precision + Recall)
#   Sustituir en la búsqueda de th_results:
#     F2 = (1 + 4) * (Precision * Recall) / (4 * Precision + Recall)

# ── IDEA 2: Pesos por clase ──────────────────────────────────────────────────
# lm() acepta un argumento `weights`. Dar más peso a la clase minoritaria
# penaliza más los errores sobre hogares pobres → aumenta recall.
#   n_pos   <- sum(y_train == 1)
#   n_neg   <- sum(y_train == 0)
#   w_train <- if_else(y_train == 1, n_neg / n_pos, 1.0)
#   lm(pobre ~ ., data = df_train_full, weights = w_train)
# Impacto esperado: recall ↑, precision ↓. Si el F1 mejora, mantener.

# ── IDEA 3: Selección de variables por valor-p o magnitud ────────────────────
# Hay ~114 variables + dummies de factores. Muchas pueden ser ruidosas.
# Alternativa 1: retener solo los predictores con p-value < 0.05 en el modelo final.
# Alternativa 2: retener solo los predictores con |beta| > 0.03 (umbral pragmático).
# Un modelo más parsimonioso puede generalizar mejor (menos overfitting).
# Implementación:
#   vars_seleccionadas <- coefs_df |> filter(abs_estimate > 0.03) |> pull(term)
#   # Ajustar con solo esas variables y comparar F1 CV

# ── IDEA 4: Interacciones adicionales ────────────────────────────────────────
# Variables que podrían capturar efectos heterogéneos no modelados:
#   zona × nivel_educ_jefe       (retornos a educación diferenciados por zona)
#   n_menores_18 × sin_ocupados  (máxima carga sin ingreso laboral)
#   estrato_hog × prop_informal  (estrato bajo + informal = doble riesgo)
#   edad_jefe × nivel_educ_jefe  (ciclo de vida con mayor o menor capital humano)
# Crear las interacciones en un nuevo tibble y usarlo como X_train alternativo.

# ── IDEA 5: Estratificar los folds de CV ─────────────────────────────────────
# Con 20% de pobres, algunos folds por azar tendrán menos/más pobres.
# Estratificar garantiza que cada fold tenga el mismo ratio de clases.
#   folds_estratificados <- caret::createFolds(
#     y    = factor(y_train),
#     k    = K_FOLDS,
#     list = TRUE      # devuelve una lista con los índices de validación por fold
#   )
#   # Reemplazar el loop usando folds_estratificados[[k]] en lugar de which(fold_ids == k)

# ── IDEA 6: Regularización (Lasso/Ridge) ─────────────────────────────────────
# glmnet con alpha=1 (Lasso) selecciona variables automáticamente y puede
# superar al LPM en F1 al manejar mejor la multicolinealidad residual.
# glmnet con alpha=0 (Ridge) encoge todos los coeficientes sin eliminar ninguno,
# lo que puede ser útil si queremos conservar todas las variables pero reducir
# el sobreajuste.
#   library(glmnet)
#   X_mat <- model.matrix(~ ., data = X_train)[, -1]  # expandir factores a dummies
#   fit   <- cv.glmnet(X_mat, y_train, alpha = 1, family = "gaussian", nfolds = K_FOLDS)
#   preds <- predict(fit, newx = X_mat_test, s = "lambda.min")
