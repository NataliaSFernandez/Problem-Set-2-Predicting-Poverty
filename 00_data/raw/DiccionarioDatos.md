# Diccionario de Datos
## Poverty Prediction Challenge — Colombia (DANE / MESE)

> **Fuente:** DANE — Empalme de las Series de Empleo, Pobreza y Desigualdad (MESE)  
> **Unidad de observación:** Hogar e Individuo  
> **Llave de unión:** `id` (identificador de hogar)

---

## Índice

1. [Base de Hogares (train_hogares / test_hogares)](#1-base-de-hogares)
2. [Base de Personas (train_personas / test_personas)](#2-base-de-personas)
   - [Identificación y geografía](#21-identificación-y-geografía)
   - [Características sociodemográficas](#22-características-sociodemográficas)
   - [Educación](#23-educación)
   - [Empleo e ingresos laborales](#24-empleo-e-ingresos-laborales)
   - [Ingresos no laborales](#25-ingresos-no-laborales)
   - [Agregados de ingreso y clasificación laboral](#26-agregados-de-ingreso-y-clasificación-laboral)
3. [Variable Objetivo](#3-variable-objetivo)
4. [Disponibilidad por conjunto de datos](#4-disponibilidad-por-conjunto-de-datos)
5. [Códigos frecuentes](#5-códigos-frecuentes)

---

## 1. Base de Hogares

> Archivos: `train_hogares.csv` | `test_hogares.csv`  
> Cada fila es un **hogar**.

| Variable | Tipo | Descripción | Valores / Unidad |
|----------|------|-------------|-----------------|
| `id` | string | Identificador único del hogar. Llave de unión con la base de personas. | Hash alfanumérico |
| `Clase` | categórica | Zona geográfica del hogar. | `1` = Urbana · `2` = Rural |
| `Dominio` | categórica | Ciudad o dominio de estudio. | Nombre de ciudad (ej. `MEDELLIN`, `SANTA MARTA`) |
| `P5000` | numérica | Número de cuartos o piezas que tiene la vivienda (sin contar cocina, baños ni garaje). | Entero ≥ 1 |
| `P5010` | numérica | Número de cuartos o piezas usados para dormir. | Entero ≥ 1 |
| `P5090` | categórica | Tenencia de la vivienda. | `1` = Propia pagada · `2` = Propia pagando · `3` = Arriendo/subarr. · `4` = Usufructo · `5` = Posesión sin título · `6` = Otra |
| `P5100` | numérica | Valor mensual de cuota de amortización. | Pesos colombianos (COP) |
| `P5130` | numérica | Valor mensual del arriendo estimado de la vivienda (si fuera a arrendarse). | COP |
| `P5140` | numérica | Valor mensual del arriendo | COP |
| `Nper` | numérica | Número total de personas en el hogar. | Entero ≥ 1 |
| `Npersug` | numérica | Número de personas en la unidad de gasto | Entero ≥ 1 |
| `Li` | numérica | Línea de indigencia (pobreza extrema) per cápita mensual. | COP |
| `Lp` | numérica | Línea de pobreza per cápita mensual. | COP |
| `Fex_c` | numérica | Factor de expansión del hogar a nivel nacional. | Ponderador muestral |
| `Depto` | categórica | Código DANE del departamento. | Código de 2 dígitos (ej. `"05"` = Antioquia, `"47"` = Magdalena) |
| `Fex_dpto` | numérica | Factor de expansión del hogar a nivel departamental. | Ponderador muestral |

> **Nota:** `train_hogares` incluye adicionalmente la variable `Ingpcug` (ingreso per cápita de la unidad de gasto) y `Pobre` (variable objetivo). Ver [Sección 3](#3-variable-objetivo).

---

## 2. Base de Personas

> Archivos: `train_personas.csv` | `test_personas.csv`  
> Cada fila es un **individuo**. Múltiples filas por hogar.

### 2.1 Identificación y geografía

| Variable | Tipo | Descripción | Valores / Unidad |
|----------|------|-------------|-----------------|
| `id` | string | Identificador del hogar al que pertenece el individuo. Llave de unión. | Hash alfanumérico |
| `Orden` | numérica | Número de orden del individuo dentro del hogar. | Entero ≥ 1 |
| `Clase` | categórica | Zona geográfica. | `1` = Urbana · `2` = Rural |
| `Dominio` | categórica | Ciudad o dominio de estudio. | Nombre de ciudad |
| `Estrato1` | numérica | Estrato socioeconómico de la vivienda. | `1`–`6` (1=más bajo) · `9` = No aplica |
| `Depto` | categórica | Código DANE del departamento. | Código de 2 dígitos |
| `Fex_c` | numérica | Factor de expansión individual nacional. | Ponderador muestral |
| `Fex_dpto` | numérica | Factor de expansión individual departamental. | Ponderador muestral |

---

### 2.2 Características sociodemográficas

| Variable | Tipo | Descripción | Valores / Unidad |
|----------|------|-------------|-----------------|
| `P6020` | categórica | Sexo del individuo. | `1` = Hombre · `2` = Mujer |
| `P6040` | numérica | Edad en años cumplidos. | 0–110 |
| `P6050` | categórica | Parentesco con el jefe del hogar. | `1` = Jefe(a) · `2` = Pareja/cónyuge · `3` = Hijo(a) · `4` = Nieto(a) · `5` = Otro pariente · `6` = Empleado(a) doméstico(a) · `7` = Pensionista · `8` = Otro no pariente |
| `P6090` | categórica | ¿Está afiliado a seguridad social en salud? | `1` = Sí · `2` = No · `9` = No sabe |
| `P6100` | categórica | Tipo de afiliación a salud. | `1` = Contributivo · `2` = Subsidiado · `3` = Especial · `9` = No sabe |

---

### 2.3 Educación

| Variable | Tipo | Descripción | Valores / Unidad |
|----------|------|-------------|-----------------|
| `P6210` | categórica | Nivel educativo más alto aprobado. | `1` = Ninguno · `2` = Preescolar · `3` = Básica primaria · `4` = Básica secundaria · `5` = Media (bachillerato) · `6` = Técnico/Tecnológico · `7` = Universitario · `8` = Especialización · `9` = Maestría/Doctorado |
| `P6210s1` | numérica | Último año/grado aprobado en el nivel educativo reportado en `P6210`. | Entero |

---

### 2.4 Empleo e ingresos laborales

#### Condición de actividad

| Variable | Tipo | Descripción | Valores / Unidad |
|----------|------|-------------|-----------------|
| `P6240` | categórica | Actividad principal la semana pasada. | `1` = Trabajó · `2` = No trabajó pero tiene trabajo · `3` = Buscó trabajo · `4` = Estudió · `5` = Oficios del hogar · `6` = Incapacitado · `7` = Otra actividad |
| `Pet` | binaria | Población en edad de trabajar (≥ 12 años urbano; ≥ 10 rural). | `1` = Sí · `0`/`NA` = No |
| `Oc` | binaria | Ocupado (empleado). | `1` = Sí · `0`/`NA` = No |
| `Des` | binaria | Desempleado (busca trabajo activamente). | `1` = Sí · `0`/`NA` = No |
| `Ina` | binaria | Inactivo (no trabaja ni busca). | `1` = Sí · `0`/`NA` = No |

#### Características del empleo principal

| Variable | Tipo | Descripción | Valores / Unidad |
|----------|------|-------------|-----------------|
| `Oficio` | categórica | Ocupación principal (código CIUO-88). | Código numérico de ocupación |
| `P6426` | numérica | Tiempo en el empleo actual (meses). | Entero ≥ 0 |
| `P6430` | categórica | Posición ocupacional en el empleo principal. | `1` = Empleado empresa particular · `2` = Empleado gobierno · `3` = Empleado doméstico · `4` = Cuenta propia · `5` = Patrón/empleador · `6` = Trabajador familiar sin remuneración · `7` = Trabajador sin remuneración no familiar · `8` = Jornalero/peón · `9` = Otro |
| `P6800` | numérica | Horas trabajadas a la semana en el empleo principal. | Horas |
| `P6870` | categórica | Tamaño de la empresa donde trabaja. | `1` = 1 persona · `2` = 2–3 · `3` = 4–5 · `4` = 6–10 · `5` = 11–19 · `6` = 20–30 · `7` = 31–50 · `8` = 51–100 · `9` = Más de 100 |

#### Ingresos del empleo principal

| Variable | Tipo | Descripción | Valores / Unidad |
|----------|------|-------------|-----------------|
| `P6500` | numérica | Ingreso laboral mensual en el empleo principal (asalariados). | COP |
| `P6510` | categórica | ¿Recibió ingresos adicionales por horas extras? | `1` = Sí · `2` = No |
| `P6510s1` | numérica | Valor mensual de horas extras. | COP |
| `P6545` | categórica | ¿Recibió subsidio de alimentación? | `1` = Sí · `2` = No |
| `P6545s1` | numérica | Valor mensual del subsidio de alimentación. | COP |
| `P6580` | categórica | ¿Recibió subsidio de transporte? | `1` = Sí · `2` = No |
| `P6580s1` | numérica | Valor mensual del subsidio de transporte. | COP |
| `P6585s1` | categórica | ¿La empresa le paga o le paga en especie por concepto de vivienda? | `1` = Sí · `2` = No |
| `P6585s1a1` | numérica | Valor mensual de la prestación en especie: vivienda. | COP |
| `P6585s2` | categórica | ¿La empresa le paga alimentación en especie? | `1` = Sí · `2` = No |
| `P6585s2a1` | numérica | Valor mensual de la prestación en especie: alimentación. | COP |
| `P6585s3` | categórica | ¿La empresa le paga transporte en especie? | `1` = Sí · `2` = No |
| `P6585s3a1` | numérica | Valor mensual de la prestación en especie: transporte. | COP |
| `P6585s4` | categórica | ¿La empresa le otorga otras prestaciones en especie? | `1` = Sí · `2` = No |
| `P6585s4a1` | numérica | Valor mensual de otras prestaciones en especie. | COP |
| `P6590` | categórica | ¿Recibió prima, bonificación u otros pagos? | `1` = Sí · `2` = No |
| `P6590s1` | numérica | Valor mensual prorrateado de prima/bonificación. | COP |
| `P6600` | categórica | ¿Recibió cesantías? | `1` = Sí · `2` = No |
| `P6600s1` | numérica | Valor mensual prorrateado de cesantías. | COP |
| `P6610` | categórica | ¿Recibió intereses sobre cesantías? | `1` = Sí · `2` = No |
| `P6610s1` | numérica | Valor mensual prorrateado de intereses sobre cesantías. | COP |
| `P6620` | categórica | ¿Recibió vacaciones pagadas? | `1` = Sí · `2` = No |
| `P6620s1` | numérica | Valor mensual prorrateado de vacaciones. | COP |
| `P6630s1` | categórica | ¿Cotiza a pensión? | `1` = Sí · `2` = No |
| `P6630s1a1` | numérica | Valor mensual cotizado a pensión. | COP |
| `P6630s2` | categórica | ¿Cotiza a salud? | `1` = Sí · `2` = No |
| `P6630s2a1` | numérica | Valor mensual cotizado a salud. | COP |
| `P6630s3` | categórica | ¿Cotiza a caja de compensación? | `1` = Sí · `2` = No |
| `P6630s3a1` | numérica | Valor mensual cotizado a caja de compensación. | COP |
| `P6630s4` | categórica | ¿Cotiza a riesgos profesionales (ARL)? | `1` = Sí · `2` = No |
| `P6630s4a1` | numérica | Valor mensual cotizado a ARL. | COP |
| `P6630s6` | categórica | ¿Le descuentan cuotas al sindicato? | `1` = Sí · `2` = No |
| `P6630s6a1` | numérica | Valor mensual del descuento sindical. | COP |
| `P6750` | numérica | Ganancia neta mensual en el trabajo por cuenta propia o patrón. | COP |
| `P6760` | numérica | Número de meses trabajados en el último año (cuenta propia/patrón). | Entero 1–12 |
| `P550` | numérica | Ingreso mensual del trabajo como empleado doméstico. | COP |

#### Segundo empleo

| Variable | Tipo | Descripción | Valores / Unidad |
|----------|------|-------------|-----------------|
| `P6920` | categórica | ¿Tiene otro empleo o negocio además del principal? | `1` = Sí · `2` = No |
| `P7040` | categórica | ¿En este otro trabajo es asalariado o cuenta propia? | `1` = Asalariado · `2` = Cuenta propia/patrón |
| `P7045` | numérica | Horas trabajadas a la semana en el segundo empleo. | Horas |
| `P7050` | numérica | Ingreso mensual del segundo empleo. | COP |

#### Búsqueda de empleo y subempleo

| Variable | Tipo | Descripción | Valores / Unidad |
|----------|------|-------------|-----------------|
| `P7070` | numérica | Semanas que lleva buscando trabajo (desempleados). | Semanas |
| `P7090` | categórica | ¿Se siente subempleado por insuficiencia de horas? | `1` = Sí · `2` = No |
| `P7110` | categórica | ¿Se siente subempleado por competencias? | `1` = Sí · `2` = No |
| `P7120` | categórica | ¿Se siente subempleado por ingresos? | `1` = Sí · `2` = No |
| `P7140s1` | categórica | ¿Hizo diligencias para conseguir trabajo en las últimas 4 semanas? | `1` = Sí · `2` = No |
| `P7140s2` | categórica | ¿Está disponible para trabajar? | `1` = Sí · `2` = No |
| `P7150` | numérica | ¿Cuánto tiempo lleva desempleado? (meses). | Meses |
| `P7160` | categórica | Razón principal por la que no buscó trabajo. | Códigos DANE |
| `P7310` | categórica | ¿Por qué no trabaja más horas? | Códigos DANE |
| `P7350` | categórica | Posición en el empleo del trabajo secundario. | Mismos códigos que `P6430` |
| `P7422` | categórica | ¿Recibió ingreso como empleado doméstico en otro hogar? | `1` = Sí · `2` = No |
| `P7422s1` | numérica | Valor mensual del ingreso como doméstico en otro hogar. | COP |
| `P7472` | categórica | ¿Recibió ingresos por trabajo en actividades agrícolas? | `1` = Sí · `2` = No |
| `P7472s1` | numérica | Valor mensual de ingresos agrícolas. | COP |
| `P7495` | categórica | ¿Recibió ingresos por trabajo en actividades no agrícolas por cuenta propia? | `1` = Sí · `2` = No |

---

### 2.5 Ingresos no laborales

| Variable | Tipo | Descripción | Valores / Unidad |
|----------|------|-------------|-----------------|
| `P7500s1` | categórica | ¿Recibió ingresos por arrendamientos? | `1` = Sí · `2` = No |
| `P7500s1a1` | numérica | Valor mensual de ingresos por arrendamientos. | COP |
| `P7500s2` | categórica | ¿Recibió pensión o jubilación? | `1` = Sí · `2` = No |
| `P7500s2a1` | numérica | Valor mensual de pensión o jubilación. | COP |
| `P7500s3` | categórica | ¿Recibió ingresos por dividendos, intereses o rendimientos financieros? | `1` = Sí · `2` = No |
| `P7500s3a1` | numérica | Valor mensual de dividendos e intereses. | COP |
| `P7505` | categórica | ¿Recibió ingresos por otros conceptos no laborales? | `1` = Sí · `2` = No |
| `P7510s1` | categórica | ¿Recibió transferencias de familiares dentro del país? | `1` = Sí · `2` = No |
| `P7510s1a1` | numérica | Valor mensual de transferencias familiares nacionales. | COP |
| `P7510s2` | categórica | ¿Recibió remesas del exterior? | `1` = Sí · `2` = No |
| `P7510s2a1` | numérica | Valor mensual de remesas del exterior. | COP |
| `P7510s3` | categórica | ¿Recibió transferencias de instituciones (subsidios del Estado, ONG)? | `1` = Sí · `2` = No |
| `P7510s3a1` | numérica | Valor mensual de subsidios institucionales. | COP |
| `P7510s5` | categórica | ¿Recibió subsidio de desempleo o ayuda del SENA? | `1` = Sí · `2` = No |
| `P7510s5a1` | numérica | Valor mensual del subsidio de desempleo. | COP |
| `P7510s6` | categórica | ¿Recibió ingreso por donaciones o limosnas? | `1` = Sí · `2` = No |
| `P7510s6a1` | numérica | Valor mensual de donaciones/limosnas. | COP |
| `P7510s7` | categórica | ¿Recibió otros ingresos no laborales? | `1` = Sí · `2` = No |
| `P7510s7a1` | numérica | Valor mensual de otros ingresos no laborales. | COP |

---

### 2.6 Agregados de ingreso y clasificación laboral

> Variables calculadas por DANE a partir de las respuestas individuales.

| Variable | Tipo | Descripción | Valores / Unidad |
|----------|------|-------------|-----------------|
| `Impa` | numérica | Ingreso monetario por trabajo asalariado (empleo principal). | COP |
| `Isa` | numérica | Ingreso total por trabajo asalariado. | COP |
| `Ie` | numérica | Ingreso de empleadores/patronos. | COP |
| `Imdi` | numérica | Ingreso monetario de independientes (cuenta propia). | COP |
| `Iof1` | numérica | Ingreso por trabajo en el oficio principal. | COP |
| `Iof2` | numérica | Ingreso por segundo empleo. | COP |
| `Iof3h` | numérica | Ingreso por trabajo doméstico remunerado. | COP |
| `Iof3i` | numérica | Ingreso por trabajo independiente adicional. | COP |
| `Iof6` | numérica | Ingreso por trabajo agropecuario. | COP |
| `Impaes` | numérica | Ingreso monetario asalariado (en pesos constantes). | COP constantes |
| `Isaes` | numérica | Ingreso asalariado total (pesos constantes). | COP constantes |
| `Iees` | numérica | Ingreso de empleadores (pesos constantes). | COP constantes |
| `Imdies` | numérica | Ingreso independientes (pesos constantes). | COP constantes |
| `Iof1es` | numérica | Ingreso oficio principal (pesos constantes). | COP constantes |
| `Iof2es` | numérica | Ingreso segundo empleo (pesos constantes). | COP constantes |
| `Iof3hes` | numérica | Ingreso doméstico (pesos constantes). | COP constantes |
| `Iof3ies` | numérica | Ingreso independiente adicional (pesos constantes). | COP constantes |
| `Iof6es` | numérica | Ingreso agropecuario (pesos constantes). | COP constantes |
| `Ingtotob` | numérica | Ingreso total observado del individuo (suma de todas las fuentes). | COP |
| `Ingtotes` | numérica | Ingreso total del individuo (pesos constantes). | COP constantes |
| `Ingtot` | numérica | Ingreso total del individuo (variable oficial DANE). | COP |
| `Cclasnr2`–`Cclasnr11` | binarias | Indicadores de clasificación de ingresos por fuente (flags internos DANE). | `0`/`1` |

---

## 3. Variable Objetivo

> Disponible **únicamente en `train_hogares.csv`**.

| Variable | Tipo | Descripción | Valores |
|----------|------|-------------|---------|
| `Pobre` | binaria | **Variable a predecir.** Indica si el hogar es pobre monetariamente. Se define como `1` si el ingreso per cápita de la unidad de gasto es inferior a la línea de pobreza (`Lp`). | `1` = Pobre · `0` = No pobre |

> **Nota sobre clase desbalanceada:** Históricamente la pobreza monetaria en Colombia ronda el 30–40% según el año y la muestra. Se recomienda revisar la distribución empírica en el train y considerar técnicas de balanceo (SMOTE, pesos de clase) si el desbalance supera 70/30.

---

## 4. Disponibilidad por conjunto de datos

| Variable / Grupo | train_hogares | test_hogares | train_personas | test_personas |
|------------------|:-------------:|:------------:|:--------------:|:-------------:|
| `id`, `Clase`, `Dominio`, `Depto` | ✅ | ✅ | ✅ | ✅ |
| Variables de vivienda (`P5000`–`P5140`) | ✅ | ✅ | — | — |
| `Nper`, `Npersug`, `Li`, `Lp` | ✅ | ✅ | — | — |
| Factores de expansión (`Fex_c`, `Fex_dpto`) | ✅ | ✅ | ✅ | ✅ |
| **`Pobre`** (variable objetivo) | ✅ | ❌ | — | — |
| Características sociodemográficas | — | — | ✅ | ✅ |
| Variables de educación | — | — | ✅ | ✅ |
| Variables de empleo y posición ocupacional | — | — | ✅ | ✅ |
| Ingresos laborales detallados | — | — | ✅ | ⚠️ parcial |
| Ingresos no laborales | — | — | ✅ | ⚠️ parcial |
| Agregados de ingreso (`Ingtot`, `Impa`, etc.) | — | — | ✅ | ⚠️ parcial |

> ⚠️ **Algunas variables de ingreso están ausentes en el test por diseño del challenge**, para simular condiciones reales donde el ingreso es costoso de medir.

---

## 5. Códigos frecuentes

### Departamentos (selección)

| Código | Departamento |
|--------|-------------|
| `05` | Antioquia |
| `08` | Atlántico |
| `11` | Bogotá D.C. |
| `13` | Bolívar |
| `17` | Caldas |
| `18` | Caquetá |
| `19` | Cauca |
| `20` | Cesar |
| `23` | Córdoba |
| `25` | Cundinamarca |
| `27` | Chocó |
| `41` | Huila |
| `44` | La Guajira |
| `47` | Magdalena |
| `50` | Meta |
| `52` | Nariño |
| `54` | Norte de Santander |
| `63` | Quindío |
| `66` | Risaralda |
| `68` | Santander |
| `70` | Sucre |
| `73` | Tolima |
| `76` | Valle del Cauca |

### Convención de respuesta dicotómica

La gran mayoría de preguntas binarias en la encuesta usan la convención:

| Código | Significado |
|--------|-------------|
| `1` | Sí / Recibió / Tiene |
| `2` | No / No recibió / No tiene |
| `9` | No sabe / No informa |
| `NA` | No aplica (persona fuera del universo de la pregunta) |

---

*Última actualización del diccionario: 2026. Basado en el módulo de Empleo e Ingresos de la GEIH-MESE (DANE).*

---

## 6. Variables adicionales construidas (Script 03)

> Generadas en `01_code/03_variables_adicionales.R` a partir de las variables base.
> Disponibles en `00_data/processed/train_features.rds` y `test_features.rds`.

### 6.1 Calidad de vivienda y hacinamiento

| Variable | Construcción | Racionalidad económica | Referencia |
|----------|-------------|----------------------|-----------|
| `hacinamiento` | `num_personas / num_cuartos` | El espacio disponible por persona es un proxy directo de privación material. Hogares hacinados enfrentan peores condiciones sanitarias, menor productividad del trabajo en casa y menor calidad de vida. | Ferreira & Ravallion (2008) |
| `hacinamiento_dorm` | `num_personas / num_cuartos_dorm` | Restringe el hacinamiento al espacio para dormir, más directamente vinculado al descanso, la salud y el desarrollo infantil. | DANE IPM Colombia (2022) |
| `cuartos_pc` | `num_cuartos / num_personas` | Versión inversa del hacinamiento; facilita la interpretación positiva del espacio como activo del hogar. | Ferreira & Ravallion (2008) |
| `tenencia_insegura` | `1` si `tipo_tenencia ∈ {3,4,5,6}` | La inseguridad en la tenencia (arriendo, usufructo, posesión sin título) impide acumular patrimonio, limita el acceso a crédito y expone al hogar a desplazamiento forzado. | Payne (2002) |
| `vivienda_propia_pag` | `1` si `tipo_tenencia == 1` | La propiedad pagada es el mayor activo acumulable para hogares de ingresos bajos; elimina el gasto en arriendo y provee garantía para crédito. | Attanasio & Székely (2001) |

### 6.2 Composición demográfica y carga económica

| Variable | Construcción | Racionalidad económica | Referencia |
|----------|-------------|----------------------|-----------|
| `prop_menores` | `n_menores_18 / num_personas` | Los menores no generan ingreso laboral; una proporción alta implica que pocos miembros deben sostener a muchos, deprimiendo el ingreso per cápita. | Lanjouw & Ravallion (1995) |
| `prop_adultos_may_p` | `n_adultos_may / num_personas` | Los adultos mayores pueden ser dependientes o perceptores de pensión; su efecto sobre la pobreza depende de si cuentan con protección social. | CEPAL (2017) |
| `menores_por_ocupado` | `n_menores_18 / max(n_ocupados, 1)` | Mide directamente cuántos niños financia cada trabajador del hogar. Es una razón de dependencia económica más precisa que el ratio estándar. | Lanjouw & Ravallion (1995) |
| `sin_ocupados` | `1` si `n_ocupados == 0` | Hogar sin ningún ingreso laboral: máxima vulnerabilidad económica, dependiente enteramente de transferencias o activos. | Chaudhuri et al. (2002) |
| `un_solo_ocupado` | `1` si `n_ocupados == 1` | Un único proveedor concentra todo el riesgo de ingreso; cualquier choque de empleo o salud empuja al hogar a la pobreza. | Morduch (1995) |
| `doble_ingreso` | `1` si `n_ocupados ≥ 2` | Dos o más trabajadores diversifican el riesgo laboral del hogar, aumentando la resiliencia ante choques idiosincráticos. | Ellis (2000) |
| `hogar_monoparental` | `1` si `mujer_jefa == 1` y `n_menores_18 > 0` | Los hogares con jefa mujer y menores combinan la brecha salarial de género con la mayor carga de cuidado, resultando en mayor vulnerabilidad estructural. | Buvinic & Gupta (1997) |

### 6.3 Capital humano — efectos no lineales y umbrales

| Variable | Construcción | Racionalidad económica | Referencia |
|----------|-------------|----------------------|-----------|
| `educ_jefe_sq` | `nivel_educ_jefe²` | Los retornos a la educación son convexos: cada nivel adicional tiene mayor impacto salarial en los tramos altos. El término cuadrático captura este efecto creciente. | Mincer (1974) |
| `educ_prom_sq` | `nivel_educ_prom²` | Mismo argumento aplicado a la educación promedio del hogar; captura el efecto no lineal del capital humano colectivo. | Card (1999) |
| `educ_media_o_mas` | `1` si `nivel_educ_jefe ≥ 5` | El bachillerato es la puerta de entrada al mercado laboral formal en Colombia. Activa el "sheepskin effect" de diploma más relevante para hogares de ingresos medios. | Hungerford & Solon (1987) |
| `educ_superior_jefe` | `1` si `nivel_educ_jefe ≥ 6` | La educación técnica o universitaria multiplica los retornos salariales y reduce drásticamente la probabilidad de informalidad. | Card (1999) |
| `brecha_educ` | `nivel_educ_max − nivel_educ_jefe` | Un valor positivo indica que algún miembro superó educativamente al jefe: señal de movilidad intergeneracional ascendente dentro del hogar. | Attanasio & Székely (2001) |

### 6.4 Calidad del empleo e índice de formalidad

| Variable | Construcción | Racionalidad económica | Referencia |
|----------|-------------|----------------------|-----------|
| `indice_formalidad` | Promedio de `prop_asalariado`, `prop_cotiza_pension`, `prop_prima_serv`, `prop_sub_transp` | La formalidad laboral es el principal mecanismo de protección contra la pobreza en América Latina. Un índice compuesto captura múltiples dimensiones del empleo de calidad. | Fields (2011) |
| `formal_estricto` | `prop_asalariado × prop_cotiza_pension` | Exige que la formalidad sea simultánea en contrato y seguridad social; filtra trabajadores asalariados sin pensión (informalidad parcial). | Levy (2008) |
| `vulnerabilidad_lab` | `prop_domestico + prop_fam_sin_rem + prop_jornalero` | Suma de las tres posiciones ocupacionales más precarias: sin contrato, sin prestaciones y con los salarios más bajos del mercado. | ILO (2018) |
| `tasa_empleo_total` | `n_ocupados / num_personas` | Complementa la tasa de ocupación sobre PET con la proporción de empleados sobre el total, capturando el esfuerzo laboral relativo al tamaño del hogar. | Fields (2011) |

### 6.5 Diversificación y resiliencia de ingresos

| Variable | Construcción | Racionalidad económica | Referencia |
|----------|-------------|----------------------|-----------|
| `n_fuentes_no_lab` | Suma de 8 indicadores de ingreso no laboral | La diversificación de fuentes reduce la varianza del ingreso total. Hogares con múltiples fuentes son más resilientes a pérdidas de empleo. | Ellis (2000) |
| `tiene_ingreso_no_lab` | `1` si al menos una fuente no laboral | Indicador binario de si el hogar tiene algún colchón de ingreso más allá del trabajo; discrimina los más vulnerables a un choque laboral. | Morduch (1995) |
| `solo_subsidio` | `1` si solo recibe subsidio estatal (sin pensión, remesas ni activos) | Hogar cuya única fuente no laboral es un subsidio del Estado: la situación de mayor precariedad en términos de dependencia de ingresos externos. | Barrientos (2011) |
| `ingresos_activos` | `1` si recibe intereses o arrendamientos | Los ingresos de activos financieros o inmobiliarios señalan acumulación de capital: característica casi exclusiva de hogares no pobres. | McKenzie & Rapoport (2011) |

### 6.6 Protección social formal e informal

| Variable | Construcción | Racionalidad económica | Referencia |
|----------|-------------|----------------------|-----------|
| `proteccion_social` | `1` si cotiza a pensión o alguien recibe jubilación | La pensión contributiva es el instrumento más efectivo de protección contra la pobreza en la vejez y, mientras se cotiza, señala empleo formal estable. | Holzmann & Jørgensen (2001) |
| `desprotegido` | `1` si alta informalidad, sin cotización y sin jubilación | Identifica hogares en el "hueco" de la protección: demasiado pobres para la seguridad contributiva y sin acceso pleno a la asistencial. | Barrientos (2011) |
| `prop_contributivo` | `prop_afil_salud − prop_subsidiado` | La afiliación contributiva (vs. subsidiada) refleja capacidad de pago y empleo formal; la diferencia aisla a quienes cotizan activamente al sistema de salud. | Glassman et al. (2010) |

### 6.7 Clasificación regional

| Variable | Construcción | Racionalidad económica | Referencia |
|----------|-------------|----------------------|-----------|
| `costa_caribe` | `1` si `dpto ∈ {08,13,20,23,44,47,70}` | La Costa Caribe concentra históricamente las tasas de pobreza más altas de Colombia, asociadas a menor capital humano, mayor informalidad y exclusión institucional histórica. | Galvis & Meisel (2010) |
| `region_pacifico` | `1` si `dpto ∈ {19,27,52}` | La región Pacífica (Cauca, Chocó, Nariño) combina alta pobreza, aislamiento geográfico y menor inversión pública histórica. Chocó es el departamento más pobre de Colombia. | Bonet (2006) |
| `bogota` | `1` si `dpto == "11"` | Bogotá concentra empleo formal, mayores salarios y mejor acceso a servicios; sus hogares tienen probabilidades de pobreza sistemáticamente menores. | DANE (2023) |
| `eje_cafetero` | `1` si `dpto ∈ {17,63,66,76}` | El Eje Cafetero y Valle tienen economías diversificadas con mayor clase media; sirve como región de referencia intermedia entre Bogotá y la periferia pobre. | Bonet (2006) |
| `rural_costa_caribe` | `zona == 2` y `dpto` de Costa Caribe | Intersección de ruralidad y región costera: captura la pobreza rural costera, que combina los dos factores de riesgo geográfico más fuertes del país. | Galvis & Meisel (2010) |

### 6.8 Términos de interacción

| Variable | Construcción | Racionalidad económica | Referencia |
|----------|-------------|----------------------|-----------|
| `educ_x_formal` | `nivel_educ_jefe × prop_cotiza_pension` | La educación y la formalidad son complementarias: el capital humano solo se traduce en menores ingresos si hay empleos formales que lo remuneren adecuadamente. | Mincer (1974); Fields (2011) |
| `educ_x_ocup` | `nivel_educ_jefe × tasa_ocupacion` | Captura hogares donde el capital humano se convierte efectivamente en empleo; un jefe educado pero desocupado no protege al hogar de la pobreza. | Card (1999) |
| `menores_x_desocup` | `n_menores_18 × (1 − tasa_ocupacion)` | La combinación de muchos menores y alta desocupación es el escenario de mayor pobreza infantil: alta demanda de recursos sin ingreso laboral que la sostenga. | Lanjouw & Ravallion (1995) |
| `depend_x_informal` | `ratio_depend × prop_informal` | Doble penalización estructural: muchos dependientes sostenidos con ingresos precarios e inestables. Identifica los hogares con mayor vulnerabilidad crónica. | Chaudhuri et al. (2002) |
| `rural_x_cta_propia` | `(zona == 2) × prop_cta_propia` | El trabajo por cuenta propia en zona rural suele ser de subsistencia, no una elección empresarial; su efecto sobre la pobreza es distinto al autoempleo urbano. | Fields (2011) |
| `educ_x_rural` | `nivel_educ_jefe × (zona == 2)` | Los retornos a la educación son menores en zonas rurales por escasez de empleos cualificados; la interacción distingue el efecto diferencial de la educación por zona. | Card (1999) |
| `subsidiado_x_menores` | `prop_subsidiado × n_menores_18` | Combina las dos señales más fuertes de pobreza del modelo (régimen subsidiado y menores); su producto amplifica la señal en los hogares más vulnerables. | Alkire & Foster (2011) |
| `jefe_mayor_pension` | `(edad_jefe > 55) × alguno_pension_jub` | Un jefe mayor con pensión representa el ciclo de vida avanzado con protección formal consolidada: prácticamente inmune a la pobreza por ingreso laboral. | Holzmann & Jørgensen (2001) |
| `estrato_x_zona` | `estrato_hog × (zona == 2)` | El estrato en zona rural está comprimido en los niveles bajos (casi todos son estrato 1); la interacción distingue el significado informativo del estrato por contexto geográfico. | DANE IPM Colombia (2022) |

### 6.9 Transformaciones no lineales

| Variable | Construcción | Racionalidad económica | Referencia |
|----------|-------------|----------------------|-----------|
| `log_num_personas` | `log(1 + num_personas)` | El impacto marginal del tamaño del hogar sobre la pobreza es decreciente: pasar de 1 a 2 personas es más relevante que pasar de 8 a 9. | Deaton (1997) |
| `log_n_menores` | `log(1 + n_menores_18)` | Mismo argumento para el número de menores: el primer hijo tiene el mayor impacto sobre el ingreso per cápita. | Lanjouw & Ravallion (1995) |
| `ratio_depend_sq` | `ratio_depend²` | La relación entre dependencia y pobreza es convexa: ratios muy altos tienen un efecto desproporcionadamente mayor que los moderados. | Alkire & Foster (2011) |
| `tasa_ocup_sq` | `tasa_ocupacion²` | Relación convexa entre empleo y bienestar: pasar de 0% a 20% de ocupación tiene un efecto mayor que de 60% a 80%. | Fields (2011) |
| `hacinamiento_sq` | `hacinamiento²` | El hacinamiento extremo tiene consecuencias sanitarias y cognitivas que crecen de forma no lineal con la densidad. | Ferreira & Ravallion (2008) |
| `edad_jefe_sq` | `edad_jefe²` | Los ingresos siguen un perfil de ciclo de vida en forma de U invertida: crecen con la experiencia y disminuyen tras la jubilación. | Mincer (1974) |

### 6.10 Índice de Privaciones Múltiples (inspirado en el IPM)

> Basado en el enfoque de conteo de Alkire & Foster (2011): un hogar es "multidimensionalmente pobre" si acumula privaciones en al menos **k = 2** dimensiones de las 5 consideradas.

| Variable | Construcción | Racionalidad económica | Referencia |
|----------|-------------|----------------------|-----------|
| `dep_educacion` | `1` si `nivel_educ_jefe ≤ 3` | Privación educativa del jefe según el umbral oficial del IPM Colombia: sin educación básica completa no hay acceso real al mercado laboral formal. | DANE IPM Colombia (2022) |
| `dep_vivienda` | `1` si `hacinamiento > 3` | Privación en condiciones de habitabilidad: umbral de hacinamiento crítico del IPM Colombia. | DANE IPM Colombia (2022) |
| `dep_empleo_formal` | `1` si hay ocupados pero ninguno es asalariado ni cotiza pensión | Privación en calidad de empleo: el hogar tiene ingresos laborales pero todos provienen de la informalidad total. | Alkire & Foster (2011) |
| `dep_proteccion` | `1` si `prop_subsidiado > 0.5` y nadie recibe jubilación | Privación en protección social: el hogar está mayoritariamente en el régimen de salud para pobres y sin pensión acumulada. | Holzmann & Jørgensen (2001) |
| `dep_dependencia` | `1` si `ratio_depend > 1` | Privación demográfica: hay más dependientes (menores + adultos mayores) que personas en edad de trabajar, deprimiendo structuralmente el ingreso per cápita. | Lanjouw & Ravallion (1995) |
| `n_privaciones` | Suma de las 5 privaciones anteriores (0–5) | Indicador cardinal de pobreza multidimensional: a mayor puntaje, mayor acumulación de desventajas simultáneas. | Alkire & Foster (2011) |
| `multi_privado` | `1` si `n_privaciones ≥ 2` | Umbral estándar del IPM: hogar que presenta privaciones en al menos dos dimensiones simultáneamente. | Alkire & Foster (2011) |
| `privado_severo` | `1` si `n_privaciones ≥ 3` | Pobreza multidimensional severa: acumulación crítica de desventajas que hace muy improbable la salida espontánea de la pobreza sin intervención externa. | Alkire & Foster (2011) |

### 6.11 Variables adicionales desde personas crudos

| Variable | Construcción | Racionalidad económica | Referencia |
|----------|-------------|----------------------|-----------|
| `estrato_hog` | Moda de `Estrato1` en el hogar (1–6) | El estrato es el indicador socioeconómico más usado en la política social colombiana: determina tarifas de servicios públicos, acceso a subsidios y focalización de programas. | DANE; Profamilia (2015) |
| `estrato_bajo` | `1` si `estrato_hog ≤ 2` | Los estratos 1 y 2 son los beneficiarios directos de subsidios de energía, agua y gas. Este umbral binario activa la clasificación oficial de vulnerabilidad en servicios públicos. | DANE IPM Colombia (2022) |
| `mujer_jefa` | `1` si el jefe del hogar (parentesco == 1) es mujer | Los hogares con jefa mujer enfrentan brechas salariales de género y mayor carga de trabajo de cuidado no remunerado, elevando su vulnerabilidad estructural a la pobreza. | Buvinic & Gupta (1997) |
| `prop_patron` | Proporción de ocupados como empleadores | Los empleadores generan ingresos superiores al promedio y crean empleo para otros; su presencia es señal de capacidad de acumulación de capital. | Fields (2011) |
| `prop_domestico` | Proporción en trabajo doméstico remunerado | El trabajo doméstico es de los peor remunerados, con escasa protección legal y alta informalidad; su presencia señala precariedad laboral estructural. | ILO (2018) |
| `prop_fam_sin_rem` | Proporción como trabajadores familiares sin remuneración | La categoría más precaria del mercado laboral: trabajo sin salario ni prestaciones, típico de microempresas familiares rurales en subsistencia. | ILO (2018) |
| `prop_jornalero` | Proporción como jornaleros o peones | El trabajo jornalero es estacional, con salario mínimo o inferior, sin contrato ni prestaciones; proxy de pobreza laboral rural extrema. | Fields (2011) |
| `prop_microempresa` | Proporción que trabaja en empresas de 1–3 personas | Las microempresas concentran el empleo informal en América Latina, con salarios sistemáticamente menores y menor cobertura de seguridad social. | Levy (2008) |
| `prop_gran_empresa` | Proporción que trabaja en empresas de 20+ personas | Las empresas grandes tienen mayor probabilidad de cumplir la normativa laboral; su presencia en el hogar se asocia con empleo formal y salarios más altos. | Levy (2008) |
| `antiguedad_prom` | Promedio de meses en el empleo actual | La antigüedad laboral refleja estabilidad del empleo y acumulación de capital específico; empleos más antiguos tienen mayor protección contra el despido. | Topel (1991) |
| `antiguedad_alta` | `1` si alguien lleva más de 24 meses en su empleo | Umbral de dos años: indica que al menos un miembro tiene empleo estable consolidado, reduciendo el riesgo de caída al desempleo. | Topel (1991) |
| `alguno_agropec` | `1` si alguien recibe ingreso agropecuario | El ingreso agropecuario es típico de hogares rurales; puede ser tanto subsistencia (pobreza) como diversificación productiva (resiliencia), según el contexto. | Ellis (2000) |
| `alguno_arriendos` | `1` si alguien recibe ingresos por arrendamiento | Los ingresos por arriendo reflejan posesión de activos inmobiliarios: característica casi exclusiva de hogares de ingresos medios y altos. | Attanasio & Székely (2001) |

---

### Referencias bibliográficas (Script 03)

- Alkire, S. & Foster, J. (2011). Counting and multidimensional poverty measurement. *Journal of Public Economics*, 95(7–8), 476–487.
- Attanasio, O. & Székely, M. (2001). *Going beyond income: Redefining poverty in Latin America*. IDB.
- Barrientos, A. (2011). Social protection and poverty. *International Journal of Social Welfare*, 20(3), 240–249.
- Bonet, J. (2006). Desequilibrios regionales en la política de descentralización en Colombia. *Documentos de Trabajo sobre Economía Regional*, Banco de la República.
- Buvinic, M. & Gupta, G. R. (1997). Female-headed households and female-maintained families: Are they worth targeting to reduce poverty in developing countries? *Economic Development and Cultural Change*, 45(2), 259–280.
- Card, D. (1999). The causal effect of education on earnings. *Handbook of Labor Economics*, 3, 1801–1863.
- Chaudhuri, S., Jalan, J. & Suryahadi, A. (2002). *Assessing household vulnerability to poverty from cross-sectional data*. World Bank Policy Research Working Paper 2882.
- DANE (2022). *Metodología del Índice de Pobreza Multidimensional — Colombia*. Departamento Administrativo Nacional de Estadística.
- Deaton, A. (1997). *The Analysis of Household Surveys*. World Bank / Johns Hopkins University Press.
- Ellis, F. (2000). *Rural Livelihoods and Diversity in Developing Countries*. Oxford University Press.
- Ferreira, F. & Ravallion, M. (2008). *Global poverty and inequality: A review of the evidence*. World Bank Policy Research Working Paper 4623.
- Fields, G. (2011). Labor market analysis for developing countries. *Labour Economics*, 18(S1), S16–S22.
- Galvis, L. A. & Meisel, A. (2010). Persistencia de las desigualdades regionales en Colombia. *Documentos de Trabajo sobre Economía Regional*, Banco de la República.
- Holzmann, R. & Jørgensen, S. (2001). Social risk management: A new conceptual framework for social protection. *International Tax and Public Finance*, 8(4), 529–556.
- Hungerford, T. & Solon, G. (1987). Sheepskin effects in the returns to education. *Review of Economics and Statistics*, 69(1), 175–177.
- ILO (2007). *Decent Work Indicators: Concepts and Definitions*. International Labour Organization.
- ILO (2018). *Women and Men in the Informal Economy: A Statistical Picture* (3rd ed.). International Labour Organization.
- Lanjouw, P. & Ravallion, M. (1995). Poverty and household size. *Economic Journal*, 105(433), 1415–1434.
- Levy, S. (2008). *Good Intentions, Bad Outcomes: Social Policy, Informality, and Economic Growth in Mexico*. Brookings Institution Press.
- McKenzie, D. & Rapoport, H. (2011). Can migration reduce educational attainment? *Journal of Population Economics*, 24(4), 1331–1358.
- Mincer, J. (1974). *Schooling, Experience and Earnings*. Columbia University Press.
- Morduch, J. (1995). Income smoothing and consumption smoothing. *Journal of Economic Perspectives*, 9(3), 103–114.
- Payne, G. (2002). *Land, Rights and Innovation: Improving Tenure Security for the Urban Poor*. ITDG Publishing.
- Topel, R. (1991). Specific capital, mobility, and wages: Wages rise with job seniority. *Journal of Political Economy*, 99(1), 145–176.