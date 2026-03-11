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
| `P5100` | numérica | Valor mensual del arriendo pagado (si aplica). | Pesos colombianos (COP) |
| `P5130` | numérica | Valor mensual del arriendo estimado de la vivienda (si fuera a arrendarse). | COP |
| `P5140` | numérica | Valor mensual del servicio de alojamiento (hoteles, pensiones, etc.). | COP |
| `Nper` | numérica | Número total de personas en el hogar. | Entero ≥ 1 |
| `Npersug` | numérica | Número de personas en la unidad de gasto. | Entero ≥ 1 |
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