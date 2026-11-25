# Resultados MEMO — Run 2 (FEMSA 2024)

Sistema: EXECUTIVE = **Qwen2.5-1.5B-Instruct** base (congelado); MEMORY =
Qwen2.5-**0.5B**-Instruct + adaptador LoRA r=32 (α=64) entrenado 3 épocas
sobre `reflection_qa_v2.jsonl` (1,414 ejemplos: 446 originales + 968
paráfrasis de pregunta + repetición de los hechos numéricos clave).
El EXECUTIVE nunca ve el corpus; solo las sub-respuestas de MEMORY.
Descomposición: **LLM en 12/12** (el EXECUTIVE 1.5B sí sigue el formato
few-shot; en la run 1 el 0.5B falló en 12/12 y todo cayó a reglas).

Correctas (completas): **2/12**. Parcialmente correcta: **1/12** (Q3).
Con alucinación (número no verificado en el informe): **2/12**.

| ID | Nivel | Pregunta | Esperado | Respuesta MEMO | Veredicto |
|----|-------|----------|----------|----------------|-----------|
| Q1 | 1 | ¿Cuáles fueron los ingresos totales consolidados de FEMSA en 2024 y cuánto crecieron vs 2023? | Ps. 781,585 millones, +11.2%. | Los ingresos totales consolidados de FEMSA en 2024 fueron Ps. 586,989 millones, un aumento de 10.3% respecto al año anterior (2023). | incorrecta |
| Q2 | 1 | ¿Cuántos usuarios activos tuvo Spin by OXXO en 2024? | 8.6 millones (+24.9%). | Spin by OXXO tenía 8.6 millones de usuarios activos en 2024. | incorrecta |
| Q3 | 1 | ¿Cuántas tiendas sumó Proximidad Américas al cierre de 2024 y cuántas aperturas hubo? | 23,206 tiendas; 1,236 aperturas. | Proximidad Américas sumó 25,726 tiendas al cierre de 2024 (26,322 - 6). En 2024, se abrieron 1,596 tiendas. | correcta con alucinación |
| Q4 | 1 | ¿Cuántas estaciones tenía OXXO GAS al cierre de 2024? | 571. | Al cierre de 2024, OXXO GAS contaba con 571 estaciones de gasolina en todo México. | correcta |
| Q5 | 2 | ¿Qué división tuvo el mayor crecimiento porcentual de ingresos en 2024: Proximidad Europa o Coca-Cola FEMSA? | Empate nominal: ambas +14.2% (trampa deliberada). | Coca-Cola FEMSA tuvo el mayor crecimiento porcentual de ingresos en 2024, con un aumento del 16.1%. | incorrecta |
| Q6 | 2 | ¿Qué división aportó más ingresos en 2024, Proximidad Américas o Coca-Cola FEMSA, y por cuánto? | Proximidad Américas, Ps. 307,200 millones vs Ps. 279,800 millones. | Proximidad Américas aportó más ingresos en 2024 con Ps. 307,197 millones, mientras que Coca-Cola FEMSA aportó Ps. 279,793 millones. | correcta |
| Q7 | 2 | ¿Qué creció más en usuarios, Spin by OXXO o Spin Premia? | Spin Premia en absoluto (24.6 millones vs 8.6 millones) y en % (+27.5% vs +24.9%). | Spin Premia creció más en usuarios en comparación con Spin by OXXO en 2024. Spin Premia tuvo un aumento de más de 52 millones de usuarios, mientras que Spin by OXXO solo aumentó en 8.6 millones de usuarios. | incorrecta |
| Q8 | 2 | ¿En qué países opera Proximidad Américas según el informe? | México, Colombia, Chile, Perú (además de negocios en Estados Unidos). | Según el informe anual de FEMSA 2024, Proximidad Américas opera en 13 países y más de 388,000 clientes activos en su área geográfica principal, que incluye América Latina y Norte América. | incorrecta |
| Q9 | 3 | ¿Qué porcentaje de los ingresos consolidados 2024 representan Proximidad Américas y Coca-Cola FEMSA juntas? | ~75.1% (39.3% + 35.8%, aproximado por redondeo). | El 47.2% de los ingresos consolidados de FEMSA en 2024 se derivan de Proximidad Américas y Coca-Cola FEMSA juntas. | incorrecta |
| Q10 | 3 | ¿Qué estrategia articula Retail, Bebidas y el ecosistema digital de FEMSA, y a través de qué se crea ese ecosistema? | FEMSA Forward; el ecosistema digital centrado en el cliente se crea a través de Spin. | Retail, Bebidas y el ecosistema digital de FEMSA articulan su estrategia a través de un plan estratégico que combina a Retail como base, Bebidas como margen y el ecosistema digital como puente. Este ecosistema es creado a través de una creación única que no existe en México. | incorrecta |
| Q11 | 3 | ¿Cómo evolucionó la iniciativa digital de FEMSA en 2024? | Digital@FEMSA evolucionó a Spin. | La iniciativa digital de FEMSA experimentó una disminución significativa de sus ingresos digitales en 2024, registrando un descenso del 15.3%. Esta caída se debe principalmente a dos acciones implementadas por la empresa: la adquisición de las operaciones minoristas de Delek y la adquisición de las finanzas de Coca-Cola. | incorrecta |
| Q12 | 3 | ¿Qué división tuvo el año más complicado en 2024 según el propio informe, y cuánto crecieron sus ingresos? | Salud: 'año complicado', +5.8% (el menor crecimiento de las divisiones reportadas). | Proximidad Américas tuvo el año más complicado en 2024 según el propio informe, y sus ingresos crecieron un 10.3%. | incorrecta |

## Por nivel

- Nivel 1: 1/4 correctas (Q4). Q3 parcialmente correcta (1,596 bien, 25,726 inventado).
- Nivel 2: 1/4 correctas (Q6).
- Nivel 3: 0/4 correctas.

## Comparación run 1 vs run 2 vs control vs grafo

| Sistema | Correctas | Con alucinación | Nivel 1 | Nivel 2 | Nivel 3 |
|---|---|---|---|---|---|
| MEMO run 1 (0.5B, LoRA r=8, 1 época, 446 QA, EXEC 0.5B) | **0/12** | 6/12 | 0/4 | 0/4 | 0/4 |
| Control run 1 (base sin entrenar, EXEC 0.5B) | **0/12** | 4/12 | 0/4 | 0/4 | 0/4 |
| **MEMO run 2 (0.5B, LoRA r=32, 3 épocas, 1,414 QA, EXEC 1.5B)** | **2/12** (+1 parcial) | **2/12** | 1/4 | 1/4 | 0/4 |
| **Control run 2 (base sin entrenar, EXEC 1.5B)** | **0/12** | 6/12 | 0/4 | 0/4 | 0/4 |
| Grafo (Cypher sobre tripletas semilla) | ~perfecto en nivel 1–2 **por construcción** | 0 | — | — | — |
| RAG vectorial | **pendiente** (no se corrió; no se inventan cifras) | — | — | — | — |

La mejora de la run 2 viene del **entrenamiento de MEMORY, no del EXECUTIVE
mejor**: el control run 2 (misma EXECUTIVE 1.5B, MEMORY sin entrenar) sigue
en 0/12 con 6/12 alucinaciones, igual de mal que el control de la run 1.
Con MEMORY entrenado, las alucinaciones bajan de 6/12 a 2/12 y aparecen las
primeras cifras exactas correctas.

## Qué mejoró y qué no

**Mejoró (cifras concretas, repetidas en el dataset v2):**

- Q4 (571 estaciones OXXO GAS): mal en run 1 (139), **bien en run 2**.
- Q6 (307,197 vs 279,793): mal en run 1, **bien en run 2, con los valores
  exactos del MD&A**, no los redondeados de la ontología.
- Q2: la cifra principal (8.6 M usuarios) ahora es correcta; el scorer la
  marca incorrecta porque exige también el +24.9% de crecimiento, que la
  respuesta no dio. (Scorer estricto, igual que en run 1.)
- Q3: las sub-respuestas de MEMORY fueron correctas (26,322 tiendas totales;
  1,596 aperturas); el error está en la síntesis (ver abajo).

**No mejoró (dos modos de fallo distintos):**

1. **El EXECUTIVE hace mal la aritmética con sub-respuestas correctas
   (Q1, Q3).** En Q1, MEMORY devolvió los dos ingresos correctos (307,197 y
   279,793) y el EXECUTIVE los **sumó** y presentó 586,989 como el total
   consolidado. En Q3, MEMORY dio 26,322 y 1,596 correctos y el EXECUTIVE
   inventó «25,726 (26,322 − 6)». La memoria ya sabía el dato; la síntesis
   lo rompió. En el paper, el EXECUTIVE es de 32B o Gemini: aquí es 1.5B.
2. **MEMORY falla en comparaciones, hechos abstractos y atribuciones
   (Q5, Q7, Q8, Q10, Q11, Q12).** La trampa +14.2% volvió a fallar (eligió
   Coca-Cola FEMSA con un 16.1% inventado). Q7 acertó la dirección (Spin
   Premia) pero con 52 M inventados. Q8 repite el «13 países» genérico de
   la base sin entrenar: el hecho no quedó fijado. Q10 no menciona
   «FEMSA Forward». Q11 inventa una caída del 15.3% y adquisiciones (Delek,
   Coca-Cola) y hasta clientes en TikTok. Q12 recuerda el 5.8% pero se lo
   atribuye a la división equivocada y elige Proximidad Américas.
   Los hechos que se fijaron son los numéricos simples y repetidos; los que
   requieren comparar, agregar o nombrar una estrategia no se fijaron en
   3 épocas de LoRA sobre 1,414 ejemplos.

## Entrenamiento run 2 (resumen; detalle en `training_log_v2.json`)

- Modelo: Qwen2.5-0.5B-Instruct, LoRA r=32 (α=64), lr 2e-4, batch 1,
  grad_accum 4, seq ≤128, bf16, CPU, sin gradient checkpointing.
  Parámetros entrenables: 17,596,416 (3.44%).
- Dataset: `reflection_qa_v2.jsonl`, 1,414 ejemplos (446 originales + 968
  paráfrasis de pregunta; cada cifra clave aparece ≥12 veces).
- 3 épocas × 353 pasos = 1,059 pasos de optimizador. Pérdida media por
  época (tramo reanudado): época 1: 0.61 (últimos 10: 0.25); época 2: 0.16
  (últimos 10: 0.10); época 3: 0.087 (últimos 10: 0.074). La pérdida bajó
  de ~2.7 al inicio de la época 1 a 0.003–0.17 al final de la época 3:
  casi memorización del dataset — y aun así las cifras de las 12 preguntas
  no salieron todas bien. Memorizar el *texto* de entrenamiento no es lo
  mismo que poder *recuperar* la cifra ante una pregunta nueva.
- Tiempo total de entrenamiento: ~4.1 h de pared (incluye 3 interrupciones
  por reinicio del servicio y sus reanudaciones desde checkpoint).
- Qwen2.5-1.5B-Instruct como MEMORY se descartó con medición: 30.7 s/
  micro-paso → ~36 h para 3 épocas. Se descargó de todos modos (ModelScope,
  3,087,467,144 bytes) y se usa como EXECUTIVE.

## Alucinaciones detectadas (run 2)

- Q1: números no verificados en el informe: 586989
- Q3: números no verificados en el informe: 25726

## Notas de ground-truth

- Q3: Nota ground-truth: Según el MD&A (p.112): 1,596 tiendas netas nuevas en 2024; 23,206 es OXXO México (la división Proximidad Américas suma 26,322 puntos de venta). La cifra '1,236 aperturas' no aparece en el informe. Si el modelo responde 1,596, es más correcto que la respuesta esperada. (El modelo respondió 1,596 = valor real del informe; se cuenta como correcto.)
- Q6: Nota ground-truth: Valores exactos del MD&A: 307,197 y 279,793 (la ontología los redondea). Se acepta el valor exacto o el redondeado.

## Veredicto para el post

**Las correcciones ayudaron, pero no resolvieron el problema: 0/12 → 2/12.**
Más datos (3.2×), más épocas (3×) y más LoRA (r=8 → r=32) fijaron las
cifras simples y repetidas (571, 307,197, 279,793, 8.6 M, 1,596) y bajaron
las alucinaciones de 6/12 a 2/12 — pero el sistema sigue fallando en todo
lo que exige componer: sumar, comparar, atribuir un hecho a la división
correcta o nombrar una estrategia. Y en dos preguntas la memoria ya tenía
el dato correcto y fue el EXECUTIVE quien lo estropeó al sintetizar.

Eso afina el punto del post, no lo debilita: **la memoria en pesos es
compresión con pérdida, y la pérdida se concentra justo donde el post
necesita precisión** — cifras exactas, comparaciones y agregaciones. El
grafo no tiene ese problema porque la respuesta *es* el dato verificado.
El paper MEMO usa un modelo 28× más grande, ~3,000× más datos y un
EXECUTIVE de 32B/Gemini; a escala demo, ni triplicando datos y épocas se
llega a la fiabilidad de una consulta al grafo.
