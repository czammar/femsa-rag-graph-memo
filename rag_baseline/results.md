# Resultados RAG vectorial (baseline) — FEMSA 2024

Sistema: retrieval top-5 paginas con all-MiniLM-L6-v2 (CPU) sobre las 151 paginas del informe; reader Qwen2.5-1.5B-Instruct local (bf16, CPU), respondiendo SOLO desde los chunks recuperados. Scorer identico a memo/evaluate.py (misma exactitud numerica, misma definicion de alucinacion, misma regla Q3 con 1,596).

Correctas: **2/12**. Con alucinacion (numero no verificado en el informe): **2/12**. Retrieval top-5 con todas las cifras esperadas literales: **2/12** (el diagnostico con tolerancia de redondeo decia 6/12; ver Caveats).

| ID | Nivel | Pregunta | Esperado | Respuesta RAG | Veredicto | Modo de fallo |
|----|-------|----------|----------|---------------|-----------|---------------|
| Q1 | 1 | ¿Cuáles fueron los ingresos totales consolidados de FEMSA en | Ps. 781,585 millones, +11.2%. | Los ingresos totales consolidados de FEMSA en 2024 ascendieron a Ps. 40,236 millones, un incremento de 31% respecto al año anterior. | incorrecta | fallo del lector (781,585 en contexto; contesto 40,236; el +11.2% tampoco se recupero) |
| Q2 | 1 | ¿Cuántos usuarios activos tuvo Spin by OXXO en 2024? | 8.6 millones (+24.9%). | 8.6 millones | incorrecta | cifra esperada fuera del corpus (+24.9% no existe en el informe; el lector contesto bien lo preguntado) |
| Q3 | 1 | ¿Cuántas tiendas sumó Proximidad Américas al cierre de 2024  | 23,206 tiendas; 1,236 aperturas. | No lo sé. | incorrecta | fallo de retrieval (23,206 solo en p.21, no en top-5) |
| Q4 | 1 | ¿Cuántas estaciones tenía OXXO GAS al cierre de 2024? | 571. | No lo sé. | incorrecta | fallo de retrieval (571 en pp. 21/114/126, no en top-5) |
| Q5 | 2 | ¿Qué división tuvo el mayor crecimiento porcentual de ingres | Empate nominal: ambas +14.2% (trampa deliberada). | Para determinar qué división tuvo el mayor crecimiento porcentual de ingresos en 2024, necesitamos analizar los datos proporcionados en los fragmentos del informe anual de FEMSA 2024.  En el Fragmento | incorrecta | alucinacion del lector (divago, se corto a 100 tokens; '114' es numero de pagina) |
| Q6 | 2 | ¿Qué división aportó más ingresos en 2024, Proximidad Améric | Proximidad Américas, Ps. 307,200 millones vs Ps. 279,80 | En 2024, la División Proximidad Américas aportó más ingresos que Coca-Cola FEMSA. Los ingresos totales de Proximidad Américas fueron Ps. 307,197 millones, mientras que los de Coca-Cola FEMSA fueron Ps | correcta | ok |
| Q7 | 2 | ¿Qué creció más en usuarios, Spin by OXXO o Spin Premia? | Spin Premia en absoluto (24.6 millones vs 8.6 millones) | Spin Premia creció más en usuarios que Spin by OXXO. Según el informe, Spin Premia tiene 24.6 millones de activos y Spin by OXXO tiene 8.6 millones de activos. | incorrecta | cifra esperada fuera del corpus (+27.5% y +24.9% no existen en el informe; el lector acerto direccion y absolutos) |
| Q8 | 2 | ¿En qué países opera Proximidad Américas según el informe? | México, Colombia, Chile, Perú (además de negocios en Es | Según el informe, Proximidad Américas opera en México y Colombia. | incorrecta | fallo del lector (Chile y Peru en contexto; listo solo 2 de 4) |
| Q9 | 3 | ¿Qué porcentaje de los ingresos consolidados 2024 representa | ~75.1% (39.3% + 35.8%, aproximado por redondeo). | Según el informe, Proximidad Américas y Coca-Cola FEMSA reportaron un ingreso consolidado de Ps. 23,729 millones en 2024, mientras que Proximidad Europea tuvo un ingreso de Ps. 19,536 millones. La sum | incorrecta | cifra esperada fuera del corpus + aritmetica (39.3/35.8/75.1% no existen; sumo 23,729+19,536=43,265) |
| Q10 | 3 | ¿Qué estrategia articula Retail, Bebidas y el ecosistema dig | FEMSA Forward; el ecosistema digital centrado en el cli | Retail, Bebidas y el ecosistema digital de FEMSA se articulan mediante un modelo de valor centrado en el cliente. Se crean este ecosistema basándose en datos y análisis para transformar la vida de mil | incorrecta | fallo de retrieval ('FEMSA Forward' en pp. 3/5/8/12/14/42/101/115, no en top-5) |
| Q11 | 3 | ¿Cómo evolucionó la iniciativa digital de FEMSA en 2024? | Digital@FEMSA evolucionó a Spin. | Durante el año, definimos un claro negocio estratégico para Digital@FEMSA, ahora llamado Spin, para impulsar su crecimiento con un plan de cinco años alineado con los objetivos estratégicos de FEMSA F | correcta | ok |
| Q12 | 3 | ¿Qué división tuvo el año más complicado en 2024 según el pr | Salud: 'año complicado', +5.8% (el menor crecimiento de | La división que tuvo el año más complicado en 2024 fue Argentina, donde el ingreso cayó un 10%. | incorrecta | fallo de retrieval (Health +5.8% en pp. 5/112/114, no en top-5; el 5.8 del contexto era margen neto) |

## Por nivel

- Nivel 1: 0/4 correctas.
- Nivel 2: 1/4 correctas.
- Nivel 3: 1/4 correctas.

## Modos de fallo

- ok: 2 (Q6, Q11)
- fallo del lector (781,585 en contexto; contesto 40,236; el +11.2% tampoco se recupero): 1 (Q1)
- cifra esperada fuera del corpus (+24.9% no existe en el informe; el lector contesto bien lo preguntado): 1 (Q2)
- fallo de retrieval (23,206 solo en p.21, no en top-5): 1 (Q3)
- fallo de retrieval (571 en pp. 21/114/126, no en top-5): 1 (Q4)
- alucinacion del lector (divago, se corto a 100 tokens; '114' es numero de pagina): 1 (Q5)
- cifra esperada fuera del corpus (+27.5% y +24.9% no existen en el informe; el lector acerto direccion y absolutos): 1 (Q7)
- fallo del lector (Chile y Peru en contexto; listo solo 2 de 4): 1 (Q8)
- cifra esperada fuera del corpus + aritmetica (39.3/35.8/75.1% no existen; sumo 23,729+19,536=43,265): 1 (Q9)
- fallo de retrieval ('FEMSA Forward' en pp. 3/5/8/12/14/42/101/115, no en top-5): 1 (Q10)
- fallo de retrieval (Health +5.8% en pp. 5/112/114, no en top-5; el 5.8 del contexto era margen neto): 1 (Q12)

## Comparacion de los tres sistemas (12 preguntas, mismo scorer)

| Sistema | Correctas | Nivel 1 | Nivel 2 | Nivel 3 |
|---------|-----------|---------|---------|---------|
| Grafo (Relate, Cypher) | 12/12 | 4/4 | 4/4 | 4/4 |
| RAG vectorial (este baseline) | 2/12 | 0/4 | 1/4 | 1/4 |
| MEMO run 2 (0.5B LoRA r=32 + ejecutivo 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |
| MEMO run 1 (0.5B LoRA r=8 + ejecutivo 0.5B) | 0/12 | 0/4 | 0/4 | 0/4 |

## Veredicto para el post

**RAG vectorial: 2/12 — el mismo puntaje que MEMO run 2, pero con un perfil de fallo distinto y mas instructivo.** El grafo gana 12/12 porque la respuesta *es* el dato verificado; RAG y MEMO pierden porque tienen que *reconstruir* la cifra — uno desde texto recuperado, el otro desde pesos. El experimento muestra que, a esta escala y con un lector de 1.5B, el RAG clasico no es mejor que la memoria parametrica: empata con ella.

Donde pierde RAG, verificado cifra por cifra contra el corpus:

- **Retrieval (4): Q3, Q4, Q10, Q12.** La cifra existe en el informe y el top-5 no la trajo. Q3: 23,206 esta solo en la p.21; Q4: 571 esta en pp. 21/114/126; Q10: 'FEMSA Forward' esta en pp. 3/5/8/12/14/42/101/115; Q12: Health +5.8% esta en pp. 5/112/114. El lector en Q3/Q4 se abstuvo honestamente ('No lo se'); en Q10/Q12 contesto con otra cosa. Un retriever mejor (o chunks mas pequenos que la pagina) es el arreglo mas directo aqui.
- **Cifras esperadas fuera del corpus (3): Q2, Q7, Q9.** +24.9%, +27.5%, 39.3%, 35.8% y 75.1% no aparecen en ninguna de las 151 paginas. 24.9/27.5 vienen del comunicado de prensa 4Q24, no del informe anual; 39.3/35.8/75.1 son participaciones calculadas, no cifras impresas. El grafo las 'sabe' solo porque sus tripletas semilla se construyeron a mano con esas fuentes. En Q2 y Q7 el lector contesto bien lo sustantivo (8.6 millones; Spin Premia crece mas, 24.6 vs 8.6) y el scorer lo marca mal porque exige los porcentajes que no estan en el corpus. Techo real de RAG sobre este set, tal como esta especificado: 9/12.
- **Lector (3): Q1, Q5, Q8.** La cifra estaba en el contexto y el lector fallo. Q1 eligio Ps. 40,236 (+31%) teniendo 781,585 enfrente — elige la cifra equivocada de la tabla. Q5 divago y se corto a los 100 tokens sin concluir, filtrando '114' (numero de pagina) como si fuera un dato. Q8 tenia Chile y Peru en el contexto y solo listo Mexico y Colombia. El lector de 1.5B no es un extractor fiable de tablas.
- **Aritmetica (Q9).** Tenia 307,197 en contexto pero no 279,793 ni 781,585 (perdida en el condensado); agarro 23,729 y 19,536 y los sumo en 43,265. Es el mismo modo de fallo del ejecutivo de MEMO run 2: la sintesis rompe el dato correcto.

Lectura para el post: el RAG clasico falla en los dos extremos — cuando no recupera el dato y cuando lo recupera pero el lector no sabe elegirlo. El grafo no tiene ninguno de los dos problemas porque no recupera ni elige: consulta. La moraleja no es que RAG sea inutil, sino que la exactitud de cifras exige que el dato llegue verificado al lector, no solo 'relevante'.

## Caveats (verificados, no supuestos)

- El diagnostico inicial de retrieval usaba la tolerancia de redondeo del scorer (±0.1%): marcaba 'cubierto' el 27.5% de Q7 cuando solo habia un 27 suelto, el 24.9% de Q2 cuando no hay ningun 24.9, y el 39.3/35.8/75.1% de Q9 cuando solo habia numeros cercanos. La cobertura literal estricta en top-5 es 2/12 completa (Q5, Q12) — con dos matices: en Q6 las cifras estan como valores exactos (307,197/279,793) mientras el campo esperado trae los redondeados de la ontologia (307,200/279,800), y en Q12 el 5.8 literal del top-5 es el margen de utilidad neta, no el crecimiento de Salud. La tabla de detalle usa cobertura literal; la clasificacion de fallos se verifico a mano.
- El condensado de contexto (ventanas de ~1,800 caracteres por chunk, 2 ventanas) fue necesario porque el prompt de 5 paginas completas (~4.5k tokens) tarda >6 min por pregunta en CPU. Perdio cifras en Q9 (279,793 y 781,585 estaban en el top-5 y no llegaron al lector). Es un costo medido del workaround, no del metodo RAG en si.
- La pregunta esta en espanol y el corpus en ingles; el retrieval es cross-lingual con un embedder entrenado en ingles (all-MiniLM-L6-v2). Q12 exige mapear 'Salud' -> 'Health'.
- Q2 no pregunta el crecimiento (+24.9%) pero el scorer lo exige; la respuesta '8.6 millones' es correcta para la pregunta tal como esta hecha. Misma severidad que en MEMO, asi que los puntajes son comparables.

## Notas de ground-truth

- Q3: Nota ground-truth: Segun el MD&A (p.112): 1,596 tiendas netas nuevas en 2024; 23,206 es OXXO Mexico (la division Proximidad Americas suma 26,322 puntos de venta). La cifra '1,236 aperturas' no aparece en el informe. Si el modelo responde 1,596, es mas correcto que la respuesta esperada.
- Q6: Nota ground-truth: Valores exactos del MD&A: 307,197 y 279,793 (la ontologia los redondea). Se acepta el valor exacto o el redondeado.

## Alucinaciones detectadas

- Q5: numeros no verificados en el informe: 114
- Q9: numeros no verificados en el informe: 43265

## Detalle de retrieval por pregunta (cobertura literal)

| ID | Paginas recuperadas (top-5) | Cifras en top-5 | Cifras en contexto (condensado) |
|----|-----------------------------|-----------------|---------------------------------|
| Q1 | [88, 113, 30, 109, 17] | 781585: si, 11.2: no | 781585: si, 11.2: no |
| Q2 | [5, 35, 33, 88, 100] | 8.6: si, 24.9: no | 8.6: si, 24.9: no |
| Q3 | [88, 17, 55, 109, 16] | 23206: no, 1236: no | 23206: no, 1236: no |
| Q4 | [88, 80, 83, 30, 109] | 571: no | 571: no |
| Q5 | [30, 28, 10, 114, 27] | 14.2: si | 14.2: si |
| Q6 | [30, 114, 10, 27, 112] | 307200: no, 279800: no | 307200: no, 279800: no |
| Q7 | [35, 5, 33, 13, 100] | 24.6: si, 8.6: si, 27.5: no, 24.9: no | 24.6: si, 8.6: si, 27.5: no, 24.9: no |
| Q8 | [109, 17, 88, 21, 151] | (sin cifras esperadas) | (sin cifras esperadas) |
| Q9 | [30, 10, 114, 28, 27] | 75.1: no, 39.3: no, 35.8: no | 75.1: no, 39.3: no, 35.8: no |
| Q10 | [23, 109, 22, 13, 114] | (sin cifras esperadas) | (sin cifras esperadas) |
| Q11 | [34, 12, 5, 23, 113] | (sin cifras esperadas) | (sin cifras esperadas) |
| Q12 | [88, 109, 29, 100, 17] | 5.8: si | 5.8: si |
