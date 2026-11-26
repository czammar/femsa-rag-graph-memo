# Resultados RAG hibrido + expansion de queries (BM25 + denso, RRF k=60) — FEMSA 2024

Sistema: retrieval hibrido top-5 paginas — BM25 Okapi (rank_bm25, k1=1.5, b=0.75) + all-MiniLM-L6-v2, fusion Reciprocal Rank Fusion k=60 — sobre las 151 paginas del informe, con queries expandidas (traduccion manual al ingles + keywords de dominio, ver expand_queries.py; sin cifras esperadas ni nombres-respuesta). Reader Qwen2.5-1.5B-Instruct local (bf16, CPU), mismo prompt y mismo condense que el baseline denso. Scorer identico a memo/evaluate.py (misma exactitud numerica, misma definicion de alucinacion, misma regla Q3 con 1,596). La unica diferencia con el hibrido es la QUERY de retrieval.

Correctas: **5/12**. Con alucinacion (numero no verificado en el informe): **3/12**. Retrieval top-5 con todas las cifras esperadas literales: **4/12** (hibrido sin expansion: 1/12; denso puro: 2/12; ver Caveats).

| ID | Nivel | Pregunta | Esperado | Respuesta RAG | Veredicto | Modo de fallo |
|----|-------|----------|----------|---------------|-----------|---------------|
| Q1 | 1 | ¿Cuáles fueron los ingresos totales consolidados de FEMSA en | Ps. 781,585 millones, +11.2%. | Los ingresos totales consolidados de FEMSA en 2024 ascendieron a Ps. 781,585 millones, un incremento del 11.2% respecto al año anterior (2023). | correcta | ok |
| Q2 | 1 | ¿Cuántos usuarios activos tuvo Spin by OXXO en 2024? | 8.6 millones (+24.9%). | 8.6 millones | incorrecta | cifra esperada fuera del corpus (+24.9% no existe en el informe; el lector contesto bien lo preguntado) |
| Q3 | 1 | ¿Cuántas tiendas sumó Proximidad Américas al cierre de 2024  | 23,206 tiendas; 1,236 aperturas. | En Proximidad América, el total de tiendas consolidadas al cierre de 2024 fue de 26,322, y surcaron un total de 438 nuevas tiendas durante ese año. | incorrecta | fallo del lector (23,206 estaba en top-5 [114,19,22,112,21] y en el contexto condensado; el lector eligio 26,322, el total de la division, y '438 nuevas tiendas'; 1,596 no llego al contexto por el condense) |
| Q4 | 1 | ¿Cuántas estaciones tenía OXXO GAS al cierre de 2024? | 571. | Al cierre de 2024, OXXO GAS contaba con un total de 571 servicios de gasolina. | correcta | ok |
| Q5 | 2 | ¿Qué división tuvo el mayor crecimiento porcentual de ingres | Empate nominal: ambas +14.2% (trampa deliberada). | Para determinar qué división tuvo el mayor crecimiento porcentual de ingresos en 2024, necesitamos analizar los datos proporcionados.  En el fragmento 113, se menciona que "Proximity Americas Division | incorrecta | fallo del lector (14.2 en top-5 y en contexto; divago y se corto a los 100 tokens; '113' es numero de pagina) |
| Q6 | 2 | ¿Qué división aportó más ingresos en 2024, Proximidad Améric | Proximidad Américas, Ps. 307,200 millones vs Ps. 279,80 | En 2024, la División Proximidad Américas aportó más ingresos que Coca-Cola FEMSA. Los ingresos totales de Proximidad Américas fueron Ps. 307,197 millones, mientras que los ingresos totales de Coca-Col | correcta | ok |
| Q7 | 2 | ¿Qué creció más en usuarios, Spin by OXXO o Spin Premia? | Spin Premia en absoluto (24.6 millones vs 8.6 millones) | Spin Premia creció un 31.5%, mientras que Spin by OXXO creció un 32.8%. | incorrecta | cifra esperada fuera del corpus + conflicto de ground-truth (ver nota: el informe p.34 dice Spin by OXXO +32.8% y Spin Premia +31.5%; el lector dio exactamente esas cifras con la atribucion correcta; los +27.5%/+24.9% esperados son del comunicado 4Q24, no del informe) |
| Q8 | 2 | ¿En qué países opera Proximidad Américas según el informe? | México, Colombia, Chile, Perú (además de negocios en Es | Según el informe anual de FEMSA 2024, Proximidad Américas opera en 18 países diferentes, incluyendo Argentina, Austria, Brasil, Colombia, Costa Rica, Chile, Ecuador, España, Guatemala, Honduras, Méxic | correcta | ok |
| Q9 | 3 | ¿Qué porcentaje de los ingresos consolidados 2024 representa | ~75.1% (39.3% + 35.8%, aproximado por redondeo). | Según el informe anual de FEMSA 2024, los ingresos consolidados totales fueron de Ps. 781,585 millones. Los ingresos de Proximidad América y Coca-Cola FEMSA juntos representaron un 11.2% del total.  P | incorrecta | cifra esperada fuera del corpus + atribucion erronea (39.3/35.8/75.1% no existen; contesto 11.2%, que es el crecimiento consolidado, no la participacion) |
| Q10 | 3 | ¿Qué estrategia articula Retail, Bebidas y el ecosistema dig | FEMSA Forward; el ecosistema digital centrado en el cli | FEMSA Forward es la estrategia que articula Retail, Bebidas y el ecosistema digital de FEMSA. Se crea este ecosistema basándose en datos y análisis para transformar la vida de millones de personas, co | correcta | ok |
| Q11 | 3 | ¿Cómo evolucionó la iniciativa digital de FEMSA en 2024? | Digital@FEMSA evolucionó a Spin. | En 2024, FEMSA avanzó significativamente su estrategia digital transformando el nombre de su división Digital@FEMSA en Spin. Este cambio reflejaba un reajuste estratégico y una renovación de imagen pa | correcta con alucinacion | ok por keywords pero con alucinacion segun el scorer ('328' = 32.8% real del informe p.34 sobre usuarios de Spin by OXXO; la respuesta se corto a los 100 tokens) |
| Q12 | 3 | ¿Qué división tuvo el año más complicado en 2024 según el pr | Salud: 'año complicado', +5.8% (el menor crecimiento de | La división que tuvo el año más complicado en 2024 fue la salud, debido a la intensa competencia en el mercado mexicano. Su ingresos aumentaron un 14.2% en relación al año anterior. | incorrecta | fallo del lector (5.8% en top-5 [29,112,114,30,26] y en contexto; contesto 14.2%, que es el crecimiento de Proximidad Europa/Coca-Cola FEMSA, no de Salud) |

## Por nivel

- Nivel 1: 2/4 correctas.
- Nivel 2: 2/4 correctas.
- Nivel 3: 1/4 correctas.

## Modos de fallo

- ok: 5 (Q1, Q4, Q6, Q8, Q10)
- cifra esperada fuera del corpus (+24.9% no existe en el informe; el lector contesto bien lo preguntado): 1 (Q2)
- fallo del lector (23,206 estaba en top-5 [114,19,22,112,21] y en el contexto condensado; el lector eligio 26,322, el total de la division, y '438 nuevas tiendas'; 1,596 no llego al contexto por el condense): 1 (Q3)
- fallo del lector (14.2 en top-5 y en contexto; divago y se corto a los 100 tokens; '113' es numero de pagina): 1 (Q5)
- cifra esperada fuera del corpus + conflicto de ground-truth (ver nota: el informe p.34 dice Spin by OXXO +32.8% y Spin Premia +31.5%; el lector dio exactamente esas cifras con la atribucion correcta; los +27.5%/+24.9% esperados son del comunicado 4Q24, no del informe): 1 (Q7)
- cifra esperada fuera del corpus + atribucion erronea (39.3/35.8/75.1% no existen; contesto 11.2%, que es el crecimiento consolidado, no la participacion): 1 (Q9)
- ok por keywords pero con alucinacion segun el scorer ('328' = 32.8% real del informe p.34 sobre usuarios de Spin by OXXO; la respuesta se corto a los 100 tokens): 1 (Q11)
- fallo del lector (5.8% en top-5 [29,112,114,30,26] y en contexto; contesto 14.2%, que es el crecimiento de Proximidad Europa/Coca-Cola FEMSA, no de Salud): 1 (Q12)

## Comparacion de los sistemas (12 preguntas, mismo scorer)

| Sistema | Correctas | Nivel 1 | Nivel 2 | Nivel 3 |
|---------|-----------|---------|---------|---------|
| Grafo (Relate, Cypher) | 12/12 | 4/4 | 4/4 | 4/4 |
| RAG hibrido + expansion de queries (este run) | 5/12 | 2/4 | 2/4 | 1/4 |
| RAG hibrido BM25+denso (sin expansion) | 2/12 | 1/4 | 0/4 | 1/4 |
| RAG vectorial denso puro | 2/12 | 0/4 | 1/4 | 1/4 |
| MEMO run 2 (0.5B LoRA r=32 + ejecutivo 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |
| MEMO run 1 (0.5B LoRA r=8 + ejecutivo 0.5B) | 0/12 | 0/4 | 0/4 | 0/4 |

## Que cambio vs el hibrido sin expansion (2/12)

De 2/12 a 5/12: cuatro flips a favor y uno en contra (por el scorer, no por la sustancia). El hibrido acertaba Q1 y Q11; el expandido acierta Q1, Q4, Q6, Q8 y Q10, y Q11 queda como 'correcta con alucinacion' (ver nota).

- **Q4: hibrido incorrecta -> expandida correcta (flip a favor, retrieval).** La query expandida ('OXXO GAS stations gas stations') metio la p.21 en el top-5 ([21,114,85,82,100]) y el lector contesto 571. En el hibrido la p.21 no entraba y el lector contesto '12 estaciones'.
- **Q6: hibrido incorrecta -> expandida correcta (flip a favor).** El lector dio 307,197 y 279,793 exactos. En el hibrido el contexto lo hizo tomar 128,736 (utilidad bruta) como ingreso de Coca-Cola FEMSA.
- **Q8: hibrido incorrecta -> expandida correcta (flip a favor, con caveat).** Los 4 paises estaban en contexto y el lector los listo (Mexico, Colombia, Chile, Peru), asi que la regla de keywords marca correcta. PERO enumero 18 paises del grupo FEMSA (incluidos Austria, Espana, Paises Bajos), no solo los 4 de Proximidad Americas: la regla es indulgente aqui (ver Caveats).
- **Q10: hibrido incorrecta -> expandida correcta (flip a favor, retrieval).** La query expandida ('strategy Retail Beverages digital ecosystem') metio la p.8 en el top-5 ([34,13,12,8,4]) y el lector nombro 'FEMSA Forward'. En el hibrido la p.8 no entraba y la respuesta era vaga, sin el nombre. Ojo: 'FEMSA Forward' NO estaba en la query (seria filtrar la respuesta); entro por 'strategy/Retail/Beverages/digital ecosystem'.
- **Q11: hibrido correcta -> expandida 'correcta con alucinacion' (flip en contra solo en el conteo).** La sustancia es la misma (Digital@FEMSA -> Spin) y la keyword 'Spin' esta. El scorer flaggea '328' como alucinacion, pero 32.8% ES una cifra real del informe (p.34, crecimiento de usuarios de Spin by OXXO): el conjunto de numeros verificados del scorer (respuestas QA + tripletas) no la contiene. Artefacto del scorer, no del lector.
- **Q3: igual de mal, distinto modo.** En el hibrido era fallo de retrieval (el lector se abstuvo); ahora 23,206 ESTA en el top-5 y en el contexto condensado, y el lector eligio 26,322 (el total de la division) y '438 nuevas tiendas'. El cuello de botella se movio del retrieval al lector. 1,596 no llego al contexto: el condense no incluyo esa linea de la p.112.
- **Q12: igual de mal, distinto modo.** En el hibrido era fallo de retrieval (el lector se abstuvo); ahora 5.8% ESTA en el top-5 y en contexto, y el lector contesto 14.2% (el crecimiento de Proximidad Europa/Coca-Cola FEMSA). Fallo del lector con el dato enfrente.
- **Q5: igual de mal.** 14.2 estaba en top-5 y en contexto en las tres corridas; el lector divaga y se corta a los 100 tokens sin concluir. Aqui el '113' es numero de pagina, igual que el '114' del denso.
- **Q7: igual de mal por el scorer, mejor por el corpus (ver nota de ground-truth).** El lector dio 31.5% y 32.8% con la atribucion correcta segun el informe p.34; los +27.5%/+24.9% esperados son del comunicado 4Q24 y no estan en el informe. El scorer marca incorrecta y flaggea 315/328 como alucinacion, pero ambas cifras estan en el informe.
- **Q9: igual de mal.** Las cifras esperadas (39.3/35.8/75.1%) no existen; el lector contesto 11.2% (el crecimiento consolidado) como si fuera la participacion. Peor que abstenerse (hibrido), mejor que inventar la suma (denso, 43,265).
- **Sin cambio:** Q1 (correcta en ambas), Q2 (cifra fuera del corpus; '8.6 millones' es correcto para la pregunta).
- **Cobertura literal ALL en top-5:** hibrido 1/12 (Q1) -> expandido 4/12 (Q1, Q4, Q5, Q12). Con tolerancia del scorer: 6/12. El retrieval mejoro; lo que no mejoro es el lector.

## Veredicto para el post

**RAG hibrido + expansion de queries: 5/12 — el primer salto real del RAG (2/12 -> 5/12).** La expansion cerro 2 de los 4 fallos de retrieval del hibrido (Q4 y Q10 traen ahora la pagina con el dato) y de rebote arreglo Q6 y Q8. La razon es la que el usuario apunto: el problema era el vocabulario de la pregunta, no el ranker. Traducir al ingles y anadir keywords de dominio ('stations', 'stores', 'Health', 'strategy', 'digital ecosystem') pone en la query los tokens que el corpus si tiene.

Pero el salto deja ver el siguiente cuello de botella con nitidez: con el dato ya en el top-5, el lector falla en Q3 (eligio 26,322 teniendo 23,206 enfrente) y en Q12 (contesto 14.2% teniendo 5.8% enfrente), y divaga en Q5 (14.2% enfrente). De los 7 errores que quedan, 3 son del lector con el dato en contexto, 3 son cifras esperadas que no estan en el informe (Q2, Q7, Q9: techo 9/12) y 1 es Q11, correcta en sustancia pero penalizada por un artefacto del scorer. El siguiente arreglo no es otro ranker ni mas keywords: es el lector (o chunks por seccion para que el condense no pierda la linea, como el 1,596 de Q3).

Dos honestidades que el post debe llevar: (1) Q8 se cuenta correcta por la regla de keywords aunque enumera 18 paises del grupo, no los 4 de la division — la regla es indulgente. (2) Q7 es correcta segun el informe mismo (p.34: Spin by OXXO +32.8%, Spin Premia +31.5%) y solo incorrecta segun la respuesta esperada, que viene del comunicado 4Q24; con una regla de ground-truth como la de Q3 (1,596), el run seria 6/12. Y el flag de 'alucinacion' de 315/328 es un artefacto: ambas cifras estan en el informe.

## Caveats (verificados, no supuestos)

- El diagnostico de retrieval con tolerancia de redondeo del scorer (±0.1%) decia 6/12 con todas las cifras cubiertas; la cobertura literal estricta en top-5 es 4/12 (Q1, Q4, Q5, Q12) — la tolerancia marca 'cubierto' un 27.5% cuando solo hay un 27 suelto, igual que en el baseline denso y en el hibrido. La tabla de detalle usa cobertura literal; la clasificacion de fallos se verifico a mano. En Q6 las cifras estan como valores exactos (307,197/279,793) mientras el campo esperado trae los redondeados de la ontologia (307,200/279,800), asi que la cobertura literal marca False aunque el dato si esta.
- La expansion cerro 2 de los 4 fallos de retrieval del hibrido: Q4 (571, p.21 en top-5) y Q10 ('FEMSA Forward', p.8 en top-5). Q3 y Q12 ahora traen la cifra al top-5 (23,206 en p.21; 5.8% en pp. 112/114) pero el lector falla con el dato enfrente: el cuello de botella se movio del retrieval al lector. Q5 traia el 14.2 en las tres corridas y el lector divaga igual.
- La pregunta original esta en espanol y el corpus en ingles; la expanded query traduce al ingles y anade keywords de dominio, asi que el retrieval ya no es cross-lingual. Q12 exige mapear 'Salud' -> 'Health' (ahora explicito en la query).
- El condensado de contexto (ventanas de ~1,800 caracteres por chunk, 2 ventanas) fue necesario porque el prompt de 5 paginas completas (~4.5k tokens) tarda >6 min por pregunta en CPU. Es el mismo workaround que en el baseline denso, con el mismo costo medido: cifras que estan en el top-5 pueden no llegar al lector (Q3: 1,596 estaba en la p.112 del top-5 y no llego al contexto). Es un costo del workaround CPU, no del metodo RAG en si.
- **Q7, conflicto de ground-truth (verificado en el corpus).** La respuesta esperada (+27.5% Spin Premia vs +24.9% Spin by OXXO) viene del comunicado de prensa 4Q24 y NO esta en el informe anual. El informe mismo, p.34, dice que los usuarios de Spin by OXXO crecieron 32.8% y los de Spin Premia 31.5%. El lector dio exactamente esas cifras con la atribucion correcta. El scorer marca incorrecta (no contiene 27.5/24.9) y flaggea 315/328 como 'alucinacion' porque el conjunto de numeros verificados se construye de las respuestas QA y las tripletas semilla, no del corpus completo — pero ambas cifras estan en el informe. Con una regla como la de Q3 (1,596), Q7 contaria como correcta y el run seria 6/12. Se deja el puntaje del scorer (5/12) para comparabilidad y se documenta la alternativa.
- **Q8, regla de keywords indulgente.** La respuesta enumera 18 paises del grupo FEMSA (Argentina, Austria, Brasil, Colombia, Costa Rica, Chile, Ecuador, Espana, Guatemala, Honduras, Mexico, Paises Bajos, Nicaragua, Panama, Peru, Uruguay, ...) — el footprint del grupo, no los 4 paises de Proximidad Americas (Mexico, Colombia, Chile, Peru). Como los 4 estan en la lista, la regla de keywords marca correcta. Misma regla que en las corridas anteriores, asi que el puntaje es comparable, pero la respuesta es sobre-inclusiva.
- **Q11, flag de alucinacion artefacto.** El '328' que el scorer flaggea es el 32.8% real del informe (p.34, crecimiento de usuarios de Spin by OXXO). La respuesta se corto a los 100 tokens ('aumentaron un 31.').
- Q2 no pregunta el crecimiento (+24.9%) pero el scorer lo exige; la respuesta '8.6 millones' es correcta para la pregunta tal como esta hecha. Misma severidad que en MEMO, asi que los puntajes son comparables.

## Notas de ground-truth

- Q3: Nota ground-truth: Segun el MD&A (p.112): 1,596 tiendas netas nuevas en 2024; 23,206 es OXXO Mexico (la division Proximidad Americas suma 26,322 puntos de venta). La cifra '1,236 aperturas' no aparece en el informe. Si el modelo responde 1,596, es mas correcto que la respuesta esperada.
- Q6: Nota ground-truth: Valores exactos del MD&A: 307,197 y 279,793 (la ontologia los redondea). Se acepta el valor exacto o el redondeado.

## Alucinaciones detectadas

- Q5: numeros no verificados en el informe: 113
- Q7: numeros no verificados en el informe: 315, 328
- Q11: numeros no verificados en el informe: 328

## Detalle de retrieval por pregunta (cobertura literal)

| ID | Paginas recuperadas (top-5) | Cifras en top-5 | Cifras en contexto (condensado) |
|----|-----------------------------|-----------------|---------------------------------|
| Q1 | [112, 113, 115, 114, 108] | 781585: si, 11.2: si | 781585: si, 11.2: si |
| Q2 | [33, 19, 13, 5, 35] | 8.6: si, 24.9: no | 8.6: si, 24.9: no |
| Q3 | [114, 19, 22, 112, 21] | 23206: si, 1236: no | 23206: si, 1236: no |
| Q4 | [21, 114, 85, 82, 100] | 571: si | 571: si |
| Q5 | [112, 114, 29, 25, 10] | 14.2: si | 14.2: si |
| Q6 | [112, 114, 19, 13, 10] | 307200: no, 279800: no | 307200: no, 279800: no |
| Q7 | [33, 19, 35, 13, 34] | 24.6: si, 8.6: si, 27.5: no, 24.9: no | 24.6: si, 8.6: si, 27.5: no, 24.9: no |
| Q8 | [10, 21, 112, 19, 20] | (sin cifras esperadas) | (sin cifras esperadas) |
| Q9 | [112, 114, 19, 127, 10] | 75.1: no, 39.3: no, 35.8: no | 75.1: no, 39.3: no, 35.8: no |
| Q10 | [34, 13, 12, 8, 4] | (sin cifras esperadas) | (sin cifras esperadas) |
| Q11 | [34, 12, 66, 13, 4] | (sin cifras esperadas) | (sin cifras esperadas) |
| Q12 | [29, 112, 114, 30, 26] | 5.8: si | 5.8: si |
