# Resultados RAG agentico (LangChain + LangGraph, ReAct textual) — FEMSA 2024

Sistema: agente ReAct con LangGraph (nodos plan -> act -> observe -> answer), modelo local Qwen2.5-1.5B-Instruct (bf16, CPU), maximo 3 pasos de herramienta por pregunta. Herramientas: `vector_search` (retrieval hibrido BM25 + MiniLM, RRF k=60, top-5 paginas), `graph_query` (Cypher contra el store Relate femsa.rl) y `calculator` (aritmetica segura por AST). El modelo de 1.5B no emite tool calls nativos de forma fiable (verificado en test_toolcall.py: cero tool_calls con bind_tools), asi que el agente usa un bucle ReAct textual: el modelo escribe `ACCION: herramienta("argumento")` o `RESPUESTA: <texto>` y el nodo act lo ejecuta. Scorer identico a memo/evaluate.py (misma exactitud numerica, misma definicion de alucinacion, misma regla Q3 con 1,596).

Correctas: **2/12**. Con alucinacion (numero no verificado en el informe): **0/12**. Preguntas donde la evidencia vista por el agente contenia todas las cifras esperadas (literal): **1/12**. Pasos de herramienta: **24** en total (0 vector_search, 24 graph_query, 0 calculator, 0 errores de formato). Llamadas al plan: **44**, de las que **13 no llegaron a ninguna accion ejecutada ni a la RESPUESTA final** (5 en Q5 y 5 en Q9: consultas gigantes truncadas a los 150 tokens; 1 en Q6, Q7 y Q12: una 4ª ACCION identica que choco con el tope de 3 pasos). Tiempo total de la corrida: **65.3 min** (media 326 s por pregunta).

| ID | Nivel | Pregunta | Esperado | Respuesta del agente | Veredicto | Modo de fallo |
|----|-------|----------|----------|----------------------|-----------|---------------|
| Q1 | 1 | ¿Cuáles fueron los ingresos totales consolidados de FEMSA en | Ps. 781,585 millones, +11.2%. | No lo sé. No inventes cifras. | incorrecta | abstencion tras 3 consultas fallidas (nombre de indicador inventado: 'Ingresos Totales Consolidados' en vez de 'Ingresos Totales'; repitio la misma consulta 3 veces pese a la guardia anti-bucle) |
| Q2 | 1 | ¿Cuántos usuarios activos tuvo Spin by OXXO en 2024? | 8.6 millones (+24.9%). | No lo sé. No inventes cifras. | incorrecta | abstencion tras 2 consultas fallidas ('Spin by OXXO' tratado como Division con OPERA_EN y propiedad inventada 's.multiples_operadoras'; es un Formato de la Division Spin) |
| Q3 | 1 | ¿Cuántas tiendas sumó Proximidad Américas al cierre de 2024  | 23,206 tiendas; 1,236 aperturas. | Proximidad Américas sumó 4 tiendas al cierre de 2024, con 1 apertura en Perú, 1 en México, 1 en Colombia y 1 en Chile. | incorrecta | evidencia mal interpretada (conto las 4 filas OPERA_EN —paises— como 4 tiendas con 1 apertura cada una; la cifra real es 23,206 tiendas y 1,596 aperturas netas) |
| Q4 | 1 | ¿Cuántas estaciones tenía OXXO GAS al cierre de 2024? | 571. | Al cierre de 2024, OXXO GAS tenía 571 estaciones. | correcta | ok |
| Q5 | 2 | ¿Qué división tuvo el mayor crecimiento porcentual de ingres | Empate nominal: ambas +14.2% (trampa deliberada). | No lo sé. | incorrecta | sin ninguna llamada a herramienta (5 planes con Cypher-SQL gigante truncado a los 150 tokens, relacion inventada 'OPERA_IN'; el modelo intenta una sola consulta comparativa en vez de dos simples) |
| Q6 | 2 | ¿Qué división aportó más ingresos en 2024, Proximidad Améric | Proximidad Américas, Ps. 307,200 millones vs Ps. 279,80 | No lo sé. | incorrecta | abstencion tras 3 consultas fallidas (SUM(r.valor) sobre OPERA_EN, que no tiene valor: 0 en Proximidad Americas; repitio la misma consulta para Coca-Cola FEMSA) |
| Q7 | 2 | ¿Qué creció más en usuarios, Spin by OXXO o Spin Premia? | Spin Premia en absoluto (24.6 millones vs 8.6 millones) | No lo sé. | incorrecta | abstencion tras 3 consultas fallidas ('Spin by OXXO' como Iniciativa con OPERA_EN y r.valor; ademas las cifras esperadas +27.5%/+24.9% no estan en el informe: techo 9/12) |
| Q8 | 2 | ¿En qué países opera Proximidad Américas según el informe? | México, Colombia, Chile, Perú (además de negocios en Es | Proximidad Américas opera en Perú, México, Colombia y Chile. | correcta | ok |
| Q9 | 3 | ¿Qué porcentaje de los ingresos consolidados 2024 representa | ~75.1% (39.3% + 35.8%, aproximado por redondeo). | No lo sé. | incorrecta | sin ninguna llamada a herramienta (5 planes con SQL anidado en calculator, truncado a los 150 tokens; ademas las cifras esperadas 39.3/35.8/75.1% no estan en el informe: techo 9/12) |
| Q10 | 3 | ¿Qué estrategia articula Retail, Bebidas y el ecosistema dig | FEMSA Forward; el ecosistema digital centrado en el cli | No lo sé. No inventes cifras. | incorrecta | abstencion tras 3 consultas fallidas (tipo de nodo inventado 'EcosistemaDigital', 'Retail' como Iniciativa y 'Bebidas' como Division; no descubrio 'FEMSA Forward') |
| Q11 | 3 | ¿Cómo evolucionó la iniciativa digital de FEMSA en 2024? | Digital@FEMSA evolucionó a Spin. | No lo sé. No inventes cifras. | incorrecta | evidencia correcta ignorada (la primera consulta devolvio {"x.nombre":"Spin"} y el agente repitio la misma consulta 2 veces mas y se abstuvo) |
| Q12 | 3 | ¿Qué división tuvo el año más complicado en 2024 según el pr | Salud: 'año complicado', +5.8% (el menor crecimiento de | No lo sé. | incorrecta | abstencion tras 3 consultas fallidas (error de sintaxis Cypher: 'LIMIT 1, 1' estilo SQL + ORDER BY sobre r.variacion_pct de OPERA_EN, que no tiene esa propiedad) |

## Por nivel

- Nivel 1: 1/4 correctas.
- Nivel 2: 1/4 correctas.
- Nivel 3: 0/4 correctas.

## Uso de herramientas por pregunta

| ID | Pasos | Herramientas (en orden) | Llamadas al plan | Segundos |
|----|-------|------------------------|------------------|----------|
| Q1 | 3 | graph_query, graph_query, graph_query | 4 | 866 |
| Q2 | 2 | graph_query, graph_query | 3 | 162 |
| Q3 | 2 | graph_query, graph_query | 3 | 179 |
| Q4 | 1 | graph_query | 2 | 108 |
| Q5 | 0 | (ninguna) | 5 | 653 |
| Q6 | 3 | graph_query, graph_query, graph_query | 4 | 249 |
| Q7 | 3 | graph_query, graph_query, graph_query | 4 | 255 |
| Q8 | 1 | graph_query | 2 | 88 |
| Q9 | 0 | (ninguna) | 5 | 633 |
| Q10 | 3 | graph_query, graph_query, graph_query | 4 | 237 |
| Q11 | 3 | graph_query, graph_query, graph_query | 4 | 195 |
| Q12 | 3 | graph_query, graph_query, graph_query | 4 | 290 |

## Modos de fallo

- ok: 2 (Q4, Q8)
- abstencion tras 3 consultas fallidas (nombre de indicador inventado: 'Ingresos Totales Consolidados' en vez de 'Ingresos Totales'; repitio la misma consulta 3 veces pese a la guardia anti-bucle): 1 (Q1)
- abstencion tras 2 consultas fallidas ('Spin by OXXO' tratado como Division con OPERA_EN y propiedad inventada 's.multiples_operadoras'; es un Formato de la Division Spin): 1 (Q2)
- evidencia mal interpretada (conto las 4 filas OPERA_EN —paises— como 4 tiendas con 1 apertura cada una; la cifra real es 23,206 tiendas y 1,596 aperturas netas): 1 (Q3)
- sin ninguna llamada a herramienta (5 planes con Cypher-SQL gigante truncado a los 150 tokens, relacion inventada 'OPERA_IN'; el modelo intenta una sola consulta comparativa en vez de dos simples): 1 (Q5)
- abstencion tras 3 consultas fallidas (SUM(r.valor) sobre OPERA_EN, que no tiene valor: 0 en Proximidad Americas; repitio la misma consulta para Coca-Cola FEMSA): 1 (Q6)
- abstencion tras 3 consultas fallidas ('Spin by OXXO' como Iniciativa con OPERA_EN y r.valor; ademas las cifras esperadas +27.5%/+24.9% no estan en el informe: techo 9/12): 1 (Q7)
- sin ninguna llamada a herramienta (5 planes con SQL anidado en calculator, truncado a los 150 tokens; ademas las cifras esperadas 39.3/35.8/75.1% no estan en el informe: techo 9/12): 1 (Q9)
- abstencion tras 3 consultas fallidas (tipo de nodo inventado 'EcosistemaDigital', 'Retail' como Iniciativa y 'Bebidas' como Division; no descubrio 'FEMSA Forward'): 1 (Q10)
- evidencia correcta ignorada (la primera consulta devolvio {"x.nombre":"Spin"} y el agente repitio la misma consulta 2 veces mas y se abstuvo): 1 (Q11)
- abstencion tras 3 consultas fallidas (error de sintaxis Cypher: 'LIMIT 1, 1' estilo SQL + ORDER BY sobre r.variacion_pct de OPERA_EN, que no tiene esa propiedad): 1 (Q12)

## Comparacion de los sistemas (12 preguntas, mismo scorer)

| Sistema | Correctas | Nivel 1 | Nivel 2 | Nivel 3 |
|---------|-----------|---------|---------|---------|
| Grafo (Relate, Cypher) | 12/12 | 4/4 | 4/4 | 4/4 |
| RAG agentico (este run) | 2/12 | 1/4 | 1/4 | 0/4 |
| RAG hibrido + expansion de queries | 5/12 | 2/4 | 2/4 | 1/4 |
| RAG hibrido BM25+denso (sin expansion) | 2/12 | 1/4 | 0/4 | 1/4 |
| RAG vectorial denso puro | 2/12 | 0/4 | 1/4 | 1/4 |
| MEMO run 2 (0.5B LoRA r=32 + ejecutivo 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |
| MEMO run 1 (0.5B LoRA r=8 + ejecutivo 0.5B) | 0/12 | 0/4 | 0/4 | 0/4 |

## Que cambio vs el RAG hibrido + expansion (5/12)

De 5/12 a 2/12: tres flips en contra y ninguno a favor. El expandido acierta Q1, Q4, Q6, Q8 y Q10; el agentico solo Q4 y Q8 (las dos que acierta son las dos consultas de plantilla directa al grafo).

- **Q1: expansion correcta -> agentico incorrecta (flip en contra, nombre de entidad).** El agente escribio 'Ingresos Totales Consolidados' en vez de 'Ingresos Totales' y repitio la misma consulta 3 veces. El lector RAG, con el dato en contexto, contesto 781,585.
- **Q6: expansion correcta -> agentico incorrecta (flip en contra, relacion equivocada).** El agente hizo SUM(r.valor) sobre OPERA_EN (que no tiene valor) y obtuvo 0; el lector RAG dio 307,197 y 279,793 exactos desde el contexto.
- **Q10: expansion correcta -> agentico incorrecta (flip en contra, esquema inventado).** El agente invento el tipo de nodo 'EcosistemaDigital' y trato 'Retail' como Iniciativa; no descubrio 'FEMSA Forward'. El lector RAG lo nombro desde la p.8 del informe.
- **Sin cambio:** Q4 (correcta en ambas: 571 estaciones) y Q8 (correcta en ambas: los 4 paises). Q2, Q3, Q5, Q7, Q9, Q11 y Q12 incorrectas en ambas (Q3 y Q12 por el lector en el RAG y por el agente aqui; Q2, Q7 y Q9 por cifras fuera del corpus en ambas).
- **Q11, el caso mas revelador:** la primera consulta del agente devolvio {"x.nombre":"Spin"} — la keyword que el scorer pide — y el agente repitio la misma consulta 2 veces mas y contesto 'No lo se'. Tenia la respuesta en la mano y no supo parar.
- **Uso de herramientas:** el agente nunca uso vector_search (0 veces) ni calculator (0 veces) pese a la instruccion de empezar siempre por vector_search y de calcular con la herramienta. Las 24 llamadas fueron graph_query, y 13 llamadas al plan no llegaron a ejecutarse (5 en Q5 y 5 en Q9 sin accion valida; 1 en Q6, Q7 y Q12 bloqueada por el tope de 3 pasos).

## Veredicto para el post

**RAG agentico: 2/12 — el bucle agentico no suma con un modelo de 1.5B; resta.** Con el mismo modelo, el mismo retrieval y el mismo scorer, el pipeline fijo con expansion saca 5/12 y el agente 2/12. La diferencia no es el acceso a los datos — el agente tiene el grafo exacto, que solo contesta 12/12, y el retrieval hibrido — sino la capacidad de *usar* las herramientas: componer Cypher sin mezclar relaciones, corregir un nombre tras un 'Sin resultados', parar cuando la evidencia ya esta y no repetir la misma llamada. El 1.5B falla en las cuatro.

Los dos aciertos (Q4, Q8) son consultas de plantilla copiadas al pie de la letra; todo lo que exige componer (comparar dos divisiones, calcular un porcentaje, descubrir un nombre) se cae. Las preguntas de comparacion/agregacion (Q5, Q9) son las peores: el modelo intenta una sola consulta gigante con sintaxis SQL anidada, se trunca a los 150 tokens y quema los 5 intentos sin ejecutar nada (653 s y 633 s de nada). Y el fallo mas caro es Q11: la evidencia correcta llego en la primera llamada y el agente no supo parar.

La lectura honesta para el post: 'agentico' no es una propiedad del bucle, es una propiedad del modelo que lo maneja. Con Qwen2.5-1.5B el agente es un RAG peor: mas lento (65 min vs ~26 min del expandido), mas fragil (una instruccion ignorada — vector_search primero — y un bug de router que tumbo la corrida en Q5) y con menos aciertos. El grafo 12/12 muestra que el dato esta; el agente muestra que llegar a el exige un modelo que sepa planificar, y el 1.5B no es ese modelo. El siguiente arreglo no es mas herramientas ni mas pasos: es un agente mas capaz, o un agente con menos grados de libertad (plantillas cerradas, verificacion automatica de la consulta antes de ejecutarla).

## Caveats (verificados, no supuestos)

- **El agente nunca uso vector_search ni calculator.** Las 24 llamadas fueron graph_query. La instruccion 'Empieza SIEMPRE con vector_search' fue ignorada en las 12 preguntas: el modelo va directo al grafo cuando la pregunta pide una cifra. El experimento, por tanto, mide un agente *solo-grafo*, no el bucle completo con sus tres herramientas.
- **Q11 tenia la respuesta y la perdio.** La primera consulta devolvio {"x.nombre":"Spin"}; el scorer pide la keyword 'Spin'. El agente repitio la consulta y se abstuvo. Con una regla de parada ('si la evidencia responde, contesta'), Q11 seria correcta y el run 3/12.
- **Q5 tumbo la corrida por un bug del router.** `_route` podia devolver 'plan' pero el mapa de `add_conditional_edges` no incluia 'plan': KeyError en Q5. Se arreglo anadiendo 'plan': 'plan' al mapa y un try/except por pregunta en run_agentic.py; la corrida se reanudo desde Q5 sin tocar Q1–Q4. El bug es del codigo, no del modelo, y esta documentado en `agent.py` y en el README.
- **La guardia anti-bucle tiene un agujero de espacios.** En Q2 la segunda llamada era identica salvo un espacio final y la guardia (comparacion exacta) no la detecto. Normalizar con strip() lo cerraria; se deja como esta para no cambiar el agente a mitad de corrida.
- **Q1 tardo 866 s por contencion de CPU** (otro agente corria un modelo en paralelo durante la primera parte). Sin contencion, las preguntas tardan 88–290 s (Q5 y Q9, con 5 planes fallidos, 653 s y 633 s). El tiempo total de preguntas es 65.3 min; el wall total con la caida y la reanudacion, ~69 min.
- **Q7 y Q9 son techo (9/12), igual que en los otros runs.** Las cifras esperadas de Q7 (+27.5%/+24.9%) vienen del comunicado 4Q24 y las de Q9 (39.3/35.8/75.1%) son participaciones calculadas: no estan en el informe. Q2 tambien pide +24.9% como parte de lo esperado. El techo real del agente sobre este set es 9/12, no 12/12.
- **Q3 es incorrecta por interpretacion, no por abstencion.** El agente conto 4 paises como 4 tiendas. Es el unico error con respuesta inventada (las demas son abstenciones): 0/12 con alucinacion segun el scorer porque '4' es un solo digito, pero la respuesta es falsa (23,206 tiendas, 1,596 aperturas).
- **Las plantillas funcionan cuando se copian.** Q4 y Q8 son copias literales de la plantilla de metrica operativa y de paises. Los fallos son desviaciones de la plantilla (relacion, nombre o propiedad cambiados) o intentos de consulta comparativa gigante. El diseno de 'plantillas cerradas' es el arreglo mas barato que el post puede proponer.

## Notas de ground-truth

- Q3: Nota ground-truth: Segun el MD&A (p.112): 1,596 tiendas netas nuevas en 2024; 23,206 es OXXO Mexico (la division Proximidad Americas suma 26,322 puntos de venta). La cifra '1,236 aperturas' no aparece en el informe. Si el modelo responde 1,596, es mas correcto que la respuesta esperada.
- Q6: Nota ground-truth: Valores exactos del MD&A: 307,197 y 279,793 (la ontologia los redondea). Se acepta el valor exacto o el redondeado.

## Alucinaciones detectadas

- Ninguna.

## Detalle de evidencia por pregunta (cobertura literal en lo que vio el agente)

| ID | Herramientas | Cifras esperadas en la evidencia |
|----|--------------|----------------------------------|
| Q1 | graph_query, graph_query, graph_query | 781585: no, 11.2: no |
| Q2 | graph_query, graph_query | 8.6: no, 24.9: no |
| Q3 | graph_query, graph_query | 23206: no, 1236: no |
| Q4 | graph_query | 571: si |
| Q5 |  | 14.2: no |
| Q6 | graph_query, graph_query, graph_query | 307200: no, 279800: no |
| Q7 | graph_query, graph_query, graph_query | 24.6: no, 8.6: no, 27.5: no, 24.9: no |
| Q8 | graph_query | (sin cifras esperadas) |
| Q9 |  | 75.1: no, 39.3: no, 35.8: no |
| Q10 | graph_query, graph_query, graph_query | (sin cifras esperadas) |
| Q11 | graph_query, graph_query, graph_query | (sin cifras esperadas) |
| Q12 | graph_query, graph_query, graph_query | 5.8: no |
