# Resultados RAG hibrido (BM25 + denso, RRF k=60) — FEMSA 2024

Sistema: retrieval hibrido top-5 paginas — BM25 Okapi (rank_bm25, k1=1.5, b=0.75) + all-MiniLM-L6-v2, fusion Reciprocal Rank Fusion k=60 — sobre las 151 paginas del informe; reader Qwen2.5-1.5B-Instruct local (bf16, CPU), mismo prompt y mismo condense que el baseline denso. Scorer identico a memo/evaluate.py (misma exactitud numerica, misma definicion de alucinacion, misma regla Q3 con 1,596). La unica diferencia con el baseline denso es el retrieval.

Correctas: **2/12**. Con alucinacion (numero no verificado en el informe): **1/12**. Retrieval top-5 con todas las cifras esperadas literales: **1/12** (denso puro: 2/12; ver Caveats).

| ID | Nivel | Pregunta | Esperado | Respuesta RAG | Veredicto | Modo de fallo |
|----|-------|----------|----------|---------------|-----------|---------------|
| Q1 | 1 | ¿Cuáles fueron los ingresos totales consolidados de FEMSA en | Ps. 781,585 millones, +11.2%. | Los ingresos totales consolidados de FEMSA en 2024 ascendieron a 781,585 millones de pesos mexicanos, representando un incremento del 11,2% frente al año anterior. | correcta | ok |
| Q2 | 1 | ¿Cuántos usuarios activos tuvo Spin by OXXO en 2024? | 8.6 millones (+24.9%). | 8.6 millones de usuarios activos en Spin by OXXO en 2024. | incorrecta | cifra esperada fuera del corpus (+24.9% no existe en el informe; el lector contesto bien lo preguntado) |
| Q3 | 1 | ¿Cuántas tiendas sumó Proximidad Américas al cierre de 2024  | 23,206 tiendas; 1,236 aperturas. | No lo sé. | incorrecta | fallo de retrieval (23,206 solo en p.21, no en top-5 hibrido [16,88,75,72,55]; el lector se abstuvo) |
| Q4 | 1 | ¿Cuántas estaciones tenía OXXO GAS al cierre de 2024? | 571. | Al cierre de 2024, OXXO GAS contaba con 12 estaciones. | incorrecta | fallo de retrieval + cifra mal atribuida (571 no en top-5 hibrido [83,100,85,88,92]; contesto '12 estaciones' con un 12 suelto del contexto) |
| Q5 | 2 | ¿Qué división tuvo el mayor crecimiento porcentual de ingres | Empate nominal: ambas +14.2% (trampa deliberada). | No lo sé. | incorrecta | fallo de retrieval (14.2 no en top-5 hibrido [28,88,128,80,100]; el lector se abstuvo) |
| Q6 | 2 | ¿Qué división aportó más ingresos en 2024, Proximidad Améric | Proximidad Américas, Ps. 307,200 millones vs Ps. 279,80 | En 2024, la División Proximidad Américas aportó más ingresos que Coca-Cola FEMSA. Los ingresos totales de la División Proximidad Américas fueron de Ps. 307,197 millones, mientras que los de Coca-Cola  | incorrecta | fallo del lector (307,197 bien; Coca-Cola FEMSA la puso en 128,736 en vez de 279,793: 128,736 es su utilidad bruta, no su ingreso; el contexto hibrido cambio vs el denso) |
| Q7 | 2 | ¿Qué creció más en usuarios, Spin by OXXO o Spin Premia? | Spin Premia en absoluto (24.6 millones vs 8.6 millones) | Spin Premia creció más en usuarios que Spin by OXXO. Según el informe, Spin Premia tiene 24.6 millones de activos y Spin by OXXO tiene 8.6 millones de activos, lo que significa que Spin Premia aumentó | incorrecta | cifra esperada fuera del corpus + aritmetica inventada (+27.5% y +24.9% no existen en el informe; el lector acerto direccion y absolutos, pero invento 'aumento un 275% (24.6 - 8.6 = 16)' y 'crecio un 50%'; el 16 cuenta como alucinacion) |
| Q8 | 2 | ¿En qué países opera Proximidad Américas según el informe? | México, Colombia, Chile, Perú (además de negocios en Es | Según el informe, Proximidad Américas opera en México y América Latina. | incorrecta | fallo del lector (los 4 paises estaban en contexto; listo solo Mexico y 'America Latina') |
| Q9 | 3 | ¿Qué porcentaje de los ingresos consolidados 2024 representa | ~75.1% (39.3% + 35.8%, aproximado por redondeo). | No lo sé. | incorrecta | cifra esperada fuera del corpus (39.3/35.8/75.1% no existen); el lector se abstuvo en vez de inventar la suma |
| Q10 | 3 | ¿Qué estrategia articula Retail, Bebidas y el ecosistema dig | FEMSA Forward; el ecosistema digital centrado en el cli | La estrategia que articula Retail, Bebidas y el ecosistema digital de FEMSA se basa en un modelo de crecimiento sostenible y duradero. Para crear este ecosistema, FEMSA ha implementado una serie de ac | incorrecta | fallo de retrieval ('FEMSA Forward' no en top-5 hibrido [13,75,4,34,117]; respuesta vaga sin el nombre) |
| Q11 | 3 | ¿Cómo evolucionó la iniciativa digital de FEMSA en 2024? | Digital@FEMSA evolucionó a Spin. | En 2024, FEMSA avanzó significativamente en su estrategia digital al transformar la marca Digital@FEMSA en Spin. Esta evolución incluyó un cambio en el branding, imagen y identidad, así como una forta | correcta | ok |
| Q12 | 3 | ¿Qué división tuvo el año más complicado en 2024 según el pr | Salud: 'año complicado', +5.8% (el menor crecimiento de | No lo sé. | incorrecta | fallo de retrieval (Health +5.8% en pp. 5/112/114, no en top-5 hibrido [88,16,75,99,96]; el lector se abstuvo) |

## Por nivel

- Nivel 1: 1/4 correctas.
- Nivel 2: 0/4 correctas.
- Nivel 3: 1/4 correctas.

## Modos de fallo

- ok: 2 (Q1, Q11)
- cifra esperada fuera del corpus (+24.9% no existe en el informe; el lector contesto bien lo preguntado): 1 (Q2)
- fallo de retrieval (23,206 solo en p.21, no en top-5 hibrido [16,88,75,72,55]; el lector se abstuvo): 1 (Q3)
- fallo de retrieval + cifra mal atribuida (571 no en top-5 hibrido [83,100,85,88,92]; contesto '12 estaciones' con un 12 suelto del contexto): 1 (Q4)
- fallo de retrieval (14.2 no en top-5 hibrido [28,88,128,80,100]; el lector se abstuvo): 1 (Q5)
- fallo del lector (307,197 bien; Coca-Cola FEMSA la puso en 128,736 en vez de 279,793: 128,736 es su utilidad bruta, no su ingreso; el contexto hibrido cambio vs el denso): 1 (Q6)
- cifra esperada fuera del corpus + aritmetica inventada (+27.5% y +24.9% no existen en el informe; el lector acerto direccion y absolutos, pero invento 'aumento un 275% (24.6 - 8.6 = 16)' y 'crecio un 50%'; el 16 cuenta como alucinacion): 1 (Q7)
- fallo del lector (los 4 paises estaban en contexto; listo solo Mexico y 'America Latina'): 1 (Q8)
- cifra esperada fuera del corpus (39.3/35.8/75.1% no existen); el lector se abstuvo en vez de inventar la suma: 1 (Q9)
- fallo de retrieval ('FEMSA Forward' no en top-5 hibrido [13,75,4,34,117]; respuesta vaga sin el nombre): 1 (Q10)
- fallo de retrieval (Health +5.8% en pp. 5/112/114, no en top-5 hibrido [88,16,75,99,96]; el lector se abstuvo): 1 (Q12)

## Comparacion de los sistemas (12 preguntas, mismo scorer)

| Sistema | Correctas | Nivel 1 | Nivel 2 | Nivel 3 |
|---------|-----------|---------|---------|---------|
| Grafo (Relate, Cypher) | 12/12 | 4/4 | 4/4 | 4/4 |
| RAG hibrido BM25+denso (este run) | 2/12 | 1/4 | 0/4 | 1/4 |
| RAG vectorial denso puro | 2/12 | 0/4 | 1/4 | 1/4 |
| MEMO run 2 (0.5B LoRA r=32 + ejecutivo 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |
| MEMO run 1 (0.5B LoRA r=8 + ejecutivo 0.5B) | 0/12 | 0/4 | 0/4 | 0/4 |

## Que cambio vs el baseline denso (2/12)

Mismo puntaje, distinta composicion: el hibrido acierta Q1 y Q11; el denso acertaba Q6 y Q11. Un flip a favor, un flip en contra, y las otras 10 preguntas igual.

- **Q1: denso incorrecta -> hibrido correcta (unico flip a favor).** El hibrido trajo el 11.2% al top-5 (paginas [88,113,16,112,115]) y el lector eligio 781,585 en vez de los 40,236 del denso. Q1 en el denso era fallo del lector, no de retrieval (781,585 si estaba en contexto); el hibrido lo arreglo de rebote, al subir la pagina con el 11.2%.
- **Q6: denso correcta -> hibrido incorrecta (flip en contra).** El contexto hibrido ([88,112,79,27,107] vs [30,114,10,27,112] del denso) cambio la tabla que vio el lector y tomo 128,736 como ingreso de Coca-Cola FEMSA. 128,736 es una cifra real del informe —su utilidad bruta, no su ingreso (279,793)—, asi que no cuenta como alucinacion, pero la respuesta esta mal. El hibrido arreglo el retrieval de Q1 y rompió el de Q6.
- **Q3, Q4, Q10: igual de mal.** Los 3 fallos de retrieval del denso siguen fallando: 23,206 solo esta en la p.21, 571 en pp. 21/114/126 y 'FEMSA Forward' en pp. 3/5/8/12/14/42/101/115, y ninguna de esas paginas entro al top-5 hibrido. BM25 no puede emparejar lo que no esta en la pregunta: '571' y 'Forward' no aparecen en las preguntas, y 'proximidad'/'estaciones' estan en espanol mientras el corpus esta en ingles.
- **Q5: peor retrieval, mejor conducta.** El denso SI traia el 14.2 en top-5 y el lector divago (alucinacion '114'); el hibrido ya no trae el 14.2 y el lector se abstuvo honestamente ('No lo se'). Sigue incorrecta, pero sin alucinar.
- **Q12: peor retrieval.** El denso traia un 5.8 literal (margen de utilidad neta, espurio, no el crecimiento de Salud); el hibrido ni ese trae. El lector se abstuvo en ambas.
- **Q9: mejor conducta.** El denso invento la suma 43,265; el hibrido se abstuvo. Sigue incorrecta (las cifras no existen en el informe), pero sin alucinar.
- **Q8: peor lector.** El denso listo Mexico y Colombia (2 de 4); el hibrido, con los 4 paises en contexto, listo solo Mexico y 'America Latina'.
- **Q7: peor conducta.** El denso contesto solo direccion y absolutos (24.6 vs 8.6) y paro ahi, sin alucinar; el hibrido siguio e invento 'aumento un 275% (24.6 - 8.6 = 16)' y 'crecio un 50%'. El 16 cuenta como alucinacion. Sigue incorrecta en ambas (los porcentajes no estan en el informe).
- **Sin cambio:** Q2 (cifra fuera del corpus), Q11 (correcta en ambas).
- **Cobertura literal ALL en top-5:** denso 2/12 (Q5, Q12) -> hibrido 1/12 (Q1). El hibrido gano Q1 y perdio Q5 y Q12.

## Veredicto para el post

**RAG hibrido BM25+denso: 2/12 — el mismo puntaje que el denso puro, con un perfil distinto.** No cerro la brecha de retrieval: de los 4 fallos de retrieval del denso (Q3, Q4, Q10, Q12), el hibrido no arreglo ninguno, y perdio Q5. Lo que arreglo fue Q1, que en el denso era fallo del lector (781,585 si estaba en contexto): el hibrido ademas trajo el 11.2% y el lector eligio bien. BM25 reordena, pero no puede arreglar un desajuste de vocabulario: las preguntas estan en espanol, el corpus en ingles, y los tokens discriminativos viven en la respuesta, no en la pregunta. El arreglo directo seria traducir o expandir la pregunta con sinonimos del dominio, o chunkear por secciones en vez de por pagina — no otro ranker lexico.

Lo que SI cambio es la conducta, en ambas direcciones: alucinaciones de 2/12 (denso: Q5 '114', Q9 '43265') a 1/12 (hibrido: Q7 '16'). Q5 y Q9 ya no alucinan — el lector se abstiene — pero Q7 ahora inventa porcentajes ('aumento un 275%', 'crecio un 50%') que el denso no inventaba. El mismo lector, con distinto contexto, divaga distinto. Abstenerse no suma puntos con este scorer, asi que el marcador no se mueve. La moraleja se mantiene y se afina: la exactitud de cifras exige que el dato llegue verificado al lector; cambiar el ranker sin cambiar el vocabulario de la pregunta es mover los muebles.

## Caveats (verificados, no supuestos)

- El diagnostico de retrieval con tolerancia de redondeo del scorer (±0.1%) decia 5/12 con todas las cifras cubiertas; la cobertura literal estricta en top-5 es 1/12 (Q1) — la tolerancia marca 'cubierto' un 27.5% cuando solo hay un 27 suelto, igual que en el baseline denso. La tabla de detalle usa cobertura literal; la clasificacion de fallos se verifico a mano. En Q6 las cifras estan como valores exactos (307,197/279,793) mientras el campo esperado trae los redondeados de la ontologia (307,200/279,800).
- El hibrido NO cerro la brecha de retrieval del denso: Q3, Q4 y Q10 siguen sin traer la cifra al top-5. La razon es estructural y se verifico en los rankings: las preguntas estan en espanol y el corpus en ingles, asi que BM25 solo empareja tokens exactos (numeros, nombres propios) y los tokens discriminativos viven en la respuesta, no en la pregunta ('571' no esta en la pregunta de Q4; 'Forward' no esta en la de Q10; '23,206' no esta en la de Q3). BM25 reordena, pero no puede inventar el vocabulario que falta. El arreglo directo seria traducir la pregunta o expandirla con sinonimos del dominio, no otro ranker lexico.
- El condensado de contexto (ventanas de ~1,800 caracteres por chunk, 2 ventanas) fue necesario porque el prompt de 5 paginas completas (~4.5k tokens) tarda >6 min por pregunta en CPU. Es el mismo workaround que en el baseline denso, con el mismo costo medido: cifras que estan en el top-5 pueden no llegar al lector. Es un costo del workaround CPU, no del metodo RAG en si.
- La pregunta esta en espanol y el corpus en ingles; el retrieval es cross-lingual con un embedder entrenado en ingles (all-MiniLM-L6-v2). Q12 exige mapear 'Salud' -> 'Health'.
- Q2 no pregunta el crecimiento (+24.9%) pero el scorer lo exige; la respuesta '8.6 millones' es correcta para la pregunta tal como esta hecha. Misma severidad que en MEMO, asi que los puntajes son comparables.

## Notas de ground-truth

- Q3: Nota ground-truth: Segun el MD&A (p.112): 1,596 tiendas netas nuevas en 2024; 23,206 es OXXO Mexico (la division Proximidad Americas suma 26,322 puntos de venta). La cifra '1,236 aperturas' no aparece en el informe. Si el modelo responde 1,596, es mas correcto que la respuesta esperada.
- Q6: Nota ground-truth: Valores exactos del MD&A: 307,197 y 279,793 (la ontologia los redondea). Se acepta el valor exacto o el redondeado.

## Alucinaciones detectadas

- Q7: numeros no verificados en el informe: 16

## Detalle de retrieval por pregunta (cobertura literal)

| ID | Paginas recuperadas (top-5) | Cifras en top-5 | Cifras en contexto (condensado) |
|----|-----------------------------|-----------------|---------------------------------|
| Q1 | [88, 113, 16, 112, 115] | 781585: si, 11.2: si | 781585: si, 11.2: si |
| Q2 | [35, 5, 33, 36, 4] | 8.6: si, 24.9: no | 8.6: si, 24.9: no |
| Q3 | [16, 88, 75, 72, 55] | 23206: no, 1236: no | 23206: no, 1236: no |
| Q4 | [83, 100, 85, 88, 92] | 571: no | 571: no |
| Q5 | [28, 88, 128, 80, 100] | 14.2: no | 14.2: no |
| Q6 | [88, 112, 79, 27, 107] | 307200: no, 279800: no | 307200: no, 279800: no |
| Q7 | [35, 5, 33, 19, 13] | 24.6: si, 8.6: si, 27.5: no, 24.9: no | 24.6: si, 8.6: si, 27.5: no, 24.9: no |
| Q8 | [88, 17, 21, 120, 80] | (sin cifras esperadas) | (sin cifras esperadas) |
| Q9 | [28, 128, 107, 88, 75] | 75.1: no, 39.3: no, 35.8: no | 75.1: no, 39.3: no, 35.8: no |
| Q10 | [13, 75, 4, 34, 117] | (sin cifras esperadas) | (sin cifras esperadas) |
| Q11 | [12, 100, 151, 34, 128] | (sin cifras esperadas) | (sin cifras esperadas) |
| Q12 | [88, 16, 75, 99, 96] | 5.8: no | 5.8: no |
