# Resultados RAGAS (estilo) — FEMSA 2024

Sistema: las tres variantes RAG del experimento — vectorial denso, híbrido
BM25+denso (RRF k=60), híbrido+expansión de queries — mismo reader
Qwen2.5-1.5B-Instruct local en CPU, mismos 12 preguntas y mismos traces que
`results.md`, `results_hybrid.md` y `results_expanded.md`. Scorer de
exactitud idéntico a `memo/evaluate.py` (para el análisis de
acuerdo/desacuerdo de la última sección).

## Método: determinista, sin juez LLM

La librería `ragas` 0.4.4 está instalada y se probó con el único juez
disponible — Qwen2.5-1.5B-Instruct local en CPU — en una pregunta (ver
"Smoke test con la librería real" abajo): **los 4 jobs (uno por métrica)
expiraron por timeout (180 s cada uno) sin completar una sola métrica**.
Por eso las cuatro métricas se calculan determinísticamente con las
definiciones estándar de RAGAS, adaptadas a chunks de página completa, a
partir de los traces de retrieval, los traces de respuesta (con el contexto
condensado que el lector realmente vio) y el conjunto de evidencia auditable
en `ragas/ground_truth_evidence.json` (derivado por grep literal sobre las
151 páginas del informe).

- **context_precision** = páginas recuperadas que son páginas de evidencia /
  páginas recuperadas (top-5). Definición estricta: una página sobre el tema
  que no contiene la cifra exacta cuenta como no relevante → este número es
  un **límite inferior** de lo que marcaría un juez LLM.
- **context_recall** = ítems de evidencia con al menos una página en el top-5
  / ítems de evidencia. Q9 no tiene ningún ítem en el corpus → recall
  **null, no 0**.
- **faithfulness** = claims de la respuesta apoyados por el contexto
  recuperado / claims totales. Claims = números + entidades (divisiones,
  iniciativas, países). Un número está apoyado si su valor normalizado
  aparece en el contexto condensado (encabezados "[Página N]" eliminados
  antes). Las entidades se comparan bilingües (la respuesta está en
  español, el corpus en inglés: "Proximidad Américas" ≡ "Proximity
  Americas", "Salud" ≡ "Health", "Brasil" ≡ "Brazil", etc.) y con el
  whitespace normalizado (el texto del PDF trae tabs dentro de los nombres).
  Sin ese arreglo, "Proximidad Américas" se marcaba sin apoyo aunque el
  concepto sí estaba en el contexto — se corrigió tras auditar los claims
  (ver `compute_metrics.py`). Abstención ("No lo sé") = 0 claims →
  faithfulness 1.0 por convención. **Límite: no detecta la mala
  atribución** — un número real usado para la división equivocada cuenta
  como apoyado. La faithfulness determinista es un **límite superior**.
- **answer_relevancy (proxy)** = similitud coseno entre embeddings
  all-MiniLM-L6-v2 de la pregunta y la respuesta. RAGAS la define con
  preguntas generadas por LLM; ese paso NO se usa aquí. Es la más débil de
  las cuatro en este set (ver "Caveats").

## Resumen por variante

| Variante | context_precision | context_recall (media por pregunta) | context_recall (pooled, ítems) | faithfulness | answer_relevancy (proxy) | Exactitud |
|----------|------------------:|------------------------------------:|-------------------------------:|-------------:|-------------------------:|----------:|
| RAG vectorial denso | 0.183 | 0.455 | 8/17 = 0.471 | 0.969 | 0.609 | 2/12 |
| RAG híbrido BM25+denso | 0.150 | 0.500 | 9/17 = 0.529 | 0.967 | 0.628 | 2/12 |
| RAG híbrido + expansión | 0.400 | 1.000 | 17/17 = 1.000 | 0.975 | 0.656 | 5/12 |

## Detalle por pregunta

### RAG vectorial denso (exactitud 2/12: Q6, Q11)

| ID | context_precision | context_recall | faithfulness (n claims) | answer_relevancy | Exactitud |
|----|------------------:|---------------:|------------------------:|-----------------:|-----------|
| Q1 | 0.20 | 0.50 (1/2) | 1.00 (3) | 0.54 | no |
| Q2 | 0.40 | 1.00 (1/1) | 1.00 (1) | 0.11 | no |
| Q3 | 0.00 | 0.00 (0/2) | 1.00 (0, abstención) | 0.26 | no |
| Q4 | 0.00 | 0.00 (0/1) | 1.00 (0, abstención) | 0.29 | no |
| Q5 | 0.00 | 0.00 (0/1) | 0.80 (5) | 0.71 | no |
| Q6 | 0.40 | 1.00 (2/2) | 1.00 (5) | 0.77 | **sí** |
| Q7 | 0.60 | 0.50 (2/4) | 1.00 (4) | 0.85 | no |
| Q8 | 0.20 | 1.00 (1/1) | 1.00 (3) | 0.87 | no |
| Q9 | 0.00 | n/a | 0.83 (6) | 0.76 | no |
| Q10 | 0.00 | 0.00 (0/1) | 1.00 (1) | 0.83 | no |
| Q11 | 0.40 | 1.00 (1/1) | 1.00 (6) | 0.57 | **sí** |
| Q12 | 0.00 | 0.00 (0/1) | 1.00 (3) | 0.76 | no |

### RAG híbrido BM25+denso (exactitud 2/12: Q1, Q11)

| ID | context_precision | context_recall | faithfulness (n claims) | answer_relevancy | Exactitud |
|----|------------------:|---------------:|------------------------:|-----------------:|-----------|
| Q1 | 0.20 | 1.00 (2/2) | 1.00 (3) | 0.65 | **sí** |
| Q2 | 0.40 | 1.00 (1/1) | 1.00 (3) | 0.78 | no |
| Q3 | 0.00 | 0.00 (0/2) | 1.00 (0, abstención) | 0.26 | no |
| Q4 | 0.00 | 0.00 (0/1) | 1.00 (3) | 0.88 | no |
| Q5 | 0.00 | 0.00 (0/1) | 1.00 (0, abstención) | 0.34 | no |
| Q6 | 0.20 | 1.00 (2/2) | 1.00 (5) | 0.77 | no |
| Q7 | 0.60 | 0.50 (2/4) | 0.80 (10) | 0.78 | no |
| Q8 | 0.20 | 1.00 (1/1) | 1.00 (2) | 0.89 | no |
| Q9 | 0.00 | n/a | 1.00 (0, abstención) | 0.21 | no |
| Q10 | 0.00 | 0.00 (0/1) | 1.00 (2) | 0.87 | no |
| Q11 | 0.20 | 1.00 (1/1) | 0.80 (5) | 0.72 | **sí** |
| Q12 | 0.00 | 0.00 (0/1) | 1.00 (0, abstención) | 0.38 | no |

### RAG híbrido + expansión (exactitud 5/12: Q1, Q4, Q6, Q8, Q10)

| ID | context_precision | context_recall | faithfulness (n claims) | answer_relevancy | Exactitud |
|----|------------------:|---------------:|------------------------:|-----------------:|-----------|
| Q1 | 0.40 | 1.00 (2/2) | 1.00 (4) | 0.58 | **sí** |
| Q2 | 0.60 | 1.00 (1/1) | 1.00 (1) | 0.11 | no |
| Q3 | 0.60 | 1.00 (2/2) | 1.00 (3) | 0.65 | no |
| Q4 | 0.40 | 1.00 (1/1) | 1.00 (3) | 0.76 | **sí** |
| Q5 | 0.20 | 1.00 (1/1) | 1.00 (8) | 0.60 | no |
| Q6 | 0.40 | 1.00 (2/2) | 0.80 (5) | 0.76 | **sí** |
| Q7 | 0.80 | 1.00 (4/4) | 1.00 (4) | 0.80 | no |
| Q8 | 0.40 | 1.00 (1/1) | 0.89 (19) | 0.69 | **sí** |
| Q9 | 0.00 | n/a | 1.00 (5) | 0.75 | no |
| Q10 | 0.40 | 1.00 (1/1) | 1.00 (2) | 0.84 | **sí** |
| Q11 | 0.20 | 1.00 (1/1) | 0.80 (5) | 0.61 | no (correcta con alucinación*) |
| Q12 | 0.40 | 1.00 (1/1) | 1.00 (3) | 0.73 | no |

\* Q11 expandida: la sustancia es correcta (Digital@FEMSA → Spin) pero el
scorer la marca "correcta con alucinación" por el '328' (32.8% real del
informe, p.34) — artefacto del scorer, no del lector. Por eso el conteo
oficial es 5/12 y no 6/12.

## Lectura: retrieval vs lector

- **El retrieval se arregló por completo con la expansión.** Context recall
  (pooled): 8/17 (denso) → 9/17 (híbrido) → **17/17 (expandido)**. La
  expansión no solo subió el recall medio a 1.00: en la variante expandida
  TODAS las preguntas contestables traen la evidencia al top-5. Los 7
  errores que quedan son del lector con el dato enfrente (Q3, Q5, Q12),
  cifras fuera del corpus (Q2, Q7, Q9) o la Q11 penalizada por un artefacto
  del scorer. El cuello de botella ya no es el retrieval: es el lector.
- **Context precision es baja en las tres variantes (0.15–0.40) y no hay
  que leerla como "el retrieval es malo".** Con chunks de página completa y
  una definición estricta (la página debe contener la cifra exacta), solo
  ~2 de cada 5 páginas recuperadas son páginas de evidencia; las demás son
  páginas sobre el tema sin la cifra. Un juez LLM marcaría varias de esas
  como relevantes. La precision aquí es un límite inferior, y su valor está
  en la comparación entre variantes (la expansión la duplica: 0.15–0.18 →
  0.40), no en el nivel absoluto.
- **Faithfulness es alta en las tres variantes (0.97) y NO sigue a la
  exactitud.** El lector casi no inventa números: los **elige mal** del
  contexto. Casos verificados claim por claim en `metrics_ragas.json`:
  - Denso Q1: F=1.00 pero incorrecta — contestó 40,236 (+31%), ambos números
    reales del contexto, pero no son el ingreso consolidado (781,585).
  - Expandido Q9: F=1.00 pero incorrecta — contestó 11.2% como si fuera la
    participación; 11.2% está en el contexto, es el crecimiento consolidado.
  - Expandido Q3: F=1.00 pero incorrecta — 26,322 y "438 nuevas tiendas",
    ambos en contexto, pero la respuesta pedida era 23,206 y 1,596.
  - Expandido Q12: F=1.00 pero incorrecta — 14.2% en contexto, pero es el
    crecimiento de otras divisiones, no de Salud (+5.8%).
  - Las únicas respuestas con algún claim sin apoyo: denso Q9 (F=0.83, solo
    '43265' sin apoyo — la suma inventada; los demás números son reales del
    contexto), denso Q5 (F=0.80, solo '114' sin apoyo — número de página
    filtrado como dato), híbrido Q7 (F=0.80, '275' y '16' sin apoyo — la
    aritmética inventada), híbrido Q11 (F=0.80, 'Spin' sin apoyo: el nombre
    no está en el contexto condensado que vio el lector), expandido Q11
    (F=0.80, '31' sin apoyo — fragmento de "31.5%" cortado a los 100
    tokens), expandido Q8 (F=0.89, España y Honduras sin apoyo en el
    contexto condensado).
- **La abstención sale bien parada en faithfulness (1.00, 0 claims) y mal
  en relevancy (0.21–0.38).** Es lo correcto en ambos casos: no afirmar nada
  falso es fiel, y "No lo sé" no responde la pregunta. El scorer de
  exactitud no premia abstenerse; RAGAS tampoco debería.
- **Answer relevancy (proxy) es la métrica menos fiable de las cuatro en
  este set.** Premia respuestas largas y fluidas aunque estén mal (híbrido
  Q4: AR=0.88 para "12 estaciones", incorrecta; denso Q10: AR=0.83 para una
  respuesta vaga sin el nombre) y castiga respuestas cortas y numéricas
  aunque sean correctas (Q2: AR=0.11 en denso y expandido para "8.6
  millones", que es lo preguntado). La relevancy por coseno de embeddings
  no distingue "habla del tema" de "contesta la pregunta". No usar el proxy
  para decisiones; la relevancy real de RAGAS (preguntas generadas por LLM)
  se comportaría distinto.

## RAGAS vs exactitud: dónde coinciden y dónde no

| Comparación | Veredicto |
|---|---|
| Context recall vs exactitud | **Coinciden en dirección.** El recall es la métrica que más se mueve con la exactitud (0.47 → 0.53 → 1.00 pooled, frente a 2/12 → 2/12 → 5/12). Es la única de las cuatro que separa a la variante expandida de las otras dos. |
| Context precision vs exactitud | **Coinciden en dirección, con ruido.** La expansión la duplica (0.15–0.18 → 0.40) junto con la exactitud, pero denso (0.183) supera al híbrido (0.150) con el mismo 2/12. |
| Faithfulness vs exactitud | **Desacuerdan sistemáticamente.** F está en 0.97 mientras la exactitud está en 2/12–5/12. El desacuerdo es informativo, no un error: el modo de fallo dominante es **selección** (elegir el número equivocado del contexto), no **invención**. Un sistema puede ser fiel y estar mal. |
| Answer relevancy (proxy) vs exactitud | **Desacuerdan en ambas direcciones.** Respuestas incorrectas con AR alto (híbrido Q4: 0.88) y correctas con AR bajo (Q2: 0.11). El proxy no sirve aquí; no es un veredicto sobre la métrica real de RAGAS. |

La lectura para el post: **RAGAS confirma el diagnóstico del scorer y lo
afina.** El problema del RAG clásico en este set no es que el modelo
alucine (faithfulness 0.97) ni que el retrieval falle después de la
expansión (recall 1.00): es que el lector, con el dato enfrente, no sabe
elegirlo. Ninguna de las cuatro métricas de RAGAS mide esa selección — por
eso la exactitud numérica sigue siendo la métrica que decide en este
experimento, y por eso el grafo gana 12/12: no selecciona, consulta.

## Smoke test con la librería real

`smoke_test_ragas.py`: `ragas` 0.4.4 + juez Qwen2.5-1.5B-Instruct local
(bf16, CPU) + embeddings all-MiniLM-L6-v2, una pregunta (denso Q1), las 4
métricas. Instalación: `ragas` 0.4.4 + `langchain-huggingface` ~8 min;
`torch` CPU + `transformers` 5.24.1 ~3 min más (el venv de `ragas` no traía
`transformers`; el índice CPU de PyTorch no lo sirve, hubo que instalarlo
desde PyPI por separado).

Resultado (log: `smoke_test.log`):

- Juez cargado: 88.1 s. Embeddings cargados: 104.3 s.
- Evaluación: **los 4 jobs (uno por métrica) lanzaron `TimeoutError()`** —
  el timeout por job de `ragas` es de 180 s y el juez local no completó
  ninguna métrica en ese tiempo. Progreso: 4/4 en 03:00, sin valores.
- El proceso fue interrumpido después por un reinicio del runtime antes de
  imprimir la tabla final; la tabla habría sido NaN en las 4 métricas
  (4/4 jobs fallidos), así que no se perdió ningún número.

Conclusión: con un juez de 1.5B en CPU, un run completo de `ragas` (36
respuestas × 4 métricas, cada métrica con varias generaciones secuenciales)
tomaría horas y cada llamado corre riesgo de timeout. Subir el timeout de
`ragas` podría dejar pasar una pregunta, pero no cambia la escala del
problema. **Los números de este directorio son RAGAS en definición, no
RAGAS con juez LLM, y así deben citarse en el post** ("juez: ninguno —
cálculo determinista"; no son equivalentes a un run juzgado por GPT-4).

## Caveats (verificados, no supuestos)

- **Q9, recall n/a.** 39.3/35.8/75.1% no existen en el informe (verificado
  por grep: ninguna página). Son participaciones calculadas. El recall no
  es 0: la pregunta no tiene evidencia que recuperar. El techo de 9/12 se
  mantiene.
- **Q2 y Q7, recall parcial por diseño.** El recall cuenta solo los ítems
  que existen en el informe. Q2: 8.6M sí (pp. 13/33/35), +24.9% no (fuera
  del corpus). Q7: 24.6M y 8.6M sí, más los ítems alternativos del propio
  informe 32.8% (p.34) y 31.5% (pp. 34/113) — la variante expandida los
  cubre los 4 (p.34 en su top-5) y aun así el scorer la marca incorrecta,
  porque exige 27.5/24.9% del comunicado 4Q24. El recall de Q7 expandido
  (1.00) contra su exactitud (no) es el conflicto de ground-truth en
  números.
- **Q5, evidencia parcial no contada.** Solo pp. 5 y 112 traen AMBAS
  divisiones en +14.2% (el empate exige las dos juntas). Pp. 25, 27, 29 y
  113 traen solo una. El denso recuperó p.27 (Coca-Cola 14.2% sola) y su
  recall es 0.00 — estricto pero correcto para la pregunta del empate.
- **Q10, corrección durante la auditoría.** El grep inicial incluía p.13
  ("FEMSA Forward") y daba recall 1.00 en denso e híbrido. La revisión del
  contexto mostró que p.13 solo trae el encabezado del diagrama ("FEMSA
  FORWARD & HEALTH"), no la estrategia. El conjunto corregido es pp. 3, 5,
  8, 12, 14, 42, 115, 116; el recall de Q10 pasó a 0.00 en denso e híbrido
  y se mantuvo 1.00 en expandido (pp. 8 y 12 en su top-5) — en línea con
  los veredictos de `results.md` y `results_hybrid.md`.
- **Q12, el 5.8% espurio.** "5.8%" aparece en pp. 5, 108, 109, 112, 113,
  114, pero solo 5, 112 y 114 son crecimiento de Salud; el resto es margen
  de utilidad neta. El ground truth usa solo [5, 112, 114]. El denso
  recuperó un 5.8% espurio y su recall es 0.00 — correcto.
- **Expandido Q8, faithfulness 0.89 (19 claims).** La respuesta enumera 18
  países del grupo; solo España y Honduras no están en el contexto
  condensado (el condense de ~1,800 caracteres por chunk no los incluyó).
  Es el costo del condense visible en faithfulness, no un fallo de
  invención: los países listados son reales, solo dos no estaban en el
  contexto que vio el lector.
- **La faithfulness determinista no ve la mala atribución.** Híbrido Q6
  (128,736 = utilidad bruta citada como ingreso) tiene F=1.00: los 5
  claims, incluido 128,736, están en el contexto — la atribución errónea
  es completamente invisible para esta métrica. Un juez LLM la detectaría.
  Los valores de F de este directorio son techos, no mediciones exactas de
  fidelidad.

## Notas de ground-truth

- Q3 usa 1,596 (MD&A p.112) en vez de 1,236 (no aparece en el informe) —
  misma regla que el scorer de exactitud.
- Q6 acepta 307,197/279,793 exactos (la ontología redondea a 307,200/279,800).
- Q7 registra 32.8%/31.5% (p.34) como ground truth alternativo del informe.
- Q8: las 4 países juntas en pp. 10, 21, 22, 55, 92, 121.
- Q11: evidencia en pp. 5 ("Digital@FEMSA becomes Spin") y 34.
