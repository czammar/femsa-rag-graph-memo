# RAG vectorial (baseline) — FEMSA 2024

Baseline de RAG clásico para la comparación del post: mismo corpus
(`pipeline/data/chunks.jsonl`, 151 páginas del informe anual FEMSA 2024 en
inglés), mismas 12 preguntas (`memo/eval_questions.json`) y mismo scorer
(`memo/evaluate.py`) que el grafo y que MEMO. Sin este baseline el README
del repo decía "RAG vectorial: pendiente (no se corrió; no se inventan
cifras)". Ahora hay cifras reales.

## Resultado

**RAG vectorial: 2/12 correctas** (Q6, Q11), 2/12 con alucinación.
Por nivel: 1: 0/4, 2: 1/4, 3: 1/4.

| Sistema | Correctas | Nivel 1 | Nivel 2 | Nivel 3 |
|---------|-----------|---------|---------|---------|
| Grafo (Relate, Cypher) | 12/12 | 4/4 | 4/4 | 4/4 |
| RAG híbrido + expansión de queries | **5/12** | 2/4 | 2/4 | 1/4 |
| RAG híbrido BM25+denso | 2/12 | 1/4 | 0/4 | 1/4 |
| RAG vectorial (este baseline) | 2/12 | 0/4 | 1/4 | 1/4 |
| MEMO run 2 | 2/12 | 1/4 | 1/4 | 0/4 |
| MEMO run 1 | 0/12 | 0/4 | 0/4 | 0/4 |

El veredicto completo y la clasificación de fallos verificada cifra por
cifra están en `results.md`. Resumen: retrieval falla en 4 (Q3, Q4, Q10,
Q12), 3 preguntas tienen cifras esperadas que no existen en el corpus
(Q2, Q7, Q9: +24.9%/+27.5% son del comunicado 4Q24; 39.3/35.8/75.1% son
participaciones calculadas), el lector falla en 3 (Q1, Q5, Q8) y Q9 suma
cifras equivocadas (43,265). Techo real de RAG sobre este set tal como
está especificado: 9/12. **La expansión de queries (abajo) sube el RAG
a 5/12: el problema era el vocabulario de la pregunta, no el ranker.**

## Cómo se corrió

Sistema: retrieval top-5 páginas con `all-MiniLM-L6-v2` (CPU) sobre las
151 páginas; reader `Qwen2.5-1.5B-Instruct` local (bf16, CPU), el mismo
modelo que hizo de EXECUTIVE en MEMO run 2, respondiendo SOLO desde los
chunks recuperados, en español. Scorer idéntico a `memo/evaluate.py`
(misma exactitud numérica, misma definición de alucinación, misma regla
Q3 con 1,596).

```bash
# desde la raíz del workspace del goal
memo/.venv/bin/python rag_baseline/retrieve.py   # embeddings + top-5 + diagnóstico
memo/.venv/bin/python rag_baseline/answer.py     # reader (lento en CPU, ver abajo)
memo/.venv/bin/python rag_baseline/score.py      # results.md
```

## Archivos

- `retrieve.py` — embeddings (MiniLM), top-5 por pregunta, diagnóstico de
  cobertura de cifras esperadas (con la tolerancia de redondeo del scorer;
  ver Caveats en `results.md` sobre sus falsos positivos).
- `answer.py` — reader Qwen2.5-1.5B con el contexto recuperado.
- `score.py` — scorer (reutiliza `digits`/`numbers_in`/`expected_hit` de
  `memo/evaluate.py`) + clasificación de fallos verificada a mano contra
  el corpus + `results.md`.
- `retrieval.json` — top-5 por pregunta (página, score, texto completo) y
  cobertura con tolerancia. **No está en el repo por tamaño** (198 KB, el
  CLI de GitHub no acepta argumentos de más de ~128 KB); se regenera con
  `make rag-retrieve`.
- `run_trace.jsonl` — respuestas del reader + contexto condensado usado
  (campo `context`) + páginas recuperadas. Mismo formato que
  `memo/run_trace*.jsonl`. **No está en el repo por tamaño** (123 KB, el
  payload JSON escapa por encima de ~128 KB); se regenera con
  `make rag-answer`.
- `results.md` — tabla por pregunta, por nivel, modos de fallo, comparación
  de los tres sistemas, veredicto y caveats.
- `answer.log` — salida del reader (24 min).

## Versiones exactas

- Python 3.12.3, `memo/.venv` (mismo venv del entrenamiento MEMO; no se
  creó un venv nuevo porque ya tenía torch CPU).
- torch 2.15.0+cpu, transformers 5.24.1, accelerate 1.16.0, peft 0.22.0,
  numpy 2.5.5, sentence-transformers 6.3.0 (instalado en `memo/.venv` para
  este baseline; el resto ya estaba).
- Embedder: `sentence-transformers/all-MiniLM-L6-v2` (descargado de
  Hugging Face, ~90 MB, cache de HF).
- Reader: `memo/models/Qwen2.5-1.5B-Instruct/` (local, ya descargado para
  MEMO run 2). Sin GPU: todo en CPU.

## Tiempos medidos (esta VM, CPU)

- Retrieval (151 chunks, MiniLM): ~66 s (primera vez incluye descarga del
  embedder; re-corridas ~60 s).
- Reader (12 preguntas): **1,438 s (~24 min)**, ~2 min por pregunta.
- Scoring: <1 s.

## Limitaciones conocidas (medidas, no supuestas)

1. **Condensado de contexto.** El prompt con 5 páginas completas
   (~17k caracteres, ~4.5k tokens) tarda >6 min por pregunta en CPU: la
   primera corrida se mató a los ~7 min sin terminar ni Q1. `answer.py`
   condensa cada chunk a ~1,800 caracteres en 2 ventanas alrededor de las
   líneas más relevantes a la pregunta (función `condense`, con el texto
   usado guardado en la traza). Costo medido: en Q9, 279,793 y 781,585
   estaban en el top-5 y no llegaron al lector. Es un costo del workaround
   CPU, no del método RAG en sí; con GPU o chunks más pequeños no haría
   falta.
2. **Cross-lingual.** Preguntas en español, corpus en inglés, embedder
   entrenado en inglés. Q12 exige mapear "Salud" → "Health".
3. **Chunks a nivel página.** Páginas de estados financieros de hasta
   13k caracteres; MiniLM trunca a 256 word pieces al embeder, así que la
   representación de las páginas largas es parcial. Un chunking por
   secciones/tablas probablemente mejoraría el retrieval (Q3, Q4, Q10,
   Q12 son fallos de retrieval con la cifra en una sola página).
4. **El lector divaga.** Qwen2.5-1.5B razona en voz alta en vez de
   responder directo; Q5 se cortó a los 100 tokens sin concluir. Un
   lector más grande o un prompt más estricto ayudaría; aquí se mantuvo
   el 1.5B para que la comparación con MEMO run 2 sea limpia (mismo
   modelo como lector/ejecutivo).
5. **El scorer exige cifras que la pregunta no pide** (Q2: +24.9% no
   preguntado) y exige cifras que no están en el corpus (Q2/Q7/Q9).
   Misma severidad que en MEMO, así que los puntajes son comparables,
   pero el techo de RAG sobre este set es 9/12, no 12/12.

## Hybrid retrieval (BM25 + denso, RRF k=60) — añadido 2026-11-25

El baseline denso puro quedaba "pelón": 4 fallos de retrieval (Q3, Q4,
Q10, Q12) donde la cifra existe pero el top-5 no la trajo. La variante
híbrida añade BM25 para esos tokens exactos.

**Resultado: RAG híbrido 2/12** (Q1, Q11), mismo puntaje que el denso
(Q6, Q11). Por nivel: 1: 1/4, 2: 0/4, 3: 1/4. Alucinaciones 1/12 (Q7).
Cobertura literal ALL en top-5: 1/12 (denso: 2/12). **No cerró la
brecha de retrieval**: Q3, Q4 y Q10 siguen sin traer la cifra; Q12
tampoco. El único flip a favor fue Q1 (el híbrido trajo el 11.2% y el
lector eligió 781,585); el flip en contra fue Q6 (el contexto híbrido
hizo que el lector tomara 128,736 —utilidad bruta de Coca-Cola FEMSA—
como su ingreso). Detalle completo en `results_hybrid.md`, con cada
flip y cada modo de fallo verificado a mano.

Método (`retrieve_hybrid.py`): BM25 Okapi (rank-bm25 0.2.2, k1=1.5,
b=0.75) + mismo MiniLM denso, fusión Reciprocal Rank Fusion k=60
estándar, ranks 1-based sobre los 151 chunks. Se eligió RRF (no suma
ponderada) para no calibrar escalas dispares ni un peso arbitrario.
Tokenización BM25: minúsculas, regex `[a-z0-9áéíóúüñ]+`, misma para
pregunta y documentos, sin stopwords. El lector es el mismo
(`answer.py`, sin cambios, solo `--retrieval` distinto) y el scorer el
mismo (`score_hybrid.py`, mismos helpers de `memo/evaluate.py`): denso
vs híbrido difieren SOLO en retrieval.

Por qué no funcionó como se esperaba (verificado en los rankings): las
preguntas están en español y el corpus en inglés, así que BM25 solo
empareja tokens exactos (números, nombres propios) y los tokens
discriminativos viven en la respuesta, no en la pregunta ('571' no está
en la pregunta de Q4; 'Forward' no está en la de Q10). BM25 reordena
pero no puede inventar el vocabulario que falta.

```bash
memo/.venv/bin/python rag_baseline/retrieve_hybrid.py  # BM25 + MiniLM + RRF -> retrieval_hybrid.json
memo/.venv/bin/python rag_baseline/answer.py --retrieval rag_baseline/retrieval_hybrid.json --out rag_baseline/run_trace_hybrid.jsonl
memo/.venv/bin/python rag_baseline/score_hybrid.py      # -> results_hybrid.md
```

Tiempos medidos (esta VM, CPU): retrieval híbrido 37.0 s; reader
1,236.2 s (~20.6 min); scoring <1 s.

Archivos nuevos: `retrieve_hybrid.py`, `retrieval_hybrid.json` (con
`bm25_rank`/`dense_rank` por chunk y top-10 de cada ranker),
`run_trace_hybrid.jsonl`, `answer_hybrid.log`, `score_hybrid.py`,
`results_hybrid.md`. `retrieval_hybrid.json` (195 KB) y
`run_trace_hybrid.jsonl` (124 KB) no están en el repo por tamaño (el
CLI de GitHub no acepta argumentos de más de ~128 KB); se regeneran
con `make rag-retrieve-hybrid` y `make rag-answer-hybrid`. El baseline
denso (`results.md`, `retrieval.json`, `run_trace.jsonl`) no se tocó.

## Query expansion (traducción + keywords de dominio) — añadido 2026-11-26

El híbrido no cerró la brecha de retrieval porque el problema era el
vocabulario de la pregunta, no el ranker: las preguntas están en español
y el corpus en inglés, y los tokens discriminativos viven en la respuesta.
La variante con queries expandidas traduce cada pregunta al inglés y
añade keywords de dominio de los términos que SÍ están en la pregunta
(`stations`, `stores`, `Health`, `strategy`, `digital ecosystem`).

**Resultado: RAG híbrido + expansión 5/12** (Q1, Q4, Q6, Q8, Q10), el
primer salto real del RAG. Por nivel: 1: 2/4, 2: 2/4, 3: 1/4.
Alucinaciones según el scorer: 3/12 (Q5 '113', Q7 '315'/'328', Q11
'328') — pero 315/328 son cifras reales del informe (p.34), así que el
conteo real de alucinaciones es 1/12 (ver Caveats de
`results_expanded.md`). Cobertura literal ALL en top-5: 4/12 (híbrido:
1/12; denso: 2/12). Flips vs híbrido: a favor Q4 (571, p.21 en top-5),
Q6 (307,197 + 279,793 exactos), Q8 (los 4 países en contexto) y Q10
('FEMSA Forward', p.8 en top-5); en contra Q11 solo en el conteo del
scorer (sustancia correcta, flag artefacto del '328'). Detalle completo
en `results_expanded.md`.

Método (`expand_queries.py`): determinista y auditable, sin modelo.
Cada pregunta tiene una traducción al inglés escrita a mano y keywords
de un mapa explícito `TERM_MAP` (término en español → equivalentes en
inglés), solo de términos presentes en la pregunta original. La expanded
query es traducción + keywords y la ven LOS DOS rankers (BM25 y denso);
todo lo demás queda fijo (mismo reader, mismo prompt, mismo scorer, la
pregunta original en español es la que ve el reader). La salida es
`expanded_queries.json` con la query exacta por pregunta.

Regla anti-filtración (verificada en código por `expand_queries.py`,
que aborta si se viola): nunca se añaden las cifras esperadas (571,
23,206, 14.2, 5.8, …) ni los nombres que SON la respuesta esperada —
en concreto, Q10 no lleva "FEMSA Forward" ni "Spin", Q11 no lleva
"Spin" y Q8 no lleva los nombres de los países. "FEMSA Forward" entró
al top-5 por 'strategy/Retail/Beverages/digital ecosystem', no por
estar en la query.

```bash
python3 rag_baseline/expand_queries.py                    # -> expanded_queries.json (con guardia anti-filtración)
memo/.venv/bin/python rag_baseline/retrieve_expanded.py   # mismo híbrido RRF k=60, con las expanded queries -> retrieval_expanded.json
memo/.venv/bin/python rag_baseline/answer.py --retrieval rag_baseline/retrieval_expanded.json --out rag_baseline/run_trace_expanded.jsonl
memo/.venv/bin/python rag_baseline/score_expanded.py      # -> results_expanded.md
```

Tiempos medidos (esta VM, CPU): expansión <1 s; retrieval expandido
33.8 s; reader 1,548.3 s (~25.8 min); scoring <1 s.

Dos honestidades documentadas en `results_expanded.md`: (1) Q8 se
cuenta correcta por la regla de keywords aunque enumera 18 países del
grupo FEMSA, no los 4 de Proximidad Américas. (2) Q7 es correcta según
el informe mismo (p.34: Spin by OXXO +32.8%, Spin Premia +31.5%) y
solo incorrecta según la respuesta esperada, que viene del comunicado
4Q24; con una regla de ground-truth como la de Q3 (1,596), el run
sería 6/12. El siguiente cuello de botella ya no es el retrieval: con
el dato en el top-5, el lector falla en Q3 y Q12 y divaga en Q5.

Archivos nuevos: `expand_queries.py`, `expanded_queries.json`,
`retrieve_expanded.py`, `retrieval_expanded.json`,
`run_trace_expanded.jsonl`, `answer_expanded.log`,
`score_expanded.py`, `results_expanded.md`.
`retrieval_expanded.json` (208 KB) y `run_trace_expanded.jsonl`
(127 KB) no están en el repo por tamaño (el CLI de GitHub no acepta
argumentos de más de ~128 KB); se regeneran con
`make rag-retrieve-expanded` y `make rag-answer-expanded`. Ni el
baseline denso ni el híbrido se tocaron.

## Para el repo

Los archivos híbridos y de expansión están en `rag_baseline/` en
`femsa-rag-graph-memo` (pusheados 2026-11-26), salvo los cuatro
artefactos grandes, que se regeneran: `retrieval_hybrid.json` (con
`make rag-retrieve-hybrid`), `run_trace_hybrid.jsonl` (con
`make rag-answer-hybrid`), `retrieval_expanded.json` (con
`make rag-retrieve-expanded`) y `run_trace_expanded.jsonl` (con
`make rag-answer-expanded`). El README raíz del repo trae la tabla
actualizada (híbrido 2/12, expansión 5/12) y la sección "Cómo
reproducir las métricas" con los comandos de cada variante.
