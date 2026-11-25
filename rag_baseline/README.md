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
| RAG vectorial (este baseline) | 2/12 | 0/4 | 1/4 | 1/4 |
| MEMO run 2 | 2/12 | 1/4 | 1/4 | 0/4 |
| MEMO run 1 | 0/12 | 0/4 | 0/4 | 0/4 |

El veredicto completo y la clasificación de fallos verificada cifra por
cifra están en `results.md`. Resumen: retrieval falla en 4 (Q3, Q4, Q10,
Q12), 3 preguntas tienen cifras esperadas que no existen en el corpus
(Q2, Q7, Q9: +24.9%/+27.5% son del comunicado 4Q24; 39.3/35.8/75.1% son
participaciones calculadas), el lector falla en 3 (Q1, Q5, Q8) y Q9 suma
cifras equivocadas (43,265). Techo real de RAG sobre este set tal como
está especificado: 9/12.

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

## Para el repo

Estos archivos están en `rag_baseline/` en `femsa-rag-graph-memo`. El README
raíz del repo dice **RAG vectorial: 2/12** con enlace a `rag_baseline/results.md`.
`retrieval.json` y `run_trace.jsonl` no están en el repo por tamaño (ver
"Archivos" arriba); se regeneran con `make rag-retrieve` y `make rag-answer`.
