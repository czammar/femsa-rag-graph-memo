# RAGAS (estilo) — FEMSA 2024

Métricas RAGAS para las tres variantes RAG (denso, híbrido BM25+denso,
híbrido+expansión), sobre las mismas 12 preguntas y los mismos traces que
`rag_baseline/results*.md`. Complementan el scorer de exactitud: RAGAS separa
la calidad del retrieval (context precision / context recall) de la calidad
de la respuesta (faithfulness / answer relevancy).

## Método: determinista, sin juez LLM (ver justificación abajo)

La librería `ragas` 0.4.4 está instalada en `ragas/.venv` y se probó con el
único juez disponible — Qwen2.5-1.5B-Instruct local en CPU — en una pregunta
(`smoke_test_ragas.py`). El resultado de ese smoke test está en
`results_ragas.md` (sección "Smoke test con la librería real") y es la razón
documentada de usar el camino determinista: un juez de 1.5B en CPU es
demasiado lento y demasiado débil para los ~300 llamados que exigiría un run
completo de ragas (36 respuestas × 4 métricas).

Las cuatro métricas se calculan con las definiciones estándar de RAGAS,
adaptadas a chunks de página completa, a partir de:

- `../retrieval*.json` — top-5 páginas por pregunta y variante.
- `../run_trace*.jsonl` — respuesta final y el contexto condensado que el
  lector realmente vio.
- `ground_truth_evidence.json` — conjunto de evidencia por pregunta, derivado
  por grep literal sobre `pipeline/data/chunks.jsonl` (151 páginas) y
  auditado a mano (ver "Verificación del ground truth" abajo). Cada ítem de
  evidencia es un hecho que la respuesta debe contener, con las páginas donde
  aparece.

Definiciones:

- **context_precision** = páginas recuperadas que son páginas de evidencia /
  páginas recuperadas (top-5). Una página es relevante si está en el conjunto
  de páginas de evidencia de esa pregunta. Definición estricta: una página
  sobre el tema que no contiene la cifra exacta cuenta como no relevante, así
  que este número es un **límite inferior** de lo que un juez LLM marcaría.
- **context_recall** = ítems de evidencia con al menos una página en el top-5
  / ítems de evidencia. Q9 no tiene ningún ítem en el corpus (39.3/35.8/75.1%
  son participaciones calculadas, no cifras impresas) → recall **null, no 0**.
- **faithfulness** = claims de la respuesta apoyados por el contexto
  recuperado / claims totales. Claims = números + entidades (divisiones,
  iniciativas, países). Un número está apoyado si su valor normalizado aparece
  en el contexto condensado (encabezados "[Página N]" eliminados antes, para
  que un número de página filtrado no cuente como apoyado por el encabezado).
  Una abstención ("No lo sé") tiene 0 claims → faithfulness 1.0 por
  convención, marcada como abstención. **Límite: no detecta la mala
  atribución** (un número real usado para la división equivocada, p. ej.
  híbrido Q6: 128,736 es utilidad bruta, no ingreso). La faithfulness
  determinista es por tanto un **límite superior**.
- **answer_relevancy (proxy)** = similitud coseno entre embeddings
  all-MiniLM-L6-v2 de la pregunta y la respuesta. RAGAS la define con
  preguntas generadas por LLM a partir de la respuesta; ese paso NO se usa
  aquí. El proxy es el más débil de los cuatro en este set: premia respuestas
  largas y fluidas (aunque estén mal) y castiga respuestas cortas y
  numéricas (aunque sean correctas) — ver `results_ragas.md`.

## Reproducir

```bash
cd rag_baseline/ragas
python3 compute_metrics.py            # precision/recall/faithfulness (~2 s)
../../memo/.venv/bin/python compute_relevancy.py   # relevancy proxy (~36 s, carga MiniLM)
```

Salidas: `metrics_ragas.json` (por pregunta, por variante, por métrica, con
detalle por claim de faithfulness y veredicto de exactitud para el análisis
de acuerdo/desacuerdo).

Smoke test con la librería real:

```bash
ragas/.venv/bin/python smoke_test_ragas.py   # 1 pregunta (denso Q1), 4 métricas, juez local 1.5B
```

## Resultados (resumen)

| Variante | context_precision | context_recall (media) | context_recall (pooled) | faithfulness | answer_relevancy (proxy) | Exactitud |
|----------|------------------:|-----------------------:|------------------------:|-------------:|-------------------------:|----------:|
| RAG vectorial denso | 0.183 | 0.455 | 8/17 = 0.471 | 0.969 | 0.609 | 2/12 |
| RAG híbrido BM25+denso | 0.150 | 0.500 | 9/17 = 0.529 | 0.967 | 0.628 | 2/12 |
| RAG híbrido + expansión | 0.400 | 1.000 | 17/17 = 1.000 | 0.975 | 0.656 | 5/12 |

Lectura corta: la expansión arregló el retrieval por completo (recall 0.47 →
1.00); la faithfulness es alta en las tres variantes (0.97 — el lector casi
no inventa números, los **elige mal** del contexto; el único invento claro
es la suma 43,265 del denso Q9); la exactitud no sigue a ninguna de las dos,
porque el fallo que queda es de selección del lector, no de retrieval ni de
invención. Detalle completo en `results_ragas.md`.

## Verificación del ground truth

Comandos usados (grep literal sobre `pipeline/data/chunks.jsonl`):

- `23,206` → p.21 única; `1,596` → pp. 23, 112, 114; `781,585` → pp. 11, 108,
  109, 112; `11.2%` → pp. 5, 108, 112; `571` → pp. 21, 114, 126;
  `307,197` → pp. 112, 114; `279,793` → pp. 112, 113; `14.2%` → pp. 5, 25,
  27, 29, 112, 113 (solo pp. 5 y 112 traen AMBAS divisiones en 14.2% — el
  empate exige las dos juntas; pp. 25/27/29/113 traen solo una y se anotan
  como evidencia parcial, no contada); `FEMSA Forward` → pp. 3, 5, 8, 12,
  13, 14, 42, 101, 115, 116, 120, 149, de las cuales solo 3, 5, 8, 12, 14,
  42, 115, 116 la nombran como estrategia (p.13 es solo el encabezado del
  diagrama "FEMSA FORWARD & HEALTH"; p.101 es una biografía; pp. 120/149
  son menciones incidentales) — **la corrección de p.13 cambió el recall de
  Q10 de 1.00 a 0.00 en denso e híbrido, en línea con el veredicto de
  `results.md`/`results_hybrid.md`**; `Digital@FEMSA` → pp. 5, 34, 100
  (p.100 es biografía; evidencia: pp. 5 y 34); `8.6 million` y `24.6 million`
  → pp. 13, 33, 35; `32.8%` → p.34; `31.5%` → pp. 34, 113; `5.8%` → pp. 5,
  108, 109, 112, 113, 114 (solo 5, 112, 114 son crecimiento de Salud; el
  resto es margen de utilidad neta); `27.5%`, `24.9%`, `39.3%`, `75.1%` →
  **ninguna página** (fuera del corpus, verificado).
- Q8: las 4 países juntos (México, Colombia, Chile, Perú) en pp. 10, 21, 22,
  55, 92, 121.

## Entorno

- `ragas/.venv`: Python 3.12.3, `ragas` 0.4.4, `langchain-huggingface`,
  `torch` (CPU), `transformers`, `sentence-transformers`. Instalación de
  `ragas` + deps: ~8 min; `torch` CPU + `transformers`: ~10 min.
- `memo/.venv`: torch 2.15.0+cpu, transformers 5.24.1,
  sentence-transformers 6.3.0 (usado solo para el proxy de relevancy).
- Juez local (solo smoke test): `memo/models/Qwen2.5-1.5B-Instruct/`
  (bf16, CPU). Embeddings: all-MiniLM-L6-v2.

## Archivos

- `ground_truth_evidence.json` — evidencia por pregunta (auditable).
- `compute_metrics.py` — precision/recall/faithfulness deterministas.
- `compute_relevancy.py` — proxy de answer relevancy (MiniLM coseno).
- `smoke_test_ragas.py` — 1 pregunta con la librería `ragas` real y juez local.
- `metrics_ragas.json` — resultados por pregunta/variante/métrica.
- `results_ragas.md` — lectura completa, acuerdo/desacuerdo con exactitud.
