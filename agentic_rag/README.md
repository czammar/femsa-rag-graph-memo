# RAG agéntico (LangChain + LangGraph) — FEMSA 2024

Agente ReAct con LangGraph que responde las mismas 12 preguntas del
experimento (`memo/eval_questions.json`) con las mismas reglas de scoring
(`memo/evaluate.py`). Es el sistema "agéntico" del post RAG clásico vs
agentic RAG: un bucle plan → act → observe → answer con herramientas,
no un pipeline fijo.

## Diseño

- **LangGraph** (`StateGraph` de dict): nodos `plan` → `act` → `plan` …
  → `answer`. Máximo 3 pasos de herramienta por pregunta; máximo 5
  llamadas al plan; guardia anti-bucle (si el modelo repite la misma
  llamada exacta, la observación le dice que cambie de estrategia).
- **Modelo**: Qwen2.5-1.5B-Instruct local (`memo/models/…`), bf16, CPU,
  cargado con transformers crudo + `apply_chat_template` (mismo patrón
  probado de `rag_baseline/answer.py`). Con `run_agentic.py --llm
  ollama` el mismo grafo corre contra un modelo servido por Ollama
  (por defecto `qwen2.5:7b-instruct`): el backend vive en
  `llm_backends.py` y solo cambia el cerebro — ver `OLLAMA.md`.
- **Herramientas** (LangChain `@tool`, reutilizando código verificado):
  1. `vector_search`: retrieval híbrido BM25 Okapi + MiniLM, RRF k=60,
     top-5 páginas sobre `pipeline/data/chunks.jsonl` (misma lógica que
     `rag_baseline/retrieve_hybrid.py` + `condense` de
     `rag_baseline/answer.py`).
  2. `graph_query`: Cypher contra el store Relate `pipeline/data/femsa.rl`
     (vía el CLI `relate`, `~/.local/bin/relate` v1.2.0). El prompt del
     plan incluye el esquema del grafo, el inventario de entidades y
     plantillas de consulta por tipo de pregunta.
  3. `calculator`: aritmética segura por AST (solo `+ - * / % **` y
     paréntesis); las cifras vienen de la evidencia, nunca de la cabeza
     del modelo.
- **Idioma**: el agente responde en español, con las cifras exactas como
  aparecen en la evidencia. Si las herramientas no devuelven evidencia,
  responde "No lo sé" en vez de inventar.

## Hallazgo previo: el 1.5B no emite tool calls nativos

`test_toolcall.py` y `test_toolcall2.py` lo verifican: con
`chat.bind_tools(...)` el modelo emite **cero** `tool_calls` (y en una
prueba repitió el transcript completo del prompt). Por eso el agente usa
un **bucle ReAct textual**: el modelo escribe una línea
`ACCIÓN: herramienta("argumento")` o `RESPUESTA: <texto>` y el nodo `act`
la parsea y ejecuta la `@tool`. Los nodos, el grafo y las herramientas
son LangChain/LangGraph; solo el formato de llamada es texto.

## Historial de prompts (Q4 como pregunta de prueba)

El 1.5B compone Cypher con dificultad; cada variante se probó solo en Q4
("¿Cuántas estaciones tenía OXXO GAS al cierre de 2024?", esperado 571):

| Variante | Cambio | Resultado Q4 |
|----------|--------|--------------|
| v1 | sin esquema | responde sin usar herramientas |
| v2 | esquema añadido | inventa `GasStation`/`HAS_STATION`, luego "No lo sé" |
| v3 | inventario de entidades + mapeo OXXO GAS→Combustibles | consulta con `OPERA_EN` y `s.valor` |
| v4 | reglas de relaciones | 3× `graph_query` con `OPERA_EN`/`COUNT(r)`, luego "No lo sé" |
| v5 (final) | **plantillas de consulta** por tipo + regla "el valor SIEMPRE es `r.valor`" + **vector_search primero** + guardia anti-bucle | ver `results_agentic.md` |

Lección medida: con reglas abstractas el modelo mezcla relaciones
(`OPERA_EN` para una métrica, `REPORTA` para una métrica operativa) y
pone `.valor` en el nodo (`m.valor`, `i.valor`) en vez de en la relación
(`r.valor`). Las plantillas —documentación de la herramienta, no
respuestas— y empezar por `vector_search` son lo que hace usable el
agente con un modelo tan pequeño.

## Setup

```bash
# 1. Servidor Relate (el grafo debe estar sirviendo para graph_query)
relate serve --database ./pipeline/data/femsa.rl --bolt 7687 --auth none
#    (en otra terminal; el agente lo comprueba al arrancar y avisa si no responde)

# 2. Entorno (desde la raíz del repo)
python3 -m venv agentic_rag/.venv
agentic_rag/.venv/bin/pip install langchain==1.5.2 langgraph==1.2.24 \
  langchain-huggingface==1.3.0 langchain-core==1.8.5 neo4j==6.4.1 \
  langsmith==0.16.2 langchain-ollama==1.2.0
# El stack ML pesado (torch, transformers, sentence-transformers,
# rank_bm25) NO se reinstala: se comparte desde memo/.venv con un
# archivo .pth (agentic_rag/.venv/lib/python3.12/site-packages/memo_venv.pth
# apunta a memo/.venv/lib/python3.12/site-packages). Evita otra descarga
# de ~2 GB y otro venv de ~4 GB. Si memo/.venv no existe, crea el venv
# completo instalando además torch==2.15.0+cpu transformers==5.24.1
# sentence-transformers==6.3.0 rank_bm25 numpy==2.5.5 pydantic==2.14.4.
```

## Reproducir

```bash
# Corrida completa (12 preguntas, reanudable: salta las ya registradas)
agentic_rag/.venv/bin/python agentic_rag/run_agentic.py
# Una sola pregunta (para depurar el prompt):
agentic_rag/.venv/bin/python agentic_rag/run_agentic.py --only Q4
# Scoring + informe
agentic_rag/.venv/bin/python agentic_rag/score_agentic.py
# o: make agentic-run / make agentic-score (desde la raíz)
```

## Con Ollama y un modelo más grande

El run de 1.5B (2/12) no muestra el poder de lo agéntico: el modelo
no sabe manejar herramientas. La misma corrida con un 7B/14B servido
por Ollama es la prueba real del bucle — mismo grafo, mismas
herramientas, mismos prompts, mismo scorer; solo cambia el cerebro.
Ver la guía completa en **`OLLAMA.md`**. Resumen:

```bash
make agentic-run-ollama                        # 7B por defecto
make agentic-run-ollama MODEL=qwen2.5:14b-instruct
make agentic-score-ollama                      # escribe results_agentic_ollama.md
```

La traza Ollama (`run_trace_agentic_ollama.jsonl`) y su informe
(`results_agentic_ollama.md`) nunca tocan los del 1.5B. Los modos de
fallo se verifican a mano igual que en la corrida local (ver
`OLLAMA.md` §8).

Resultados: `run_trace_agentic.jsonl` (un JSON por pregunta: pasos,
observaciones, `plan_raw_outputs`, segundos) y
`results_agentic.md` (tabla por pregunta, por nivel, uso de
herramientas, modos de fallo verificados a mano, comparación con los
seis sistemas).

## Tiempos medidos en CPU (VM: 4 CPUs, ~8 GB RAM)

- Carga del modelo Qwen2.5-1.5B: ~7 s.
- Cada llamada al LLM (plan o respuesta): ~55–130 s sin contención de
  CPU; hasta ~220 s cuando otro proceso pesado comparte la VM (la corrida
  completa de Q4 en v5 tardó 1,084.5 s por la contención con otro agente
  corriendo un modelo en paralelo).
- Corrida completa (12 preguntas): **65.3 min de tiempo de preguntas**
  (media 326 s por pregunta; Q1 866 s con contención de CPU, resto
  88–653 s). Wall total ~69 min incluyendo una caída por un bug del
  router en Q5 (ver abajo) y la reanudación desde Q5. Resultado: **2/12**
  (Q4 y Q8), 0/12 con alucinación. Ver `results_agentic.md`.

## Incidencias de la corrida

- **Bug del router (Q5):** `_route` podía devolver `"plan"` pero el
  mapa de `add_conditional_edges` no incluía `"plan"` → `KeyError` y
  la corrida se cayó en Q5. Se arregló añadiendo `"plan": "plan"` al
  mapa y un `try/except` por pregunta en `run_agentic.py` (un fallo en
  una pregunta se registra como error y la corrida sigue). La corrida
  se reanudó desde Q5 sin tocar Q1–Q4.
- **La guardia anti-bucle tiene un agujero de espacios:** en Q2 la
  segunda llamada era idéntica salvo un espacio final y la comparación
  exacta no la detectó. Normalizar con `strip()` lo cerraría; se deja
  como está para no cambiar el agente a mitad de corrida.

## Archivos

| Archivo | Qué es |
|---------|--------|
| `agent.py` | Definición LangGraph + herramientas + prompts |
| `llm_backends.py` | Backends del LLM: local 1.5B (transformers) y Ollama (ChatOllama); `call_llm` es el único punto de contacto del agente con el modelo |
| `OLLAMA.md` | Guía: mismo agente con un modelo 7B/14B en tu máquina con Ollama |
| `run_agentic.py` | Corre las 12 preguntas, escribe el trace (reanudable); flags `--llm local\|ollama --model --host` |
| `score_agentic.py` | Scorer (mismo criterio que `memo/evaluate.py`) + `results_agentic.md`; flags `--model-label --system-name --blank-modes` para runs nuevos |
| `results_agentic.md` | Informe de resultados |
| `test_toolcall.py`, `test_toolcall2.py` | Diagnóstico: tool calling nativo falla (cero tool_calls) |
| `test_cypher.py` | Diagnóstico: el 1.5B compone Cypher con errores de relación |
| `run_trace_agentic.jsonl` | Trace completo — **no se sube al repo** (como ninguna otra traza del repo; regenerable con `make agentic-run`) |
| `run_trace_agentic_ollama.jsonl` | Trace del run Ollama — **tampoco se sube** (regenerable con `make agentic-run-ollama`) |
| `results_agentic_ollama.md` | Informe del run Ollama (se escribe con `make agentic-score-ollama`; los modos de fallo se verifican a mano) |

## Versiones

- langchain 1.5.2, langchain-core 1.8.5, langchain-huggingface 1.3.0,
  langchain-ollama 1.2.0, langgraph 1.2.24, neo4j 6.4.1, langsmith 0.16.2 (instalados en
  `agentic_rag/.venv`).
- Compartidos desde `memo/.venv`: torch 2.15.0+cpu, transformers 5.24.1,
  sentence-transformers 6.3.0, rank_bm25, numpy 2.5.5, pydantic 2.14.4.
- Modelo: Qwen2.5-1.5B-Instruct local, bf16, CPU.
- Relate CLI v1.2.0 (`~/.local/bin/relate`), store `pipeline/data/femsa.rl`.
