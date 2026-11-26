# Notebooks de Google Colab — Experimento FEMSA 2024

Repite el experimento completo (**RAG clásico vs grafo vs MEMO vs RAG agéntico**,
informe anual FEMSA 2024) en **Google Colab con GPU T4**. Todo corre en Colab:
nada de pasos de máquina local. Cada notebook es autocontenido: monta Drive,
clona el repo, instala sus dependencias y, donde hace falta, instala y arranca
Ollama dentro de la propia instancia. **Los datos procesados no se regeneran
en cada notebook: cada uno los busca primero en Drive y, si no están, te pide
subirlos** (el widget de Colab); generarlos es el último recurso. Y **cada
notebook guarda sus modelos y outputs en Drive al final**, con una celda de
verificación que lista lo guardado: nada se pierde si la sesión se corta.

## Autocontención total: toda la lógica vive inline en las celdas

Cada notebook es **plenamente autocontenido**: ninguna celda de código
importa ni ejecuta scripts `.py` del repo. Toda la lógica —extracción del
PDF, generación de Cypher y carga a Relate, generación y aumento del dataset
MEMO, entrenamiento LoRA, ejecutivo y scorer, retrieval denso e híbrido,
expansión de queries, juez agéntico, cálculo determinista de RAGAS— vive
**inline en las celdas del propio notebook**, con los mismos prompts, las
mismas constantes y la misma lógica de scorer que las corridas en CPU.

Los scripts del repo (`pipeline/`, `memo/`, `rag_baseline/`,
`agentic_rag/`) siguen siendo la **referencia canónica** de las corridas en
CPU y se citan en el markdown de cada notebook como procedencia, pero los
notebooks **no dependen de ellos**: si borras todos los `.py` del repo
clonado, los notebooks siguen corriendo (solo necesitan los archivos de
**datos**, que son otra cosa — ver la tabla de abajo).

| Dato | De dónde lo saca cada notebook |
|---|---|
| `pipeline/data/chunks.jsonl` (151 págs.) | Drive → subir con `files.upload()` → generar con la extracción inline del NB01 (último recurso) |
| `pipeline/triples_seed.json` (31 tripletas) | Del repo clonado (dato JSON, no código); lo leen el NB02 (Cypher) y el NB04 (eval ampliado) |
| `pipeline/queries.cypher` (12 preguntas) | Del repo clonado (dato); lo lee el NB02 |
| `memo/eval_questions.json` (12 preguntas) | Del repo clonado (dato); lo leen los NB03–08 |
| `rag_baseline/ragas/ground_truth_evidence.json` | Del repo clonado (dato); lo leen el NB07 (diagnóstico del juez) y el NB08 (RAGAS) |
| `memo/reflection_qa.jsonl` (v1, 446) y `reflection_qa_v2.jsonl` (v2, 1,414) | Los **genera inline** el NB03 desde los chunks (misma lógica que los scripts; v1 verificada byte a byte, md5 `a4ccf20c…`); también se pueden subir a Drive y el NB03 los reutiliza si existen |
| `memo/reflection_qa_v3.jsonl` (~3,300) | Lo genera inline el NB03 (segunda pasada de aumento, sin tope) |
| Adaptador LoRA, trazas, retrievals, resultados | Los produce cada etapa y se guardan en Drive; los notebooks que los consumen los verifican con la celda de prerrequisitos (NB04, NB08) |

`validate_notebooks.py` lo verifica: ninguna celda de código contiene
nombres de scripts del repo ni `sys.path.insert` (el chequeo 7 del validador).

## Orden de ejecución

| # | Notebook | Qué hace | Ollama |
|---|----------|----------|--------|
| 00 | `00_ollama_setup.ipynb` | Setup canónico de Ollama: instala, arranca el servidor, baja qwen2.5:1.5b y qwen2.5:7b | sí (completo) |
| 01 | `01_setup_y_extraccion.ipynb` | Drive, repo, dependencias, descarga del PDF y extracción (151 págs.) | solo verifica |
| 02 | `02_grafo_relate.ipynb` | Cypher de carga, instala Relate, carga el grafo (27 nodos, 39 relaciones), 12 preguntas | no |
| 03 | `03_memo_entrenamiento.ipynb` | Dataset expandido v3 (~3,300 pares, generado inline) + entreno MEMORY 1.5B con LoRA en GPU | no (entrena con HF) |
| 04 | `04_memo_evaluacion.ipynb` | Protocolo MEMO (ejecutivo vía Ollama) + control + scorer; genera set de eval ampliado (opcional) | sí, ejecutivo |
| 05 | `05_rag_baseline.ipynb` | Retrieval denso + lector vía Ollama + scorer | sí, lector |
| 06 | `06_rag_hibrido_y_expansion.ipynb` | BM25 híbrido (RRF k=60) + expansión de queries + lector vía Ollama + scorers | sí, lector |
| 07 | `07_rag_agentico.ipynb` | RAG agéntico JUEZ: retrieval híbrido top-10 + un llamado al juez por pregunta (califica los 10 chunks de golpe) + lector solo con los seleccionados; máx. 3 reintentos, abstención si no hay veredictos | sí, juez y lector |
| 08 | `08_ragas.ipynb` | RAGAS deterministas + run real con la librería `ragas` y juez vía Ollama | sí, juez |

## Datos procesados: primero en Drive, si no, súbelos

Cada notebook que consume `pipeline/data/chunks.jsonl` (las 151 páginas del
informe extraídas) lleva las mismas celdas de datos, en este orden:

1. **Buscar en Drive** (ruta estándar, bajo el repo clonado en
   `/content/drive/MyDrive/femsa-rag/femsa-rag-graph-memo`).
2. **Si no está, subirlo** con el widget de Colab (`files.upload()`): el
   archivo es `chunks.jsonl`. Lo consigues de una corrida anterior en Drive
   o generándolo con el notebook 01 (su celda de último recurso extrae las
   151 páginas del PDF). La celda te dice qué subir y dónde conseguirlo.
3. **Solo si no lo subes, generarlo** con la lógica de extracción inline
   de la propia celda (descarga el PDF ~19 MB y extrae las 151 páginas,
   ~1–2 min). Es el último recurso, no el camino principal.

`triples_seed.json` y `eval_questions.json` vienen con el clon del repo: no
hace falta subirlos.

Los insumos que produce una etapa anterior **no** se regeneran ni se suben
por defecto: se verifican con una celda de prerrequisitos que falla con un
mensaje claro si faltan, diciendo qué notebook los produce:

- **NB04** verifica que exista el adaptador MEMORY del NB03
  (`memo/memory_model_colab`): sin adaptador entrenado, no hay nada que evaluar.
- **NB08** verifica las 3 trazas y los 3 retrievals de los NB05/06
  (`run_trace_colab.jsonl`, `run_trace_hybrid_colab.jsonl`,
  `run_trace_expanded_colab.jsonl`, `retrieval.json`,
  `retrieval_hybrid.json`, `retrieval_expanded.json`): si faltan, te pide
  subirlos; si existen los de Colab, los **alinea** copiándolos a los nombres
  de CPU (`run_trace.jsonl`, ...) que espera el cálculo determinista inline.
  Sin trazas no hay métricas.

## Qué guarda cada notebook en Drive

Cada notebook termina con una celda de guardado + verificación: lista lo
producido (ruta, tamaño, fecha) y avisa si falta algo obligatorio. Todo vive
bajo el repo clonado en Drive, así que sobrevive aunque la sesión se corte.
Los logs de la sesión (`ollama.log`, `relate.log`) también se copian al repo.

| # | Lo que guarda en Drive |
|---|---|
| 00 | `ollama-manifest.txt` (versión de Ollama + tags bajados). Los modelos **no** se copian a Drive (4.7–9 GB cada uno; bajar de nuevo por sesión es más rápido que servirlos desde Drive) |
| 01 | `pipeline/data/chunks.jsonl` (151 páginas), `pipeline/data/FEMSA_2024_Annual_Report_EN.pdf`, `pipeline/extracted_triples.jsonl` (13 candidatas por reglas, pendientes de revisión) |
| 02 | `pipeline/data/femsa.rl` (store Relate: 27 nodos, 39 relaciones), `pipeline/load_graph.cypher`, `pipeline/schema.cypher`, `relate.log` |
| 03 | `memo/memory_model_colab` (adaptador LoRA final, MEMORY 1.5B), `memo/memory_model_colab_ckpt` (checkpoints cada 25 pasos), `memo/reflection_qa_v3.jsonl` (dataset v3) + v2 y v1 |
| 04 | `memo/run_trace_colab.jsonl`, `memo/run_trace_colab_control.jsonl`, `memo/results_colab.md`, `memo/results_colab_control.md`, `memo/eval_questions_expanded.json` (si se generó; `needs_review: true`) |
| 05 | `rag_baseline/retrieval.json` (denso top-5), `rag_baseline/run_trace_colab.jsonl`, `rag_baseline/results_colab.md` |
| 06 | `rag_baseline/retrieval_hybrid.json`, `retrieval_expanded.json`, `run_trace_hybrid_colab.jsonl`, `run_trace_expanded_colab.jsonl`, `results_hybrid_colab.md`, `results_expanded_colab.md`, `expanded_queries.json` |
| 07 | `rag_baseline/retrieval_hybrid_k10.json` (top-10), `agentic_rag/run_trace_judge_colab.jsonl`, `agentic_rag/results_agentic_judge_colab.md` |
| 08 | `rag_baseline/ragas/metrics_ragas.json` (deterministas, actualizadas en Colab), `rag_baseline/ragas/metrics_ragas_llm_colab.json` (run real con juez Ollama, solo si completa), `ground_truth_evidence.json` |

## Antes de empezar

1. **Activa la GPU:** Runtime > Change runtime type > **T4 GPU**. Verifica con
   `!nvidia-smi` (el notebook 00 lo hace).
2. **Drive:** todo vive en `/content/drive/MyDrive/femsa-rag` (cámbiala en la
   celda de parámetros de cada notebook). Ahí quedan: el repo clonado, los
   chunks, las trazas, los resultados y los adaptadores LoRA.
3. **Corre los notebooks en orden** (o solo el que necesites: cada uno es
   autocontenido, pero los prerrequisitos de etapa están escritos arriba de
   cada uno y se verifican con una celda que falla con un mensaje claro si
   faltan: NB04 necesita el adaptador del NB03, NB08 necesita las trazas y
   retrievals de los NB05/06).

## Persistencia: qué sobrevive a un reinicio de sesión y qué no

La VM de Colab es efímera: al desconectarse se pierde todo lo instalado.

| Persiste (en Drive) | Se pierde (reinstalar por sesión) |
|---|---|
| Repo clonado, chunks, tripletas, trazas, resultados | Ollama (binario + servidor + modelos bajados) |
| Adaptadores LoRA y checkpoints (`memo/memory_model_colab*`) | Binario `relate` y su servidor (el store `femsa.rl` sí persiste) |
| Cache de Hugging Face (`HF_HOME` en Drive) | pip installs de la sesión |
| `eval_questions_expanded.json` (si se genera) | — |

Por eso cada notebook que usa Ollama o Relate lleva sus celdas de setup
**idempotentes**: verifican si el binario existe, si el servidor ya responde
y si el modelo ya está bajado antes de hacer nada. Correr un notebook dos
veces (o después del 00) no rompe nada ni descarga de más.

## Diseño Ollama: por qué los modelos intermedios van por Ollama

Los modelos intermedios —lector RAG, ejecutivo MEMO, juez RAGAS, juez y lector
del RAG agéntico— corren vía **Ollama instalado en la propia instancia de Colab**:

- **Un solo servidor** para todos los notebooks (puerto 11434, localhost).
- **Cambio de modelo con un parámetro** en la parte superior de cada notebook
  (`qwen2.5:1.5b` para paridad con los runs originales en CPU, `qwen2.5:7b`
  para el run con modelo más grande; el juez del notebook 07 usa 7B por defecto).
- **Sin cargar pesos en cada notebook**: el modelo se baja una vez por sesión
  y todos los notebooks lo reutilizan.

El **entrenamiento MEMO** (notebook 03) sigue siendo Hugging Face + LoRA:
entrenar no se puede hacer vía Ollama. Y el **MEMORY** (notebook 04) se carga
con HF en GPU porque es un LoRA entrenado — Ollama no sirve adaptadores
LoRA; el EXECUTIVE sí va por Ollama.

**Modelos y VRAM (T4, 16 GB):** `qwen2.5:1.5b` ~1 GB, `qwen2.5:7b` (Q4_K_M)
~4.7 GB, `qwen2.5:14b` Q4 ~9 GB (ajustado; opcional, comentado en el 00).
Los modelos viven en la VM y **hay que volver a bajarlos en cada sesión
nueva**, salvo que se apunte `OLLAMA_MODELS` a una ruta en Drive (opcional,
más lento: se menciona en el 00).

**Fuente de los comandos de Ollama en Colab:** guías publicadas de Ollama en
Colab (apt `zstd` —que Colab no trae y el instalador exige— + `install.sh` +
`ollama serve` en background con `nohup` + pull de modelos) y verificación
propia de que el instalador exige `zstd`. El notebook 00 es el lugar canónico
con la explicación completa; los demás llevan la versión compacta idempotente.

## Tiempos (estimaciones, no mediciones)

En CPU local los tiempos medidos fueron: entrenamiento MEMO run 2 ~4.1 h,
lector RAG ~24 min, agente 65 min, RAGAS real inviable (timeout a 1800 s).
En la T4 los tiempos **estimados** son mucho menores (medir con las celdas
de `TIME_TEST` / piloto antes de la corrida completa):

| Paso | Estimación en T4 |
|---|---|
| pip installs (1ª vez) | ~5–10 min |
| Extracción PDF (NB01) | ~1–2 min |
| Grafo Relate (NB02) | ~2 min |
| Entreno MEMO 1.5B, 3 épocas (NB03) | ~10–20 min (medir con `TIME_TEST_STEPS`) |
| Eval MEMO 12 preguntas (NB04) | ~5 min |
| RAG denso / híbrido / expandido (NB05–06) | ~5–10 min cada uno |
| Juez + lector 7B, 12 preguntas (NB07) | ~10–20 min (12 llamados al juez + hasta 12 al lector) |
| RAGAS real, 36 muestras (NB08) | acotado por `TIME_BUDGET_MIN` (default 90 min) |

Colab gratuito desconecta tras ~90 min de inactividad y 12 h como máximo:
mantén la pestaña abierta en las corridas largas. Los checkpoints (NB03) y
las trazas reanudables (NB07) permiten seguir donde se cortó.

## El RAG agéntico es el juez (NB07), no el agente ReAct

El notebook 07 implementa el **segundo diseño** de RAG agéntico, el que pidió
el usuario: el sistema solo califica los chunks del RAG híbrido y selecciona
los útiles con un juez. Sin `graph_query`, sin `calculator`. El primer diseño
—el agente ReAct con LangChain + LangGraph y 3 herramientas— sacó 2/12 en CPU
y nunca usó `vector_search` ni `calculator` (24/24 llamadas fueron
`graph_query`): midió un agente solo-grafo, no un RAG agéntico, y queda como
negativo documentado (`agentic_rag/results_agentic.md`, no se toca). El código
de referencia del juez vive en el repo (`agentic_rag/judge_rerank.py`,
`agentic_rag/run_judge.py`, `agentic_rag/score_judge.py`): es la misma
lógica que el notebook 07 lleva inline, para las corridas en CPU; el
notebook no importa ni ejecuta esos scripts. El notebook 07 no corre
nada en este sandbox: la evaluación del juez corre **solo en Colab**.

## Las dos mejoras que permite Colab

1. **Más preguntas:** el notebook 04 genera un set de evaluación ampliado
   desde las tripletas semilla (plantillas por tipo de tripleta). Cada cifra
   esperada se verifica por grep literal en el corpus: lo que no aparece
   literal, se descarta — nunca se adivina. La salida lleva
   `"needs_review": true`: hay que revisar las preguntas a mano antes de
   evaluar con ellas.
2. **Mejor entrenamiento MEMO:** notebook 03 entrena MEMORY con el 1.5B
   (en CPU era el 0.5B) sobre un dataset expandido v3 (~3,300 pares, más
   variación de paráfrasis por hecho), 3 épocas, LoRA r=32, en GPU.

## Mismo scorer, mismas reglas, en todas partes

- Mismas 12 preguntas (`memo/eval_questions.json`), misma lógica de scorer
  (inline en cada notebook, tomada de `memo/evaluate.py` y sus derivados
  en `rag_baseline/` y `agentic_rag/`).
- Techo real **9/12** (3 cifras esperadas no están en el informe), regla Q3
  con **1,596** aperturas netas, conflicto de ground-truth Q7 documentado.
- Las cifras RAGAS deterministas siguen siendo las citables si el run real
  con juez Ollama no completa; el notebook 08 compara ambos si completa.

## Regenerar los notebooks

`build_notebooks.py` genera los 9 `.ipynb` (nbformat). `validate_notebooks.py`
comprueba: JSON válido, `py_compile` de las celdas Python (saltando celdas
`%pip` / `!` / `%%bash`), existencia de los archivos del repo referenciados,
presencia del bloque Ollama en 00/04/05/06/07/08, base autocontenida en todos
(GPU check + Drive + repo), celda de datos (Drive → subir → generar) en
01–08, celda final de guardado en Drive en todos, checks de prerrequisitos
de etapa (adaptador en 04, trazas/retrievals de 05/06 en 08), ausencia de
pasos de máquina local, y **autocontención total: ninguna celda de código
importa ni ejecuta scripts `.py` del repo** (chequeo 7 del validador).

```bash
python3 build_notebooks.py
python3 validate_notebooks.py
```

## Opcional: `patch_gpu.py` (correr los scripts originales en GPU)

`patch_gpu.py` es otra vía, complementaria: parchea la copia clonada del repo
para que los scripts originales (que son 100% CPU) usen la GPU cuando existe
(`device_map` + tensores a CUDA en `train_memory*.py`, `answer.py`,
`executive*.py` y el backend local del agente). Los notebooks de arriba **no
lo necesitan** (usan Ollama para los modelos intermedios y código GPU
autocontenido para el entreno), pero si prefieres correr los scripts
originales tal cual en Colab, aplícalo sobre el clon después del notebook 01:

```bash
python colab/patch_gpu.py --root femsa-rag-graph-memo
```

Es idempotente (la segunda vez salta lo ya parcheado) y no toca el repo
publicado: solo la copia en Drive. Verificado aquí sobre copias de los 6
archivos: 18 reemplazos, segunda corrida sin cambios, archivos parcheados
que compilan (`py_compile` OK).

## Qué se validó aquí y qué falta por probar en Colab real

**Validado aquí (sin Colab):**

- Los 9 `.ipynb` son JSON nbformat válido; todas las celdas Python compilan
  (`py_compile`).
- Todos los archivos del repo que los notebooks referencian existen.
- El bloque Ollama (instalar si falta / arrancar si no responde / pull si
  falta el tag exacto) está en 00, 04, 05, 06, 07 y 08.
- Cada notebook lleva la misma base (GPU check + Drive + repo), la celda de
  datos (Drive → subir con `files.upload()` → generar con la extracción
  inline como último recurso) en 01–08, y la celda final de guardado en Drive
  con verificación en los 9. El NB02 además instala el driver `neo4j` (lo
  necesita la celda de carga inline) y llevaba el GPU check que le faltaba.
- Los prerrequisitos de etapa están verificados: el NB04 falla con mensaje
  claro si falta el adaptador del NB03, y el NB08 verifica las 6 entradas de
  05/06 (con opción de subirlas) y alinea los nombres de traza de Colab a los
  de CPU que espera el cálculo determinista inline.
- La lógica de parseo de `queries.cypher` del NB02 se probó contra el archivo
  real: 12 bloques, Q9 y Q10 con sus dos cláusulas MATCH juntas (un bug
  anterior solo tomaba 3 consultas; corregido y verificado).
- `agentic_rag/score_judge.py` compila y se probó (smoke test) con una traza
  fabricada en `/tmp` —solo verifica la lógica del scorer, no son resultados—:
  Q4 correcta con precisión de selección 1.00, Q1 incorrecta por fallo del
  juez tras 3 intentos, con su abstención.
- Ningún notebook depende de pasos de máquina local.

**Sin probar (hay que verificar en la primera corrida en Colab):**

- Instalación de Ollama en la instancia (apt `zstd` + `install.sh`) y que el
  servidor levante en background y sobreviva entre celdas.
- Que `qwen2.5:7b` quepa y rinda en la T4 junto a los embeddings (y que el
  14B quepa, si se prueba).
- El entrenamiento LoRA 1.5B en GPU (tiempos reales, memoria con batch 8).
- Que `relate serve` en background sobreviva entre celdas de Colab.
- El run real de `ragas` 0.4.4 con juez ChatOllama (si el piloto de una
  muestra completa, el resto es cuestión de tiempo).
- La corrida del juez (NB07): un llamado al juez por pregunta con 10 chunks,
  y si el 7B sigue los veredictos mejor que el 1.5B en CPU.
- Los tiempos estimados de la tabla de arriba.

Si algo de eso falla en la primera corrida, el fallo queda en el log de la
celda: no se inventan resultados.
