# femsa-rag-graph-memo

Este repositorio contiene experimentos, modelos y código de investigación del post de Substack «RAG clásico vs RAG agentic» (en preparación), centrados en comparar tres formas de darle conocimiento a un modelo sobre el mismo documento público: el **Informe Anual Integrado 2024 de FEMSA**. El proyecto utiliza la suite de herramientas de Python basadas en [`uv`](https://github.com/astral-sh/uv) para un manejo eficiente de entornos virtuales y dependencias.

Los tres sistemas contestan las **mismas 12 preguntas** sobre el informe, así que la comparación es directa: (a) RAG clásico, (b) grafo de conocimiento con ontología cerrada, consultado con Cypher en [Relate](https://relatedb.ai) (la base en memoria de Neo4j), y (c) MEMO (*Memory as a Model*, arXiv 2605.15156), un modelo pequeño entrenado con preguntas sintetizadas del propio informe.

---

## 🛠️ Requisitos Previos

* **Python 3.12+**
* [**`uv`**](https://github.com/astral-sh/uv) instalado en el sistema.
* [**Relate**](https://relatedb.ai) instalado para la parte del grafo (`curl -sSfL https://relatedb.ai/install.sh | bash`).
* CPU con ~6 GB de RAM para el demo de MEMO (el entrenamiento de la run 2 tomó ~4 h en CPU).

---

## 🚀 Instalación y Configuración

El proyecto está configurado para gestionar automáticamente el entorno virtual y las dependencias utilizando el `Makefile` y `pyproject.toml`.

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/czammar/femsa-rag-graph-memo.git
   cd femsa-rag-graph-memo
   ```

2. **Crear el entorno virtual e instalar dependencias:**
   ```bash
   make setup
   ```
   *Esto creará el entorno `.venv` e instalará las librerías necesarias mediante `uv sync`.*

3. **Descargar el informe y los modelos base:**
   ```bash
   make download          # PDF de FEMSA 2024 + extracción de texto (151 pp.)
   make download-models   # Qwen2.5-0.5B y 1.5B-Instruct a memo/models/
   ```

---

## 📂 Estructura del Proyecto

```text
femsa-rag-graph-memo/
├── Makefile              # Comandos de automatización para uv y el experimento
├── pyproject.toml        # Configuración del proyecto y dependencias
├── .python-version       # Versión de Python fijada (3.12.3)
├── docs/
│   └── ontology-and-questions.md   # Ontología v1 (9 tipos, 8 relaciones) + 12 preguntas
├── pipeline/             # Extracción, carga y consultas del knowledge graph
│   ├── extract.py            # Descarga el PDF, extrae texto y tripletas por reglas
│   ├── triples_seed.json     # 31 tripletas semilla verificadas
│   ├── build_cypher.py       # Genera load_graph.cypher desde las semilla
│   ├── schema.cypher         # 9 restricciones de unicidad
│   ├── load_graph.cypher     # MERGE idempotentes
│   ├── load_relate.py        # Carga esquema + grafo en Relate por Bolt
│   ├── queries.cypher        # Las 12 preguntas en Cypher, con respuesta esperada
│   ├── qa_to_triples.py      # Convierte pares QA en tripletas candidatas
│   ├── extracted_triples*.jsonl  # Candidatas (revisión humana antes de cargar)
│   ├── needs_computed_path.json  # Agregaciones que no son una tripleta
│   └── data/chunks.jsonl     # Texto extraído, una línea JSON por página (se genera con make extract)
├── memo/                 # Demo MEMO (Memory as a Model) a escala pequeña
    ├── generate_reflections.py   # Regenera el dataset de reflexiones (determinista)
    ├── reflection_qa.jsonl       # 446 pares QA (run 1)
    ├── reflection_qa_v2.jsonl    # 1,414 ejemplos con paráfrasis (run 2; se genera con make augment-memo-v2)
    ├── augment_v2.py             # 446 → 1,414 ejemplos
    ├── train_memory.py           # SFT LoRA run 1 (r=8, 1 época)
    ├── train_memory_v2.py        # SFT LoRA run 2 (r=32, 3 épocas)
    ├── executive.py / executive_v2.py  # Protocolo de inferencia MEMO
    ├── evaluate.py               # Scorer: exactitud + alucinación
    ├── eval_questions.json       # Las 12 preguntas con respuestas esperadas
    ├── results*.md / run_trace*.jsonl  # Resultados y trazas de cada corrida
    └── README.md                 # Runbook completo del demo MEMO
└── rag_baseline/         # Baseline RAG vectorial (MiniLM + lector Qwen2.5-1.5B)
    ├── retrieve.py           # Embeddings + top-5 por pregunta
    ├── answer.py             # Reader (responde solo desde los chunks)
    ├── score.py              # Scorer + clasificación de fallos
    ├── results.md            # Tabla por pregunta, modos de fallo, veredicto
    ├── retrieval.json        # Top-5 por pregunta (se genera con make rag-retrieve)
    ├── run_trace.jsonl       # Respuestas del reader + contexto usado (se genera con make rag-answer)
    └── README.md             # Runbook del baseline RAG
```

Los detalles de cada parte están en `pipeline/README.md`, `memo/README.md` y `rag_baseline/README.md`.

---

## 📊 El experimento y los resultados

Mismo corpus (FEMSA 2024), mismas 12 preguntas en 3 niveles (hecho directo, comparación/multi-salto, agregación). Una de ellas es una trampa deliberada: Proximidad Europa y Coca-Cola FEMSA crecieron ambas +14.2%.

| Sistema | Correctas | Con alucinación |
|---|---|---|
| **Grafo** (Cypher sobre tripletas semilla, Relate) | ~perfecto en nivel 1–2 **por construcción** | 0 |
| **MEMO run 1** (0.5B, LoRA r=8, 1 época, 446 QA) | **0/12** | 6/12 |
| **MEMO run 2** (0.5B, LoRA r=32, 3 épocas, 1,414 QA, EXEC 1.5B) | **2/12** (+1 parcial) | **2/12** |
| **RAG vectorial** (MiniLM top-5 + lector Qwen2.5-1.5B) | **2/12** (correctas Q6 y Q11; techo real 9/12: 3 cifras esperadas no están en el informe; detalle en [rag_baseline/results.md](rag_baseline/results.md)) | **2/12** |

El veredicto honesto: más datos (3.2×), más épocas (3×) y más LoRA (r=8 → r=32) fijaron las cifras simples y repetidas (571 estaciones, 307,197 vs 279,793) y bajaron las alucinaciones de 6/12 a 2/12, pero el sistema sigue fallando en todo lo que exige componer: sumar, comparar, atribuir un hecho a su división o nombrar una estrategia. En dos preguntas la memoria ya tenía el dato correcto y fue el ejecutivo quien lo estropeó al sintetizar. **La memoria en pesos es compresión con pérdida, y la pérdida se concentra justo donde se necesita precisión**: cifras exactas, comparaciones y agregaciones. El grafo no tiene ese problema porque la respuesta *es* el dato verificado.

Eso no invalida a MEMO: el paper usa un modelo 28× más grande, ~3,000× más datos y un ejecutivo de 32B/Gemini. Lo que el demo muestra es el punto del post: para cifras exactas, el sistema necesita el dato afuera (grafo o retrieval), no dentro del modelo.

---

## 📏 Cómo reproducir las métricas

Cada cifra de la tabla sale de correr el sistema contra las **mismas 12 preguntas** (`memo/eval_questions.json`). RAG y MEMO se puntúan con el mismo scorer (`memo/evaluate.py`: exactitud numérica + alucinación); el grafo se verifica con las 12 consultas Cypher de `pipeline/queries.cypher`. Todo en CPU.

### Grafo: 12/12

```bash
make setup             # entorno
make download          # PDF + texto + tripletas por reglas
make load-relate       # carga el grafo (con el servidor corriendo en otra terminal:
                       # relate serve --database ./pipeline/data/femsa.rl --bolt 7687 --auth none)
make queries           # las 12 preguntas en Cypher
```

Resultado: las 12 consultas devuelven la respuesta esperada (ver `pipeline/queries.cypher`, que trae la respuesta esperada por pregunta).

### RAG vectorial: 2/12

```bash
make setup             # entorno (incluye sentence-transformers para los embeddings)
make download          # PDF + texto (los chunks viven en pipeline/data/chunks.jsonl)
make rag-retrieve      # embeddings MiniLM + top-5 por pregunta (~1 min en CPU)
make rag-answer        # reader Qwen2.5-1.5B en CPU, solo desde los chunks (~24 min)
make rag-score         # escribe rag_baseline/results.md
```

Resultado: **2/12** (correctas Q6 y Q11), 2/12 con alucinación. Leer en `rag_baseline/results.md` (tabla por pregunta, modos de fallo y caveats). `make rag-answer` descarga el modelo 1.5B automáticamente (`make download-models`). Techo real sobre este set: 9/12 — las cifras esperadas de Q2 (+24.9%), Q7 (+27.5%) y Q9 (39.3%/35.8%/75.1%) no están en el informe (las dos primeras son del comunicado 4Q24; las terceras son participaciones calculadas).

### MEMO run 1: 0/12

```bash
make train-memo        # SFT LoRA r=8, 1 época (~35 min en CPU)
make eval-memo         # evaluación + control + puntuación
```

Resultado: **0/12**, 6/12 con alucinación. Leer en `memo/results.md` (control sin entrenar en `memo/results_control.md`).

### MEMO run 2: 2/12

```bash
make train-memo-v2     # SFT LoRA r=32, 3 épocas (~4 h en CPU, reanudable por época)
make eval-memo-v2      # evaluación + control + puntuación
```

Resultado: **2/12** (correctas Q4 y Q6, +1 parcial en Q3), 2/12 con alucinación. Leer en `memo/results_v2.md` (control en `memo/results_v2_control.md`).

---

## ▶️ Cómo reproducir

```bash
make setup             # entorno
make download          # PDF + extracción de texto + tripletas por reglas
make build-cypher      # genera pipeline/load_graph.cypher
make load-relate       # carga el grafo en Relate (con el servidor corriendo)
make queries           # corre las 12 preguntas en Cypher
make download-models   # modelos base Qwen2.5-0.5B y 1.5B
make train-memo        # run 1: LoRA r=8, 1 época (~35 min en CPU)
make eval-memo         # run 1: evaluación + control + puntuación
make train-memo-v2     # run 2: LoRA r=32, 3 épocas (~4 h en CPU)
make eval-memo-v2      # run 2: evaluación + control + puntuación
make rag-retrieve      # RAG: embeddings MiniLM + top-5 (~1 min en CPU)
make rag-answer        # RAG: reader Qwen2.5-1.5B en CPU (~24 min)
make rag-score         # RAG: escribe rag_baseline/results.md (2/12)
```

Ver `make help` para la lista completa.

---

## 📦 Qué no está en el repo (y por qué)

* **El PDF del informe** (18.8 MB): es público, pero el sitio en español bloquea descargas directas; `make download` lo baja (prueba la edición en español y, si falla, la edición en inglés, con las mismas cifras) y `pipeline/extract.py` regenera `data/chunks.jsonl`.
* **Los pesos de los modelos** (Qwen2.5-0.5B y 1.5B, ~4 GB): se descargan con `make download-models` desde Hugging Face.
* **Los adaptadores LoRA entrenados y los checkpoints** (28 MB – 79 MB por adaptador, checkpoints de hasta ~900 MB): binarios grandes; se regeneran con `make train-memo` / `make train-memo-v2`.
* **El store de Relate** (`pipeline/data/femsa.rl`): se genera al correr el runbook del grafo.
* **Los chunks extraídos** (`pipeline/data/chunks.jsonl`, 424 KB): se regeneran con `make extract`.
* **El dataset v2 de MEMO** (`memo/reflection_qa_v2.jsonl`, 431 KB): se regenera de forma determinista con `make augment-memo-v2` a partir de `memo/reflection_qa.jsonl` (que sí está en el repo); `make train-memo-v2` y `make eval-memo-v2` ya lo generan automáticamente.
* **El top-5 del RAG** (`rag_baseline/retrieval.json`, 198 KB): se regenera con `make rag-retrieve`.
* **La traza del reader RAG** (`rag_baseline/run_trace.jsonl`, 123 KB): se regenera con `make rag-answer`.
* **Logs de entrenamiento** (`*.log`): ruido de corrida, los resúmenes están en `memo/training_log_v2.json` y en los `results*.md`.

---

## 🧹 Limpieza

Para eliminar el entorno virtual generado y los archivos de caché:

```bash
make clean
```
