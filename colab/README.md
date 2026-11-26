# Notebooks de Google Colab — Experimento FEMSA 2024

Repite el experimento completo (**RAG clásico vs grafo vs MEMO vs RAG agéntico**,
informe anual FEMSA 2024) en **Google Colab con GPU T4**. Todo corre en Colab:
nada de pasos de máquina local. Cada notebook es autocontenido (monta Drive,
clona el repo, instala sus dependencias y, donde hace falta, instala y arranca
Ollama dentro de la propia instancia).

## Orden de ejecución

| # | Notebook | Qué hace | Ollama |
|---|----------|----------|--------|
| 00 | `00_ollama_setup.ipynb` | Setup canónico de Ollama: instala, arranca el servidor, baja qwen2.5:1.5b y qwen2.5:7b | sí (completo) |
| 01 | `01_setup_y_extraccion.ipynb` | Drive, repo, dependencias, descarga del PDF y extracción (151 págs.) | solo verifica |
| 02 | `02_grafo_relate.ipynb` | Cypher de carga, instala Relate, carga el grafo (27 nodos, 39 relaciones), 12 preguntas | no |
| 03 | `03_memo_entrenamiento.ipynb` | Dataset expandido v3 (objetivo 1,500–2,000 pares) + entreno MEMORY 1.5B con LoRA en GPU | no (entrena con HF) |
| 04 | `04_memo_evaluacion.ipynb` | Protocolo MEMO (ejecutivo vía Ollama) + control + scorer; genera set de eval ampliado (opcional) | sí, ejecutivo |
| 05 | `05_rag_baseline.ipynb` | Retrieval denso + lector vía Ollama + scorer | sí, lector |
| 06 | `06_rag_hibrido_y_expansion.ipynb` | BM25 híbrido (RRF k=60) + expansión de queries + lector vía Ollama + scorers | sí, lector |
| 07 | `07_rag_agentico.ipynb` | Agente LangChain + LangGraph con LLM vía Ollama (default 7B) + scorer | sí, LLM del agente |
| 08 | `08_ragas.ipynb` | RAGAS deterministas + run real con la librería `ragas` y juez vía Ollama | sí, juez |

## Antes de empezar

1. **Activa la GPU:** Runtime > Change runtime type > **T4 GPU**. Verifica con
   `!nvidia-smi` (el notebook 00 lo hace).
2. **Drive:** todo vive en `/content/drive/MyDrive/femsa-rag` (cámbiala en la
   celda de parámetros de cada notebook). Ahí quedan: el repo clonado, los
   chunks, las trazas, los resultados y los adaptadores LoRA.
3. **Corre los notebooks en orden** (o solo el que necesites: cada uno es
   autocontenido, pero los prerrequisitos están escritos arriba de cada uno).

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

Los modelos intermedios —lector RAG, ejecutivo MEMO, juez RAGAS, LLM del
agente— corren vía **Ollama instalado en la propia instancia de Colab**:

- **Un solo servidor** para todos los notebooks (puerto 11434, localhost).
- **Cambio de modelo con un parámetro** en la parte superior de cada notebook
  (`qwen2.5:1.5b` para paridad con los runs originales en CPU, `qwen2.5:7b`
  para el run con modelo más grande; el agente usa 7B por defecto).
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
| Agente 7B, 12 preguntas (NB07) | ~10–20 min |
| RAGAS real, 36 muestras (NB08) | acotado por `TIME_BUDGET_MIN` (default 90 min) |

Colab gratuito desconecta tras ~90 min de inactividad y 12 h como máximo:
mantén la pestaña abierta en las corridas largas. Los checkpoints (NB03) y
las trazas reanudables (NB07) permiten seguir donde se cortó.

## Las dos mejoras que permite Colab

1. **Más preguntas:** el notebook 04 genera un set de evaluación ampliado
   desde las tripletas semilla (plantillas por tipo de tripleta). Cada cifra
   esperada se verifica por grep literal en el corpus: lo que no aparece
   literal, se descarta — nunca se adivina. La salida lleva
   `"needs_review": true`: hay que revisar las preguntas a mano antes de
   evaluar con ellas.
2. **Mejor entrenamiento MEMO:** notebook 03 entrena MEMORY con el 1.5B
   (en CPU era el 0.5B) sobre un dataset expandido v3 (objetivo 1,500–2,000
   pares, más variación de paráfrasis por hecho), 3 épocas, LoRA r=32, en GPU.

## Mismo scorer, mismas reglas, en todas partes

- Mismas 12 preguntas (`memo/eval_questions.json`), mismo scorer
  (`memo/evaluate.py` y sus derivados en `rag_baseline/` y `agentic_rag/`).
- Techo real **9/12** (3 cifras esperadas no están en el informe), regla Q3
  con **1,596** aperturas netas, conflicto de ground-truth Q7 documentado.
- Las cifras RAGAS deterministas siguen siendo las citables si el run real
  con juez Ollama no completa; el notebook 08 compara ambos si completa.

## Regenerar los notebooks

`build_notebooks.py` genera los 9 `.ipynb` (nbformat). `validate_notebooks.py`
comprueba: JSON válido, `py_compile` de las celdas Python (saltando celdas
`%pip` / `!` / `%%bash`), existencia de los archivos del repo referenciados,
presencia del bloque Ollama en 00/04/05/06/07/08, y ausencia de pasos de
máquina local.

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
- La lógica de parseo de `queries.cypher` del NB02 se probó contra el archivo
  real: 12 bloques, Q9 y Q10 con sus dos cláusulas MATCH juntas (un bug
  anterior solo tomaba 3 consultas; corregido y verificado).
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
- Los tiempos estimados de la tabla de arriba.

Si algo de eso falla en la primera corrida, el fallo queda en el log de la
celda: no se inventan resultados.
