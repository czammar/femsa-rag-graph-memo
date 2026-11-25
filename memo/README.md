# MEMO (Memory as a Model) — demo a escala pequeña sobre FEMSA 2024

Réplica en miniatura del paper **«MeMo: Memory as a Model»** (arXiv 2605.15156,
NUS/MIT CSAIL, mayo 2026) aplicada al informe anual FEMSA 2024, para el post
«RAG clásico vs RAG agentic» de Substack. **El demo se corrió de verdad de
principio a fin; los resultados son los reales, incluidos los malos.**

## Qué es MEMO (el paper)

En vez de buscar en un índice (RAG) o en un grafo, se entrena un modelo
pequeño de **MEMORY** con un dataset de preguntas-respuestas sintetizado del
corpus por un modelo GENERATOR congelado (5 pasos: extracción de hechos,
consolidación, verificación, surfacing de entidades, síntesis
cross-document). En inferencia, un modelo **EXECUTIVE** congelado responde
descomponiendo la pregunta en sub-preguntas, consultando a MEMORY y razonando
sobre sus respuestas. El EXECUTIVE nunca ve el corpus.

## Escala: el paper vs este demo

| | Paper | Este demo |
|---|---|---|
| MEMORY | Qwen2.5-14B-Instruct, 3 épocas | Qwen2.5-**0.5B**-Instruct, 1 época, LoRA r=8 |
| EXECUTIVE | Qwen2.5-32B-Instruct o Gemini-3-Flash | Qwen2.5-**0.5B** base (el mismo modelo, sin adaptador) |
| Dataset de reflexiones | ~1.3 M QA | **446 QA** |
| Hardware | 8×H100, ~72 GPU-horas | CPU (4 vCPU, ~6 GB RAM), **~34 min** |
| Corpus | NarrativeQA / MuSiQue | Informe anual FEMSA 2024 (151 pp., edición en inglés) |

## Qué se construyó (archivos en `memo/`)

- `generate_reflections.py` — regenera el dataset (determinista).
  Lee `pipeline/data/chunks.jsonl` y produce QA en español.
- `reflection_qa.jsonl` — **446 pares** (246 factoid, 93 comparación,
  40 multi-hop, 67 agregación; 103 citan 2+ páginas). Cada cifra se verificó
  contra el texto del chunk; las 12 preguntas de evaluación están excluidas
  (Jaccard máx. 0.71 vs umbral 0.80). Se usan los valores exactos del MD&A
  (307,197 no 307,200; 1,596 tiendas netas nuevas, no 1,236 — esa cifra no
  existe en el informe).
- `train_memory.py` — SFT con LoRA (r=8, α=16, lr 2e-4, batch 1,
  grad_accum 4, seq ≤128, bf16, CPU, `--max-seconds` como cortafuegos).
  Guarda checkpoints cada 25 pasos en `memory_model_ckpt/` y el adaptador
  final en `memory_model/`.
- `executive.py` — protocolo de inferencia: descomposición (prompt few-shot
  con el modelo base; si falla el parseo, cae a una regla documentada en la
  traza como `"rule"`), sub-respuestas de MEMORY (greedy, cortas) y síntesis.
  `--control` corre la base sin entrenar como MEMORY.
- `evaluate.py` — puntúa las 12 preguntas: exactitud del número esperado
  (con tolerancia de redondeo) y alucinación (cualquier número de la respuesta
  que no esté en el conjunto verificado del informe). Las preguntas sin
  números esperados se evalúan por palabras clave; sin keywords ni números,
  nunca se marcan correctas automáticamente.
- `eval_questions.json` — las 12 preguntas con respuestas esperadas y notas
  de ground-truth (Q3: 1,596 cuenta como correcto; Q6: se acepta 307,197 o
  307,200).
- `run_trace.jsonl` / `run_trace_control.jsonl` — traza completa de cada
  corrida (descomposición, sub-respuestas, respuesta final).
- `results.md` / `results_control.md` — tablas de resultados.
- `models/` — pesos locales de los modelos base (Qwen2.5-0.5B y 1.5B). **No incluidos en el repo**: se descargan con `make download-models` (ver el README principal).
- `train.log` / `train_run1.log` — logs de entrenamiento.

## Cómo se corrió (reproducible)

```bash
# 1. Dataset (determinista; md5 a4ccf20c2ea38d61fe50e992b20241c6)
python memo/generate_reflections.py

# 2. Entrenamiento (CPU, ~34 min, 1 época)
.venv/bin/python memo/train_memory.py \
  --model memo/models/Qwen2.5-0.5B-Instruct --epochs 1 \
  --max-seconds 3600 --out memo/memory_model

# 3. Evaluación MEMO (12 preguntas, ~7 min) y control (base sin entrenar)
.venv/bin/python memo/executive.py --all --adapter memo/memory_model --out memo/run_trace.jsonl
.venv/bin/python memo/executive.py --all --control --out memo/run_trace_control.jsonl

# 4. Puntuación
.venv/bin/python memo/evaluate.py --trace memo/run_trace.jsonl --out memo/results.md
.venv/bin/python memo/evaluate.py --trace memo/run_trace_control.jsonl --out memo/results_control.md
```

## Resultados reales (las 12 preguntas de evaluación)

| Sistema | Correctas | Con alucinación (nº no verificado) |
|---|---|---|
| MEMO (MEMORY entrenado) | **0/12** | 6/12 |
| Control (base sin entrenar) | **0/12** | 4/12 |
| Grafo (Cypher sobre tripletas semilla) | ~perfecto en nivel 1–2 **por construcción** | 0 (las respuestas son las tripletas verificadas) |
| RAG vectorial | **pendiente** (no se corrió; no se inventan cifras) | — |

Por nivel, MEMO entrenado: nivel 1: 0/4, nivel 2: 0/4, nivel 3: 0/4.

Ejemplos de lo que respondió MEMORY entrenado (todo mal en la cifra):

| Pregunta | Esperado | Respondió |
|---|---|---|
| Ingresos consolidados 2024 | Ps. 781,585 M, +11.2% | Ps. 157,968 M, +11.4% |
| Usuarios Spin by OXXO | 8.6 M (+24.9%) | 15 millones |
| Estaciones OXXO GAS | 571 | 139 estaciones |
| ¿Qué división creció más? (trampa +14.2%) | Empate | Coca-Cola FEMSA (cayó en la trampa) |
| Países de Proximidad Américas | México, Colombia, Chile, Perú | 13 países |

Y el control (base sin entrenar) respondió cosas como «13 estaciones» y
«13 tiendas, 57 aperturas»: también mal, pero con números genéricos.

## Run 2 — ¿mejora con más datos, más épocas y LoRA r=32? (sí, pero poco)

**Cambios** (todo con sufijo v2 en `memo/`):

- **Datos:** `augment_v2.py` expande `reflection_qa.jsonl` a
  `reflection_qa_v2.jsonl`: **446 → 1,414 ejemplos** (968 paráfrasis de
  pregunta, misma respuesta y mismas páginas fuente; cada cifra clave
  aparece ≥12 veces). Verificado por script: ninguna respuesta cambió sus
  números vs el par original; chequeo de fuga por Jaccard vs las 12
  preguntas de evaluación (umbral 0.80, 0 descartadas por fuga).
- **Entrenamiento:** `train_memory_v2.py`, mismo 0.5B, LoRA **r=32** (α=64,
  17,596,416 parámetros, 3.44%), **3 épocas** × 353 pasos = 1,059 pasos,
  CPU, checkpoints cada 25 pasos, reanudable (`--init-adapter`,
  `--skip-micro`). Pérdida media por época: 0.61 → 0.16 → 0.087
  (últimos 10 pasos: 0.25 → 0.10 → 0.074). Tiempo total ~4.1 h de pared,
  con 3 interrupciones por reinicio del servicio (épocas 1, 2 y 3 murieron
  en los pasos 60, 210 y 180; cada una se reanudó desde su checkpoint).
  Detalle en `training_log_v2.json`.
- **1.5B como MEMORY, descartado con medición:** 30.7 s/micro-paso →
  ~36 h para 3 épocas. Los pesos se descargaron de todos modos (ModelScope,
  porque HF estaba limitado a ~34 KB/s; 3,087,467,144 bytes) y el 1.5B se
  usa como EXECUTIVE.
- **EXECUTIVE:** `executive_v2.py` — EXECUTIVE = Qwen2.5-1.5B-Instruct
  (local), MEMORY = 0.5B + adaptador v2. La descomposición LLM **ahora sí
  funciona: 12/12** (con el prompt corregido: sin el ejemplo de OXXO GAS
  que el modelo copiaba, y con la instrucción de no copiar los ejemplos).
  Ver `decomp_test_1p5b_v2.json`.

**Resultados** (`results_v2.md`, `run_trace_v2.jsonl`, control en
`results_v2_control.md` / `run_trace_v2_control.jsonl`):

| Sistema | Correctas | Con alucinación |
|---|---|---|
| MEMO run 2 (MEMORY entrenado) | **2/12** (+1 parcial, Q3) | **2/12** |
| Control run 2 (base sin entrenar, misma EXEC 1.5B) | **0/12** | 6/12 |

Por nivel, MEMO run 2: nivel 1: 1/4, nivel 2: 1/4, nivel 3: 0/4.

**Qué mejoró:** Q4 (571 estaciones, antes 139) y Q6 (307,197 vs 279,793,
valores exactos del MD&A) ahora son correctas; Q2 da el 8.6 M bien (el
scorer exige también el +24.9%); las alucinaciones bajaron de 6/12 a 2/12.
**La mejora viene del entrenamiento de MEMORY, no del EXECUTIVE mejor:**
el control run 2 sigue en 0/12 con 6/12 alucinaciones.

**Qué no mejoró — dos modos de fallo:**

1. **El EXECUTIVE hace mal la aritmética con sub-respuestas correctas
   (Q1, Q3).** En Q1, MEMORY devolvió 307,197 y 279,793 correctos y el
   EXECUTIVE los sumó y presentó 586,989 como total consolidado. En Q3,
   MEMORY dio 26,322 y 1,596 correctos y el EXECUTIVE inventó
   «25,726 (26,322 − 6)».
2. **MEMORY falla en comparaciones y hechos abstractos (Q5, Q7, Q8, Q10,
   Q11, Q12).** La trampa +14.2% volvió a fallar (Coca-Cola FEMSA, 16.1%
   inventado). Q7 acierta la dirección pero con 52 M inventados. Q8 repite
   el «13 países» de la base. Q10 no menciona «FEMSA Forward». Q11 inventa
   una caída del 15.3% y adquisiciones. Q12 recuerda el 5.8% pero lo
   atribuye a la división equivocada.

**Veredicto run 2:** 0/12 → 2/12. Más datos (3.2×), más épocas (3×) y más
LoRA (r=8 → r=32) fijaron las cifras simples y repetidas, pero el sistema
sigue fallando en todo lo que exige componer: sumar, comparar, atribuir o
nombrar una estrategia. La pérdida llegó a ~0.07 (casi memorización del
dataset) y aun así las cifras no salieron: memorizar el texto de
entrenamiento no es lo mismo que recuperar la cifra ante una pregunta
nueva. Para el post, el punto queda más fino: la memoria en pesos es
compresión con pérdida, y la pérdida se concentra justo donde se necesita
precisión.

Reproducir:

```bash
# 1. Dataset v2
.venv/bin/python memo/augment_v2.py

# 2. Entrenamiento v2 (3 épocas, ~4 h en CPU con las reanudaciones)
.venv/bin/python memo/train_memory_v2.py --data memo/reflection_qa_v2.jsonl \
  --model memo/models/Qwen2.5-0.5B-Instruct --lora-r 32 --lora-alpha 64 \
  --epochs 1 --no-grad-checkpointing --max-seconds 6600 --out memo/memory_model_v2_ep1
# ... época 2 con --init-adapter memo/memory_model_v2_ep1, época 3 con
# --init-adapter memo/memory_model_v2_ep2 (mismo comando, --out distinto)

# 3. Evaluación v2 (~17 min) y control v2 (~15 min)
.venv/bin/python memo/executive_v2.py --all --adapter memo/memory_model_v2 --out memo/run_trace_v2.jsonl
.venv/bin/python memo/executive_v2.py --all --control --out memo/run_trace_v2_control.jsonl

# 4. Puntuación
.venv/bin/python memo/evaluate.py --trace memo/run_trace_v2.jsonl --eval memo/eval_questions.json --qa memo/reflection_qa_v2.jsonl --out memo/results_v2.md
.venv/bin/python memo/evaluate.py --trace memo/run_trace_v2_control.jsonl --eval memo/eval_questions.json --qa memo/reflection_qa_v2.jsonl --out memo/results_v2_control.md
```

## Veredicto para el post

**A esta escala, la memoria paramétrica no memoriza cifras exactas.**
El entrenamiento SFT sí cambió el *estilo* de las respuestas (la base dice
«13»; el modelo entrenado dice números con pinta de informe, 157,968), pero
no la exactitud: 0/12 en ambos casos. El modelo memoriza la *forma* de los
hechos (qué división, qué métrica, qué orden de magnitud) y rellena la cifra
con algo plausible. Por eso el grafo gana en cifras exactas: ahí la respuesta
*es* el dato verificado, no una reconstrucción.

Eso no invalida a MEMO: el paper usa un modelo 28× más grande, ~3,000× más
datos de entrenamiento y un EXECUTIVE de 32B/Gemini. Lo que el demo muestra
es justo el punto del post: **la memoria en pesos es compresión con pérdida;
para cifras exactas, el sistema necesita el dato afuera (grafo o retrieval),
no dentro del modelo.** Y también muestra que 1 época de LoRA sobre 446 QA
no basta para fijar cifras: harían falta más épocas, más datos y un modelo
mayor — exactamente lo que el paper ya dice (limitaciones: costo de
entrenamiento y escalado de la capacidad de MEMORY con el tamaño del corpus).

## Limitaciones honestas

- **EXECUTIVE y MEMORY comparten los pesos base** (0.5B): MEMORY = base +
  adaptador LoRA; EXECUTIVE = base con el adaptador desactivado. En el paper
  son modelos distintos y mucho mayores.
- **La descomposición LLM nunca funcionó**: el EXECUTIVE 0.5B no siguió el
  formato few-shot en ninguna de las 12 preguntas; todas cayeron al fallback
  por reglas (queda registrado como `"decomposition": "rule"` en la traza).
- **El EXECUTIVE de este demo es el agente** (yo) en la síntesis final:
  `executive.py` genera la respuesta final a partir de las sub-respuestas de
  MEMORY con un prompt de síntesis; no hay un modelo ejecutivo independiente
  razonando. Documentado, no disfrazado.
- **1 época, 446 ejemplos, LoRA r=8**: el entrenamiento bajó la pérdida de
  2.56 a ~1.0 (media 1.21, media últimos 10 pasos 1.04), pero eso no se
  tradujo en cifras correctas.
- **Evaluación pequeña**: 12 preguntas; el 0/12 es un dato demo, no una
  medida con significancia estadística.
- **El scorer inicial tenía un bug**: las preguntas sin números esperados
  (Q8, Q10, Q11) se marcaban correctas por vacuidad. Se corrigió con
  palabras clave y el resultado quedó en 0/12 (antes decía 3/12).

## Incidentes durante la corrida

- **Descarga del modelo**: el cliente de Hugging Face se atascaba (0 KB/30–60 s).
  Se bajó con curl con resume; el archivo se completó a 988,097,824 bytes
  (el tamaño objetivo asumido al inicio era erróneo y dio un 416 al reanudar:
  el archivo ya estaba completo). Los pesos viven en
  `memo/models/Qwen2.5-0.5B-Instruct/`; `executive.py` apunta ahí, no al
  caché de HF.
- **Primera corrida de entrenamiento murió en el paso 95/112** por un
  reinicio del servicio (no por OOM: el log termina limpio). Se re-corrió
  desde cero tras añadir checkpoints cada 25 pasos
  (`memory_model_ckpt/step_25|50|75|100`); la segunda corrida completó los
  112 pasos en 2,029 s sin cortafuegos.
- **La corrida 1 se guardó** como `train_run1.log` (pasos 1–95, pérdida
  2.56 → ~0.76).

## Extra: extractor desacoplado (idea del usuario)

`pipeline/qa_to_triples.py` convierte los pares de `reflection_qa.jsonl` en
tripletas candidatas restringidas a la ontología cerrada (9 tipos de entidad,
8 relaciones), cada una con su página fuente. Las preguntas de agregación no
se fuerzan a tripleta: van a una lista separada `needs_computed_path`.
Salidas:

- `pipeline/extracted_triples_from_qa.jsonl` — **92 tripletas**, todas con
  `estado: "pendiente_revision"` (misma regla que siempre: **nunca se cargan
  tripletas sin revisar**).
- `pipeline/needs_computed_path.json` — 67 ítems de agregación.
- Cobertura de las 31 tripletas semilla: **29/31**. Faltan
  (Proximidad Américas, mide, Aperturas) porque 1,236 no está en el informe
  (ausencia correcta) y (Coca-Cola FEMSA, mide, Clientes Juntos+), que sí
  está capturada como (Juntos+, mide, Clientes Juntos+, 1.3).
- Imperfecciones conocidas (las atrapará la revisión humana): el triple
  «Proximidad Américas mide Tiendas = 249» es solo tiendas en EE. UU. (sin
  calificador de país en el formato); se pierde el signo en variaciones
  negativas (p. ej. −59.3% queda como 59.3).
