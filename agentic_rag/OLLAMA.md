# RAG agéntico con Ollama y un modelo más grande — FEMSA 2024

El mismo agente que sacó **2/12 con Qwen2.5-1.5B-Instruct local**, pero con
un cerebro más grande servido por Ollama. Mismo grafo LangGraph, mismas
3 herramientas, mismos prompts, mismo scorer, mismas 12 preguntas:
**solo cambia el modelo**. Por eso la comparación es limpia: si el
marcador sube, subió por el modelo, no por cambiar el experimento.

## Por qué

El run de 1.5B (ver `results_agentic.md`) no mide lo agéntico, mide un
modelo que no sabe manejar herramientas: hizo 24 llamadas, las 24 a
`graph_query`, nunca usó `vector_search` ni `calculator`, repitió la
misma consulta fallida 3 veces y en Q11 tenía la respuesta y se
abstuvo. La lectura del post es que "agéntico" es una propiedad del
modelo, no del bucle — y este run es la prueba: mismo bucle, modelo
capaz. Si con 7B/14B el agente usa las tres herramientas y compone
Cypher sin inventar el esquema, el bucle sí suma.

## Requisitos

| Modelo | Disco (descarga) | RAM recomendada | Para qué |
|--------|------------------|-----------------|----------|
| `qwen2.5:7b-instruct` (por defecto) | ~4.7 GB | ~8 GB | El primer salto: 7B ya sigue instrucciones y compone mejor |
| `qwen2.5:14b-instruct` | ~9 GB | ~16 GB | La prueba fuerte, si tu máquina lo aguanta |

Apple Silicon (M1/M2/M3/M4) va muy bien: Ollama usa la GPU integrada
y la memoria unificada. En Linux solo-CPU funciona pero es más lento
(ver tiempos estimados abajo). Windows: Ollama corre con GPU NVIDIA
si la hay, si no, CPU.

## 1. Instalar Ollama

```bash
# macOS (con Homebrew; la app se inicia sola)
brew install --cask ollama
# o descarga el instalador en https://ollama.com/download

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: instalador en https://ollama.com/download
```

Comprueba que responde:

```bash
curl http://localhost:11434
# -> "Ollama is running"
```

Si no responde, arráncalo a mano (`ollama serve`) o abre la app de
Ollama. El agente se conecta a `http://localhost:11434` por defecto
(cámbialo con la variable `OLLAMA_HOST` o el flag `--host`).

## 2. Bajar el modelo

```bash
ollama pull qwen2.5:7b-instruct
# Opción más grande (si tienes ~16 GB de RAM):
ollama pull qwen2.5:14b-instruct

ollama list   # verifica que el modelo está descargado
ollama run qwen2.5:7b-instruct "Hola, ¿qué modelo eres?"
# Prueba rápida: debe contestar en español que es Qwen2.5-7B.
```

## 3. El experimento

```bash
git clone https://github.com/czammar/femsa-rag-graph-memo
cd femsa-rag-graph-memo
make setup            # entorno raíz (uv sync)
make download         # PDF del informe + extracción de texto (151 pp.)

# Relate (el grafo; si no lo tienes):
curl -sSfL https://relatedb.ai/install.sh | bash
# En una terminal aparte, sirve el store:
relate serve --database ./pipeline/data/femsa.rl --bolt 7687 --auth none
make load-relate      # carga el grafo (otra terminal, con el servidor corriendo)
```

## 4. El venv del agente

El agente vive en su propio venv (`agentic_rag/.venv`) y comparte el
stack ML pesado (torch, transformers, sentence-transformers) desde
`memo/.venv` con un archivo `.pth`, igual que la corrida local:

```bash
python3 -m venv agentic_rag/.venv
agentic_rag/.venv/bin/pip install langchain==1.5.2 langgraph==1.2.24 \
  langchain-huggingface==1.3.0 langchain-core==1.8.5 neo4j==6.4.1 \
  langsmith==0.16.2 langchain-ollama==1.2.0
# El stack ML pesado NO se reinstala: se comparte desde memo/.venv con
# un archivo .pth (agentic_rag/.venv/lib/python3.12/site-packages/memo_venv.pth
# apunta a memo/.venv/lib/python3.12/site-packages). Si memo/.venv no
# existe, crea el venv completo instalando además torch==2.15.0+cpu
# transformers==5.24.1 sentence-transformers==6.3.0 rank_bm25
# numpy==2.5.5 pydantic==2.14.4.
```

`langchain-ollama` es la única dependencia nueva respecto a la corrida
local: es el cliente LangChain para Ollama (`ChatOllama`).

## 5. Correr

```bash
# Corrida completa (12 preguntas, reanudable)
make agentic-run-ollama
# Con el modelo de 14B:
make agentic-run-ollama MODEL=qwen2.5:14b-instruct
# O directo, sin make:
agentic_rag/.venv/bin/python agentic_rag/run_agentic.py --llm ollama \
  --model qwen2.5:7b-instruct
# Una sola pregunta (para probar que todo conecta):
agentic_rag/.venv/bin/python agentic_rag/run_agentic.py --llm ollama --only Q4

# Scoring + informe (su propio archivo, nunca toca el del 1.5B)
make agentic-score-ollama
```

La traza va a `agentic_rag/run_trace_agentic_ollama.jsonl` y el
informe a `agentic_rag/results_agentic_ollama.md`. La corrida de 1.5B
(`run_trace_agentic.jsonl`, `results_agentic.md`) no se toca.

## 6. Tiempos (estimación, no medición)

La corrida de 1.5B en CPU tardó **65.3 min** (media 326 s por
pregunta). Con Ollama en Apple Silicon, **estimación: ~10–20 min para
las 12 preguntas con el 7B** (las llamadas al LLM son de 150 tokens
como máximo y el modelo corre en GPU integrada). En Linux solo-CPU
será más lento; con el 14B, más lento aún. Son estimaciones: los
tiempos reales se anotan en el informe al correrlo.

## 7. Qué comparar

Mismo scorer, mismas 12 preguntas, así que la tabla es directamente
comparable:

| Sistema | Correctas |
|---------|-----------|
| Grafo (Relate, Cypher) | 12/12 |
| RAG híbrido + expansión | 5/12 |
| **RAG agéntico Ollama 7B (este run)** | **?/12** |
| RAG agéntico local 1.5B | 2/12 |

Y además del marcador, mira en la traza lo que el 1.5B no hizo:

- **¿Usó `vector_search` y `calculator`?** En el run de 1.5B fueron 0
  y 0 de 24 llamadas. Si el 7B las usa, el bucle ya está vivo.
- **¿Corrige tras un error?** El 1.5B repetía la misma consulta 3
  veces. Un modelo capaz cambia el nombre, la relación o el RETURN.
- **¿Para cuando ya tiene la evidencia?** Q11 es el caso de prueba:
  la primera consulta devolvió "Spin" y el 1.5B se abstuvo igual.

## 8. Después de la corrida: verificar los modos de fallo a mano

El scorer escribe los modos de fallo como `PENDIENTE` la primera vez
(igual que en la corrida local: los modos se verifican contra el
corpus y la traza, no se inventan). Para completar el informe:

1. Lee `results_agentic_ollama.md` y, para cada pregunta incorrecta,
   verifica en `run_trace_agentic_ollama.jsonl` qué pasó y escríbelo
   en `VERIFIED_MODES` dentro de `agentic_rag/score_agentic.py`
   (una frase por pregunta, como las del run de 1.5B).
2. Vuelve a correr el scorer **sin** `--blank-modes` (con el mismo
   `--model-label`), para que escriba los modos reales y el veredicto:

```bash
agentic_rag/.venv/bin/python agentic_rag/score_agentic.py \
  --trace agentic_rag/run_trace_agentic_ollama.jsonl \
  --out agentic_rag/results_agentic_ollama.md \
  --model-label "Ollama qwen2.5:7b-instruct" \
  --system-name "RAG agentico Ollama (este run)"
```

## Troubleshooting

- **"Ollama no responde en http://localhost:11434"**: Ollama no está
  corriendo. `ollama serve` en una terminal (o abre la app). Si tu
  servidor está en otra máquina o puerto, usa
  `OLLAMA_HOST=http://tu-host:11434 make agentic-run-ollama` o el
  flag `--host`.
- **"model not found" / error 404 del modelo**: el modelo no está
  bajado. `ollama pull qwen2.5:7b-instruct` (ver paso 2).
- **Out of memory / el modelo no carga**: baja al 7B si estabas en
  14B, o cierra otras apps. Ollama descarga por defecto el
  cuantizado Q4_K_M; no hay un 7B "más chico" oficial que valga la
  pena para este experimento.
- **Muy lento en CPU vieja**: esperado; el 7B en CPU sin GPU es
  varias veces más lento que en Apple Silicon. Deja la corrida
  corriendo (es reanudable: si se interrumpe, `make
  agentic-run-ollama` retoma donde quedó).
- **El cliente de Ollama falla al arrancar con un error raro de URL o
  de puerto**: algunas terminales exportan variables de proxy
  (`http_proxy`, `no_proxy`) que el cliente lee al importarse. Prueba
  corriendo con las variables de proxy quitadas para ese comando.
- **`relate` no responde / grafo vacío**: `graph_query` necesita el
  servidor Relate corriendo (paso 3) y el store cargado
  (`make load-relate`). Sin grafo, el agente solo tiene
  `vector_search` y `calculator`.

## Qué NO se probó aquí

La ruta Ollama **no se corrió de punta a punta en la VM del autor**
(no hay servidor Ollama ni modelo de 7B ahí, y la VM tiene 7 GB de
RAM: el 7B apenas cabría). Lo verificado en esta VM: el código
compila, los flags existen (`run_agentic.py --help`), el backend
Ollama se instancia sin servidor, y sin servidor el agente falla
rápido con el error amable de arriba y registra el fallo por pregunta
sin tumbar la corrida. La primera corrida real es la tuya: si algo
falla en tu máquina, el error queda en la traza y en
`results_agentic_ollama.md`.
