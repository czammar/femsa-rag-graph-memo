# COLORS
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

TARGET_MAX_CHAR_NUM=20

# Variables
UV := $(shell command -v uv 2> /dev/null)
PYTHON_VERSION := 3.12.3
VENV_BIN := .venv/bin
PY := $(VENV_BIN)/python

.PHONY: all check-python-version get-python-version install sync setup clean help \
	download extract build-cypher load-relate queries download-models \
	train-memo train-memo-v2 eval-memo eval-memo-v2 \
	rag-retrieve rag-answer rag-score \
	rag-retrieve-hybrid rag-answer-hybrid rag-score-hybrid \
	rag-expand rag-retrieve-expanded rag-answer-expanded rag-score-expanded \
	ragas-metrics agentic-run agentic-score

## Check is python version of the project
check-python-version:
	@test "$$(cat .python-version)" = "$(PYTHON_VERSION)" || \
		(echo ".python-version must be $(PYTHON_VERSION)"; exit 1)

## Download & pin python version for the project
get-python-version:
	uv python install $(PYTHON_VERSION)
	uv python pin $(PYTHON_VERSION)

## Show help with `make help`
help:
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  ${YELLOW}%-$(TARGET_MAX_CHAR_NUM)s${RESET} ${GREEN}%s${RESET}\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)

## Default action by executing `make` (install uv + sync)
all: install sync

## Installing uv
install:
ifdef UV
	@echo "✅ 'uv' ya está instalado en: $(UV)"
else
	@echo "📥 'uv' no fue detectado. Instalando mediante el script oficial..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "⚠️  Nota: Si es la primera vez que instalas 'uv', es posible que necesites reiniciar tu terminal o recargar tu perfil (source ~/.zshrc o ~/.bashrc)."
endif

## Syncing project dependencies with uv
sync: get-python-version check-python-version
	@echo "🔄 Sincronizando dependencias del proyecto con uv..."
	@if command -v uv > /dev/null; then \
		uv sync; \
	else \
		echo "❌ Error: 'uv' no está disponible en el PATH actual. Si lo acabas de instalar, ejecuta 'source ~/.zshrc' (o tu equivalente) e intenta de nuevo."; \
		exit 1; \
	fi
	@echo "🚀 Entorno virtual listo y actualizado."

## Setup alias: install + sync (lo que dice el README)
setup: all

## Cleaning the venv environment
clean:
	@echo "🧹 Deleting .venv..."
	@rm -rf .venv
	@find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@echo " Cleaning .venv complete"

## Download FEMSA 2024 PDF + extract text (151 pp.) and rule-based triples
download: sync
	$(PY) pipeline/extract.py

## Extract text and triples from an already downloaded PDF (alias de download)
extract: download

## Generate pipeline/load_graph.cypher from the seed triples
build-cypher: sync
	$(PY) pipeline/build_cypher.py

## Load schema + graph into Relate via Bolt (needs `relate serve` running)
load-relate: sync build-cypher
	@echo "ℹ️  Si Relate no está corriendo: relate serve --database ./pipeline/data/femsa.rl --bolt 7687 --auth none"
	$(PY) pipeline/load_relate.py

## Run the 12 questions in Cypher against the Relate store
queries:
	relate -d ./pipeline/data/femsa.rl query < pipeline/queries.cypher

## Download base models (Qwen2.5-0.5B and 1.5B-Instruct) to memo/models/
download-models: sync
	$(PY) -c "from huggingface_hub import snapshot_download; \
		snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='memo/models/Qwen2.5-0.5B-Instruct'); \
		snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='memo/models/Qwen2.5-1.5B-Instruct')"

## Train MEMO run 1: LoRA r=8, 1 epoch (~35 min CPU)
train-memo: sync download-models
	$(PY) memo/train_memory.py --model memo/models/Qwen2.5-0.5B-Instruct --epochs 1 --max-seconds 3600 --out memo/memory_model

## Evaluate MEMO run 1 + control, then score
eval-memo: sync
	$(PY) memo/executive.py --all --adapter memo/memory_model --out memo/run_trace.jsonl
	$(PY) memo/executive.py --all --control --out memo/run_trace_control.jsonl
	$(PY) memo/evaluate.py --trace memo/run_trace.jsonl --out memo/results.md
	$(PY) memo/evaluate.py --trace memo/run_trace_control.jsonl --out memo/results_control.md

## Generate MEMO run 2 dataset (1,414 examples; deterministic, from reflection_qa.jsonl)
augment-memo-v2: sync
	$(PY) memo/augment_v2.py

## Train MEMO run 2: LoRA r=32, 3 epochs (~4 h CPU, resumable per epoch)
train-memo-v2: sync download-models augment-memo-v2
	$(PY) memo/train_memory_v2.py --data memo/reflection_qa_v2.jsonl --model memo/models/Qwen2.5-0.5B-Instruct --lora-r 32 --lora-alpha 64 --epochs 1 --no-grad-checkpointing --max-seconds 6600 --out memo/memory_model_v2_ep1
	$(PY) memo/train_memory_v2.py --data memo/reflection_qa_v2.jsonl --model memo/models/Qwen2.5-0.5B-Instruct --lora-r 32 --lora-alpha 64 --epochs 1 --no-grad-checkpointing --max-seconds 6600 --init-adapter memo/memory_model_v2_ep1 --out memo/memory_model_v2_ep2
	$(PY) memo/train_memory_v2.py --data memo/reflection_qa_v2.jsonl --model memo/models/Qwen2.5-0.5B-Instruct --lora-r 32 --lora-alpha 64 --epochs 1 --no-grad-checkpointing --max-seconds 6600 --init-adapter memo/memory_model_v2_ep2 --out memo/memory_model_v2

## Evaluate MEMO run 2 + control, then score
eval-memo-v2: sync augment-memo-v2
	$(PY) memo/executive_v2.py --all --adapter memo/memory_model_v2 --out memo/run_trace_v2.jsonl
	$(PY) memo/executive_v2.py --all --control --out memo/run_trace_v2_control.jsonl
	$(PY) memo/evaluate.py --trace memo/run_trace_v2.jsonl --eval memo/eval_questions.json --qa memo/reflection_qa_v2.jsonl --out memo/results_v2.md
	$(PY) memo/evaluate.py --trace memo/run_trace_v2_control.jsonl --eval memo/eval_questions.json --qa memo/reflection_qa_v2.jsonl --out memo/results_v2_control.md

## Retrieve top-5 chunks per question with MiniLM embeddings (~1 min CPU)
rag-retrieve: download
	$(PY) rag_baseline/retrieve.py

## Answer the 12 questions with the Qwen2.5-1.5B reader, only from retrieved chunks (~24 min CPU)
rag-answer: rag-retrieve download-models
	$(PY) rag_baseline/answer.py

## Score the RAG baseline and write rag_baseline/results.md
rag-score: rag-answer
	$(PY) rag_baseline/score.py

## Retrieve top-5 with hybrid BM25 Okapi + MiniLM, RRF k=60 (~1 min CPU)
rag-retrieve-hybrid: download
	$(PY) rag_baseline/retrieve_hybrid.py

## Answer the 12 questions with the hybrid retrieval top-5 (~21 min CPU)
rag-answer-hybrid: rag-retrieve-hybrid download-models
	$(PY) rag_baseline/answer.py --retrieval rag_baseline/retrieval_hybrid.json --out rag_baseline/run_trace_hybrid.jsonl

## Score the hybrid RAG run and write rag_baseline/results_hybrid.md
rag-score-hybrid: rag-answer-hybrid
	$(PY) rag_baseline/score_hybrid.py

## Expand the 12 questions: manual English translation + domain keywords (<1 s, with anti-leakage guard)
rag-expand:
	$(PY) rag_baseline/expand_queries.py

## Retrieve top-5 with the expanded queries, same hybrid RRF k=60 (~1 min CPU)
rag-retrieve-expanded: rag-expand download
	$(PY) rag_baseline/retrieve_expanded.py

## Answer the 12 questions with the expanded retrieval top-5 (~26 min CPU)
rag-answer-expanded: rag-retrieve-expanded download-models
	$(PY) rag_baseline/answer.py --retrieval rag_baseline/retrieval_expanded.json --out rag_baseline/run_trace_expanded.jsonl

## Score the expanded RAG run and write rag_baseline/results_expanded.md
rag-score-expanded: rag-answer-expanded
	$(PY) rag_baseline/score_expanded.py

## Compute RAGAS metrics (deterministic, from RAGAS definitions) for the 3 RAG variants (~2 s + ~36 s relevancy proxy)
ragas-metrics: rag-score-expanded
	$(PY) rag_baseline/ragas/compute_metrics.py
	$(PY) rag_baseline/ragas/compute_relevancy.py

## Run the agentic RAG loop (LangChain + LangGraph, ReAct textual) over the 12 questions (~65 min CPU, resumable; needs `relate serve` running + agentic_rag/.venv, see agentic_rag/README.md)
agentic-run: download download-models
	agentic_rag/.venv/bin/python agentic_rag/run_agentic.py

## Score the agentic RAG run and write agentic_rag/results_agentic.md (2/12)
agentic-score: agentic-run
	agentic_rag/.venv/bin/python agentic_rag/score_agentic.py
