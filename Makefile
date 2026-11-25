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
	train-memo train-memo-v2 eval-memo eval-memo-v2

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

## Train MEMO run 2: LoRA r=32, 3 epochs (~4 h CPU, resumable per epoch)
train-memo-v2: sync download-models
	$(PY) memo/train_memory_v2.py --data memo/reflection_qa_v2.jsonl --model memo/models/Qwen2.5-0.5B-Instruct --lora-r 32 --lora-alpha 64 --epochs 1 --no-grad-checkpointing --max-seconds 6600 --out memo/memory_model_v2_ep1
	$(PY) memo/train_memory_v2.py --data memo/reflection_qa_v2.jsonl --model memo/models/Qwen2.5-0.5B-Instruct --lora-r 32 --lora-alpha 64 --epochs 1 --no-grad-checkpointing --max-seconds 6600 --init-adapter memo/memory_model_v2_ep1 --out memo/memory_model_v2_ep2
	$(PY) memo/train_memory_v2.py --data memo/reflection_qa_v2.jsonl --model memo/models/Qwen2.5-0.5B-Instruct --lora-r 32 --lora-alpha 64 --epochs 1 --no-grad-checkpointing --max-seconds 6600 --init-adapter memo/memory_model_v2_ep2 --out memo/memory_model_v2

## Evaluate MEMO run 2 + control, then score
eval-memo-v2: sync
	$(PY) memo/executive_v2.py --all --adapter memo/memory_model_v2 --out memo/run_trace_v2.jsonl
	$(PY) memo/executive_v2.py --all --control --out memo/run_trace_v2_control.jsonl
	$(PY) memo/evaluate.py --trace memo/run_trace_v2.jsonl --eval memo/eval_questions.json --qa memo/reflection_qa_v2.jsonl --out memo/results_v2.md
	$(PY) memo/evaluate.py --trace memo/run_trace_v2_control.jsonl --eval memo/eval_questions.json --qa memo/reflection_qa_v2.jsonl --out memo/results_v2_control.md
