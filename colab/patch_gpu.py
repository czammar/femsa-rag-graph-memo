#!/usr/bin/env python3
"""Parche GPU para la copia clonada de femsa-rag-graph-memo (uso en Colab).

Los scripts del repo son 100% CPU: cargan los modelos con
``torch_dtype=torch.bfloat16`` sin ``device_map`` y los tensores quedan en
CPU, asi que en Colab correrian en CPU aunque el runtime tenga GPU (T4).
Este script aplica el cambio minimo para que usen la GPU cuando existe
y sigan igual en CPU cuando no existe:

  * ``device = "cuda" if torch.cuda.is_available() else "cpu"``
  * ``device_map=device`` en cada ``from_pretrained`` (train, reader,
    executive v1/v2, backend local del agente).
  * los tensores de entrada se mueven a ``device`` antes de
    ``generate`` / ``model(**batch)``.

Uso (dentro del notebook 00, despues de clonar el repo):

    !python colab/patch_gpu.py --root femsa-rag-graph-memo

No modifica el repo publicado: se corre sobre la copia clonada en Colab.
Si algun patron no se encuentra exactamente una vez, falla con error
en vez de dejar un parche a medias (un no-op silencioso seria peor).
Idempotencia: correrlo dos veces es seguro, la segunda vez salta cada
archivo ya parcheado (marca `# colab-gpu-patch aplicado`) sin duplicar
nada; si un patron no se encuentra, falla con error en vez de dejar un
parche a medias (un no-op silencioso seria peor).
"""

import argparse
import sys
from pathlib import Path

MARKER = "# colab-gpu-patch aplicado"

# (archivo, texto exacto a buscar, reemplazo, ocurrencias esperadas)
# El primer parche de cada archivo incluye MARKER; si el archivo ya lo
# trae, se salta entero (re-correr el parche es seguro: no duplica nada).
PATCHES = [
    # --- train_memory.py / train_memory_v2.py (mismo patron en ambos) ---
    ("memo/train_memory.py",
     "    tok = AutoTokenizer.from_pretrained(args.model)",
     "    device = \"cuda\" if torch.cuda.is_available() else \"cpu\"  # colab-gpu-patch aplicado\n"
     "    print(f\"device: {device}\")\n"
     "    tok = AutoTokenizer.from_pretrained(args.model)", 1),
    ("memo/train_memory.py",
     "        args.model, torch_dtype=torch.bfloat16)",
     "        args.model, torch_dtype=torch.bfloat16, device_map=device)", 1),
    ("memo/train_memory.py",
     "            out = model(**batch)",
     "            batch = {k: v.to(device) for k, v in batch.items()}\n"
     "            out = model(**batch)", 1),
    ("memo/train_memory_v2.py",
     "    tok = AutoTokenizer.from_pretrained(args.model)",
     "    device = \"cuda\" if torch.cuda.is_available() else \"cpu\"  # colab-gpu-patch aplicado\n"
     "    print(f\"device: {device}\")\n"
     "    tok = AutoTokenizer.from_pretrained(args.model)", 1),
    ("memo/train_memory_v2.py",
     "        args.model, torch_dtype=torch.bfloat16)",
     "        args.model, torch_dtype=torch.bfloat16, device_map=device)", 1),
    ("memo/train_memory_v2.py",
     "            out = model(**batch)",
     "            batch = {k: v.to(device) for k, v in batch.items()}\n"
     "            out = model(**batch)", 1),
    # --- rag_baseline/answer.py (reader 1.5B) ---
    ("rag_baseline/answer.py",
     "    tok = AutoTokenizer.from_pretrained(args.model)",
     "    device = \"cuda\" if torch.cuda.is_available() else \"cpu\"  # colab-gpu-patch aplicado\n"
     "    print(f\"device: {device}\")\n"
     "    tok = AutoTokenizer.from_pretrained(args.model)", 1),
    ("rag_baseline/answer.py",
     "    model = AutoModelForCausalLM.from_pretrained(args.model,\n"
     "                                                 torch_dtype=torch.bfloat16)",
     "    model = AutoModelForCausalLM.from_pretrained(args.model,\n"
     "                                                 torch_dtype=torch.bfloat16,\n"
     "                                                 device_map=device)", 1),
    ("rag_baseline/answer.py",
     "            ids = tok(text, return_tensors=\"pt\")",
     "            ids = {k: v.to(device) for k, v in tok(text, return_tensors=\"pt\").items()}", 1),
    # --- memo/executive.py / executive_v2.py (chat() compartido) ---
    ("memo/executive.py",
     "from peft import PeftModel",
     "from peft import PeftModel\n\nDEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"  # colab-gpu-patch aplicado", 1),
    ("memo/executive.py",
     "    ids = tok(text, return_tensors=\"pt\")",
     "    ids = {k: v.to(DEVICE) for k, v in tok(text, return_tensors=\"pt\").items()}", 1),
    ("memo/executive.py",
     "torch_dtype=torch.bfloat16)",
     "torch_dtype=torch.bfloat16, device_map=DEVICE)", 1),
    ("memo/executive_v2.py",
     "from peft import PeftModel",
     "from peft import PeftModel\n\nDEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"  # colab-gpu-patch aplicado", 1),
    ("memo/executive_v2.py",
     "    ids = tok(text, return_tensors=\"pt\")",
     "    ids = {k: v.to(DEVICE) for k, v in tok(text, return_tensors=\"pt\").items()}", 1),
    ("memo/executive_v2.py",
     "torch_dtype=torch.bfloat16)",
     "torch_dtype=torch.bfloat16, device_map=DEVICE)", 2),
    # --- agentic_rag/llm_backends.py (backend local del agente) ---
    ("agentic_rag/llm_backends.py",
     "        self._tok = None\n        self._model = None",
     "        self._tok = None\n        self._model = None\n"
     "        self._device = \"cuda\" if torch.cuda.is_available() else \"cpu\"  # colab-gpu-patch aplicado", 1),
    ("agentic_rag/llm_backends.py",
     "            str(self.model_path), torch_dtype=torch.bfloat16)",
     "            str(self.model_path), torch_dtype=torch.bfloat16,\n"
     "            device_map=self._device)", 1),
    ("agentic_rag/llm_backends.py",
     "        ids = self._tok(text, return_tensors=\"pt\")",
     "        ids = {k: v.to(self._device) for k, v in self._tok(text, return_tensors=\"pt\").items()}", 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".",
                    help="raiz del clon de femsa-rag-graph-memo")
    args = ap.parse_args()
    root = Path(args.root)
    # Agrupar por archivo para poder saltar un archivo ya parcheado.
    by_file = {}
    for rel, old, new, expected in PATCHES:
        by_file.setdefault(rel, []).append((old, new, expected))
    failures = []
    applied = 0
    for rel, patches in by_file.items():
        path = root / rel
        if not path.exists():
            failures.append(f"{rel}: archivo no encontrado")
            continue
        text = path.read_text()
        if MARKER in text:
            print(f"skip {rel}: parche ya aplicado")
            continue
        for old, new, expected in patches:
            count = text.count(old)
            if count != expected:
                failures.append(
                    f"{rel}: patron encontrado {count} veces, se esperaba {expected}: {old[:70]!r}")
                break
            text = text.replace(old, new)
            applied += 1
            print(f"ok  {rel}: {old[:60]!r} ({count}x)")
        else:
            path.write_text(text)
            continue
        # Si un parche del archivo fallo, no se escribe nada de ese archivo.
    if failures:
        print("\nFALLO el parche (los archivos con fallo no se tocaron):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    print(f"\nParche GPU aplicado: {applied} reemplazos en {len(by_file)} archivos.")


if __name__ == "__main__":
    main()
