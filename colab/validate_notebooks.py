#!/usr/bin/env python3
"""Valida los notebooks de colab/ sin ejecutarlos en Colab.

- nbformat.validate (JSON bien formado, esquema nbformat 4).
- py_compile de cada celda de codigo Python (se saltan celdas %pip, !shell
  y %%bash, que no son Python).
- Existencia de los archivos del repo que los notebooks referencian.
- Presencia del bloque Ollama (instalar/arrancar/pull) en 00, 04, 05, 06, 07, 08.
- Ausencia de pasos de maquina local (la guia OLLAMA.md local no es el camino).
"""
import ast
import json
import re
import sys
from pathlib import Path

import nbformat as nbf

COLAB = Path(__file__).resolve().parent
GOAL = COLAB.parent

NOTEBOOKS = sorted(COLAB.glob("*.ipynb"))
errors, warnings = [], []


def check(cond, msg):
    if not cond:
        errors.append(msg)


for nb_path in NOTEBOOKS:
    nb = nbf.read(nb_path, as_version=4)
    try:
        nbf.validate(nb)
    except Exception as e:
        errors.append(f"{nb_path.name}: nbformat invalido: {e}")
    py_cells = [c for c in nb.cells if c.cell_type == "code"]
    md_text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    code_text = "\n".join(c.source for c in py_cells)
    # 1. py_compile de celdas Python (saltar magics/shell)
    for i, c in enumerate(py_cells):
        src = c.source.strip()
        if not src:
            continue
        first = src.splitlines()[0].strip()
        if first.startswith("%") or first.startswith("!"):
            continue
        try:
            compile(src, f"{nb_path.name}:celda{i}", "exec")
        except SyntaxError as e:
            errors.append(f"{nb_path.name} celda {i}: SyntaxError: {e}")
    # 2. Ollama presente donde toca
    if nb_path.name in ("00_ollama_setup.ipynb", "04_memo_evaluacion.ipynb",
                        "05_rag_baseline.ipynb", "06_rag_hibrido_y_expansion.ipynb",
                        "07_rag_agentico.ipynb", "08_ragas.ipynb"):
        check("ollama" in code_text.lower() and "11434" in code_text,
              f"{nb_path.name}: falta el bloque Ollama (servidor 11434)")
        check("ollama show" in code_text or "ollama pull" in code_text,
              f"{nb_path.name}: falta el pull idempotente de modelos")
    if nb_path.name in ("01_setup_y_extraccion.ipynb", "02_grafo_relate.ipynb",
                        "03_memo_entrenamiento.ipynb"):
        check("ollama serve" not in code_text,
              f"{nb_path.name}: no deberia arrancar Ollama (no lo usa)")
    # 3. Sin pasos de maquina local
    check("OLLAMA.md" not in md_text and "OLLAMA.md" not in code_text,
          f"{nb_path.name}: referencia la guia local OLLAMA.md (fuera de alcance)")
    check("tu máquina" not in md_text and "tu maquina" not in md_text.lower(),
          f"{nb_path.name}: menciona la maquina local")
    # 4. Sin rutas absolutas locales del VM de desarrollo
    check("/home/hatch" not in code_text and "/home/hatch" not in md_text,
          f"{nb_path.name}: contiene ruta /home/hatch")
    print(f"{nb_path.name}: {len(nb.cells)} celdas, "
          f"{len(py_cells)} de codigo, OK" if not errors else
          f"{nb_path.name}: revisado")

# 5. Archivos del repo referenciados existen (solo ENTRADAS: los scripts y
# datos que deben pre-existir; las salidas que generan los notebooks — trazas,
# resultados, v3, adaptador, eval expandido — no existen aun por definicion).
OUTPUT_PATTERNS = ("_colab", "reflection_qa_v3", "eval_questions_expanded",
                   "memory_model_colab")
referenced = set()
for nb_path in NOTEBOOKS:
    nb = nbf.read(nb_path, as_version=4)
    text = "\n".join(c.source for c in nb.cells)
    for m in re.finditer(r'"([a-z_]+/[\w./]+\.(?:py|json|jsonl|cypher|md))"', text):
        referenced.add(m.group(1))
    for m in re.finditer(r"REPO \+ \"/([^\"]+)\"", text):
        referenced.add(m.group(1))
referenced = {p for p in referenced
              if p != ".git" and not any(o in p for o in OUTPUT_PATTERNS)}
missing = sorted(p for p in referenced if not (GOAL / p).exists())
check(not missing, f"archivos referenciados que no existen: {missing}")

# 6. Modelos Ollama mencionados son tags validos segun el encargo
for nb_path in NOTEBOOKS:
    nb = nbf.read(nb_path, as_version=4)
    text = "\n".join(c.source for c in nb.cells)
    for m in re.finditer(r"qwen2\.5:([0-9.]+b)(?![0-9.\w])", text):
        if m.group(1) not in ("1.5b", "7b", "14b"):
            warnings.append(f"{nb_path.name}: tag Ollama no esperado: qwen2.5:{m.group(1)}")

print()
print("archivos referenciados:", len(referenced))
if warnings:
    print("AVISOS:")
    for w in warnings:
        print(" -", w)
if errors:
    print("ERRORES:")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print("VALIDACION OK: nbformat + py_compile + rutas + Ollama + sin pasos locales")
