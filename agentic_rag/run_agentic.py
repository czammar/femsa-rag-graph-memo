#!/usr/bin/env python3
"""Corre el RAG agéntico sobre las 12 preguntas y escribe la traza.

Reanudable: salta las preguntas cuyo id ya está en la traza de salida
(por si el proceso se interrumpe; cada pregunta tarda varios minutos
en CPU). Uso:
  agentic_rag/.venv/bin/python agentic_rag/run_agentic.py [--only Q4]
  agentic_rag/.venv/bin/python agentic_rag/run_agentic.py --llm ollama
      [--model qwen2.5:7b-instruct] [--host http://localhost:11434]

Backend local (por defecto): Qwen2.5-1.5B-Instruct en disco; la traza
va a run_trace_agentic.jsonl. Backend Ollama: el mismo agente contra
un modelo servido por Ollama; la traza va a
run_trace_agentic_ollama.jsonl y NUNCA toca la del 1.5B (ver
agentic_rag/OLLAMA.md).
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(ROOT / "agentic_rag"))

from agent import build_graph, configure_llm, run_question  # noqa: E402

EVAL = ROOT / "memo/eval_questions.json"
OUT_LOCAL = ROOT / "agentic_rag/run_trace_agentic.jsonl"
OUT_OLLAMA = ROOT / "agentic_rag/run_trace_agentic_ollama.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="solo esta pregunta, p. ej. Q4")
    ap.add_argument("--out", default=None,
                    help="traza de salida (por defecto: local -> "
                    "run_trace_agentic.jsonl, ollama -> "
                    "run_trace_agentic_ollama.jsonl)")
    ap.add_argument("--llm", default="local", choices=["local", "ollama"],
                    help="backend del LLM: local 1.5B (por defecto) u "
                    "Ollama (modelo mas grande, ver OLLAMA.md)")
    ap.add_argument("--model", default=None,
                    help="local: ruta del modelo en disco; ollama: nombre "
                    "del modelo en Ollama (por defecto qwen2.5:7b-instruct)")
    ap.add_argument("--host", default=None,
                    help="solo ollama: URL del servidor "
                    "(por defecto $OLLAMA_HOST o http://localhost:11434)")
    args = ap.parse_args()

    if args.out is None:
        args.out = str(OUT_OLLAMA if args.llm == "ollama" else OUT_LOCAL)

    configure_llm(args.llm, model=args.model, host=args.host, root=ROOT)

    done = set()
    if Path(args.out).exists():
        for line in open(args.out):
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    if done:
        print(f"[run] reanudando: {sorted(done)} ya en {args.out}", flush=True)

    qs = json.load(open(EVAL))["questions"]
    if args.only:
        qs = [q for q in qs if q["id"] == args.only]
    graph = build_graph()
    t0 = time.time()
    with open(args.out, "a") as f:
        for q in qs:
            if q["id"] in done:
                print(f"[run] {q['id']} ya hecha, salto", flush=True)
                continue
            print(f"[run] {q['id']} L{q['level']}: {q['question']}", flush=True)
            try:
                rec = run_question(q["question"], q["id"], q["level"], graph)
            except Exception as e:  # noqa: BLE001 - un fallo en una pregunta
                # no debe matar la corrida: se registra como error y se sigue.
                import traceback
                traceback.print_exc()
                rec = {"id": q["id"], "level": q["level"],
                       "question": q["question"],
                       "llm_backend": args.llm,
                       "final_answer": f"ERROR del agente: {type(e).__name__}: {e}",
                       "steps": [], "plan_raw_outputs": [],
                       "seconds": round(time.time() - t0, 1)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            tools = [s["tool"] for s in rec["steps"]]
            print(f"[run] {q['id']} -> {rec['final_answer'][:160]} "
                  f"(tools={tools}, {rec['seconds']}s)", flush=True)
    print(f"[run] listo en {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
