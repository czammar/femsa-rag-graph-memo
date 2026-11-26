#!/usr/bin/env python3
"""Corre el RAG agéntico sobre las 12 preguntas y escribe la traza.

Reanudable: salta las preguntas cuyo id ya está en la traza de salida
(por si el proceso se interrumpe; cada pregunta tarda varios minutos
en CPU). Uso:
  agentic_rag/.venv/bin/python agentic_rag/run_agentic.py [--only Q4]
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(ROOT / "agentic_rag"))

from agent import build_graph, run_question  # noqa: E402

EVAL = ROOT / "memo/eval_questions.json"
OUT = ROOT / "agentic_rag/run_trace_agentic.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="solo esta pregunta, p. ej. Q4")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

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
