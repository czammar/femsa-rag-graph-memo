#!/usr/bin/env python3
"""RAG agentico (FEMSA 2024) -- juez de seleccion, pipeline completo.

Segundo diseno de RAG agentico (el que pidio el usuario): el sistema
SOLO califica los chunks del RAG hibrido y selecciona los utiles con un
juez. Sin graph_query, sin calculator. El agente ReAct anterior
(agent.py, 2/12) midio un agente solo-grafo y queda como resultado
negativo documentado; este archivo no lo toca.

Por pregunta:
  1. Retrieval hibrido top-10 (BM25 Okapi + MiniLM, RRF k=60) desde
     rag_baseline/retrieval_hybrid_k10.json (generado con
     retrieve_hybrid.py --k 10).
  2. UN llamado al juez con los 10 chunks numerados (ver
     judge_rerank.py): devuelve un veredicto por chunk (UTIL / NO UTIL)
     con su razon corta. Maximo 3 reintentos si los veredictos no se
     pueden parsear o el llamado falla; si a los 3 sigue sin veredictos
     validos, la pregunta se abstiene ("No lo se", sin contestar desde
     chunks sin juzgar).
  3. El lector responde usando UNICAMENTE los chunks seleccionados
     (misma logica que rag_baseline/answer.py: mismo lector Qwen2.5-1.5B,
     mismo condense, mismo SYSTEM, mismos parametros de generacion).
     Si el juez no selecciona ningun chunk, la respuesta es "No lo se"
     sin llamar al lector. El llamado del lector tambien tiene maximo
     3 reintentos si falla (excepcion o respuesta vacia); si no se logra
     respuesta, la respuesta es abstencion ("No lo se").

El juez ve la version condensada de cada chunk, la misma que vera el
lector: un chunk es "util" si lo que el lector va a ver de el sirve
para responder. La traza guarda los chunks presentados, los veredictos
por chunk, los reintentos de cada llamado y la respuesta final.

Uso:
  memo/.venv/bin/python agentic_rag/run_judge.py [--only Q1]
      [--time-test] [--retrieval rag_baseline/retrieval_hybrid_k10.json]
      [--out agentic_rag/run_trace_judge.jsonl]

  --time-test: corre solo Q1 de punta a punta (juez + lector) y reporta
  tiempos + extrapolacion a las 12 preguntas; no escribe la traza.
  La corrida es reanudable: salta las preguntas ya registradas en la traza.
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)   # judge_rerank.py vive en agentic_rag/
sys.path.insert(0, _ROOT)   # rag_baseline.answer vive en la raiz

import torch
from judge_rerank import Judge

from rag_baseline.answer import SYSTEM as READER_SYSTEM, condense  # noqa: E402

MODEL = "memo/models/Qwen2.5-1.5B-Instruct"
MAX_NEW_READER = 100
MAX_READER_ATTEMPTS = 3
ABSTAIN = "No lo sé."


def reader_call(model, tok, context, question):
    """Un llamado al lector. Devuelve (respuesta, error_o_None)."""
    user = (f"Fragmentos del informe anual de FEMSA 2024:\n\n{context}\n\n"
            f"Pregunta: {question}")
    msgs = [{"role": "system", "content": READER_SYSTEM},
            {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAX_NEW_READER,
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
    ans = tok.decode(out[0][ids["input_ids"].shape[1]:],
                     skip_special_tokens=True).strip()
    return ans


def run_question(judge, item):
    """Corre una pregunta completa. Devuelve el registro de la traza."""
    t0 = time.time()
    chunks = [(r["page"], condense(r["text"], item["question"]))
              for r in item["retrieved"]]
    presented = [{"n": i + 1, "page": p, "condensed": c}
                 for i, (p, c) in enumerate(chunks)]

    # 1 llamado al juez con los 10 chunks (con sus 3 reintentos internos)
    jres = judge.judge(item["question"], chunks)
    rec = {"id": item["id"], "level": item["level"],
           "question": item["question"],
           "decomposition": "judge-rerank",
           "reader_model": MODEL,
           "retrieved_pages": [r["page"] for r in item["retrieved"]],
           "retrieval_scores": [r["score"] for r in item["retrieved"]],
           "chunks_presented": presented,
           "judge": {"attempts": jres["attempts"],
                     "attempts_raw": jres["attempts_raw"],
                     "verdicts": jres["verdicts"],
                     "ok": jres["ok"],
                     "seconds": jres["seconds"]},
           "selected_pages": [],
           "reader": {"attempts": 0, "seconds": 0.0,
                      "failed": False, "skipped": False},
           "context": "",
           "final_answer": ABSTAIN,
           "seconds": 0.0}

    if not jres["ok"]:
        # Sin veredictos validos tras 3 intentos: abstencion, sin lector.
        rec["reader"]["skipped"] = True
        rec["seconds"] = round(time.time() - t0, 1)
        return rec

    selected = [i for i in range(1, len(chunks) + 1)
                if jres["verdicts"][i]["useful"]]
    rec["selected_pages"] = [chunks[i - 1][0] for i in selected]
    if not selected:
        # El juez no selecciono nada: abstencion honesta, sin lector.
        rec["reader"]["skipped"] = True
        rec["seconds"] = round(time.time() - t0, 1)
        return rec

    ctx = "\n\n".join(
        f"[Página {chunks[i - 1][0]}]\n{chunks[i - 1][1]}" for i in selected)
    rec["context"] = ctx

    # Lector con maximo 3 reintentos si el llamado falla.
    t_reader = time.time()
    answer, failed = None, False
    for attempt in range(1, MAX_READER_ATTEMPTS + 1):
        try:
            ans = reader_call(judge.model, judge.tok, ctx, item["question"])
        except Exception as e:
            ans = ""
            print(f"  {item['id']} reader intento {attempt} fallo: "
                  f"{type(e).__name__}: {e}", flush=True)
        if ans:
            answer = ans
            rec["reader"]["attempts"] = attempt
            break
        rec["reader"]["attempts"] = attempt
    else:
        failed = True
    rec["reader"]["seconds"] = round(time.time() - t_reader, 1)
    rec["reader"]["failed"] = failed
    if failed:
        rec["final_answer"] = ABSTAIN
    else:
        rec["final_answer"] = answer
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval",
                    default="rag_baseline/retrieval_hybrid_k10.json")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="agentic_rag/run_trace_judge.jsonl")
    ap.add_argument("--only", default=None)
    ap.add_argument("--time-test", action="store_true")
    args = ap.parse_args()

    data = json.load(open(args.retrieval))
    questions = data["questions"]
    if args.only:
        questions = [q for q in questions if q["id"] == args.only]
    if args.time_test:
        questions = [q for q in data["questions"] if q["id"] == "Q1"]

    t0 = time.time()
    judge = Judge.load(args.model)
    print(f"judge/reader model loaded in {time.time() - t0:.1f}s", flush=True)

    done = set()
    if not args.time_test and os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out)
                if l.strip()}
        print(f"resuming: {len(done)} preguntas ya en {args.out}", flush=True)

    total_judge, total_reader = 0.0, 0.0
    fout = open(args.out, "a") if not args.time_test else None
    try:
        for item in questions:
            if item["id"] in done:
                print(f"{item['id']} ya registrada, se salta", flush=True)
                continue
            rec = run_question(judge, item)
            total_judge += rec["judge"]["seconds"]
            total_reader += rec["reader"]["seconds"]
            n_sel = len(rec["selected_pages"])
            print(f"{item['id']} juez_ok={rec['judge']['ok']} "
                  f"intentos_juez={rec['judge']['attempts']} "
                  f"seleccionados={n_sel}/10 "
                  f"lector_intentos={rec['reader']['attempts']} "
                  f"-> {rec['final_answer'][:160]}", flush=True)
            if fout is not None:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
    finally:
        if fout is not None:
            fout.close()
    if args.time_test:
        n = len(questions)
        per_q = (total_judge + total_reader) / max(n, 1)
        print(f"\nTIME-TEST Q1: juez {total_judge:.1f}s + lector "
              f"{total_reader:.1f}s = {per_q:.1f}s/pregunta")
        print(f"Extrapolacion a 12 preguntas: ~{per_q * 12 / 60:.1f} min "
              f"(12 llamados al juez + 12 al lector).")
        print("No se escribio la traza (time-test).")
    else:
        print(f"trace -> {args.out} "
              f"(juez {total_judge:.1f}s + lector {total_reader:.1f}s, "
              f"total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
