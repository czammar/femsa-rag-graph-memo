#!/usr/bin/env python3
"""RAG baseline (FEMSA 2024) — Paso 1: retrieval.

Embeds the 151 pages of pipeline/data/chunks.jsonl with
sentence-transformers/all-MiniLM-L6-v2 (CPU) and retrieves top-k chunks per
eval question. Also runs a retrieval-only diagnostic: does the union of the
retrieved chunks contain the expected figures? This separates retrieval
failures from reader failures, exactly like the MEMO diagnosis did.

Uso:
  memo/.venv/bin/python rag_baseline/retrieve.py [--k 5] [--out rag_baseline/retrieval.json]

Salida: rag_baseline/retrieval.json con, por pregunta, los k chunks
(pagina, score, texto) y el diagnostico de cobertura de cifras esperadas.
"""
import argparse
import json
import re
import time

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS = "pipeline/data/chunks.jsonl"
EVAL = "memo/eval_questions.json"


def digits(s):
    return re.sub(r"[.,]", "", s)


def numbers_in(text):
    return {digits(m.group(0)) for m in re.finditer(r"\d[\d.,]*", text or "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunks", default=CHUNKS)
    ap.add_argument("--eval", default=EVAL)
    ap.add_argument("--out", default="rag_baseline/retrieval.json")
    args = ap.parse_args()

    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    chunks = [json.loads(l) for l in open(args.chunks)]
    texts = [c["text"] for c in chunks]
    print(f"embedding {len(chunks)} chunks with {EMBED_MODEL} ...", flush=True)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    print(f"embeddings: {emb.shape} in {time.time() - t0:.1f}s", flush=True)

    eval_qs = json.load(open(args.eval))["questions"]
    results = []
    for q in eval_qs:
        q_emb = model.encode([q["question"]], normalize_embeddings=True,
                             show_progress_bar=False)[0]
        sims = emb @ q_emb
        top = np.argsort(-sims)[: args.k]
        retrieved = []
        for idx in top:
            c = chunks[int(idx)]
            retrieved.append({"page": c["page"], "score": round(float(sims[idx]), 4),
                              "text": c["text"]})
        union_nums = set()
        for r in retrieved:
            union_nums |= numbers_in(r["text"])
        # diagnostico: cada cifra esperada aparece en algun chunk recuperado?
        coverage = {}
        for e in q.get("expected_numbers", []):
            d = digits(str(e))
            hit = d in union_nums
            if not hit:  # tolerancia de redondeo igual que en memo/evaluate.py
                try:
                    v = float(e)
                    hit = any(abs(float(a) - v) <= max(1.0, abs(v) * 0.001)
                              for a in union_nums if a.replace(".", "", 1).isdigit())
                except ValueError:
                    hit = False
            coverage[e] = bool(hit)
        results.append({"id": q["id"], "level": q["level"],
                        "question": q["question"],
                        "expected": q["expected"],
                        "expected_numbers": q.get("expected_numbers", []),
                        "retrieved": retrieved,
                        "coverage": coverage,
                        "all_covered": all(coverage.values()) if coverage else None})
        cov = "ALL" if results[-1]["all_covered"] else (
            "PARTIAL" if any(coverage.values()) else "NONE")
        print(f"{q['id']} L{q['level']} coverage={cov} "
              f"pages={[r['page'] for r in retrieved]}", flush=True)

    out = {"embed_model": EMBED_MODEL, "k": args.k, "n_chunks": len(chunks),
           "retrieval_seconds": round(time.time() - t0, 1),
           "questions": results}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    n_all = sum(1 for r in results if r["all_covered"])
    n_none = sum(1 for r in results if r["all_covered"] is False
                 and not any(r["coverage"].values()))
    print(f"retrieval: {n_all}/12 preguntas con TODAS las cifras esperadas "
          f"en top-{args.k}; {n_none}/12 sin ninguna -> {args.out}")


if __name__ == "__main__":
    main()
