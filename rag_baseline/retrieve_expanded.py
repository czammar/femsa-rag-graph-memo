#!/usr/bin/env python3
"""RAG FEMSA 2024 — Paso 1c: retrieval HIBRIDO con queries expandidas.

Identico a retrieve_hybrid.py (BM25 Okapi + MiniLM, RRF k=60, mismos 151
page-chunks, mismo top-5) salvo en una cosa: la query que ven los dos
rankers es la expanded query de rag_baseline/expanded_queries.json
(traduccion al ingles + keywords de dominio), no la pregunta original en
espanol. La pregunta original se conserva en la salida como "question"
—es la que ve el reader y la que usa condense— y la expandida va en
"expanded_query". Asi denso-vs-hibrido-vs-expandido difieren SOLO en la
query de retrieval.

Uso:
  memo/.venv/bin/python rag_baseline/retrieve_expanded.py [--k 5]
      [--queries rag_baseline/expanded_queries.json]
      [--out rag_baseline/retrieval_expanded.json]

Salida: rag_baseline/retrieval_expanded.json, mismo formato que
retrieval_hybrid.json + campo "expanded_query" por pregunta.
"""
import argparse
import json
import re
import time

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS = "pipeline/data/chunks.jsonl"
EVAL = "memo/eval_questions.json"
QUERIES = "rag_baseline/expanded_queries.json"
RRF_K = 60

TOKEN_RE = re.compile(r"[a-z0-9áéíóúüñ]+")


def tokenize(text):
    return TOKEN_RE.findall((text or "").lower())


def digits(s):
    return re.sub(r"[.,]", "", s)


def numbers_in(text):
    return {digits(m.group(0)) for m in re.finditer(r"\d[\d.,]*", text or "")}


def rrf_ranks(bm25_scores, dense_sims):
    """Ranks 1-based (1 = mejor) de cada ranker sobre los 151 chunks."""
    bm25_rank = np.empty(len(bm25_scores), dtype=int)
    bm25_rank[np.argsort(-np.asarray(bm25_scores))] = np.arange(1, len(bm25_scores) + 1)
    dense_rank = np.empty(len(dense_sims), dtype=int)
    dense_rank[np.argsort(-np.asarray(dense_sims))] = np.arange(1, len(dense_sims) + 1)
    return bm25_rank, dense_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunks", default=CHUNKS)
    ap.add_argument("--eval", default=EVAL)
    ap.add_argument("--queries", default=QUERIES)
    ap.add_argument("--out", default="rag_baseline/retrieval_expanded.json")
    args = ap.parse_args()

    t0 = time.time()
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    chunks = [json.loads(l) for l in open(args.chunks)]
    texts = [c["text"] for c in chunks]
    print(f"BM25 Okapi (k1=1.5, b=0.75) + {EMBED_MODEL} sobre {len(chunks)} chunks ...",
          flush=True)
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    print(f"BM25 index listo en {time.time() - t0:.1f}s", flush=True)

    model = SentenceTransformer(EMBED_MODEL)
    emb = np.asarray(model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False), dtype=np.float32)
    print(f"embeddings: {emb.shape} en {time.time() - t0:.1f}s", flush=True)

    eval_qs = json.load(open(args.eval))["questions"]
    expanded = {q["id"]: q["expanded_query"]
                for q in json.load(open(args.queries))["questions"]}
    results = []
    for q in eval_qs:
        query = expanded[q["id"]]  # <-- la unica diferencia vs retrieve_hybrid.py
        q_tokens = tokenize(query)
        b_scores = np.asarray(bm25.get_scores(q_tokens), dtype=np.float64)
        q_emb = model.encode([query], normalize_embeddings=True,
                             show_progress_bar=False)[0]
        sims = emb @ q_emb
        b_rank, d_rank = rrf_ranks(b_scores, sims)
        rrf = 1.0 / (RRF_K + b_rank) + 1.0 / (RRF_K + d_rank)
        top = np.argsort(-rrf)[: args.k]
        retrieved = []
        for idx in top:
            c = chunks[int(idx)]
            retrieved.append({"page": c["page"],
                              "score": round(float(rrf[idx]), 6),
                              "bm25_rank": int(b_rank[idx]),
                              "dense_rank": int(d_rank[idx]),
                              "bm25_score": round(float(b_scores[idx]), 4),
                              "dense_score": round(float(sims[idx]), 4),
                              "text": c["text"]})
        union_nums = set()
        for r in retrieved:
            union_nums |= numbers_in(r["text"])
        coverage = {}
        for e in q.get("expected_numbers", []):
            d = digits(str(e))
            hit = d in union_nums
            if not hit:
                try:
                    v = float(e)
                    hit = any(abs(float(a) - v) <= max(1.0, abs(v) * 0.001)
                              for a in union_nums if a.replace(".", "", 1).isdigit())
                except ValueError:
                    hit = False
            coverage[e] = bool(hit)
        b_top10 = [int(chunks[i]["page"]) for i in np.argsort(-b_scores)[:10]]
        d_top10 = [int(chunks[i]["page"]) for i in np.argsort(-sims)[:10]]
        results.append({"id": q["id"], "level": q["level"],
                        "question": q["question"],
                        "expanded_query": query,
                        "expected": q["expected"],
                        "expected_numbers": q.get("expected_numbers", []),
                        "retrieved": retrieved,
                        "coverage": coverage,
                        "all_covered": all(coverage.values()) if coverage else None,
                        "bm25_top10_pages": b_top10,
                        "dense_top10_pages": d_top10})
        cov = "ALL" if results[-1]["all_covered"] else (
            "PARTIAL" if any(coverage.values()) else "NONE")
        print(f"{q['id']} L{q['level']} coverage={cov} "
              f"pages={[r['page'] for r in retrieved]} "
              f"(bm25_top10={b_top10[:5]} dense_top10={d_top10[:5]})", flush=True)

    out = {"method": "hybrid BM25Okapi (k1=1.5, b=0.75) + MiniLM dense, RRF k=60, "
                     "con queries expandidas (traduccion EN + keywords de dominio)",
           "embed_model": EMBED_MODEL, "k": args.k, "n_chunks": len(chunks),
           "retrieval_seconds": round(time.time() - t0, 1),
           "questions": results}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    n_all = sum(1 for r in results if r["all_covered"])
    n_none = sum(1 for r in results if r["all_covered"] is False
                 and not any(r["coverage"].values()))
    print(f"retrieval expandido: {n_all}/12 preguntas con TODAS las cifras "
          f"esperadas en top-{args.k}; {n_none}/12 sin ninguna -> {args.out}")


if __name__ == "__main__":
    main()
