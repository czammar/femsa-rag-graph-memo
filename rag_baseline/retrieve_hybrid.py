#!/usr/bin/env python3
"""RAG baseline (FEMSA 2024) — Paso 1b: retrieval HIBRIDO (BM25 + denso).

El baseline denso puro (retrieve.py) quedo en 2/12 con 4 fallos de
retrieval (Q3, Q4, Q10, Q12) donde la cifra existe en el informe pero el
top-5 denso no la trajo. BM25 deberia ayudar justo ahi: son tokens exactos
("571", "FEMSA Forward", "Health", "23,206") que el embedder semantico
diluye en paginas largas.

Metodo (el reportado; no hay otro):
- BM25 Okapi (rank_bm25, k1=1.5, b=0.75, valores por defecto de la
  libreria) sobre los mismos 151 page-chunks.
- Denso: mismo all-MiniLM-L6-v2, misma codificacion que retrieve.py.
- Fusion: Reciprocal Rank Fusion (RRF), k=60 estandar:
  RRF(d) = 1/(60 + rank_bm25(d)) + 1/(60 + rank_denso(d)), ranks 1-based
  sobre los 151 chunks. Se eligio RRF (no suma ponderada) porque no
  requiere normalizar ni calibrar escalas dispares (BM25 ~0-30, coseno
  ~0-1) ni un peso arbitrario.
- Tokenizacion BM25: minusculas, regex [a-z0-9a-z...] con acentos
  ([a-z0-9áéíóúüñ]+), misma para pregunta y documentos, sin stopwords
  (BM25 ya penaliza por IDF; quitar stopwords en espanol no ayuda a un
  corpus en ingles y borraria tokens como "spin"). Las preguntas estan en
  espanol y el corpus en ingles: BM25 solo empareja tokens exactos
  (numeros, nombres propios, "FEMSA", "OXXO", "Spin"); la parte
  semantica cross-lingual la aporta el denso. Eso es una limitacion
  medida del hibrido, no un detalle menor.

Uso:
  memo/.venv/bin/python rag_baseline/retrieve_hybrid.py [--k 5]
      [--out rag_baseline/retrieval_hybrid.json]

Salida: rag_baseline/retrieval_hybrid.json con el mismo formato que
retrieval.json (top-k por pregunta: pagina, score=RRF, texto) mas, por
chunk recuperado, bm25_rank y dense_rank para auditoria, y por pregunta
los top-10 de cada ranker.
"""
import argparse
import json
import re
import time

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS = "pipeline/data/chunks.jsonl"
EVAL = "memo/eval_questions.json"
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
    ap.add_argument("--out", default="rag_baseline/retrieval_hybrid.json")
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
    results = []
    for q in eval_qs:
        q_tokens = tokenize(q["question"])
        b_scores = np.asarray(bm25.get_scores(q_tokens), dtype=np.float64)
        q_emb = model.encode([q["question"]], normalize_embeddings=True,
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
        # top-10 de cada ranker, para auditoria
        b_top10 = [int(chunks[i]["page"]) for i in np.argsort(-b_scores)[:10]]
        d_top10 = [int(chunks[i]["page"]) for i in np.argsort(-sims)[:10]]
        results.append({"id": q["id"], "level": q["level"],
                        "question": q["question"],
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

    out = {"method": "hybrid BM25Okapi (k1=1.5, b=0.75) + MiniLM dense, RRF k=60",
           "embed_model": EMBED_MODEL, "k": args.k, "n_chunks": len(chunks),
           "retrieval_seconds": round(time.time() - t0, 1),
           "questions": results}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    n_all = sum(1 for r in results if r["all_covered"])
    n_none = sum(1 for r in results if r["all_covered"] is False
                 and not any(r["coverage"].values()))
    print(f"retrieval hibrido: {n_all}/12 preguntas con TODAS las cifras "
          f"esperadas en top-{args.k}; {n_none}/12 sin ninguna -> {args.out}")


if __name__ == "__main__":
    main()
