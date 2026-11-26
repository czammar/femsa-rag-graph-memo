#!/usr/bin/env python3
"""answer_relevancy (proxy): cosine similarity between all-MiniLM-L6-v2
embeddings of the question and the generated answer.
RAGAS defines answer_relevancy via LLM-generated questions from the answer;
that generation step is NOT used here — this is an embedding proxy, labelled
as such in metrics_ragas.json and results_ragas.md.
Runs in memo/.venv (sentence-transformers already installed there)."""
import json, time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent.parent
RAGAS_DIR = BASE / "ragas"
TRACES = {"dense": "run_trace.jsonl", "hybrid": "run_trace_hybrid.jsonl", "expanded": "run_trace_expanded.jsonl"}

t0 = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
metrics = json.load(open(RAGAS_DIR / "metrics_ragas.json"))
for name, trace_file in TRACES.items():
    trace = [json.loads(l) for l in open(BASE / trace_file)]
    qs = [t["question"] for t in trace]
    ans = [t["final_answer"] for t in trace]
    eq = model.encode(qs, normalize_embeddings=True, show_progress_bar=False)
    ea = model.encode(ans, normalize_embeddings=True, show_progress_bar=False)
    for t, a, b in zip(trace, eq, ea):
        metrics["variants"][name]["per_question"][t["id"]]["answer_relevancy"] = round(float(np.dot(a, b)), 4)
    print(name, "done")
metrics["answer_relevancy_method"] = "proxy: cosine(all-MiniLM-L6-v2(question), all-MiniLM-L6-v2(answer)); RAGAS defines it via LLM-generated questions, not used here"
json.dump(metrics, open(RAGAS_DIR / "metrics_ragas.json", "w"), ensure_ascii=False, indent=1)
print("relevancy computed in %.1fs" % (time.time() - t0))
