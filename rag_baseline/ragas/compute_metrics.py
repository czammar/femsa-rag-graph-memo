#!/usr/bin/env python3
"""RAGAS-style metrics for the three FEMSA RAG variants, computed DETERMINISTICALLY.

Why deterministic (no LLM judge): the real `ragas` library computes
context_precision / context_recall / faithfulness with an LLM judge. On this
VM the only judge available is a local Qwen2.5-1.5B on CPU; a full ragas run
needs ~8-10 judge calls per answer x 36 answers (see ragas/README.md for the
timed smoke test that justified the fallback). Instead, the four metrics are
computed with the exact definitions below, from the retrieval traces, the
run traces (answers + the condensed context actually shown to the reader)
and the auditable ground-truth page set in ground_truth_evidence.json.

Definitions (standard RAGAS, adapted to page-level chunks):
- context_precision = |retrieved pages that are ground-truth evidence pages|
  / |retrieved pages| (top-5). A page counts as relevant iff it is in the
  ground-truth page set for that question.
- context_recall = |ground-truth evidence items with >=1 page in the top-5|
  / |ground-truth evidence items|. Q9 has no evidence item in the corpus
  (its figures are computed shares) -> recall is null, not 0.
- faithfulness = |claims in the answer supported by the retrieved context|
  / |claims in the answer|. Claims = numeric claims + entity claims
  (division / initiative / country names). A numeric claim is supported iff
  its normalized value appears in the condensed context (page headers
  "[Página N]" stripped first, so a leaked page number does not count as
  supported by the header). An entity claim is supported iff the entity
  string appears in the context (accent/case-insensitive). Abstention
  ("No lo sé") has 0 claims -> faithfulness 1.0 by convention, flagged.
  LIMIT: this cannot catch misattribution (a real number used for the wrong
  division, e.g. hybrid Q6's 128,736 gross profit quoted as revenue) — an
  LLM judge would. Deterministic faithfulness is therefore an UPPER BOUND.
- answer_relevancy = cosine similarity between all-MiniLM-L6-v2 embeddings
  of the question and the answer. PROXY: RAGAS defines it via LLM-generated
  questions from the answer; that generation step is not used here.

Outputs: metrics_ragas.json (per question, per variant, per metric, with
per-claim faithfulness detail and exactness verdicts for agreement analysis).
"""
import json
import re
import unicodedata
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # rag_baseline/
RAGAS_DIR = BASE / "ragas"
GT = json.load(open(RAGAS_DIR / "ground_truth_evidence.json"))
GT_BY_ID = {q["id"]: q for q in GT["questions"]}

VARIANTS = {
    "dense": {"retrieval": "retrieval.json", "trace": "run_trace.jsonl"},
    "hybrid": {"retrieval": "retrieval_hybrid.json", "trace": "run_trace_hybrid.jsonl"},
    "expanded": {"retrieval": "retrieval_expanded.json", "trace": "run_trace_expanded.jsonl"},
}

# Exactness verdicts from the published results files (same scorer, memo/evaluate.py).
EXACTNESS = {
    "dense": {"Q6", "Q11"},
    "hybrid": {"Q1", "Q11"},
    # Q11 expanded is "correcta con alucinacion" in results_expanded.md: the
    # substance is right (Digital@FEMSA -> Spin) but the scorer flags '328';
    # the published count is 5/12, so Q11 is NOT in the correct set here.
    "expanded": {"Q1", "Q4", "Q6", "Q8", "Q10"},
}
EXACTNESS_NOTES = {"expanded": {"Q11": "correcta con alucinacion segun el scorer (artefacto: '328' = 32.8% real, p.34)"}}


def norm_text(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # collapse all whitespace (PDF text has tabs/newlines inside names:
    # "Coca-Cola\tFEMSA" must match "coca-cola femsa")
    return re.sub(r"\s+", " ", s).strip().lower()


NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?")


def norm_number(tok):
    """'781,585' -> '781585'; '11,2' -> '11.2'; '8.6' -> '8.6'; '571' -> '571'."""
    tok = tok.strip().rstrip("%")
    if "," in tok and "." not in tok:
        parts = tok.split(",")
        if len(parts) == 2 and len(parts[1]) == 1:
            return parts[0] + "." + parts[1]  # decimal comma (Spanish)
        return "".join(parts)  # thousands separator
    return tok.replace(",", "")


def extract_numbers(text):
    return [norm_number(m.group(0)) for m in NUM_RE.finditer(text)]


ENTITIES = {
    # canonical -> accepted surface forms (Spanish answer vs English corpus:
    # a claim is supported if ANY form appears in the context)
    "FEMSA Forward": ["femsa forward"],
    "Digital@FEMSA": ["digital@femsa"],
    "Spin by OXXO": ["spin by oxxo"],
    "Spin Premia": ["spin premia"],
    "Proximidad Americas": ["proximidad americas", "proximity americas"],
    "Coca-Cola FEMSA": ["coca-cola femsa"],
    "Proximidad Europa": ["proximidad europa", "proximity europe"],
    "OXXO GAS": ["oxxo gas"],
    "Salud": ["salud", "health"],
    "Spin": ["spin"],
    "Mexico": ["mexico"],
    "Colombia": ["colombia"],
    "Chile": ["chile"],
    "Peru": ["peru"],
    "Argentina": ["argentina"],
    "Austria": ["austria"],
    "Brasil": ["brasil", "brazil"],
    "Costa Rica": ["costa rica"],
    "Ecuador": ["ecuador"],
    "Espana": ["espana", "spain"],
    "Guatemala": ["guatemala"],
    "Honduras": ["honduras"],
    "Paises Bajos": ["paises bajos", "netherlands"],
    "Nicaragua": ["nicaragua"],
    "Panama": ["panama"],
    "Uruguay": ["uruguay"],
    "Estados Unidos": ["estados unidos", "united states"],
}


def extract_entities(text):
    nt = norm_text(text)
    found = []
    for canonical, forms in ENTITIES.items():
        if any(f in nt for f in forms):
            found.append(canonical)
    # drop "Spin" if a longer Spin entity is present
    if any(e in found for e in ("Spin by OXXO", "Spin Premia")) and "Spin" in found:
        found.remove("Spin")
    return found


def context_numbers(context):
    body = re.sub(r"\[P[aá]gina \d+\]", " ", context)  # strip page headers
    return set(extract_numbers(body))


def context_entities(context):
    return set(extract_entities(context))


def load_variant(name):
    cfg = VARIANTS[name]
    ret = json.load(open(BASE / cfg["retrieval"]))
    pages = {q["id"]: [c["page"] for c in q["retrieved"]] for q in ret["questions"]}
    trace = [json.loads(l) for l in open(BASE / cfg["trace"])]
    answers = {t["id"]: t for t in trace}
    return pages, answers


def main():
    out = {"variants": {}, "method": "deterministic (see ragas/README.md)", "judge": None}
    for name in VARIANTS:
        pages_by_q, answers = load_variant(name)
        per_q = {}
        for qid, gt in GT_BY_ID.items():
            retrieved = pages_by_q[qid]
            gt_pages = set()
            for item in gt["evidence"]:
                gt_pages.update(item["pages"])
            # context precision
            relevant = [p for p in retrieved if p in gt_pages]
            ctx_precision = len(relevant) / len(retrieved) if retrieved else None
            # context recall (item level)
            n_items = len(gt["evidence"])
            if n_items == 0:
                ctx_recall = None
                recall_detail = {"covered": 0, "total": 0, "note": "no hay evidencia en el corpus (cifras calculadas, no impresas)"}
            else:
                covered = sum(1 for item in gt["evidence"] if any(p in retrieved for p in item["pages"]))
                ctx_recall = covered / n_items
                recall_detail = {"covered": covered, "total": n_items}
            # faithfulness
            ans = answers[qid]["final_answer"]
            ctx = answers[qid]["context"]
            abstain = ans.strip().rstrip(".").lower().startswith("no lo s")
            ctx_nums = context_numbers(ctx)
            ctx_ents = context_entities(ctx)
            claims = []
            if abstain:
                faithfulness = 1.0
                claims = []
            else:
                for n in extract_numbers(ans):
                    claims.append({"claim": n, "type": "numeric", "supported": n in ctx_nums})
                for e in extract_entities(ans):
                    claims.append({"claim": e, "type": "entity", "supported": e in ctx_ents})
                faithfulness = (sum(1 for c in claims if c["supported"]) / len(claims)) if claims else 1.0
            per_q[qid] = {
                "question": gt["question"],
                "retrieved_pages": retrieved,
                "ground_truth_pages": sorted(gt_pages),
                "context_precision": ctx_precision,
                "context_recall": ctx_recall,
                "recall_detail": recall_detail,
                "faithfulness": faithfulness,
                "n_claims": len(claims),
                "abstention": abstain,
                "claims": claims,
                "exactness_correct": qid in EXACTNESS[name],
                "exactness_note": EXACTNESS_NOTES.get(name, {}).get(qid, ""),
            }
        # answer relevancy (embedding cosine) — computed separately if sentence-transformers available
        out["variants"][name] = {"per_question": per_q}
    json.dump(out, open(RAGAS_DIR / "metrics_ragas.json", "w"), ensure_ascii=False, indent=1)
    print("wrote metrics_ragas.json (precision/recall/faithfulness; relevancy pending)")


if __name__ == "__main__":
    main()
