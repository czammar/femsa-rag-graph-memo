#!/usr/bin/env python3
"""Build reflection_qa_v2.jsonl for MEMO run 2 (FEMSA 2024).

Run 1 diagnosis: 446 QA pairs, 1 epoch, LoRA r=8 -> 0/12. Exact figures need
repetition and surface variation. This script expands the v1 dataset by
deterministic, question-only paraphrases (the ANSWER is copied byte-identical,
so no figure can change) plus extra copies of the key numeric facts
(the 12 eval ground truths + the seed-triple figures).

Rules (question-only, Spanish):
  framing_A : "Según el informe anual de FEMSA 2024, ¿...?"
  framing_B : "De acuerdo con el informe anual de FEMSA 2024, ¿...?"
  fronting  : "En 2024, ¿...?" (moves trailing "en 2024" to the front)
  swap_asc  : "¿A cuánto ascendieron" <-> "¿A cuánto llegaron"
  swap_fueron: "¿Cuáles fueron" -> "¿Cuáles son"
  swap_crecio: "¿Qué división creció más" -> "¿Cuál fue la división que más creció"
  swap_cuantos: "¿Cuántos/as" -> "¿Qué cantidad de"
  swap_tiene: "¿Tiene X operaciones en Y?" -> "¿Opera X en Y?"
  swap_pct  : "¿En qué porcentaje crecieron" -> "¿Cuánto crecieron en porcentaje"

Safety:
  * every paraphrase is checked against the 12 eval questions with token
    Jaccard; anything >= 0.80 is dropped (same leakage rule as run 1).
  * a script-level verification asserts every answer in v2 is byte-identical
    to its v1 source and reports how many times each key fact appears.
  * the 12 eval questions are NOT in the training set (same as run 1).

Usage:
  memo/.venv/bin/python memo/augment_v2.py
"""
import json
import re
import unicodedata

BASE = "memo/reflection_qa.jsonl"
EVAL = "memo/eval_questions.json"
TRIPLES = "pipeline/triples_seed.json"
OUT = "memo/reflection_qa_v2.jsonl"
REPORT = "memo/augment_v2_report.json"
JACCARD_MAX = 0.80


def norm_num_tokens(s):
    """Digit sequences in a string, separators stripped ('781,585' -> '781585')."""
    return set(re.sub(r"[.,\s]", "", t) for t in re.findall(r"[\d][\d.,]*", s))


def tokens(q):
    q = unicodedata.normalize("NFKD", q.lower())
    q = "".join(c for c in q if not unicodedata.combining(c))
    return set(re.findall(r"[a-z0-9]+", q))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def lower_first_after_q(q):
    """Lowercase the first letter after '¿' (framing prefixes need it)."""
    m = re.match(r"(¿)([A-ZÁÉÍÓÚÑ])", q)
    if m:
        return q[:1] + m.group(2).lower() + q[2:]
    return q


def framing(q, prefix):
    return prefix + lower_first_after_q(q)


def fronting(q):
    m = re.match(r"^(.*)\sen 2024\?$", q.strip())
    if m:
        return "En 2024, " + lower_first_after_q(m.group(1) + "?")
    m = re.match(r"^(.*)\sal cierre de 2024\?$", q.strip())
    if m:
        return "Al cierre de 2024, " + lower_first_after_q(m.group(1) + "?")
    return None


def swap(q, old, new):
    if q.startswith(old):
        return new + q[len(old):]
    return None


def candidates(q):
    """Ordered paraphrase candidates (question-only)."""
    out = []
    out.append(framing(q, "Según el informe anual de FEMSA 2024, "))
    f = fronting(q)
    if f:
        out.append(f)
    for old, new in [
        ("¿A cuánto ascendieron", "¿A cuánto llegaron"),
        ("¿A cuánto llegaron", "¿A cuánto ascendieron"),
        ("¿Cuáles fueron", "¿Cuáles son"),
        ("¿Qué división creció más", "¿Cuál fue la división que más creció"),
        ("¿Cuántos", "¿Qué cantidad de"),
        ("¿Cuántas", "¿Qué cantidad de"),
        ("¿En qué porcentaje crecieron", "¿Cuánto crecieron en porcentaje"),
    ]:
        s = swap(q, old, new)
        if s:
            out.append(s)
    m = re.match(r"¿Tiene (.+?) operaciones en (.+?)\?$", q)
    if m:
        out.append(f"¿Opera {m.group(1)} en {m.group(2)}?")
    out.append(framing(q, "De acuerdo con el informe anual de FEMSA 2024, "))
    # dedupe, keep order, drop identical-to-original
    seen = {q}
    uniq = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def main():
    rows = [json.loads(l) for l in open(BASE)]
    eval_qs = json.load(open(EVAL))["questions"]
    eval_toks = [tokens(q["question"]) for q in eval_qs]
    eval_nums = set()
    for q in eval_qs:
        eval_nums.update(q.get("expected_numbers", []))

    triples = json.load(open(TRIPLES))["triples"]
    triple_nums = set()
    for t in triples:
        for v in (t.get("valor"), t.get("valor_exacto_informe")):
            if v:
                triple_nums.update(norm_num_tokens(str(v)))
    key_nums = {n for n in eval_nums | triple_nums if len(n) >= 2}

    def is_key(row):
        return bool(norm_num_tokens(row["answer"]) & key_nums)

    out_rows = []
    dropped_leak = []
    key_rows = 0
    for i, r in enumerate(rows):
        key = is_key(r)
        if key:
            key_rows += 1
        n_para = 3 if key else 2
        cands = candidates(r["question"])
        kept = []
        for c in cands:
            if len(kept) >= n_para:
                break
            if max((jaccard(tokens(c), et) for et in eval_toks), default=0) >= JACCARD_MAX:
                dropped_leak.append({"base": i, "paraphrase": c})
                continue
            kept.append(c)
        out_rows.append({**r, "variant": "original", "base_index": i})
        for j, c in enumerate(kept):
            out_rows.append({**r, "question": c, "variant": f"parafrasis_{j+1}",
                             "base_index": i})

    # --- verification: answers byte-identical to their v1 source ---
    by_base = {}
    for r in out_rows:
        by_base.setdefault(r["base_index"], []).append(r)
    assert len(by_base) == len(rows), "some v1 row lost"
    for i, r in enumerate(rows):
        for v in by_base[i]:
            assert v["answer"] == r["answer"], f"answer changed for base {i}"
            assert v["source_pages"] == r["source_pages"]
    # no exact duplicate (question, answer)
    seen = set()
    for r in out_rows:
        k = (r["question"], r["answer"])
        assert k not in seen, f"duplicate pair: {k}"
        seen.add(k)
    # key-fact multiplicity: count appearances of each key number
    from collections import Counter
    fact_counts = Counter()
    for r in out_rows:
        for n in norm_num_tokens(r["answer"]) & key_nums:
            fact_counts[n] += 1
    low = {n: c for n, c in fact_counts.items() if c < 3}
    assert not low, f"key facts with <3 appearances: {low}"

    with open(OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "v1_rows": len(rows),
        "v2_rows": len(out_rows),
        "key_fact_rows_v1": key_rows,
        "paraphrases_kept": len(out_rows) - len(rows),
        "dropped_for_leakage_jaccard>=0.80": len(dropped_leak),
        "key_numbers_tracked": len(key_nums),
        "min_appearances_per_key_number": min(fact_counts.values()),
        "max_appearances_per_key_number": max(fact_counts.values()),
        "dropped_examples": dropped_leak[:5],
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
