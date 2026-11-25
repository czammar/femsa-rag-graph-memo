#!/usr/bin/env python3
"""Test whether a given base model follows the few-shot decomposition format.

Loads the model, runs llm_decompose (from executive_v2) on the 12 eval
questions, and reports per-question success. Used to decide the EXECUTIVE
for run 2: run 1's 0.5B base failed on all 12.
"""
import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "memo")
from executive_v2 import llm_decompose, rule_decompose  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--eval", default="memo/eval_questions.json")
ap.add_argument("--out", default="memo/decomp_test.json")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
print(f"loading {args.model} (bf16, CPU)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
model.eval()

qs = json.load(open(args.eval))["questions"]
results = []
for item in qs:
    subs = llm_decompose(tok, model, item["question"])
    ok = bool(subs)
    results.append({"id": item["id"], "question": item["question"],
                    "llm_decomposed": ok,
                    "sub_questions": subs or rule_decompose(item["question"]),
                    "method": "llm" if ok else "rule"})
    print(f"{item['id']}: {'LLM' if ok else 'rule'} -> {results[-1]['sub_questions']}",
          flush=True)

n_llm = sum(r["llm_decomposed"] for r in results)
print(f"\nLLM decomposition: {n_llm}/12")
with open(args.out, "w") as f:
    json.dump({"model": args.model, "llm_success": n_llm,
               "results": results}, f, indent=2, ensure_ascii=False)
