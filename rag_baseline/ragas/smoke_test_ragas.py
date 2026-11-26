#!/usr/bin/env python3
"""Smoke test: real `ragas` 0.4.4 with a LOCAL judge (Qwen2.5-1.5B-Instruct,
CPU) on ONE question (dense Q1) to measure viability.

Q1 dense is the interesting case: the reader answered 40,236 (+31%) with
781,585 in context — wrong by exactness, deterministic faithfulness 1.00.
If the LLM judge agrees with the deterministic number, the fallback loses
little; if it takes minutes per question or the judge output is incoherent,
the full 36-answer run is not viable on CPU.

Timed; prints per-metric values and elapsed time.
"""
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAGAS_DIR = BASE / "ragas"

t0 = time.time()
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

print("imports: %.1fs" % (time.time() - t0), flush=True)

tok = AutoTokenizer.from_pretrained(str(BASE.parent / "memo/models/Qwen2.5-1.5B-Instruct"))
model = AutoModelForCausalLM.from_pretrained(
    str(BASE.parent / "memo/models/Qwen2.5-1.5B-Instruct"), torch_dtype="bfloat16"
)
gen = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=512,
               temperature=0.0, do_sample=False, return_full_text=False)
llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=gen))
print("judge loaded: %.1fs" % (time.time() - t0), flush=True)

emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
print("embeddings loaded: %.1fs" % (time.time() - t0), flush=True)

trace = [json.loads(l) for l in open(BASE / "run_trace.jsonl")][0]  # dense Q1
gt = json.load(open(RAGAS_DIR / "ground_truth_evidence.json"))["questions"][0]
contexts = [c for c in trace["context"].split("\n\n") if c.strip()]
sample = SingleTurnSample(
    user_input=trace["question"],
    retrieved_contexts=contexts,
    response=trace["final_answer"],
    reference=gt["question"] + " Esperado: Ps. 781,585 millones, +11.2%.",
)
t1 = time.time()
result = evaluate(
    EvaluationDataset(samples=[sample]),
    metrics=[ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy()],
    llm=llm,
    embeddings=emb,
)
print("evaluate: %.1fs" % (time.time() - t1), flush=True)
print(result.to_pandas().to_string())
print("total: %.1fs" % (time.time() - t0))
