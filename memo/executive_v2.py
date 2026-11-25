#!/usr/bin/env python3
"""MEMO inference protocol (demo scale) — RUN 2, FEMSA 2024.

v2 changes vs executive.py (run 1):
  * EXECUTIVE and MEMORY are separate models. EXECUTIVE defaults to the local
    Qwen2.5-1.5B-Instruct (larger than the 0.5B base that could not follow the
    few-shot decomposition format in run 1); MEMORY is Qwen2.5-0.5B-Instruct +
    the LoRA adapter trained in run 2 (memo/memory_model_v2). Both run on CPU.
  * The decomposition method per question is recorded in the trace
    ("decomposition": "llm" | "rule"), so we can report honestly how many of
    the 12 questions the EXECUTIVE decomposed itself.
  * --exec-model / --memory-model / --adapter are configurable; --control runs
    MEMORY as the untrained 0.5B base.
  * The EXECUTIVE never sees the corpus text or the expected answers; it only
    sees MEMORY's sub-answers.

Usage:
  memo/.venv/bin/python memo/executive_v2.py --all --out memo/run_trace_v2.jsonl
  memo/.venv/bin/python memo/executive_v2.py --all --control --out memo/run_trace_v2_control.jsonl
"""
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

EXEC_BASE = "memo/models/Qwen2.5-1.5B-Instruct"
MEM_BASE = "memo/models/Qwen2.5-0.5B-Instruct"
ADAPTER = "memo/memory_model_v2"
MAX_NEW = 80

DECOMP_SYSTEM = (
    "Descompón la pregunta del usuario en 1 a 3 sub-preguntas simples, cada una "
    "sobre un solo hecho del informe anual de FEMSA 2024. Responde solo con las "
    "sub-preguntas, una por línea, sin numerar. Crea sub-preguntas nuevas sobre "
    "la pregunta del usuario; NO copies las preguntas de los ejemplos.")
DECOMP_EXAMPLES = (
    "Pregunta: ¿Qué división aportó más ingresos en 2024, Proximidad Américas o Coca-Cola FEMSA, y por cuánto?\n"
    "¿Cuáles fueron los ingresos de Proximidad Américas en 2024?\n"
    "¿Cuáles fueron los ingresos de Coca-Cola FEMSA en 2024?\n\n"
    "Pregunta: ¿Qué porcentaje de los ingresos consolidados 2024 representan Proximidad Américas y Coca-Cola FEMSA juntas?\n"
    "¿Cuáles fueron los ingresos totales consolidados de FEMSA en 2024?\n"
    "¿Cuáles fueron los ingresos de Proximidad Américas en 2024?\n"
    "¿Cuáles fueron los ingresos de Coca-Cola FEMSA en 2024?\n\n")


def chat(tok, model, system, user, max_new=MAX_NEW):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def rule_decompose(q):
    """Descomposicion por reglas (fallback documentado)."""
    low = q.lower()
    divs = ["Proximidad Américas", "Proximidad Europa", "Coca-Cola FEMSA",
            "Salud", "Combustibles", "Spin"]
    named = [d for d in divs if d.lower() in low]
    if len(named) >= 2 and ("ingresos" in low or "creci" in low or "aport" in low):
        metric = "ingresos" if "ingresos" in low else "crecimiento"
        return [f"¿Cuáles fueron los {metric} de {d} en 2024?" for d in named]
    if "porcentaje" in low and "juntas" in low:
        return ["¿Cuáles fueron los ingresos totales consolidados de FEMSA en 2024?",
                "¿Cuáles fueron los ingresos de Proximidad Américas en 2024?",
                "¿Cuáles fueron los ingresos de Coca-Cola FEMSA en 2024?"]
    if "creció más" in low or "crecio mas" in low:
        return [q]  # la comparacion la resuelve el EXECUTIVE con las dos cifras
    return [q]


def llm_decompose(tok, model, q):
    out = chat(tok, model, DECOMP_SYSTEM, DECOMP_EXAMPLES + f"Pregunta: {q}\n")
    lines = [l.strip().lstrip("0123456789.-) ").strip()
             for l in out.splitlines() if l.strip().endswith("?")]
    lines = [l for l in lines if len(l) > 10][:3]
    return lines or None


def memory_answer(tok, model, subq):
    system = ("Responde con el dato exacto del informe anual de FEMSA 2024, "
              "en una frase corta. Si no sabes el dato, di 'No lo sé'.")
    return chat(tok, model, system, subq, max_new=60)


def synthesize(tok, model, q, sub_answers):
    context = "\n".join(f"- {s}: {a}" for s, a in sub_answers)
    system = ("Responde la pregunta usando SOLO la información siguiente, "
              "que viene del modelo de memoria. No inventes cifras. "
              "Si la información no alcanza, dilo.")
    user = f"Información del modelo de memoria:\n{context}\n\nPregunta: {q}"
    return chat(tok, model, system, user, max_new=120)


class MemoPipelineV2:
    def __init__(self, exec_model=EXEC_BASE, memory_model=MEM_BASE,
                 adapter_dir=ADAPTER, control=False):
        self.control = control
        self.exec_tok = AutoTokenizer.from_pretrained(exec_model)
        print(f"loading EXECUTIVE {exec_model} (bf16, CPU)...", flush=True)
        self.exec_model = AutoModelForCausalLM.from_pretrained(
            exec_model, torch_dtype=torch.bfloat16)
        self.exec_model.eval()
        self.mem_tok = AutoTokenizer.from_pretrained(memory_model)
        print(f"loading MEMORY {memory_model} (bf16, CPU)...", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            memory_model, torch_dtype=torch.bfloat16)
        if not control:
            print(f"loading adapter {adapter_dir} ...", flush=True)
            base = PeftModel.from_pretrained(base, adapter_dir)
        base.eval()
        self.mem_model = base

    def run(self, q):
        subs = llm_decompose(self.exec_tok, self.exec_model, q)
        method = "llm"
        if not subs:
            subs = rule_decompose(q)
            method = "rule"
        sub_answers = [(s, memory_answer(self.mem_tok, self.mem_model, s))
                       for s in subs]
        final = synthesize(self.exec_tok, self.exec_model, q, sub_answers)
        return {"question": q, "decomposition": method,
                "exec_model": EXEC_BASE, "memory_model": MEM_BASE,
                "sub_questions": subs,
                "sub_answers": [{"sub_question": s, "memory_answer": a}
                                for s, a in sub_answers],
                "final_answer": final,
                "control": self.control}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--eval", default="memo/eval_questions.json")
    ap.add_argument("--exec-model", default=EXEC_BASE)
    ap.add_argument("--memory-model", default=MEM_BASE)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--control", action="store_true",
                    help="usa la base sin entrenar como MEMORY (control)")
    ap.add_argument("--out", default="memo/run_trace_v2.jsonl")
    args = ap.parse_args()

    pipe = MemoPipelineV2(args.exec_model, args.memory_model,
                          args.adapter, control=args.control)
    if args.all:
        eval_qs = json.load(open(args.eval))["questions"]
        with open(args.out, "w") as f:
            for item in eval_qs:
                r = pipe.run(item["question"])
                r["id"] = item["id"]
                r["level"] = item["level"]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"{item['id']} [{r['decomposition']}] -> {r['final_answer'][:120]}", flush=True)
        print(f"trace -> {args.out}")
    elif args.question:
        print(json.dumps(pipe.run(args.question), ensure_ascii=False, indent=2))
    else:
        ap.error("usa --question o --all")


if __name__ == "__main__":
    main()
