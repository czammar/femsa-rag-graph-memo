#!/usr/bin/env python3
"""MEMO inference protocol (demo scale) sobre el informe FEMSA 2024.

Arquitectura del paper (arXiv 2605.15156):
  EXECUTIVE (congelado) -> descompone la pregunta en sub-preguntas dirigidas
  -> MEMORY (entrenado por SFT en reflexiones) responde cada sub-pregunta
  -> EXECUTIVE razona sobre las sub-respuestas y produce la respuesta final.
  El EXECUTIVE NUNCA ve el corpus.

Demo: aqui EXECUTIVE y MEMORY comparten los mismos pesos base
(Qwen2.5-0.5B-Instruct); MEMORY es base + adaptador LoRA (memo/memory_model),
EXECUTIVE es la base sin adaptador (peft disable_adapter). Es una simplificacion
documentada del demo: en el paper son dos modelos distintos. Si no hay adaptador
entrenado, se usa --control para correr con la base sin entrenar y se etiqueta
el resultado como CONTROL (modelo de memoria sin entrenar).

Descomposicion: el EXECUTIVE (LLM base) intenta descomponer con un prompt
few-shot; si la salida no se puede parsear, se usa descomposicion por reglas
y se registra el metodo usado por pregunta (campo "decomposition" en el trace).

Uso:
  .venv/bin/python memo/executive.py --question "¿Cuántas estaciones tenía OXXO GAS al cierre de 2024?"
  .venv/bin/python memo/executive.py --all --out memo/run_trace.jsonl
  .venv/bin/python memo/executive.py --all --control --out memo/run_trace_control.jsonl
"""
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "memo/models/Qwen2.5-0.5B-Instruct"  # copia local (ver memo/README.md:
# la descarga desde HF se atascaba; los pesos se bajaron con curl + resume)
ADAPTER = "memo/memory_model"
MAX_NEW = 80

DECOMP_SYSTEM = (
    "Descompón la pregunta del usuario en 1 a 3 sub-preguntas simples, cada una "
    "sobre un solo hecho del informe anual de FEMSA 2024. Responde solo con las "
    "sub-preguntas, una por línea, sin numerar.")
DECOMP_EXAMPLES = (
    "Pregunta: ¿Qué división aportó más ingresos en 2024, Proximidad Américas o Coca-Cola FEMSA, y por cuánto?\n"
    "¿Cuáles fueron los ingresos de Proximidad Américas en 2024?\n"
    "¿Cuáles fueron los ingresos de Coca-Cola FEMSA en 2024?\n\n"
    "Pregunta: ¿Qué porcentaje de los ingresos consolidados 2024 representan Proximidad Américas y Coca-Cola FEMSA juntas?\n"
    "¿Cuáles fueron los ingresos totales consolidados de FEMSA en 2024?\n"
    "¿Cuáles fueron los ingresos de Proximidad Américas en 2024?\n"
    "¿Cuáles fueron los ingresos de Coca-Cola FEMSA en 2024?\n\n"
    "Pregunta: ¿Cuántas estaciones tenía OXXO GAS al cierre de 2024?\n"
    "¿Cuántas estaciones tenía OXXO GAS al cierre de 2024?\n\n")


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


class MemoPipeline:
    def __init__(self, adapter_dir=ADAPTER, control=False):
        self.tok = AutoTokenizer.from_pretrained(BASE)
        print(f"loading {BASE} (bf16, CPU)...")
        base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
        self.model = base
        self.control = control
        if not control:
            print(f"loading adapter {adapter_dir} ...")
            self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()

    def ask_memory(self, subq):
        # MEMORY = adaptador LoRA activo (en modo control, la base sin entrenar).
        return memory_answer(self.tok, self.model, subq)

    def ask_executive(self, fn, *a, **k):
        if self.control:
            return fn(self.tok, self.model, *a, **k)
        with self.model.disable_adapter():
            return fn(self.tok, self.model, *a, **k)

    def run(self, q):
        subs = self.ask_executive(
            lambda tok, m, qq: llm_decompose(tok, m, qq), q)
        method = "llm"
        if not subs:
            subs = rule_decompose(q)
            method = "rule"
        sub_answers = [(s, self.ask_memory(s)) for s in subs]
        final = self.ask_executive(synthesize, q, sub_answers)
        return {"question": q, "decomposition": method,
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
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--control", action="store_true",
                    help="usa la base sin entrenar como MEMORY (control)")
    ap.add_argument("--out", default="memo/run_trace.jsonl")
    args = ap.parse_args()

    pipe = MemoPipeline(args.adapter, control=args.control)
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
