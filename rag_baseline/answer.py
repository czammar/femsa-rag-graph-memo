#!/usr/bin/env python3
"""RAG baseline (FEMSA 2024) — Paso 2: reader.

Lee rag_baseline/retrieval.json y responde cada pregunta con
Qwen2.5-1.5B-Instruct (local, CPU, bf16) usando UNICAMENTE los k chunks
recuperados como contexto. Es el mismo modelo que hizo de EXECUTIVE en
MEMO run 2 (descompuso 12/12), asi que puede seguir instrucciones.

Uso:
  memo/.venv/bin/python rag_baseline/answer.py [--retrieval rag_baseline/retrieval.json]
                                              [--out rag_baseline/run_trace.jsonl]

Salida: run_trace.jsonl en el mismo formato que memo/run_trace*.jsonl
(id, level, question, final_answer, decomposition="rag", retrieved_pages).
"""
import argparse
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

READER_MODEL = "memo/models/Qwen2.5-1.5B-Instruct"
MAX_NEW = 100
MAX_CHUNK_CHARS = 1800  # ventana por chunk (ver condense): el contexto completo
# de 5 paginas (~17k caracteres) hace el prefill en CPU inviable; se conserva
# la ventana mas relevante a la pregunta y se guarda en la traza para auditar.

STOPWORDS = set("""de la el en y a los del se las por un para con no una su al lo como
mas pero sus le ya o este mi que qué cuando donde cuales cuál cómo cuánto""".split())


def condense(text, question, max_chars=MAX_CHUNK_CHARS):
    """~max_chars caracteres por chunk: ventanas alrededor de las 2 lineas
    del chunk mas relevantes a la pregunta (overlap de palabras + lineas con
    cifras). Dos ventanas porque las dos cifras que una pregunta compara
    (p. ej. 39.3% y 35.8% en Q9) suelen estar lejos en la misma pagina.
    Sin esto el prompt de 5 paginas (~4.5k tokens) tarda >6 min por pregunta
    en CPU; con la ventana el prefill baja a ~2k tokens. Las ventanas usadas
    quedan guardadas en la traza (campo 'context')."""
    qwords = {w.lower() for w in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", question)
              if len(w) > 3 and w.lower() not in STOPWORDS}
    lines = text.splitlines() or [text]
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1

    def line_score(ln):
        low = ln.lower()
        s = sum(2 for w in qwords if w in low)
        if re.search(r"\d", ln):
            s += 1
        return s

    scored = sorted(((line_score(ln), i) for i, ln in enumerate(lines)
                     if line_score(ln) > 0), reverse=True)
    # top 2 lineas no adyacentes (ventanas que no se solapan)
    chosen = []
    for s, i in scored:
        if all(abs(i - j) > 3 for j in chosen):
            chosen.append(i)
        if len(chosen) == 2:
            break
    if not chosen:
        return text[:max_chars]
    half = max_chars // max(len(chosen), 1)
    parts = []
    for i in sorted(chosen):
        start = offsets[i]
        lo = max(0, start - half // 3)
        hi = min(len(text), lo + half)
        lo = max(0, hi - half)
        parts.append(text[lo:hi])
    out = "\n[...]\n".join(parts)
    return out[:max_chars]

SYSTEM = (
    "Responde la pregunta usando UNICAMENTE la información de los fragmentos "
    "del informe anual de FEMSA 2024 que se te dan a continuación. Copia las "
    "cifras exactamente como aparecen en los fragmentos, sin redondear ni "
    "inventar nada. Solo si la pregunta pide explícitamente sumar o comparar "
    "cifras que están en los fragmentos, haz esa operación con esas cifras. "
    "Si la información de los fragmentos no alcanza para responder, responde "
    "exactamente: No lo sé. Responde en español, en una o dos frases cortas.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", default="rag_baseline/retrieval.json")
    ap.add_argument("--model", default=READER_MODEL)
    ap.add_argument("--out", default="rag_baseline/run_trace.jsonl")
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"loading reader {args.model} (bf16, CPU)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.bfloat16)
    model.eval()
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    data = json.load(open(args.retrieval))
    with open(args.out, "w") as f:
        for item in data["questions"]:
            condensed = [condense(r["text"], item["question"])
                         for r in item["retrieved"]]
            ctx = "\n\n".join(
                f"[Página {r['page']}]\n{c}"
                for r, c in zip(item["retrieved"], condensed))
            user = (f"Fragmentos del informe anual de FEMSA 2024:\n\n{ctx}\n\n"
                    f"Pregunta: {item['question']}")
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            ids = tok(text, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=MAX_NEW,
                                     do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            ans = tok.decode(out[0][ids["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            rec = {"id": item["id"], "level": item["level"],
                   "question": item["question"],
                   "decomposition": "rag",
                   "reader_model": args.model,
                   "retrieved_pages": [r["page"] for r in item["retrieved"]],
                   "retrieval_scores": [r["score"] for r in item["retrieved"]],
                   "context": ctx,
                   "final_answer": ans}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{item['id']} -> {ans[:160]}", flush=True)
    print(f"trace -> {args.out} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
