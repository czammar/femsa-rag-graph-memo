#!/usr/bin/env python3
"""Evalua la corrida MEMO sobre las 12 preguntas y escribe memo/results.md.

Metricas por pregunta:
  - exactitud del numero: todos los numeros esperados aparecen en la respuesta
    final (tolerancia: se comparan digitos normalizados; se aceptan los valores
    exactos del MD&A y los redondeados de la ontologia).
  - alucinacion: cualquier numero en la respuesta final que NO este en el
    conjunto de numeros verificados del informe (construido desde
    reflection_qa.jsonl + triples_seed.json). Un numero alucinado = fallo.
  - Q3 tiene una nota de ground-truth: la ontologia dice 1,236 aperturas, pero
    el informe dice 1,596 tiendas netas nuevas. Si el modelo responde 1,596 se
    marca "correcto segun el informe" aunque difiera de la respuesta esperada.

Comparacion con el grafo: el sistema de grafo es ~perfecto en nivel 1-2 POR
CONSTRUCCION (las respuestas son las tripletas semilla); se dice asi, no se
fabrican numeros. La baseline RAG vectorial queda como "pendiente" si no se
corrio (no se inventan cifras de RAG).

Uso:
  .venv/bin/python memo/evaluate.py [--trace memo/run_trace.jsonl] [--out memo/results.md]
"""
import argparse
import json
import re


def digits(s):
    return re.sub(r"[.,]", "", s)


def numbers_in(text):
    return {digits(m.group(0)) for m in re.finditer(r"\d[\d.,]*", text or "")}


def norm_num(x):
    """Normaliza un numero esperado ('781585', '11.2') a digitos comparables."""
    return digits(str(x))


def expected_hit(answer_nums, expected):
    """True si cada numero esperado aparece (o su redondeo a entero) en la respuesta."""
    for e in expected:
        d = norm_num(e)
        if d in answer_nums:
            continue
        # tolerancia de redondeo: 307197 vs 307200, 279793 vs 279800
        try:
            v = float(e)
            ok = any(abs(float(a) - v) <= max(1.0, abs(v) * 0.001)
                     for a in answer_nums if a.replace(".", "", 1).isdigit())
        except ValueError:
            ok = False
        if not ok:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="memo/run_trace.jsonl")
    ap.add_argument("--eval", default="memo/eval_questions.json")
    ap.add_argument("--qa", default="memo/reflection_qa.jsonl")
    ap.add_argument("--seed", default="pipeline/triples_seed.json")
    ap.add_argument("--out", default="memo/results.md")
    args = ap.parse_args()

    eval_data = json.load(open(args.eval))
    questions = {q["id"]: q for q in eval_data["questions"]}
    trace = [json.loads(l) for l in open(args.trace)]

    # conjunto de numeros verificados del informe
    verified = set()
    try:
        for line in open(args.qa):
            r = json.loads(line)
            verified |= numbers_in(r["answer"])
    except FileNotFoundError:
        pass
    try:
        seed = json.load(open(args.seed))
        triples = seed if isinstance(seed, list) else seed.get("triples", [])
        for t in triples:
            for k in ("valor", "variacion_pct"):
                if t.get(k) is not None:
                    verified.add(norm_num(t[k]))
    except FileNotFoundError:
        pass
    # anios y periodos no son alucinaciones
    verified |= {"2024", "2023", "2022"}
    # las cifras esperadas del propio set de evaluacion tambien estan
    # verificadas (con el set de 100, si no, respuestas correctas nuevas se
    # marcarian como alucinacion por no estar en reflection_qa/triples_seed)
    for _q in questions.values():
        verified |= numbers_in(_q.get("expected", ""))
        verified |= {norm_num(n) for n in _q.get("expected_numbers", [])}

    rows = []
    for r in trace:
        q = questions[r["id"]]
        ans = r["final_answer"]
        ans_nums = numbers_in(ans)
        halluc = sorted(n for n in ans_nums if n not in verified and len(n) > 1)
        ok = expected_hit(ans_nums, q.get("expected_numbers", []))
        # Preguntas sin numeros esperados: se evaluan por palabras clave.
        # Sin keywords ni numeros, NUNCA se marca correcta automaticamente
        # (el bug original: expected_hit([]) devolvia True por vacuidad y
        # Q8/Q10/Q11 salian "correctas" con respuestas erroneas).
        kws = q.get("expected_keywords", [])
        if not q.get("expected_numbers") and kws:
            low = ans.lower()
            ok = all(k.lower() in low for k in kws)
        elif not q.get("expected_numbers") and not kws:
            ok = False
        note = ""
        if q.get("ground_truth_note"):
            note = "Nota ground-truth: " + q["ground_truth_note"]
        if r["id"] == "Q3" and "1596" in ans_nums:
            ok = True
            note += " (El modelo respondio 1,596 = valor real del informe; se cuenta como correcto.)"
        verdict = "correcta" if ok and not halluc else (
            "correcta con alucinacion" if ok else "incorrecta")
        rows.append({"id": r["id"], "level": q["level"], "question": q["question"],
                     "expected": q["expected"], "answer": ans,
                     "decomposition": r.get("decomposition"),
                     "verdict": verdict, "hallucinated": halluc, "note": note})

    by_level = {}
    for row in rows:
        by_level.setdefault(row["level"], []).append(row)
    total_ok = sum(1 for r in rows if r["verdict"] == "correcta")
    total_halluc = sum(1 for r in rows if r["hallucinated"])

    lines = ["# Resultados MEMO (demo) — FEMSA 2024",
             "",
             "Sistema: EXECUTIVE = Qwen2.5-0.5B-Instruct base (congelado); "
             "MEMORY = base + adaptador LoRA entrenado en reflexiones del informe. "
             "El EXECUTIVE nunca ve el corpus; solo las sub-respuestas de MEMORY.",
             "",
             f"Correctas: **{total_ok}/12**. Con alucinacion (numero no verificado en el informe): **{total_halluc}/12**.",
             "",
             "| ID | Nivel | Pregunta | Esperado | Respuesta MEMO | Veredicto |",
             "|----|-------|----------|----------|----------------|-----------|"]
    for r in rows:
        a = r["answer"].replace("|", "/").replace("\n", " ")[:220]
        lines.append(f"| {r['id']} | {r['level']} | {r['question'][:70]} | "
                     f"{r['expected'][:60]} | {a} | {r['verdict']} |")
    lines += ["",
              "## Por nivel",
              ""]
    for lv in sorted(by_level):
        ok = sum(1 for r in by_level[lv] if r["verdict"] == "correcta")
        lines.append(f"- Nivel {lv}: {ok}/{len(by_level[lv])} correctas.")
    lines += ["",
              "## Comparacion con el grafo (declarada, no medida con numeros inventados)",
              "",
              "- Grafo (Cypher sobre tripletas semilla): ~perfecto en nivel 1-2 **por "
              "construccion**, porque las respuestas son las tripletas verificadas. "
              "En nivel 3 depende de que el agente sepa agregar sin inventar.",
              "- MEMO (demo): ver tabla. La memoria es un modelo entrenado, no una base "
              "de datos: puede equivocar cifras aunque haya visto el hecho en "
              "entrenamiento. Esa es justo la diferencia que el post debe mostrar.",
              "- RAG vectorial: **pendiente** (no se corrio en este demo; no se "
              "inventan cifras).",
              "",
              "## Notas de ground-truth",
              ""]
    for r in rows:
        if r["note"]:
            lines.append(f"- {r['id']}: {r['note']}")
    if not any(r["note"] for r in rows):
        lines.append("- Ninguna.")
    lines += ["",
              "## Alucinaciones detectadas",
              ""]
    for r in rows:
        if r["hallucinated"]:
            lines.append(f"- {r['id']}: numeros no verificados en el informe: "
                         f"{', '.join(r['hallucinated'])}")
    if not any(r["hallucinated"] for r in rows):
        lines.append("- Ninguna.")

    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"correctas {total_ok}/12, con alucinacion {total_halluc}/12 -> {args.out}")
    for r in rows:
        print(f"  {r['id']} L{r['level']} [{r['decomposition']}] {r['verdict']}"
              + (f" halluc={r['hallucinated']}" if r["hallucinated"] else ""))


if __name__ == "__main__":
    main()
