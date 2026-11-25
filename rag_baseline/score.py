#!/usr/bin/env python3
"""RAG baseline (FEMSA 2024) — Paso 3: scoring.

Reutiliza la logica de memo/evaluate.py (mismos helpers: digits,
numbers_in, expected_hit; mismo conjunto de numeros verificados del
informe; misma regla Q3 con 1,596; mismas keywords para Q8/Q10/Q11), asi
que el puntaje RAG es directamente comparable con MEMO run 1 (0/12) y
run 2 (2/12). Ademas cruza el resultado con el diagnostico de retrieval
para clasificar cada fallo: retrieval fallo (la cifra no estaba en los
chunks recuperados) vs reader fallo (la cifra SI estaba, el lector
contesto mal) vs alucinacion.

Uso:
  memo/.venv/bin/python rag_baseline/score.py [--trace rag_baseline/run_trace.jsonl]
                                             [--retrieval rag_baseline/retrieval.json]
                                             [--out rag_baseline/results.md]
"""
import argparse
import json
import sys

sys.path.insert(0, "memo")
from evaluate import digits, numbers_in, expected_hit  # noqa: E402


def literal_present(text, expected):
    """True si la cifra esperada aparece literalmente en el texto (con ',' o
    '.' como separador). Sin tolerancia de redondeo: la tolerancia de
    expected_hit marca 'cubierto' un 27.5% cuando solo hay un 27 suelto en el
    texto, y eso falsea el diagnostico de retrieval."""
    e = str(expected).replace(",", "")
    if "." in e:
        return e in text or e.replace(".", ",") in text
    # entero: aparece con separador de miles o sin el
    grouped = f"{int(e):,}"
    return grouped in text or e in text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="rag_baseline/run_trace.jsonl")
    ap.add_argument("--retrieval", default="rag_baseline/retrieval.json")
    ap.add_argument("--eval", default="memo/eval_questions.json")
    ap.add_argument("--qa", default="memo/reflection_qa.jsonl")
    ap.add_argument("--seed", default="pipeline/triples_seed.json")
    ap.add_argument("--chunks", default="pipeline/data/chunks.jsonl")
    ap.add_argument("--out", default="rag_baseline/results.md")
    args = ap.parse_args()

    eval_data = json.load(open(args.eval))
    questions = {q["id"]: q for q in eval_data["questions"]}
    trace = [json.loads(l) for l in open(args.trace)]
    retr = {r["id"]: r for r in json.load(open(args.retrieval))["questions"]}
    corpus_text = "\n".join(json.loads(l)["text"]
                            for l in open(args.chunks))

    # mismo conjunto de numeros verificados que memo/evaluate.py
    verified = set()
    try:
        for line in open(args.qa):
            verified |= numbers_in(json.loads(line)["answer"])
    except FileNotFoundError:
        pass
    try:
        seed = json.load(open(args.seed))
        triples = seed if isinstance(seed, list) else seed.get("triples", [])
        for t in triples:
            for k in ("valor", "variacion_pct"):
                if t.get(k) is not None:
                    verified.add(digits(str(t[k])))
    except FileNotFoundError:
        pass
    verified |= {"2024", "2023", "2022"}

    rows = []
    for r in trace:
        q = questions[r["id"]]
        ans = r["final_answer"]
        ans_nums = numbers_in(ans)
        halluc = sorted(n for n in ans_nums if n not in verified and len(n) > 1)
        ok = expected_hit(ans_nums, q.get("expected_numbers", []))
        kws = q.get("expected_keywords", [])
        if not q.get("expected_numbers") and kws:
            ok = all(k.lower() in ans.lower() for k in kws)
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
        exp_nums = q.get("expected_numbers", [])
        top5_text = "\n".join(x["text"] for x in retr[r["id"]]["retrieved"])
        ctx_text = r.get("context", "")
        missing_top5 = [e for e in exp_nums
                        if not literal_present(top5_text, e)]
        missing_ctx = [e for e in exp_nums
                       if not literal_present(ctx_text, e)]
        # Clasificacion del fallo: VERIFICADA A MANO contra el corpus (no
        # derivada por heuristica — la heuristica automatica etiqueta mal Q1,
        # Q10 y Q12: en Q1 el lector tenia 781,585 enfrente y contesto 40,236;
        # en Q10 'FEMSA Forward' no estaba en el top-5; en Q12 el 5.8 del
        # contexto era margen de utilidad neta, no el crecimiento de Salud).
        # Cada etiqueta se comprobo buscando la cifra literal en chunks.jsonl.
        VERIFIED_MODES = {
            "Q1": "fallo del lector (781,585 en contexto; contesto 40,236; el +11.2% tampoco se recupero)",
            "Q2": "cifra esperada fuera del corpus (+24.9% no existe en el informe; el lector contesto bien lo preguntado)",
            "Q3": "fallo de retrieval (23,206 solo en p.21, no en top-5)",
            "Q4": "fallo de retrieval (571 en pp. 21/114/126, no en top-5)",
            "Q5": "alucinacion del lector (divago, se corto a 100 tokens; '114' es numero de pagina)",
            "Q6": "ok",
            "Q7": "cifra esperada fuera del corpus (+27.5% y +24.9% no existen en el informe; el lector acerto direccion y absolutos)",
            "Q8": "fallo del lector (Chile y Peru en contexto; listo solo 2 de 4)",
            "Q9": "cifra esperada fuera del corpus + aritmetica (39.3/35.8/75.1% no existen; sumo 23,729+19,536=43,265)",
            "Q10": "fallo de retrieval ('FEMSA Forward' en pp. 3/5/8/12/14/42/101/115, no en top-5)",
            "Q11": "ok",
            "Q12": "fallo de retrieval (Health +5.8% en pp. 5/112/114, no en top-5; el 5.8 del contexto era margen neto)",
        }
        mode = VERIFIED_MODES[r["id"]]
        if verdict == "correcta":
            mode = "ok"
        # cobertura literal (sin tolerancia) por cifra, en top-5 y en contexto
        cov_literal = {e: literal_present(top5_text, e) for e in exp_nums}
        ctx_literal = {e: literal_present(ctx_text, e) for e in exp_nums}
        rows.append({"id": r["id"], "level": q["level"],
                     "question": q["question"], "expected": q["expected"],
                     "answer": ans, "verdict": verdict,
                     "hallucinated": halluc, "note": note,
                     "retrieval_pages": r.get("retrieved_pages", []),
                     "coverage": cov_literal,
                     "coverage_ctx": ctx_literal,
                     "failure_mode": mode,
                     "missing_top5": missing_top5,
                     "missing_ctx": missing_ctx})

    by_level = {}
    for row in rows:
        by_level.setdefault(row["level"], []).append(row)
    total_ok = sum(1 for r in rows if r["verdict"] == "correcta")
    total_halluc = sum(1 for r in rows if r["hallucinated"])
    n_retr_full = sum(1 for r in rows if r["coverage"]
                      and all(r["coverage"].values()))
    n_retr_none = sum(1 for r in rows if r["coverage"]
                      and not any(r["coverage"].values()))

    lines = ["# Resultados RAG vectorial (baseline) — FEMSA 2024",
             "",
             "Sistema: retrieval top-5 paginas con all-MiniLM-L6-v2 (CPU) sobre "
             "las 151 paginas del informe; reader Qwen2.5-1.5B-Instruct local "
             "(bf16, CPU), respondiendo SOLO desde los chunks recuperados. "
             "Scorer identico a memo/evaluate.py (misma exactitud numerica, "
             "misma definicion de alucinacion, misma regla Q3 con 1,596).",
             "",
             f"Correctas: **{total_ok}/12**. Con alucinacion (numero no verificado en el informe): **{total_halluc}/12**. "
             f"Retrieval top-5 con todas las cifras esperadas literales: **{n_retr_full}/12** "
             f"(el diagnostico con tolerancia de redondeo decia 6/12; ver Caveats).",
             "",
             "| ID | Nivel | Pregunta | Esperado | Respuesta RAG | Veredicto | Modo de fallo |",
             "|----|-------|----------|----------|---------------|-----------|---------------|"]
    for r in rows:
        a = r["answer"].replace("|", "/").replace("\n", " ")[:200]
        lines.append(f"| {r['id']} | {r['level']} | {r['question'][:60]} | "
                     f"{r['expected'][:55]} | {a} | {r['verdict']} | {r['failure_mode']} |")
    lines += ["",
              "## Por nivel",
              ""]
    for lv in sorted(by_level):
        ok = sum(1 for r in by_level[lv] if r["verdict"] == "correcta")
        lines.append(f"- Nivel {lv}: {ok}/{len(by_level[lv])} correctas.")
    lines += ["",
              "## Modos de fallo",
              ""]
    from collections import Counter
    for mode, n in Counter(r["failure_mode"] for r in rows).most_common():
        ids = ", ".join(r["id"] for r in rows if r["failure_mode"] == mode)
        lines.append(f"- {mode}: {n} ({ids})")
    lines += ["",
              "## Comparacion de los tres sistemas (12 preguntas, mismo scorer)",
              "",
              "| Sistema | Correctas | Nivel 1 | Nivel 2 | Nivel 3 |",
              "|---------|-----------|---------|---------|---------|",
              "| Grafo (Relate, Cypher) | 12/12 | 4/4 | 4/4 | 4/4 |",
              f"| RAG vectorial (este baseline) | {total_ok}/12 | "
              f"{sum(1 for r in by_level[1] if r['verdict'] == 'correcta')}/4 | "
              f"{sum(1 for r in by_level[2] if r['verdict'] == 'correcta')}/4 | "
              f"{sum(1 for r in by_level[3] if r['verdict'] == 'correcta')}/4 |",
              "| MEMO run 2 (0.5B LoRA r=32 + ejecutivo 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |",
              "| MEMO run 1 (0.5B LoRA r=8 + ejecutivo 0.5B) | 0/12 | 0/4 | 0/4 | 0/4 |",
              "",
              "## Veredicto para el post",
              "",
              "**RAG vectorial: 2/12 — el mismo puntaje que MEMO run 2, pero con un "
              "perfil de fallo distinto y mas instructivo.** El grafo gana 12/12 "
              "porque la respuesta *es* el dato verificado; RAG y MEMO pierden "
              "porque tienen que *reconstruir* la cifra — uno desde texto "
              "recuperado, el otro desde pesos. El experimento muestra que, a "
              "esta escala y con un lector de 1.5B, el RAG clasico no es mejor "
              "que la memoria parametrica: empata con ella.",
              "",
              "Donde pierde RAG, verificado cifra por cifra contra el corpus:",
              "",
              "- **Retrieval (4): Q3, Q4, Q10, Q12.** La cifra existe en el "
              "informe y el top-5 no la trajo. Q3: 23,206 esta solo en la p.21; "
              "Q4: 571 esta en pp. 21/114/126; Q10: 'FEMSA Forward' esta en pp. "
              "3/5/8/12/14/42/101/115; Q12: Health +5.8% esta en pp. 5/112/114. "
              "El lector en Q3/Q4 se abstuvo honestamente ('No lo se'); en "
              "Q10/Q12 contesto con otra cosa. Un retriever mejor (o chunks "
              "mas pequenos que la pagina) es el arreglo mas directo aqui.",
              "- **Cifras esperadas fuera del corpus (3): Q2, Q7, Q9.** "
              "+24.9%, +27.5%, 39.3%, 35.8% y 75.1% no aparecen en ninguna de "
              "las 151 paginas. 24.9/27.5 vienen del comunicado de prensa 4Q24, "
              "no del informe anual; 39.3/35.8/75.1 son participaciones "
              "calculadas, no cifras impresas. El grafo las 'sabe' solo porque "
              "sus tripletas semilla se construyeron a mano con esas fuentes. "
              "En Q2 y Q7 el lector contesto bien lo sustantivo (8.6 millones; "
              "Spin Premia crece mas, 24.6 vs 8.6) y el scorer lo marca mal "
              "porque exige los porcentajes que no estan en el corpus. Techo "
              "real de RAG sobre este set, tal como esta especificado: 9/12.",
              "- **Lector (3): Q1, Q5, Q8.** La cifra estaba en el contexto y "
              "el lector fallo. Q1 eligio Ps. 40,236 (+31%) teniendo 781,585 "
              "enfrente — elige la cifra equivocada de la tabla. Q5 divago y "
              "se corto a los 100 tokens sin concluir, filtrando '114' (numero "
              "de pagina) como si fuera un dato. Q8 tenia Chile y Peru en el "
              "contexto y solo listo Mexico y Colombia. El lector de 1.5B no "
              "es un extractor fiable de tablas.",
              "- **Aritmetica (Q9).** Tenia 307,197 en contexto pero no "
              "279,793 ni 781,585 (perdida en el condensado); agarro 23,729 y "
              "19,536 y los sumo en 43,265. Es el mismo modo de fallo del "
              "ejecutivo de MEMO run 2: la sintesis rompe el dato correcto.",
              "",
              "Lectura para el post: el RAG clasico falla en los dos extremos — "
              "cuando no recupera el dato y cuando lo recupera pero el lector "
              "no sabe elegirlo. El grafo no tiene ninguno de los dos problemas "
              "porque no recupera ni elige: consulta. La moraleja no es que RAG "
              "sea inutil, sino que la exactitud de cifras exige que el dato "
              "llegue verificado al lector, no solo 'relevante'.",
              "",
              "## Caveats (verificados, no supuestos)",
              "",
              "- El diagnostico inicial de retrieval usaba la tolerancia de "
              "redondeo del scorer (±0.1%): marcaba 'cubierto' el 27.5% de Q7 "
              "cuando solo habia un 27 suelto, el 24.9% de Q2 cuando no hay "
              "ningun 24.9, y el 39.3/35.8/75.1% de Q9 cuando solo habia "
              "numeros cercanos. La cobertura literal estricta en top-5 es "
              "2/12 completa (Q5, Q12) — con dos matices: en Q6 las cifras "
              "estan como valores exactos (307,197/279,793) mientras el campo "
              "esperado trae los redondeados de la ontologia (307,200/279,800), "
              "y en Q12 el 5.8 literal del top-5 es el margen de utilidad neta, "
              "no el crecimiento de Salud. La tabla de detalle usa cobertura "
              "literal; la clasificacion de fallos se verifico a mano.",
              "- El condensado de contexto (ventanas de ~1,800 caracteres por "
              "chunk, 2 ventanas) fue necesario porque el prompt de 5 paginas "
              "completas (~4.5k tokens) tarda >6 min por pregunta en CPU. "
              "Perdio cifras en Q9 (279,793 y 781,585 estaban en el top-5 y "
              "no llegaron al lector). Es un costo medido del workaround, no "
              "del metodo RAG en si.",
              "- La pregunta esta en espanol y el corpus en ingles; el "
              "retrieval es cross-lingual con un embedder entrenado en ingles "
              "(all-MiniLM-L6-v2). Q12 exige mapear 'Salud' -> 'Health'.",
              "- Q2 no pregunta el crecimiento (+24.9%) pero el scorer lo "
              "exige; la respuesta '8.6 millones' es correcta para la pregunta "
              "tal como esta hecha. Misma severidad que en MEMO, asi que los "
              "puntajes son comparables.",
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
    lines += ["",
              "## Detalle de retrieval por pregunta (cobertura literal)",
              "",
              "| ID | Paginas recuperadas (top-5) | Cifras en top-5 | Cifras en contexto (condensado) |",
              "|----|-----------------------------|-----------------|---------------------------------|"]
    for r in rows:
        cov_s = ", ".join(f"{k}: {'si' if v else 'no'}"
                          for k, v in r["coverage"].items()) \
            if r["coverage"] else "(sin cifras esperadas)"
        ctx_s = ", ".join(f"{k}: {'si' if v else 'no'}"
                          for k, v in r["coverage_ctx"].items()) \
            if r["coverage_ctx"] else "(sin cifras esperadas)"
        lines.append(f"| {r['id']} | {r['retrieval_pages']} | {cov_s} | {ctx_s} |")

    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"correctas {total_ok}/12, con alucinacion {total_halluc}/12, "
          f"retrieval literal completo {n_retr_full}/12 -> {args.out}")
    for r in rows:
        print(f"  {r['id']} L{r['level']} {r['verdict']} [{r['failure_mode']}]"
              + (f" halluc={r['hallucinated']}" if r["hallucinated"] else ""))


if __name__ == "__main__":
    main()
