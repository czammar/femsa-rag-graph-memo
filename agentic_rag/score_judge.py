#!/usr/bin/env python3
"""RAG agentico (FEMSA 2024) -- juez de seleccion, Paso 3: scoring.

Misma logica que rag_baseline/score_expanded.py y agentic_rag/score_agentic.py
(helpers de memo/evaluate.py, misma exactitud numerica, misma definicion de
alucinacion, misma regla Q3 con 1,596, mismas keywords Q8/Q10/Q11), asi que el
puntaje del juez es directamente comparable con los otros sistemas. La unica
diferencia es el SISTEMA: retrieval hibrido top-10 (BM25 + MiniLM, RRF k=60),
UN llamado al juez por pregunta que califica los 10 chunks de golpe
(UTIL / NO UTIL) con maximo 3 reintentos, y el lector responde UNICAMENTE con
los chunks seleccionados. Si el juez no selecciona nada (o no hay veredictos
validos tras 3 intentos), la respuesta es abstencion ("No lo se").

Ademas del puntaje, este scorer diagnostica AL JUEZ: compara las paginas
seleccionadas contra las paginas de evidencia del ground truth
(rag_baseline/ragas/ground_truth_evidence.json): precision de seleccion
(seleccionadas que son evidencia / seleccionadas) y recall de seleccion
(evidencia conservada / evidencia total). Eso separa "el juez tiro la
evidencia" de "el lector fallo con la evidencia enfrente".
VERIFIED_MODES se llena a mano DESPUES de la corrida, verificando cada
fallo contra el corpus (igual que en score_agentic.py).

Uso:
  agentic_rag/.venv/bin/python agentic_rag/score_judge.py [--trace agentic_rag/run_trace_judge.jsonl]
                                                  [--out agentic_rag/results_agentic_judge.md]
  # Despues de verificar los modos de fallo a mano en VERIFIED_MODES,
  # se vuelve a correr SIN --blank-modes (con el mismo --model-label)
  # para que el informe lleve los modos reales y el veredicto.
  # En Colab, el notebook 07 corre este mismo script contra la traza de
  # Colab (--trace .../run_trace_judge_colab.jsonl --out .../results_agentic_judge_colab.md).
"""
import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, "memo")
from evaluate import digits, numbers_in, expected_hit  # noqa: E402


def literal_present(text, expected):
    """True si la cifra esperada aparece literalmente en el texto (con ',' o
    '.' como separador). Sin tolerancia de redondeo."""
    e = str(expected).replace(",", "")
    if "." in e:
        return e in text or e.replace(".", ",") in text
    grouped = f"{int(e):,}"
    return grouped in text or e in text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="agentic_rag/run_trace_judge.jsonl")
    ap.add_argument("--eval", default="memo/eval_questions.json")
    ap.add_argument("--qa", default="memo/reflection_qa.jsonl")
    ap.add_argument("--seed", default="pipeline/triples_seed.json")
    ap.add_argument("--ground-truth",
                    default="rag_baseline/ragas/ground_truth_evidence.json")
    ap.add_argument("--out", default="agentic_rag/results_agentic_judge.md")
    ap.add_argument("--model-label", default="local Qwen2.5-1.5B-Instruct (bf16, CPU)",
                    help="modelo juez/lector que corrio (aparece en el informe)")
    ap.add_argument("--system-name", default="RAG agentico juez (este run)",
                    help="etiqueta de la fila de este run en la tabla comparativa")
    ap.add_argument("--blank-modes", action="store_true",
                    help="modos de fallo en PENDIENTE (para un run nuevo: "
                    "se verifican a mano y se llenan en VERIFIED_MODES; "
                    "el veredicto se escribe despues)")
    args = ap.parse_args()

    eval_data = json.load(open(args.eval))
    questions = {q["id"]: q for q in eval_data["questions"]}
    trace = [json.loads(l) for l in open(args.trace)]

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

    # Paginas de evidencia por pregunta (ground truth auditable, RAGAS).
    evidence_pages = {}
    try:
        gt = json.load(open(args.ground_truth))
        for q in gt.get("questions", []):
            pages = set()
            for item in q.get("evidence", []):
                pages |= set(item.get("pages", []))
            evidence_pages[q["id"]] = pages
    except FileNotFoundError:
        evidence_pages = {}

    # Clasificacion del fallo: SE LLENA A MANO DESPUES de la corrida,
    # verificando cada respuesta contra el corpus y contra el trace.
    # NO correr el scorer con los PENDIENTE sin llenar.
    VERIFIED_MODES = {f"Q{i}": "PENDIENTE" for i in range(1, 13)}

    if args.blank_modes:
        VERIFIED_MODES = {qid: "PENDIENTE" for qid in VERIFIED_MODES}

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
        # Cobertura literal: en lo que presento el retrieval (top-10) y en
        # lo que selecciono el juez (el contexto que vio el lector).
        presented_text = "\n".join(c.get("condensed", "")
                                   for c in r.get("chunks_presented", []))
        cov_presented = {e: literal_present(presented_text, e) for e in exp_nums}
        cov_selected = {e: literal_present(r.get("context", ""), e) for e in exp_nums}
        # Diagnostico del juez contra el ground truth de evidencia.
        ev = evidence_pages.get(r["id"], set())
        sel = set(r.get("selected_pages", []))
        sel_prec = (len(sel & ev) / len(sel)) if sel else None
        sel_rec = (len(sel & ev) / len(ev)) if ev else None
        judge = r.get("judge", {})
        mode = VERIFIED_MODES[r["id"]]
        if verdict == "correcta":
            mode = "ok"
        rows.append({"id": r["id"], "level": q["level"],
                     "question": q["question"], "expected": q["expected"],
                     "answer": ans, "verdict": verdict,
                     "hallucinated": halluc, "note": note,
                     "judge_ok": judge.get("ok"),
                     "judge_attempts": judge.get("attempts"),
                     "judge_seconds": judge.get("seconds", 0),
                     "n_selected": len(sel),
                     "selected_pages": sorted(sel),
                     "sel_precision": sel_prec,
                     "sel_recall": sel_rec,
                     "evidence_pages": sorted(ev),
                     "reader_attempts": r.get("reader", {}).get("attempts", 0),
                     "reader_skipped": r.get("reader", {}).get("skipped", False),
                     "reader_failed": r.get("reader", {}).get("failed", False),
                     "seconds": r.get("seconds", 0),
                     "coverage_presented": cov_presented,
                     "coverage_selected": cov_selected,
                     "failure_mode": mode})

    by_level = {}
    for row in rows:
        by_level.setdefault(row["level"], []).append(row)
    total_ok = sum(1 for r in rows if r["verdict"] == "correcta")
    total_halluc = sum(1 for r in rows if r["hallucinated"])
    n_ev_full = sum(1 for r in rows if r["coverage_selected"]
                    and all(r["coverage_selected"].values()))
    n_judge_ok = sum(1 for r in rows if r["judge_ok"])
    n_abstain_judge = sum(1 for r in rows if not r["judge_ok"])
    n_abstain_empty = sum(1 for r in rows
                          if r["judge_ok"] and r["n_selected"] == 0)
    total_selected = sum(r["n_selected"] for r in rows)
    # Precision/recall de seleccion agregados (pooled sobre las 12).
    sel_inter = sum(len(set(r["selected_pages"]) & set(r["evidence_pages"]))
                    for r in rows)
    sel_total = sum(len(r["selected_pages"]) for r in rows)
    ev_total = sum(len(r["evidence_pages"]) for r in rows)
    pooled_prec = (sel_inter / sel_total) if sel_total else None
    pooled_rec = (sel_inter / ev_total) if ev_total else None
    total_judge_seconds = sum(r["judge_seconds"] for r in rows)
    total_seconds = sum(r["seconds"] for r in rows)
    pending = [r["id"] for r in rows if r["failure_mode"] == "PENDIENTE"]
    if pending:
        print(f"AVISO: modos de fallo PENDIENTES de verificar a mano: {pending}")

    def lvl_ok(lv):
        return sum(1 for r in by_level.get(lv, [])
                   if r["verdict"] == "correcta")

    def pct(x):
        return f"{x:.2f}" if x is not None else "n/a"

    lines = ["# Resultados RAG agentico juez (seleccion de chunks con juez) — FEMSA 2024",
             "",
             "Sistema: retrieval hibrido top-10 (BM25 Okapi + MiniLM, RRF k=60), "
             f"UN llamado al juez por pregunta que califica los 10 chunks de golpe "
             f"(UTIL / NO UTIL, modelo {args.model_label}), maximo 3 reintentos si los "
             "veredictos no se pueden parsear (si a los 3 sigue sin veredictos validos, "
             "la pregunta se abstiene sin llamar al lector), y el lector responde "
             "UNICAMENTE con los chunks seleccionados (mismo lector, mismo condense, "
             "mismo prompt que rag_baseline/answer.py; si el juez no selecciona nada, "
             "abstencion sin llamar al lector). Sin graph_query, sin calculator. "
             "Scorer identico a memo/evaluate.py (misma exactitud numerica, misma "
             "definicion de alucinacion, misma regla Q3 con 1,596).",
             "",
             f"Correctas: **{total_ok}/12**. Con alucinacion (numero no verificado en el informe): **{total_halluc}/12**. "
             f"Preguntas donde el contexto seleccionado contenia todas las cifras "
             f"esperadas (literal): **{n_ev_full}/12**. "
             f"Juez con veredictos validos: **{n_judge_ok}/12** preguntas "
             f"({n_abstain_judge} abstencion{'es' if n_abstain_judge != 1 else ''} por juez fallido, "
             f"{n_abstain_empty} por seleccion vacia). "
             f"Chunks seleccionados: **{total_selected}/120** "
             f"(media {total_selected / max(len(rows), 1):.1f} por pregunta). "
             f"Seleccion del juez vs evidencia (pooled): precision "
             f"**{pct(pooled_prec)}**, recall **{pct(pooled_rec)}**. "
             f"Tiempo total de la corrida: **{total_seconds / 60:.1f} min** "
             f"(juez {total_judge_seconds / 60:.1f} min, media "
             f"{total_seconds / max(len(rows), 1):.0f} s por pregunta).",
             "",
             "| ID | Nivel | Pregunta | Esperado | Respuesta del agente | Veredicto | Sel. | Juez | Modo de fallo |",
             "|----|-------|----------|----------|----------------------|-----------|------|------|---------------|"]
    for r in rows:
        a = r["answer"].replace("|", "/").replace("\n", " ")[:200]
        judge_s = f"{r['judge_attempts']} int." if r["judge_ok"] else "fallo"
        lines.append(f"| {r['id']} | {r['level']} | {r['question'][:60]} | "
                     f"{r['expected'][:55]} | {a} | {r['verdict']} | "
                     f"{r['n_selected']}/10 | {judge_s} | {r['failure_mode']} |")
    lines += ["",
              "## Por nivel",
              ""]
    for lv in sorted(by_level):
        lines.append(f"- Nivel {lv}: {lvl_ok(lv)}/{len(by_level[lv])} correctas.")
    lines += ["",
              "## Diagnostico del juez (seleccion vs evidencia)",
              "",
              "Precision = paginas seleccionadas que son evidencia / paginas "
              "seleccionadas. Recall = paginas de evidencia conservadas / paginas de "
              "evidencia totales (ground truth auditable, "
              "rag_baseline/ragas/ground_truth_evidence.json). Un recall bajo dice que "
              "el juez tiro la evidencia; una precision baja dice que el juez dejo pasar "
              "ruido y el lector tuvo que elegir entre mas chunks.",
              "",
              "| ID | Seleccionados | Precision | Recall | Paginas de evidencia |",
              "|----|---------------|-----------|--------|----------------------|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['n_selected']}/10 ({', '.join(str(p) for p in r['selected_pages']) or 'ninguna'}) | "
                     f"{pct(r['sel_precision'])} | {pct(r['sel_recall'])} | "
                     f"{', '.join(str(p) for p in r['evidence_pages']) or 'n/a'} |")
    lines += ["",
              f"**Pooled:** precision {pct(pooled_prec)} ({sel_inter}/{sel_total}), "
              f"recall {pct(pooled_rec)} ({sel_inter}/{ev_total}).",
              "",
              "## Modos de fallo",
              ""]
    for mode, n in Counter(r["failure_mode"] for r in rows).most_common():
        ids = ", ".join(r["id"] for r in rows if r["failure_mode"] == mode)
        lines.append(f"- {mode}: {n} ({ids})")
    lines += ["",
              "## Comparacion de los sistemas (12 preguntas, mismo scorer)",
              "",
              "| Sistema | Correctas | Nivel 1 | Nivel 2 | Nivel 3 |",
              "|---------|-----------|---------|---------|---------|",
              "| Grafo (Relate, Cypher) | 12/12 | 4/4 | 4/4 | 4/4 |",
              f"| {args.system_name} | {total_ok}/12 | {lvl_ok(1)}/4 | {lvl_ok(2)}/4 | {lvl_ok(3)}/4 |",
              "| RAG hibrido + expansion de queries | 5/12 | 2/4 | 2/4 | 1/4 |",
              "| RAG hibrido BM25+denso (sin expansion) | 2/12 | 1/4 | 0/4 | 1/4 |",
              "| RAG vectorial denso puro | 2/12 | 0/4 | 1/4 | 1/4 |",
              "| RAG agentico ReAct (LangChain + LangGraph, 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |",
              "| MEMO run 2 (0.5B LoRA r=32 + ejecutivo 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |",
              "| MEMO run 1 (0.5B LoRA r=8 + ejecutivo 0.5B) | 0/12 | 0/4 | 0/4 | 0/4 |",
              ""]
    if not args.blank_modes:
        lines += ["## Que cambio vs el RAG hibrido + expansion (5/12)",
                  "",
                  "PENDIENTE de redactar tras verificar los modos de fallo a mano: "
                  "comparar pregunta por pregunta cuales flips son por la seleccion del "
                  "juez (tiro la evidencia / dejo pasar ruido) y cuales por el lector.",
                  "",
                  "## Veredicto para el post",
                  "",
                  "PENDIENTE de redactar tras verificar los modos de fallo a mano: la "
                  "pregunta es si la seleccion con juez supera el top-5 fijo del pipeline "
                  "hibrido+expansion (5/12) con el mismo modelo y el mismo scorer.",
                  "",
                  ]
    else:
        lines += ["## Veredicto para el post",
                  "",
                  "Pendiente de revision manual. Los numeros de la tabla de arriba ya son "
                  "comparables con los demas sistemas (mismo scorer, mismas 12 preguntas); "
                  "lo que falta es verificar cada fallo contra el corpus y contra la "
                  "traza, llenar VERIFIED_MODES en score_judge.py y volver a correr el "
                  "scorer sin --blank-modes para que escriba los modos de fallo y el "
                  "veredicto.",
                  "",
                  ]
    lines += ["## Caveats (verificados, no supuestos)",
              "",
              "- **Q7 y Q9 son techo (9/12), igual que en los otros runs.** Las cifras "
              "esperadas de Q7 (+27.5%/+24.9%) vienen del comunicado 4Q24 y las de Q9 "
              "(39.3/35.8/75.1%) son participaciones calculadas: no estan en el informe. "
              "Q2 tambien pide +24.9% como parte de lo esperado. El techo real del juez "
              "sobre este set es 9/12, no 12/12.",
              "- **El juez ve la version condensada de cada chunk, la misma que vera el "
              "lector.** Un chunk es util si lo que el lector va a ver de el sirve para "
              "responder; la seleccion se mide contra esa misma version, no contra la "
              "pagina completa.",
              "- **El resto de los caveats esta pendiente.** Los modos de fallo de este "
              "run todavia no se verificaron a mano: ver la seccion de veredicto arriba.",
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
              "## Detalle de evidencia por pregunta (cobertura literal)",
              "",
              "| ID | Cifras esperadas en top-10 presentado | Cifras esperadas en contexto seleccionado |",
              "|----|----------------------------------------|-------------------------------------------|"]
    for r in rows:
        cov_p = ", ".join(f"{k}: {'si' if v else 'no'}"
                          for k, v in r["coverage_presented"].items()) \
            if r["coverage_presented"] else "(sin cifras esperadas)"
        cov_s = ", ".join(f"{k}: {'si' if v else 'no'}"
                          for k, v in r["coverage_selected"].items()) \
            if r["coverage_selected"] else "(sin cifras esperadas)"
        lines.append(f"| {r['id']} | {cov_p} | {cov_s} |")

    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"correctas {total_ok}/12, con alucinacion {total_halluc}/12, "
          f"seleccionados {total_selected}/120, juez ok {n_judge_ok}/12, "
          f"precision {pct(pooled_prec)}, recall {pct(pooled_rec)}, "
          f"{total_seconds / 60:.1f} min -> {args.out}")
    for r in rows:
        print(f"  {r['id']} L{r['level']} {r['verdict']} [{r['failure_mode']}] "
              f"sel={r['n_selected']}/10 juez={'ok' if r['judge_ok'] else 'fallo'} "
              f"({r['judge_attempts']} int.) {r['seconds']:.0f}s"
              + (f" halluc={r['hallucinated']}" if r["hallucinated"] else ""))


if __name__ == "__main__":
    main()
