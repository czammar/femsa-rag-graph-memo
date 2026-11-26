#!/usr/bin/env python3
"""RAG hibrido BM25 + denso (FEMSA 2024) — Paso 3: scoring.

Misma logica que rag_baseline/score.py (helpers de memo/evaluate.py,
misma regla Q3 con 1,596, mismas keywords Q8/Q10/Q11), asi que el puntaje
hibrido es directamente comparable con el denso puro (2/12). La unica
diferencia entre ambas corridas es el retrieval: RRF k=60 sobre BM25
Okapi + MiniLM, mismo reader (Qwen2.5-1.5B), mismo prompt, mismo
condense. VERIFIED_MODES se lleno a mano DESPUES de la corrida,
verificando cada fallo contra el corpus (igual que en score.py).

Uso:
  memo/.venv/bin/python rag_baseline/score_hybrid.py [--trace rag_baseline/run_trace_hybrid.jsonl]
                                             [--retrieval rag_baseline/retrieval_hybrid.json]
                                             [--out rag_baseline/results_hybrid.md]
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
    ap.add_argument("--trace", default="rag_baseline/run_trace_hybrid.jsonl")
    ap.add_argument("--retrieval", default="rag_baseline/retrieval_hybrid.json")
    ap.add_argument("--eval", default="memo/eval_questions.json")
    ap.add_argument("--qa", default="memo/reflection_qa.jsonl")
    ap.add_argument("--seed", default="pipeline/triples_seed.json")
    ap.add_argument("--chunks", default="pipeline/data/chunks.jsonl")
    ap.add_argument("--out", default="rag_baseline/results_hybrid.md")
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
        # Clasificacion del fallo: SE LLENA A MANO DESPUES de la corrida,
        # verificando cada respuesta contra el corpus (misma regla que en
        # score.py: la heuristica automatica etiqueta mal varios casos).
        # NO correr el scorer con los PENDIENTE sin llenar.
        VERIFIED_MODES = {
            "Q1": "ok",
            "Q2": "cifra esperada fuera del corpus (+24.9% no existe en el informe; el lector contesto bien lo preguntado)",
            "Q3": "fallo de retrieval (23,206 solo en p.21, no en top-5 hibrido [16,88,75,72,55]; el lector se abstuvo)",
            "Q4": "fallo de retrieval + cifra mal atribuida (571 no en top-5 hibrido [83,100,85,88,92]; contesto '12 estaciones' con un 12 suelto del contexto)",
            "Q5": "fallo de retrieval (14.2 no en top-5 hibrido [28,88,128,80,100]; el lector se abstuvo)",
            "Q6": "fallo del lector (307,197 bien; Coca-Cola FEMSA la puso en 128,736 en vez de 279,793: 128,736 es su utilidad bruta, no su ingreso; el contexto hibrido cambio vs el denso)",
            "Q7": "cifra esperada fuera del corpus + aritmetica inventada (+27.5% y +24.9% no existen en el informe; el lector acerto direccion y absolutos, pero invento 'aumento un 275% (24.6 - 8.6 = 16)' y 'crecio un 50%'; el 16 cuenta como alucinacion)",
            "Q8": "fallo del lector (los 4 paises estaban en contexto; listo solo Mexico y 'America Latina')",
            "Q9": "cifra esperada fuera del corpus (39.3/35.8/75.1% no existen); el lector se abstuvo en vez de inventar la suma",
            "Q10": "fallo de retrieval ('FEMSA Forward' no en top-5 hibrido [13,75,4,34,117]; respuesta vaga sin el nombre)",
            "Q11": "ok",
            "Q12": "fallo de retrieval (Health +5.8% en pp. 5/112/114, no en top-5 hibrido [88,16,75,99,96]; el lector se abstuvo)",
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

    lines = ["# Resultados RAG hibrido (BM25 + denso, RRF k=60) — FEMSA 2024",
             "",
             "Sistema: retrieval hibrido top-5 paginas — BM25 Okapi (rank_bm25, "
             "k1=1.5, b=0.75) + all-MiniLM-L6-v2, fusion Reciprocal Rank Fusion "
             "k=60 — sobre las 151 paginas del informe; reader Qwen2.5-1.5B-Instruct "
             "local (bf16, CPU), mismo prompt y mismo condense que el baseline "
             "denso. Scorer identico a memo/evaluate.py (misma exactitud numerica, "
             "misma definicion de alucinacion, misma regla Q3 con 1,596). La unica "
             "diferencia con el baseline denso es el retrieval.",
             "",
             f"Correctas: **{total_ok}/12**. Con alucinacion (numero no verificado en el informe): **{total_halluc}/12**. "
             f"Retrieval top-5 con todas las cifras esperadas literales: **{n_retr_full}/12** "
             f"(denso puro: 2/12; ver Caveats).",
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
              "## Comparacion de los sistemas (12 preguntas, mismo scorer)",
              "",
              "| Sistema | Correctas | Nivel 1 | Nivel 2 | Nivel 3 |",
              "|---------|-----------|---------|---------|---------|",
              "| Grafo (Relate, Cypher) | 12/12 | 4/4 | 4/4 | 4/4 |",
              f"| RAG hibrido BM25+denso (este run) | {total_ok}/12 | "
              f"{sum(1 for r in by_level[1] if r['verdict'] == 'correcta')}/4 | "
              f"{sum(1 for r in by_level[2] if r['verdict'] == 'correcta')}/4 | "
              f"{sum(1 for r in by_level[3] if r['verdict'] == 'correcta')}/4 |",
              "| RAG vectorial denso puro | 2/12 | 0/4 | 1/4 | 1/4 |",
              "| MEMO run 2 (0.5B LoRA r=32 + ejecutivo 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |",
              "| MEMO run 1 (0.5B LoRA r=8 + ejecutivo 0.5B) | 0/12 | 0/4 | 0/4 | 0/4 |",
              "",
              "## Que cambio vs el baseline denso (2/12)",
              "",
              "Mismo puntaje, distinta composicion: el hibrido acierta Q1 y Q11; "
              "el denso acertaba Q6 y Q11. Un flip a favor, un flip en contra, "
              "y las otras 10 preguntas igual.",
              "",
              "- **Q1: denso incorrecta -> hibrido correcta (unico flip a favor).** "
              "El hibrido trajo el 11.2% al top-5 (paginas [88,113,16,112,115]) "
              "y el lector eligio 781,585 en vez de los 40,236 del denso. Q1 "
              "en el denso era fallo del lector, no de retrieval (781,585 si "
              "estaba en contexto); el hibrido lo arreglo de rebote, al subir "
              "la pagina con el 11.2%.",
              "- **Q6: denso correcta -> hibrido incorrecta (flip en contra).** "
              "El contexto hibrido ([88,112,79,27,107] vs [30,114,10,27,112] "
              "del denso) cambio la tabla que vio el lector y tomo 128,736 como "
              "ingreso de Coca-Cola FEMSA. 128,736 es una cifra real del informe "
              "—su utilidad bruta, no su ingreso (279,793)—, asi que no cuenta "
              "como alucinacion, pero la respuesta esta mal. El hibrido arreglo "
              "el retrieval de Q1 y rompió el de Q6.",
              "- **Q3, Q4, Q10: igual de mal.** Los 3 fallos de retrieval del "
              "denso siguen fallando: 23,206 solo esta en la p.21, 571 en pp. "
              "21/114/126 y 'FEMSA Forward' en pp. 3/5/8/12/14/42/101/115, y "
              "ninguna de esas paginas entro al top-5 hibrido. BM25 no puede "
              "emparejar lo que no esta en la pregunta: '571' y 'Forward' no "
              "aparecen en las preguntas, y 'proximidad'/'estaciones' estan en "
              "espanol mientras el corpus esta en ingles.",
              "- **Q5: peor retrieval, mejor conducta.** El denso SI traia el "
              "14.2 en top-5 y el lector divago (alucinacion '114'); el hibrido "
              "ya no trae el 14.2 y el lector se abstuvo honestamente ('No lo "
              "se'). Sigue incorrecta, pero sin alucinar.",
              "- **Q12: peor retrieval.** El denso traia un 5.8 literal (margen "
              "de utilidad neta, espurio, no el crecimiento de Salud); el "
              "hibrido ni ese trae. El lector se abstuvo en ambas.",
              "- **Q9: mejor conducta.** El denso invento la suma 43,265; el "
              "hibrido se abstuvo. Sigue incorrecta (las cifras no existen en "
              "el informe), pero sin alucinar.",
              "- **Q8: peor lector.** El denso listo Mexico y Colombia (2 de 4); "
              "el hibrido, con los 4 paises en contexto, listo solo Mexico y "
              "'America Latina'.",
              "- **Q7: peor conducta.** El denso contesto solo direccion y "
              "absolutos (24.6 vs 8.6) y paro ahi, sin alucinar; el hibrido "
              "siguio e invento 'aumento un 275% (24.6 - 8.6 = 16)' y 'crecio "
              "un 50%'. El 16 cuenta como alucinacion. Sigue incorrecta en "
              "ambas (los porcentajes no estan en el informe).",
              "- **Sin cambio:** Q2 (cifra fuera del corpus), Q11 "
              "(correcta en ambas).",
              "- **Cobertura literal ALL en top-5:** denso 2/12 (Q5, Q12) -> "
              "hibrido 1/12 (Q1). El hibrido gano Q1 y perdio Q5 y Q12.",
              "",
              "## Veredicto para el post",
              "",
              "**RAG hibrido BM25+denso: 2/12 — el mismo puntaje que el denso "
              "puro, con un perfil distinto.** No cerro la brecha de retrieval: "
              "de los 4 fallos de retrieval del denso (Q3, Q4, Q10, Q12), el "
              "hibrido no arreglo ninguno, y perdio Q5. Lo que arreglo fue Q1, "
              "que en el denso era fallo del lector (781,585 si estaba en "
              "contexto): el hibrido ademas trajo el 11.2% y el lector eligio "
              "bien. BM25 reordena, pero no puede arreglar un desajuste de "
              "vocabulario: las preguntas estan en espanol, el corpus en "
              "ingles, y los tokens discriminativos viven en la respuesta, no "
              "en la pregunta. El arreglo directo seria traducir o expandir la "
              "pregunta con sinonimos del dominio, o chunkear por secciones en "
              "vez de por pagina — no otro ranker lexico.",
              "",
              "Lo que SI cambio es la conducta, en ambas direcciones: "
              "alucinaciones de 2/12 (denso: Q5 '114', Q9 '43265') a 1/12 "
              "(hibrido: Q7 '16'). Q5 y Q9 ya no alucinan — el lector se "
              "abstiene — pero Q7 ahora inventa porcentajes ('aumento un "
              "275%', 'crecio un 50%') que el denso no inventaba. El mismo "
              "lector, con distinto contexto, divaga distinto. Abstenerse no "
              "suma puntos con este scorer, asi que el marcador no se mueve. "
              "La moraleja se mantiene y se afina: la exactitud de cifras "
              "exige que el dato llegue verificado al lector; cambiar el "
              "ranker sin cambiar el vocabulario de la pregunta es mover los "
              "muebles.",
              "",
              "## Caveats (verificados, no supuestos)",
              "",
              "- El diagnostico de retrieval con tolerancia de redondeo del "
              "scorer (±0.1%) decia 5/12 con todas las cifras cubiertas; la "
              "cobertura literal estricta en top-5 es 1/12 (Q1) — la "
              "tolerancia marca 'cubierto' un 27.5% cuando solo hay un 27 "
              "suelto, igual que en el baseline denso. La tabla de detalle "
              "usa cobertura literal; la clasificacion de fallos se verifico "
              "a mano. En Q6 las cifras estan como valores exactos "
              "(307,197/279,793) mientras el campo esperado trae los "
              "redondeados de la ontologia (307,200/279,800).",
              "- El hibrido NO cerro la brecha de retrieval del denso: Q3, "
              "Q4 y Q10 siguen sin traer la cifra al top-5. La razon es "
              "estructural y se verifico en los rankings: las preguntas estan "
              "en espanol y el corpus en ingles, asi que BM25 solo empareja "
              "tokens exactos (numeros, nombres propios) y los tokens "
              "discriminativos viven en la respuesta, no en la pregunta "
              "('571' no esta en la pregunta de Q4; 'Forward' no esta en la "
              "de Q10; '23,206' no esta en la de Q3). BM25 reordena, pero no "
              "puede inventar el vocabulario que falta. El arreglo directo "
              "seria traducir la pregunta o expandirla con sinonimos del "
              "dominio, no otro ranker lexico.",
              "- El condensado de contexto (ventanas de ~1,800 caracteres por "
              "chunk, 2 ventanas) fue necesario porque el prompt de 5 paginas "
              "completas (~4.5k tokens) tarda >6 min por pregunta en CPU. "
              "Es el mismo workaround que en el baseline denso, con el mismo "
              "costo medido: cifras que estan en el top-5 pueden no llegar "
              "al lector. Es un costo del workaround CPU, no del metodo RAG "
              "en si.",
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
