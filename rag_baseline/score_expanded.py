#!/usr/bin/env python3
"""RAG hibrido + expansion de queries (FEMSA 2024) — Paso 3: scoring.

Misma logica que rag_baseline/score_hybrid.py (helpers de memo/evaluate.py,
misma regla Q3 con 1,596, mismas keywords Q8/Q10/Q11), asi que el puntaje
expandido es directamente comparable con el hibrido (2/12) y el denso puro (2/12). La unica
diferencia entre esta corrida y el hibrido es la QUERY de retrieval: la expanded query (traduccion EN + keywords de dominio) en vez de la pregunta original en espanol. Mismo retrieval: RRF k=60 sobre BM25
Okapi + MiniLM, mismo reader (Qwen2.5-1.5B), mismo prompt, mismo
condense. VERIFIED_MODES se lleno a mano DESPUES de la corrida,
verificando cada fallo contra el corpus (igual que en score.py).

Uso:
  memo/.venv/bin/python rag_baseline/score_expanded.py [--trace rag_baseline/run_trace_expanded.jsonl]
                                             [--retrieval rag_baseline/retrieval_expanded.json]
                                             [--out rag_baseline/results_expanded.md]
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
    ap.add_argument("--trace", default="rag_baseline/run_trace_expanded.jsonl")
    ap.add_argument("--retrieval", default="rag_baseline/retrieval_expanded.json")
    ap.add_argument("--eval", default="memo/eval_questions.json")
    ap.add_argument("--qa", default="memo/reflection_qa.jsonl")
    ap.add_argument("--seed", default="pipeline/triples_seed.json")
    ap.add_argument("--chunks", default="pipeline/data/chunks.jsonl")
    ap.add_argument("--out", default="rag_baseline/results_expanded.md")
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
        # score.py y score_hybrid.py: la heuristica automatica etiqueta mal
        # varios casos). NO correr el scorer con los PENDIENTE sin llenar.
        # Se llena en un segundo paso, tras leer run_trace_expanded.jsonl.
        VERIFIED_MODES = {
            "Q1": "ok",
            "Q2": "cifra esperada fuera del corpus (+24.9% no existe en el informe; el lector contesto bien lo preguntado)",
            "Q3": "fallo del lector (23,206 estaba en top-5 [114,19,22,112,21] y en el contexto condensado; el lector eligio 26,322, el total de la division, y '438 nuevas tiendas'; 1,596 no llego al contexto por el condense)",
            "Q4": "ok",
            "Q5": "fallo del lector (14.2 en top-5 y en contexto; divago y se corto a los 100 tokens; '113' es numero de pagina)",
            "Q6": "ok",
            "Q7": "cifra esperada fuera del corpus + conflicto de ground-truth (ver nota: el informe p.34 dice Spin by OXXO +32.8% y Spin Premia +31.5%; el lector dio exactamente esas cifras con la atribucion correcta; los +27.5%/+24.9% esperados son del comunicado 4Q24, no del informe)",
            "Q8": "ok",
            "Q9": "cifra esperada fuera del corpus + atribucion erronea (39.3/35.8/75.1% no existen; contesto 11.2%, que es el crecimiento consolidado, no la participacion)",
            "Q10": "ok",
            "Q11": "ok por keywords pero con alucinacion segun el scorer ('328' = 32.8% real del informe p.34 sobre usuarios de Spin by OXXO; la respuesta se corto a los 100 tokens)",
            "Q12": "fallo del lector (5.8% en top-5 [29,112,114,30,26] y en contexto; contesto 14.2%, que es el crecimiento de Proximidad Europa/Coca-Cola FEMSA, no de Salud)",
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

    lines = ["# Resultados RAG hibrido + expansion de queries (BM25 + denso, RRF k=60) — FEMSA 2024",
             "",
             "Sistema: retrieval hibrido top-5 paginas — BM25 Okapi (rank_bm25, "
             "k1=1.5, b=0.75) + all-MiniLM-L6-v2, fusion Reciprocal Rank Fusion "
             "k=60 — sobre las 151 paginas del informe, con queries expandidas "
             "(traduccion manual al ingles + keywords de dominio, ver "
             "expand_queries.py; sin cifras esperadas ni nombres-respuesta). "
             "Reader Qwen2.5-1.5B-Instruct local (bf16, CPU), mismo prompt y "
             "mismo condense que el baseline denso. Scorer identico a "
             "memo/evaluate.py (misma exactitud numerica, misma definicion de "
             "alucinacion, misma regla Q3 con 1,596). La unica diferencia con "
             "el hibrido es la QUERY de retrieval.",
             "",
             f"Correctas: **{total_ok}/12**. Con alucinacion (numero no verificado en el informe): **{total_halluc}/12**. "
             f"Retrieval top-5 con todas las cifras esperadas literales: **{n_retr_full}/12** "
             f"(hibrido sin expansion: 1/12; denso puro: 2/12; ver Caveats).",
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
              f"| RAG hibrido + expansion de queries (este run) | {total_ok}/12 | "
              f"{sum(1 for r in by_level[1] if r['verdict'] == 'correcta')}/4 | "
              f"{sum(1 for r in by_level[2] if r['verdict'] == 'correcta')}/4 | "
              f"{sum(1 for r in by_level[3] if r['verdict'] == 'correcta')}/4 |",
              "| RAG hibrido BM25+denso (sin expansion) | 2/12 | 1/4 | 0/4 | 1/4 |",
              "| RAG vectorial denso puro | 2/12 | 0/4 | 1/4 | 1/4 |",
              "| MEMO run 2 (0.5B LoRA r=32 + ejecutivo 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |",
              "| MEMO run 1 (0.5B LoRA r=8 + ejecutivo 0.5B) | 0/12 | 0/4 | 0/4 | 0/4 |",
              "",
              "## Que cambio vs el hibrido sin expansion (2/12)",
              "",
              "De 2/12 a 5/12: cuatro flips a favor y uno en contra (por el "
              "scorer, no por la sustancia). El hibrido acertaba Q1 y Q11; "
              "el expandido acierta Q1, Q4, Q6, Q8 y Q10, y Q11 queda como "
              "'correcta con alucinacion' (ver nota).",
              "",
              "- **Q4: hibrido incorrecta -> expandida correcta (flip a favor, "
              "retrieval).** La query expandida ('OXXO GAS stations gas "
              "stations') metio la p.21 en el top-5 ([21,114,85,82,100]) y "
              "el lector contesto 571. En el hibrido la p.21 no entraba y "
              "el lector contesto '12 estaciones'.",
              "- **Q6: hibrido incorrecta -> expandida correcta (flip a favor).** "
              "El lector dio 307,197 y 279,793 exactos. En el hibrido el "
              "contexto lo hizo tomar 128,736 (utilidad bruta) como ingreso "
              "de Coca-Cola FEMSA.",
              "- **Q8: hibrido incorrecta -> expandida correcta (flip a favor, "
              "con caveat).** Los 4 paises estaban en contexto y el lector "
              "los listo (Mexico, Colombia, Chile, Peru), asi que la regla "
              "de keywords marca correcta. PERO enumero 18 paises del grupo "
              "FEMSA (incluidos Austria, Espana, Paises Bajos), no solo los "
              "4 de Proximidad Americas: la regla es indulgente aqui (ver "
              "Caveats).",
              "- **Q10: hibrido incorrecta -> expandida correcta (flip a favor, "
              "retrieval).** La query expandida ('strategy Retail Beverages "
              "digital ecosystem') metio la p.8 en el top-5 ([34,13,12,8,4]) "
              "y el lector nombro 'FEMSA Forward'. En el hibrido la p.8 no "
              "entraba y la respuesta era vaga, sin el nombre. Ojo: 'FEMSA "
              "Forward' NO estaba en la query (seria filtrar la respuesta); "
              "entro por 'strategy/Retail/Beverages/digital ecosystem'.",
              "- **Q11: hibrido correcta -> expandida 'correcta con "
              "alucinacion' (flip en contra solo en el conteo).** La "
              "sustancia es la misma (Digital@FEMSA -> Spin) y la keyword "
              "'Spin' esta. El scorer flaggea '328' como alucinacion, pero "
              "32.8% ES una cifra real del informe (p.34, crecimiento de "
              "usuarios de Spin by OXXO): el conjunto de numeros verificados "
              "del scorer (respuestas QA + tripletas) no la contiene. Artefacto "
              "del scorer, no del lector.",
              "- **Q3: igual de mal, distinto modo.** En el hibrido era fallo "
              "de retrieval (el lector se abstuvo); ahora 23,206 ESTA en el "
              "top-5 y en el contexto condensado, y el lector eligio 26,322 "
              "(el total de la division) y '438 nuevas tiendas'. El cuello "
              "de botella se movio del retrieval al lector. 1,596 no llego "
              "al contexto: el condense no incluyo esa linea de la p.112.",
              "- **Q12: igual de mal, distinto modo.** En el hibrido era fallo "
              "de retrieval (el lector se abstuvo); ahora 5.8% ESTA en el "
              "top-5 y en contexto, y el lector contesto 14.2% (el "
              "crecimiento de Proximidad Europa/Coca-Cola FEMSA). Fallo del "
              "lector con el dato enfrente.",
              "- **Q5: igual de mal.** 14.2 estaba en top-5 y en contexto en "
              "las tres corridas; el lector divaga y se corta a los 100 "
              "tokens sin concluir. Aqui el '113' es numero de pagina, igual "
              "que el '114' del denso.",
              "- **Q7: igual de mal por el scorer, mejor por el corpus (ver "
              "nota de ground-truth).** El lector dio 31.5% y 32.8% con la "
              "atribucion correcta segun el informe p.34; los +27.5%/+24.9% "
              "esperados son del comunicado 4Q24 y no estan en el informe. "
              "El scorer marca incorrecta y flaggea 315/328 como alucinacion, "
              "pero ambas cifras estan en el informe.",
              "- **Q9: igual de mal.** Las cifras esperadas (39.3/35.8/75.1%) "
              "no existen; el lector contesto 11.2% (el crecimiento "
              "consolidado) como si fuera la participacion. Peor que "
              "abstenerse (hibrido), mejor que inventar la suma (denso, "
              "43,265).",
              "- **Sin cambio:** Q1 (correcta en ambas), Q2 (cifra fuera del "
              "corpus; '8.6 millones' es correcto para la pregunta).",
              "- **Cobertura literal ALL en top-5:** hibrido 1/12 (Q1) -> "
              "expandido 4/12 (Q1, Q4, Q5, Q12). Con tolerancia del scorer: "
              "6/12. El retrieval mejoro; lo que no mejoro es el lector.",
              "",
              "## Veredicto para el post",
              "",
              "**RAG hibrido + expansion de queries: 5/12 — el primer salto "
              "real del RAG (2/12 -> 5/12).** La expansion cerro 2 de los 4 "
              "fallos de retrieval del hibrido (Q4 y Q10 traen ahora la "
              "pagina con el dato) y de rebote arreglo Q6 y Q8. La razon es "
              "la que el usuario apunto: el problema era el vocabulario de "
              "la pregunta, no el ranker. Traducir al ingles y anadir "
              "keywords de dominio ('stations', 'stores', 'Health', "
              "'strategy', 'digital ecosystem') pone en la query los tokens "
              "que el corpus si tiene.",
              "",
              "Pero el salto deja ver el siguiente cuello de botella con "
              "nitidez: con el dato ya en el top-5, el lector falla en Q3 "
              "(eligio 26,322 teniendo 23,206 enfrente) y en Q12 (contesto "
              "14.2% teniendo 5.8% enfrente), y divaga en Q5 (14.2% "
              "enfrente). De los 7 errores que quedan, 3 son del lector con "
              "el dato en contexto, 3 son cifras esperadas que no estan en "
              "el informe (Q2, Q7, Q9: techo 9/12) y 1 es Q11, correcta en "
              "sustancia pero penalizada por un artefacto del scorer. El "
              "siguiente arreglo no es otro ranker ni mas keywords: es el "
              "lector (o chunks por seccion para que el condense no pierda "
              "la linea, como el 1,596 de Q3).",
              "",
              "Dos honestidades que el post debe llevar: (1) Q8 se cuenta "
              "correcta por la regla de keywords aunque enumera 18 paises "
              "del grupo, no los 4 de la division — la regla es indulgente. "
              "(2) Q7 es correcta segun el informe mismo (p.34: Spin by "
              "OXXO +32.8%, Spin Premia +31.5%) y solo incorrecta segun la "
              "respuesta esperada, que viene del comunicado 4Q24; con una "
              "regla de ground-truth como la de Q3 (1,596), el run seria "
              "6/12. Y el flag de 'alucinacion' de 315/328 es un artefacto: "
              "ambas cifras estan en el informe.",
              "",
              "## Caveats (verificados, no supuestos)",
              "",
              "- El diagnostico de retrieval con tolerancia de redondeo del "
              "scorer (±0.1%) decia 6/12 con todas las cifras cubiertas; la "
              "cobertura literal estricta en top-5 es 4/12 (Q1, Q4, Q5, Q12) "
              "— la tolerancia marca 'cubierto' un 27.5% cuando solo hay un "
              "27 suelto, igual que en el baseline denso y en el hibrido. "
              "La tabla de detalle usa cobertura literal; la clasificacion "
              "de fallos se verifico a mano. En Q6 las cifras estan como "
              "valores exactos (307,197/279,793) mientras el campo esperado "
              "trae los redondeados de la ontologia (307,200/279,800), asi "
              "que la cobertura literal marca False aunque el dato si esta.",
              "- La expansion cerro 2 de los 4 fallos de retrieval del "
              "hibrido: Q4 (571, p.21 en top-5) y Q10 ('FEMSA Forward', p.8 "
              "en top-5). Q3 y Q12 ahora traen la cifra al top-5 (23,206 en "
              "p.21; 5.8% en pp. 112/114) pero el lector falla con el dato "
              "enfrente: el cuello de botella se movio del retrieval al "
              "lector. Q5 traia el 14.2 en las tres corridas y el lector "
              "divaga igual.",
              "- La pregunta original esta en espanol y el corpus en ingles; "
              "la expanded query traduce al ingles y anade keywords de "
              "dominio, asi que el retrieval ya no es cross-lingual. Q12 "
              "exige mapear 'Salud' -> 'Health' (ahora explicito en la "
              "query).",
              "- El condensado de contexto (ventanas de ~1,800 caracteres por "
              "chunk, 2 ventanas) fue necesario porque el prompt de 5 paginas "
              "completas (~4.5k tokens) tarda >6 min por pregunta en CPU. "
              "Es el mismo workaround que en el baseline denso, con el mismo "
              "costo medido: cifras que estan en el top-5 pueden no llegar "
              "al lector (Q3: 1,596 estaba en la p.112 del top-5 y no llego "
              "al contexto). Es un costo del workaround CPU, no del metodo "
              "RAG en si.",
              "- **Q7, conflicto de ground-truth (verificado en el corpus).** "
              "La respuesta esperada (+27.5% Spin Premia vs +24.9% Spin by "
              "OXXO) viene del comunicado de prensa 4Q24 y NO esta en el "
              "informe anual. El informe mismo, p.34, dice que los usuarios "
              "de Spin by OXXO crecieron 32.8% y los de Spin Premia 31.5%. "
              "El lector dio exactamente esas cifras con la atribucion "
              "correcta. El scorer marca incorrecta (no contiene 27.5/24.9) "
              "y flaggea 315/328 como 'alucinacion' porque el conjunto de "
              "numeros verificados se construye de las respuestas QA y las "
              "tripletas semilla, no del corpus completo — pero ambas cifras "
              "estan en el informe. Con una regla como la de Q3 (1,596), Q7 "
              "contaria como correcta y el run seria 6/12. Se deja el "
              "puntaje del scorer (5/12) para comparabilidad y se documenta "
              "la alternativa.",
              "- **Q8, regla de keywords indulgente.** La respuesta enumera "
              "18 paises del grupo FEMSA (Argentina, Austria, Brasil, "
              "Colombia, Costa Rica, Chile, Ecuador, Espana, Guatemala, "
              "Honduras, Mexico, Paises Bajos, Nicaragua, Panama, Peru, "
              "Uruguay, ...) — el footprint del grupo, no los 4 paises de "
              "Proximidad Americas (Mexico, Colombia, Chile, Peru). Como "
              "los 4 estan en la lista, la regla de keywords marca correcta. "
              "Misma regla que en las corridas anteriores, asi que el "
              "puntaje es comparable, pero la respuesta es sobre-inclusiva.",
              "- **Q11, flag de alucinacion artefacto.** El '328' que el "
              "scorer flaggea es el 32.8% real del informe (p.34, "
              "crecimiento de usuarios de Spin by OXXO). La respuesta se "
              "corto a los 100 tokens ('aumentaron un 31.').",
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
