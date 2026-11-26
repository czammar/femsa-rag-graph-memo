#!/usr/bin/env python3
"""RAG agentico (LangChain + LangGraph, ReAct textual) — Paso 3: scoring.

Misma logica que rag_baseline/score_expanded.py (helpers de
memo/evaluate.py, misma exactitud numerica, misma definicion de
alucinacion, misma regla Q3 con 1,596, mismas keywords Q8/Q10/Q11), asi
que el puntaje agentico es directamente comparable con los otros cinco
sistemas. La unica diferencia es el SISTEMA: un agente ReAct con
Qwen2.5-1.5B-Instruct que decide que herramienta usar (vector_search,
graph_query, calculator), con maximo 3 pasos de herramienta por pregunta.
El trace (run_trace_agentic.jsonl) trae los pasos con sus observaciones,
asi que el scorer tambien resume el uso de herramientas por pregunta.
VERIFIED_MODES se llena a mano DESPUES de la corrida, verificando cada
fallo contra el corpus (igual que en score_expanded.py).

Uso:
  agentic_rag/.venv/bin/python agentic_rag/score_agentic.py [--trace agentic_rag/run_trace_agentic.jsonl]
                                                   [--out agentic_rag/results_agentic.md]
"""
import argparse
import json
import re
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
    ap.add_argument("--trace", default="agentic_rag/run_trace_agentic.jsonl")
    ap.add_argument("--eval", default="memo/eval_questions.json")
    ap.add_argument("--qa", default="memo/reflection_qa.jsonl")
    ap.add_argument("--seed", default="pipeline/triples_seed.json")
    ap.add_argument("--out", default="agentic_rag/results_agentic.md")
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

    # Clasificacion del fallo: SE LLENA A MANO DESPUES de la corrida,
    # verificando cada respuesta contra el corpus y contra el trace.
    # NO correr el scorer con los PENDIENTE sin llenar.
    VERIFIED_MODES = {
        "Q1": "abstencion tras 3 consultas fallidas (nombre de indicador inventado: 'Ingresos Totales Consolidados' en vez de 'Ingresos Totales'; repitio la misma consulta 3 veces pese a la guardia anti-bucle)",
        "Q2": "abstencion tras 2 consultas fallidas ('Spin by OXXO' tratado como Division con OPERA_EN y propiedad inventada 's.multiples_operadoras'; es un Formato de la Division Spin)",
        "Q3": "evidencia mal interpretada (conto las 4 filas OPERA_EN —paises— como 4 tiendas con 1 apertura cada una; la cifra real es 23,206 tiendas y 1,596 aperturas netas)",
        "Q4": "PENDIENTE",
        "Q5": "sin ninguna llamada a herramienta (5 planes con Cypher-SQL gigante truncado a los 150 tokens, relacion inventada 'OPERA_IN'; el modelo intenta una sola consulta comparativa en vez de dos simples)",
        "Q6": "abstencion tras 3 consultas fallidas (SUM(r.valor) sobre OPERA_EN, que no tiene valor: 0 en Proximidad Americas; repitio la misma consulta para Coca-Cola FEMSA)",
        "Q7": "abstencion tras 3 consultas fallidas ('Spin by OXXO' como Iniciativa con OPERA_EN y r.valor; ademas las cifras esperadas +27.5%/+24.9% no estan en el informe: techo 9/12)",
        "Q8": "PENDIENTE",
        "Q9": "sin ninguna llamada a herramienta (5 planes con SQL anidado en calculator, truncado a los 150 tokens; ademas las cifras esperadas 39.3/35.8/75.1% no estan en el informe: techo 9/12)",
        "Q10": "abstencion tras 3 consultas fallidas (tipo de nodo inventado 'EcosistemaDigital', 'Retail' como Iniciativa y 'Bebidas' como Division; no descubrio 'FEMSA Forward')",
        "Q11": "evidencia correcta ignorada (la primera consulta devolvio {\"x.nombre\":\"Spin\"} y el agente repitio la misma consulta 2 veces mas y se abstuvo)",
        "Q12": "abstencion tras 3 consultas fallidas (error de sintaxis Cypher: 'LIMIT 1, 1' estilo SQL + ORDER BY sobre r.variacion_pct de OPERA_EN, que no tiene esa propiedad)",
    }

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
        # evidencia vista por el agente: observaciones de todos los pasos
        evidence = "\n".join(str(s.get("observation", ""))
                             for s in r.get("steps", []))
        cov_evidence = {e: literal_present(evidence, e) for e in exp_nums}
        tools_used = [s["tool"] for s in r.get("steps", [])]
        # Un plan "fallido" es una llamada al plan que no produjo ni una
        # accion valida ni la RESPUESTA final (p. ej. Cypher/SQL gigante
        # truncado a los 150 tokens, sin cerrar comillas). Q5 y Q9 queman
        # 5 cada una: el modelo no llega a ejecutar ninguna herramienta.
        raws = r.get("plan_raw_outputs", [])
        last_is_answer = bool(raws) and raws[-1].lstrip().startswith("RESPUESTA:")
        failed_plans = len(raws) - len(tools_used) - (1 if last_is_answer else 0)
        mode = VERIFIED_MODES[r["id"]]
        if verdict == "correcta":
            mode = "ok"
        rows.append({"id": r["id"], "level": q["level"],
                     "question": q["question"], "expected": q["expected"],
                     "answer": ans, "verdict": verdict,
                     "hallucinated": halluc, "note": note,
                     "tools": tools_used,
                     "n_steps": len(tools_used),
                     "n_plan_calls": len(raws),
                     "failed_plans": max(failed_plans, 0),
                     "seconds": r.get("seconds", 0),
                     "coverage": cov_evidence,
                     "failure_mode": mode})

    by_level = {}
    for row in rows:
        by_level.setdefault(row["level"], []).append(row)
    total_ok = sum(1 for r in rows if r["verdict"] == "correcta")
    total_halluc = sum(1 for r in rows if r["hallucinated"])
    n_ev_full = sum(1 for r in rows if r["coverage"]
                    and all(r["coverage"].values()))
    tool_counter = Counter(t for r in rows for t in r["tools"])
    total_steps = sum(r["n_steps"] for r in rows)
    total_plan_calls = sum(r["n_plan_calls"] for r in rows)
    total_failed_plans = sum(r["failed_plans"] for r in rows)
    total_seconds = sum(r["seconds"] for r in rows)
    pending = [r["id"] for r in rows if r["failure_mode"] == "PENDIENTE"]
    if pending:
        print(f"AVISO: modos de fallo PENDIENTES de verificar a mano: {pending}")

    def lvl_ok(lv):
        return sum(1 for r in by_level.get(lv, [])
                   if r["verdict"] == "correcta")

    lines = ["# Resultados RAG agentico (LangChain + LangGraph, ReAct textual) — FEMSA 2024",
             "",
             "Sistema: agente ReAct con LangGraph (nodos plan -> act -> "
             "observe -> answer), modelo local Qwen2.5-1.5B-Instruct (bf16, "
             "CPU), maximo 3 pasos de herramienta por pregunta. Herramientas: "
             "`vector_search` (retrieval hibrido BM25 + MiniLM, RRF k=60, "
             "top-5 paginas), `graph_query` (Cypher contra el store Relate "
             "femsa.rl) y `calculator` (aritmetica segura por AST). El modelo "
             "de 1.5B no emite tool calls nativos de forma fiable (verificado "
             "en test_toolcall.py: cero tool_calls con bind_tools), asi que el "
             "agente usa un bucle ReAct textual: el modelo escribe "
             "`ACCION: herramienta(\"argumento\")` o `RESPUESTA: <texto>` y el "
             "nodo act lo ejecuta. Scorer identico a memo/evaluate.py (misma "
             "exactitud numerica, misma definicion de alucinacion, misma "
             "regla Q3 con 1,596).",
             "",
             f"Correctas: **{total_ok}/12**. Con alucinacion (numero no verificado en el informe): **{total_halluc}/12**. "
             f"Preguntas donde la evidencia vista por el agente contenia todas "
             f"las cifras esperadas (literal): **{n_ev_full}/12**. "
             f"Pasos de herramienta: **{total_steps}** en total "
             f"({tool_counter.get('vector_search', 0)} vector_search, "
             f"{tool_counter.get('graph_query', 0)} graph_query, "
             f"{tool_counter.get('calculator', 0)} calculator, "
             f"{tool_counter.get('parse_error', 0)} errores de formato). "
             f"Llamadas al plan: **{total_plan_calls}**, de las que "
             f"**{total_failed_plans} no llegaron a ninguna accion ejecutada "
             f"ni a la RESPUESTA final** (5 en Q5 y 5 en Q9: consultas "
             f"gigantes truncadas a los 150 tokens; 1 en Q6, Q7 y Q12: una "
             f"4ª ACCION identica que choco con el tope de 3 pasos). "
             f"Tiempo total de la corrida: **{total_seconds / 60:.1f} min** "
             f"(media {total_seconds / max(len(rows), 1):.0f} s por pregunta).",
             "",
             "| ID | Nivel | Pregunta | Esperado | Respuesta del agente | Veredicto | Modo de fallo |",
             "|----|-------|----------|----------|----------------------|-----------|---------------|"]
    for r in rows:
        a = r["answer"].replace("|", "/").replace("\n", " ")[:200]
        lines.append(f"| {r['id']} | {r['level']} | {r['question'][:60]} | "
                     f"{r['expected'][:55]} | {a} | {r['verdict']} | {r['failure_mode']} |")
    lines += ["",
              "## Por nivel",
              ""]
    for lv in sorted(by_level):
        lines.append(f"- Nivel {lv}: {lvl_ok(lv)}/{len(by_level[lv])} correctas.")
    lines += ["",
              "## Uso de herramientas por pregunta",
              "",
              "| ID | Pasos | Herramientas (en orden) | Llamadas al plan | Segundos |",
              "|----|-------|------------------------|------------------|----------|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['n_steps']} | {', '.join(r['tools']) or '(ninguna)'} | "
                     f"{r['n_plan_calls']} | {r['seconds']:.0f} |")
    lines += ["",
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
              f"| RAG agentico (este run) | {total_ok}/12 | {lvl_ok(1)}/4 | {lvl_ok(2)}/4 | {lvl_ok(3)}/4 |",
              "| RAG hibrido + expansion de queries | 5/12 | 2/4 | 2/4 | 1/4 |",
              "| RAG hibrido BM25+denso (sin expansion) | 2/12 | 1/4 | 0/4 | 1/4 |",
              "| RAG vectorial denso puro | 2/12 | 0/4 | 1/4 | 1/4 |",
              "| MEMO run 2 (0.5B LoRA r=32 + ejecutivo 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |",
              "| MEMO run 1 (0.5B LoRA r=8 + ejecutivo 0.5B) | 0/12 | 0/4 | 0/4 | 0/4 |",
              "",
              "## Que cambio vs el RAG hibrido + expansion (5/12)",
              "",
              "De 5/12 a 2/12: tres flips en contra y ninguno a favor. El "
              "expandido acierta Q1, Q4, Q6, Q8 y Q10; el agentico solo Q4 y "
              "Q8 (las dos que acierta son las dos consultas de plantilla "
              "directa al grafo).",
              "",
              "- **Q1: expansion correcta -> agentico incorrecta (flip en "
              "contra, nombre de entidad).** El agente escribio "
              "'Ingresos Totales Consolidados' en vez de 'Ingresos Totales' "
              "y repitio la misma consulta 3 veces. El lector RAG, con el "
              "dato en contexto, contesto 781,585.",
              "- **Q6: expansion correcta -> agentico incorrecta (flip en "
              "contra, relacion equivocada).** El agente hizo SUM(r.valor) "
              "sobre OPERA_EN (que no tiene valor) y obtuvo 0; el lector "
              "RAG dio 307,197 y 279,793 exactos desde el contexto.",
              "- **Q10: expansion correcta -> agentico incorrecta (flip en "
              "contra, esquema inventado).** El agente invento el tipo de "
              "nodo 'EcosistemaDigital' y trato 'Retail' como Iniciativa; "
              "no descubrio 'FEMSA Forward'. El lector RAG lo nombro desde "
              "la p.8 del informe.",
              "- **Sin cambio:** Q4 (correcta en ambas: 571 estaciones) y Q8 "
              "(correcta en ambas: los 4 paises). Q2, Q3, Q5, Q7, Q9, Q11 y "
              "Q12 incorrectas en ambas (Q3 y Q12 por el lector en el RAG y "
              "por el agente aqui; Q2, Q7 y Q9 por cifras fuera del corpus "
              "en ambas).",
              "- **Q11, el caso mas revelador:** la primera consulta del "
              "agente devolvio {\"x.nombre\":\"Spin\"} — la keyword que el "
              "scorer pide — y el agente repitio la misma consulta 2 veces "
              "mas y contesto 'No lo se'. Tenia la respuesta en la mano y "
              "no supo parar.",
              "- **Uso de herramientas:** el agente nunca uso vector_search "
              "(0 veces) ni calculator (0 veces) pese a la instruccion de "
              "empezar siempre por vector_search y de calcular con la "
              "herramienta. Las 24 llamadas fueron graph_query, y 13 "
              "llamadas al plan no llegaron a ejecutarse (5 en Q5 y 5 en "
              "Q9 sin accion valida; 1 en Q6, Q7 y Q12 bloqueada por el "
              "tope de 3 pasos).",
              "",
              "## Veredicto para el post",
              "",
              "**RAG agentico: 2/12 — el bucle agentico no suma con un "
              "modelo de 1.5B; resta.** Con el mismo modelo, el mismo "
              "retrieval y el mismo scorer, el pipeline fijo con expansion "
              "saca 5/12 y el agente 2/12. La diferencia no es el acceso a "
              "los datos — el agente tiene el grafo exacto, que solo "
              "contesta 12/12, y el retrieval hibrido — sino la capacidad "
              "de *usar* las herramientas: componer Cypher sin mezclar "
              "relaciones, corregir un nombre tras un 'Sin resultados', "
              "parar cuando la evidencia ya esta y no repetir la misma "
              "llamada. El 1.5B falla en las cuatro.",
              "",
              "Los dos aciertos (Q4, Q8) son consultas de plantilla "
              "copiadas al pie de la letra; todo lo que exige componer "
              "(comparar dos divisiones, calcular un porcentaje, descubrir "
              "un nombre) se cae. Las preguntas de comparacion/agregacion "
              "(Q5, Q9) son las peores: el modelo intenta una sola consulta "
              "gigante con sintaxis SQL anidada, se trunca a los 150 tokens "
              "y quema los 5 intentos sin ejecutar nada (653 s y 633 s de "
              "nada). Y el fallo mas caro es Q11: la evidencia correcta "
              "llego en la primera llamada y el agente no supo parar.",
              "",
              "La lectura honesta para el post: 'agentico' no es una "
              "propiedad del bucle, es una propiedad del modelo que lo "
              "maneja. Con Qwen2.5-1.5B el agente es un RAG peor: mas lento "
              "(65 min vs ~26 min del expandido), mas fragil (una "
              "instruccion ignorada — vector_search primero — y un bug de "
              "router que tumbo la corrida en Q5) y con menos aciertos. El "
              "grafo 12/12 muestra que el dato esta; el agente muestra que "
              "llegar a el exige un modelo que sepa planificar, y el 1.5B "
              "no es ese modelo. El siguiente arreglo no es mas "
              "herramientas ni mas pasos: es un agente mas capaz, o un "
              "agente con menos grados de libertad (plantillas cerradas, "
              "verificacion automatica de la consulta antes de ejecutarla).",
              "",
              "## Caveats (verificados, no supuestos)",
              "",
              "- **El agente nunca uso vector_search ni calculator.** Las 24 "
              "llamadas fueron graph_query. La instruccion 'Empieza SIEMPRE "
              "con vector_search' fue ignorada en las 12 preguntas: el "
              "modelo va directo al grafo cuando la pregunta pide una cifra. "
              "El experimento, por tanto, mide un agente *solo-grafo*, no "
              "el bucle completo con sus tres herramientas.",
              "- **Q11 tenia la respuesta y la perdio.** La primera consulta "
              "devolvio {\"x.nombre\":\"Spin\"}; el scorer pide la keyword "
              "'Spin'. El agente repitio la consulta y se abstuvo. Con una "
              "regla de parada ('si la evidencia responde, contesta'), Q11 "
              "seria correcta y el run 3/12.",
              "- **Q5 tumbo la corrida por un bug del router.** `_route` "
              "podia devolver 'plan' pero el mapa de "
              "`add_conditional_edges` no incluia 'plan': KeyError en Q5. "
              "Se arreglo anadiendo 'plan': 'plan' al mapa y un try/except "
              "por pregunta en run_agentic.py; la corrida se reanudo desde "
              "Q5 sin tocar Q1–Q4. El bug es del codigo, no del modelo, y "
              "esta documentado en `agent.py` y en el README.",
              "- **La guardia anti-bucle tiene un agujero de espacios.** En "
              "Q2 la segunda llamada era identica salvo un espacio final y "
              "la guardia (comparacion exacta) no la detecto. Normalizar "
              "con strip() lo cerraria; se deja como esta para no cambiar el "
              "agente a mitad de corrida.",
              "- **Q1 tardo 866 s por contencion de CPU** (otro agente corria "
              "un modelo en paralelo durante la primera parte). Sin "
              "contencion, las preguntas tardan 88–290 s (Q5 y Q9, con 5 "
              "planes fallidos, 653 s y 633 s). El tiempo total de "
              "preguntas es 65.3 min; el wall total con la caida y la "
              "reanudacion, ~69 min.",
              "- **Q7 y Q9 son techo (9/12), igual que en los otros runs.** "
              "Las cifras esperadas de Q7 (+27.5%/+24.9%) vienen del "
              "comunicado 4Q24 y las de Q9 (39.3/35.8/75.1%) son "
              "participaciones calculadas: no estan en el informe. Q2 "
              "tambien pide +24.9% como parte de lo esperado. El techo real "
              "del agente sobre este set es 9/12, no 12/12.",
              "- **Q3 es incorrecta por interpretacion, no por abstencion.** "
              "El agente conto 4 paises como 4 tiendas. Es el unico error "
              "con respuesta inventada (las demas son abstenciones): 0/12 "
              "con alucinacion segun el scorer porque '4' es un solo "
              "digito, pero la respuesta es falsa (23,206 tiendas, 1,596 "
              "aperturas).",
              "- **Las plantillas funcionan cuando se copian.** Q4 y Q8 son "
              "copias literales de la plantilla de metrica operativa y de "
              "paises. Los fallos son desviaciones de la plantilla "
              "(relacion, nombre o propiedad cambiados) o intentos de "
              "consulta comparativa gigante. El diseno de 'plantillas "
              "cerradas' es el arreglo mas barato que el post puede "
              "proponer.",
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
              "## Detalle de evidencia por pregunta (cobertura literal en lo que vio el agente)",
              "",
              "| ID | Herramientas | Cifras esperadas en la evidencia |",
              "|----|--------------|----------------------------------|"]
    for r in rows:
        cov_s = ", ".join(f"{k}: {'si' if v else 'no'}"
                          for k, v in r["coverage"].items()) \
            if r["coverage"] else "(sin cifras esperadas)"
        lines.append(f"| {r['id']} | {', '.join(r['tools'])} | {cov_s} |")

    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"correctas {total_ok}/12, con alucinacion {total_halluc}/12, "
          f"pasos {total_steps}, {total_seconds / 60:.1f} min -> {args.out}")
    for r in rows:
        print(f"  {r['id']} L{r['level']} {r['verdict']} [{r['failure_mode']}] "
              f"tools={r['tools']} {r['seconds']:.0f}s"
              + (f" halluc={r['hallucinated']}" if r["hallucinated"] else ""))


if __name__ == "__main__":
    main()
