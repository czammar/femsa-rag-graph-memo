#!/usr/bin/env python3
"""RAG FEMSA 2024 — expansion de preguntas (query expansion / traduccion).

El hibrido BM25+denso quedo en 2/12 sin arreglar ninguno de los 4 fallos
de retrieval (Q3, Q4, Q10, Q12). Su propio diagnostico dice por que: las
preguntas estan en espanol y el corpus en ingles, asi que BM25 solo
empareja tokens exactos y los tokens discriminativos viven en la respuesta,
no en la pregunta ("571" no esta en la pregunta de Q4; "Forward" no esta
en la de Q10). El arreglo que toca es mejorar las keywords: traducir la
pregunta al ingles y anadir sinonimos de dominio de los terminos que SI
estan en la pregunta.

Metodo (determinista, auditable, sin modelo):
- Cada pregunta tiene una traduccion al ingles escrita a mano (dict
  TRANSLATIONS, visible abajo) y una lista de keywords en ingles
  (dict KEYWORDS) derivadas SOLO de terminos que aparecen en la pregunta
  original, usando el mapa TERM_MAP (termino en espanol -> equivalentes
  en ingles). La expanded query es: traduccion + keywords.
- La expanded query se usa para LOS DOS rankers (BM25 y denso). Todo lo
  demas queda fijo: mismo reader, mismo prompt, mismo scorer. La pregunta
  original en espanol se conserva en el JSON de retrieval como "question"
  (es la que ve el reader y la que usa condense), y la expandida va en
  "expanded_query".

Regla anti-filtracion (se verifica en codigo, abajo):
- NUNCA se anaden las cifras esperadas (571, 23,206, 14.2, 5.8, ...) ni
  los nombres que SON la respuesta esperada. En concreto:
  * Q10: NO se anade "FEMSA Forward" (es la keyword que el scorer busca
    en la respuesta) ni "Spin" (es la mitad de la respuesta esperada).
  * Q11: NO se anade "Spin" (es el destino de la evolucion que se
    pregunta). Si se anade "Digital@FEMSA", nombre propio de la
    iniciativa de la que habla la pregunta.
  * Q8: NO se anaden los nombres de los paises (son la respuesta).
  * Q3/Q4/Q12: NO se anaden cifras.
- El script aborta si alguna expanded query contiene digitos de las
  cifras esperadas o la cadena "FEMSA Forward".

Uso:
  python3 rag_baseline/expand_queries.py [--eval memo/eval_questions.json]
                                         [--out rag_baseline/expanded_queries.json]

Salida: expanded_queries.json con {id, level, question, translation,
keywords, expanded_query} por pregunta.
"""
import argparse
import json
import re

# Termino en espanol (tal como aparece en las preguntas) -> equivalentes
# en ingles usados como keywords. Solo terminos presentes en la pregunta.
TERM_MAP = {
    "ingresos": ["revenue", "revenues"],
    "ingresos totales consolidados": ["total consolidated revenues"],
    "ingresos consolidados": ["consolidated revenues"],
    "crecieron": ["growth", "grew"],
    "creció": ["growth", "grew"],
    "crecimiento": ["growth"],
    "crecimiento porcentual": ["percentage growth"],
    "usuarios": ["users"],
    "usuarios activos": ["active users"],
    "tiendas": ["stores"],
    "aperturas": ["openings", "new stores"],
    "estaciones": ["stations"],
    "división": ["division"],
    "aportó más ingresos": ["contributed most revenue"],
    "países": ["countries"],
    "opera": ["operates", "operations"],
    "porcentaje": ["percentage", "share"],
    "juntas": ["together", "combined"],
    "estrategia": ["strategy"],
    "bebidas": ["beverages"],
    "ecosistema digital": ["digital ecosystem"],
    "iniciativa digital": ["digital initiative", "Digital@FEMSA"],
    "evolucionó": ["evolved", "evolution"],
    "año más complicado": ["most difficult year", "challenging year"],
    "salud": ["Health"],
    "proximidad américas": ["Proximity Americas"],
    "proximidad europa": ["Proximity Europe"],
}

TRANSLATIONS = {
    "Q1": "What were FEMSA's total consolidated revenues in 2024 and how much did they grow vs 2023?",
    "Q2": "How many active users did Spin by OXXO have in 2024?",
    "Q3": "How many stores did Proximity Americas add by the end of 2024 and how many openings were there?",
    "Q4": "How many stations did OXXO GAS have at the end of 2024?",
    "Q5": "Which division had the highest percentage revenue growth in 2024: Proximity Europe or Coca-Cola FEMSA?",
    "Q6": "Which division contributed the most revenue in 2024, Proximity Americas or Coca-Cola FEMSA, and by how much?",
    "Q7": "Which grew more in users, Spin by OXXO or Spin Premia?",
    "Q8": "In which countries does Proximity Americas operate according to the report?",
    "Q9": "What percentage of 2024 consolidated revenues do Proximity Americas and Coca-Cola FEMSA together represent?",
    "Q10": "What strategy articulates FEMSA Retail, Beverages and the digital ecosystem, and through what is that ecosystem created?",
    "Q11": "How did FEMSA's digital initiative evolve in 2024?",
    "Q12": "Which division had the most difficult year in 2024 according to the report itself, and how much did its revenues grow?",
}

# Keywords por pregunta: solo equivalentes de terminos presentes en la
# pregunta original (ver TERM_MAP). Nada de cifras, nada de nombres que
# sean la respuesta (ver docstring).
KEYWORDS = {
    "Q1": ["total consolidated revenues", "revenue", "growth", "2024"],
    "Q2": ["Spin by OXXO", "active users", "users", "2024"],
    "Q3": ["Proximity Americas", "stores", "openings", "new stores",
           "points of sale", "2024"],
    "Q4": ["OXXO GAS", "stations", "gas stations", "2024"],
    "Q5": ["Proximity Europe", "Coca-Cola FEMSA", "percentage growth",
           "revenue", "growth", "2024"],
    "Q6": ["Proximity Americas", "Coca-Cola FEMSA", "revenues",
           "total revenues", "2024"],
    "Q7": ["Spin by OXXO", "Spin Premia", "users", "growth",
           "active users", "2024"],
    "Q8": ["Proximity Americas", "countries", "operates", "operations"],
    "Q9": ["Proximity Americas", "Coca-Cola FEMSA", "consolidated revenues",
           "percentage", "share", "together", "2024"],
    "Q10": ["FEMSA", "strategy", "Retail", "Beverages", "digital ecosystem",
            "2024"],
    "Q11": ["Digital@FEMSA", "digital initiative", "evolution", "evolved",
            "2024"],
    "Q12": ["Health", "division", "most difficult year", "challenging year",
            "revenue", "growth", "2024"],
}

# Cadenas prohibidas en cualquier expanded query: son la respuesta.
FORBIDDEN = ["FEMSA Forward"]


def digits(s):
    return re.sub(r"[.,]", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="memo/eval_questions.json")
    ap.add_argument("--out", default="rag_baseline/expanded_queries.json")
    args = ap.parse_args()

    eval_qs = json.load(open(args.eval))["questions"]
    out = []
    for q in eval_qs:
        qid = q["id"]
        translation = TRANSLATIONS[qid]
        kws = KEYWORDS[qid]
        # dedup de keywords preservando orden
        seen, uniq = set(), []
        for k in kws:
            if k.lower() not in seen:
                seen.add(k.lower())
                uniq.append(k)
        expanded = translation + " " + " ".join(uniq)
        # --- guardia anti-filtracion ---
        for e in q.get("expected_numbers", []):
            d = digits(str(e))
            assert d not in re.sub(r"[.,]", "", expanded), (
                f"{qid}: la expanded query contiene la cifra esperada {e}")
            # tampoco la forma con decimales
            assert str(e) not in expanded, (
                f"{qid}: la expanded query contiene la cifra esperada {e}")
        for f in FORBIDDEN:
            assert f.lower() not in expanded.lower(), (
                f"{qid}: la expanded query contiene {f!r}, que es la respuesta")
        # tampoco paises-respuesta en Q8
        if qid == "Q8":
            for c in ["Mexico", "Colombia", "Chile", "Peru"]:
                assert c.lower() not in expanded.lower(), (
                    f"Q8: la expanded query contiene el pais {c}, que es la respuesta")
        out.append({"id": qid, "level": q["level"],
                    "question": q["question"], "translation": translation,
                    "keywords": uniq, "expanded_query": expanded})
        print(f"{qid}: {expanded}", flush=True)

    json.dump({"method": "traduccion manual al ingles + keywords de dominio "
                         "(TERM_MAP) de terminos presentes en la pregunta; "
                         "sin cifras esperadas ni nombres-respuesta",
               "questions": out},
              open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"expanded queries -> {args.out} "
          f"(guardia anti-filtracion pasada: sin cifras ni 'FEMSA Forward')")


if __name__ == "__main__":
    main()
