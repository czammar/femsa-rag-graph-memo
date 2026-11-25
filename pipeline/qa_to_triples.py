#!/usr/bin/env python3
"""Convierte pares QA de reflexion (MEMO) en tripletas candidatas para el grafo.

Idea (del usuario): el dataset de reflexiones ya contiene los hechos del informe
en forma de pregunta/respuesta; este conversor los descompone de vuelta a tripletas
de la ontologia cerrada, para alimentar el knowledge graph sin re-extraer del PDF.

Reglas duras (las mismas del pipeline):
  - Ontologia CERRADA: 9 tipos de entidad, 8 relaciones (ver ONTOLOGY abajo, que
    replica docs/ontology-and-questions.md). Nada fuera de ella.
  - Cada tripleta lleva su pagina fuente, tomada del par QA.
  - Las preguntas de tipo "aggregation" NO se fuerzan a una tripleta: van a la
    lista separada pipeline/needs_computed_path.json (necesitan un calculo, no
    un hecho). Las de tipo "comparison"/"multi-hop" se descomponen en sus
    tripletas componentes (cada hecho por separado).
  - Salida en el MISMO formato que pipeline/extracted_triples.jsonl.
  - Todo lo emitido queda marcado "pendiente_revision": NUNCA se carga al grafo
    sin revision humana (misma regla que extracted_triples.jsonl).

Uso:
  python3 pipeline/qa_to_triples.py [--qa memo/reflection_qa.jsonl]
"""
import argparse
import json
import re
import unicodedata

DOCUMENTO = ("FEMSA 2024 Integrated Annual Report, «Evolving» "
             "(English edition, same figures)")

# --- Ontologia cerrada (replica de docs/ontology-and-questions.md) ---
ENTITY_TYPES = {"Empresa", "Division", "Formato", "IndicadorFinanciero",
                "MetricaOperativa", "Periodo", "Pais", "Iniciativa", "Riesgo"}
RELATIONS = {"tiene_division", "tiene_formato", "opera_en", "reporta",
             "mide", "crecio_en", "impulsa", "evoluciono_a"}

# Entidades conocidas (nombres canonicos del informe) con su tipo.
KNOWN = {
    "FEMSA": "Empresa",
    "Coca-Cola FEMSA": "Division",
    "Proximidad Américas": "Division",
    "Proximidad Europa": "Division",
    "Salud": "Division",
    "Combustibles": "Division",
    "Spin": "Division",
    "OXXO México": "Formato",
    "OXXO GAS": "Formato",
    "Spin by OXXO": "Formato",
    "Spin Premia": "Formato",
    "Juntos+": "Formato",
    "OXXO": "Formato",
    "Bara": "Formato",
    "OXXO SMART": "Formato",
    "OXXO Nicho": "Formato",
    "FEMSA Forward": "Iniciativa",
    "Digital@FEMSA": "Iniciativa",
    "Retina Media": "Iniciativa",
    "Huellas Mayores": "Iniciativa",
    "Juntos+ Advisor": "Formato",
    "Premia Juntos+": "Formato",
    "México": "Pais", "Colombia": "Pais", "Chile": "Pais", "Perú": "Pais",
    "Brasil": "Pais", "Estados Unidos": "Pais", "Suiza": "Pais",
    "Alemania": "Pais", "Ecuador": "Pais", "Argentina": "Pais",
    "Guatemala": "Pais", "Luxemburgo": "Pais", "Países Bajos": "Pais",
    "Austria": "Pais", "Uruguay": "Pais",
}
DIVISIONS = ["Coca-Cola FEMSA", "Proximidad Américas", "Proximidad Europa",
             "Salud", "Combustibles", "Spin"]
FINANCIAL_INDICATORS = [
    ("Ingresos Totales", [r"ingresos?\s+totales?", r"total\s+revenues?"]),
    ("Utilidad de Operación", [r"utilidad\s+de\s+operaci[oó]n", r"income\s+from\s+operations?"]),
    ("Utilidad Bruta", [r"utilidad\s+bruta", r"gross\s+profit"]),
    ("EBITDA", [r"\bebitda\b"]),
    ("Utilidad Neta", [r"utilidad\s+neta", r"net\s+income"]),
    ("Ventas Mismas-Tiendas", [r"ventas\s+mismas[- ]tiendas", r"same[- ]store\s+sales"]),
]
OPERATIONAL_METRICS = [
    ("Tiendas", [r"tiendas?", r"stores?"]),
    ("Estaciones", [r"estaciones?", r"stations?"]),
    ("Usuarios Activos", [r"usuarios?\s+activos?", r"active\s+users?"]),
    ("Volumen", [r"volumen", r"volume"]),
    ("Aperturas", [r"aperturas?", r"openings?"]),
    ("Clientes Juntos+", [r"(?=.*clientes?)(?=.*juntos\+)"]),
    ("Transacciones", [r"transacciones?"]),
    ("Puntos de Venta", [r"puntos?\s+de\s+venta", r"sales\s+points?"]),
    ("Tickets", [r"tickets?"]),
    ("Plantas", [r"plantas?", r"plants?"]),
    ("Centros de Distribución", [r"centros?\s+de\s+distribuci[oó]n",
                                  r"distribution\s+centers?"]),
    ("Capacidad de Producción", [r"capacidad\s+de\s+producci[oó]n",
                                  r"production\s+capacity"]),
]


def norm(s):
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower()


def find_all(hay, patterns):
    n = norm(hay)
    return any(re.search(p, n) for p in patterns)


def numbers_in(text):
    """Devuelve los numeros del texto como cadenas de digitos normalizadas."""
    out = []
    for m in re.finditer(r"\d[\d.,]*", text or ""):
        out.append(re.sub(r"[.,]", "", m.group(0)))
    return out


def first_number(text):
    m = re.search(r"(\d[\d.,]*)", text or "")
    if not m:
        return None
    raw = m.group(1)
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def pct_in(text):
    m = re.search(r"([+-]?\d[\d.,]*)\s*%", text or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def name_pattern(name):
    """Regex para una entidad que NO casa dentro de un nombre mas largo.

    Evita que 'FEMSA' case dentro de 'Coca-Cola FEMSA', 'Spin' dentro de
    'Spin Premia' / 'Spin by OXXO', 'OXXO' dentro de 'OXXO Mexico', etc.
    """
    base = accent_regex(name)
    if name == "FEMSA":
        return r"(?<!Coca-Cola )" + base + r"(?!\s+Forward\b)"
    if name == "Spin":
        return base + r"(?!\s+(?:by\b|Premia\b))"
    if name == "OXXO":
        return base + r"(?!\s+(?:M[eé]xico|GAS|SMART|Nicho|PAY)\b)"
    if name == "Juntos+":
        return r"(?<!Premia )" + base
    return base


def known_entity_in(text):
    """Entidades conocidas mencionadas en el texto, en orden de aparicion."""
    found = []
    for name in KNOWN:
        if re.search(name_pattern(name), text, flags=re.I):
            found.append(name)
    found.sort(key=lambda nm: re.search(name_pattern(nm), text,
                                        flags=re.I).start())
    return found


def make_triple(subject, subject_type, relation, obj, object_type,
                page, valor=None, unidad=None, variacion_pct=None,
                periodo="2024", qa=None):
    assert subject_type in ENTITY_TYPES, subject_type
    assert object_type in ENTITY_TYPES, object_type
    assert relation in RELATIONS, relation
    t = {"subject": subject, "subject_type": subject_type,
         "relation": relation, "object": obj, "object_type": object_type,
         "fuente": {"documento": DOCUMENTO, "pagina": page},
         "periodo": periodo,
         "origen": "reflection_qa",
         "estado": "pendiente_revision",
         "confianza": "regla"}
    if valor is not None:
        t["valor"] = valor
    if unidad:
        t["unidad"] = unidad
    if variacion_pct is not None:
        t["variacion_pct"] = variacion_pct
    if qa:
        t["qa_question"] = qa["question"]
    return t


def accent_regex(name):
    """Regex que casa el nombre con o sin acentos, insensible a mayusculas."""
    out = []
    for ch in name:
        if ch in "aA":
            out.append("[aáAÁ]")
        elif ch in "eE":
            out.append("[eéEÉ]")
        elif ch in "iI":
            out.append("[iíIÍ]")
        elif ch in "oO":
            out.append("[oóOÓ]")
        elif ch in "uU":
            out.append("[uúUÚ]")
        elif ch in "nN":
            out.append("[nñNÑ]")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def mentioned(text, name):
    return re.search(name_pattern(name), text, flags=re.I) is not None


def segment_values(seg):
    """Extrae (valor, moneda, variacion_pct) del primer numero del segmento.

    Moneda: 'Ps. millones' si el segmento habla de pesos, 'US$ millones' si
    habla de dolares, None en otro caso ('N millones' a secas NO es pesos).
    Los anios (2023, 2024...) nunca cuentan como valor.
    """
    money = re.search(r"Ps\.\s*([\d,]+(?:\.\d+)?)", seg) or \
        re.search(r"([\d,]+(?:\.\d+)?)\s*millones\s+de\s+pesos", seg)
    plain = None
    for m in re.finditer(r"([\d,]+(?:\.\d+)?)", seg):
        tok = m.group(1).replace(",", "")
        if re.fullmatch(r"(19|20)\d\d", tok):
            continue  # anio, no es un valor
        plain = m
        break
    pct = re.search(r"([+-]?\d[\d.,]*)\s*%", seg)
    valor, moneda, var = None, None, None
    if money:
        try:
            valor = float(money.group(1).replace(",", ""))
            moneda = "Ps. millones"
        except ValueError:
            pass
    elif plain:
        try:
            valor = float(plain.group(1).replace(",", ""))
        except ValueError:
            pass
    # "116 mil millones" = 116,000 millones, no 116
    if valor is not None and re.search(r"mil\s+millones", seg, flags=re.I):
        valor *= 1000
    if re.search(r"US\$|d[oó]lar", seg, flags=re.I) and valor is not None:
        moneda = "US$ millones"
    if pct:
        try:
            var = float(pct.group(1).replace(",", ""))
        except ValueError:
            pass
    # si el unico numero es el porcentaje, no es un valor absoluto
    if valor is not None and var is not None and valor == var:
        valor, moneda = None, None
    return valor, moneda, var


def subject_segment(combined, answer, subject):
    """Segmento (clausula) alrededor de la mencion del sujeto.

    Prefiere menciones en la respuesta. El texto se parte en clausulas por
    comas / ' y ' / puntos (sin partir dentro de numeros como 279,793 ni
    abreviaturas como 'Ps.'); se toma la clausula de la mencion y, si no tiene
    numero, la siguiente (o la anterior). Asi el numero pertenece al sujeto
    nombrado en esa clausula, no al sujeto anterior ni al siguiente.
    Devuelve (segmento, en_respuesta).
    """
    pat = name_pattern(subject)
    splitter = re.compile(r",(?!\d)\s*|\s+y\s+|\.\s+(?![0-9])")
    for text, in_answer in ((answer, True), (combined, False)):
        m = re.search(pat, text, flags=re.I)
        if not m:
            continue
        parts = splitter.split(text)
        pos = 0
        for i, part in enumerate(parts):
            start = text.find(part, pos)
            end = start + len(part)
            pos = end
            if start <= m.start() < end:
                seg = part
                if not re.search(r"\d", seg):
                    if i + 1 < len(parts):
                        seg = seg + " " + parts[i + 1]
                    elif i > 0:
                        seg = parts[i - 1] + " " + seg
                return seg, in_answer
    return None, False


def other_keyword_in_segment(seg, own_pats):
    """True si el segmento menciona OTRO indicador/metrica distinto del candidato.

    Sirve para no atribuir a una metrica un numero que pertenece a otra
    (p. ej. el EBITDA dentro de una pregunta sobre usuarios activos).
    """
    for _name, pats in FINANCIAL_INDICATORS + OPERATIONAL_METRICS:
        if pats is own_pats:
            continue
        if find_all(seg, pats):
            return True
    return False


def decompose(qa):
    """Descompone un par QA en tripletas candidatas (lista) o None si no mapea.

    Devuelve (triples, diverted): triples es la lista de tripletas; diverted es
    True si el par va a needs_computed_path en vez de a tripletas.
    """
    q, a = qa["question"], qa["answer"]
    text = q + " " + a
    n = norm(text)
    pages = qa.get("source_pages") or [None]
    page = pages[0]
    triples = []

    # --- Agregacion: no se fuerza a tripleta ---
    if qa.get("type") == "aggregation":
        return [], True

    # --- Estructura: tiene_division / tiene_formato / opera_en ---
    if find_all(text, [r"division", r"segmento", r"vertical"]):
        for d in DIVISIONS:
            if mentioned(text, d) and find_all(q, [r"division(es)?", r"segmentos?",
                                                    r"vertical(es)?", r"negocios?"]):
                triples.append(make_triple("FEMSA", "Empresa", "tiene_division",
                                           d, "Division", page, qa=qa))
    if find_all(text, [r"opera en", r"operan en", r"pa[ií]ses?"]):
        for ent in known_entity_in(text):
            if KNOWN[ent] == "Division":
                for c in known_entity_in(text):
                    if KNOWN.get(c) == "Pais":
                        triples.append(make_triple(ent, "Division", "opera_en",
                                                   c, "Pais", page, qa=qa))
    for fmt, div in [("OXXO México", "Proximidad Américas"),
                     ("OXXO GAS", "Combustibles"),
                     ("Spin by OXXO", "Spin"),
                     ("Spin Premia", "Spin"),
                     ("Juntos+", "Coca-Cola FEMSA")]:
        if mentioned(text, fmt) and find_all(q, [r"formato", r"marca", r"parte de",
                                                 r"pertenece", r"de qu[eé] division"]):
            triples.append(make_triple(div, "Division", "tiene_formato",
                                       fmt, "Formato", page, qa=qa))

    # --- Iniciativas ---
    if "femsa forward" in n:
        # FEMSA Forward impulsa las tres verticales; se emite solo si la
        # pregunta nombra la division concreta.
        for d in ["Proximidad Américas", "Coca-Cola FEMSA", "Spin"]:
            if mentioned(text, d):
                triples.append(make_triple("FEMSA Forward", "Iniciativa",
                                           "impulsa", d, "Division", page, qa=qa))
    if "digital@femsa" in n and "spin" in n and \
            find_all(q, [r"evolucion", r"se convirti[oó]", r"ahora se llama",
                         r"antes"]):
        triples.append(make_triple("Digital@FEMSA", "Iniciativa",
                                   "evoluciono_a", "Spin", "Division",
                                   page, qa=qa))

    # --- Indicadores financieros / metricas operativas ---
    # La metrica debe aparecer en la PREGUNTA (el tema), y el numero se atribuye
    # por proximidad a la mencion del sujeto en la respuesta. Si el segmento del
    # sujeto menciona OTRA metrica, no se emite (el numero pertenece a esa otra).
    # Preguntas sobre ratios (p. ej. deuda neta / EBITDA) no producen tripletas
    # de valor: el numero es un ratio, no el valor del indicador.
    if re.search(r"ratio|deuda\s+neta|apalancamiento", norm(q)):
        return triples, False
    subjects = [e for e in known_entity_in(text)
                if KNOWN[e] in ("Empresa", "Division", "Formato")]

    def emit_for(s, pats, relation, obj, obj_type):
        if not find_all(q, pats):
            return
        seg, _in_answer = subject_segment(text, a, s)
        if seg is None:
            return
        if other_keyword_in_segment(seg, pats):
            return
        v, u, pc = segment_values(seg)
        # reporta exige valor absoluto: un % solo (p. ej. un margen) no es un
        # reporte; el crecimiento va en crecio_en, no aqui.
        if relation == "reporta" and v is None:
            return
        if v is None and pc is None:
            return
        triples.append(make_triple(s, KNOWN[s], relation, obj, obj_type,
                                   page, valor=v, unidad=u,
                                   variacion_pct=pc, qa=qa))

    for ind, pats in FINANCIAL_INDICATORS:
        for s in subjects:
            emit_for(s, pats, "reporta", ind, "IndicadorFinanciero")
    # crecio_en: solo cuando la pregunta es explicitamente sobre el crecimiento
    # de UN solo indicador ("crecio/crecieron/crecimiento"); "aumento" solo no
    # basta (puede ser un margen o un nivel), y con dos indicadores en la
    # pregunta el primer % de la respuesta no se puede atribuir a uno de ellos.
    if pct_in(a) is not None and re.search(r"creci", norm(q)):
        matching = [ind for ind, pats in FINANCIAL_INDICATORS
                    if find_all(q, pats)]
        if len(matching) == 1:
            triples.append(make_triple(matching[0], "IndicadorFinanciero",
                                       "crecio_en", "2024", "Periodo",
                                       page, variacion_pct=pct_in(a), qa=qa))
    for met, pats in OPERATIONAL_METRICS:
        # "tiendas nuevas netas" / "aperturas" es Aperturas, no Tiendas totales;
        # "ventas mismas-tienda" es un indicador financiero, no Tiendas.
        if met == "Tiendas" and re.search(
                r"nuevas?\s+netas?|aperturas?|mismas?\s*-?\s*tiendas?", norm(q)):
            continue
        for s in subjects:
            seg, _in_answer = subject_segment(text, a, s)
            if seg is None or not find_all(q, pats):
                continue
            if other_keyword_in_segment(seg, pats):
                continue
            v, u, pc = segment_values(seg)
            if v is None and pc is None:
                continue
            unidad = u
            if unidad is None:
                if "millones de usuarios" in norm(a) or "usuarios" in norm(a):
                    unidad = "millones de usuarios"
                elif "transacciones" in norm(a):
                    unidad = "millones de transacciones"
                elif "puntos de venta" in norm(a):
                    unidad = "millones de puntos de venta"
                elif "tiendas" in norm(a):
                    unidad = "tiendas"
                elif "estaciones" in norm(a):
                    unidad = "estaciones"
            triples.append(make_triple(s, KNOWN[s], "mide", met,
                                       "MetricaOperativa", page,
                                       valor=v, unidad=unidad,
                                       variacion_pct=pc, qa=qa))
    return triples, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="memo/reflection_qa.jsonl")
    ap.add_argument("--out", default="pipeline/extracted_triples_from_qa.jsonl")
    ap.add_argument("--computed", default="pipeline/needs_computed_path.json")
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.qa)]
    triples, computed, skipped = [], [], 0
    by_relation = {}
    for qa in pairs:
        try:
            ts, diverted = decompose(qa)
        except Exception as e:  # noqa: BLE001 - un par roto no tumba el lote
            print(f"WARN: no se pudo descomponer: {qa.get('question','?')[:80]} ({e})")
            skipped += 1
            continue
        if diverted:
            computed.append({
                "question": qa["question"], "answer": qa["answer"],
                "source_pages": qa.get("source_pages"),
                "type": qa.get("type"),
                "reason": ("Agregacion/sintesis: requiere calculo sobre varias "
                           "tripletas (suma, participacion, ranking), no un hecho unico."),
                "estado": "pendiente_revision"})
            continue
        for t in ts:
            # verificacion: el valor debe aparecer en la respuesta del par QA.
            # "116 mil millones" en la respuesta equivale a valor 116000.
            if t.get("valor") is not None:
                v = t["valor"]
                v_str = str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
                digits = re.sub(r"[.,]", "", v_str)
                ans_nums = numbers_in(qa["answer"])
                ok = digits in ans_nums
                if not ok and "mil millones" in norm(qa["answer"]):
                    try:
                        ok = str(int(v // 1000)) in ans_nums if v % 1000 == 0 else False
                    except (ValueError, ZeroDivisionError):
                        ok = False
                if not ok:
                    skipped += 1
                    continue
            triples.append(t)
            by_relation[t["relation"]] = by_relation.get(t["relation"], 0) + 1

    # deduplicacion por (subject, relation, object, valor)
    seen, dedup = set(), []
    for t in triples:
        key = (t["subject"], t["relation"], t["object"],
               t.get("valor"), t.get("variacion_pct"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(t)

    with open(args.out, "w") as f:
        for t in dedup:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(args.computed, "w") as f:
        json.dump(computed, f, ensure_ascii=False, indent=2)

    print(f"pares QA leidos: {len(pairs)}")
    print(f"tripletas candidatas (dedup): {len(dedup)} -> {args.out}")
    print(f"  por relacion: {by_relation}")
    print(f"pares de agregacion -> needs_computed_path: {len(computed)} -> {args.computed}")
    print(f"descartadas (valor no verificable / error): {skipped}")
    print("ESTADO: pendiente_revision — NO cargar al grafo sin revision humana.")


if __name__ == "__main__":
    main()
