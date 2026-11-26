#!/usr/bin/env python3
"""Genera memo/eval_questions_100.json (12 originales + 88 nuevas) y
rag_baseline/ragas/ground_truth_evidence_100.json, con verificacion.

Reglas (las mismas del experimento):
- Cada cifra esperada se verifica por grep literal (normalizado) en
  pipeline/data/chunks.jsonl (151 paginas, edicion en ingles del informe
  anual FEMSA 2024). Lo que no aparece en el corpus se descarta: este
  script falla (exit != 0) si alguna cifra no se encuentra.
- Las preguntas de agregacion (nivel 3) llevan sus insumos y su formula:
  el script verifica cada insumo en el corpus y recomputa el resultado.
- Las 12 originales se copian byte-identicas (mismos objetos JSON) desde
  memo/eval_questions.json; no se reescriben a mano.
- Q7 y Q3 conservan sus conflictos documentados (ver ground_truth_note).

Uso:
    python memo/build_eval_100.py [--verify-only]
Escribe:
    memo/eval_questions_100.json
    rag_baseline/ragas/ground_truth_evidence_100.json
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CHUNKS = HERE / "pipeline" / "data" / "chunks.jsonl"
EVAL12 = HERE / "memo" / "eval_questions.json"
OUT100 = HERE / "memo" / "eval_questions_100.json"
GT100 = HERE / "rag_baseline" / "ragas" / "ground_truth_evidence_100.json"
GT12 = HERE / "rag_baseline" / "ragas" / "ground_truth_evidence.json"


# ---------------------------------------------------------------------------
# Corpus: texto por pagina + normalizacion de numeros
# ---------------------------------------------------------------------------

def load_corpus():
    pages = {}
    for line in open(CHUNKS):
        c = json.loads(line)
        pages[c["page"]] = c["text"]
    return pages


def norm_digits(s):
    """'781,585' -> '781585'; '4,224.6' -> '4224.6'; '11,2' -> '11.2'."""
    s = str(s).strip().rstrip("%")
    if "," in s and "." not in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) == 1:
            return parts[0] + "." + parts[1]  # coma decimal
        return "".join(parts)
    return s.replace(",", "")


NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?")


def numbers_on_page(text):
    return {norm_digits(m.group(0)) for m in NUM_RE.finditer(text)}


def pages_with_number(pages, num):
    """Paginas donde aparece el numero (normalizado)."""
    target = norm_digits(num)
    return sorted(p for p, t in pages.items() if target in numbers_on_page(t))


def check_number(pages, num, evidence_pages):
    """Devuelve (ok, paginas_donde_aparece). ok exige que aparezca en el
    corpus Y en al menos una de las paginas de evidencia declaradas."""
    found = pages_with_number(pages, num)
    ok = bool(found) and bool(set(found) & set(evidence_pages))
    return ok, found


# ---------------------------------------------------------------------------
# Las 88 preguntas nuevas (Q13-Q100).
# Campos: id, level, question, expected, expected_numbers (o
# expected_keywords), evidence_pages, type, notes (opcional),
# ground_truth_note (opcional), inputs + formula (solo L3).
# ---------------------------------------------------------------------------

NEW = [
    # ---- Nivel 1: hecho directo (Q13-Q48) ----
    dict(id="Q13", level=1,
         question="¿Cuál fue la utilidad de operación consolidada de FEMSA en 2024 y cuánto creció vs 2023?",
         expected="Ps. 70,667 millones, +19.8%.",
         expected_numbers=["70667", "19.8"], evidence_pages=[108, 109, 112]),
    dict(id="Q14", level=1,
         question="¿Cuál fue el EBITDA consolidado de FEMSA en 2024?",
         expected="Ps. 115,593 millones.",
         expected_numbers=["115593"], evidence_pages=[108, 109]),
    dict(id="Q15", level=1,
         question="¿Cuál fue la utilidad neta consolidada de FEMSA en 2024?",
         expected="Ps. 40,236 millones.",
         expected_numbers=["40236"], evidence_pages=[108, 109],
         ground_truth_note="La tabla del estado de resultados (pp. 108-109) dice Ps. 40,236 millones; la narrativa del MD&A (p. 113) dice Ps. 41,687 millones. Se usa el valor de la tabla."),
    dict(id="Q16", level=1,
         question="¿Cuáles fueron los ingresos de cada división de FEMSA en 2024?",
         expected="Proximidad Américas Ps. 307,197 millones; Coca-Cola FEMSA Ps. 279,793 millones; Salud Ps. 79,755 millones; Combustibles Ps. 65,365 millones; Proximidad Europa Ps. 49,755 millones.",
         expected_numbers=["307197", "279793", "79755", "65365", "49755"],
         evidence_pages=[112]),
    dict(id="Q17", level=1,
         question="¿Cuál fue la utilidad bruta consolidada de FEMSA en 2024?",
         expected="Ps. 321,513 millones.",
         expected_numbers=["321513"], evidence_pages=[109, 112],
         ground_truth_note="El estado de resultados y la tabla del MD&A (pp. 109, 112) dicen Ps. 321,513 millones; la narrativa del MD&A (p. 112) dice Ps. 321,417 millones. Se usa el valor de la tabla."),
    dict(id="Q18", level=1,
         question="¿Cuál fue el margen bruto consolidado de FEMSA en 2024?",
         expected="41.1%.",
         expected_numbers=["41.1"], evidence_pages=[109, 112]),
    dict(id="Q19", level=1,
         question="¿Cuánto sumaron los activos totales de FEMSA al cierre de 2024?",
         expected="Ps. 851,536 millones.",
         expected_numbers=["851536"], evidence_pages=[108, 110]),
    dict(id="Q20", level=1,
         question="¿Cuánto invirtió FEMSA en CAPEX en 2024 y cuánto creció vs 2023?",
         expected="Ps. 51,074 millones, +31.1%.",
         expected_numbers=["51074", "31.1"], evidence_pages=[108]),
    dict(id="Q21", level=1,
         question="¿Cuánto efectivo y equivalentes tenía FEMSA al cierre de 2024?",
         expected="Ps. 139,834 millones.",
         expected_numbers=["139834"], evidence_pages=[108]),
    dict(id="Q22", level=1,
         question="¿Cuál fue la utilidad neta de la participación controladora en 2024?",
         expected="Ps. 26,735 millones.",
         expected_numbers=["26735"], evidence_pages=[108, 109]),
    dict(id="Q23", level=1,
         question="¿Cuál fue el volumen de Coca-Cola FEMSA en 2024?",
         expected="4,224.6 millones de cajas unidad.",
         expected_numbers=["4224.6"], evidence_pages=[113]),
    dict(id="Q24", level=1,
         question="¿Cuál fue la utilidad bruta de Coca-Cola FEMSA y de Proximidad Américas en 2024?",
         expected="Coca-Cola FEMSA Ps. 128,736 millones; Proximidad Américas Ps. 136,993 millones.",
         expected_numbers=["128736", "136993"], evidence_pages=[112]),
    dict(id="Q25", level=1,
         question="¿Cuánto crecieron las ventas mismas tiendas de Proximidad Américas en 2024?",
         expected="+4.2%.",
         expected_numbers=["4.2"], evidence_pages=[114]),
    dict(id="Q26", level=1,
         question="¿Cuánto crecieron las ventas mismas estaciones de Combustibles (OXXO GAS) en 2024?",
         expected="+9.9%.",
         expected_numbers=["9.9"], evidence_pages=[114]),
    dict(id="Q27", level=1,
         question="¿Cuántas tiendas OXXO había en total al cierre de 2024?",
         expected="24,462 tiendas.",
         expected_numbers=["24462"], evidence_pages=[23, 114]),
    dict(id="Q28", level=1,
         question="¿Cuántos puntos de venta tenía Proximidad Américas al cierre de 2024?",
         expected="26,322 puntos de venta.",
         expected_numbers=["26322"], evidence_pages=[21]),
    dict(id="Q29", level=1,
         question="¿Cuántas farmacias tenía FEMSA Salud al cierre de 2024?",
         expected="4,661 farmacias.",
         expected_numbers=["4661"], evidence_pages=[21]),
    dict(id="Q30", level=1,
         question="¿Cuántos puntos de venta tenía Proximidad Europa al cierre de 2024?",
         expected="2,778 puntos de venta.",
         expected_numbers=["2778"], evidence_pages=[21]),
    dict(id="Q31", level=1,
         question="¿Cuántos colaboradores tenía FEMSA en total en 2024?",
         expected="388,999 colaboradores.",
         expected_numbers=["388999"], evidence_pages=[108]),
    dict(id="Q32", level=1,
         question="¿En cuántos países opera FEMSA según el informe anual 2024?",
         expected="18 países.",
         expected_numbers=["18"], evidence_pages=[10, 21]),
    dict(id="Q33", level=1,
         question="¿Cuántos clientes tenía Juntos+ en 2024?",
         expected="1.3 millones de clientes.",
         expected_numbers=["1.3"], evidence_pages=[27]),
    dict(id="Q34", level=1,
         question="¿Cuántos usuarios tenía Premia Juntos+ en 2024?",
         expected="Más de 1.1 millones de usuarios.",
         expected_numbers=["1.1"], evidence_pages=[27]),
    dict(id="Q35", level=1,
         question="¿Cuántas plantas embotelladoras y cuántos centros de distribución tenía Coca-Cola FEMSA en 2024?",
         expected="56 plantas embotelladoras y 256 centros de distribución.",
         expected_numbers=["56", "256"], evidence_pages=[30]),
    dict(id="Q36", level=1,
         question="¿Cuántas líneas de producción nuevas instaló Coca-Cola FEMSA en 2024?",
         expected="7 líneas nuevas.",
         expected_numbers=["7"], evidence_pages=[30]),
    dict(id="Q37", level=1,
         question="¿Cuál fue la tasa efectiva de impuestos de FEMSA en 2024?",
         expected="35.9%.",
         expected_numbers=["35.9"], evidence_pages=[113]),
    dict(id="Q38", level=1,
         question="¿Cuántas transacciones mensuales procesaba Spin by OXXO en 2024?",
         expected="Más de 63 millones de transacciones mensuales.",
         expected_numbers=["63"], evidence_pages=[35]),
    dict(id="Q39", level=1,
         question="¿Cuántos clientes nuevos sumó Spin by OXXO en 2024?",
         expected="Más de 13 millones de clientes.",
         expected_numbers=["13"], evidence_pages=[35]),
    dict(id="Q40", level=1,
         question="¿Cuántos usuarios nuevos sumó Spin Premia en 2024?",
         expected="Más de 52 millones de usuarios.",
         expected_numbers=["52"], evidence_pages=[35]),
    dict(id="Q41", level=1,
         question="¿Cuántas tazas de café andatti se servían al día en 2024?",
         expected="Más de 811,000 tazas al día.",
         expected_numbers=["811000"], evidence_pages=[20]),
    dict(id="Q42", level=1,
         question="¿Cuántas tiendas OXXO había en Colombia al cierre de 2024?",
         expected="569 tiendas.",
         expected_numbers=["569"], evidence_pages=[21]),
    dict(id="Q43", level=1,
         question="¿Cuántas tiendas OXXO había en México al cierre de 2024?",
         expected="23,206 tiendas.",
         expected_numbers=["23206"], evidence_pages=[21]),
    dict(id="Q44", level=1,
         question="¿Cuál fue el margen de utilidad de operación consolidado de FEMSA en 2024?",
         expected="9.0%.",
         expected_numbers=["9.0"], evidence_pages=[108]),
    dict(id="Q45", level=1,
         question="¿Cuál fue el margen EBITDA consolidado de FEMSA en 2024?",
         expected="14.8%.",
         expected_numbers=["14.8"], evidence_pages=[108]),
    dict(id="Q46", level=1,
         question="¿Cuánto fueron las ventas netas consolidadas de FEMSA en 2024?",
         expected="Ps. 775,551 millones.",
         expected_numbers=["775551"], evidence_pages=[109]),
    dict(id="Q47", level=1,
         question="¿Cuánto fue el costo de ventas consolidado de FEMSA en 2024?",
         expected="Ps. 460,072 millones.",
         expected_numbers=["460072"], evidence_pages=[109]),
    dict(id="Q48", level=1,
         question="¿Cuánto sumaron los pasivos totales de FEMSA al cierre de 2024?",
         expected="Ps. 470,405 millones.",
         expected_numbers=["470405"], evidence_pages=[108, 110]),
]

NEW_L2 = [
    # ---- Nivel 2: comparación / multi-salto (Q49-Q74) ----
    dict(id="Q49", level=2,
         question="¿Cuál creció más en usuarios en 2024 según el informe anual, Spin by OXXO o Spin Premia, y cuánto creció cada uno?",
         expected="Spin by OXXO creció más: +32.8% vs +31.5% de Spin Premia.",
         expected_numbers=["32.8", "31.5"], evidence_pages=[34],
         ground_truth_note="TRAMPA DOCUMENTADA: estas son las cifras del informe anual (p. 34). La Q7 espera +27.5% / +24.9%, que vienen del comunicado 4Q24, no del informe. Ambas preguntas se mantienen para medir el conflicto."),
    dict(id="Q50", level=2,
         question="¿Qué división tuvo el mayor margen bruto en 2024?",
         expected="Coca-Cola FEMSA, 46.0%.",
         expected_numbers=["46.0"], evidence_pages=[112, 113]),
    dict(id="Q51", level=2,
         question="¿Qué división tuvo el menor margen bruto en 2024?",
         expected="Combustibles, 12.1%.",
         expected_numbers=["12.1"], evidence_pages=[114]),
    dict(id="Q52", level=2,
         question="¿Qué división tuvo los menores ingresos en 2024?",
         expected="Proximidad Europa, Ps. 49,755 millones.",
         expected_numbers=["49755"], evidence_pages=[112]),
    dict(id="Q53", level=2,
         question="¿Qué división tuvo el mayor crecimiento de gastos de administración en 2024?",
         expected="Salud, +56.0%.",
         expected_numbers=["56.0"], evidence_pages=[114]),
    dict(id="Q54", level=2,
         question="¿En qué país tenía FEMSA más colaboradores en 2024?",
         expected="México, 268,407 colaboradores.",
         expected_numbers=["268407"], evidence_pages=[55]),
    dict(id="Q55", level=2,
         question="¿Quién tenía más puntos de venta, Coca-Cola FEMSA o Proximidad Américas?",
         expected="Coca-Cola FEMSA, con ~2.2 millones de puntos de venta, frente a 26,322 de Proximidad Américas.",
         expected_numbers=["2.2", "26322"], evidence_pages=[13, 21]),
    dict(id="Q56", level=2,
         question="¿En qué país consumió FEMSA más energía renovable (MWh) en 2024?",
         expected="México, 1,724,031 MWh.",
         expected_numbers=["1724031"], evidence_pages=[84]),
    dict(id="Q57", level=2,
         question="¿Qué material de empaque tenía el mayor porcentaje reciclado en 2024?",
         expected="Aluminio, 80% reciclado.",
         expected_numbers=["80"], evidence_pages=[125]),
    dict(id="Q58", level=2,
         question="¿En qué tema se impartieron más horas de capacitación en 2024?",
         expected="Conocimiento técnico, 4,242,222 horas.",
         expected_numbers=["4242222"], evidence_pages=[123]),
    dict(id="Q59", level=2,
         question="¿Cuál fue la desinversión más grande de FEMSA en 2024?",
         expected="Imbera / Torrey, Ps. 8,000 millones.",
         expected_numbers=["8000"], evidence_pages=[115, 116]),
    dict(id="Q60", level=2,
         question="¿Cuál fue la recompra acelerada de acciones (ASR) más grande de 2024, la de marzo o la de junio?",
         expected="La de junio, hasta US$ 600 millones (la de marzo fue de US$ 400 millones).",
         expected_numbers=["600", "400"], evidence_pages=[115, 116]),
    dict(id="Q61", level=2,
         question="¿Hubo más horas de capacitación en 2023 o en 2024, y cuántas en cada año?",
         expected="Más en 2023: 9,787,020 horas vs 6,181,088 en 2024.",
         expected_numbers=["9787020", "6181088"], evidence_pages=[122]),
    dict(id="Q62", level=2,
         question="¿Hubo más fatalidades entre colaboradores directos o entre contratistas en 2024?",
         expected="Empate: 2 fatalidades de colaboradores directos y 2 de contratistas.",
         expected_numbers=["2"], evidence_pages=[122],
         ground_truth_note="TRAMPA: la respuesta es un empate (2 y 2). En 2023 no lo era (9 directos vs 4 contratistas): quien responda con las cifras de 2023 falla."),
    dict(id="Q63", level=2,
         question="¿Qué fue mayor en 2024, las emisiones de Alcance 1 o las de Alcance 2?",
         expected="Alcance 1: 1,003,012 t CO2eq vs 444,618 t de Alcance 2 (base de mercado).",
         expected_numbers=["1003012", "444618"], evidence_pages=[125]),
    dict(id="Q64", level=2,
         question="¿En qué país tenía FEMSA Salud más farmacias en 2024?",
         expected="México, 1,739 farmacias.",
         expected_numbers=["1739"], evidence_pages=[21]),
    dict(id="Q65", level=2,
         question="¿En qué país tenía Valora más tiendas en 2024?",
         expected="Alemania, 1,376 tiendas.",
         expected_numbers=["1376"], evidence_pages=[21]),
    dict(id="Q66", level=2,
         question="¿Qué indicador financiero creció más en 2024, los ingresos (+11.2%), la utilidad de operación (+19.8%) o el EBITDA (+20.6%)?",
         expected="El EBITDA, +20.6%.",
         expected_numbers=["20.6"], evidence_pages=[108]),
    dict(id="Q67", level=2,
         question="¿Qué cayó más en 2024, la utilidad neta consolidada (-47.5%) o la de la participación controladora (-59.3%)?",
         expected="La de la participación controladora, -59.3% (vs -47.5% consolidada).",
         expected_numbers=["59.3", "47.5"], evidence_pages=[108]),
    dict(id="Q68", level=2,
         question="¿Quién tenía más clientes en 2024, Juntos+ o Premia Juntos+?",
         expected="Juntos+, 1.3 millones de clientes, frente a más de 1.1 millones de Premia Juntos+.",
         expected_numbers=["1.3", "1.1"], evidence_pages=[27]),
    dict(id="Q69", level=2,
         question="¿Qué porcentaje del Consejo de Administración era independiente y qué porcentaje eran mujeres en 2024?",
         expected="46% independientes y 40% mujeres.",
         expected_numbers=["46", "40"], evidence_pages=[115, 116]),
    dict(id="Q70", level=2,
         question="¿Qué fue mayor en Spin by OXXO en 2024, los clientes nuevos (+13 millones) o los usuarios activos (8.6 millones)?",
         expected="Los clientes nuevos: más de 13 millones, frente a 8.6 millones de usuarios activos.",
         expected_numbers=["13", "8.6"], evidence_pages=[33, 35]),
    dict(id="Q71", level=2,
         question="¿En qué país tenía FEMSA más colaboradores después de México, Brasil o Colombia?",
         expected="Brasil, 46,016 colaboradores (Colombia: 25,563).",
         expected_numbers=["46016", "25563"], evidence_pages=[55]),
    dict(id="Q72", level=2,
         question="¿Qué era mayor en 2024, las tiendas OXXO en México (23,206) o las farmacias de FEMSA Salud en total (4,661)?",
         expected="Las tiendas OXXO en México: 23,206 frente a 4,661 farmacias.",
         expected_numbers=["23206", "4661"], evidence_pages=[21]),
    dict(id="Q73", level=2,
         question="¿En qué pilar invirtió más FEMSA en 2024 según el informe de sostenibilidad?",
         expected="En Personas (People), Ps. 3,855 millones.",
         expected_numbers=["3855"], evidence_pages=[121]),
    dict(id="Q74", level=2,
         question="¿Qué meta 2030 estaba más cerca de cumplirse en 2024, la de compromiso de colaboradores (90%) o la de proveedores locales (90%)?",
         expected="La de compromiso de colaboradores: 88.5% en 2024 (proveedores locales: 77%).",
         expected_numbers=["88.5", "77"], evidence_pages=[46]),
]

NEW_L3 = [
    # ---- Nivel 3: agregacion (Q75-Q100) ----
    # inputs: lista de dicts {label, value (numero tal cual en el informe),
    #         pages}. formula: expresion Python con variables a..z en el
    #         orden de inputs; el resultado se compara con expected_numbers.
    dict(id="Q75", level=3,
         question="¿Cuánto suman los ingresos de las cinco divisiones de FEMSA en 2024?",
         expected="Ps. 781,865 millones (la suma de las divisiones; difiere del consolidado de Ps. 781,585 millones por eliminaciones y otros).",
         expected_numbers=["781865"], evidence_pages=[112],
         inputs=[dict(label="Proximidad Américas 307,197", value="307197", pages=[112]),
                 dict(label="Coca-Cola FEMSA 279,793", value="279793", pages=[112]),
                 dict(label="Proximidad Europa 49,755", value="49755", pages=[112]),
                 dict(label="Salud 79,755", value="79755", pages=[112]),
                 dict(label="Combustibles 65,365", value="65365", pages=[112])],
         formula="a+b+c+d+e",
         notes="La suma (781,865) difiere del consolidado (781,585) por 280: eliminaciones y otros. El modelo debe dar la suma de las divisiones."),
    dict(id="Q76", level=3,
         question="¿Cuántas tiendas OXXO había en total si sumas las de México, Colombia, Chile, Perú y Estados Unidos?",
         expected="24,462 tiendas.",
         expected_numbers=["24462"], evidence_pages=[21, 23, 114],
         inputs=[dict(label="México 23,206", value="23206", pages=[21]),
                 dict(label="Colombia 569", value="569", pages=[21]),
                 dict(label="Chile 235", value="235", pages=[21]),
                 dict(label="Perú 203", value="203", pages=[21]),
                 dict(label="Estados Unidos 249", value="249", pages=[21])],
         formula="a+b+c+d+e"),
    dict(id="Q77", level=3,
         question="¿Cuántas farmacias tenía FEMSA Salud en total si sumas las de México, Chile, Ecuador y Colombia?",
         expected="4,661 farmacias.",
         expected_numbers=["4661"], evidence_pages=[21],
         inputs=[dict(label="México 1,739", value="1739", pages=[21]),
                 dict(label="Chile 997", value="997", pages=[21]),
                 dict(label="Ecuador 980", value="980", pages=[21]),
                 dict(label="Colombia 945", value="945", pages=[21])],
         formula="a+b+c+d"),
    dict(id="Q78", level=3,
         question="¿Cuántos puntos de venta tenía Proximidad Europa en total si sumas Alemania, Suiza, Luxemburgo, Países Bajos y Austria?",
         expected="2,778 puntos de venta.",
         expected_numbers=["2778"], evidence_pages=[21],
         inputs=[dict(label="Alemania 1,376", value="1376", pages=[21]),
                 dict(label="Suiza 1,241", value="1241", pages=[21]),
                 dict(label="Luxemburgo 76", value="76", pages=[21]),
                 dict(label="Países Bajos 54", value="54", pages=[21]),
                 dict(label="Austria 31", value="31", pages=[21])],
         formula="a+b+c+d+e"),
    dict(id="Q79", level=3,
         question="¿Cuántos usuarios activos tenía Spin en total en 2024 (Spin by OXXO + Spin Premia)?",
         expected="33.2 millones de usuarios activos.",
         expected_numbers=["33.2"], evidence_pages=[13, 33, 35],
         inputs=[dict(label="Spin by OXXO 8.6", value="8.6", pages=[13, 33, 35]),
                 dict(label="Spin Premia 24.6", value="24.6", pages=[13, 33, 35])],
         formula="a+b"),
    dict(id="Q80", level=3,
         question="¿Cuántos clientes sumaban Juntos+ y Premia Juntos+ en 2024?",
         expected="2.4 millones de clientes.",
         expected_numbers=["2.4"], evidence_pages=[27],
         inputs=[dict(label="Juntos+ 1.3", value="1.3", pages=[27]),
                 dict(label="Premia Juntos+ 1.1", value="1.1", pages=[27])],
         formula="a+b"),
    dict(id="Q81", level=3,
         question="¿Qué porcentaje de la utilidad bruta consolidada aportó Coca-Cola FEMSA en 2024?",
         expected="40.0% (128,736 de 321,513).",
         expected_numbers=["40.0"], evidence_pages=[109, 112],
         inputs=[dict(label="Utilidad bruta Coca-Cola FEMSA 128,736", value="128736", pages=[112]),
                 dict(label="Utilidad bruta consolidada 321,513", value="321513", pages=[109, 112])],
         formula="round(a/b*100,1)",
         notes="TRAMPA DE PORCENTAJE: el 40.0% es la participación en la utilidad bruta, no un margen."),
    dict(id="Q82", level=3,
         question="¿Por cuánto superaron los ingresos de Proximidad Américas a los de Coca-Cola FEMSA en 2024?",
         expected="Ps. 27,404 millones.",
         expected_numbers=["27404"], evidence_pages=[112],
         inputs=[dict(label="Proximidad Américas 307,197", value="307197", pages=[112]),
                 dict(label="Coca-Cola FEMSA 279,793", value="279793", pages=[112])],
         formula="a-b"),
    dict(id="Q83", level=3,
         question="¿Cuánto suman las ventas netas y las otras partidas que llegan a los ingresos totales de 2024 (781,585 menos 775,551)?",
         expected="Ps. 6,034 millones.",
         expected_numbers=["6034"], evidence_pages=[109],
         inputs=[dict(label="Ingresos totales 781,585", value="781585", pages=[109]),
                 dict(label="Ventas netas 775,551", value="775551", pages=[109])],
         formula="a-b"),
    dict(id="Q84", level=3,
         question="¿Cuánto suman los pasivos y el capital para llegar a los activos totales de 2024?",
         expected="Ps. 851,536 millones (470,405 + 381,131).",
         expected_numbers=["851536"], evidence_pages=[108, 110],
         inputs=[dict(label="Pasivos totales 470,405", value="470405", pages=[108, 110]),
                 dict(label="Capital total 381,131", value="381131", pages=[108, 110])],
         formula="a+b"),
    dict(id="Q85", level=3,
         question="¿Cuánto suman los pasivos corrientes y no corrientes de FEMSA en 2024?",
         expected="Ps. 470,405 millones (202,930 + 267,475).",
         expected_numbers=["470405"], evidence_pages=[108, 110],
         inputs=[dict(label="Pasivos corrientes 202,930", value="202930", pages=[108, 110]),
                 dict(label="Pasivos no corrientes 267,475", value="267475", pages=[108, 110])],
         formula="a+b"),
    dict(id="Q86", level=3,
         question="¿Qué porcentaje de la energía total consumida por FEMSA en 2024 era renovable?",
         expected="36.2% (7,588,556 de 20,948,731 GJ).",
         expected_numbers=["36.2"], evidence_pages=[125],
         inputs=[dict(label="Energía renovable 7,588,556 GJ", value="7588556", pages=[125]),
                 dict(label="Energía total 20,948,731 GJ", value="20948731", pages=[125])],
         formula="round(a/b*100,1)",
         notes="TRAMPA: el 65.3% del informe es electricidad renovable, NO energía total. La respuesta correcta es 36.2%."),
    dict(id="Q87", level=3,
         question="¿Qué porcentaje de los materiales usados por FEMSA en 2024 era reciclado?",
         expected="36.9% (214,902 de 582,453 toneladas).",
         expected_numbers=["36.9"], evidence_pages=[125],
         inputs=[dict(label="Materiales reciclados 214,902 t", value="214902", pages=[125]),
                 dict(label="Materiales totales 582,453 t", value="582453", pages=[125])],
         formula="round(a/b*100,1)"),
    dict(id="Q88", level=3,
         question="¿Qué porcentaje del agua extraída por FEMSA en 2024 era subterránea?",
         expected="52.6% (22,494 de 42,734 miles de m³).",
         expected_numbers=["52.6"], evidence_pages=[125],
         inputs=[dict(label="Agua subterránea 22,494", value="22494", pages=[125]),
                 dict(label="Agua total extraída 42,734", value="42734", pages=[125])],
         formula="round(a/b*100,1)"),
    dict(id="Q89", level=3,
         question="¿Cuánto sumaron las emisiones de Alcance 1 y Alcance 2 de FEMSA en 2024?",
         expected="1,447,630 t CO2eq (1,003,012 + 444,618).",
         expected_numbers=["1447630"], evidence_pages=[125],
         inputs=[dict(label="Alcance 1: 1,003,012", value="1003012", pages=[125]),
                 dict(label="Alcance 2: 444,618", value="444618", pages=[125])],
         formula="a+b"),
    dict(id="Q90", level=3,
         question="¿Cuánta agua extrajo FEMSA en total en 2024 si sumas la subterránea, la de terceros y la superficial?",
         expected="42,734 miles de m³ (22,494 + 19,464 + 776).",
         expected_numbers=["42734"], evidence_pages=[125],
         inputs=[dict(label="Subterránea 22,494", value="22494", pages=[125]),
                 dict(label="Terceros 19,464", value="19464", pages=[125]),
                 dict(label="Superficial 776", value="776", pages=[125])],
         formula="a+b+c"),
    dict(id="Q91", level=3,
         question="¿Cuántos residuos generó FEMSA en total en 2024 (peligrosos + no peligrosos)?",
         expected="326,038 toneladas (4,705 + 321,333).",
         expected_numbers=["326038"], evidence_pages=[126],
         inputs=[dict(label="Peligrosos 4,705", value="4705", pages=[126]),
                 dict(label="No peligrosos 321,333", value="321333", pages=[126])],
         formula="a+b"),
    dict(id="Q92", level=3,
         question="¿Cuánta energía consumió FEMSA en total en 2024 (renovable + no renovable)?",
         expected="20,948,731 GJ (7,588,556 + 13,360,175).",
         expected_numbers=["20948731"], evidence_pages=[125],
         inputs=[dict(label="Renovable 7,588,556", value="7588556", pages=[125]),
                 dict(label="No renovable 13,360,175", value="13360175", pages=[125])],
         formula="a+b"),
    dict(id="Q93", level=3,
         question="¿Cuántos materiales usó FEMSA en total en 2024 (vírgenes + reciclados)?",
         expected="582,453 toneladas (367,551 + 214,902).",
         expected_numbers=["582453"], evidence_pages=[125],
         inputs=[dict(label="Vírgenes 367,551", value="367551", pages=[125]),
                 dict(label="Reciclados 214,902", value="214902", pages=[125])],
         formula="a+b"),
    dict(id="Q94", level=3,
         question="¿Cuántos colaboradores sumaban los tres países con más personal de FEMSA en 2024 (México, Brasil y Colombia)?",
         expected="339,986 colaboradores (268,407 + 46,016 + 25,563).",
         expected_numbers=["339986"], evidence_pages=[55],
         inputs=[dict(label="México 268,407", value="268407", pages=[55]),
                 dict(label="Brasil 46,016", value="46016", pages=[55]),
                 dict(label="Colombia 25,563", value="25563", pages=[55])],
         formula="a+b+c"),
    dict(id="Q95", level=3,
         question="¿Cuántos colaboradores tenía FEMSA fuera de México en 2024?",
         expected="120,592 colaboradores (388,999 − 268,407).",
         expected_numbers=["120592"], evidence_pages=[55, 108],
         inputs=[dict(label="Total 388,999", value="388999", pages=[108]),
                 dict(label="México 268,407", value="268407", pages=[55])],
         formula="a-b"),
    dict(id="Q96", level=3,
         question="¿Cuánto sumaron las tres desinversiones de FEMSA en 2024 (Imbera/Torrey, Solistica y plásticos)?",
         expected="Ps. 15,225 millones (8,000 + 4,060 + 3,165).",
         expected_numbers=["15225"], evidence_pages=[115, 116, 117],
         inputs=[dict(label="Imbera/Torrey 8,000", value="8000", pages=[115, 116]),
                 dict(label="Solistica 4,060", value="4060", pages=[117]),
                 dict(label="Plásticos 3,165", value="3165", pages=[117])],
         formula="a+b+c"),
    dict(id="Q97", level=3,
         question="¿Cuánto sumaron las dos recompras aceleradas de acciones (ASR) de FEMSA en 2024?",
         expected="US$ 1,000 millones (400 + 600).",
         expected_numbers=["1000"], evidence_pages=[115, 116],
         inputs=[dict(label="ASR marzo US$ 400M", value="400", pages=[115, 116]),
                 dict(label="ASR junio US$ 600M", value="600", pages=[115, 116])],
         formula="a+b"),
    dict(id="Q98", level=3,
         question="¿Por cuántos millones superaba Spin Premia a Spin by OXXO en usuarios activos en 2024?",
         expected="16.0 millones (24.6 − 8.6).",
         expected_numbers=["16.0"], evidence_pages=[13, 33, 35],
         inputs=[dict(label="Spin Premia 24.6", value="24.6", pages=[13, 33, 35]),
                 dict(label="Spin by OXXO 8.6", value="8.6", pages=[13, 33, 35])],
         formula="a-b"),
    dict(id="Q99", level=3,
         question="¿Cuántas horas de capacitación sumaron cultura/liderazgo y conocimiento técnico en 2024, y qué porcentaje del total representan?",
         expected="4,979,608 horas (737,386 + 4,242,222), el 80.6% del total de 6,181,088.",
         expected_numbers=["4979608", "80.6"], evidence_pages=[122, 123],
         inputs=[dict(label="Cultura y liderazgo 737,386", value="737386", pages=[123]),
                 dict(label="Conocimiento técnico 4,242,222", value="4242222", pages=[123]),
                 dict(label="Total capacitación 6,181,088", value="6181088", pages=[122])],
         formula="a+b",
         formula2="round((a+b)/c*100,1)",
         notes="Dos resultados: la suma (4,979,608) y su participación (80.6%)."),
    dict(id="Q100", level=3,
         question="¿Cuántas tiendas OXXO había en Chile y Perú juntas al cierre de 2024?",
         expected="438 tiendas (235 + 203).",
         expected_numbers=["438"], evidence_pages=[21, 22],
         inputs=[dict(label="Chile 235", value="235", pages=[21]),
                 dict(label="Perú 203", value="203", pages=[21])],
         formula="a+b"),
]

NEW = NEW + NEW_L2 + NEW_L3


# ---------------------------------------------------------------------------
# Verificacion + escritura
# ---------------------------------------------------------------------------

def fmt_num(v):
    v = float(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return str(round(v, 1))


def verify_question(pages, q, failures, warnings):
    """Verifica una pregunta nueva. Devuelve True si pasa."""
    ok = True
    ev = q["evidence_pages"]
    # Nivel 3 (con insumos): el resultado es calculado, no impreso en el
    # informe; no se exige en el corpus. Lo valida la formula de abajo.
    check_expected = not q.get("inputs")
    for n in q.get("expected_numbers", []) if check_expected else []:
        good, found = check_number(pages, n, ev)
        if not good:
            failures.append(f"{q['id']}: numero esperado {n} no esta en el corpus "
                            f"o no en las paginas de evidencia {ev} "
                            f"(aparece en: {found or 'ninguna'})")
            ok = False
    for kw in q.get("expected_keywords", []):
        hit = [p for p in ev if kw.lower() in pages.get(p, "").lower()]
        if not hit:
            failures.append(f"{q['id']}: keyword {kw!r} no aparece en las "
                            f"paginas de evidencia {ev}")
            ok = False
    for inp in q.get("inputs", []):
        good, found = check_number(pages, inp["value"], inp["pages"])
        if not good:
            failures.append(f"{q['id']}: insumo {inp['label']} no esta en el "
                            f"corpus o no en sus paginas {inp['pages']} "
                            f"(aparece en: {found or 'ninguna'})")
            ok = False
    if q.get("formula"):
        env = {"round": round}
        for i, inp in enumerate(q["inputs"]):
            env[chr(ord("a") + i)] = float(norm_digits(inp["value"]))
        try:
            res = fmt_num(eval(q["formula"], {"__builtins__": {}}, env))
        except Exception as e:  # noqa: BLE001
            failures.append(f"{q['id']}: formula no evaluable: {e}")
            return False
        exp_vals = [float(norm_digits(n)) for n in q["expected_numbers"]]
        if not any(abs(float(res) - e) <= 0.05 for e in exp_vals):
            failures.append(f"{q['id']}: la formula {q['formula']} da {res}, "
                            f"pero el esperado es {q['expected_numbers']}")
            ok = False
        if q.get("formula2"):
            try:
                res2 = fmt_num(eval(q["formula2"], {"__builtins__": {}}, env))
            except Exception as e:  # noqa: BLE001
                failures.append(f"{q['id']}: formula2 no evaluable: {e}")
                return False
            exp_vals = [float(norm_digits(n)) for n in q["expected_numbers"]]
            if not any(abs(float(res2) - e) <= 0.05 for e in exp_vals):
                failures.append(f"{q['id']}: la formula2 {q['formula2']} da "
                                f"{res2}, pero el esperado es "
                                f"{q['expected_numbers']}")
                ok = False
    return ok


def public_question(q):
    """La pregunta tal como va al JSON (sin campos internos del builder)."""
    out = {
        "id": q["id"],
        "level": q["level"],
        "question": q["question"],
        "expected": q["expected"],
        "evidence_pages": q["evidence_pages"],
        "type": "numeric" if q.get("expected_numbers") else "keyword",
    }
    if q.get("expected_numbers"):
        out["expected_numbers"] = q["expected_numbers"]
    if q.get("expected_keywords"):
        out["expected_keywords"] = q["expected_keywords"]
    if q.get("ground_truth_note"):
        out["ground_truth_note"] = q["ground_truth_note"]
    if q.get("notes"):
        out["notes"] = q["notes"]
    return out


def gt_entry(q):
    """Entrada de ground_truth_evidence para una pregunta nueva."""
    items = []
    for inp in q.get("inputs", []):
        items.append({"label": f"{inp['label']} (insumo para el calculo)",
                      "pages": sorted(inp["pages"])})
    if q.get("inputs"):
        all_pages = sorted({p for inp in q["inputs"] for p in inp["pages"]})
        items.append({"label": f"Resultado calculado: {q['expected']}",
                      "pages": all_pages})
    else:
        for n in q.get("expected_numbers", []):
            items.append({"label": f"{n} ({q['expected'][:60]})",
                          "pages": sorted(q["evidence_pages"])})
        for kw in q.get("expected_keywords", []):
            items.append({"label": f"keyword: {kw}",
                          "pages": sorted(q["evidence_pages"])})
    return {"id": q["id"], "question": q["question"], "evidence": items}


def main():
    verify_only = "--verify-only" in sys.argv
    pages = load_corpus()
    print(f"corpus: {len(pages)} paginas")

    orig = json.load(open(EVAL12))
    orig_qs = orig["questions"]
    assert len(orig_qs) == 12, f"se esperaban 12 originales, hay {len(orig_qs)}"
    ids = [q["id"] for q in orig_qs] + [q["id"] for q in NEW]
    assert len(ids) == len(set(ids)) == 100, "IDs duplicados o total != 100"

    # --- verificar las 88 nuevas ---
    failures, warnings = [], []
    for q in NEW:
        verify_question(pages, q, failures, warnings)

    # --- verificar las 12 originales (con las excepciones documentadas) ---
    known_missing = {"Q3": {"1236"}, "Q7": {"27.5", "24.9"}, "Q2": {"24.9"},
                   "Q9": {"75.1", "39.3", "35.8"}}  # Q9: participaciones calculadas, fuera del corpus (documentado)
    all_corpus_nums = set()
    for t in pages.values():
        all_corpus_nums |= numbers_on_page(t)
    for q in orig_qs:
        for n in q.get("expected_numbers", []):
            if n in known_missing.get(q["id"], set()):
                continue
            if norm_digits(n) in all_corpus_nums:
                continue
            # tolerancia de redondeo del scorer (+-0.1%): Q6 usa 307,200/279,800
            v = float(norm_digits(n))
            if any(abs(float(x) - v) <= max(1.0, abs(v) * 0.001)
                   for x in all_corpus_nums if x.replace(".", "", 1).isdigit()):
                continue
            failures.append(f"{q['id']} (original): numero {n} ya no se "
                            f"encuentra en el corpus")

    # --- duplicados entre las nuevas (Jaccard >= 0.80) ---
    import unicodedata
    def tokens(s):
        s = unicodedata.normalize("NFKD", s.lower())
        s = "".join(c for c in s if not unicodedata.combining(c))
        return set(re.findall(r"[a-z0-9]+", s))
    toks = {q["id"]: tokens(q["question"]) for q in NEW}
    # Pares revisados a mano: misma redaccion, cifra distinta (EBITDA vs
    # margen EBITDA; utilidad neta vs bruta; activos vs pasivos; OXXO total
    # vs por pais; mayor vs menor margen). No son duplicados.
    KNOWN_SIMILAR = {frozenset(("Q14", "Q45")), frozenset(("Q15", "Q17")),
                     frozenset(("Q18", "Q45")), frozenset(("Q19", "Q48")),
                     frozenset(("Q27", "Q42")), frozenset(("Q27", "Q43")),
                     frozenset(("Q28", "Q30")), frozenset(("Q42", "Q43")),
                     frozenset(("Q50", "Q51"))}
    for i, a in enumerate(NEW):
        for b in NEW[i + 1:]:
            inter = toks[a["id"]] & toks[b["id"]]
            union = toks[a["id"]] | toks[b["id"]]
            if union and len(inter) / len(union) >= 0.80 and \
                    frozenset((a["id"], b["id"])) not in KNOWN_SIMILAR:
                warnings.append(f"posible duplicado: {a['id']} ~ {b['id']} "
                                f"(Jaccard {len(inter)/len(union):.2f})")

    # --- solape con el dataset de entrenamiento (para NB03) ---
    leak_ids = set()
    for qa_file in ("reflection_qa.jsonl", "reflection_qa_v2.jsonl",
                    "reflection_qa_v3.jsonl"):
        p = HERE / "memo" / qa_file
        if not p.exists():
            continue
        train_qs = [tokens(json.loads(l)["question"]) for l in open(p)]
        for q in NEW:
            t = toks[q["id"]]
            if any(t & tq and len(t & tq) / len(t | tq) >= 0.80 for tq in train_qs):
                leak_ids.add(q["id"])
    print(f"solape con training (Jaccard >= 0.80, preguntas nuevas): "
          f"{len(leak_ids)} {sorted(leak_ids)} "
          f"(NB03 las excluye al regenerar con el set de 100)")

    for w in warnings:
        print("AVISO:", w)
    if failures:
        print(f"\nFALLOS: {len(failures)}")
        for f in failures:
            print("  -", f)
        sys.exit(1)

    from collections import Counter
    split = Counter(q["level"] for q in orig_qs + NEW)
    print(f"verificacion OK: 100 preguntas (L1 {split[1]}, L2 {split[2]}, "
          f"L3 {split[3]}), 0 fallos")

    if verify_only:
        return

    # --- escribir eval_questions_100.json ---
    # Las 12 originales se copian con todos sus campos intactos; solo se
    # les ANADEN evidence_pages (union de paginas de su entrada en
    # ground_truth_evidence.json; Q9 usa las paginas de sus insumos) y type,
    # para que las 100 entradas tengan el mismo esquema.
    gt12_map = {q["id"]: q for q in json.load(open(GT12))["questions"]}
    orig_out = []
    for q in orig_qs:
        qq = dict(q)
        gt_pages = sorted({p for it in gt12_map[q["id"]].get("evidence", [])
                           for p in it["pages"]})
        if not gt_pages:  # Q9: participaciones calculadas
            gt_pages = [11, 108, 109, 112]
        qq["evidence_pages"] = gt_pages
        qq["type"] = "numeric" if q.get("expected_numbers") else "keyword"
        orig_out.append(qq)
    all_qs = orig_out + [public_question(q) for q in NEW]
    out = {
        "note": ("100 preguntas de evaluacion sobre el informe anual FEMSA "
                 "2024 (edicion en ingles). Q1-Q12: el set original, sin "
                 "cambios. Q13-Q100: 88 nuevas, cada cifra verificada por "
                 "grep literal en pipeline/data/chunks.jsonl; las de nivel 3 "
                 "son agregaciones calculables desde cifras del informe "
                 "(insumos y formulas en memo/build_eval_100.py). "
                 "Excepciones documentadas: Q3 (1,236 no aparece; el "
                 "informe dice 1,596) y Q7 (+27.5%/+24.9% vienen del "
                 "comunicado 4Q24, no del informe; p. 34 dice +32.8%/+31.5%, "
                 "que es lo que pregunta Q49)."),
        "questions": all_qs,
        "source": orig.get("source", "FEMSA 2024 Integrated Annual Report"),
    }
    json.dump(out, open(OUT100, "w"), ensure_ascii=False, indent=2,
              sort_keys=True)
    print(f"-> {OUT100}")

    # --- escribir ground_truth_evidence_100.json ---
    gt12 = json.load(open(GT12))
    gt_qs = [dict(q) for q in gt12["questions"]] + [gt_entry(q) for q in NEW]
    gt_out = {
        "note": ("Ground-truth evidence para las 100 preguntas de "
                 "evaluacion, derivada por grep literal en el corpus "
                 "(pipeline/data/chunks.jsonl, 151 paginas, edicion en "
                 "ingles del informe anual FEMSA 2024). Q1-Q12: entradas "
                 "originales, sin cambios. Q13-Q100: un item por cifra "
                 "esperada (nivel 1-2) o por insumo del calculo (nivel 3). "
                 "Ver memo/build_eval_100.py para la verificacion."),
        "questions": gt_qs,
    }
    json.dump(gt_out, open(GT100, "w"), ensure_ascii=False, sort_keys=True)
    print(f"-> {GT100}")


if __name__ == "__main__":
    main()
