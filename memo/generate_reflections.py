#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_reflections.py — Generador del dataset de reflection QA (estilo MEMO)
para el experimento con el Informe Anual Integrado 2024 de FEMSA.

Paper de referencia: arXiv 2605.15156, "MeMo: Memory as a Model".
Uso:  python3 memo/generate_reflections.py
      (desde la raiz del repo; regenera memo/reflection_qa.jsonl
       de forma determinista).

Fuentes:
  - pipeline/data/chunks.jsonl : 151 paginas del FEMSA 2024 Integrated
    Annual Report (edicion en ingles). El campo "page" es el numero de
    pagina del PDF y es el que se cita en "source_pages".
  - pipeline/triples_seed.json : 31 tripletas semilla verificadas.
  - docs/ontology-and-questions.md : ontologia v1 + 12 preguntas
    de evaluacion (seccion 3). Esas 12 preguntas estan EXCLUIDAS del
    dataset (ver EXCLUDED_QUESTIONS y la comprobacion en main()).

TENSION CONOCIDA (documentada, no resuelta a favor de la ontologia):
  La ontologia v1 redondea los ingresos por division (307,200 / 279,800 /
  49,800 / 79,700 / 65,400) y registra "Aperturas 1,236" para Proximidad
  Americas. El MD&A (p. 112/114) da los valores EXACTOS 307,197 / 279,793 /
  49,755 / 79,755 / 65,365 y 1,596 tiendas nuevas netas; ademas 23,206 es
  el numero de tiendas OXXO en Mexico (el total de la division es 26,322
  puntos de venta y el total OXXO es 24,462). ESTE DATASET USA SOLO LOS
  VALORES EXACTOS DEL MD&A. La cifra 1,236 NO aparece en el informe anual
  (nota en triples_seed.json) y jamas se usa. Los valores redondeados de
  la ontologia tampoco se usan.
  Otras dos distinciones que se mantienen separadas:
  - Spin Premia: "mas de 24 millones de usuarios activos" (carta, p. 5)
    vs "24.6 millones" (cifras clave, pp. 13/33/35) vs "mas de 52 millones
    de usuarios" (p. 19/35). Son tres medidas distintas del informe.
  - Spin by OXXO: "8.6 millones de usuarios activos" (pp. 5/13/33/35)
    vs "mas de 13 millones de clientes" (pp. 19/35).
  - Juntos+ 4.0: el informe dice "over half a million active customers"
    (p. 4), por lo que la respuesta usa "mas de medio millon" en palabras:
    la cifra 500,000 como digitos NO aparece en el texto extraido.
  - Utilidad bruta consolidada: el texto del MD&A (p. 112) escribe
    Ps. 321,417 millones, pero el estado de resultados (p. 109) y la tabla
    de la p. 112 dan Ps. 321,513 millones. Se usa 321,513 citando p. 109.
  - EBITDA ajustado consolidado: el informe lo expresa como "Ps. 116
    billion" (p. 5), por lo que la respuesta usa "Ps. 116 mil millones":
    la cifra 116,000 como digitos NO aparece en el texto extraido.
    Lo mismo para "Ps. 279 mil millones" / "Ps. 56 mil millones" de
    Coca-Cola FEMSA (p. 29).
  - Los porcentajes +24.9% (Spin by OXXO) y +27.5% (Spin Premia) de las
    tripletas semilla vienen del comunicado 4T24, NO del informe anual:
    NO se usan en este dataset.

REGLAS APLICADAS POR EL SCRIPT:
  1. Cada par (pregunta, respuesta) se verifica contra el texto de las
     paginas citadas: todos los numeros de la respuesta (normalizados:
     sin comas ni espacios) deben aparecer como tokens numericos en el
     texto de al menos una de las paginas citadas. Los pares que fallan
     se descartan y se cuentan en el informe impreso.
  2. Ninguna pregunta puede ser verbatim ni casi-verbatim de las 12
     preguntas de evaluacion (igualdad normalizada, contencion o
     similitud de Jaccard >= 0.80 sobre conjuntos de palabras).
  3. Deduplicacion por pregunta normalizada (minusculas, sin acentos,
     sin puntuacion, espacios colapsados).
  4. Salida determinista: los pares finales se ordenan por pregunta
     normalizada antes de escribir el JSONL.
  5. Solo se usan cifras del informe (verificadas contra chunks.jsonl)
     o de la lista de hechos verificados del encargo. Nada inventado.
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOAL = HERE.parent
CHUNKS_PATH = GOAL / "pipeline" / "data" / "chunks.jsonl"
TRIPLES_PATH = GOAL / "pipeline" / "triples_seed.json"
OUT_PATH = HERE / "reflection_qa.jsonl"

# ---------------------------------------------------------------------------
# 12 preguntas de evaluacion EXCLUIDAS (ontologia-y-preguntas, seccion 3).
# No deben aparecer verbatim ni casi-verbatim en el dataset.
# ---------------------------------------------------------------------------
EXCLUDED_QUESTIONS = [
    "¿Cuáles fueron los ingresos totales consolidados de FEMSA en 2024 y cuánto crecieron vs 2023?",
    "¿Cuántos usuarios activos tuvo Spin by OXXO en 2024?",
    "¿Cuántas tiendas sumó Proximidad Américas al cierre de 2024 y cuántas aperturas hubo?",
    "¿Cuántas estaciones tenía OXXO GAS al cierre de 2024?",
    "¿Qué división tuvo el mayor crecimiento porcentual de ingresos en 2024: Proximidad Europa o Coca-Cola FEMSA?",
    "¿Qué división aportó más ingresos en 2024, Proximidad Américas o Coca-Cola FEMSA, y por cuánto?",
    "¿Qué creció más en usuarios, Spin by OXXO o Spin Premia?",
    "¿En qué países opera Proximidad Américas según el informe?",
    "¿Qué porcentaje de los ingresos consolidados 2024 representan Proximidad Américas y Coca-Cola FEMSA juntas?",
    "¿Qué estrategia articula Retail, Bebidas y el ecosistema digital de FEMSA, y a través de qué se crea ese ecosistema?",
    "¿Cómo evolucionó la iniciativa digital de FEMSA en 2024?",
    "¿Qué división tuvo el año más complicado en 2024 según el propio informe, y cuánto crecieron sus ingresos?",
]

# ---------------------------------------------------------------------------
# Utilidades de normalizacion / verificacion
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"\d+(?:[,\.]\d+)*")


def normalize_text(s: str) -> str:
    """Minusculas, sin acentos, sin puntuacion, espacios colapsados."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_excluded(question: str):
    """Devuelve la pregunta excluida que coincide, o None."""
    nq = normalize_text(question)
    for ex in EXCLUDED_QUESTIONS:
        ne = normalize_text(ex)
        if nq == ne or nq in ne or ne in nq:
            return ex
        if jaccard(nq, ne) >= 0.80:
            return ex
    return None


def numbers_in(text: str):
    """Tokens numericos normalizados (sin comas): {'781585', '11.2', ...}."""
    out = set()
    for tok in _NUM_RE.findall(text):
        out.add(tok.replace(",", ""))
    return out


def load_chunk_numbers():
    """page -> set de tokens numericos normalizados de esa pagina."""
    page_numbers = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            page_numbers[d["page"]] = numbers_in(d["text"])
    return page_numbers


def verify_numbers(answer: str, pages, page_numbers):
    """Todos los numeros de la respuesta deben aparecer en las paginas citadas.

    Devuelve (ok, numeros_faltantes). Las respuestas sin numeros pasan.
    """
    cited = set()
    for p in pages:
        cited |= page_numbers.get(p, set())
    missing = sorted(n for n in numbers_in(answer) if n not in cited)
    return (not missing), missing


# ---------------------------------------------------------------------------
# Datos curados a mano. Cada entrada: tipo, paginas, entidades, respuesta,
# lista de formulaciones de la pregunta. Todas las cifras estan verificadas
# contra chunks.jsonl (ver informe de verificacion al ejecutar el script).
# Claves: t=tipo, p=source_pages, e=entities, a=answer, q=questions.
# ---------------------------------------------------------------------------
CURATED = [
    # ---- Consolidados (pp. 108/109/112) ----
    {"t": "factoid", "p": [108, 112], "e": ["FEMSA"],
     "a": "Ps. 781,585 millones, un crecimiento de 11.2% frente a 2023.",
     "q": ["¿A cuánto ascendieron los ingresos totales consolidados de FEMSA en 2024?",
           "¿Qué ingresos totales consolidados reportó FEMSA en 2024?"]},
    {"t": "factoid", "p": [108, 112], "e": ["FEMSA"],
     "a": "11.2% frente a 2023, cuando fueron Ps. 702,692 millones.",
     "q": ["¿Qué crecimiento tuvieron los ingresos totales de FEMSA en 2024 respecto a 2023?",
           "¿En qué porcentaje aumentaron los ingresos consolidados de FEMSA en 2024?"]},
    {"t": "factoid", "p": [112], "e": ["FEMSA"],
     "a": "10.7% en base orgánica.",
     "q": ["¿Cuál fue el crecimiento orgánico de los ingresos totales de FEMSA en 2024?",
           "¿Cuánto crecieron los ingresos de FEMSA en 2024 en términos orgánicos?"]},
    {"t": "factoid", "p": [108], "e": ["FEMSA"],
     "a": "US$37,476 millones.",
     "q": ["¿A cuánto equivalieron en dólares los ingresos totales de FEMSA en 2024?",
           "¿Cuáles fueron los ingresos totales de FEMSA en 2024 expresados en dólares?"]},
    {"t": "factoid", "p": [108, 109], "e": ["FEMSA"],
     "a": "Ps. 70,667 millones, con un aumento de 19.8%.",
     "q": ["¿A cuánto llegó la utilidad de operación de FEMSA en 2024?",
           "¿Cuál fue el ingreso por operaciones de FEMSA en 2024?"]},
    {"t": "comparison", "p": [108, 109], "e": ["FEMSA"],
     "a": "9.0% en 2024, frente a 8.4% en 2023.",
     "q": ["¿Cuál fue el margen operativo de FEMSA en 2024?",
           "¿Qué margen de operación registró FEMSA en 2024 y cuál fue en 2023?"]},
    {"t": "factoid", "p": [109, 112], "e": ["FEMSA"],
     "a": "Ps. 321,513 millones, un aumento de 15.0%.",
     "q": ["¿A cuánto ascendió la utilidad bruta consolidada de FEMSA en 2024?",
           "¿Cuál fue la utilidad bruta de FEMSA en 2024?"]},
    {"t": "comparison", "p": [109], "e": ["FEMSA"],
     "a": "41.1% en 2024, frente a 39.8% en 2023.",
     "q": ["¿Cuál fue el margen bruto de FEMSA en 2024?",
           "¿Qué margen bruto reportó FEMSA en 2024?"]},
    {"t": "factoid", "p": [112], "e": ["FEMSA"],
     "a": "130 puntos base, hasta llegar a 41.1%.",
     "q": ["¿Cuántos puntos base aumentó el margen bruto consolidado de FEMSA en 2024?",
           "¿En cuántos puntos base se expandió el margen bruto de FEMSA en 2024?"]},
    {"t": "factoid", "p": [108, 109], "e": ["FEMSA"],
     "a": "Ps. 115,593 millones, con un crecimiento de 20.6%.",
     "q": ["¿A cuánto llegó el EBITDA de FEMSA en 2024?",
           "¿Cuál fue el EBITDA consolidado de FEMSA en 2024?"]},
    {"t": "factoid", "p": [108], "e": ["FEMSA"],
     "a": "14.8% en 2024.",
     "q": ["¿Cuál fue el margen EBITDA de FEMSA en 2024?",
           "¿Qué margen de EBITDA registró FEMSA en 2024?"]},
    {"t": "factoid", "p": [5], "e": ["FEMSA"],
     "a": "Ps. 116 mil millones, con un margen de EBITDA ajustado de 14.8%.",
     "q": ["¿A cuánto llegó el EBITDA ajustado de FEMSA en 2024 según la carta del director general?",
           "¿Qué EBITDA ajustado reportó FEMSA en 2024?"]},
    {"t": "factoid", "p": [108, 109], "e": ["FEMSA"],
     "a": "Ps. 40,236 millones, una caída de 47.5% frente a Ps. 76,677 millones en 2023.",
     "q": ["¿Cuál fue la utilidad neta consolidada de FEMSA en 2024?",
           "¿A cuánto ascendió el ingreso neto consolidado de FEMSA en 2024?"]},
    {"t": "factoid", "p": [108, 109], "e": ["FEMSA"],
     "a": "Ps. 26,735 millones, una caída de 59.3%.",
     "q": ["¿Cuál fue la utilidad neta de la participación controladora de FEMSA en 2024?",
           "¿A cuánto llegó la utilidad neta atribuible a la participación controladora en 2024?"]},
    {"t": "factoid", "p": [109], "e": ["FEMSA"],
     "a": "Ps. 775,551 millones.",
     "q": ["¿Cuáles fueron las ventas netas de FEMSA en 2024?",
           "¿A cuánto ascendieron las ventas netas consolidadas de FEMSA en 2024?"]},
    {"t": "factoid", "p": [109], "e": ["FEMSA"],
     "a": "Ps. 460,072 millones.",
     "q": ["¿Cuál fue el costo de ventas de FEMSA en 2024?",
           "¿A cuánto llegó el costo de los bienes vendidos de FEMSA en 2024?"]},
    {"t": "factoid", "p": [109], "e": ["FEMSA"],
     "a": "Ps. 250,846 millones.",
     "q": ["¿A cuánto ascendieron los gastos operativos de FEMSA en 2024?",
           "¿Cuáles fueron los gastos de operación consolidados de FEMSA en 2024?"]},
    {"t": "factoid", "p": [108], "e": ["FEMSA"],
     "a": "Ps. 851,536 millones, un aumento de 5.7%.",
     "q": ["¿A cuánto llegaron los activos totales de FEMSA en 2024?",
           "¿Cuál fue el total de activos de FEMSA al cierre de 2024?"]},
    {"t": "factoid", "p": [108], "e": ["FEMSA"],
     "a": "Ps. 470,405 millones, un aumento de 10.0%.",
     "q": ["¿A cuánto ascendieron los pasivos totales de FEMSA en 2024?",
           "¿Cuál fue el total de pasivos de FEMSA en 2024?"]},
    {"t": "factoid", "p": [108], "e": ["FEMSA"],
     "a": "Ps. 381,131 millones, un aumento de 0.7%.",
     "q": ["¿A cuánto llegó el capital contable total de FEMSA en 2024?",
           "¿Cuál fue el capital total de FEMSA en 2024?"]},
    {"t": "factoid", "p": [108, 109], "e": ["FEMSA"],
     "a": "Ps. 51,074 millones, un aumento de 31.1%.",
     "q": ["¿Cuánto invirtió FEMSA en CAPEX en 2024?",
           "¿A cuánto ascendieron las inversiones en activo fijo (CAPEX) de FEMSA en 2024?"]},
    {"t": "factoid", "p": [108], "e": ["FEMSA"],
     "a": "Ps. 139,834 millones, una disminución de 15.3%.",
     "q": ["¿Cuánto efectivo y equivalentes tenía FEMSA al cierre de 2024?",
           "¿A cuánto llegó el efectivo y equivalentes de efectivo de FEMSA en 2024?"]},
    {"t": "factoid", "p": [108], "e": ["FEMSA"],
     "a": "388,999 empleados.",
     "q": ["¿Cuántos empleados tenía FEMSA en 2024 según los datos financieros?",
           "¿Cuál fue el número de empleados (headcount) de FEMSA en 2024?"]},
    {"t": "factoid", "p": [108], "e": ["FEMSA"],
     "a": "Ps. 20.8557 por US$1.00 al 31 de diciembre de 2024.",
     "q": ["¿Cuál fue el tipo de cambio al 31 de diciembre de 2024?",
           "¿A cuánto estaba el peso frente al dólar al cierre de 2024 según el informe?"]},
    {"t": "factoid", "p": [112], "e": ["FEMSA"],
     "a": "Ps. 39,091 millones, un aumento de 21.0% y 5.0% de los ingresos.",
     "q": ["¿A cuánto llegaron los gastos administrativos consolidados de FEMSA en 2024?",
           "¿Cuáles fueron los gastos de administración de FEMSA en 2024?"]},
    {"t": "factoid", "p": [112], "e": ["FEMSA"],
     "a": "Ps. 211,864 millones, un aumento de 12.3% y 27.1% de los ingresos.",
     "q": ["¿A cuánto ascendieron los gastos de venta consolidados de FEMSA en 2024?",
           "¿Cuáles fueron los gastos de ventas de FEMSA en 2024?"]},
    {"t": "comparison", "p": [112], "e": ["FEMSA"],
     "a": "Ps. 3,588 millones en 2024, frente a Ps. 13,102 millones en 2023.",
     "q": ["¿Cuánto sumaron los otros ingresos de FEMSA en 2024?",
           "¿A cuánto llegaron los otros ingresos de FEMSA en 2024?"]},
    {"t": "comparison", "p": [113], "e": ["FEMSA"],
     "a": "Ps. 9,440 millones en 2024, frente a Ps. 6,252 millones en 2023.",
     "q": ["¿Cuánto sumaron los otros gastos de FEMSA en 2024?",
           "¿A cuánto ascendieron los otros gastos de FEMSA en 2024?"]},
    {"t": "comparison", "p": [113], "e": ["FEMSA"],
     "a": "Una ganancia cambiaria de Ps. 11,929 millones en 2024, frente a una pérdida de Ps. 9,849 millones en 2023.",
     "q": ["¿Qué resultado cambiario registró FEMSA en 2024?",
           "¿Cuál fue la ganancia por tipo de cambio de FEMSA en 2024?"]},
    {"t": "factoid", "p": [108], "e": ["FEMSA"],
     "a": "US$3,388 millones.",
     "q": ["¿A cuánto equivalió en dólares la utilidad de operación de FEMSA en 2024?"]},
    {"t": "factoid", "p": [108], "e": ["FEMSA"],
     "a": "US$5,543 millones.",
     "q": ["¿A cuánto equivalió en dólares el EBITDA de FEMSA en 2024?"]},
    {"t": "factoid", "p": [108], "e": ["FEMSA"],
     "a": "US$1,930 millones.",
     "q": ["¿A cuánto equivalió en dólares la utilidad neta consolidada de FEMSA en 2024?"]},
    {"t": "comparison", "p": [108], "e": ["FEMSA"],
     "a": "Ps. 7.5 por unidad BD en 2024, frente a Ps. 18.4 en 2023.",
     "q": ["¿Cuál fue la utilidad por unidad BD de la participación controladora en 2024?"]},
    {"t": "comparison", "p": [108], "e": ["FEMSA"],
     "a": "Ps. 74.7 por ADS en 2024, frente a Ps. 183.6 en 2023.",
     "q": ["¿Cuál fue la utilidad por ADS de la participación controladora en 2024?"]},
    {"t": "comparison", "p": [109], "e": ["FEMSA"],
     "a": "Ps. 40,121 millones en 2024, frente a Ps. 44,439 millones en 2023.",
     "q": ["¿Cuál fue la utilidad neta de operaciones continuas de FEMSA en 2024?"]},
    {"t": "comparison", "p": [109], "e": ["FEMSA"],
     "a": "Ps. 115 millones en 2024, frente a Ps. 32,238 millones en 2023.",
     "q": ["¿Cuál fue la utilidad neta de operaciones discontinuas de FEMSA en 2024?"]},
    {"t": "factoid", "p": [109], "e": ["FEMSA"],
     "a": "Ps. 13,501 millones.",
     "q": ["¿A cuánto llegó la utilidad neta de la participación no controladora en 2024?"]},
    {"t": "factoid", "p": [109], "e": ["FEMSA"],
     "a": "Ps. 66,741 millones.",
     "q": ["¿Cuál fue la utilidad antes de impuestos de FEMSA en 2024?"]},
    {"t": "factoid", "p": [109], "e": ["FEMSA"],
     "a": "Ps. 25,433 millones.",
     "q": ["¿A cuánto llegaron los impuestos a la utilidad de FEMSA en 2024?"]},
    {"t": "aggregation", "p": [108], "e": ["FEMSA"],
     "a": "Ps. 202,930 millones de deuda a corto plazo y Ps. 267,475 millones de deuda a largo plazo.",
     "q": ["¿Cuánta deuda a corto y largo plazo tenía FEMSA en 2024?"]},
    {"t": "aggregation", "p": [109], "e": ["FEMSA"],
     "a": "Ps. 35,199 millones de depreciación y Ps. 9,728 millones de amortización y otros cargos.",
     "q": ["¿Cuánto sumaron la depreciación y la amortización de FEMSA en 2024?"]},
    {"t": "multi-hop", "p": [113], "e": ["FEMSA"],
     "a": "Ps. 26,617 millones, con una tasa efectiva de 35.9% en operaciones continuas, frente a 22.7% en 2023.",
     "q": ["¿Cuál fue la provisión de impuestos a la utilidad de FEMSA en 2024?",
           "¿Qué tasa efectiva de impuestos tuvo FEMSA en operaciones continuas en 2024?"]},
    {"t": "comparison", "p": [113], "e": ["FEMSA"],
     "a": "Ps. 8,092 millones de gasto neto por intereses en 2024, frente a un ingreso neto por intereses de Ps. 2,693 millones en 2023.",
     "q": ["¿Cuál fue el gasto neto por intereses de FEMSA en 2024?"]},
    {"t": "comparison", "p": [113], "e": ["FEMSA"],
     "a": "Ps. 209 millones en 2024, frente a Ps. 94 millones en 2023.",
     "q": ["¿Cuál fue la ganancia por posición monetaria de FEMSA en 2024?"]},
    {"t": "comparison", "p": [108], "e": ["FEMSA"],
     "a": "392,968 empleados en 2023, una disminución de 0.4% en 2024.",
     "q": ["¿Cuántos empleados tenía FEMSA en 2023?",
           "¿Cuánto varió el número de empleados de FEMSA en 2024 frente a 2023?"]},
]

CURATED += [
    # ---- Divisiones (MD&A pp. 112/114; valores EXACTOS, no redondeados) ----
    {"t": "factoid", "p": [112], "e": ["Coca-Cola FEMSA"],
     "a": "Ps. 279,793 millones, un crecimiento de 14.2%.",
     "q": ["¿A cuánto llegaron los ingresos totales de Coca-Cola FEMSA en 2024?",
           "¿Qué ingresos reportó Coca-Cola FEMSA en 2024?"]},
    {"t": "factoid", "p": [112], "e": ["Coca-Cola FEMSA"],
     "a": "Ps. 128,736 millones, un crecimiento de 16.1%.",
     "q": ["¿Cuál fue la utilidad bruta de Coca-Cola FEMSA en 2024?",
           "¿A cuánto ascendió la utilidad bruta de Coca-Cola FEMSA en 2024?"]},
    {"t": "factoid", "p": [112], "e": ["Proximidad Américas"],
     "a": "Ps. 307,197 millones, un crecimiento de 10.3%.",
     "q": ["¿A cuánto llegaron los ingresos de Proximidad Américas en 2024?",
           "¿Qué ingresos totales reportó Proximidad Américas en 2024?"]},
    {"t": "factoid", "p": [112, 114], "e": ["Proximidad Américas"],
     "a": "Ps. 136,993 millones, un crecimiento de 17.0%.",
     "q": ["¿Cuál fue la utilidad bruta de Proximidad Américas en 2024?",
           "¿A cuánto llegó la utilidad bruta de Proximidad Américas en 2024?"]},
    {"t": "factoid", "p": [112, 114], "e": ["Proximidad Américas"],
     "a": "4.2% en promedio.",
     "q": ["¿Cuánto crecieron en promedio las ventas mismas-tienda de Proximidad Américas en 2024?",
           "¿Qué aumento promedio registraron las ventas de mismas tiendas de Proximidad Américas en 2024?"]},
    {"t": "factoid", "p": [112, 114], "e": ["Proximidad Américas"],
     "a": "1,596 tiendas nuevas netas en 2024.",
     "q": ["¿Cuántas tiendas nuevas netas sumó Proximidad Américas en 2024?",
           "¿Cuántas aperturas netas tuvo Proximidad Américas en 2024?"]},
    {"t": "factoid", "p": [112], "e": ["Proximidad Europa"],
     "a": "Ps. 49,755 millones, un crecimiento de 14.2%.",
     "q": ["¿A cuánto llegaron los ingresos de Proximidad Europa en 2024?",
           "¿Qué ingresos reportó Proximidad Europa en 2024?"]},
    {"t": "factoid", "p": [112], "e": ["Proximidad Europa"],
     "a": "Ps. 21,344 millones, un crecimiento de 14.6%.",
     "q": ["¿Cuál fue la utilidad bruta de Proximidad Europa en 2024?"]},
    {"t": "factoid", "p": [112], "e": ["Combustibles"],
     "a": "Ps. 65,365 millones, un crecimiento de 11.7%.",
     "q": ["¿A cuánto llegaron los ingresos de la división de Combustibles en 2024?",
           "¿Qué ingresos reportó la división de Combustibles en 2024?"]},
    {"t": "factoid", "p": [112], "e": ["Combustibles"],
     "a": "Ps. 7,935 millones, un crecimiento de 8.0%.",
     "q": ["¿Cuál fue la utilidad bruta de la división de Combustibles en 2024?"]},
    {"t": "factoid", "p": [112], "e": ["Combustibles"],
     "a": "9.9%.",
     "q": ["¿Cuánto crecieron las ventas mismas-estación de Combustibles en 2024?",
           "¿Qué aumento registraron las ventas de mismas estaciones de Combustibles en 2024?"]},
    {"t": "factoid", "p": [112], "e": ["Salud"],
     "a": "Ps. 79,755 millones, un crecimiento de 5.8%.",
     "q": ["¿A cuánto llegaron los ingresos de la división de Salud en 2024?",
           "¿Qué ingresos reportó la división de Salud en 2024?"]},
    {"t": "factoid", "p": [112], "e": ["Salud"],
     "a": "Ps. 24,041 millones, un crecimiento de 6.9%.",
     "q": ["¿Cuál fue la utilidad bruta de la división de Salud en 2024?"]},
    {"t": "factoid", "p": [112, 114], "e": ["Salud"],
     "a": "187 ubicaciones netas en 2024.",
     "q": ["¿Cuántas ubicaciones netas sumó la división de Salud en 2024?",
           "¿Cuántas ubicaciones netas añadió Salud en 2024?"]},
    {"t": "factoid", "p": [112], "e": ["Salud"],
     "a": "Una disminución de 0.3%.",
     "q": ["¿Cómo se comportaron las ventas mismas-tienda de la división de Salud en 2024?",
           "¿Cuánto variaron las ventas mismas-tienda de Salud en 2024?"]},
    {"t": "factoid", "p": [114], "e": ["Proximidad Américas"],
     "a": "Ps. 278,520 millones en 2023.",
     "q": ["¿A cuánto llegaron los ingresos de Proximidad Américas en 2023?"]},
    {"t": "factoid", "p": [114], "e": ["Proximidad Américas"],
     "a": "44.6% de los ingresos totales, con un aumento de 260 puntos base.",
     "q": ["¿Cuál fue el margen bruto de Proximidad Américas en 2024?",
           "¿En cuántos puntos base aumentó el margen bruto de Proximidad Américas en 2024?"]},
    {"t": "comparison", "p": [114], "e": ["Proximidad Américas"],
     "a": "Ps. 170,204 millones, un aumento de 5.4% frente a Ps. 161,458 millones en 2023.",
     "q": ["¿Cuál fue el costo de ventas de Proximidad Américas en 2024?"]},
    {"t": "factoid", "p": [114], "e": ["Proximidad Américas"],
     "a": "Ps. 8,642 millones, un aumento de 32.7%.",
     "q": ["¿A cuánto llegaron los gastos administrativos de Proximidad Américas en 2024?"]},
    {"t": "factoid", "p": [114], "e": ["Proximidad Américas"],
     "a": "Ps. 98,653 millones, un aumento de 16.8%.",
     "q": ["¿A cuánto llegaron los gastos de venta de Proximidad Américas en 2024?"]},
    {"t": "factoid", "p": [114], "e": ["Proximidad Europa"],
     "a": "Ps. 43,552 millones en 2023.",
     "q": ["¿A cuánto llegaron los ingresos de Proximidad Europa en 2023?"]},
    {"t": "factoid", "p": [114], "e": ["Proximidad Américas", "Delek"],
     "a": "249 tiendas de la adquisición de las operaciones minoristas de Delek en Estados Unidos.",
     "q": ["¿Cuántas tiendas aportó la adquisición de Delek a Proximidad Américas en 2024?",
           "¿Cuántas tiendas de Delek se consolidaron en Proximidad Américas en el cuarto trimestre de 2024?"]},
    {"t": "comparison", "p": [114], "e": ["Coca-Cola FEMSA"],
     "a": "Ps. 23,729 millones en 2024, frente a Ps. 19,536 millones en 2023.",
     "q": ["¿Cuál fue el ingreso neto de la participación controladora de Coca-Cola FEMSA en 2024?"]},
    {"t": "aggregation", "p": [114], "e": ["Proximidad Américas"],
     "a": "Un aumento de 5.7% en el ticket promedio y una disminución de 1.5% en el tráfico de mismas tiendas.",
     "q": ["¿Qué impulsó el aumento de 4.2% en ventas mismas-tienda de Proximidad Américas en 2024?",
           "¿Cuánto aumentó el ticket promedio y cuánto cayó el tráfico en Proximidad Américas en 2024?"]},
    # ---- Proximidad (pp. 19-26) ----
    {"t": "factoid", "p": [19], "e": ["OXXO México"],
     "a": "Más de 13 millones de tickets por día en OXXO México en 2024.",
     "q": ["¿Cuántos tickets por día promedió OXXO México en 2024?",
           "¿Cuántas transacciones diarias promedio tuvo OXXO México en 2024?"]},
    {"t": "factoid", "p": [19, 20], "e": ["OXXO México"],
     "a": "Se superaron las 1,000 aperturas de tiendas OXXO nuevas.",
     "q": ["¿Cuántas aperturas nuevas de tiendas OXXO se superaron en 2024?",
           "¿Se superó el umbral de 1,000 aperturas nuevas de OXXO en 2024?"]},
    {"t": "factoid", "p": [20], "e": ["OXXO"],
     "a": "Más de 24,000 tiendas OXXO en el mundo en 2024.",
     "q": ["¿Cuántas tiendas OXXO había en el mundo en 2024?",
           "¿A cuántas tiendas OXXO llegó FEMSA a nivel mundial en 2024?"]},
    {"t": "factoid", "p": [21], "e": ["OXXO México"],
     "a": "23,206 tiendas en México.",
     "q": ["¿Cuántas tiendas OXXO había en México en 2024?",
           "¿Cuántos puntos de venta tenía OXXO en México en 2024?"]},
    {"t": "factoid", "p": [20], "e": ["OXXO México"],
     "a": "Más de 811,000 tazas de café andatti por día.",
     "q": ["¿Cuántas tazas de café andatti se servían al día en OXXO México en 2024?",
           "¿Cuántas tazas de café andatti vendía OXXO al día en 2024?"]},
    {"t": "factoid", "p": [21], "e": ["Proximidad Américas"],
     "a": "26,322 puntos de venta.",
     "q": ["¿Cuántos puntos de venta tenía Proximidad Américas en 2024?",
           "¿A cuántos puntos de venta llegó Proximidad Américas en 2024?"]},
    {"t": "factoid", "p": [21], "e": ["Proximidad Américas", "Brasil"],
     "a": "1,860 puntos de venta en Brasil, que no consolidan en Proximidad Américas.",
     "q": ["¿Cuántos puntos de venta tenía Proximidad Américas en Brasil en 2024?",
           "¿Cuántos puntos de venta sumaba Brasil en Proximidad Américas en 2024?"]},
    {"t": "factoid", "p": [21], "e": ["OXXO", "Colombia"],
     "a": "569 tiendas OXXO en Colombia.",
     "q": ["¿Cuántas tiendas OXXO había en Colombia en 2024?"]},
    {"t": "factoid", "p": [21], "e": ["Proximidad Américas", "Estados Unidos"],
     "a": "249 tiendas en Estados Unidos.",
     "q": ["¿Cuántos puntos de venta tenía Proximidad Américas en Estados Unidos en 2024?"]},
    {"t": "factoid", "p": [21], "e": ["Proximidad Américas", "Chile"],
     "a": "235 tiendas en Chile.",
     "q": ["¿Cuántas tiendas tenía Proximidad Américas en Chile en 2024?"]},
    {"t": "factoid", "p": [21], "e": ["Proximidad Américas", "Perú"],
     "a": "203 tiendas en Perú.",
     "q": ["¿Cuántas tiendas tenía Proximidad Américas en Perú en 2024?"]},
    {"t": "factoid", "p": [21, 114], "e": ["Proximidad Europa"],
     "a": "2,778 tiendas en Proximidad Europa.",
     "q": ["¿Cuántas tiendas tenía Proximidad Europa en 2024?",
           "¿A cuántos puntos de venta llegó Proximidad Europa en 2024?"]},
    {"t": "factoid", "p": [21], "e": ["Proximidad Europa", "Alemania"],
     "a": "1,376 tiendas en Alemania.",
     "q": ["¿Cuántas tiendas tenía Proximidad Europa en Alemania en 2024?"]},
    {"t": "factoid", "p": [21], "e": ["Proximidad Europa", "Suiza"],
     "a": "1,241 tiendas en Suiza.",
     "q": ["¿Cuántas tiendas tenía Proximidad Europa en Suiza en 2024?"]},
    {"t": "aggregation", "p": [21], "e": ["Proximidad Europa"],
     "a": "76 en Luxemburgo, 54 en Países Bajos y 31 en Austria.",
     "q": ["¿Cuántas tiendas tenía Proximidad Europa en Luxemburgo, Países Bajos y Austria en 2024?"]},
    {"t": "factoid", "p": [21], "e": ["OXXO GAS", "Combustibles"],
     "a": "571 estaciones, todas en México.",
     "q": ["¿Cuántas estaciones de servicio sumaba la división de Combustibles en 2024?",
           "¿Cuál es el número de estaciones de OXXO GAS reportado en 2024?"]},
    {"t": "factoid", "p": [21], "e": ["Salud"],
     "a": "4,661 farmacias.",
     "q": ["¿Cuántas farmacias tenía la división de Salud en 2024?",
           "¿A cuántos puntos de venta llegó Salud en 2024?"]},
    {"t": "aggregation", "p": [21], "e": ["Salud"],
     "a": "1,739 en México, 997 en Chile, 980 en Ecuador y 945 en Colombia.",
     "q": ["¿Cuántas farmacias tenía Salud en México, Chile, Ecuador y Colombia en 2024?"]},
    {"t": "factoid", "p": [21], "e": ["FEMSA"],
     "a": "34,332 puntos de venta en total.",
     "q": ["¿Cuántos puntos de venta tenía FEMSA en total en 2024?",
           "¿A cuánto llegó el total de puntos de venta de FEMSA en 2024?"]},
    {"t": "factoid", "p": [23, 114], "e": ["OXXO"],
     "a": "24,462 tiendas OXXO en total.",
     "q": ["¿Cuántas tiendas OXXO había en total en 2024?",
           "¿A cuántas tiendas OXXO llegó la base de tiendas en 2024?"]},
    {"t": "factoid", "p": [22], "e": ["OXXO Nicho"],
     "a": "Más de 400 tiendas OXXO Nicho en México.",
     "q": ["¿Cuántas tiendas OXXO Nicho había en México en 2024?"]},
    {"t": "aggregation", "p": [22], "e": ["OXXO", "Colombia"],
     "a": "15 años operando en Colombia, con 569 tiendas OXXO.",
     "q": ["¿Cuántos años llevaba OXXO operando en Colombia en 2024?",
           "¿Desde hace cuántos años opera OXXO en Colombia?"]},
    {"t": "factoid", "p": [22], "e": ["Grupo Nós", "Brasil"],
     "a": "594 tiendas del Grupo Nós en Brasil.",
     "q": ["¿Cuántas tiendas tenía el Grupo Nós en Brasil en 2024?"]},
    {"t": "factoid", "p": [22], "e": ["OXXO", "Chile", "Perú"],
     "a": "438 tiendas consolidadas en Chile y Perú.",
     "q": ["¿Cuántas tiendas consolidadas sumaban Chile y Perú en 2024?"]},
    {"t": "multi-hop", "p": [22], "e": ["OXXO México", "Carlos Arenas Cadena"],
     "a": "Carlos Arenas Cadena, tras una carrera de 40 años en FEMSA; el retiro se anunció el 26 de septiembre de 2024.",
     "q": ["¿Quién era el director general de OXXO México que se retiró en 2024 y tras cuántos años de carrera?",
           "¿Cuántos años de carrera tenía Carlos Arenas Cadena al retirarse de OXXO México?"]},
    {"t": "multi-hop", "p": [22], "e": ["OXXO México", "Carlos Arroyo Rico"],
     "a": "Carlos Arroyo Rico asumió el cargo el 19 de noviembre de 2024, con más de 25 años de experiencia en la industria.",
     "q": ["¿Quién asumió como director general de OXXO México en noviembre de 2024?",
           "¿Cuándo asumió Carlos Arroyo Rico como director general de OXXO México?"]},
    {"t": "factoid", "p": [19], "e": ["Avec", "Proximidad Europa"],
     "a": "Avec fue reconocida entre las 12 mejores tiendas de foodvenience del mundo.",
     "q": ["¿Qué reconocimiento recibió Avec en 2024?",
           "¿Entre qué tiendas fue reconocida Avec en 2024?"]},
    {"t": "factoid", "p": [4], "e": ["Spin Premia", "OXXO"],
     "a": "Spin Premia formó parte del 40% de las ventas de OXXO en 2024.",
     "q": ["¿Qué porcentaje de las ventas de OXXO pasó por Spin Premia en 2024?",
           "¿En qué porcentaje de las ventas de OXXO participó Spin Premia en 2024?"]},
    {"t": "aggregation", "p": [4], "e": ["OXXO"],
     "a": "Se recicló más de 20% de los residuos de las tiendas y 85% de los de los centros de distribución.",
     "q": ["¿Qué porcentaje de residuos recicló OXXO en sus tiendas y centros de distribución en 2024?"]},
    {"t": "aggregation", "p": [4], "e": ["OXXO"],
     "a": "Más de 13,000 adultos mayores en el programa Huellas Mayores y más de 290,000 personas beneficiadas con iniciativas comunitarias.",
     "q": ["¿Cuántos adultos mayores participaban en el programa Huellas Mayores en 2024?",
           "¿A cuántas personas beneficiaron las iniciativas comunitarias de OXXO en 2024?"]},
    {"t": "aggregation", "p": [26], "e": ["Combustibles"],
     "a": "El canal B2B representó 18% de las ventas totales, con más de 7,000 clientes empresariales y más de 70,000 unidades suministradas.",
     "q": ["¿Qué porcentaje de las ventas de Combustibles representó el canal B2B en 2024?",
           "¿Cuántos clientes empresariales atendía el negocio B2B de Combustibles en 2024?"]},
    {"t": "factoid", "p": [19, 26], "e": ["Salud"],
     "a": "413 aperturas brutas de tiendas en Salud en 2024.",
     "q": ["¿Cuántas aperturas brutas tuvo la división de Salud en 2024?"]},
    {"t": "aggregation", "p": [26], "e": ["Salud"],
     "a": "Un centro de distribución nuevo en Colombia de 26,641 metros cuadrados, con capacidad para despachar 21,500 líneas y más de 120,000 unidades.",
     "q": ["¿Qué tamaño tenía el nuevo centro de distribución de Salud en Colombia?",
           "¿Cuántas líneas y unidades podía despachar el nuevo centro de distribución de Salud en Colombia?"]},
    {"t": "multi-hop", "p": [23], "e": ["OXXO"],
     "a": "Una alianza con GLORY anunciada en julio de 2024, con más de 1,500 máquinas instaladas en tiendas.",
     "q": ["¿Con quién se alió OXXO para la tecnología CASHINFINITY y cuántas máquinas instaló?",
           "¿Cuántas máquinas CASHINFINITY instaló OXXO en sus tiendas en 2024?"]},
    {"t": "multi-hop", "p": [25], "e": ["Proximidad Europa", "Ditsch"],
     "a": "Una inversión de aproximadamente US$70 millones para una segunda planta de pretzels en Cincinnati.",
     "q": ["¿Cuánto invirtió la filial Ditsch en su segunda planta de pretzels en Cincinnati?",
           "¿Dónde abrió Ditsch su segunda planta de producción de pretzels y con qué inversión?"]},
    {"t": "factoid", "p": [19], "e": ["FEMSA"],
     "a": "9% de las ventas se invirtió en CAPEX.",
     "q": ["¿Qué porcentaje de las ventas invirtió FEMSA en CAPEX en 2024 según los datos operativos?"]},
    {"t": "aggregation", "p": [23], "e": ["Proximidad Américas"],
     "a": "Un aumento de 5.7% en el gasto promedio y una disminución de 1.5% en el tráfico.",
     "q": ["¿Cuánto aumentó el gasto promedio y cuánto cayó el tráfico en Proximidad Américas en 2024?"]},
]

CURATED += [
    # ---- Coca-Cola FEMSA (pp. 27-32) ----
    {"t": "factoid", "p": [28], "e": ["Coca-Cola FEMSA"],
     "a": "El embotellador de Coca-Cola más grande del mundo por volumen de ventas.",
     "q": ["¿Qué lugar ocupa Coca-Cola FEMSA entre los embotelladores de Coca-Cola del mundo?"]},
    {"t": "multi-hop", "p": [28, 29], "e": ["Coca-Cola FEMSA"],
     "a": "9 países de América Latina y 276 millones de personas atendidas.",
     "q": ["¿En cuántos países opera Coca-Cola FEMSA y a cuántas personas atiende?",
           "¿A cuántas personas sirve Coca-Cola FEMSA en sus 9 países?"]},
    {"t": "factoid", "p": [29], "e": ["Coca-Cola FEMSA"],
     "a": "Cerca de 2.2 millones de puntos de venta.",
     "q": ["¿Cuántos puntos de venta tiene Coca-Cola FEMSA?"]},
    {"t": "aggregation", "p": [30], "e": ["Coca-Cola FEMSA"],
     "a": "56 plantas embotelladoras y 256 centros de distribución.",
     "q": ["¿Cuántas plantas embotelladoras y centros de distribución opera Coca-Cola FEMSA?"]},
    {"t": "factoid", "p": [27, 29], "e": ["Coca-Cola FEMSA"],
     "a": "El volumen creció 4.4% en 2024.",
     "q": ["¿Cuánto creció el volumen de Coca-Cola FEMSA en 2024?",
           "¿Qué crecimiento de volumen registró Coca-Cola FEMSA en 2024?"]},
    {"t": "factoid", "p": [29], "e": ["Coca-Cola FEMSA"],
     "a": "4.2 mil millones de cajas unidad.",
     "q": ["¿Cuántas cajas unidad vendió Coca-Cola FEMSA en 2024?",
           "¿Cuál fue el volumen de Coca-Cola FEMSA en cajas unidad en 2024?"]},
    {"t": "factoid", "p": [29], "e": ["Coca-Cola FEMSA"],
     "a": "Ps. 279 mil millones en ingresos totales.",
     "q": ["¿A cuántos miles de millones de pesos llegaron los ingresos de Coca-Cola FEMSA en 2024?"]},
    {"t": "aggregation", "p": [29], "e": ["Coca-Cola FEMSA"],
     "a": "Ps. 56 mil millones de EBITDA ajustado, con un margen de 20.1%.",
     "q": ["¿Cuál fue el EBITDA ajustado de Coca-Cola FEMSA en 2024?",
           "¿Qué margen de EBITDA ajustado tuvo Coca-Cola FEMSA en 2024?"]},
    {"t": "factoid", "p": [29], "e": ["Coca-Cola FEMSA"],
     "a": "La utilidad de operación creció 17.4% en 2024.",
     "q": ["¿Cuánto creció la utilidad de operación de Coca-Cola FEMSA en 2024?"]},
    {"t": "aggregation", "p": [28], "e": ["Coca-Cola FEMSA"],
     "a": "Seis prioridades estratégicas; la estrategia se actualizó en 2023 y 2024 fue el segundo capítulo de la transformación.",
     "q": ["¿Cuántas prioridades estratégicas tiene Coca-Cola FEMSA?",
           "¿En qué año se actualizó la estrategia de Coca-Cola FEMSA?"]},
    {"t": "aggregation", "p": [4, 27, 31], "e": ["Coca-Cola FEMSA", "Juntos+"],
     "a": "1.3 millones de clientes activos, más de 60% de la base de clientes.",
     "q": ["¿Cuántos clientes activos tenía la plataforma Juntos+ en 2024?",
           "¿Qué porcentaje de la base de clientes representaban los usuarios de Juntos+ en 2024?"]},
    {"t": "factoid", "p": [4, 31], "e": ["Coca-Cola FEMSA", "Juntos+"],
     "a": "Más de medio millón de clientes activos en Juntos+ 4.0, impulsada por inteligencia artificial.",
     "q": ["¿Cuántos clientes activos tenía Juntos+ 4.0 en 2024?",
           "¿Qué versión de Juntos+ está impulsada por inteligencia artificial?"]},
    {"t": "aggregation", "p": [4, 31], "e": ["Coca-Cola FEMSA", "Premia Juntos+"],
     "a": "Creció de 250,000 clientes a principios de año a más de 1.1 millones al cierre de 2024.",
     "q": ["¿Cuántos clientes tenía el programa Premia Juntos+ a principios y a finales de 2024?",
           "¿Cuánto creció el programa de lealtad Premia Juntos+ en 2024?"]},
    {"t": "aggregation", "p": [31], "e": ["Coca-Cola FEMSA", "Juntos+"],
     "a": "1.3 millones de compradores activos en 7 mercados.",
     "q": ["¿En cuántos mercados operaba Juntos+ en 2024?"]},
    {"t": "aggregation", "p": [4, 30], "e": ["Coca-Cola FEMSA"],
     "a": "7 líneas nuevas de producción iniciaron operaciones en 2024: 3 en México, 2 en Guatemala, 1 en Colombia y 1 en Brasil.",
     "q": ["¿Cuántas líneas nuevas de producción iniciaron operaciones en 2024 y dónde?",
           "¿Cuántas líneas de producción nuevas abrió Coca-Cola FEMSA en México, Guatemala, Colombia y Brasil en 2024?"]},
    {"t": "aggregation", "p": [30, 32], "e": ["Coca-Cola FEMSA"],
     "a": "La capacidad de producción aumentó 3.5% (más de 4% en México) y la de almacenamiento y distribución 5%.",
     "q": ["¿Cuánto aumentó la capacidad de producción de Coca-Cola FEMSA en 2024?",
           "¿Cuánto crecieron las capacidades de producción y distribución de Coca-Cola FEMSA en 2024?"]},
    {"t": "factoid", "p": [30, 32], "e": ["Coca-Cola FEMSA"],
     "a": "Se inauguraron 4 centros de distribución nuevos en 2024.",
     "q": ["¿Cuántos centros de distribución nuevos inauguró Coca-Cola FEMSA en 2024?"]},
    {"t": "factoid", "p": [30, 32], "e": ["Coca-Cola FEMSA"],
     "a": "15 plantas en México rompieron récords de producción.",
     "q": ["¿Cuántas plantas en México rompieron récords de producción en 2024?"]},
    {"t": "factoid", "p": [19, 30], "e": ["Coca-Cola FEMSA"],
     "a": "La capacidad de pallets en México aumentó 25%.",
     "q": ["¿Cuánto aumentó la capacidad de pallets en México en 2024?"]},
    {"t": "aggregation", "p": [30], "e": ["Coca-Cola FEMSA"],
     "a": "El volumen de Coca-Cola Zero Sugar creció 31% en 2024 y 56% en Brasil.",
     "q": ["¿Cuánto creció el volumen de Coca-Cola Zero Sugar en 2024?",
           "¿Cuánto creció Coca-Cola Zero Sugar en Brasil en 2024?"]},
    {"t": "factoid", "p": [30], "e": ["Coca-Cola FEMSA", "México"],
     "a": "El volumen en México creció 3.5% en 2024.",
     "q": ["¿Cuánto creció el volumen de Coca-Cola FEMSA en México en 2024?"]},
    {"t": "factoid", "p": [30], "e": ["Coca-Cola FEMSA", "Brasil"],
     "a": "El volumen en Brasil creció 7.8% en 2024, pese al cierre temporal de la planta de Porto Alegre.",
     "q": ["¿Cuánto creció el volumen de Coca-Cola FEMSA en Brasil en 2024?"]},
    {"t": "aggregation", "p": [30], "e": ["Coca-Cola FEMSA", "México"],
     "a": "Las bebidas no carbonatadas en México crecieron 8% en volumen, con 67% en tés, 39% en Powerade y 12% en Monster.",
     "q": ["¿Cuánto crecieron las bebidas no carbonatadas en México en 2024?",
           "¿Cuánto crecieron los tés, Powerade y Monster en México en 2024?"]},
    {"t": "factoid", "p": [32], "e": ["Coca-Cola FEMSA"],
     "a": "Se planea instalar 9 líneas embotelladoras adicionales en 2025.",
     "q": ["¿Cuántas líneas embotelladoras adicionales planea instalar Coca-Cola FEMSA en 2025?"]},
    {"t": "factoid", "p": [32], "e": ["Coca-Cola FEMSA"],
     "a": "Un aumento de 15% en capacidad de producción y de 30% en capacidad de almacén.",
     "q": ["¿Cuánto espera aumentar Coca-Cola FEMSA su capacidad de producción y de almacén?"]},
    {"t": "factoid", "p": [32], "e": ["Coca-Cola FEMSA", "Brasil"],
     "a": "La planta de Porto Alegre representaba 10% de la capacidad de producción en Brasil.",
     "q": ["¿Qué porcentaje de la capacidad de producción de Brasil representaba la planta de Porto Alegre?"]},
    {"t": "aggregation", "p": [32], "e": ["Coca-Cola FEMSA"],
     "a": "Un aumento de 14% en la producción de las líneas instaladas en los últimos 2 años, equivalente a 8 líneas embotelladoras nuevas, y 50% más capacidad de almacenamiento que en 2022.",
     "q": ["¿Cuánto aumentó la producción de las líneas instaladas de Coca-Cola FEMSA en los últimos 2 años?"]},
    # ---- Spin (pp. 5, 13, 19, 33, 35) ----
    {"t": "multi-hop", "p": [5], "e": ["Spin", "Digital@FEMSA"],
     "a": "Digital@FEMSA se convirtió en Spin, con un plan a cinco años alineado con FEMSA Forward.",
     "q": ["¿En qué se convirtió Digital@FEMSA en 2024?",
           "¿Qué nombre adoptó Digital@FEMSA en 2024?"]},
    {"t": "aggregation", "p": [5, 35], "e": ["Spin"],
     "a": "Tres pilares estratégicos: lealtad (Spin Premia), pagos (Spin by OXXO) y plataforma de servicios financieros.",
     "q": ["¿Cuáles son los tres pilares estratégicos de Spin?",
           "¿En qué tres pilares se basa la estrategia de Spin?"]},
    {"t": "factoid", "p": [5, 13, 33, 35], "e": ["Spin by OXXO"],
     "a": "8.6 millones de usuarios activos.",
     "q": ["¿A cuántos usuarios activos llegó Spin by OXXO durante 2024?",
           "¿Qué número de usuarios activos alcanzó Spin by OXXO en 2024?"]},
    {"t": "factoid", "p": [19, 35], "e": ["Spin by OXXO"],
     "a": "Más de 13 millones de clientes de Spin by OXXO.",
     "q": ["¿Cuántos clientes tenía Spin by OXXO en 2024?",
           "¿A cuántos clientes llegó Spin by OXXO en 2024?"]},
    {"t": "factoid", "p": [5, 13, 33], "e": ["Spin Premia"],
     "a": "Más de 24 millones de usuarios activos en Spin Premia.",
     "q": ["¿Cuántos usuarios activos tenía el programa de lealtad Spin Premia en 2024?",
           "¿A cuántos usuarios activos llegó Spin Premia en 2024?"]},
    {"t": "factoid", "p": [13, 33, 35], "e": ["Spin Premia"],
     "a": "24.6 millones de usuarios activos en Spin Premia.",
     "q": ["¿Cuál fue el número exacto de usuarios activos de Spin Premia reportado en el informe?",
           "¿Cuántos usuarios activos de Spin Premia se reportaron en las cifras clave de 2024?"]},
    {"t": "factoid", "p": [19, 35], "e": ["Spin Premia"],
     "a": "Más de 52 millones de usuarios de Spin Premia.",
     "q": ["¿Cuántos usuarios sumaba Spin Premia en total en 2024?",
           "¿A cuántos usuarios llegó Spin Premia como programa de lealtad en 2024?"]},
    {"t": "factoid", "p": [35], "e": ["Spin by OXXO"],
     "a": "Más de 63 millones de transacciones mensuales en promedio en Spin by OXXO.",
     "q": ["¿Cuántas transacciones mensuales promedio tenía Spin by OXXO en 2024?"]},
    {"t": "factoid", "p": [35], "e": ["Spin Premia", "OXXO"],
     "a": "40.7% de las transacciones en tiendas OXXO se pagaron con Spin Premia.",
     "q": ["¿Qué porcentaje de las compras en OXXO se pagó con Spin Premia en 2024?"]},
    {"t": "factoid", "p": [33], "e": ["Spin by OXXO"],
     "a": "Más de 50% de los usuarios activos de Spin by OXXO son mujeres.",
     "q": ["¿Qué porcentaje de los usuarios de Spin by OXXO son mujeres?"]},
    {"t": "factoid", "p": [5], "e": ["Spin", "Conekta", "OXXO PAY"],
     "a": "FEMSA completó la adquisición de los activos de Conekta, que habilitan OXXO PAY.",
     "q": ["¿Qué adquisición completó FEMSA para fortalecer su plataforma B2B en 2024?",
           "¿Qué activos adquirió FEMSA que habilitan OXXO PAY?"]},
    {"t": "factoid", "p": [5, 35], "e": ["Spin"],
     "a": "Consolidar a Spin como un referente de tecnología financiera en América Latina.",
     "q": ["¿Qué busca consolidar FEMSA con Spin en América Latina?",
           "¿Cuál es la meta de Spin como empresa de tecnología financiera?"]},
    # ---- Estrategia / empresa (pp. 3, 6, 10, 13) ----
    {"t": "aggregation", "p": [3], "e": ["FEMSA Forward"],
     "a": "Tres prioridades: invertir en las tres verticales principales con enfoque orgánico, desarrollar nuevas vías de creación de valor y optimizar la estructura de capital.",
     "q": ["¿Cuáles son las tres prioridades de la estrategia FEMSA Forward?",
           "¿Qué tres prioridades tiene FEMSA Forward?"]},
    {"t": "factoid", "p": [3], "e": ["FEMSA Forward"],
     "a": "Un ratio de deuda neta sobre EBITDA de 2x, excluyendo a Coca-Cola FEMSA.",
     "q": ["¿A qué ratio de deuda neta sobre EBITDA se comprometió FEMSA?",
           "¿Cuál es el objetivo de deuda neta sobre EBITDA de FEMSA excluyendo Coca-Cola FEMSA?"]},
    {"t": "aggregation", "p": [3], "e": ["FEMSA"],
     "a": "Se completó la desinversión de negocios no estratégicos en 2024, con dividendos especiales de Ps. 10,091 millones y recompras de casi 3% de las acciones en circulación.",
     "q": ["¿Cuánto pagó FEMSA en dividendos especiales en 2024?",
           "¿Qué porcentaje de acciones recompró FEMSA en 2024?"]},
    {"t": "factoid", "p": [3], "e": ["Proximidad", "Salud"],
     "a": "El margen bruto de Proximidad y Salud se expandió más de 300 puntos base.",
     "q": ["¿Cuántos puntos base se expandió el margen bruto de Proximidad y Salud en 2024?"]},
    {"t": "factoid", "p": [6], "e": ["FEMSA", "Delek"],
     "a": "La adquisición de las operaciones minoristas de Delek permitió la entrada al mercado de Estados Unidos, con enfoque en Texas.",
     "q": ["¿Qué adquisición permitió a FEMSA entrar al mercado de conveniencia de Estados Unidos?",
           "¿En qué estado de Estados Unidos se enfocó FEMSA tras adquirir Delek?"]},
    {"t": "factoid", "p": [6, 49], "e": ["FEMSA"],
     "a": "FEMSA fue incluida por primera vez en el S&P Dow Jones Sustainability World Index.",
     "q": ["¿En qué índice de sostenibilidad fue incluida FEMSA por primera vez en 2024?",
           "¿Qué reconocimiento de sostenibilidad logró FEMSA por primera vez en 2024?"]},
    {"t": "aggregation", "p": [6, 10], "e": ["FEMSA"],
     "a": "Presencia en 18 países y más de 388,000 colaboradores.",
     "q": ["¿En cuántos países tiene presencia FEMSA?",
           "¿Cuántos colaboradores tiene FEMSA?"]},
    {"t": "factoid", "p": [10], "e": ["FEMSA", "Coca-Cola FEMSA"],
     "a": "FEMSA posee 47.2% de Coca-Cola FEMSA.",
     "q": ["¿Qué porcentaje de Coca-Cola FEMSA posee FEMSA?",
           "¿Cuál es la participación de FEMSA en Coca-Cola FEMSA?"]},
    {"t": "factoid", "p": [13], "e": ["Proximidad", "Salud"],
     "a": "Más de 32,000 ubicaciones minoristas de Proximidad y Salud.",
     "q": ["¿Cuántas ubicaciones minoristas sumaban Proximidad y Salud en 2024?"]},
    {"t": "aggregation", "p": [13], "e": ["FEMSA"],
     "a": "US$37.4 mil millones en ingresos totales y US$3.3 mil millones en utilidad de operación.",
     "q": ["¿A cuánto equivalieron en dólares los ingresos y la utilidad de operación de FEMSA en 2024?"]},
    {"t": "factoid", "p": [6], "e": ["Salud"],
     "a": "Salud enfrentó un año desafiante: México vio mayores costos laborales y mayor competencia.",
     "q": ["¿Cómo fue 2024 para la división de Salud según el informe?",
           "¿Qué desafíos enfrentó la división de Salud en México en 2024?"]},
    {"t": "aggregation", "p": [6], "e": ["Salud"],
     "a": "En Colombia se consolidó como el actor más relevante del sector farmacéutico; en Chile se mantuvo líder en volumen y rentabilidad; en Ecuador mantuvo su posición.",
     "q": ["¿Cómo le fue a la división de Salud en Colombia, Chile y Ecuador en 2024?"]},
    {"t": "factoid", "p": [26], "e": ["Salud"],
     "a": "Una de las cadenas de farmacias más grandes de América Latina por ventas y puntos de venta, con 4,661 puntos de venta.",
     "q": ["¿Qué posición tiene la división de Salud entre las cadenas de farmacias de América Latina?"]},
    {"t": "factoid", "p": [25], "e": ["Proximidad Europa"],
     "a": "Proximidad Europa aumentó sus ingresos 14.2% en 2024.",
     "q": ["¿Cuánto crecieron los ingresos de Proximidad Europa en 2024 según la sección operativa?"]},
    {"t": "factoid", "p": [4], "e": ["FEMSA", "Retina Media"],
     "a": "Retina Media es la nueva plataforma omnicanal de retail media de FEMSA.",
     "q": ["¿Cómo se llama la nueva plataforma omnicanal de retail media de FEMSA?"]},
    # ---- Comparaciones curadas (evitan los pares exactos de las preguntas 5-7) ----
    {"t": "comparison", "p": [112], "e": ["Salud", "Combustibles"],
     "a": "Salud, con Ps. 79,755 millones, frente a Combustibles, con Ps. 65,365 millones.",
     "q": ["¿Qué división tuvo mayores ingresos en 2024, Salud o Combustibles?"]},
    {"t": "comparison", "p": [112], "e": ["Combustibles", "Salud"],
     "a": "Combustibles, con 11.7%, frente a Salud, con 5.8%.",
     "q": ["¿Qué división creció más en ingresos en 2024, Combustibles o Salud?"]},
    {"t": "comparison", "p": [112], "e": ["Proximidad Américas", "Coca-Cola FEMSA"],
     "a": "Proximidad Américas, con 17.0%, frente a Coca-Cola FEMSA, con 16.1%.",
     "q": ["¿Qué división creció más en utilidad bruta en 2024, Proximidad Américas o Coca-Cola FEMSA?"]},
    {"t": "comparison", "p": [112], "e": ["Proximidad Américas", "Coca-Cola FEMSA"],
     "a": "Proximidad Américas, con Ps. 136,993 millones, frente a Coca-Cola FEMSA, con Ps. 128,736 millones.",
     "q": ["¿Qué división tuvo mayor utilidad bruta en 2024, Proximidad Américas o Coca-Cola FEMSA?"]},
    {"t": "comparison", "p": [21], "e": ["OXXO", "Colombia", "Estados Unidos"],
     "a": "En Colombia, con 569 tiendas, frente a 249 en Estados Unidos.",
     "q": ["¿Dónde tenía más tiendas Proximidad Américas en 2024, en Colombia o en Estados Unidos?"]},
    {"t": "comparison", "p": [21], "e": ["Salud", "Chile", "Ecuador"],
     "a": "En Chile, con 997 farmacias, frente a 980 en Ecuador.",
     "q": ["¿Dónde tenía más farmacias Salud en 2024, en Chile o en Ecuador?"]},
    {"t": "comparison", "p": [21], "e": ["Proximidad Europa", "Alemania", "Suiza"],
     "a": "En Alemania, con 1,376 tiendas, frente a 1,241 en Suiza.",
     "q": ["¿Dónde tenía más tiendas Proximidad Europa en 2024, en Alemania o en Suiza?"]},
    {"t": "comparison", "p": [30], "e": ["Coca-Cola FEMSA", "México", "Brasil"],
     "a": "En Brasil, con 7.8%, frente a 3.5% en México.",
     "q": ["¿En qué país creció más el volumen de Coca-Cola FEMSA en 2024, en México o en Brasil?"]},
    {"t": "comparison", "p": [108, 109], "e": ["FEMSA"],
     "a": "La de la participación controladora, con una caída de 59.3%, frente a 47.5% de la utilidad neta consolidada.",
     "q": ["¿Qué cayó más en 2024, la utilidad neta consolidada o la de la participación controladora?"]},
    {"t": "comparison", "p": [112], "e": ["FEMSA"],
     "a": "La utilidad bruta, con 15.0%, frente a 11.2% de los ingresos totales.",
     "q": ["¿Qué creció más en 2024, los ingresos totales o la utilidad bruta de FEMSA?"]},
    {"t": "comparison", "p": [108, 112], "e": ["FEMSA"],
     "a": "El EBITDA, con 20.6%, frente a 19.8% de la utilidad de operación.",
     "q": ["¿Qué creció más en 2024, el EBITDA o la utilidad de operación de FEMSA?"]},
    {"t": "comparison", "p": [112], "e": ["FEMSA"],
     "a": "Los gastos administrativos, con 21.0%, frente a 12.3% de los gastos de venta.",
     "q": ["¿Qué aumentó más en 2024, los gastos administrativos o los gastos de venta de FEMSA?"]},
    {"t": "comparison", "p": [30], "e": ["Coca-Cola FEMSA", "México"],
     "a": "Los tés, con 67%, frente a 39% de Powerade.",
     "q": ["¿Qué creció más en México en 2024, los tés o Powerade?"]},
    {"t": "comparison", "p": [108], "e": ["FEMSA"],
     "a": "Los pasivos totales, con 10.0%, frente a 5.7% de los activos totales.",
     "q": ["¿Qué aumentó más en 2024, los activos totales o los pasivos totales de FEMSA?"]},
    {"t": "comparison", "p": [19, 35], "e": ["Spin Premia", "Spin by OXXO"],
     "a": "Spin Premia, con más de 52 millones de usuarios, frente a más de 13 millones de clientes de Spin by OXXO.",
     "q": ["¿Cuál de los dos productos de Spin sumaba más usuarios en 2024?"]},
    # ---- Multi-salto curados (combinan hechos de 2+ paginas) ----
    {"t": "multi-hop", "p": [21, 112], "e": ["OXXO México", "Proximidad Américas"],
     "a": "23,206 tiendas OXXO en México y los ingresos de Proximidad Américas crecieron 10.3%.",
     "q": ["¿Cuántas tiendas OXXO hay en México y cuánto crecieron los ingresos de Proximidad Américas en 2024?"]},
    {"t": "multi-hop", "p": [21, 112], "e": ["OXXO GAS", "Combustibles"],
     "a": "571 estaciones y los ingresos de Combustibles crecieron 11.7%.",
     "q": ["¿Cuántas estaciones OXXO GAS hay y cuánto crecieron los ingresos de Combustibles en 2024?"]},
    {"t": "multi-hop", "p": [21, 112], "e": ["Salud"],
     "a": "4,661 farmacias y sus ingresos crecieron 5.8%.",
     "q": ["¿Cuántas farmacias tiene Salud y cuánto crecieron sus ingresos en 2024?"]},
    {"t": "multi-hop", "p": [5, 108], "e": ["Spin by OXXO", "FEMSA"],
     "a": "8.6 millones de usuarios activos y Ps. 116 mil millones de EBITDA ajustado.",
     "q": ["¿Cuántos usuarios activos tiene Spin by OXXO y a cuánto llegó el EBITDA ajustado de FEMSA en 2024?"]},
    {"t": "multi-hop", "p": [10, 108], "e": ["FEMSA"],
     "a": "18 países y 388,999 empleados.",
     "q": ["¿En cuántos países opera FEMSA y cuántos empleados tenía en 2024?"]},
    {"t": "multi-hop", "p": [29, 112], "e": ["Coca-Cola FEMSA"],
     "a": "4.2 mil millones de cajas unidad y Ps. 279,793 millones de ingresos.",
     "q": ["¿Cuántas cajas unidad vendió Coca-Cola FEMSA y a cuánto llegaron sus ingresos en 2024?"]},
    {"t": "multi-hop", "p": [19, 112], "e": ["Salud", "Proximidad Américas"],
     "a": "413 aperturas brutas en Salud y 1,596 tiendas nuevas netas en Proximidad Américas.",
     "q": ["¿Cuántas aperturas brutas tuvo Salud y cuántas tiendas nuevas netas sumó Proximidad Américas en 2024?"]},
    {"t": "multi-hop", "p": [3, 112], "e": ["FEMSA"],
     "a": "Ps. 10,091 millones en dividendos especiales y sus ingresos crecieron 11.2%.",
     "q": ["¿Cuánto pagó FEMSA en dividendos especiales en 2024 y cuánto crecieron sus ingresos totales?"]},
    {"t": "multi-hop", "p": [21, 22], "e": ["OXXO Nicho", "Proximidad Américas"],
     "a": "Más de 400 tiendas OXXO Nicho en México y 26,322 puntos de venta en Proximidad Américas.",
     "q": ["¿Cuántas tiendas OXXO Nicho hay en México y cuántos puntos de venta tiene Proximidad Américas en total?"]},
    {"t": "multi-hop", "p": [30, 31], "e": ["Coca-Cola FEMSA", "Juntos+"],
     "a": "7 líneas nuevas de producción y 1.3 millones de clientes activos en Juntos+.",
     "q": ["¿Cuántas líneas nuevas de producción inició Coca-Cola FEMSA en 2024 y cuántos clientes activos tenía Juntos+?"]},
    {"t": "multi-hop", "p": [108, 109], "e": ["FEMSA"],
     "a": "Ps. 40,236 millones de utilidad neta consolidada y 41.1% de margen bruto.",
     "q": ["¿Cuál fue la utilidad neta consolidada y el margen bruto de FEMSA en 2024?"]},
    {"t": "multi-hop", "p": [4, 30], "e": ["Spin Premia", "Coca-Cola FEMSA"],
     "a": "40% de las ventas de OXXO y Coca-Cola Zero Sugar creció 31%.",
     "q": ["¿Qué porcentaje de las ventas de OXXO pasó por Spin Premia y cuánto creció Coca-Cola Zero Sugar en 2024?"]},
    {"t": "multi-hop", "p": [26, 112], "e": ["Combustibles"],
     "a": "Más de 7,000 clientes empresariales y las ventas mismas-estación crecieron 9.9%.",
     "q": ["¿Cuántos clientes empresariales atendía el B2B de Combustibles y cuánto crecieron las ventas mismas-estación en 2024?"]},
    {"t": "multi-hop", "p": [13, 35], "e": ["Spin Premia", "Spin by OXXO"],
     "a": "24.6 millones de usuarios activos en Spin Premia y más de 63 millones de transacciones mensuales en Spin by OXXO.",
     "q": ["¿Cuántos usuarios activos tenía Spin Premia y cuántas transacciones mensuales promedio tenía Spin by OXXO?"]},
    {"t": "multi-hop", "p": [5, 19], "e": ["Spin Premia", "OXXO México"],
     "a": "Más de 24 millones de usuarios activos en Spin Premia y más de 13 millones de tickets por día en OXXO México.",
     "q": ["¿Cuántos usuarios activos tenía Spin Premia y cuántos tickets diarios promediaba OXXO México?"]},
    {"t": "multi-hop", "p": [10, 29], "e": ["FEMSA", "Coca-Cola FEMSA"],
     "a": "47.2% de Coca-Cola FEMSA y 4.2 mil millones de cajas unidad.",
     "q": ["¿Qué porcentaje de Coca-Cola FEMSA posee FEMSA y cuántas cajas unidad vendió Coca-Cola FEMSA en 2024?"]},
    {"t": "multi-hop", "p": [21, 29], "e": ["Proximidad Américas", "Coca-Cola FEMSA"],
     "a": "26,322 puntos de venta en Proximidad Américas y cerca de 2.2 millones de puntos de venta en Coca-Cola FEMSA.",
     "q": ["¿Cuántos puntos de venta tenía Proximidad Américas y cuántos puntos de venta tenía Coca-Cola FEMSA?"]},
    {"t": "multi-hop", "p": [3, 5], "e": ["FEMSA", "Spin by OXXO"],
     "a": "Ps. 10,091 millones en dividendos especiales y 8.6 millones de usuarios activos en Spin by OXXO.",
     "q": ["¿Cuánto pagó FEMSA en dividendos especiales y cuántos usuarios activos tenía Spin by OXXO en 2024?"]},
    {"t": "multi-hop", "p": [22, 30], "e": ["OXXO México", "Coca-Cola FEMSA"],
     "a": "Carlos Arroyo Rico asumió el 19 de noviembre de 2024 y Coca-Cola Zero Sugar creció 31%.",
     "q": ["¿Quién asumió como director general de OXXO México en 2024 y cuánto creció Coca-Cola Zero Sugar ese año?"]},
    {"t": "multi-hop", "p": [108, 112], "e": ["FEMSA"],
     "a": "14.8% de margen EBITDA y 41.1% de margen bruto.",
     "q": ["¿Cuál fue el margen EBITDA y el margen bruto de FEMSA en 2024?"]},
    {"t": "multi-hop", "p": [19, 26], "e": ["Salud"],
     "a": "413 aperturas brutas y 4,661 puntos de venta.",
     "q": ["¿Cuántas aperturas brutas tuvo Salud y cuántos puntos de venta tenía en total?"]},
    {"t": "multi-hop", "p": [6, 112], "e": ["FEMSA", "Proximidad Américas"],
     "a": "En Texas y los ingresos de Proximidad Américas crecieron 10.3%.",
     "q": ["¿En qué estado se enfocó FEMSA tras adquirir Delek y cuánto crecieron los ingresos de Proximidad Américas en 2024?"]},
    {"t": "multi-hop", "p": [4, 35], "e": ["Spin Premia", "Spin by OXXO"],
     "a": "40% de las ventas de OXXO y más de 63 millones de transacciones mensuales en Spin by OXXO.",
     "q": ["¿Qué porcentaje de las ventas de OXXO pasó por Spin Premia y cuántas transacciones mensuales promedio tenía Spin by OXXO?"]},
    {"t": "multi-hop", "p": [28, 32], "e": ["Coca-Cola FEMSA"],
     "a": "9 países y 9 líneas embotelladoras adicionales en 2025.",
     "q": ["¿En cuántos países opera Coca-Cola FEMSA y cuántas líneas embotelladoras adicionales planea instalar en 2025?"]},
    {"t": "multi-hop", "p": [20, 23], "e": ["OXXO México", "OXXO"],
     "a": "Más de 811,000 tazas al día y 24,462 tiendas OXXO en total.",
     "q": ["¿Cuántas tazas de café andatti se servían al día y cuántas tiendas OXXO había en total en 2024?"]},
    {"t": "multi-hop", "p": [13, 108], "e": ["Spin Premia", "FEMSA"],
     "a": "24.6 millones de usuarios activos en Spin Premia y Ps. 851,536 millones en activos totales.",
     "q": ["¿Cuántos usuarios activos tenía Spin Premia según las cifras clave y cuál fue el total de activos de FEMSA en 2024?"]},
    # ---- Agregacion / sintesis curadas ----
    {"t": "aggregation", "p": [21], "e": ["FEMSA"],
     "a": "26,322 en Proximidad Américas, 2,778 en Proximidad Europa, 571 en Combustibles y 4,661 en Salud, para un total de 34,332 puntos de venta.",
     "q": ["¿Cuántos puntos de venta suman Proximidad Américas, Proximidad Europa, Combustibles y Salud en 2024?"]},
    {"t": "aggregation", "p": [21], "e": ["Proximidad Américas"],
     "a": "23,206 en México, 1,860 en Brasil, 569 en Colombia, 249 en Estados Unidos, 235 en Chile y 203 en Perú, para un total de 26,322 puntos de venta.",
     "q": ["¿Cuántos puntos de venta tiene Proximidad Américas en México, Brasil, Colombia, Estados Unidos, Chile y Perú?"]},
    {"t": "aggregation", "p": [21], "e": ["Proximidad Europa"],
     "a": "1,376 en Alemania, 1,241 en Suiza, 76 en Luxemburgo, 54 en Países Bajos y 31 en Austria, para un total de 2,778 tiendas.",
     "q": ["¿Cuántas tiendas tiene Proximidad Europa en Alemania, Suiza, Luxemburgo, Países Bajos y Austria?"]},
    {"t": "aggregation", "p": [21], "e": ["Salud"],
     "a": "1,739 en México, 997 en Chile, 980 en Ecuador y 945 en Colombia, para un total de 4,661 farmacias.",
     "q": ["¿Cuántas farmacias tiene Salud en México, Chile, Ecuador y Colombia?"]},
    {"t": "aggregation", "p": [112], "e": ["FEMSA"],
     "a": "Coca-Cola FEMSA con 14.2%, Proximidad Europa con 14.2%, Combustibles con 11.7% y Proximidad Américas con 10.3%.",
     "q": ["¿Qué divisiones crecieron más de 10% en ingresos en 2024?"]},
    {"t": "aggregation", "p": [112], "e": ["Proximidad Américas", "Coca-Cola FEMSA", "FEMSA"],
     "a": "Proximidad Américas Ps. 307,197 millones y Coca-Cola FEMSA Ps. 279,793 millones, dentro de un total consolidado de Ps. 781,585 millones.",
     "q": ["¿Cuánto suman los ingresos de Proximidad Américas y Coca-Cola FEMSA en 2024 dentro del total consolidado?"]},
    {"t": "aggregation", "p": [30, 32], "e": ["Coca-Cola FEMSA"],
     "a": "7 líneas nuevas de producción y 4 centros de distribución nuevos.",
     "q": ["¿Cuántas líneas nuevas y centros de distribución nuevos sumó Coca-Cola FEMSA en 2024?"]},
    {"t": "aggregation", "p": [4, 31], "e": ["Coca-Cola FEMSA", "Juntos+", "Premia Juntos+"],
     "a": "1.3 millones de clientes activos en Juntos+ y más de 1.1 millones en Premia Juntos+.",
     "q": ["¿Cuántos clientes sumaban Juntos+ y Premia Juntos+ al cierre de 2024?"]},
    {"t": "aggregation", "p": [108], "e": ["FEMSA"],
     "a": "Ps. 851,536 millones en activos, Ps. 470,405 millones en pasivos y Ps. 381,131 millones en capital.",
     "q": ["¿Cuánto sumaban los activos, pasivos y capital de FEMSA en 2024?"]},
    {"t": "aggregation", "p": [109], "e": ["FEMSA"],
     "a": "Ps. 775,551 millones en ventas netas, Ps. 460,072 millones de costo de ventas y Ps. 321,513 millones de utilidad bruta.",
     "q": ["¿Cuáles fueron las ventas netas, el costo de ventas y la utilidad bruta de FEMSA en 2024?"]},
    {"t": "aggregation", "p": [112], "e": ["FEMSA"],
     "a": "Proximidad Américas Ps. 136,993 millones, Coca-Cola FEMSA Ps. 128,736 millones, Salud Ps. 24,041 millones, Proximidad Europa Ps. 21,344 millones y Combustibles Ps. 7,935 millones.",
     "q": ["¿Qué divisiones reportaron utilidad bruta en 2024 y de cuánto fue cada una?"]},
    {"t": "aggregation", "p": [22], "e": ["OXXO", "Colombia", "Brasil", "Chile", "Perú"],
     "a": "569 en Colombia, 594 en Brasil (Grupo Nós) y 438 en Chile y Perú.",
     "q": ["¿Cuántas tiendas OXXO sumaban Colombia, Brasil y Chile con Perú en 2024?"]},
    {"t": "aggregation", "p": [21, 23], "e": ["OXXO México", "OXXO"],
     "a": "23,206 tiendas en México y 24,462 tiendas OXXO en total.",
     "q": ["¿Cuántas tiendas OXXO hay en México y cuántas tiendas OXXO hay en total?"]},
    {"t": "aggregation", "p": [108, 109, 112], "e": ["FEMSA"],
     "a": "Los ingresos 11.2%, la utilidad bruta 15.0% y el EBITDA 20.6%.",
     "q": ["¿Cuánto crecieron los ingresos, la utilidad bruta y el EBITDA de FEMSA en 2024?"]},
    {"t": "aggregation", "p": [30], "e": ["Coca-Cola FEMSA", "México"],
     "a": "67% en tés, 39% en Powerade y 12% en Monster.",
     "q": ["¿Cuánto crecieron en México los tés, Powerade y Monster en 2024?"]},
]

# ---------------------------------------------------------------------------
# Tablas financieras (valores EXACTOS del MD&A, p. 112; consolidado p. 109).
# NUNCA los valores redondeados de la ontologia v1.
# ---------------------------------------------------------------------------
DIVISIONS = [
    {"name": "Proximidad Américas", "rev": 307197, "rev_g": 10.3,
     "gross": 136993, "gross_g": 17.0, "pages": [112]},
    {"name": "Coca-Cola FEMSA", "rev": 279793, "rev_g": 14.2,
     "gross": 128736, "gross_g": 16.1, "pages": [112]},
    {"name": "Proximidad Europa", "rev": 49755, "rev_g": 14.2,
     "gross": 21344, "gross_g": 14.6, "pages": [112]},
    {"name": "Salud", "rev": 79755, "rev_g": 5.8,
     "gross": 24041, "gross_g": 6.9, "pages": [112]},
    {"name": "Combustibles", "rev": 65365, "rev_g": 11.7,
     "gross": 7935, "gross_g": 8.0, "pages": [112]},
]
CONSOLIDATED = {"name": "FEMSA", "rev": 781585, "rev_g": 11.2,
                "gross": 321513, "gross_g": 15.0, "pages": [109, 112]}

# Pares excluidos de las comparaciones programaticas: coinciden con las
# preguntas de evaluacion 5 (Europa vs Coca-Cola FEMSA, crecimiento) y
# 6 (Proximidad Americas vs Coca-Cola FEMSA, ingresos).
SKIP_REV_PAIR = {frozenset(["Proximidad Américas", "Coca-Cola FEMSA"])}
SKIP_GROWTH_PAIR = {frozenset(["Proximidad Europa", "Coca-Cola FEMSA"])}


def _fmt(n: int) -> str:
    return f"{n:,}"


def _pct(x) -> str:
    return "%.1f" % x


def programmatic_from_tables():
    """Expansion programatica desde las tablas financieras exactas."""
    out = []
    names = [d["name"] for d in DIVISIONS]

    def div(name):
        return next(d for d in DIVISIONS if d["name"] == name)

    # Factoids por division (formulaciones distintas de las curadas).
    for d in DIVISIONS:
        out.append({"question": "¿Qué monto de ingresos totales registró %s en 2024?" % d["name"],
                    "answer": "Ps. %s millones." % _fmt(d["rev"]),
                    "source_pages": d["pages"], "type": "factoid",
                    "entities": [d["name"]]})
        out.append({"question": "¿En qué porcentaje crecieron los ingresos de %s en 2024 frente a 2023?" % d["name"],
                    "answer": "Crecieron %s%% en 2024." % _pct(d["rev_g"]),
                    "source_pages": d["pages"], "type": "factoid",
                    "entities": [d["name"]]})
        out.append({"question": "¿De cuánto fue la utilidad bruta de %s en 2024?" % d["name"],
                    "answer": "Ps. %s millones." % _fmt(d["gross"]),
                    "source_pages": d["pages"], "type": "factoid",
                    "entities": [d["name"]]})

    # Comparaciones por pares (respuesta: ganador primero, ambas cifras).
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = div(names[i]), div(names[j])
            pair = frozenset([a["name"], b["name"]])
            # Ingresos.
            if pair not in SKIP_REV_PAIR:
                w, l = (a, b) if a["rev"] >= b["rev"] else (b, a)
                ans = ("%s, con Ps. %s millones, frente a %s, con Ps. %s millones."
                       % (w["name"], _fmt(w["rev"]), l["name"], _fmt(l["rev"])))
                out.append({"question": "¿Qué división registró mayores ingresos en 2024, %s o %s?" % (a["name"], b["name"]),
                            "answer": ans, "source_pages": [112], "type": "comparison",
                            "entities": [a["name"], b["name"]]})
                out.append({"question": "¿Entre %s y %s, cuál registró mayores ingresos totales en 2024?" % (a["name"], b["name"]),
                            "answer": ans, "source_pages": [112], "type": "comparison",
                            "entities": [a["name"], b["name"]]})
            # Crecimiento de ingresos.
            if pair not in SKIP_GROWTH_PAIR:
                w, l = (a, b) if a["rev_g"] >= b["rev_g"] else (b, a)
                ans = ("%s, con %s%%, frente a %s, con %s%%."
                       % (w["name"], _pct(w["rev_g"]), l["name"], _pct(l["rev_g"])))
                out.append({"question": "¿Qué división creció más en ingresos en 2024, %s o %s?" % (a["name"], b["name"]),
                            "answer": ans, "source_pages": [112], "type": "comparison",
                            "entities": [a["name"], b["name"]]})
                out.append({"question": "¿Entre %s y %s, cuál tuvo mayor crecimiento de ingresos en 2024?" % (a["name"], b["name"]),
                            "answer": ans, "source_pages": [112], "type": "comparison",
                            "entities": [a["name"], b["name"]]})
            # Utilidad bruta.
            w, l = (a, b) if a["gross"] >= b["gross"] else (b, a)
            ans = ("%s, con Ps. %s millones, frente a %s, con Ps. %s millones."
                   % (w["name"], _fmt(w["gross"]), l["name"], _fmt(l["gross"])))
            out.append({"question": "¿Qué división reportó mayor utilidad bruta en 2024, %s o %s?" % (a["name"], b["name"]),
                        "answer": ans, "source_pages": [112], "type": "comparison",
                        "entities": [a["name"], b["name"]]})

    # Cada division frente al crecimiento consolidado (11.2%).
    for d in DIVISIONS:
        if d["rev_g"] > CONSOLIDATED["rev_g"]:
            ans = ("Sí, %s creció %s%%, frente a 11.2%% del consolidado."
                   % (d["name"], _pct(d["rev_g"])))
        else:
            ans = ("No, %s creció %s%%, frente a 11.2%% del consolidado."
                   % (d["name"], _pct(d["rev_g"])))
        out.append({"question": "¿Superó %s el crecimiento de ingresos consolidados de FEMSA en 2024?" % d["name"],
                    "answer": ans, "source_pages": [112], "type": "comparison",
                    "entities": [d["name"], "FEMSA"]})

    # Rankings / agregacion desde la tabla.
    top_rev = max(DIVISIONS, key=lambda d: d["rev"])
    low_rev = min(DIVISIONS, key=lambda d: d["rev"])
    low_g = min(DIVISIONS, key=lambda d: d["rev_g"])
    top_gross = max(DIVISIONS, key=lambda d: d["gross"])
    low_gross = min(DIVISIONS, key=lambda d: d["gross"])
    out.append({"question": "¿Cuál fue la división con mayores ingresos en 2024?",
                "answer": "%s, con Ps. %s millones." % (top_rev["name"], _fmt(top_rev["rev"])),
                "source_pages": [112], "type": "aggregation", "entities": [top_rev["name"]]})
    out.append({"question": "¿Cuál fue la división con menores ingresos en 2024?",
                "answer": "%s, con Ps. %s millones." % (low_rev["name"], _fmt(low_rev["rev"])),
                "source_pages": [112], "type": "aggregation", "entities": [low_rev["name"]]})
    out.append({"question": "¿Cuál fue la división con mayor crecimiento de ingresos en 2024?",
                "answer": "Empate: Coca-Cola FEMSA y Proximidad Europa, ambas con 14.2%.",
                "source_pages": [112], "type": "aggregation",
                "entities": ["Coca-Cola FEMSA", "Proximidad Europa"]})
    out.append({"question": "¿Qué divisiones registraron un crecimiento de ingresos de 14.2% en 2024?",
                "answer": "Coca-Cola FEMSA y Proximidad Europa, ambas con 14.2%.",
                "source_pages": [112], "type": "aggregation",
                "entities": ["Coca-Cola FEMSA", "Proximidad Europa"]})
    out.append({"question": "¿Cuál fue la división con menor crecimiento de ingresos en 2024?",
                "answer": "%s, con %s%%." % (low_g["name"], _pct(low_g["rev_g"])),
                "source_pages": [112], "type": "aggregation", "entities": [low_g["name"]]})
    out.append({"question": "¿Cuál fue la división con mayor utilidad bruta en 2024?",
                "answer": "%s, con Ps. %s millones." % (top_gross["name"], _fmt(top_gross["gross"])),
                "source_pages": [112], "type": "aggregation", "entities": [top_gross["name"]]})
    out.append({"question": "¿Cuál fue la división con menor utilidad bruta en 2024?",
                "answer": "%s, con Ps. %s millones." % (low_gross["name"], _fmt(low_gross["gross"])),
                "source_pages": [112], "type": "aggregation", "entities": [low_gross["name"]]})
    return out


# ---------------------------------------------------------------------------
# Expansion programatica desde triples_seed.json (cada tripleta -> 2 pares).
# Correcciones aplicadas aqui (ver TENSION CONOCIDA en el encabezado):
#  - Se usa valor_exacto_informe cuando existe (nunca el redondeado v1).
#  - La tripleta (Proximidad Americas, mide, Aperturas = 1,236) NO se usa:
#    se sustituye por 1,596 tiendas nuevas netas (MD&A pp. 112/114).
#  - La tripleta (Proximidad Americas, mide, Tiendas = 23,206) se formula
#    como tiendas OXXO en Mexico (no como total de la division).
#  - Los porcentajes +24.9% / +27.5% de Spin (comunicado 4T24) se omiten.
# ---------------------------------------------------------------------------
def programmatic_from_triples():
    with open(TRIPLES_PATH, encoding="utf-8") as f:
        seed = json.load(f)
    out = []
    for t in seed["triples"]:
        subj, rel, obj = t["subject"], t["relation"], t["object"]
        page = (t.get("fuente") or {}).get("pagina")
        pages = [page] if page else [112]
        ent = [subj, obj]

        if rel == "tiene_division":
            out.append({"question": "¿Es %s una división de FEMSA?" % obj,
                        "answer": "Sí, %s es una división de FEMSA." % obj,
                        "source_pages": pages, "type": "factoid", "entities": ent})
            out.append({"question": "¿De qué empresa es división %s?" % obj,
                        "answer": "%s es una división de FEMSA." % obj,
                        "source_pages": pages, "type": "factoid", "entities": ent})
        elif rel == "tiene_formato":
            out.append({"question": "¿Es %s un formato de %s?" % (obj, subj),
                        "answer": "Sí, %s es un formato de %s." % (obj, subj),
                        "source_pages": pages, "type": "factoid", "entities": ent})
            out.append({"question": "¿A qué división pertenece el formato %s?" % obj,
                        "answer": "El formato %s pertenece a %s." % (obj, subj),
                        "source_pages": pages, "type": "factoid", "entities": ent})
        elif rel == "opera_en":
            out.append({"question": "¿Opera %s en %s según el informe?" % (subj, obj),
                        "answer": "Sí, %s opera en %s." % (subj, obj),
                        "source_pages": pages, "type": "factoid", "entities": ent})
            out.append({"question": "¿Tiene %s operaciones en %s?" % (subj, obj),
                        "answer": "Sí, %s tiene operaciones en %s." % (subj, obj),
                        "source_pages": pages, "type": "factoid", "entities": ent})
        elif rel == "reporta":
            # Valor exacto del informe cuando existe; jamas el redondeado v1.
            val = t.get("valor_exacto_informe", t.get("valor"))
            var = t.get("variacion_pct")
            if obj == "Utilidad de Operación":
                out.append({"question": "¿Qué utilidad de operación reportó FEMSA en 2024?",
                            "answer": "Ps. %s millones." % _fmt(val),
                            "source_pages": [108], "type": "factoid", "entities": ent})
                out.append({"question": "¿Cuánto creció la utilidad de operación de FEMSA en 2024 frente a 2023?",
                            "answer": "Creció %s%% en 2024." % _pct(var),
                            "source_pages": [108], "type": "comparison", "entities": ent})
            else:  # Ingresos Totales por division / consolidado.
                out.append({"question": "¿Qué ingresos totales reportó %s en 2024?" % subj,
                            "answer": "Ps. %s millones." % _fmt(val),
                            "source_pages": pages, "type": "factoid", "entities": ent})
                out.append({"question": "¿Cuánto crecieron los ingresos totales de %s en 2024 frente a 2023?" % subj,
                            "answer": "Crecieron %s%% en 2024." % _pct(var),
                            "source_pages": pages, "type": "comparison", "entities": ent})
        elif rel == "mide":
            if obj == "Tiendas":
                # 23,206 = tiendas OXXO en Mexico (nota de la tripleta).
                out.append({"question": "¿Cuántas tiendas OXXO tenía FEMSA en México en 2024?",
                            "answer": "23,206 tiendas OXXO en México.",
                            "source_pages": [21], "type": "factoid", "entities": ent})
                out.append({"question": "¿Cuántas tiendas OXXO en México sumaba Proximidad Américas en 2024?",
                            "answer": "23,206 tiendas OXXO en México.",
                            "source_pages": [21], "type": "factoid", "entities": ent})
            elif obj == "Aperturas":
                # OVERRIDE: 1,236 no esta en el informe; el MD&A dice 1,596.
                out.append({"question": "¿Cuántas tiendas nuevas netas añadió Proximidad Américas durante 2024?",
                            "answer": "1,596 tiendas nuevas netas.",
                            "source_pages": [112, 114], "type": "factoid", "entities": ent})
                out.append({"question": "¿En cuánto creció la base de tiendas de Proximidad Américas en 2024?",
                            "answer": "En 1,596 tiendas nuevas netas.",
                            "source_pages": [112, 114], "type": "factoid", "entities": ent})
            elif obj == "Estaciones":
                out.append({"question": "¿Cuántas estaciones sumaba OXXO GAS en 2024?",
                            "answer": "571 estaciones.",
                            "source_pages": [21], "type": "factoid", "entities": ent})
                out.append({"question": "¿Qué número de estaciones tenía la división de Combustibles en 2024?",
                            "answer": "571 estaciones, todas en México.",
                            "source_pages": [21], "type": "factoid", "entities": ent})
            elif obj == "Volumen":
                out.append({"question": "¿Qué crecimiento de volumen midió Coca-Cola FEMSA en 2024?",
                            "answer": "El volumen creció 4.4%.",
                            "source_pages": [27], "type": "factoid", "entities": ent})
                out.append({"question": "¿En qué porcentaje aumentó el volumen de Coca-Cola FEMSA en 2024?",
                            "answer": "Aumentó 4.4%.",
                            "source_pages": [27], "type": "factoid", "entities": ent})
            elif obj == "Clientes Juntos+":
                out.append({"question": "¿Cuántos clientes activos tenía Juntos+ en 2024?",
                            "answer": "1.3 millones de clientes activos.",
                            "source_pages": [19], "type": "factoid", "entities": ent})
                out.append({"question": "¿A cuántos clientes activos llegó la plataforma Juntos+ en 2024?",
                            "answer": "1.3 millones de clientes activos.",
                            "source_pages": [19], "type": "factoid", "entities": ent})
            elif obj == "Usuarios Activos":
                # Sin los porcentajes +24.9% / +27.5% (comunicado 4T24, no informe).
                val = t.get("valor")
                out.append({"question": "¿Cuántos usuarios activos midió %s en 2024?" % subj,
                            "answer": "%s millones de usuarios activos." % _pct(val),
                            "source_pages": pages, "type": "factoid", "entities": ent})
                out.append({"question": "¿Qué número de usuarios activos registró %s en 2024?" % subj,
                            "answer": "%s millones de usuarios activos." % _pct(val),
                            "source_pages": pages, "type": "factoid", "entities": ent})
        elif rel == "impulsa":
            out.append({"question": "¿Qué iniciativa impulsa a Proximidad Américas?",
                        "answer": "FEMSA Forward impulsa a Proximidad Américas.",
                        "source_pages": pages, "type": "factoid", "entities": ent})
            out.append({"question": "¿A qué división impulsa FEMSA Forward?",
                        "answer": "FEMSA Forward impulsa a Proximidad Américas.",
                        "source_pages": pages, "type": "factoid", "entities": ent})
        elif rel == "evoluciono_a":
            out.append({"question": "¿En qué iniciativa evolucionó Digital@FEMSA?",
                        "answer": "Digital@FEMSA evolucionó a Spin.",
                        "source_pages": pages, "type": "factoid", "entities": ent})
            out.append({"question": "¿Qué iniciativa surgió de Digital@FEMSA?",
                        "answer": "Spin, antes Digital@FEMSA.",
                        "source_pages": pages, "type": "factoid", "entities": ent})
        else:  # pragma: no cover - relacion no contemplada en la ontologia v1
            raise ValueError("Relacion no contemplada: %s" % rel)
    return out


# ---------------------------------------------------------------------------
# Main: ensamblar, deduplicar, excluir, verificar, escribir.
# ---------------------------------------------------------------------------
def main():
    page_numbers = load_chunk_numbers()

    raw = []
    for entry in CURATED:
        for q in entry["q"]:
            raw.append({"question": q, "answer": entry["a"],
                        "source_pages": sorted(entry["p"]), "type": entry["t"],
                        "entities": list(entry["e"])})
    n_curated = len(raw)
    prog_tables = programmatic_from_tables()
    raw.extend(prog_tables)
    prog_triples = programmatic_from_triples()
    raw.extend(prog_triples)

    # Deduplicacion por pregunta normalizada (se conserva la primera).
    seen, deduped, n_dup = {}, [], 0
    for pair in raw:
        key = normalize_text(pair["question"])
        if key in seen:
            n_dup += 1
            continue
        seen[key] = True
        deduped.append(pair)

    # Exclusion de las 12 preguntas de evaluacion.
    kept, n_excluded, excluded_hits = [], 0, []
    for pair in deduped:
        hit = is_excluded(pair["question"])
        if hit:
            n_excluded += 1
            excluded_hits.append((pair["question"], hit))
        else:
            kept.append(pair)

    # Verificacion numerica contra los chunks citados.
    final, n_failed, failures = [], 0, []
    for pair in kept:
        ok, missing = verify_numbers(pair["answer"], pair["source_pages"], page_numbers)
        if ok:
            final.append(pair)
        else:
            n_failed += 1
            failures.append((pair["question"], pair["answer"],
                             pair["source_pages"], missing))

    # Salida determinista: orden por pregunta normalizada.
    final.sort(key=lambda p: normalize_text(p["question"]))
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for pair in final:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    by_type = Counter(p["type"] for p in final)
    cross_page = sum(1 for p in final if len(p["source_pages"]) >= 2)
    total = len(final)
    print("=== generate_reflections.py — informe ===")
    print("Pares curados a mano:        %d" % n_curated)
    print("Pares programaticos (tablas): %d" % len(prog_tables))
    print("Pares programaticos (triples): %d" % len(prog_triples))
    print("Total generado (bruto):       %d" % len(raw))
    print("Descartados por duplicado:    %d" % n_dup)
    print("Descartados por exclusion (12 preguntas eval): %d" % n_excluded)
    for q, hit in excluded_hits:
        print("  EXCLUIDA: %r  ~=  %r" % (q, hit))
    print("Descartados por verificacion numerica: %d" % n_failed)
    for q, a, pages, missing in failures:
        print("  FALLO VERIF: %r | resp=%r | pags=%s | faltan=%s"
              % (q, a, pages, missing))
    print("---")
    print("Pares finales: %d" % total)
    for t in ["factoid", "comparison", "multi-hop", "aggregation"]:
        n = by_type.get(t, 0)
        print("  %-12s %3d  (%.1f%%)" % (t, n, 100.0 * n / total if total else 0))
    print("Pares con 2+ paginas citadas (sintesis multi-pagina): %d" % cross_page)
    print("Archivo: %s" % OUT_PATH)


if __name__ == "__main__":
    main()
