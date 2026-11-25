#!/usr/bin/env python3
"""Genera pipeline/load_graph.cypher a partir de las tripletas semilla.

Uso:
    python build_cypher.py
    python build_cypher.py --include-extracted extracted_triples.jsonl
        # añade tripletas extraídas (¡solo después de revisarlas a mano!)

Modelo de grafo (ontología v1):
- Un nodo por entidad, con etiqueta = tipo (Empresa, Division, Formato,
  IndicadorFinanciero, MetricaOperativa, Periodo, Pais, Iniciativa, Riesgo)
  y propiedad `nombre` (única por etiqueta, ver schema.cypher).
- Las relaciones usan los nombres de la ontología en UPPER_SNAKE_CASE
  (tiene_division -> TIENE_DIVISION, etc.).
- Los valores van en la RELACIÓN (REPORTA / MIDE), no en el nodo del
  indicador: un mismo IndicadorFinanciero ("Ingresos Totales") tiene un
  valor distinto por división. La relación lleva valor, unidad,
  variacion_pct, periodo y la fuente (documento + página).
- CRECIO_EN (IndicadorFinanciero -> Periodo) se DERIVA de las tripletas
  REPORTA/MIDE que traen variacion_pct: es el mismo hecho, en otra forma.
  No añade ningún valor nuevo.

Todo es MERGE: correr load_graph.cypher dos veces no duplica nada.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED_PATH = HERE / "triples_seed.json"
OUT_PATH = HERE / "load_graph.cypher"

REL_MAP = {
    "tiene_division": "TIENE_DIVISION",
    "tiene_formato": "TIENE_FORMATO",
    "opera_en": "OPERA_EN",
    "reporta": "REPORTA",
    "mide": "MIDE",
    "crecio_en": "CRECIO_EN",
    "impulsa": "IMPULSA",
    "evoluciono_a": "EVOLUCIONO_A",
}

# Relaciones que llevan valor en la relación (el resto solo lleva fuente).
VALUE_RELATIONS = {"reporta", "mide"}


def cypher_str(value: str) -> str:
    """Escapa un string para usarlo como literal Cypher entre comillas simples."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def cypher_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if value is None:
        return "null"
    return cypher_str(str(value))


def rel_props(t: dict) -> dict:
    """Propiedades de la relación para una tripleta."""
    props: dict = {}
    fuente = t.get("fuente", {}) or {}
    if fuente.get("documento"):
        props["fuente_documento"] = fuente["documento"]
    if fuente.get("pagina") is not None:
        props["fuente_pagina"] = fuente["pagina"]
    if t["relation"] in VALUE_RELATIONS:
        for key in ("valor", "unidad", "variacion_pct", "periodo",
                    "valor_exacto_informe"):
            if t.get(key) is not None:
                props[key] = t[key]
    if t.get("nota"):
        props["nota"] = t["nota"]
    return props


def props_cypher(props: dict) -> str:
    if not props:
        return ""
    items = ", ".join(f"{k}: {cypher_value(v)}" for k, v in props.items())
    return " {" + items + "}"


def node_merge(label: str, name: str, var: str) -> str:
    return f"MERGE ({var}:{label} {{nombre: {cypher_str(name)}}})"


def triple_to_cypher(t: dict, idx: int) -> list[str]:
    """Una tripleta -> 1-2 sentencias MERGE (nodos + relación).

    Las variables llevan sufijo único por tripleta (s1/o1, s2/o2...):
    Relate ejecuta un script completo con ámbito de variables compartido,
    así que reutilizar `s`/`o` en cada sentencia falla con
    VariableAlreadyBound. En Neo4j no haría falta, pero así el archivo
    corre igual en ambos.
    """
    rel = REL_MAP[t["relation"]]
    lines = [
        node_merge(t["subject_type"], t["subject"], f"s{idx}") + ";",
        node_merge(t["object_type"], t["object"], f"o{idx}") + ";",
        f"MERGE (s{idx})-[:{rel}{props_cypher(rel_props(t))}]->(o{idx});",
    ]
    return lines


def derived_crecio_en(t: dict, idx: int) -> list[str] | None:
    """Deriva CRECIO_EN desde REPORTA/MIDE con variacion_pct."""
    if t["relation"] not in VALUE_RELATIONS:
        return None
    if t.get("variacion_pct") is None or t.get("periodo") is None:
        return None
    fuente = t.get("fuente", {}) or {}
    props = {"variacion_pct": t["variacion_pct"]}
    if fuente.get("documento"):
        props["fuente_documento"] = fuente["documento"]
    if fuente.get("pagina") is not None:
        props["fuente_pagina"] = fuente["pagina"]
    return [
        node_merge(t["object_type"], t["object"], f"s{idx}") + ";",
        node_merge("Periodo", t["periodo"], f"o{idx}") + ";",
        f"MERGE (s{idx})-[:CRECIO_EN{props_cypher(props)}]->(o{idx});",
    ]


def load_triples(seed_path: Path, extra_paths: list[Path]) -> list[dict]:
    with open(seed_path, encoding="utf-8") as fh:
        data = json.load(fh)
    triples = list(data["triples"])
    for path in extra_paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    triples.append(json.loads(line))
    return triples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=Path, default=SEED_PATH)
    ap.add_argument("--include-extracted", type=Path, action="append",
                    default=[], metavar="JSONL",
                    help="Tripletas extraídas ya revisadas a mano (se añaden "
                         "a las semilla). Repetible.")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    triples = load_triples(args.seed, args.include_extracted)
    unknown = {t["relation"] for t in triples} - set(REL_MAP)
    if unknown:
        raise SystemExit(f"Relaciones desconocidas en la ontología v1: {unknown}")

    # Orden determinista para que el archivo generado sea estable.
    triples.sort(key=lambda t: (t["relation"], t["subject"], t["object"]))

    out: list[str] = []
    out.append("// load_graph.cypher — generado por build_cypher.py")
    out.append("// Fuente: triples_seed.json (ontología v1, cifras verificadas)")
    if args.include_extracted:
        out.append("// + tripletas extraídas revisadas: "
                   + ", ".join(str(p) for p in args.include_extracted))
    out.append("// Todo es MERGE: se puede correr varias veces sin duplicar.")
    out.append("// Relaciones en UPPER_SNAKE_CASE (ontología: minúsculas).")
    out.append("")

    derived: list[str] = []
    seen_crecio: set = set()
    for i, t in enumerate(triples, start=1):
        out.append(f"// ({t['subject']}, {t['relation']}, {t['object']})")
        out.extend(triple_to_cypher(t, i))
        d = derived_crecio_en(t, 1000 + i)
        if d:
            # Misma (indicador, variación, periodo, fuente) = misma relación:
            # p. ej. Proximidad Europa y Coca-Cola FEMSA, ambas +14.2% en
            # Ingresos Totales, colapsan en una sola CRECIO_EN.
            key = (t["object"], t.get("variacion_pct"), t.get("periodo"),
                   (t.get("fuente") or {}).get("pagina"))
            if key not in seen_crecio:
                seen_crecio.add(key)
                derived.extend(d)
        out.append("")

    if derived:
        out.append("// --- CRECIO_EN derivadas de REPORTA/MIDE (mismo hecho) ---")
        out.append("// Nota: si dos divisiones comparten indicador y variación")
        out.append("// (p. ej. Proximidad Europa y Coca-Cola FEMSA, ambas +14.2%")
        out.append("// en Ingresos Totales), MERGE las colapsa en UNA sola")
        out.append("// relación: es el mismo hecho a nivel de indicador.")
        out.append("")
        out.extend(derived)

    args.out.write_text("\n".join(out), encoding="utf-8")
    n_nodes = len({(t["subject_type"], t["subject"]) for t in triples}
                  | {(t["object_type"], t["object"]) for t in triples})
    n_derived = sum(1 for line in out if "[:CRECIO_EN" in line)
    print(f"[build] {len(triples)} tripletas -> {args.out}")
    print(f"[build] Nodos distintos: {n_nodes}; "
          f"relaciones CRECIO_EN derivadas (distintas tras MERGE): {n_derived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
