#!/usr/bin/env python3
"""Carga schema.cypher + load_graph.cypher en una instancia de Relate.

Uso:
    python load_relate.py                      # Bolt local por defecto
    python load_relate.py --schema schema.cypher --load load_graph.cypher

Conexión (variables de entorno):
    RELATE_URI       Bolt URI, por defecto bolt://localhost:7687
    RELATE_USER      usuario (por defecto: sin autenticación)
    RELATE_PASSWORD  contraseña

Relate no tiene usuarios ni roles: si arrancaste el servidor con
`relate serve --auth none` (lo recomendado en local), deja RELATE_USER
sin definir. Si lo arrancaste con `--auth usuario:clave` (o con el token
generado que imprime al arrancar), define RELATE_USER y RELATE_PASSWORD.

El script es idempotente: todo es MERGE, así que correrlo dos veces no
duplica nodos ni relaciones. Al final imprime conteos de verificación
(nodos por etiqueta, relaciones por tipo, y la respuesta a la pregunta 1
como prueba de que el grafo responde).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SCHEMA = HERE / "schema.cypher"
DEFAULT_LOAD = HERE / "load_graph.cypher"

DEFAULT_URI = "bolt://localhost:7687"

START_RELATE_HINT = """
No se pudo conectar a Relate. Para arrancar un servidor local:

    relate serve --database ./data/femsa.rl --bolt 7687 --auth none

(Primera vez: instala Relate con `curl -sSfL https://relatedb.ai/install.sh | bash`.)
También puedes cargar el grafo sin servidor, directamente con el CLI:

    relate -d ./data/femsa.rl query < schema.cypher
    relate -d ./data/femsa.rl query < load_graph.cypher

Si tu servidor usa autenticación, define RELATE_USER y RELATE_PASSWORD.
"""


def split_statements(text: str) -> list[str]:
    """Divide un archivo .cypher en sentencias por ';' (sin ';' en strings
    en estos archivos: los textos con ';' no aparecen en los literales)."""
    out = []
    buf: list[str] = []
    in_str = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'":
            # Comilla escapada \' no cierra el string.
            if i > 0 and text[i - 1] == "\\":
                buf.append(ch)
            else:
                in_str = not in_str
                buf.append(ch)
        elif ch == ";" and not in_str:
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def read_statements(path: Path) -> list[str]:
    """Lee un .cypher ignorando líneas de comentario // y vacías."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("//"):
            continue
        lines.append(line)
    return split_statements("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uri", default=os.environ.get("RELATE_URI", DEFAULT_URI))
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    ap.add_argument("--load", type=Path, default=DEFAULT_LOAD)
    args = ap.parse_args()

    try:
        from neo4j import GraphDatabase
        from neo4j.exceptions import AuthError, ServiceUnavailable
    except ImportError:
        print("[load] Falta el driver de Neo4j para Python. Instálalo con:",
              file=sys.stderr)
        print("[load]   pip install neo4j", file=sys.stderr)
        print("[load] (o usa el venv de este pipeline: .venv/bin/pip install neo4j)",
              file=sys.stderr)
        return 2

    user = os.environ.get("RELATE_USER")
    password = os.environ.get("RELATE_PASSWORD")
    auth = (user, password) if user else None

    def connect(auth_value):
        driver = GraphDatabase.driver(args.uri, auth=auth_value)
        driver.verify_connectivity()
        return driver

    try:
        driver = connect(auth)
    except AuthError:
        # El servidor pide autenticación y no se dio (o viceversa).
        fallback = ("neo4j", password or "neo4j") if not auth else None
        if fallback is None:
            print("[load] Autenticación rechazada con las credenciales dadas.",
                  file=sys.stderr)
            print(START_RELATE_HINT, file=sys.stderr)
            return 2
        print("[load] Sin autenticación no funcionó; reintentando con "
              "usuario 'neo4j'...")
        try:
            driver = connect(fallback)
        except Exception:
            print("[load] Autenticación rechazada.", file=sys.stderr)
            print(START_RELATE_HINT, file=sys.stderr)
            return 2
    except (ServiceUnavailable, Exception) as e:  # noqa: BLE001
        print(f"[load] No se pudo conectar a Relate en {args.uri}: {e}",
              file=sys.stderr)
        print(START_RELATE_HINT, file=sys.stderr)
        return 2

    print(f"[load] Conectado a {args.uri}")

    schema_stmts = read_statements(args.schema)
    load_stmts = read_statements(args.load)
    print(f"[load] {args.schema.name}: {len(schema_stmts)} sentencias; "
          f"{args.load.name}: {len(load_stmts)} sentencias")

    with driver.session() as session:
        # Esquema: si Relate no soporta CREATE CONSTRAINT, avisa y sigue.
        for stmt in schema_stmts:
            try:
                session.run(stmt)
            except Exception as e:  # noqa: BLE001
                print(f"[load] AVISO: no se pudo aplicar una sentencia de "
                      f"esquema ({e}). Sigo sin esa restricción; los MERGE "
                      f"siguen funcionando.")
                break
        print("[load] Esquema aplicado (o avisado).")
        for i, stmt in enumerate(load_stmts, start=1):
            session.run(stmt)
            if i % 25 == 0 or i == len(load_stmts):
                print(f"[load]   {i}/{len(load_stmts)} sentencias...")
        print("[load] Carga completa.")

        # Verificación.
        total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        by_label = session.run(
            "MATCH (n) RETURN labels(n) AS etiqueta, count(n) AS n "
            "ORDER BY n DESC").data()
        by_rel = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS tipo, count(r) AS n "
            "ORDER BY n DESC").data()
        q1 = session.run(
            "MATCH (e:Empresa {nombre: 'FEMSA'})-[r:REPORTA]->"
            "(i:IndicadorFinanciero {nombre: 'Ingresos Totales'}) "
            "RETURN r.valor AS valor, r.variacion_pct AS variacion").single()
        print(f"[load] Verificación: {total_nodes} nodos, {total_rels} relaciones.")
        print("[load] Nodos por etiqueta:")
        for row in by_label:
            print(f"[load]   {row['etiqueta']}: {row['n']}")
        print("[load] Relaciones por tipo:")
        for row in by_rel:
            print(f"[load]   {row['tipo']}: {row['n']}")
        print(f"[load] Prueba Q1: ingresos totales FEMSA 2024 = "
              f"Ps. {q1['valor']:,} M (+{q1['variacion']}%).")
    driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
