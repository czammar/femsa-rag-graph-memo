# Pipeline FEMSA 2024 — RAG clásico vs agéntico

Extracción, carga y consultas del knowledge graph del Informe Anual
Integrado 2024 de FEMSA, para el post de Substack "RAG clásico vs RAG
agéntico". Ontología v1 en `../docs/ontology-and-questions.md`.

## Archivos

| Archivo | Qué es |
|---|---|
| `extract.py` | Descarga el PDF, extrae texto por página (`data/chunks.jsonl`) y corre la extracción de tripletas (reglas + LLM opcional). |
| `triples_seed.json` | Las 31 tripletas semilla verificadas de la ontología v1 (el borrador decía 25; la lista real tiene 31). |
| `extracted_triples.jsonl` | Tripletas candidatas que encontró `extract.py` (13 por reglas). **Revisar a mano antes de cargar.** |
| `build_cypher.py` | Genera `load_graph.cypher` desde las semilla (+ extraídas revisadas). |
| `schema.cypher` | 9 restricciones de unicidad (`nombre` por etiqueta). |
| `load_graph.cypher` | MERGE idempotentes (se puede correr varias veces sin duplicar). |
| `load_relate.py` | Carga esquema + grafo en Relate por Bolt y verifica conteos. |
| `queries.cypher` | Las 12 preguntas del set, en Cypher, con respuesta esperada. |
| `data/FEMSA_2024_Annual_Report_EN.pdf` | El informe en inglés (18.8 MB, 151 pp.). **No incluido en el repo**: lo descarga `extract.py` (`make download`). Ver "nota sobre la edición" abajo. |
| `data/chunks.jsonl` | Texto extraído, una línea JSON por página. |
| `data/femsa.rl` | Store de Relate ya cargado (27 nodos, 39 relaciones). **No incluido en el repo**: se genera al correr el runbook. |
| `.venv/` (en la raíz del repo) | Entorno Python del proyecto (`pypdf`, `cryptography`, `neo4j`); se crea con `make setup`. |

## Runbook (de cero)

```bash
# Desde la raíz del repo (el entorno vive en .venv/, ver el README principal)
# 1. Descargar el PDF y extraer texto (151 páginas) + tripletas por reglas
.venv/bin/python pipeline/extract.py

# 2. Generar el Cypher de carga desde las tripletas semilla
.venv/bin/python pipeline/build_cypher.py

# 3. Arrancar Relate (instalar primero si falta:
#    curl -sSfL https://relatedb.ai/install.sh | bash)
relate serve --database ./pipeline/data/femsa.rl --bolt 7687 --auth none

# 4. Cargar esquema + grafo por Bolt (idempotente) y verificar conteos
.venv/bin/python pipeline/load_relate.py

# 5. Correr las 12 preguntas (p. ej. con el CLI, sin servidor)
relate -d ./pipeline/data/femsa.rl query < pipeline/queries.cypher
# o una por una, p. ej.:
echo "MATCH (e:Empresa {nombre: 'FEMSA'})-[r:REPORTA]->(i:IndicadorFinanciero {nombre: 'Ingresos Totales'}) RETURN r.valor, r.variacion_pct;" \
  | relate -d ./pipeline/data/femsa.rl query
```

Sin servidor, los pasos 3–4 se sustituyen por:

```bash
relate -d ./pipeline/data/femsa.rl query < pipeline/schema.cypher
relate -d ./pipeline/data/femsa.rl query < pipeline/load_graph.cypher
```

### Añadir tripletas extraídas (después de revisarlas)

```bash
# Revisa extracted_triples.jsonl a mano, corrige lo que haga falta y:
.venv/bin/python pipeline/build_cypher.py --include-extracted pipeline/extracted_triples.jsonl
.venv/bin/python pipeline/load_relate.py   # o el CLI de arriba
```

## Verificado 2026-11-25

- PDF en inglés: 151 páginas, todas con texto; la cifra 781,585 aparece
  en las pp. 11, 108, 109 y 112.
- Reglas deterministas: 13 tripletas candidatas, todas con página real.
- Grafo cargado en Relate v1.2.0 (CLI y por Bolt): **27 nodos, 39 relaciones**
  (31 semilla + 8 `CRECIO_EN` derivadas; el +14.2% de Proximidad Europa y
  Coca-Cola FEMSA colapsa en una sola relación `CRECIO_EN`, porque a nivel
  de indicador es el mismo hecho).
- Las 12 preguntas corren en Relate y devuelven la respuesta esperada,
  incluida la trampa de la Q5 (empate +14.2% / +14.2%) y la Q9 (~75.10%).
- Recargar `load_graph.cypher` no cambia nada (segunda carga: "the graph
  is unchanged").
- `load_relate.py` sin servidor: sale con código 2 y el mensaje de cómo
  arrancar Relate, sin traceback.

## Nota sobre la edición del informe (importante)

La ontología v1 se verificó contra el informe **en español**
(`informeanual.femsa.com/pdf/FEMSA_Informe-Anual-2024.pdf`), pero ese
sitio está detrás de un WAF (Imperva/Incapsula) que bloquea descargas
directas, y su única captura en Wayback Machine está corrupta (sin
trailer, 0 páginas; SHA-1 verificado contra el CDX). Por eso el pipeline
usa la **edición en inglés** (`annualreport.femsa.com/pdf/FEMSA_2024_Annual_Report.pdf`),
el mismo informe con las mismas cifras. Las páginas citadas en
`triples_seed.json` son páginas del PDF en inglés.

Si quieres las páginas de la edición en español: bájala a mano desde el
navegador a `pipeline/data/FEMSA_Informe-Anual-2024.pdf` y corre
`.venv/bin/python pipeline/extract.py --pdf-file pipeline/data/FEMSA_Informe-Anual-2024.pdf`.

## Discrepancias encontradas (no inventadas, verificadas en el PDF)

La ontología v1 redondea las cifras a centenas. El MD&A (p. 112) da los
valores exactos, guardados en `valor_exacto_informe` en `triples_seed.json`:

| División | Ontología v1 | Exacto (MD&A p. 112) |
|---|---|---|
| Proximidad Américas | Ps. 307,200 M | Ps. 307,197 M |
| Proximidad Europa | Ps. 49,800 M | Ps. 49,755 M |
| Salud | Ps. 79,700 M | Ps. 79,755 M |
| Combustibles | Ps. 65,400 M | Ps. 65,365 M |
| Coca-Cola FEMSA | Ps. 279,800 M | Ps. 279,793 M |

Además:

- **Tiendas 23,206** = tiendas OXXO en **México** (key data, p. 21). El
  total de Proximidad Américas es 26,322 puntos de venta y el total OXXO
  es 24,462 tiendas (p. 23).
- **Aperturas 1,236**: no localizada en el texto de la edición en inglés.
  El informe reporta **1,596 aperturas netas** en 2024 (p. 23/112/114).
  La tripleta conserva el valor de la ontología con `pagina: null` y una
  nota: verificar contra la edición en español antes de citarla.
- **Spin +24.9% / +27.5%**: vienen del comunicado de resultados 4T24
  (vs 4T23); no aparecen en el texto del informe anual.
- **Utilidad de operación**: la ontología solo registra +19.8%; el valor
  absoluto Ps. 70,667 M (p. 108) está en `valor_exacto_informe`.

## Límites honestos de Relate

- **En memoria**: el grafo vive en RAM; el store en disco (`femsa.rl`)
  es el respaldo. No es un almacén mayor que la memoria.
- **Un solo escritor**: un proceso sirve un grafo; no hay multiusuario.
- **Sin usuarios ni roles**: `--auth none` en local está bien; no lo
  expongas a la red (por defecto solo escucha en 127.0.0.1 y sin TLS).
- **Un grafo por store**.
- Las consultas de este proyecto son 100% openCypher estándar: el mismo
  `schema.cypher`, `load_graph.cypher` y `queries.cypher` corren en Neo4j
  5+ sin cambios.

## Reglas de extracción (ontología v1)

- Toda tripleta lleva `fuente` (documento + página). Sin fuente, no entra.
- Ningún valor inventado: si el informe no da el dato, la tripleta no
  existe (y la pregunta cae al RAG vectorial: ese es el caso que se evalúa).
- Valores monetarios en Ps. millones, con su periodo. Variaciones vs 2023
  salvo que se diga 4T24.
- La extracción por LLM es opcional (`extract.py --llm`, con `LLM_API_KEY`
  y `LLM_API_BASE_URL` en el entorno; nunca hay claves en el código) y sus
  tripletas también son candidatas: revisión humana antes de cargar.
