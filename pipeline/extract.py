#!/usr/bin/env python3
"""Extracción de texto y tripletas del Informe Anual Integrado 2024 de FEMSA.

Uso:
    python extract.py                 # descarga el PDF (si falta), extrae texto
                                      # por página y escribe data/chunks.jsonl
    python extract.py --llm           # además intenta la extracción guiada por
                                      # LLM (requiere LLM_API_KEY y
                                      # LLM_API_BASE_URL en el entorno)

Sin LLM configurado, el script sigue funcionando: extrae el texto y corre
las reglas deterministas para las cifras conocidas de la ontología v1.
Toda tripleta que sale de aquí lleva `fuente` (página del PDF).
Ningún valor se inventa: si el patrón no aparece en el texto, no hay tripleta.

Fuente: https://informeanual.femsa.com/pdf/FEMSA_Informe-Anual-2024.pdf
Ontología: ../docs/ontology-and-questions.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
EXTRACTED_PATH = HERE / "extracted_triples.jsonl"

PDF_URL_ES = "https://informeanual.femsa.com/pdf/FEMSA_Informe-Anual-2024.pdf"
PDF_URL_EN = "https://annualreport.femsa.com/pdf/FEMSA_2024_Annual_Report.pdf"
# Orden de intento: primero el informe en español (fuente canónica de la
# ontología), luego el espejo en inglés (mismo informe, mismas cifras).
PDF_URLS = [PDF_URL_ES, PDF_URL_EN]
PDF_PATH_ES = DATA_DIR / "FEMSA_Informe-Anual-2024.pdf"
PDF_PATH_EN = DATA_DIR / "FEMSA_2024_Annual_Report_EN.pdf"
PDF_PATH = PDF_PATH_ES  # ruta por defecto (edición en español, si está en caché)
SOURCE_LABEL = "Informe Anual Integrado 2024 de FEMSA, «Evolucionando»"
SOURCE_LABEL_EN = "FEMSA 2024 Integrated Annual Report, «Evolving» (English edition, same figures)"

# Etiqueta de la fuente realmente usada (la fija main() tras la descarga).
ACTIVE_SOURCE_LABEL = SOURCE_LABEL

# ---------------------------------------------------------------------------
# 1. Descarga y extracción de texto
# ---------------------------------------------------------------------------


def _is_valid_pdf(path: Path) -> bool:
    """Comprueba que el archivo sea un PDF legible con al menos 1 página."""
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages) > 0
    except Exception:
        return False


def download_pdf(url: str | None = None, dest: Path | None = None) -> tuple[Path, str]:
    """Descarga el PDF público si no está en caché.

    Intenta primero la edición en español (informeanual.femsa.com) y, si
    falla o el archivo no es un PDF válido, la edición en inglés
    (annualreport.femsa.com). Devuelve (ruta_local, etiqueta_de_fuente).

    Nota verificada 2026-11-25: el sitio en español está detrás de un
    WAF (Imperva/Incapsula) que bloquea descargas directas, y su única
    captura en Wayback Machine está corrupta (sin trailer, 0 páginas;
    SHA-1 verificado contra el CDX). La edición en inglés es el mismo
    informe con las mismas cifras.
    """
    candidates = []
    if url:
        candidates.append((url, dest or PDF_PATH_ES, SOURCE_LABEL))
    else:
        candidates = [
            (PDF_URL_ES, PDF_PATH_ES, SOURCE_LABEL),
            (PDF_URL_EN, PDF_PATH_EN, SOURCE_LABEL_EN),
        ]
    for pdf_url, pdf_dest, label in candidates:
        if pdf_dest.exists() and _is_valid_pdf(pdf_dest):
            print(f"[extract] PDF en caché: {pdf_dest} ({pdf_dest.stat().st_size/1e6:.1f} MB)")
            return pdf_dest, label
        if pdf_dest.exists():
            print(f"[extract] Caché inválido, se vuelve a descargar: {pdf_dest}")
            pdf_dest.unlink()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[extract] Descargando {pdf_url} ...")
        try:
            req = urllib.request.Request(
                pdf_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as resp, \
                    open(pdf_dest, "wb") as fh:
                fh.write(resp.read())
        except Exception as e:  # noqa: BLE001 - se intenta el siguiente espejo
            print(f"[extract] Falló la descarga ({e}); probando siguiente fuente...")
            if pdf_dest.exists():
                pdf_dest.unlink()
            continue
        if _is_valid_pdf(pdf_dest):
            print(f"[extract] Guardado en {pdf_dest} ({pdf_dest.stat().st_size/1e6:.1f} MB)")
            return pdf_dest, label
        print(f"[extract] El archivo descargado no es un PDF válido; probando siguiente fuente...")
        pdf_dest.unlink()
    raise RuntimeError(
        "No se pudo descargar un PDF válido del informe. "
        "Descarga manual: https://informeanual.femsa.com/pdf/FEMSA_Informe-Anual-2024.pdf "
        "o https://annualreport.femsa.com/pdf/FEMSA_2024_Annual_Report.pdf "
        "y colócalo en pipeline/data/."
    )


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extrae el texto de cada página con pypdf. Una chunk por página."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    chunks = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        chunks.append({"page": i, "text": text, "chars": len(text),
                       "fuente": ACTIVE_SOURCE_LABEL})
    return chunks


def write_chunks(chunks: list[dict], dest: Path = CHUNKS_PATH) -> None:
    with open(dest, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[extract] {len(chunks)} páginas -> {dest}")


# ---------------------------------------------------------------------------
# 2. Extracción de tripletas guiada por la ontología v1
# ---------------------------------------------------------------------------
#
# Reglas deterministas: patrones anclados a las cifras de la ontología v1.
# Son *verificación*, no invención: solo emiten una tripleta si el patrón
# aparece literalmente en el texto de la página. Cualquier cifra nueva debe
# pasar por el path LLM + revisión humana antes de cargarse al grafo.

# (regex, constructor de tripleta). El constructor recibe el match y la página.
RULES: list[tuple[str, callable]] = []


def _triple(subject, subject_type, relation, obj, object_type, page, **kw):
    t = {
        "subject": subject,
        "subject_type": subject_type,
        "relation": relation,
        "object": obj,
        "object_type": object_type,
        "fuente": {"documento": ACTIVE_SOURCE_LABEL, "pagina": page},
    }
    t.update(kw)
    return t


def _rule(pattern: str, needs: str | None = None):
    """Registra una regla determinista.

    `needs` es un regex opcional que debe aparecer en la misma página para
    que la regla dispare (evita falsos positivos: p. ej. '79,755' también
    es el número de personas beneficiadas, no solo el ingreso de Salud).
    """
    def deco(fn):
        RULES.append((pattern, fn, needs))
        return fn
    return deco


@_rule(r"781[,\s]?585")
def _ingresos_consolidados(m, page):
    return _triple("FEMSA", "Empresa", "reporta", "Ingresos Totales",
                   "IndicadorFinanciero", page,
                   valor=781585, unidad="Ps. millones",
                   variacion_pct=11.2, periodo="2024")


@_rule(r"307[,\s]?197", needs=r"Proximity\s+Americas")
def _ingresos_prox_americas(m, page):
    return _triple("Proximidad Américas", "Division", "reporta",
                   "Ingresos Totales", "IndicadorFinanciero", page,
                   valor=307197, unidad="Ps. millones",
                   variacion_pct=10.3, periodo="2024",
                   nota="La ontología v1 redondea a Ps. 307,200 M; valor exacto del MD&A.")


@_rule(r"23[,\s]?206")
def _tiendas_prox_americas(m, page):
    return _triple("Proximidad Américas", "Division", "mide", "Tiendas",
                   "MetricaOperativa", page,
                   valor=23206, unidad="tiendas", periodo="2024",
                   nota="23,206 = tiendas OXXO en México (key data); el total de "
                        "Proximidad Américas es 26,322 puntos de venta.")


@_rule(r"1[,\s]?596")
def _aperturas_netas(m, page):
    return _triple("Proximidad Américas", "Division", "mide", "Aperturas Netas",
                   "MetricaOperativa", page,
                   valor=1596, unidad="tiendas netas nuevas", periodo="2024",
                   nota="La ontología v1 registra 1,236 aperturas (cifra del "
                        "comunicado 4T24); el informe reporta 1,596 aperturas "
                        "netas. No son la misma métrica: revisar antes de cargar.")


@_rule(r"279[,\s]?793", needs=r"Coca-Cola\s+FEMSA")
def _ingresos_kof(m, page):
    return _triple("Coca-Cola FEMSA", "Division", "reporta",
                   "Ingresos Totales", "IndicadorFinanciero", page,
                   valor=279793, unidad="Ps. millones",
                   variacion_pct=14.2, periodo="2024",
                   nota="La ontología v1 redondea a Ps. 279,800 M; valor exacto del MD&A.")


@_rule(r"49[,\s]?755", needs=r"Proximity\s+Europe")
def _ingresos_prox_europa(m, page):
    return _triple("Proximidad Europa", "Division", "reporta",
                   "Ingresos Totales", "IndicadorFinanciero", page,
                   valor=49755, unidad="Ps. millones",
                   variacion_pct=14.2, periodo="2024",
                   nota="La ontología v1 redondea a Ps. 49,800 M; valor exacto del MD&A.")


@_rule(r"79[,\s]?755", needs=r"Health\s+Division")
def _ingresos_salud(m, page):
    return _triple("Salud", "Division", "reporta",
                   "Ingresos Totales", "IndicadorFinanciero", page,
                   valor=79755, unidad="Ps. millones",
                   variacion_pct=5.8, periodo="2024",
                   nota="La ontología v1 redondea a Ps. 79,700 M; valor exacto del MD&A.")


@_rule(r"65[,\s]?365", needs=r"Fuel")
def _ingresos_combustibles(m, page):
    return _triple("Combustibles", "Division", "reporta",
                   "Ingresos Totales", "IndicadorFinanciero", page,
                   valor=65365, unidad="Ps. millones",
                   variacion_pct=11.7, periodo="2024",
                   nota="La ontología v1 redondea a Ps. 65,400 M; valor exacto del MD&A.")


@_rule(r"8\.6\s*million")
def _usuarios_spin(m, page):
    return _triple("Spin by OXXO", "Formato", "mide", "Usuarios Activos",
                   "MetricaOperativa", page,
                   valor=8.6, unidad="millones de usuarios",
                   variacion_pct=24.9, periodo="2024",
                   nota="El +24.9% viene del comunicado 4T24 (vs 4T23); no "
                        "aparece en el texto del informe anual.")


@_rule(r"24\.6\s*million")
def _usuarios_spin_premia(m, page):
    return _triple("Spin Premia", "Formato", "mide", "Usuarios Activos",
                   "MetricaOperativa", page,
                   valor=24.6, unidad="millones de usuarios",
                   variacion_pct=27.5, periodo="2024",
                   nota="El +27.5% viene del comunicado 4T24 (vs 4T23); no "
                        "aparece en el texto del informe anual.")


@_rule(r"1\.3\s*million", needs=r"Juntos\+")
def _clientes_juntos(m, page):
    return _triple("Coca-Cola FEMSA", "Division", "mide", "Clientes Juntos+",
                   "MetricaOperativa", page,
                   valor=1.3, unidad="millones de clientes", periodo="2024")


@_rule(r"volume\s+(?:increased|grew)\s+4\.4%", needs=r"Coca-Cola\s+FEMSA")
def _volumen_kof(m, page):
    return _triple("Coca-Cola FEMSA", "Division", "mide", "Volumen",
                   "MetricaOperativa", page,
                   valor=4.4, unidad="% crecimiento vs 2023", periodo="2024")


@_rule(r"\b571\b")
def _estaciones_oxxo_gas(m, page):
    # 571 solo es significativo si la página habla de OXXO GAS / estaciones.
    # La regla se evalúa con el texto completo de la página (ver abajo).
    return None  # se maneja en extract_triples_rule_based con contexto


def extract_triples_rule_based(chunk: dict) -> list[dict]:
    """Aplica las reglas deterministas a una página. Devuelve tripletas candidatas."""
    text = chunk["text"]
    page = chunk["page"]
    out: list[dict] = []
    for pattern, fn, needs in RULES:
        if needs and not re.search(needs, text):
            continue
        for m in re.finditer(pattern, text):
            t = fn(m, page)
            if t is not None:
                out.append(t)
    # Regla con contexto: 571 estaciones solo si la página menciona OXXO GAS.
    if re.search(r"\b571\b", text) and re.search(r"OXXO\s*GAS", text, re.I):
        out.append(_triple("Combustibles", "Division", "mide", "Estaciones",
                           "MetricaOperativa", page,
                           valor=571, unidad="estaciones", periodo="2024"))
    # Deduplica por (subject, relation, object, valor).
    seen, dedup = set(), []
    for t in out:
        key = (t["subject"], t["relation"], t["object"], t.get("valor"))
        if key not in seen:
            seen.add(key)
            dedup.append(t)
    return dedup


# ---------------------------------------------------------------------------
# 3. Path LLM (opcional, requiere credenciales por variables de entorno)
# ---------------------------------------------------------------------------

ONTOLOGY_PROMPT = """\
Eres un extractor de conocimiento. Del texto siguiente, extrae SOLO tripletas
que usen exactamente estos tipos de entidad y estas relaciones:

Tipos: Empresa, Division, Formato, IndicadorFinanciero, MetricaOperativa,
Periodo, Pais, Iniciativa, Riesgo.
Relaciones: tiene_division (Empresa->Division), tiene_formato
(Division->Formato), opera_en (Division/Formato->Pais), reporta
(Division->IndicadorFinanciero, con valor, variacion_pct, periodo), mide
(Division->MetricaOperativa, con valor, periodo), crecio_en
(IndicadorFinanciero->Periodo), impulsa (Iniciativa->Division),
evoluciono_a (Iniciativa->Iniciativa).

Reglas estrictas:
- No inventes ningún valor: solo cifras que aparezcan literalmente en el texto.
- Los valores monetarios van en Ps. millones, con su periodo.
- Cada tripleta debe incluir "fuente": {"documento": "Informe Anual Integrado 2024 de FEMSA", "pagina": N}.
- Si el texto no contiene ninguna tripleta válida, responde con una lista vacía.
- Responde ÚNICAMENTE con JSON: una lista de objetos con las claves
  subject, subject_type, relation, object, object_type y, cuando apliquen,
  valor, unidad, variacion_pct, periodo, fuente.
"""


def extract_triples_llm(chunk: dict) -> list[dict]:
    """Extracción con LLM vía API compatible con OpenAI (opcional).

    Requiere en el entorno:
        LLM_API_KEY       clave de la API (nunca se escribe en el código)
        LLM_API_BASE_URL  p. ej. https://api.openai.com/v1
        LLM_MODEL         opcional, por defecto el modelo que indique el endpoint
    Si falta la configuración, lanza RuntimeError y el caller lo ignora.
    """
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_API_BASE_URL")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    if not api_key or not base_url:
        raise RuntimeError(
            "LLM no configurado: define LLM_API_KEY y LLM_API_BASE_URL "
            "para usar la extracción con LLM."
        )
    import urllib.request

    payload = json.dumps({
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": ONTOLOGY_PROMPT},
            {"role": "user", "content":
             f"Página {chunk['page']} del informe:\n\n{chunk['text'][:6000]}"},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.load(resp)
    content = body["choices"][0]["message"]["content"]
    triples = json.loads(content)
    # Forzar la fuente a la página real (el LLM no debe inventar páginas).
    for t in triples:
        t.setdefault("fuente", {})["pagina"] = chunk["page"]
        t["fuente"]["documento"] = ACTIVE_SOURCE_LABEL
    return triples


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf-url", default=None,
                    help="URL de un PDF concreto (por defecto: intenta el "
                         "informe en español y luego el espejo en inglés).")
    ap.add_argument("--pdf-file", default=None,
                    help="Ruta a un PDF ya descargado (p. ej. el informe en "
                         "español bajado a mano desde el navegador).")
    ap.add_argument("--llm", action="store_true",
                    help="Intentar extracción con LLM (requiere LLM_API_KEY "
                         "y LLM_API_BASE_URL). Sin ellas, solo reglas.")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    global ACTIVE_SOURCE_LABEL
    if args.pdf_file:
        pdf = Path(args.pdf_file)
        if not _is_valid_pdf(pdf):
            print(f"[extract] ERROR: {pdf} no es un PDF válido.", file=sys.stderr)
            return 1
        ACTIVE_SOURCE_LABEL = SOURCE_LABEL
    else:
        if args.force_download:
            for p in (PDF_PATH_ES, PDF_PATH_EN):
                if p.exists():
                    p.unlink()
        pdf, ACTIVE_SOURCE_LABEL = download_pdf(args.pdf_url)
    chunks = extract_pages(pdf)
    non_empty = sum(1 for c in chunks if c["chars"] > 0)
    print(f"[extract] Páginas: {len(chunks)} ({non_empty} con texto)")
    write_chunks(chunks)

    # Reglas deterministas sobre todas las páginas.
    rule_triples: list[dict] = []
    for c in chunks:
        rule_triples.extend(extract_triples_rule_based(c))
    # Deduplicado global: un hecho, una tripleta (se conserva la primera página).
    seen: set = set()
    deduped: list[dict] = []
    for t in rule_triples:
        key = (t["subject"], t["relation"], t["object"], t.get("valor"))
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    rule_triples = deduped
    print(f"[extract] Tripletas por reglas: {len(rule_triples)} "
          f"(páginas con hits: {len({t['fuente']['pagina'] for t in rule_triples})})")
    for t in rule_triples:
        print(f"  p.{t['fuente']['pagina']}: ({t['subject']}, {t['relation']}, "
              f"{t['object']} = {t.get('valor')} {t.get('unidad', '')})")

    llm_triples: list[dict] = []
    if args.llm:
        try:
            for c in chunks:
                if c["chars"] < 200:
                    continue
                llm_triples.extend(extract_triples_llm(c))
            print(f"[extract] Tripletas por LLM: {len(llm_triples)}")
        except RuntimeError as e:
            print(f"[extract] LLM omitido: {e}", file=sys.stderr)

    all_triples = rule_triples + llm_triples
    if all_triples:
        with open(EXTRACTED_PATH, "w", encoding="utf-8") as fh:
            for t in all_triples:
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"[extract] {len(all_triples)} tripletas candidatas -> {EXTRACTED_PATH}")
        print("[extract] NOTA: las tripletas extraídas (reglas o LLM) son "
              "candidatas: revísalas antes de cargarlas al grafo "
              "(ver pipeline/README.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
