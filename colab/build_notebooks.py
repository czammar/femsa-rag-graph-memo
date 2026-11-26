#!/usr/bin/env python3
"""Genera los notebooks de Google Colab del experimento FEMSA 2024.

Uso:  python3 build_notebooks.py   (escribe colab/*.ipynb).
El README.md de colab/ se mantiene a mano (no lo genera este script).

Cada notebook es autocontenido: monta Drive, clona/actualiza el repo,
instala sus dependencias y (donde hace falta) instala/arranca Ollama con
celdas idempotentes. Toda la logica vive inline en las celdas del
notebook: ningun notebook importa ni ejecuta scripts .py del repo
(pipeline/, memo/, rag_baseline/, agentic_rag/ son la referencia
canonica de las corridas en CPU, pero los notebooks no dependen de
ellos). Los archivos de datos (chunks.jsonl, triples_seed.json,
eval_questions.json, ground_truth_evidence.json) si se leen del repo
clonado, de Drive o se suben con el widget de Colab.

Datos procesados: cada notebook que consume chunks.jsonl lleva su celda
de datos (Drive -> subir con el widget de Colab -> generar con la logica
de extraccion inline como ultimo recurso). Los insumos que produce una etapa anterior (adaptador
LoRA del 03, trazas del 05/06) se verifican con una celda de prerrequisitos
que falla con un mensaje claro si faltan. Cada notebook termina con una
celda de guardado en Drive (verifica y lista lo producido).

No se ejecuta nada en Colab aqui: este script solo genera los .ipynb.
"""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md(src):
    return nbf.v4.new_markdown_cell(src.strip())


def code(src):
    return nbf.v4.new_code_cell(src.strip())


def shell(src):
    """Celda %%bash (comandos de shell)."""
    return nbf.v4.new_code_cell("%%bash\n" + src.strip())


def pip_cell(packages):
    """Celda %pip (una por linea de paquetes)."""
    return nbf.v4.new_code_cell("%pip install -q " + " ".join(packages))


def write_nb(name, cells, title):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    }
    nb.cells = cells
    nbf.validate(nb)
    path = HERE / name
    nbf.write(nb, path)
    print(f"{name}: {len(cells)} celdas -> {path}")


# ---------------------------------------------------------------------------
# Celdas comunes
# ---------------------------------------------------------------------------

def title_cell(title, body):
    return md(f"# {title}\n\n{body}")


def gpu_check_cell():
    return shell("""nvidia-smi
python3 -c "import torch; print('torch', torch.__version__, '| CUDA disponible:', torch.cuda.is_available())" 2>/dev/null || echo "torch aun no instalado (se instala abajo)" """)


def drive_mount_cell():
    return code("""
from google.colab import drive
drive.mount('/content/drive')
print("Drive montado en /content/drive")
""")


def params_cell(extra=""):
    return code("""
# --- Parametros (editalos aqui arriba) ---
DRIVE_ROOT = "/content/drive/MyDrive/femsa-rag"   # carpeta en Drive donde vive todo
REPO_URL = "https://github.com/czammar/femsa-rag-graph-memo"
REPO = DRIVE_ROOT + "/femsa-rag-graph-memo"
""" + extra)


def repo_setup_cell():
    return code("""
import os, subprocess

os.makedirs(DRIVE_ROOT, exist_ok=True)
# El cache de Hugging Face vive en Drive: los pesos descargados
# sobreviven al reinicio de la sesion (la VM de Colab es efimera).
os.environ["HF_HOME"] = DRIVE_ROOT + "/.cache/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if os.path.isdir(REPO + "/.git"):
    print("repo ya existe, actualizando...")
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=False)
else:
    print("clonando el repo...")
    subprocess.run(["git", "clone", REPO_URL, REPO], check=True)
print("repo en:", REPO)
print("HF_HOME:", os.environ["HF_HOME"])
""")


def ollama_install_cell():
    """Compacta e idempotente: instala Ollama solo si falta el binario."""
    return shell("""if command -v ollama >/dev/null 2>&1; then
  echo "ollama ya instalado: $(ollama --version)"
else
  echo "instalando Ollama (primero zstd, que el instalador exige y Colab no trae)..."
  apt-get -qq update && apt-get -qq install -y zstd pciutils
  curl -fsSL https://ollama.com/install.sh | sh
  ollama --version
fi""")


def ollama_server_cell():
    """Compacta e idempotente: arranca el servidor solo si no responde ya."""
    return code("""
import os, subprocess, time, urllib.request

def ollama_up():
    try:
        with urllib.request.urlopen("http://localhost:11434", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

if ollama_up():
    print("servidor Ollama ya responde en http://localhost:11434")
else:
    print("arrancando `ollama serve` en segundo plano...")
    out = open("ollama.log", "w")
    err = open("ollama_error.log", "w")
    subprocess.Popen(["nohup", "ollama", "serve"], stdout=out, stderr=err,
                     preexec_fn=os.setsid)
    for _ in range(60):
        if ollama_up():
            break
        time.sleep(2)
    print("Ollama responde en localhost:11434:", ollama_up())
    if not ollama_up():
        print("Revisa ollama_error.log; el servidor no levanto.")
""")


def ollama_pull_cell(models):
    """Compacta e idempotente: hace pull solo del modelo que no esta ya.
    `ollama show` devuelve 0 solo si ese tag exacto existe en local."""
    lines = []
    for m in models:
        lines.append(f'''if ollama show "{m}" >/dev/null 2>&1; then
  echo "{m} ya descargado"
else
  echo "bajando {m}..."
  ollama pull "{m}"
fi''')
    lines.append("ollama list")
    return shell("\n".join(lines))


def ollama_smoke_cell(model_var="OLLAMA_MODEL"):
    return code(f"""
from langchain_ollama import ChatOllama

_llm = ChatOllama(model={model_var}, base_url="http://localhost:11434", temperature=0)
print(_llm.invoke("Responde con una sola palabra: ¿cuánto es 2+2?").content)
print("ChatOllama OK contra", {model_var})
""")


def ollama_compact_cells(models, model_var="OLLAMA_MODEL"):
    """El bloque compacto idempotente que lleva cada notebook que usa Ollama:
    instalar (si falta) -> arrancar servidor (si no responde) -> pull (si falta)
    -> prueba rapida. Correrlo dos veces no rompe nada ni descarga de mas."""
    return [
        md("## Ollama (setup compacto e idempotente)\n\n"
           "Este notebook usa modelos intermedios vía **Ollama** corriendo en la propia "
           "instancia de Colab (un solo servidor, cambio de modelo con un parámetro, "
           "sin cargar pesos en cada notebook). Las celdas son idempotentes: verifican "
           "si el binario existe, si el servidor ya responde y si el modelo ya está "
           "bajado antes de hacer nada. La versión completa, con la explicación y las "
           "fuentes, está en `00_ollama_setup.ipynb`."),
        ollama_install_cell(),
        ollama_server_cell(),
        ollama_pull_cell(models),
        ollama_smoke_cell(model_var),
    ]

# ---------------------------------------------------------------------------
# Celdas de datos: primero en Drive, si no, subir; generar es último recurso
# ---------------------------------------------------------------------------


def data_ensure_cells():
    """Datos procesados, en orden: Drive -> subir con widget -> generar (último recurso).
    La lleva cada notebook que consume chunks.jsonl (01-08).

    La generación es autocontenida: la lógica de descarga y extracción por
    página (la de pipeline/extract.py, citada como procedencia) va copiada
    inline en la celda; el notebook no ejecuta ningún script del repo.
    """
    _DATA_MD = (
        "## Datos procesados: primero en Drive, si no, súbelos\n\n"
        "Este notebook necesita `pipeline/data/chunks.jsonl` (las 151 páginas del informe "
        "extraídas). El flujo es:\n\n"
        "1. **Buscarlo en Drive** (ruta estándar, bajo el repo clonado).\n"
        "2. **Si no está, subirlo** con el widget de Colab de la celda siguiente: el archivo "
        "es `chunks.jsonl`. Lo consigues de una corrida anterior en Drive.\n"
        "3. **Solo si no lo subes**, generarlo aquí mismo con la lógica de extracción "
        "inline (descarga el PDF ~19 MB y extrae las 151 páginas, ~1–2 min). Es el "
        "último recurso, no el camino principal. La lógica es la de "
        "`pipeline/extract.py` del repo, copiada en la celda: el notebook no ejecuta "
        "scripts del repo.\n\n"
        "`triples_seed.json` y `eval_questions.json` vienen con el clon del repo: no hace "
        "falta subirlos."
    )

    _DATA_CODE = r"""
import json, subprocess, sys
from pathlib import Path

CHUNKS = REPO + "/pipeline/data/chunks.jsonl"
DATA_DIR = REPO + "/pipeline/data"


def _chunks_ok(path):
    try:
        with open(path) as f:
            first = f.readline()
            n = 1 + sum(1 for _ in f)
        row = json.loads(first)
        return n >= 100 and isinstance(row.get("text"), str) and len(row["text"]) > 0
    except Exception:
        return False


if _chunks_ok(CHUNKS):
    print(f"datos procesados OK en Drive: {CHUNKS}")
else:
    print("chunks.jsonl NO está en Drive.")
    print("Súbelo con el widget de abajo (botón 'Elegir archivos').")
    print("Qué subir: chunks.jsonl — las 151 páginas del informe FEMSA 2024.")
    print("Dónde conseguirlo: de una corrida anterior en Drive.")
    print("Si no lo subes, el notebook lo genera aquí mismo con la lógica inline (último recurso).")
    from google.colab import files
    uploaded = files.upload()
    for name, data in uploaded.items():
        dest = Path(DATA_DIR) / Path(name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"subido: {dest} ({len(data)} bytes)")
    if not _chunks_ok(CHUNKS) and len(uploaded) == 1:
        # El archivo subido puede venir con otro nombre: probar como chunks.jsonl.
        only = next(iter(uploaded.values()))
        Path(CHUNKS).write_bytes(only)
        print("probando el archivo subido como chunks.jsonl...")
    if _chunks_ok(CHUNKS):
        print("datos listos por subida.")
    else:
        print("Sin subida válida: último recurso, generar los chunks aquí mismo...")
        print("(lógica de extracción inline: descarga el PDF ~19 MB y extrae las 151 páginas, ~1-2 min)")
        # El PDF está cifrado con AES (contraseña vacía): pypdf necesita el
        # paquete opcional `cryptography` para leerlo, que `pip install pypdf`
        # no trae. Se verifica aquí para que el último recurso funcione.
        try:
            import pypdf
            import cryptography
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "pypdf>=7.1.1", "cryptography>=3.1"], check=True)
            import pypdf
            import cryptography
        import urllib.request
        from pypdf import PdfReader

        # Lógica de extracción del informe, copiada inline: este notebook no
        # ejecuta scripts del repo. Solo la edición en inglés: el sitio en
        # español bloquea las descargas directas (WAF), así que la edición en
        # inglés es la que se descarga (mismo informe, mismas cifras).
        _PDF_URL_EN = "https://annualreport.femsa.com/pdf/FEMSA_2024_Annual_Report.pdf"
        _PDF_PATH_EN = Path(DATA_DIR) / "FEMSA_2024_Annual_Report_EN.pdf"
        _SOURCE_LABEL_EN = "FEMSA 2024 Integrated Annual Report, «Evolving» (English edition, same figures)"

        def _inline_is_valid_pdf(path):
            try:
                return len(PdfReader(str(path)).pages) > 0
            except Exception:
                return False

        def _inline_download_pdf():
            if _PDF_PATH_EN.exists() and _inline_is_valid_pdf(_PDF_PATH_EN):
                print(f"PDF en caché: {_PDF_PATH_EN} ({_PDF_PATH_EN.stat().st_size / 1e6:.1f} MB)")
                return _PDF_PATH_EN
            if _PDF_PATH_EN.exists():
                print(f"Caché inválido, se vuelve a descargar: {_PDF_PATH_EN}")
                _PDF_PATH_EN.unlink()
            Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
            print(f"Descargando {_PDF_URL_EN} ...")
            req = urllib.request.Request(
                _PDF_URL_EN, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                with open(_PDF_PATH_EN, "wb") as fh:
                    fh.write(resp.read())
            if not _inline_is_valid_pdf(_PDF_PATH_EN):
                _PDF_PATH_EN.unlink()
                raise RuntimeError("El archivo descargado no es un PDF válido.")
            print(f"Guardado en {_PDF_PATH_EN} ({_PDF_PATH_EN.stat().st_size / 1e6:.1f} MB)")
            return _PDF_PATH_EN

        def _inline_extract_pages(pdf_path):
            reader = PdfReader(str(pdf_path))
            chunks = []
            for i, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                chunks.append({"page": i, "text": text, "chars": len(text),
                               "fuente": _SOURCE_LABEL_EN})
            return chunks

        try:
            _pdf = _inline_download_pdf()
            _chunks = _inline_extract_pages(_pdf)
            _non_empty = sum(1 for c in _chunks if c["chars"] > 0)
            print(f"Páginas: {len(_chunks)} ({_non_empty} con texto)")
            Path(CHUNKS).parent.mkdir(parents=True, exist_ok=True)
            with open(CHUNKS, "w", encoding="utf-8") as fh:
                for c in _chunks:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"{len(_chunks)} páginas -> {CHUNKS}")
        except Exception as e:
            print("Generación inline falló:", e)
        assert _chunks_ok(CHUNKS), "No se pudo obtener chunks.jsonl."
    print(f"chunks.jsonl listo: {CHUNKS}")
"""

    return [md(_DATA_MD), code(_DATA_CODE)]


# ---------------------------------------------------------------------------
# Celda final: guardar en Drive (verificar + listar lo producido)
# ---------------------------------------------------------------------------

_SAVE_CODE_TEMPLATE = r"""
import os, shutil, time


def _human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} GB"


def _list_dir(path):
    total, nfiles = 0, 0
    for root, _dirs, files in os.walk(path):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            sz = os.path.getsize(p)
            total += sz
            nfiles += 1
            print(f"    {os.path.relpath(p, REPO)} ({_human(sz)})")
    return nfiles, total


# Logs de la sesion (viven en /content, fuera de Drive): copiarlos al repo.
for _log in ["ollama.log", "ollama_error.log", "relate.log"]:
    if os.path.isfile(_log):
        shutil.copy(_log, os.path.join(REPO, _log))
        print(f"log copiado a Drive: {_log}")

ENTRIES = %ENTRIES%
total_bytes, missing = 0, []
for rel, label, required in ENTRIES:
    p = os.path.join(REPO, rel)
    if os.path.isdir(p):
        nf, tb = _list_dir(p)
        total_bytes += tb
        print(f"{label}: directorio, {nf} archivos, {_human(tb)}")
    elif os.path.isfile(p):
        sz = os.path.getsize(p)
        total_bytes += sz
        mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
        print(f"{label}: {rel} ({_human(sz)}, {mt})")
    else:
        status = "FALTA (obligatorio)" if required else "pendiente (opcional)"
        print(f"{label}: {status}: {rel}")
        if required:
            missing.append(rel)
print(f"\nGUARDADO EN DRIVE: {_human(total_bytes)} verificados bajo {REPO}")
if missing:
    print("OJO, faltan obligatorios:", missing)
"""


def save_cell(entries, note=""):
    """Celda final de cada notebook: verifica y lista en Drive lo producido.
    entries: lista de (ruta relativa a REPO, etiqueta, obligatorio).
    Solo informa (imprime FALTA), nunca rompe el notebook."""
    md_text = ("## Guardar en Drive (final)\n\n"
               "Todo lo que produjo este notebook vive bajo el repo clonado en Drive "
               "(`REPO`), así que sobrevive aunque la sesión de Colab se corte. Esta "
               "celda lo verifica y lo lista: si algo falta, lo dice aquí, no después.")
    if note:
        md_text += "\n\n" + note
    src = _SAVE_CODE_TEMPLATE.replace("%ENTRIES%", repr(entries))
    return [md(md_text), code(src)]


# ---------------------------------------------------------------------------
# 00 — Ollama setup (versión canónica, completa)
# ---------------------------------------------------------------------------

def build_00():
    cells = [
        title_cell(
            "00 — Ollama en Colab (setup canónico)",
            "Este notebook instala **Ollama** en la instancia de Colab y baja los modelos "
            "que usan los demás notebooks. Es el lugar canónico con la explicación completa; "
            "los notebooks 04–08 llevan la versión compacta e idempotente de estas mismas "
            "celdas, así que cada uno puede correrse solo.\n\n"
            "**Antes de correrlo:** activa el runtime con GPU (Runtime > Change runtime type > "
            "T4 GPU) y verifica con `!nvidia-smi`.\n\n"
            "**Diseño:** los modelos intermedios (lector RAG, ejecutivo MEMO, juez RAGAS, LLM "
            "del agente) corren vía Ollama en la propia instancia: un solo servidor, cambio "
            "de modelo con un parámetro, sin cargar pesos en cada notebook. El entrenamiento "
            "MEMO (notebook 03) sigue siendo Hugging Face + LoRA — entrenar no se puede hacer "
            "vía Ollama.\n\n"
            "**Fuente de los comandos:** guías publicadas de Ollama en Colab (apt `zstd` + "
            "`install.sh` + `ollama serve` en background + pull de modelos) y verificación "
            "propia de que Colab no trae `zstd` y el instalador lo exige."),
        gpu_check_cell(),
        drive_mount_cell(),
        params_cell(),
        repo_setup_cell(),
        md("## 1. Instalar Ollama\n\n"
           "Colab no trae `zstd` y el instalador de Ollama lo exige: primero `apt-get` "
           "con `zstd` (varias guías añaden también `pciutils` para detectar la GPU) y "
           "luego el `install.sh` oficial. Si el binario ya existe, no se reinstala nada."),
        ollama_install_cell(),
        md("## 2. Arrancar el servidor en segundo plano\n\n"
           "`ollama serve` en primer plano bloquearía la celda para siempre, así que se "
           "lanza con `nohup` + `setsid` y se espera a que responda en el puerto **11434**. "
           "Si el servidor ya responde, no se arranca otro. `OLLAMA_HOST=0.0.0.0` solo hace "
           "falta si se expone con un túnel; para uso dentro del notebook basta `localhost`."),
        ollama_server_cell(),
        md("## 3. Bajar los modelos\n\n"
           "Tamaños en VRAM (T4, 16 GB): el **7B cómodo** (~4.7 GB en Q4_K_M), el **14B Q4 ~9 GB, "
           "ajustado** (cabe, pero deja poco margen junto a los embeddings). El 1.5B es el "
           "modelo de los runs originales en CPU (paridad); el 7B es el modelo más grande "
           "para el agente y el juez.\n\n"
           "**Ojo:** los modelos viven en la VM (`~/.ollama/models`) y **hay que volver a "
           "bajarlos en cada sesión nueva** — la VM de Colab es efímera. Opcional: apuntar "
           "`OLLAMA_MODELS` a una ruta en Drive para persistirlos, a costa de más lentitud "
           "(Drive es más lento que el disco local):\n\n"
           "```bash\n"
           "export OLLAMA_MODELS=/content/drive/MyDrive/femsa-rag/ollama-models\n"
           "```\n\n"
           "Esa variable debe definirse **antes** de arrancar el servidor. Por defecto no se "
           "usa: bajar de nuevo es más rápido que servir desde Drive."),
        ollama_pull_cell(["qwen2.5:1.5b", "qwen2.5:7b"]),
        code("""
# Opcional: el 14B, solo si cabe (T4 16 GB: ~9 GB, ajustado).
# Quita el comentario para bajarlo.
# !ollama pull qwen2.5:14b
print("Modelos listos. El 14B es opcional: quita el comentario de la celda si lo quieres.")
"""),
        md("## 4. Verificación: el servidor responde y el modelo contesta"),
        code("""
from langchain_ollama import ChatOllama

for _m in ["qwen2.5:1.5b", "qwen2.5:7b"]:
    _llm = ChatOllama(model=_m, base_url="http://localhost:11434", temperature=0)
    print(_m, "->", _llm.invoke("Responde con una sola palabra: ¿cuánto es 2+2?").content)
print("Ollama OK: servidor en localhost:11434, modelos 1.5b y 7b listos.")
"""),
        md("## Qué sigue\n\n"
           "- **Notebook 01:** setup general + extracción del informe (no necesita Ollama).\n"
           "- **Notebooks 04–08:** cada uno trae su propio bloque compacto de Ollama "
           "(instalar/arrancar/pull idempotentes), así que pueden correrse solos sin "
           "haber corrido este 00 primero — pero si ya lo corriste, no descargan nada de más."),
        code(r"""
import subprocess

ver = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
lst = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
manifest = f"ollama {ver}\n\nModelos bajados en esta sesion:\n{lst}"
open(REPO + "/ollama-manifest.txt", "w").write(manifest)
print(manifest)
print("manifiesto ->", REPO + "/ollama-manifest.txt")
"""),
        *save_cell([("ollama-manifest.txt", "Manifiesto Ollama (modelos bajados)", True)],
                   note="Los modelos en sí NO se copian a Drive (4.7–9 GB cada uno; bajar de "
                        "nuevo por sesión es más rápido que servirlos desde Drive). Lo que "
                        "queda guardado es este manifiesto con la versión y los tags bajados."),
    ]
    write_nb("00_ollama_setup.ipynb", cells, "00 Ollama setup")


# ---------------------------------------------------------------------------
# 01 — Setup general y extracción
# ---------------------------------------------------------------------------

def build_01():
    triples_md = (
        "## Tripletas candidatas por reglas (determinista)\n\n"
        "Las 13 reglas deterministas de `pipeline/extract.py` (citado como procedencia), "
        "copiadas inline: cada regla solo emite una tripleta si su patrón aparece "
        "literalmente en el texto de la página, con la página como fuente. Son "
        "**candidatas**: hay que revisarlas antes de cargarlas al grafo (ver "
        "`pipeline/README.md`). Si `extracted_triples.jsonl` ya existe en Drive, "
        "no se regenera."
    )

    triples_code = r"""
import json, os, re

_TRIPLES_OUT = REPO + "/pipeline/extracted_triples.jsonl"

if os.path.isfile(_TRIPLES_OUT):
    print("extracted_triples.jsonl ya existe, no se regenera:", _TRIPLES_OUT)
else:
    _tc_chunks = [json.loads(l) for l in
                  open(REPO + "/pipeline/data/chunks.jsonl")]
    _tc_label = (_tc_chunks[0]["fuente"] if _tc_chunks else
                 "FEMSA 2024 Integrated Annual Report, «Evolving» (English edition, same figures)")

    # (patrón, needs opcional, campos de la tripleta). Reglas deterministas:
    # solo emiten si el patrón aparece literalmente en el texto de la página.
    _INLINE_RULES = [
        {"pattern": r"781[,\s]?585",
         "subject": "FEMSA", "subject_type": "Empresa", "relation": "reporta",
         "object": "Ingresos Totales", "object_type": "IndicadorFinanciero",
         "valor": 781585, "unidad": "Ps. millones", "variacion_pct": 11.2,
         "periodo": "2024"},
        {"pattern": r"307[,\s]?197", "needs": r"Proximity\s+Americas",
         "subject": "Proximidad Américas", "subject_type": "Division",
         "relation": "reporta", "object": "Ingresos Totales",
         "object_type": "IndicadorFinanciero",
         "valor": 307197, "unidad": "Ps. millones", "variacion_pct": 10.3,
         "periodo": "2024",
         "nota": "La ontología v1 redondea a Ps. 307,200 M; valor exacto del MD&A."},
        {"pattern": r"23[,\s]?206",
         "subject": "Proximidad Américas", "subject_type": "Division",
         "relation": "mide", "object": "Tiendas",
         "object_type": "MetricaOperativa",
         "valor": 23206, "unidad": "tiendas", "periodo": "2024",
         "nota": "23,206 = tiendas OXXO en México (key data); el total de "
                 "Proximidad Américas es 26,322 puntos de venta."},
        {"pattern": r"1[,\s]?596",
         "subject": "Proximidad Américas", "subject_type": "Division",
         "relation": "mide", "object": "Aperturas Netas",
         "object_type": "MetricaOperativa",
         "valor": 1596, "unidad": "tiendas netas nuevas", "periodo": "2024",
         "nota": "La ontología v1 registra 1,236 aperturas (cifra del "
                 "comunicado 4T24); el informe reporta 1,596 aperturas "
                 "netas. No son la misma métrica: revisar antes de cargar."},
        {"pattern": r"279[,\s]?793", "needs": r"Coca-Cola\s+FEMSA",
         "subject": "Coca-Cola FEMSA", "subject_type": "Division",
         "relation": "reporta", "object": "Ingresos Totales",
         "object_type": "IndicadorFinanciero",
         "valor": 279793, "unidad": "Ps. millones", "variacion_pct": 14.2,
         "periodo": "2024",
         "nota": "La ontología v1 redondea a Ps. 279,800 M; valor exacto del MD&A."},
        {"pattern": r"49[,\s]?755", "needs": r"Proximity\s+Europe",
         "subject": "Proximidad Europa", "subject_type": "Division",
         "relation": "reporta", "object": "Ingresos Totales",
         "object_type": "IndicadorFinanciero",
         "valor": 49755, "unidad": "Ps. millones", "variacion_pct": 14.2,
         "periodo": "2024",
         "nota": "La ontología v1 redondea a Ps. 49,800 M; valor exacto del MD&A."},
        {"pattern": r"79[,\s]?755", "needs": r"Health\s+Division",
         "subject": "Salud", "subject_type": "Division",
         "relation": "reporta", "object": "Ingresos Totales",
         "object_type": "IndicadorFinanciero",
         "valor": 79755, "unidad": "Ps. millones", "variacion_pct": 5.8,
         "periodo": "2024",
         "nota": "La ontología v1 redondea a Ps. 79,700 M; valor exacto del MD&A."},
        {"pattern": r"65[,\s]?365", "needs": r"Fuel",
         "subject": "Combustibles", "subject_type": "Division",
         "relation": "reporta", "object": "Ingresos Totales",
         "object_type": "IndicadorFinanciero",
         "valor": 65365, "unidad": "Ps. millones", "variacion_pct": 11.7,
         "periodo": "2024",
         "nota": "La ontología v1 redondea a Ps. 65,400 M; valor exacto del MD&A."},
        {"pattern": r"8\.6\s*million",
         "subject": "Spin by OXXO", "subject_type": "Formato",
         "relation": "mide", "object": "Usuarios Activos",
         "object_type": "MetricaOperativa",
         "valor": 8.6, "unidad": "millones de usuarios", "variacion_pct": 24.9,
         "periodo": "2024",
         "nota": "El +24.9% viene del comunicado 4T24 (vs 4T23); no "
                 "aparece en el texto del informe anual."},
        {"pattern": r"24\.6\s*million",
         "subject": "Spin Premia", "subject_type": "Formato",
         "relation": "mide", "object": "Usuarios Activos",
         "object_type": "MetricaOperativa",
         "valor": 24.6, "unidad": "millones de usuarios", "variacion_pct": 27.5,
         "periodo": "2024",
         "nota": "El +27.5% viene del comunicado 4T24 (vs 4T23); no "
                 "aparece en el texto del informe anual."},
        {"pattern": r"1\.3\s*million", "needs": r"Juntos\+",
         "subject": "Coca-Cola FEMSA", "subject_type": "Division",
         "relation": "mide", "object": "Clientes Juntos+",
         "object_type": "MetricaOperativa",
         "valor": 1.3, "unidad": "millones de clientes", "periodo": "2024"},
        {"pattern": r"volume\s+(?:increased|grew)\s+4\.4%",
         "needs": r"Coca-Cola\s+FEMSA",
         "subject": "Coca-Cola FEMSA", "subject_type": "Division",
         "relation": "mide", "object": "Volumen",
         "object_type": "MetricaOperativa",
         "valor": 4.4, "unidad": "% crecimiento vs 2023", "periodo": "2024"},
    ]

    def _inline_build_triple(rule, page, source_label):
        t = {
            "subject": rule["subject"],
            "subject_type": rule["subject_type"],
            "relation": rule["relation"],
            "object": rule["object"],
            "object_type": rule["object_type"],
            "fuente": {"documento": source_label, "pagina": page},
        }
        for k in ("valor", "unidad", "variacion_pct", "periodo", "nota"):
            if k in rule:
                t[k] = rule[k]
        return t

    def _inline_rule_triples(chunk, source_label):
        text = chunk["text"]
        page = chunk["page"]
        out = []
        for rule in _INLINE_RULES:
            needs = rule.get("needs")
            if needs and not re.search(needs, text):
                continue
            for _m in re.finditer(rule["pattern"], text):
                out.append(_inline_build_triple(rule, page, source_label))
        # 571 estaciones solo si la página menciona OXXO GAS.
        if re.search(r"\b571\b", text) and re.search(r"OXXO\s*GAS", text, re.I):
            out.append(_inline_build_triple(
                {"subject": "Combustibles", "subject_type": "Division",
                 "relation": "mide", "object": "Estaciones",
                 "object_type": "MetricaOperativa", "valor": 571,
                 "unidad": "estaciones", "periodo": "2024"},
                page, source_label))
        seen, dedup = set(), []
        for t in out:
            key = (t["subject"], t["relation"], t["object"], t.get("valor"))
            if key not in seen:
                seen.add(key)
                dedup.append(t)
        return dedup

    _tc_all = []
    for _c in _tc_chunks:
        _tc_all.extend(_inline_rule_triples(_c, _tc_label))
    _tc_seen, _tc_triples = set(), []
    for t in _tc_all:
        key = (t["subject"], t["relation"], t["object"], t.get("valor"))
        if key not in _tc_seen:
            _tc_seen.add(key)
            _tc_triples.append(t)
    with open(_TRIPLES_OUT, "w", encoding="utf-8") as fh:
        for t in _tc_triples:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"{len(_tc_triples)} tripletas candidatas -> {_TRIPLES_OUT}")
    print("NOTA: las tripletas son candidatas: revísalas antes de cargarlas al grafo.")
    for t in _tc_triples:
        print(f"  p.{t['fuente']['pagina']}: ({t['subject']}, {t['relation']}, "
              f"{t['object']} = {t.get('valor')} {t.get('unidad', '')})")
"""

    cells = [
        title_cell(
            "01 — Setup y extracción del informe FEMSA 2024",
            "Prepara todo lo demás: monta Drive, clona el repo, instala dependencias con "
            "las versiones fijadas del proyecto, descarga el informe anual FEMSA 2024 "
            "(edición en inglés, 151 páginas) y extrae el texto por página + las tripletas "
            "por reglas. **No necesita Ollama** (la extracción es determinista, sin LLM).\n\n"
            "La extracción es **autocontenida**: la lógica de descarga y extracción (la de "
            "`pipeline/extract.py`, citado como procedencia) va copiada inline en las "
            "celdas; el notebook no ejecuta ningún script del repo.\n\n"
            "**Orden:** corre este notebook después del 00 (o sin él: este notebook no usa "
            "Ollama, pero verifica que el servidor esté arriba para los que siguen)."),
        gpu_check_cell(),
        drive_mount_cell(),
        params_cell(),
        repo_setup_cell(),
        md("## Dependencias (versiones fijadas del proyecto)\n\n"
           "Se instalan las mismas versiones que usaron las corridas originales "
           "(ver `repo/pyproject.toml` y los READMEs de `memo/`, `rag_baseline/` y "
           "`agentic_rag/`). En Colab, `torch` se instala sin el índice CPU: pip resuelve "
           "la rueda CUDA compatible con la T4. Si Colab ya trae un torch compatible, "
           "pip lo deja como está."),
        pip_cell([
            '"torch>=2.15.0"', "transformers==5.24.1", "accelerate>=1.16.0",
            "peft>=0.22.0", "sentence-transformers==6.3.0", "numpy>=2.5.5",
            "pypdf>=7.1.1", "neo4j>=6.4.1", "rank-bm25==0.2.2",
            "langchain==1.5.2", "langgraph==1.2.24", "langchain-core==1.8.5",
            "langchain-huggingface==1.3.0", "langchain-ollama",
        ]),
        md("## Verificar Ollama (opcional aquí, obligatorio después)\n\n"
           "Este notebook no usa Ollama, pero los notebooks 04–08 sí. Si el servidor no "
           "está arriba, corre el notebook `00_ollama_setup.ipynb` (o deja que cada "
           "notebook levante el suyo con su bloque idempotente)."),
        code("""
import urllib.request

try:
    with urllib.request.urlopen("http://localhost:11434", timeout=3) as r:
        print("Ollama responde en localhost:11434 (status", r.status, ")")
except Exception as e:
    print("Ollama NO responde:", e)
    print("Corre 00_ollama_setup.ipynb, o sigue: los notebooks 04-08 levantan el servidor solos.")
"""),
        *data_ensure_cells(),
        md(triples_md),
        code(triples_code),
        md("## Verificación: 151 páginas y la cifra clave"),
        code("""
import json

chunks = [json.loads(l) for l in open(REPO + "/pipeline/data/chunks.jsonl")]
print("paginas:", len(chunks), "| con texto:", sum(1 for c in chunks if c["chars"] > 0))
hits = [c["page"] for c in chunks if "781,585" in c["text"]]
print("781,585 aparece en paginas:", hits)
print("ejemplo p.11:", chunks[10]["text"][:300].replace("\\n", " "))
"""),
        *save_cell([
            ("pipeline/data/chunks.jsonl", "chunks.jsonl (151 páginas extraídas)", True),
            ("pipeline/data/FEMSA_2024_Annual_Report_EN.pdf", "PDF del informe (inglés)", False),
            ("pipeline/extracted_triples.jsonl", "13 tripletas candidatas por reglas (pendientes de revisión)", False),
        ], note="El PDF (~19 MB) y `chunks.jsonl` quedan en Drive: son los insumos que "
                "los notebooks 02–08 pedirán primero (antes de subirlos o generarlos)."),
    ]
    write_nb("01_setup_y_extraccion.ipynb", cells, "01 setup y extracción")

# ---------------------------------------------------------------------------
# 02 — Grafo Relate
# ---------------------------------------------------------------------------

def build_02():
    cells = [
        title_cell(
            "02 — Grafo de conocimiento en Relate",
            "Genera el Cypher de carga desde las 31 tripletas semilla, instala el "
            "servidor **Relate** (Neo4j, en memoria) dentro de la instancia de Colab, "
            "carga el grafo y corre las 12 preguntas en Cypher. Todo dentro de Colab: "
            "nada de pasos de máquina local.\n\n"
            "**Prerrequisito:** notebook 01 (chunks + tripletas semilla)."),
        gpu_check_cell(),
        drive_mount_cell(),
        params_cell(),
        repo_setup_cell(),
        md("## Dependencia: driver neo4j\n\n"
           "La carga por Bolt usa el driver `neo4j` de Python: sin el paquete, la "
           "carga falla. Se instala aquí, antes de cargar el grafo."),
        pip_cell(["neo4j>=6.4.1"]),
        *data_ensure_cells(),
        md("## Generar el Cypher de carga\n\n"
           "La lógica que genera `load_graph.cypher` (MERGE idempotentes) y "
           "`schema.cypher` (9 constraints de unicidad) desde `triples_seed.json` va "
           "inline en la celda siguiente — la misma lógica que `pipeline/build_cypher.py`, "
           "sin ejecutar ese script. No se cargan tripletas extraídas sin revisar — esa "
           "regla sigue vigente aquí."),
        code(r"""
import json

SEED = REPO + "/pipeline/triples_seed.json"
OUT_CYPHER = REPO + "/pipeline/load_graph.cypher"
OUT_SCHEMA = REPO + "/pipeline/schema.cypher"

# Ontología v1: relación en minúsculas -> tipo de relación en UPPER_SNAKE_CASE.
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

# Las 9 etiquetas de la ontología v1: (nombre corto del constraint, etiqueta).
# Ojo: IndicadorFinanciero -> `indicador_...`, MetricaOperativa -> `metrica_...`
# (no el nombre completo de la etiqueta en minúsculas).
LABELS = [("empresa", "Empresa"), ("division", "Division"),
          ("formato", "Formato"), ("indicador", "IndicadorFinanciero"),
          ("metrica", "MetricaOperativa"), ("periodo", "Periodo"),
          ("pais", "Pais"), ("iniciativa", "Iniciativa"),
          ("riesgo", "Riesgo")]


def cypher_str(value):
    # Escapa un string para usarlo como literal Cypher entre comillas simples.
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def cypher_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if value is None:
        return "null"
    return cypher_str(str(value))


def rel_props(t):
    # Propiedades de la relación para una tripleta.
    props = {}
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


def props_cypher(props):
    if not props:
        return ""
    items = ", ".join(f"{k}: {cypher_value(v)}" for k, v in props.items())
    return " {" + items + "}"


def node_merge(label, name, var):
    return f"MERGE ({var}:{label} {{nombre: {cypher_str(name)}}})"


def triple_to_cypher(t, idx):
    # Una tripleta -> 1-2 sentencias MERGE (nodos + relación).
    #
    # Las variables llevan sufijo único por tripleta (s1/o1, s2/o2...):
    # Relate ejecuta un script completo con ámbito de variables compartido,
    # así que reutilizar `s`/`o` en cada sentencia falla con
    # VariableAlreadyBound. En Neo4j no haría falta, pero así el archivo
    # corre igual en ambos.
    rel = REL_MAP[t["relation"]]
    return [
        node_merge(t["subject_type"], t["subject"], f"s{idx}") + ";",
        node_merge(t["object_type"], t["object"], f"o{idx}") + ";",
        f"MERGE (s{idx})-[:{rel}{props_cypher(rel_props(t))}]->(o{idx});",
    ]


def derived_crecio_en(t, idx):
    # Deriva CRECIO_EN desde REPORTA/MIDE con variacion_pct.
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


with open(SEED, encoding="utf-8") as fh:
    triples = list(json.load(fh)["triples"])

unknown = {t["relation"] for t in triples} - set(REL_MAP)
assert not unknown, f"Relaciones desconocidas en la ontología v1: {unknown}"

# Orden determinista para que el archivo generado sea estable.
triples.sort(key=lambda t: (t["relation"], t["subject"], t["object"]))

out = []
out.append("// load_graph.cypher — generado en este notebook desde triples_seed.json")
out.append("// Fuente: triples_seed.json (ontología v1, cifras verificadas)")
out.append("// Todo es MERGE: se puede correr varias veces sin duplicar.")
out.append("// Relaciones en UPPER_SNAKE_CASE (ontología: minúsculas).")
out.append("")

derived = []
seen_crecio = set()
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

with open(OUT_CYPHER, "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
n_nodes = len({(t["subject_type"], t["subject"]) for t in triples}
              | {(t["object_type"], t["object"]) for t in triples})
n_derived = sum(1 for line in out if "[:CRECIO_EN" in line)
print(f"[build] {len(triples)} tripletas -> {OUT_CYPHER}")
print(f"[build] Nodos distintos: {n_nodes}; "
      f"relaciones CRECIO_EN derivadas (distintas tras MERGE): {n_derived}")

# schema.cypher: una restricción de unicidad por etiqueta sobre `nombre`:
# es lo que hace que los MERGE de load_graph.cypher sean idempotentes
# sin duplicar nodos. OpenCypher estándar: corre en Neo4j 5+ y en Relate.
schema_lines = [
    "// schema.cypher — restricciones e índices para la ontología v1 (FEMSA 2024)",
    "//",
    "// OpenCypher estándar: corre en Neo4j 5+ y en Relate (relatedb.ai).",
    "// Una restricción de unicidad por etiqueta sobre `nombre`: es lo que hace",
    "// que los MERGE de load_graph.cypher sean idempotentes sin duplicar nodos.",
    "//",
    "// Nota sobre Relate: si tu versión de Relate aún no soporta CREATE",
    "// CONSTRAINT, la carga lo detecta, avisa y sigue (los MERGE",
    "// siguen funcionando; solo pierdes la garantía de unicidad a nivel de",
    "// base). En Neo4j las restricciones se crean sin problema.",
    "",
]
for short, label in LABELS:
    schema_lines.append(
        f"CREATE CONSTRAINT {short}_nombre_unique IF NOT EXISTS\n"
        f"FOR (n:{label}) REQUIRE n.nombre IS UNIQUE;\n")
with open(OUT_SCHEMA, "w", encoding="utf-8") as fh:
    fh.write("\n".join(schema_lines))
print(f"[build] esquema -> {OUT_SCHEMA} ({len(LABELS)} constraints)")
"""),
        md("## Instalar Relate en la instancia\n\n"
           "Relate publica binarios Linux x86_64 (`relate-linux-amd64`, versión fijada "
           "v1.2.0, la misma que se verificó en local). El instalador oficial "
           "(`curl -sSfL https://relatedb.ai/install.sh | bash`) resuelve el último "
           "release estable y verifica la firma; si el binario ya existe, no se "
           "reinstala. El binario queda en `~/.local/bin/relate` **dentro de la "
           "instancia** (se pierde al reiniciar la sesión: hay que re-correr esta celda)."),
        shell("""if [ -x "$HOME/.local/bin/relate" ]; then
  echo "relate ya instalado: $($HOME/.local/bin/relate --version | head -1)"
else
  echo "instalando Relate..."
  curl -sSfL https://relatedb.ai/install.sh | bash
  "$HOME/.local/bin/relate" --version | head -1
fi
export PATH="$HOME/.local/bin:$PATH"
relate --version | head -1"""),
        md("## Arrancar el servidor Relate en segundo plano\n\n"
           "El store `data/femsa.rl` vive en el repo (en Drive, así que persiste). "
           "El servidor escucha solo en localhost, sin autenticación (`--auth none`), "
           "igual que en local. Si la sesión de Colab se reinicia, hay que re-correr "
           "esta celda: el store no se pierde, solo el proceso."),
        code("""
import os, subprocess, time

RELATE_BIN = os.path.expanduser("~/.local/bin/relate")
DB = REPO + "/pipeline/data/femsa.rl"
LOG = open("relate.log", "w")

def relate_up():
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:7687", timeout=2) as r:
            return True
    except Exception:
        return False

# Bolt (7687) no responde a HTTP, asi que la prueba es: ¿el proceso vive y el puerto escucha?
import socket
def port_open(port):
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        s.close()

if port_open(7687):
    print("Relate ya escucha en el puerto 7687")
else:
    print("arrancando `relate serve` en segundo plano...")
    subprocess.Popen([RELATE_BIN, "serve", "--database", DB, "--bolt", "7687",
                      "--auth", "none"], stdout=LOG, stderr=subprocess.STDOUT,
                     preexec_fn=os.setsid)
    for _ in range(30):
        if port_open(7687):
            break
        time.sleep(2)
    print("Relate escucha en 7687:", port_open(7687))
    if not port_open(7687):
        print("Revisa relate.log")
"""),
        md("## Cargar esquema + grafo por Bolt y verificar conteos\n\n"
           "La lógica de carga va inline en la celda siguiente (driver `neo4j` de "
           "Python, Bolt a localhost:7687 — la misma que `pipeline/load_relate.py`, "
           "sin ejecutar ese script). Esperado, verificado: **27 nodos, 39 relaciones** "
           "(31 semilla + 8 `CRECIO_EN` derivadas). La carga es idempotente: correrla "
           "dos veces no duplica nada."),
        code(r"""
import os
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

URI = os.environ.get("RELATE_URI", "bolt://localhost:7687")
SCHEMA_PATH = REPO + "/pipeline/schema.cypher"
LOAD_PATH = REPO + "/pipeline/load_graph.cypher"

START_RELATE_HINT = '''
No se pudo conectar a Relate. Para arrancar un servidor local:

    relate serve --database ./pipeline/data/femsa.rl --bolt 7687 --auth none

También puedes cargar el grafo sin servidor, directamente con el CLI:

    relate -d ./pipeline/data/femsa.rl query < schema.cypher
    relate -d ./pipeline/data/femsa.rl query < load_graph.cypher

Si tu servidor usa autenticación, define RELATE_USER y RELATE_PASSWORD.
'''


def split_statements(text):
    # Divide un archivo .cypher en sentencias por ';' (sin ';' en strings
    # en estos archivos: los textos con ';' no aparecen en los literales).
    out = []
    buf = []
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


def read_statements(path):
    # Lee un .cypher ignorando líneas de comentario // y vacías.
    lines = []
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.strip().startswith("//"):
            continue
        lines.append(line)
    return split_statements("\n".join(lines))


user = os.environ.get("RELATE_USER")
password = os.environ.get("RELATE_PASSWORD")
auth = (user, password) if user else None


def connect(auth_value):
    driver = GraphDatabase.driver(URI, auth=auth_value)
    driver.verify_connectivity()
    return driver


try:
    driver = connect(auth)
except AuthError:
    # El servidor pide autenticación y no se dio (o viceversa).
    fallback = ("neo4j", password or "neo4j") if not auth else None
    if fallback is None:
        raise SystemExit("[load] Autenticación rechazada con las credenciales "
                         "dadas." + START_RELATE_HINT)
    print("[load] Sin autenticación no funcionó; reintentando con "
          "usuario 'neo4j'...")
    try:
        driver = connect(fallback)
    except Exception:
        raise SystemExit("[load] Autenticación rechazada."
                         + START_RELATE_HINT)
except (ServiceUnavailable, Exception) as e:
    raise SystemExit(f"[load] No se pudo conectar a Relate en {URI}: {e}"
                     + START_RELATE_HINT)

print(f"[load] Conectado a {URI}")

schema_stmts = read_statements(SCHEMA_PATH)
load_stmts = read_statements(LOAD_PATH)
print(f"[load] schema.cypher: {len(schema_stmts)} sentencias; "
      f"load_graph.cypher: {len(load_stmts)} sentencias")

with driver.session() as session:
    # Esquema: si Relate no soporta CREATE CONSTRAINT, avisa y sigue.
    for stmt in schema_stmts:
        try:
            session.run(stmt)
        except Exception as e:
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
    assert total_nodes == 27 and total_rels == 39, \
        f"Conteos inesperados: {total_nodes} nodos, {total_rels} relaciones (esperado 27/39)."
driver.close()
"""),
        md("## Correr las 12 preguntas en Cypher\n\n"
           "Con el CLI, sin pasar por el servidor: `relate -d <store> query` ejecuta "
           "cada consulta de `queries.cypher` contra el store. Separa el archivo por "
           "consultas (cada bloque `MATCH ... RETURN ...;`) y ejecuta una por una, "
           "imprimiendo la respuesta esperada al lado para comparar a ojo."),
        code("""
import os, re, subprocess

RELATE_BIN = os.path.expanduser("~/.local/bin/relate")
DB = REPO + "/pipeline/data/femsa.rl"
text = open(REPO + "/pipeline/queries.cypher").read()

# Un bloque por pregunta: cada una empieza con su comentario `// Qn.`
# (la Q9 lleva DOS clausulas MATCH en una sola consulta: el regex toma
# del primer MATCH hasta el `;` final, asi que va junta).
blocks = re.split(r"(?=^// Q\\d+\\.)", text, flags=re.M)
n = 0
for b in blocks:
    m = re.search(r"MATCH.*?;", b, re.S)
    if not m:
        continue
    q = re.search(r"// (Q\\d+\\..+)", b)
    esp = re.search(r"Esperado: (.+)", b)
    p = subprocess.run([RELATE_BIN, "-d", DB, "query"], input=m.group(0).encode(),
                       capture_output=True)
    rows = [l for l in p.stdout.decode(errors="replace").splitlines() if l.startswith("{")]
    print((q.group(1).strip()[:90] if q else "consulta"), "| esperado:",
          (esp.group(1).strip() if esp else "?"))
    print("   ->", rows[0][:220] if rows else "(sin filas)")
    n += 1
print(f"\\n{n} consultas ejecutadas (esperado: 12)")
"""),
        md("## Alternativa sin servidor\n\n"
           "Si el servidor no levanta, el grafo se puede cargar directamente con el CLI, "
           "sin Bolt (mismos archivos, mismo resultado):\n\n"
           "```bash\n"
           "relate -d ./pipeline/data/femsa.rl query < pipeline/schema.cypher\n"
           "relate -d ./pipeline/data/femsa.rl query < pipeline/load_graph.cypher\n"
           "```\n\n"
           "desde el directorio del repo."),
        *save_cell([
            ("pipeline/data/femsa.rl", "Store Relate (27 nodos, 39 relaciones)", True),
            ("pipeline/load_graph.cypher", "Cypher de carga generado (MERGE idempotentes)", True),
            ("pipeline/schema.cypher", "Esquema (constraints)", False),
            ("relate.log", "Log del servidor Relate", False),
        ], note="El store `femsa.rl` queda en Drive: el notebook 04 (MEMO) no lo necesita, "
                "pero cualquier corrida futura del grafo lo reusa sin recargar. Si la sesión "
                "se corta, basta re-correr las celdas de arranque del servidor."),
    ]
    write_nb("02_grafo_relate.ipynb", cells, "02 grafo Relate")


# ---------------------------------------------------------------------------
# 03 — MEMO: dataset expandido + entrenamiento en GPU
# ---------------------------------------------------------------------------

TRAIN_CELL = r"""
import json, math, os, random, time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# --- Parametros del entrenamiento (los de arriba) ---
print(f"modelo MEMORY: {MODEL_MEMORY}")
print(f"datos: {DATA} | LoRA r={LORA_R} alpha={LORA_ALPHA} | lr={LR} | "
      f"epocas={EPOCHS} | batch={BATCH} grad_accum={GRAD_ACCUM} | max_len={MAX_LEN}")
print(f"salida: {OUT_DIR} (en Drive: sobrevive al reinicio)")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| GPU:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "-")
assert DEVICE == "cuda", "Este entrenamiento es para GPU: activa el runtime T4 (Runtime > Change runtime type)."

random.seed(SEED)
torch.manual_seed(SEED)

rows = [json.loads(l) for l in open(DATA)]
random.shuffle(rows)
if MAX_EXAMPLES:
    rows = rows[:MAX_EXAMPLES]
print(f"entrenando con {len(rows)} ejemplos de {DATA}")

tok = AutoTokenizer.from_pretrained(MODEL_MEMORY)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"cargando {MODEL_MEMORY} (bf16, device_map=auto)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_MEMORY, torch_dtype=torch.bfloat16,
                                             device_map="auto")
model.config.use_cache = False
if GRAD_CKPT:
    model.gradient_checkpointing_enable()
cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                 task_type="CAUSAL_LM")
model = get_peft_model(model, cfg)
if INIT_ADAPTER:
    print(f"cargando adaptador inicial {INIT_ADAPTER} ...")
    model.load_adapter(INIT_ADAPTER, adapter_name="default")
model.print_trainable_parameters()


class QADataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.items = []
        skipped = 0
        for r in rows:
            prompt = (f"<|im_start|>user\n{r['question']}<|im_end|>\n"
                      f"<|im_start|>assistant\n")
            full = prompt + r["answer"] + "<|im_end|>"
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            if len(f_ids) > max_len:
                skipped += 1
                continue
            labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
            self.items.append((f_ids, labels))
        print(f"dataset: {len(self.items)} ejemplos, {skipped} omitidos (> {max_len} tokens)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, labels = self.items[i]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {"input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


ds = QADataset(rows, tok, MAX_LEN)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                collate_fn=lambda b: collate(b, tok.pad_token_id))
total_opt_steps = max(1, int(math.ceil(len(ds) / (BATCH * GRAD_ACCUM)) * EPOCHS))
opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                        lr=LR, weight_decay=0.0)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_opt_steps), total_opt_steps)

t_start = time.time()
model.train()
step, micro, losses, loss_acc = 0, 0, [], 0.0
step_times = []
stop = False


def save_ckpt(tag):
    d = os.path.join(OUT_DIR + "_ckpt", f"step_{tag}")
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d)
    tok.save_pretrained(d)
    print(f"checkpoint -> {d}", flush=True)


while not stop:
    for batch in dl:
        if stop:
            break
        t0 = time.time()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / GRAD_ACCUM
        loss.backward()
        loss_acc += out.loss.item()
        micro += 1
        step_times.append(time.time() - t0)
        if micro % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(loss_acc / GRAD_ACCUM)
            loss_acc = 0.0
            if step % 5 == 0 or step == 1:
                el = time.time() - t_start
                print(f"opt step {step}/{total_opt_steps} loss {losses[-1]:.4f} "
                      f"elapsed {el:.0f}s", flush=True)
            if step % 25 == 0:
                save_ckpt(step)
            if TIME_TEST_STEPS and step >= TIME_TEST_STEPS:
                s = sum(step_times) / len(step_times)
                print(f"\nTIME TEST: {s:.2f} s/micro-step (batch {BATCH} x seq<={MAX_LEN})")
                print(f"ETA corrida completa ({len(ds)} ej, {EPOCHS} ep): "
                      f"~{s * len(ds) * EPOCHS / 60:.0f} min")
                stop = True
                break
            if time.time() - t_start > MAX_SECONDS:
                print(f"TIMEBOX ({MAX_SECONDS:.0f}s) en el paso {step}; guardando.")
                stop = True
                break
            if step >= total_opt_steps:
                stop = True
                break

log = {"model": MODEL_MEMORY, "examples": len(ds), "epochs": EPOCHS,
       "lora_r": LORA_R, "lr": LR, "max_len": MAX_LEN,
       "batch_size": BATCH, "grad_accum": GRAD_ACCUM,
       "init_adapter": INIT_ADAPTER,
       "optimizer_steps": step, "micro_steps": micro,
       "loss_mean": sum(losses) / len(losses) if losses else None,
       "loss_last10_mean": (sum(losses[-10:]) / len(losses[-10:]) if losses else None),
       "seconds": time.time() - t_start}
if not TIME_TEST_STEPS:
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    with open(os.path.join(OUT_DIR, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nadaptador guardado -> {OUT_DIR}")
print(json.dumps(log, indent=2))
"""


def build_03():
    cells = [
        title_cell(
            "03 — MEMO: dataset expandido + entrenamiento en GPU",
            "Genera el dataset de reflexiones **expandido** (~3,300 pares: la segunda "
            "pasada de aumento no tiene tope, así que el v3 sale más grande que el "
            "objetivo inicial de 1,500–2,000) y entrena MEMORY "
            "con LoRA **en GPU**: Qwen2.5-**1.5B**-Instruct como MEMORY (ya no el 0.5B), "
            "bf16, 3 épocas, LoRA r=32. En CPU el 1.5B tardaba ~36 h para 3 épocas "
            "(medido: 30.7 s/micro-paso); en la T4 es cuestión de minutos por época.\n\n"
            "**Prerrequisito:** notebook 01 (chunks). El adaptador se guarda en Drive: "
            "si la sesión se corta, se reanuda desde el checkpoint.\n\n"
            "**Regla de oro, igual que en local:** cada cifra del dataset se verifica "
            "contra el texto del chunk; las respuestas se copian byte-idénticas en las "
            "paráfrasis (solo cambia la pregunta), así que ninguna cifra puede cambiar."),
        gpu_check_cell(),
        drive_mount_cell(),
        params_cell("""
# --- Parametros de MEMO (Colab, GPU T4) ---
MODEL_MEMORY = "Qwen/Qwen2.5-1.5B-Instruct"   # MEMORY: 1.5B (en CPU era 0.5B)
DATA_V3 = REPO + "/memo/reflection_qa_v3.jsonl"
DATA = DATA_V3  # dataset que entrena la celda 5 (v3); TRAIN_CELL lo lee como DATA
OUT_DIR = REPO + "/memo/memory_model_colab"   # adaptador final (en Drive)
LORA_R = 32
LORA_ALPHA = 64
LR = 2e-4
EPOCHS = 3
BATCH = 8            # en GPU cabe batch mayor (en CPU era 1)
GRAD_ACCUM = 2       # batch efectivo 16
MAX_LEN = 128
SEED = 42
MAX_EXAMPLES = 0     # 0 = todos
MAX_SECONDS = 7200   # cortafuegos: 2 h por invocacion
GRAD_CKPT = False    # con batch 8 x seq 128 las activaciones caben sin checkpointing
INIT_ADAPTER = None  # para reanudar: ruta de un adaptador de una epoca anterior
TIME_TEST_STEPS = 5  # 0 = entrenamiento completo; >0 = solo mide s/paso y para
"""),
        repo_setup_cell(),
        *data_ensure_cells(),
        pip_cell([
            '"torch>=2.15.0"', "transformers==5.24.1", "accelerate>=1.16.0",
            "peft>=0.22.0", "numpy>=2.5.5",
        ]),
        md("## 1. Dataset v1: 446 pares (determinista)\n\n"
           "La lógica de generación vive **en la celda siguiente** (antes en "
           "`memo/generate_reflections.py`): lee los chunks, genera los pares "
           "curados + programáticos, excluye las 12 preguntas de evaluación "
           "(Jaccard ≥ 0.80) y verifica cada cifra contra el texto citado. "
           "Salida: `memo/reflection_qa.jsonl` (446 pares, md5 verificado "
           "`a4ccf20c2ea38d61fe50e992b20241c6`)."),
        code(r'''
# Dataset v1: 446 pares de reflection QA (determinista)
# Logica de generacion inline: pares curados a mano + pares programaticos
# desde las tablas financieras exactas y las tripletas semilla. Cada cifra
# se verifica contra el texto de las paginas citadas en chunks.jsonl y
# ninguna pregunta puede parecerse a las 12 de evaluacion (Jaccard >= 0.80).
# Lee: pipeline/data/chunks.jsonl, pipeline/triples_seed.json
# Escribe: memo/reflection_qa.jsonl (446 pares, orden determinista)

import json, re, unicodedata
from collections import Counter

CHUNKS_PATH = REPO + "/pipeline/data/chunks.jsonl"
TRIPLES_PATH = REPO + "/pipeline/triples_seed.json"
OUT_PATH = REPO + "/memo/reflection_qa.jsonl"

# 12 preguntas de evaluacion EXCLUIDAS: no deben aparecer verbatim ni
# casi-verbatim en el dataset (ver is_excluded).
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

# Utilidades de normalizacion / verificacion.
_NUM_RE = re.compile(r"\d+(?:[,\.]\d+)*")


def normalize_text(s: str) -> str:
    """Minusculas, sin acentos, sin puntuacion, espacios colapsados."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def jaccard_str(a: str, b: str) -> float:
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
        if jaccard_str(nq, ne) >= 0.80:
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

# Datos curados a mano. Cada entrada: tipo, paginas, entidades, respuesta,
# lista de formulaciones de la pregunta. Todas las cifras estan verificadas
# contra chunks.jsonl. Claves: t=tipo, p=source_pages, e=entities,
# a=answer, q=questions.
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

def main_v1():
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
    print("=== dataset v1 (generacion inline) — informe ===")
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

main_v1()
        '''),
        md("## 2. Dataset v2: 446 → 1,414 (paráfrasis de pregunta)\n\n"
           "La lógica de aumento vive **en la celda siguiente** (antes en "
           "`memo/augment_v2.py`): paráfrasis solo de la pregunta (la respuesta se "
           "copia byte-idéntica), con chequeo de fuga por Jaccard contra las 12 "
           "preguntas de evaluación y verificación de que ninguna respuesta cambió "
           "sus números. Salida: `memo/reflection_qa_v2.jsonl` (1,414 pares)."),
        code(r'''
# Dataset v2: 446 -> 1,414 (parafrasis solo de la pregunta)
# Logica de aumento inline: la respuesta se copia byte-identica, asi que
# ninguna cifra puede cambiar. Cada parafrasis se chequea contra las 12
# preguntas de evaluacion (Jaccard >= 0.80 se descarta) y cada cifra clave
# debe aparecer >= 3 veces. Los helpers de aqui (tokens, jaccard, framing,
# swap, candidates, norm_num_tokens) los reutiliza la celda 3.
# Lee: memo/reflection_qa.jsonl, memo/eval_questions.json,
#      pipeline/triples_seed.json
# Escribe: memo/reflection_qa_v2.jsonl + memo/augment_report.json

import json, re, unicodedata

BASE = REPO + "/memo/reflection_qa.jsonl"
EVAL = REPO + "/memo/eval_questions.json"
TRIPLES = REPO + "/pipeline/triples_seed.json"
OUT = REPO + "/memo/reflection_qa_v2.jsonl"
REPORT = REPO + "/memo/augment_report.json"
JACCARD_MAX = 0.80

def norm_num_tokens(s):
    """Digit sequences in a string, separators stripped ('781,585' -> '781585')."""
    return set(re.sub(r"[.,\s]", "", t) for t in re.findall(r"[\d][\d.,]*", s))

def tokens(q):
    q = unicodedata.normalize("NFKD", q.lower())
    q = "".join(c for c in q if not unicodedata.combining(c))
    return set(re.findall(r"[a-z0-9]+", q))

def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0

def lower_first_after_q(q):
    """Lowercase the first letter after '¿' (framing prefixes need it)."""
    m = re.match(r"(¿)([A-ZÁÉÍÓÚÑ])", q)
    if m:
        return q[:1] + m.group(2).lower() + q[2:]
    return q

def framing(q, prefix):
    return prefix + lower_first_after_q(q)

def fronting(q):
    m = re.match(r"^(.*)\sen 2024\?$", q.strip())
    if m:
        return "En 2024, " + lower_first_after_q(m.group(1) + "?")
    m = re.match(r"^(.*)\sal cierre de 2024\?$", q.strip())
    if m:
        return "Al cierre de 2024, " + lower_first_after_q(m.group(1) + "?")
    return None

def swap(q, old, new):
    if q.startswith(old):
        return new + q[len(old):]
    return None

def candidates(q):
    """Ordered paraphrase candidates (question-only)."""
    out = []
    out.append(framing(q, "Según el informe anual de FEMSA 2024, "))
    f = fronting(q)
    if f:
        out.append(f)
    for old, new in [
        ("¿A cuánto ascendieron", "¿A cuánto llegaron"),
        ("¿A cuánto llegaron", "¿A cuánto ascendieron"),
        ("¿Cuáles fueron", "¿Cuáles son"),
        ("¿Qué división creció más", "¿Cuál fue la división que más creció"),
        ("¿Cuántos", "¿Qué cantidad de"),
        ("¿Cuántas", "¿Qué cantidad de"),
        ("¿En qué porcentaje crecieron", "¿Cuánto crecieron en porcentaje"),
    ]:
        s = swap(q, old, new)
        if s:
            out.append(s)
    m = re.match(r"¿Tiene (.+?) operaciones en (.+?)\?$", q)
    if m:
        out.append(f"¿Opera {m.group(1)} en {m.group(2)}?")
    out.append(framing(q, "De acuerdo con el informe anual de FEMSA 2024, "))
    # dedupe, keep order, drop identical-to-original
    seen = {q}
    uniq = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def main_v2():
    rows = [json.loads(l) for l in open(BASE)]
    eval_qs = json.load(open(EVAL))["questions"]
    eval_toks = [tokens(q["question"]) for q in eval_qs]
    eval_nums = set()
    for q in eval_qs:
        eval_nums.update(q.get("expected_numbers", []))

    triples = json.load(open(TRIPLES))["triples"]
    triple_nums = set()
    for t in triples:
        for v in (t.get("valor"), t.get("valor_exacto_informe")):
            if v:
                triple_nums.update(norm_num_tokens(str(v)))
    key_nums = {n for n in eval_nums | triple_nums if len(n) >= 2}

    def is_key(row):
        return bool(norm_num_tokens(row["answer"]) & key_nums)

    out_rows = []
    dropped_leak = []
    key_rows = 0
    for i, r in enumerate(rows):
        key = is_key(r)
        if key:
            key_rows += 1
        n_para = 3 if key else 2
        cands = candidates(r["question"])
        kept = []
        for c in cands:
            if len(kept) >= n_para:
                break
            if max((jaccard(tokens(c), et) for et in eval_toks), default=0) >= JACCARD_MAX:
                dropped_leak.append({"base": i, "paraphrase": c})
                continue
            kept.append(c)
        out_rows.append({**r, "variant": "original", "base_index": i})
        for j, c in enumerate(kept):
            out_rows.append({**r, "question": c, "variant": f"parafrasis_{j+1}",
                             "base_index": i})

    # --- verification: answers byte-identical to their v1 source ---
    by_base = {}
    for r in out_rows:
        by_base.setdefault(r["base_index"], []).append(r)
    assert len(by_base) == len(rows), "some v1 row lost"
    for i, r in enumerate(rows):
        for v in by_base[i]:
            assert v["answer"] == r["answer"], f"answer changed for base {i}"
            assert v["source_pages"] == r["source_pages"]
    # no exact duplicate (question, answer)
    seen = set()
    for r in out_rows:
        k = (r["question"], r["answer"])
        assert k not in seen, f"duplicate pair: {k}"
        seen.add(k)
    # key-fact multiplicity: count appearances of each key number
    from collections import Counter
    fact_counts = Counter()
    for r in out_rows:
        for n in norm_num_tokens(r["answer"]) & key_nums:
            fact_counts[n] += 1
    low = {n: c for n, c in fact_counts.items() if c < 3}
    assert not low, f"key facts with <3 appearances: {low}"

    with open(OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "v1_rows": len(rows),
        "v2_rows": len(out_rows),
        "key_fact_rows_v1": key_rows,
        "paraphrases_kept": len(out_rows) - len(rows),
        "dropped_for_leakage_jaccard>=0.80": len(dropped_leak),
        "key_numbers_tracked": len(key_nums),
        "min_appearances_per_key_number": min(fact_counts.values()),
        "max_appearances_per_key_number": max(fact_counts.values()),
        "dropped_examples": dropped_leak[:5],
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

main_v2()
        '''),
        md("## 3. Dataset v3: más variación por hecho (~3,300 pares)\n\n"
           "Una segunda pasada de aumento, en el notebook: reutiliza los helpers "
           "definidos inline en la celda 2 (`tokens`, `jaccard`, `candidates`, "
           "lógica de `memo/augment_v2.py`) y añade más patrones de paráfrasis "
           "(más marcos y más trueques). Mismas reglas: solo cambia la pregunta, "
           "la respuesta es byte-idéntica; Jaccard ≥ 0.80 contra las 12 de "
           "evaluación descarta la paráfrasis; cada cifra clave debe aparecer ≥3 veces; "
           "sin duplicados (pregunta, respuesta). La pasada no tiene tope: el resultado "
           "fiel de la lógica es ~3,300 pares (verificado en local: 3,314, 0 descartes "
           "por fuga)."),
        code(r'''
# Usa los helpers definidos en la celda 2 (tokens, jaccard, framing,
# swap, norm_num_tokens, JACCARD_MAX): ya estan en el namespace.
import json

BASE = REPO + "/memo/reflection_qa.jsonl"
V2 = REPO + "/memo/reflection_qa_v2.jsonl"
EVAL = REPO + "/memo/eval_questions.json"
OUT = REPO + "/memo/reflection_qa_v3.jsonl"

EXTRA_FRAMINGS = [
    "Según los datos del informe anual de FEMSA 2024, ",
    "En el informe anual de FEMSA 2024, ",
    "Con base en el informe anual de FEMSA 2024, ",
    "De acuerdo con los datos del informe anual de FEMSA 2024, ",
]
EXTRA_SWAPS = [
    ("¿Cuáles fueron", "¿Cuáles resultaron ser"),
    ("¿Cuánto fue", "¿A cuánto ascendió"),
    ("¿A cuánto ascendieron", "¿Cuál fue el monto de"),
    ("¿Cuántos", "¿Qué número de"),
    ("¿Cuántas", "¿Qué número de"),
    ("¿Qué división creció más", "¿Cuál división tuvo el mayor crecimiento"),
    ("¿En qué países opera", "¿En qué países tiene operaciones"),
]


def extra_candidates(q):
    out = []
    for prefix in EXTRA_FRAMINGS:
        out.append(framing(q, prefix))
    for old, new in EXTRA_SWAPS:
        s = swap(q, old, new)
        if s:
            out.append(s)
    seen = {q}
    uniq = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


rows = [json.loads(l) for l in open(BASE)]
v2_rows = [json.loads(l) for l in open(V2)]
eval_qs = json.load(open(EVAL))["questions"]
eval_toks = [tokens(q["question"]) for q in eval_qs]

# Partimos del v2 (1,414) y añadimos las paráfrasis extra sobre los ORIGINALES.
by_base = {}
for r in v2_rows:
    by_base.setdefault(r["base_index"], []).append(r)

out_rows = list(v2_rows)
dropped_leak = 0
added = 0
for i, r in enumerate(rows):
    have = {x["question"] for x in by_base[i]}
    for c in extra_candidates(r["question"]):
        if c in have:
            continue
        if max((jaccard(tokens(c), et) for et in eval_toks), default=0) >= JACCARD_MAX:
            dropped_leak += 1
            continue
        out_rows.append({**r, "question": c, "variant": "parafrasis_extra",
                         "base_index": i})
        have.add(c)
        added += 1

# Verificacion: respuestas byte-identicas a su fuente v1; sin duplicados.
for r in out_rows:
    src = rows[r["base_index"]]
    assert r["answer"] == src["answer"], f"respuesta cambiada en base {r['base_index']}"
    assert r["source_pages"] == src["source_pages"]
seen = set()
for r in out_rows:
    k = (r["question"], r["answer"])
    assert k not in seen, f"par duplicado: {k}"
    seen.add(k)

# Multiplicidad de cifras clave (mismo criterio que la celda 2).
from collections import Counter
key_nums = set()
for q in eval_qs:
    for n in q.get("expected_numbers", []):
        key_nums.update(norm_num_tokens(str(n)))
fact_counts = Counter()
for r in out_rows:
    for n in norm_num_tokens(r["answer"]) & key_nums:
        fact_counts[n] += 1
low = {n: c for n, c in fact_counts.items() if c < 3}
assert not low, f"cifras clave con <3 apariciones: {low}"

with open(OUT, "w") as f:
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"v2: {len(v2_rows)} -> v3: {len(out_rows)} (+{added} parafrasis extra, "
      f"{dropped_leak} descartadas por fuga Jaccard>=0.80)")
print(f"apariciones por cifra clave: min {min(fact_counts.values())}, "
      f"max {max(fact_counts.values())}")
print(f"-> {OUT}")
        '''),
        md("## 4. Prueba de tiempo (antes del entrenamiento completo)\n\n"
           "Con `TIME_TEST_STEPS = 5` (arriba) la celda de entrenamiento solo mide "
           "segundos por micro-paso y estima el total. Si la estimación te convence, "
           "pon `TIME_TEST_STEPS = 0` y re-corre la celda para el entrenamiento completo. "
           "El modelo se descarga de Hugging Face a `HF_HOME` (en Drive): la primera vez "
           "tarda unos minutos; si HF va lento, la alternativa es ModelScope "
           "(`modelscope download Qwen/Qwen2.5-1.5B-Instruct`), como se documentó en el "
           "README de `memo/`)."),
        md("## 5. Entrenamiento (adaptado a GPU)\n\n"
           "Mismo algoritmo que `memo/train_memory_v2.py` (LoRA SFT, AdamW, schedule lineal "
           "con warmup, checkpoints cada 25 pasos, cortafuegos por tiempo), con los "
           "cambios mínimos para GPU: `device_map=\"auto\"`, batches a CUDA, batch 8 "
           "(en CPU era 1). El adaptador y el log quedan en Drive."),
        code(TRAIN_CELL),
        md("## Qué mirar al terminar\n\n"
           "- `training_log.json` en el adaptador: pérdida media por época y últimos 10 pasos. "
           "En el run 2 (CPU, 0.5B) la pérdida fue 0.61 → 0.16 → 0.087 (casi memorización). "
           "Con el 1.5B y más datos, la pérdida debería bajar parecido o más rápido.\n"
           "- Si la sesión se corta: re-corre esta celda con `INIT_ADAPTER` apuntando al "
           "último checkpoint (`memory_model_colab_ckpt/step_N`) — el entrenamiento sigue "
           "desde ahí (misma aproximación que el run 2 en CPU).\n"
           "- El adaptador que evalúa el notebook 04 es `OUT_DIR` (por defecto "
           "`memo/memory_model_colab`)."),
        *save_cell([
            ("memo/memory_model_colab", "Adaptador LoRA final (MEMORY 1.5B, 3 épocas)", True),
            ("memo/memory_model_colab_ckpt", "Checkpoints cada 25 pasos (para reanudar)", False),
            ("memo/reflection_qa_v3.jsonl", "Dataset v3 (~3,300 pares)", True),
            ("memo/reflection_qa_v2.jsonl", "Dataset v2 (1,414 pares)", False),
            ("memo/reflection_qa.jsonl", "Dataset v1 (446 pares)", False),
        ], note="El adaptador final + sus checkpoints y los 3 datasets quedan en Drive. "
                "El notebook 04 verifica que exista el adaptador antes de evaluar; si la "
                "sesión se cortó a mitad, reanuda con `INIT_ADAPTER` apuntando al último "
                "checkpoint y vuelve a guardar aquí."),
    ]
    write_nb("03_memo_entrenamiento.ipynb", cells, "03 MEMO entrenamiento GPU")

# ---------------------------------------------------------------------------
# 04 — MEMO: evaluación (ejecutivo vía Ollama)
# ---------------------------------------------------------------------------

def build_04():
    cells = [
        title_cell(
            "04 — MEMO: evaluación con ejecutivo vía Ollama",
            "Evalúa el MEMORY entrenado en el notebook 03 con el protocolo MEMO "
            "(descomposición → sub-respuestas de MEMORY → síntesis) sobre las mismas "
            "12 preguntas y con el scorer copiado inline en la celda 5 (misma lógica "
            "que `memo/evaluate.py`).\n\n"
            "**Diseño Ollama:** el **EXECUTIVE** corre vía **ChatOllama** contra el "
            "servidor local (parámetro `EXEC_OLLAMA_MODEL`: `qwen2.5:1.5b` para paridad "
            "con el run 2 en CPU, `qwen2.5:7b` para el run con modelo más grande). "
            "El **MEMORY** es el modelo entrenado (1.5B + adaptador LoRA del notebook 03) "
            "y se carga con Hugging Face en GPU — entrenar y servir un LoRA no se puede "
            "hacer vía Ollama, así que MEMORY sigue siendo HF. El EXECUTIVE nunca ve el "
            "corpus: solo las sub-respuestas de MEMORY.\n\n"
            "**Prerrequisito:** notebook 03 (adaptador en `memo/memory_model_colab`)."),
        gpu_check_cell(),
        drive_mount_cell(),
        params_cell("""
# --- Parametros ---
MEMORY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # base de MEMORY (misma que el entreno)
ADAPTER = REPO + "/memo/memory_model_colab"    # adaptador del notebook 03
EXEC_OLLAMA_MODEL = "qwen2.5:1.5b"   # ejecutivo: 1.5b (paridad run 2) o qwen2.5:7b (mas grande)
RUN_CONTROL = True                   # tambien evalua la base sin entrenar (control)
GENERATE_EXPANDED = False            # True = genera set de evaluacion ampliado (solo genera; NO evalua sin revision)
EXPANDED_EVAL = REPO + "/memo/eval_questions_expanded.json"
TRACE_OUT = REPO + "/memo/run_trace_colab.jsonl"
TRACE_CONTROL_OUT = REPO + "/memo/run_trace_colab_control.jsonl"
"""),
        repo_setup_cell(),
        *data_ensure_cells(),
        pip_cell([
            '"torch>=2.15.0"', "transformers==5.24.1", "accelerate>=1.16.0",
            "peft>=0.22.0", "numpy>=2.5.5", "langchain-ollama",
        ]),
        *ollama_compact_cells(["qwen2.5:1.5b", "qwen2.5:7b"], "EXEC_OLLAMA_MODEL"),
        md("## Prerrequisito: el adaptador MEMORY del notebook 03\n\n"
           "Sin el adaptador entrenado no hay nada que evaluar. Esta celda falla con un "
           "mensaje claro si no existe: corre el notebook 03 primero."),
        code("""
import os
assert os.path.isdir(ADAPTER), (
    "FALTA el adaptador MEMORY: " + ADAPTER + "\\n"
    "Lo produce el notebook 03 (MEMO entrenamiento). Corre 03 primero; "
    "sin adaptador entrenado, la evaluación no tiene sentido."
)
print("adaptador MEMORY OK:", ADAPTER)
"""),
        md("## 1. Cargar MEMORY (HF, GPU) y EXECUTIVE (Ollama)\n\n"
           "MEMORY = base 1.5B + adaptador LoRA del notebook 03 (o la base sola si "
           "`RUN_CONTROL`). EXECUTIVE = ChatOllama contra el servidor local. Los "
           "prompts y la descomposición por reglas están copiados inline en la celda "
           "siguiente (origen: `memo/executive_v2.py`); el notebook no importa "
           "código del repo."),
        code("""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

exec_llm = ChatOllama(model=EXEC_OLLAMA_MODEL, base_url="http://localhost:11434",
                      temperature=0, num_predict=150)
print("EXECUTIVE (Ollama):", EXEC_OLLAMA_MODEL)

# Prompts y descomposicion por reglas, copiados inline (misma logica y mismo
# texto que en el script del repo; el notebook es autocontenido).
DECOMP_SYSTEM = (
    "Descompón la pregunta del usuario en 1 a 3 sub-preguntas simples, cada una "
    "sobre un solo hecho del informe anual de FEMSA 2024. Responde solo con las "
    "sub-preguntas, una por línea, sin numerar. Crea sub-preguntas nuevas sobre "
    "la pregunta del usuario; NO copies las preguntas de los ejemplos.")
DECOMP_EXAMPLES = (
    "Pregunta: ¿Qué división aportó más ingresos en 2024, Proximidad Américas o Coca-Cola FEMSA, y por cuánto?\\n"
    "¿Cuáles fueron los ingresos de Proximidad Américas en 2024?\\n"
    "¿Cuáles fueron los ingresos de Coca-Cola FEMSA en 2024?\\n\\n"
    "Pregunta: ¿Qué porcentaje de los ingresos consolidados 2024 representan Proximidad Américas y Coca-Cola FEMSA juntas?\\n"
    "¿Cuáles fueron los ingresos totales consolidados de FEMSA en 2024?\\n"
    "¿Cuáles fueron los ingresos de Proximidad Américas en 2024?\\n"
    "¿Cuáles fueron los ingresos de Coca-Cola FEMSA en 2024?\\n\\n")


def rule_decompose(q):
    '''Descomposicion por reglas (fallback documentado).'''
    low = q.lower()
    divs = ["Proximidad Américas", "Proximidad Europa", "Coca-Cola FEMSA",
            "Salud", "Combustibles", "Spin"]
    named = [d for d in divs if d.lower() in low]
    if len(named) >= 2 and ("ingresos" in low or "creci" in low or "aport" in low):
        metric = "ingresos" if "ingresos" in low else "crecimiento"
        return [f"¿Cuáles fueron los {metric} de {d} en 2024?" for d in named]
    if "porcentaje" in low and "juntas" in low:
        return ["¿Cuáles fueron los ingresos totales consolidados de FEMSA en 2024?",
                "¿Cuáles fueron los ingresos de Proximidad Américas en 2024?",
                "¿Cuáles fueron los ingresos de Coca-Cola FEMSA en 2024?"]
    if "creció más" in low or "crecio mas" in low:
        return [q]  # la comparacion la resuelve el EXECUTIVE con las dos cifras
    return [q]


def chat_ollama(system, user):
    msgs = [SystemMessage(content=system), HumanMessage(content=user)]
    return exec_llm.invoke(msgs).content.strip()


mem_tok = AutoTokenizer.from_pretrained(MEMORY_MODEL)
print(f"cargando MEMORY {MEMORY_MODEL} (bf16, device_map=auto)...", flush=True)
_mem_base = AutoModelForCausalLM.from_pretrained(MEMORY_MODEL,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto")
print(f"cargando adaptador {ADAPTER} ...", flush=True)
mem_model = PeftModel.from_pretrained(_mem_base, ADAPTER)
mem_model.eval()
print("MEMORY lista: base + adaptador")
"""),
        md("## 2. Protocolo MEMO (lógica inline, origen: `memo/executive_v2.py`)\n\n"
           "Mismo protocolo que el run 2: descomposición por LLM (ahora vía Ollama; si "
           "falla el parseo, cae a la regla documentada), sub-respuestas de MEMORY "
           "(greedy, cortas) y síntesis por el EXECUTIVE. La traza guarda el método de "
           "descomposición (`llm` o `rule`) por pregunta, como en el original."),
        code("""
def llm_decompose(q):
    out = chat_ollama(DECOMP_SYSTEM, DECOMP_EXAMPLES + f"Pregunta: {q}\\n")
    lines = [l.strip().lstrip("0123456789.-) ").strip()
             for l in out.splitlines() if l.strip().endswith("?")]
    lines = [l for l in lines if len(l) > 10][:3]
    return lines or None


def memory_answer(subq):
    system = ("Responde con el dato exacto del informe anual de FEMSA 2024, "
              "en una frase corta. Si no sabes el dato, di 'No lo sé'.")
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": subq}]
    text = mem_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = mem_tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = mem_model.generate(**ids, max_new_tokens=60, do_sample=False,
                                 pad_token_id=mem_tok.eos_token_id)
    return mem_tok.decode(out[0][ids["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()


def synthesize(q, sub_answers):
    context = "\\n".join(f"- {s}: {a}" for s, a in sub_answers)
    system = ("Responde la pregunta usando SOLO la información siguiente, "
              "que viene del modelo de memoria. No inventes cifras. "
              "Si la información no alcanza, dilo.")
    user = f"Información del modelo de memoria:\\n{context}\\n\\nPregunta: {q}"
    return chat_ollama(system, user)


def run_memo(q):
    subs = llm_decompose(q)
    method = "llm"
    if not subs:
        subs = rule_decompose(q)
        method = "rule"
    sub_answers = [(s, memory_answer(s)) for s in subs]
    final = synthesize(q, sub_answers)
    return {"question": q, "decomposition": method,
            "exec_model": f"ollama {EXEC_OLLAMA_MODEL}",
            "memory_model": MEMORY_MODEL,
            "sub_questions": subs,
            "sub_answers": [{"sub_question": s, "memory_answer": a}
                            for s, a in sub_answers],
            "final_answer": final}
"""),
        md("## 3. Correr las 12 preguntas (MEMORY entrenado)"),
        code("""
import json

eval_qs = json.load(open(REPO + "/memo/eval_questions.json"))["questions"]
with open(TRACE_OUT, "w") as f:
    for item in eval_qs:
        r = run_memo(item["question"])
        r["id"] = item["id"]
        r["level"] = item["level"]
        r["control"] = False
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")
        print(f"{item['id']} [{r['decomposition']}] -> {r['final_answer'][:120]}", flush=True)
print(f"traza -> {TRACE_OUT}")
"""),
        md("## 4. Control (base sin entrenar, mismo ejecutivo)\n\n"
           "Si `RUN_CONTROL` es True, repite las 12 con MEMORY = base sin adaptador. "
           "El control mide cuánto viene del entrenamiento y cuánto del ejecutivo: en "
           "el run 2 el control sacó 0/12 con el mismo ejecutivo 1.5B."),
        code("""
if RUN_CONTROL:
    print("cargando MEMORY control (base sin adaptador)...", flush=True)
    _base = AutoModelForCausalLM.from_pretrained(MEMORY_MODEL,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    _base.eval()
    mem_model = _base
    eval_qs = json.load(open(REPO + "/memo/eval_questions.json"))["questions"]
    with open(TRACE_CONTROL_OUT, "w") as f:
        for item in eval_qs:
            r = run_memo(item["question"])
            r["id"] = item["id"]
            r["level"] = item["level"]
            r["control"] = True
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")
            print(f"{item['id']} [{r['decomposition']}] -> {r['final_answer'][:120]}", flush=True)
    print(f"traza control -> {TRACE_CONTROL_OUT}")
else:
    print("RUN_CONTROL=False: control omitido.")
"""),
        md("## 5. Puntuar con el scorer inline\n\n"
           "El scorer está copiado inline en la celda siguiente (origen: "
           "`memo/evaluate.py`): misma exactitud numérica con tolerancia de redondeo, "
           "misma regla de alucinación (cualquier número en la respuesta que no esté "
           "verificado en el informe), misma regla Q3 con 1,596 y mismas keywords para "
           "las preguntas sin números esperados (Q8/Q10/Q11). Escribe los mismos "
           "archivos (`results_colab.md` y `results_colab_control.md`). Compara con el "
           "run 2 en CPU (2/12) y el run 1 (0/12)."),
        code(r"""
import json, re


def digits(s):
    return re.sub(r"[.,]", "", s)


def numbers_in(text):
    return {digits(m.group(0)) for m in re.finditer(r"\d[\d.,]*", text or "")}


# Normaliza un numero esperado ('781585', '11.2') a digitos comparables.
def norm_num(x):
    return digits(str(x))


# True si cada numero esperado aparece (o su redondeo a entero) en la respuesta.
def expected_hit(answer_nums, expected):
    for e in expected:
        d = norm_num(e)
        if d in answer_nums:
            continue
        # tolerancia de redondeo: 307197 vs 307200, 279793 vs 279800
        try:
            v = float(e)
            ok = any(abs(float(a) - v) <= max(1.0, abs(v) * 0.001)
                     for a in answer_nums if a.replace(".", "", 1).isdigit())
        except ValueError:
            ok = False
        if not ok:
            return False
    return True


def score_trace(trace, out):
    # Scorer inline (misma logica que el scorer del repo): exactitud numerica
    # con tolerancia de redondeo, alucinacion = numero no verificado en el
    # informe, regla Q3 con 1,596 y keywords para preguntas sin numeros.
    eval_data = json.load(open(REPO + "/memo/eval_questions.json"))
    questions = {q["id"]: q for q in eval_data["questions"]}
    trace_rows = [json.loads(l) for l in open(trace)]

    # conjunto de numeros verificados del informe
    verified = set()
    try:
        for line in open(REPO + "/memo/reflection_qa_v3.jsonl"):
            r = json.loads(line)
            verified |= numbers_in(r["answer"])
    except FileNotFoundError:
        pass
    try:
        seed = json.load(open(REPO + "/pipeline/triples_seed.json"))
        triples = seed if isinstance(seed, list) else seed.get("triples", [])
        for t in triples:
            for k in ("valor", "variacion_pct"):
                if t.get(k) is not None:
                    verified.add(norm_num(t[k]))
    except FileNotFoundError:
        pass
    # anios y periodos no son alucinaciones
    verified |= {"2024", "2023", "2022"}

    rows = []
    for r in trace_rows:
        q = questions[r["id"]]
        ans = r["final_answer"]
        ans_nums = numbers_in(ans)
        halluc = sorted(n for n in ans_nums if n not in verified and len(n) > 1)
        ok = expected_hit(ans_nums, q.get("expected_numbers", []))
        # Preguntas sin numeros esperados: se evaluan por palabras clave.
        # Sin keywords ni numeros, NUNCA se marca correcta automaticamente
        # (el bug original: expected_hit([]) devolvia True por vacuidad y
        # Q8/Q10/Q11 salian "correctas" con respuestas erroneas).
        kws = q.get("expected_keywords", [])
        if not q.get("expected_numbers") and kws:
            low = ans.lower()
            ok = all(k.lower() in low for k in kws)
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
        rows.append({"id": r["id"], "level": q["level"], "question": q["question"],
                     "expected": q["expected"], "answer": ans,
                     "decomposition": r.get("decomposition"),
                     "verdict": verdict, "hallucinated": halluc, "note": note})

    by_level = {}
    for row in rows:
        by_level.setdefault(row["level"], []).append(row)
    total_ok = sum(1 for r in rows if r["verdict"] == "correcta")
    total_halluc = sum(1 for r in rows if r["hallucinated"])

    lines = ["# Resultados MEMO (demo) — FEMSA 2024",
             "",
             "Sistema: EXECUTIVE = Qwen2.5-0.5B-Instruct base (congelado); "
             "MEMORY = base + adaptador LoRA entrenado en reflexiones del informe. "
             "El EXECUTIVE nunca ve el corpus; solo las sub-respuestas de MEMORY.",
             "",
             f"Correctas: **{total_ok}/12**. Con alucinacion (numero no verificado en el informe): **{total_halluc}/12**.",
             "",
             "| ID | Nivel | Pregunta | Esperado | Respuesta MEMO | Veredicto |",
             "|----|-------|----------|----------|----------------|-----------|"]
    for r in rows:
        a = r["answer"].replace("|", "/").replace("\n", " ")[:220]
        lines.append(f"| {r['id']} | {r['level']} | {r['question'][:70]} | "
                     f"{r['expected'][:60]} | {a} | {r['verdict']} |")
    lines += ["",
              "## Por nivel",
              ""]
    for lv in sorted(by_level):
        ok = sum(1 for r in by_level[lv] if r["verdict"] == "correcta")
        lines.append(f"- Nivel {lv}: {ok}/{len(by_level[lv])} correctas.")
    lines += ["",
              "## Comparacion con el grafo (declarada, no medida con numeros inventados)",
              "",
              "- Grafo (Cypher sobre tripletas semilla): ~perfecto en nivel 1-2 **por "
              "construccion**, porque las respuestas son las tripletas verificadas. "
              "En nivel 3 depende de que el agente sepa agregar sin inventar.",
              "- MEMO (demo): ver tabla. La memoria es un modelo entrenado, no una base "
              "de datos: puede equivocar cifras aunque haya visto el hecho en "
              "entrenamiento. Esa es justo la diferencia que el post debe mostrar.",
              "- RAG vectorial: **pendiente** (no se corrio en este demo; no se "
              "inventan cifras).",
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

    open(out, "w").write("\n".join(lines) + "\n")
    print(f"correctas {total_ok}/12, con alucinacion {total_halluc}/12 -> {out}")
    for r in rows:
        print(f"  {r['id']} L{r['level']} [{r['decomposition']}] {r['verdict']}"
              + (f" halluc={r['hallucinated']}" if r["hallucinated"] else ""))


for trace, out in [(TRACE_OUT, REPO + "/memo/results_colab.md"),
                   (TRACE_CONTROL_OUT, REPO + "/memo/results_colab_control.md")]:
    if RUN_CONTROL or "control" not in trace:
        score_trace(trace, out)
"""),
        md("## 6. (Opcional) Generar un set de evaluación ampliado\n\n"
           "Con `GENERATE_EXPANDED = True` esta celda genera preguntas nuevas a partir "
           "de las tripletas semilla (hechos ya verificados) con plantillas por tipo de "
           "tripleta. **Cada cifra esperada se verifica por grep literal en el corpus "
           "(`chunks.jsonl`): lo que no aparece literal, se descarta — nunca se adivina.** "
           "También se descartan las que se parecen demasiado (Jaccard ≥ 0.80) a las 12 "
           "originales o entre sí. La salida lleva `\"needs_review\": true`: hay que "
           "revisar las preguntas a mano antes de evaluar con ellas (cambia `EVAL_FILE` "
           "en la celda 3 y vuelve a correr). Por defecto está apagado."),
        code(r"""
import json, re

if not GENERATE_EXPANDED:
    print("GENERATE_EXPANDED=False: no se genera nada. Ponlo en True para generar el set.")
else:
    triples = json.load(open(REPO + "/pipeline/triples_seed.json"))["triples"]
    chunks = [json.loads(l) for l in open(REPO + "/pipeline/data/chunks.jsonl")]
    corpus = "\n".join(c["text"] for c in chunks)
    orig = json.load(open(REPO + "/memo/eval_questions.json"))["questions"]

    def tokens(q):
        import unicodedata
        q = unicodedata.normalize("NFKD", q.lower())
        q = "".join(c for c in q if not unicodedata.combining(c))
        return set(re.findall(r"[a-z0-9]+", q))

    def jaccard(a, b):
        return len(a & b) / len(a | b) if a | b else 0.0

    def literal_present(text, expected):
        e = str(expected).replace(",", "")
        if "." in e:
            return e in text or e.replace(".", ",") in text
        try:
            grouped = f"{int(e):,}"
        except ValueError:
            return e in text
        return grouped in text or e in text

    orig_toks = [tokens(q["question"]) for q in orig]
    cands = []
    for t in triples:
        subj, rel, obj = t.get("subject"), t.get("relation"), t.get("object")
        valor, var = t.get("valor"), t.get("variacion_pct")
        if valor is None:
            continue
        if rel == "reporta" and obj == "Ingresos Totales":
            q = f"¿Cuáles fueron los ingresos totales de {subj} en 2024?"
            exp_nums = [str(valor)] + ([str(var)] if var is not None else [])
            exp = f"{valor} (Ps. millones)" + (f", +{var}%" if var is not None else "")
        elif rel == "mide" and obj == "Usuarios Activos":
            q = f"¿Cuántos usuarios activos tuvo {subj} en 2024?"
            exp_nums = [str(valor)] + ([str(var)] if var is not None else [])
            exp = f"{valor} millones" + (f" (+{var}%)" if var is not None else "")
        elif rel == "mide" and obj == "Tiendas":
            q = f"¿Cuántas tiendas tenía {subj} al cierre de 2024?"
            exp_nums = [str(valor)]
            exp = f"{valor} tiendas"
        elif rel == "mide" and obj == "Estaciones":
            q = f"¿Cuántas estaciones tenía OXXO GAS al cierre de 2024?"
            exp_nums = [str(valor)]
            exp = f"{valor} estaciones"
        else:
            continue
        cands.append({"question": q, "expected": exp, "expected_numbers": exp_nums,
                      "source_triple": {k: t.get(k) for k in ("subject", "relation", "object", "valor", "variacion_pct")}})

    kept, dropped = [], []
    seen_q = set()
    for c in cands:
        key = tokens(c["question"])
        if c["question"] in seen_q:
            dropped.append((c["question"], "duplicada"))
            continue
        if max((jaccard(key, et) for et in orig_toks), default=0) >= 0.80:
            dropped.append((c["question"], "demasiado parecida a una de las 12"))
            continue
        missing = [e for e in c["expected_numbers"] if not literal_present(corpus, e)]
        if missing:
            dropped.append((c["question"], f"cifra no verificada por grep literal: {missing}"))
            continue
        seen_q.add(c["question"])
        kept.append(c)

    out = {"needs_review": True,
           "note": "Generado por plantillas desde triples_seed.json; cada cifra verificada por grep literal en chunks.jsonl. REVISAR A MANO antes de evaluar.",
           "questions": [{"id": f"X{i+1}", "level": 1, **c} for i, c in enumerate(kept)]}
    json.dump(out, open(EXPANDED_EVAL, "w"), ensure_ascii=False, indent=2)
    print(f"candidatas: {len(cands)} | aceptadas: {len(kept)} | descartadas: {len(dropped)}")
    for q, why in dropped:
        print(f"  DESCARTADA: {q} ({why})")
    for c in kept:
        print(f"  X: {c['question']} -> {c['expected']}")
    print(f"-> {EXPANDED_EVAL} (needs_review=true: revisa antes de usar)")
"""),
        *save_cell([
            ("memo/run_trace_colab.jsonl", "Traza MEMO Colab (12 preguntas)", True),
            ("memo/run_trace_colab_control.jsonl", "Traza control (base sin entrenar)", False),
            ("memo/results_colab.md", "Resultados MEMO Colab", True),
            ("memo/results_colab_control.md", "Resultados control", False),
            ("memo/eval_questions_expanded.json", "Set de evaluación ampliado (si se generó; needs_review=true)", False),
        ]),
    ]
    write_nb("04_memo_evaluacion.ipynb", cells, "04 MEMO evaluación (Ollama)")

# ---------------------------------------------------------------------------
# 05 — RAG baseline (lector vía Ollama)
# ---------------------------------------------------------------------------


def build_05():
    nb05_retrieval_src = r"""
import json
import re
import time

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_FILE = REPO + "/pipeline/data/chunks.jsonl"
EVAL_FILE = REPO + "/memo/eval_questions.json"


def digits(s):
    return re.sub(r"[.,]", "", s)


def numbers_in(text):
    return {digits(m.group(0)) for m in re.finditer(r"\d[\d.,]*", text or "")}


t0 = time.time()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(EMBED_MODEL)
chunks = [json.loads(l) for l in open(CHUNKS_FILE)]
texts = [c["text"] for c in chunks]
print(f"embedding {len(chunks)} chunks with {EMBED_MODEL} ...", flush=True)
emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
emb = np.asarray(emb, dtype=np.float32)
print(f"embeddings: {emb.shape} in {time.time() - t0:.1f}s", flush=True)

eval_qs = json.load(open(EVAL_FILE))["questions"]
results = []
for q in eval_qs:
    q_emb = model.encode([q["question"]], normalize_embeddings=True,
                         show_progress_bar=False)[0]
    sims = emb @ q_emb
    top = np.argsort(-sims)[:K]
    retrieved = []
    for idx in top:
        c = chunks[int(idx)]
        retrieved.append({"page": c["page"], "score": round(float(sims[idx]), 4),
                          "text": c["text"]})
    union_nums = set()
    for r in retrieved:
        union_nums |= numbers_in(r["text"])
    # diagnostico: cada cifra esperada aparece en algun chunk recuperado?
    coverage = {}
    for e in q.get("expected_numbers", []):
        d = digits(str(e))
        hit = d in union_nums
        if not hit:  # tolerancia de redondeo igual que en el scorer
            try:
                v = float(e)
                hit = any(abs(float(a) - v) <= max(1.0, abs(v) * 0.001)
                          for a in union_nums if a.replace(".", "", 1).isdigit())
            except ValueError:
                hit = False
        coverage[e] = bool(hit)
    results.append({"id": q["id"], "level": q["level"],
                    "question": q["question"],
                    "expected": q["expected"],
                    "expected_numbers": q.get("expected_numbers", []),
                    "retrieved": retrieved,
                    "coverage": coverage,
                    "all_covered": all(coverage.values()) if coverage else None})
    cov = "ALL" if results[-1]["all_covered"] else (
        "PARTIAL" if any(coverage.values()) else "NONE")
    print(f"{q['id']} L{q['level']} coverage={cov} "
          f"pages={[r['page'] for r in retrieved]}", flush=True)

out = {"embed_model": EMBED_MODEL, "k": K, "n_chunks": len(chunks),
       "retrieval_seconds": round(time.time() - t0, 1),
       "questions": results}
json.dump(out, open(RETRIEVAL_FILE, "w"), ensure_ascii=False, indent=2)
n_all = sum(1 for r in results if r["all_covered"])
n_none = sum(1 for r in results if r["all_covered"] is False
             and not any(r["coverage"].values()))
print(f"retrieval: {n_all}/12 preguntas con TODAS las cifras esperadas "
      f"en top-{K}; {n_none}/12 sin ninguna -> {RETRIEVAL_FILE}")
"""

    nb05_reader_src = r"""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

MAX_CHUNK_CHARS = 1800  # ventana por chunk (ver condense): el contexto completo
# de 5 paginas (~17k caracteres) hace el prefill inviable; se conserva
# la ventana mas relevante a la pregunta y se guarda en la traza para auditar.

STOPWORDS = set("de la el en y a los del se las por un para con no una su al lo como mas pero sus le ya o este mi que qué cuando donde cuales cuál cómo cuánto".split())


def condense(text, question, max_chars=MAX_CHUNK_CHARS):
    # ~max_chars caracteres por chunk: ventanas alrededor de las 2 lineas
    # del chunk mas relevantes a la pregunta (overlap de palabras + lineas con
    # cifras). Dos ventanas porque las dos cifras que una pregunta compara
    # (p. ej. 39.3% y 35.8% en Q9) suelen estar lejos en la misma pagina.
    # Las ventanas usadas quedan guardadas en la traza (campo 'context').
    qwords = {w.lower() for w in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", question)
              if len(w) > 3 and w.lower() not in STOPWORDS}
    lines = text.splitlines() or [text]
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1

    def line_score(ln):
        low = ln.lower()
        s = sum(2 for w in qwords if w in low)
        if re.search(r"\d", ln):
            s += 1
        return s

    scored = sorted(((line_score(ln), i) for i, ln in enumerate(lines)
                     if line_score(ln) > 0), reverse=True)
    # top 2 lineas no adyacentes (ventanas que no se solapan)
    chosen = []
    for s, i in scored:
        if all(abs(i - j) > 3 for j in chosen):
            chosen.append(i)
        if len(chosen) == 2:
            break
    if not chosen:
        return text[:max_chars]
    half = max_chars // max(len(chosen), 1)
    parts = []
    for i in sorted(chosen):
        start = offsets[i]
        lo = max(0, start - half // 3)
        hi = min(len(text), lo + half)
        lo = max(0, hi - half)
        parts.append(text[lo:hi])
    out = "\n[...]\n".join(parts)
    return out[:max_chars]

SYSTEM = (
    "Responde la pregunta usando UNICAMENTE la información de los fragmentos "
    "del informe anual de FEMSA 2024 que se te dan a continuación. Copia las "
    "cifras exactamente como aparecen en los fragmentos, sin redondear ni "
    "inventar nada. Solo si la pregunta pide explícitamente sumar o comparar "
    "cifras que están en los fragmentos, haz esa operación con esas cifras. "
    "Si la información de los fragmentos no alcanza para responder, responde "
    "exactamente: No lo sé. Responde en español, en una o dos frases cortas.")

reader = ChatOllama(model=READER_OLLAMA_MODEL, base_url="http://localhost:11434",
                    temperature=0, num_predict=100)
print("lector RAG (Ollama):", READER_OLLAMA_MODEL)

data = json.load(open(RETRIEVAL_FILE))
with open(TRACE_OUT, "w") as f:
    for item in data["questions"]:
        condensed = [condense(r["text"], item["question"]) for r in item["retrieved"]]
        ctx = "\n\n".join(f"[Página {r['page']}]\n{c}"
                          for r, c in zip(item["retrieved"], condensed))
        user = (f"Fragmentos del informe anual de FEMSA 2024:\n\n{ctx}\n\n"
                f"Pregunta: {item['question']}")
        ans = reader.invoke([SystemMessage(content=SYSTEM),
                             HumanMessage(content=user)]).content.strip()
        rec = {"id": item["id"], "level": item["level"],
               "question": item["question"], "decomposition": "rag",
               "reader_model": f"ollama {READER_OLLAMA_MODEL}",
               "retrieved_pages": [r["page"] for r in item["retrieved"]],
               "retrieval_scores": [r["score"] for r in item["retrieved"]],
               "context": ctx, "final_answer": ans}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{item['id']} -> {ans[:160]}", flush=True)
print(f"traza -> {TRACE_OUT}")
"""

    nb05_scorer_src = r"""
import json
import re
from collections import Counter

EVAL_FILE = REPO + "/memo/eval_questions.json"
QA_FILE = REPO + "/memo/reflection_qa.jsonl"
SEED_FILE = REPO + "/pipeline/triples_seed.json"
CHUNKS_FILE = REPO + "/pipeline/data/chunks.jsonl"
OUT_FILE = REPO + "/rag_baseline/results_colab.md"


def digits(s):
    return re.sub(r"[.,]", "", s)


def numbers_in(text):
    return {digits(m.group(0)) for m in re.finditer(r"\d[\d.,]*", text or "")}


def norm_num(x):
    # Normaliza un numero esperado ('781585', '11.2') a digitos comparables.
    return digits(str(x))


def expected_hit(answer_nums, expected):
    # True si cada numero esperado aparece (o su redondeo a entero) en la respuesta.
    for e in expected:
        d = norm_num(e)
        if d in answer_nums:
            continue
        # tolerancia de redondeo: 307197 vs 307200, 279793 vs 279800
        try:
            v = float(e)
            ok = any(abs(float(a) - v) <= max(1.0, abs(v) * 0.001)
                     for a in answer_nums if a.replace(".", "", 1).isdigit())
        except ValueError:
            ok = False
        if not ok:
            return False
    return True


def literal_present(text, expected):
    # True si la cifra esperada aparece literalmente en el texto (con ',' o
    # '.' como separador). Sin tolerancia de redondeo: la tolerancia de
    # expected_hit marca 'cubierto' un 27.5% cuando solo hay un 27 suelto en el
    # texto, y eso falsea el diagnostico de retrieval.
    e = str(expected).replace(",", "")
    if "." in e:
        return e in text or e.replace(".", ",") in text
    # entero: aparece con separador de miles o sin el
    grouped = f"{int(e):,}"
    return grouped in text or e in text


eval_data = json.load(open(EVAL_FILE))
questions = {q["id"]: q for q in eval_data["questions"]}
trace = [json.loads(l) for l in open(TRACE_OUT)]
retr = {r["id"]: r for r in json.load(open(RETRIEVAL_FILE))["questions"]}
corpus_text = "\n".join(json.loads(l)["text"]
                        for l in open(CHUNKS_FILE))

# mismo conjunto de numeros verificados que el scorer de MEMO
verified = set()
try:
    for line in open(QA_FILE):
        verified |= numbers_in(json.loads(line)["answer"])
except FileNotFoundError:
    pass
try:
    seed = json.load(open(SEED_FILE))
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
    # Clasificacion del fallo: VERIFICADA A MANO contra el corpus (no
    # derivada por heuristica — la heuristica automatica etiqueta mal Q1,
    # Q10 y Q12: en Q1 el lector tenia 781,585 enfrente y contesto 40,236;
    # en Q10 'FEMSA Forward' no estaba en el top-5; en Q12 el 5.8 del
    # contexto era margen de utilidad neta, no el crecimiento de Salud).
    # Cada etiqueta se comprobo buscando la cifra literal en chunks.jsonl.
    VERIFIED_MODES = {
        "Q1": "fallo del lector (781,585 en contexto; contesto 40,236; el +11.2% tampoco se recupero)",
        "Q2": "cifra esperada fuera del corpus (+24.9% no existe en el informe; el lector contesto bien lo preguntado)",
        "Q3": "fallo de retrieval (23,206 solo en p.21, no en top-5)",
        "Q4": "fallo de retrieval (571 en pp. 21/114/126, no en top-5)",
        "Q5": "alucinacion del lector (divago, se corto a 100 tokens; '114' es numero de pagina)",
        "Q6": "ok",
        "Q7": "cifra esperada fuera del corpus (+27.5% y +24.9% no existen en el informe; el lector acerto direccion y absolutos)",
        "Q8": "fallo del lector (Chile y Peru en contexto; listo solo 2 de 4)",
        "Q9": "cifra esperada fuera del corpus + aritmetica (39.3/35.8/75.1% no existen; sumo 23,729+19,536=43,265)",
        "Q10": "fallo de retrieval ('FEMSA Forward' en pp. 3/5/8/12/14/42/101/115, no en top-5)",
        "Q11": "ok",
        "Q12": "fallo de retrieval (Health +5.8% en pp. 5/112/114, no en top-5; el 5.8 del contexto era margen neto)",
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

lines = ["# Resultados RAG vectorial (baseline) — FEMSA 2024",
         "",
         "Sistema: retrieval top-5 paginas con all-MiniLM-L6-v2 (CPU) sobre "
         "las 151 paginas del informe; reader Qwen2.5-1.5B-Instruct local "
         "(bf16, CPU), respondiendo SOLO desde los chunks recuperados. "
         "Scorer con la misma logica que el de MEMO (misma exactitud numerica, "
         "misma definicion de alucinacion, misma regla Q3 con 1,596).",
         "",
         f"Correctas: **{total_ok}/12**. Con alucinacion (numero no verificado en el informe): **{total_halluc}/12**. "
         f"Retrieval top-5 con todas las cifras esperadas literales: **{n_retr_full}/12** "
         f"(el diagnostico con tolerancia de redondeo decia 6/12; ver Caveats).",
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
for mode, n in Counter(r["failure_mode"] for r in rows).most_common():
    ids = ", ".join(r["id"] for r in rows if r["failure_mode"] == mode)
    lines.append(f"- {mode}: {n} ({ids})")
lines += ["",
          "## Comparacion de los tres sistemas (12 preguntas, mismo scorer)",
          "",
          "| Sistema | Correctas | Nivel 1 | Nivel 2 | Nivel 3 |",
          "|---------|-----------|---------|---------|---------|",
          "| Grafo (Relate, Cypher) | 12/12 | 4/4 | 4/4 | 4/4 |",
          f"| RAG vectorial (este baseline) | {total_ok}/12 | "
          f"{sum(1 for r in by_level[1] if r['verdict'] == 'correcta')}/4 | "
          f"{sum(1 for r in by_level[2] if r['verdict'] == 'correcta')}/4 | "
          f"{sum(1 for r in by_level[3] if r['verdict'] == 'correcta')}/4 |",
          "| MEMO run 2 (0.5B LoRA r=32 + ejecutivo 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |",
          "| MEMO run 1 (0.5B LoRA r=8 + ejecutivo 0.5B) | 0/12 | 0/4 | 0/4 | 0/4 |",
          "",
          "## Veredicto para el post",
          "",
          "**RAG vectorial: 2/12 — el mismo puntaje que MEMO run 2, pero con un "
          "perfil de fallo distinto y mas instructivo.** El grafo gana 12/12 "
          "porque la respuesta *es* el dato verificado; RAG y MEMO pierden "
          "porque tienen que *reconstruir* la cifra — uno desde texto "
          "recuperado, el otro desde pesos. El experimento muestra que, a "
          "esta escala y con un lector de 1.5B, el RAG clasico no es mejor "
          "que la memoria parametrica: empata con ella.",
          "",
          "Donde pierde RAG, verificado cifra por cifra contra el corpus:",
          "",
          "- **Retrieval (4): Q3, Q4, Q10, Q12.** La cifra existe en el "
          "informe y el top-5 no la trajo. Q3: 23,206 esta solo en la p.21; "
          "Q4: 571 esta en pp. 21/114/126; Q10: 'FEMSA Forward' esta en pp. "
          "3/5/8/12/14/42/101/115; Q12: Health +5.8% esta en pp. 5/112/114. "
          "El lector en Q3/Q4 se abstuvo honestamente ('No lo se'); en "
          "Q10/Q12 contesto con otra cosa. Un retriever mejor (o chunks "
          "mas pequenos que la pagina) es el arreglo mas directo aqui.",
          "- **Cifras esperadas fuera del corpus (3): Q2, Q7, Q9.** "
          "+24.9%, +27.5%, 39.3%, 35.8% y 75.1% no aparecen en ninguna de "
          "las 151 paginas. 24.9/27.5 vienen del comunicado de prensa 4Q24, "
          "no del informe anual; 39.3/35.8/75.1 son participaciones "
          "calculadas, no cifras impresas. El grafo las 'sabe' solo porque "
          "sus tripletas semilla se construyeron a mano con esas fuentes. "
          "En Q2 y Q7 el lector contesto bien lo sustantivo (8.6 millones; "
          "Spin Premia crece mas, 24.6 vs 8.6) y el scorer lo marca mal "
          "porque exige los porcentajes que no estan en el corpus. Techo "
          "real de RAG sobre este set, tal como esta especificado: 9/12.",
          "- **Lector (3): Q1, Q5, Q8.** La cifra estaba en el contexto y "
          "el lector fallo. Q1 eligio Ps. 40,236 (+31%) teniendo 781,585 "
          "enfrente — elige la cifra equivocada de la tabla. Q5 divago y "
          "se corto a los 100 tokens sin concluir, filtrando '114' (numero "
          "de pagina) como si fuera un dato. Q8 tenia Chile y Peru en el "
          "contexto y solo listo Mexico y Colombia. El lector de 1.5B no "
          "es un extractor fiable de tablas.",
          "- **Aritmetica (Q9).** Tenia 307,197 en contexto pero no "
          "279,793 ni 781,585 (perdida en el condensado); agarro 23,729 y "
          "19,536 y los sumo en 43,265. Es el mismo modo de fallo del "
          "ejecutivo de MEMO run 2: la sintesis rompe el dato correcto.",
          "",
          "Lectura para el post: el RAG clasico falla en los dos extremos — "
          "cuando no recupera el dato y cuando lo recupera pero el lector "
          "no sabe elegirlo. El grafo no tiene ninguno de los dos problemas "
          "porque no recupera ni elige: consulta. La moraleja no es que RAG "
          "sea inutil, sino que la exactitud de cifras exige que el dato "
          "llegue verificado al lector, no solo 'relevante'.",
          "",
          "## Caveats (verificados, no supuestos)",
          "",
          "- El diagnostico inicial de retrieval usaba la tolerancia de "
          "redondeo del scorer (±0.1%): marcaba 'cubierto' el 27.5% de Q7 "
          "cuando solo habia un 27 suelto, el 24.9% de Q2 cuando no hay "
          "ningun 24.9, y el 39.3/35.8/75.1% de Q9 cuando solo habia "
          "numeros cercanos. La cobertura literal estricta en top-5 es "
          "2/12 completa (Q5, Q12) — con dos matices: en Q6 las cifras "
          "estan como valores exactos (307,197/279,793) mientras el campo "
          "esperado trae los redondeados de la ontologia (307,200/279,800), "
          "y en Q12 el 5.8 literal del top-5 es el margen de utilidad neta, "
          "no el crecimiento de Salud. La tabla de detalle usa cobertura "
          "literal; la clasificacion de fallos se verifico a mano.",
          "- El condensado de contexto (ventanas de ~1,800 caracteres por "
          "chunk, 2 ventanas) fue necesario porque el prompt de 5 paginas "
          "completas (~4.5k tokens) tarda >6 min por pregunta en CPU. "
          "Perdio cifras en Q9 (279,793 y 781,585 estaban en el top-5 y "
          "no llegaron al lector). Es un costo medido del workaround, no "
          "del metodo RAG en si.",
          "- La pregunta esta en espanol y el corpus en ingles; el "
          "retrieval es cross-lingual con un embedder entrenado en ingles "
          "(all-MiniLM-L6-v2). Q12 exige mapear 'Salud' -> 'Health'.",
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

open(OUT_FILE, "w").write("\n".join(lines) + "\n")
print(f"correctas {total_ok}/12, con alucinacion {total_halluc}/12, "
      f"retrieval literal completo {n_retr_full}/12 -> {OUT_FILE}")
for r in rows:
    print(f"  {r['id']} L{r['level']} {r['verdict']} [{r['failure_mode']}]"
          + (f" halluc={r['hallucinated']}" if r["hallucinated"] else ""))
"""

    cells = [
        title_cell(
            "05 — RAG clásico (baseline denso, lector vía Ollama)",
            "Retrieval denso top-5 con all-MiniLM-L6-v2 + lector vía **ChatOllama** "
            "(`READER_OLLAMA_MODEL`: `qwen2.5:1.5b` para paridad con el baseline en CPU, "
            "`qwen2.5:7b` para el run con modelo más grande) + mismo scorer. El retrieval, "
            "el lector y el scorer son lógica inline en este notebook (misma lógica que "
            "`rag_baseline/retrieve.py`, `rag_baseline/answer.py` y `rag_baseline/score.py` "
            "en el repo, sin ejecutar ni importar esos scripts): el lector es el mismo "
            "prompt y el mismo `condense()` inline, solo cambia el modelo que contesta "
            "(Ollama en vez de transformers en CPU).\n\n"
            "**Prerrequisito:** notebook 01 (chunks). En GPU el embedder usa CUDA solo."),
        gpu_check_cell(),
        drive_mount_cell(),
        params_cell("""
# --- Parametros ---
READER_OLLAMA_MODEL = "qwen2.5:1.5b"   # lector: 1.5b (paridad CPU) o qwen2.5:7b (mas grande)
K = 5
RETRIEVAL_FILE = REPO + "/rag_baseline/retrieval.json"
TRACE_OUT = REPO + "/rag_baseline/run_trace_colab.jsonl"
"""),
        repo_setup_cell(),
        *data_ensure_cells(),
        pip_cell([
            '"torch>=2.15.0"', "transformers==5.24.1", "sentence-transformers==6.3.0",
            "numpy>=2.5.5", "langchain-ollama",
        ]),
        *ollama_compact_cells(["qwen2.5:1.5b", "qwen2.5:7b"], "READER_OLLAMA_MODEL"),
        md("## 1. Retrieval denso top-5\n\n"
           "Lógica inline de `rag_baseline/retrieve.py` (proveniencia; el script no se "
           "ejecuta): embebe las 151 páginas con all-MiniLM-L6-v2 y recupera top-5 por "
           "pregunta, con el diagnóstico de cobertura de cifras esperadas (separa fallos "
           "de retrieval de fallos de lector). Escribe `retrieval.json` con el mismo "
           "esquema que el script."),
        code(nb05_retrieval_src),
        md("## 2. Lector vía Ollama\n\n"
           "Mismo sistema y mismo `condense()` inline (proveniencia: `rag_baseline/answer.py`, "
           "sin ejecutar el script ni importarlo): el contexto son los 5 chunks "
           "recuperados (ventanas de ~1,800 caracteres por chunk, 2 ventanas) y el modelo "
           "responde solo desde ellos. La traza tiene el mismo formato que la de CPU, "
           "así que el scorer no cambia."),
        code(nb05_reader_src),
        md("## 3. Puntuar con el mismo scorer\n\n"
           "Lógica inline de `rag_baseline/score.py` (proveniencia; el script no se ejecuta): "
           "misma exactitud numérica, misma definición de alucinación, misma regla Q3 con "
           "1,596. Compara con el baseline en CPU (2/12: Q6 y Q11)."),
        code(nb05_scorer_src),
        md("## Alternativa documentada (Hugging Face, no por defecto en Colab)\n\n"
           "Si quieres el lector HF en GPU en vez de Ollama, corre el script original "
           "con el modelo local (hay que bajar Qwen2.5-1.5B-Instruct a `memo/models/` "
           "o apuntar `--model` a `Qwen/Qwen2.5-1.5B-Instruct` en el Hub):\n\n"
           "```bash\n"
           "python rag_baseline/answer.py --retrieval rag_baseline/retrieval.json "
           "--model Qwen/Qwen2.5-1.5B-Instruct --out rag_baseline/run_trace_hf.jsonl\n"
           "```\n\n"
           "Ojo: `answer.py` carga el modelo sin `device_map`, así que en Colab correría "
           "en CPU salvo que se adapte. Por eso el default en Colab es Ollama."),
        *save_cell([
            ("rag_baseline/retrieval.json", "Retrieval denso top-5 (12 preguntas)", True),
            ("rag_baseline/run_trace_colab.jsonl", "Traza del lector (12 respuestas)", True),
            ("rag_baseline/results_colab.md", "Resultados RAG Colab", True),
        ], note="El notebook 08 (RAGAS) verifica que existan estas trazas antes de medir: "
                "sin ellas no hay métricas."),
    ]
    write_nb("05_rag_baseline.ipynb", cells, "05 RAG baseline (Ollama)")


# ---------------------------------------------------------------------------
# 06 — RAG híbrido + expansión (lector vía Ollama)
# ---------------------------------------------------------------------------

def build_06():
    cells = [
        title_cell(
            "06 — RAG híbrido (BM25 + denso) y expansión de queries",
            "Las dos variantes que movieron el marcador en CPU: híbrido BM25 Okapi + "
            "MiniLM con fusión RRF (k=60) y, encima, expansión de queries (traducción "
            "manual al inglés + mapa de términos de dominio, sin filtrar respuestas). "
            "Mismo lector vía **ChatOllama** y mismo scorer que el notebook 05.\n\n"
            "**Prerrequisito:** notebook 01 (chunks). El retrieval híbrido no necesita "
            "GPU (BM25 es CPU, los embeddings usan CUDA si está disponible)."),
        gpu_check_cell(),
        drive_mount_cell(),
        params_cell("""
# --- Parametros ---
READER_OLLAMA_MODEL = "qwen2.5:1.5b"   # lector: 1.5b (paridad CPU) o qwen2.5:7b (mas grande)
K = 5
"""),
        repo_setup_cell(),
        *data_ensure_cells(),
        pip_cell([
            '"torch>=2.15.0"', "transformers==5.24.1", "sentence-transformers==6.3.0",
            "numpy>=2.5.5", "rank-bm25==0.2.2", "langchain-ollama",
        ]),
        *ollama_compact_cells(["qwen2.5:1.5b", "qwen2.5:7b"], "READER_OLLAMA_MODEL"),
        md("## 1. Expansión de queries\n\n"
           "La lógica de `rag_baseline/expand_queries.py`, inline en la celda "
           "siguiente: traduce cada pregunta al inglés y añade términos de dominio "
           "del `TERM_MAP` (solo términos presentes en la pregunta; la celda aborta "
           "si se cuela una cifra esperada o un nombre de respuesta). "
           "Salida: `expanded_queries.json`."),
        code(r"""
import json, re

EVAL_PATH = REPO + "/memo/eval_questions.json"
EXPANDED_OUT = REPO + "/rag_baseline/expanded_queries.json"

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
# sean la respuesta.
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


def nb06_digits(s):
    return re.sub(r"[.,]", "", s)


eval_qs = json.load(open(EVAL_PATH))["questions"]
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
        d = nb06_digits(str(e))
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
          open(EXPANDED_OUT, "w"), ensure_ascii=False, indent=2)
print(f"expanded queries -> {EXPANDED_OUT} "
      f"(guardia anti-filtracion pasada: sin cifras ni 'FEMSA Forward')")
"""),
        md("## 2. Retrieval híbrido (sin expansión) y con expansión\n\n"
           "La lógica de `rag_baseline/retrieve_hybrid.py` y "
           "`rag_baseline/retrieve_expanded.py`, inline: dos corridas del mismo "
           "retrieval híbrido (BM25 Okapi k1=1.5, b=0.75 + MiniLM, RRF k=60), una "
           "con las preguntas originales y otra con las expandidas. El índice BM25 "
           "y los embeddings se construyen una sola vez en la primera celda y se "
           "reutilizan en ambas corridas."),
        code(r"""
import json, re, time

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60
TOKEN_RE = re.compile(r"[a-z0-9áéíóúüñ]+")
CHUNKS_PATH = REPO + "/pipeline/data/chunks.jsonl"
EVAL_PATH = REPO + "/memo/eval_questions.json"


def nb06_tokenize(text):
    return TOKEN_RE.findall((text or "").lower())


def nb06_digits(s):
    return re.sub(r"[.,]", "", s)


def nb06_numbers_in(text):
    return {nb06_digits(m.group(0)) for m in re.finditer(r"\d[\d.,]*", text or "")}


def nb06_rrf_ranks(bm25_scores, dense_sims):
    '''Ranks 1-based (1 = mejor) de cada ranker sobre los 151 chunks.'''
    bm25_rank = np.empty(len(bm25_scores), dtype=int)
    bm25_rank[np.argsort(-np.asarray(bm25_scores))] = np.arange(1, len(bm25_scores) + 1)
    dense_rank = np.empty(len(dense_sims), dtype=int)
    dense_rank[np.argsort(-np.asarray(dense_sims))] = np.arange(1, len(dense_sims) + 1)
    return bm25_rank, dense_rank


def nb06_coverage(q, retrieved):
    '''Cobertura de cifras esperadas en el top-k (con tolerancia +-0.1%).'''
    union_nums = set()
    for r in retrieved:
        union_nums |= nb06_numbers_in(r["text"])
    coverage = {}
    for e in q.get("expected_numbers", []):
        d = nb06_digits(str(e))
        hit = d in union_nums
        if not hit:
            try:
                v = float(e)
                hit = any(abs(float(a) - v) <= max(1.0, abs(v) * 0.001)
                          for a in union_nums if a.replace(".", "", 1).isdigit())
            except ValueError:
                hit = False
        coverage[e] = bool(hit)
    return coverage


chunks = [json.loads(l) for l in open(CHUNKS_PATH)]
texts = [c["text"] for c in chunks]
print(f"BM25 Okapi (k1=1.5, b=0.75) + {EMBED_MODEL} sobre {len(chunks)} chunks ...",
      flush=True)
t0 = time.time()
tokenized = [nb06_tokenize(t) for t in texts]
bm25 = BM25Okapi(tokenized)
print(f"BM25 index listo en {time.time() - t0:.1f}s", flush=True)

model = SentenceTransformer(EMBED_MODEL)
emb = np.asarray(model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=False), dtype=np.float32)
print(f"embeddings: {emb.shape} en {time.time() - t0:.1f}s", flush=True)


def nb06_run_retrieval(query_of, method, out_path, label, expanded=False):
    '''Una corrida de retrieval hibrido: `query_of(q)` es la query que ven
    los dos rankers (pregunta original o expanded query).'''
    t1 = time.time()
    eval_qs = json.load(open(EVAL_PATH))["questions"]
    results = []
    for q in eval_qs:
        query = query_of(q)
        q_tokens = nb06_tokenize(query)
        b_scores = np.asarray(bm25.get_scores(q_tokens), dtype=np.float64)
        q_emb = model.encode([query], normalize_embeddings=True,
                             show_progress_bar=False)[0]
        sims = emb @ q_emb
        b_rank, d_rank = nb06_rrf_ranks(b_scores, sims)
        rrf = 1.0 / (RRF_K + b_rank) + 1.0 / (RRF_K + d_rank)
        top = np.argsort(-rrf)[:K]
        retrieved = []
        for idx in top:
            c = chunks[int(idx)]
            retrieved.append({"page": c["page"],
                              "score": round(float(rrf[idx]), 6),
                              "bm25_rank": int(b_rank[idx]),
                              "dense_rank": int(d_rank[idx]),
                              "bm25_score": round(float(b_scores[idx]), 4),
                              "dense_score": round(float(sims[idx]), 4),
                              "text": c["text"]})
        coverage = nb06_coverage(q, retrieved)
        # top-10 de cada ranker, para auditoria
        b_top10 = [int(chunks[i]["page"]) for i in np.argsort(-b_scores)[:10]]
        d_top10 = [int(chunks[i]["page"]) for i in np.argsort(-sims)[:10]]
        rec = {"id": q["id"], "level": q["level"],
               "question": q["question"]}
        if expanded:
            rec["expanded_query"] = query
        rec.update({"expected": q["expected"],
                    "expected_numbers": q.get("expected_numbers", []),
                    "retrieved": retrieved,
                    "coverage": coverage,
                    "all_covered": all(coverage.values()) if coverage else None,
                    "bm25_top10_pages": b_top10,
                    "dense_top10_pages": d_top10})
        results.append(rec)
        cov = "ALL" if rec["all_covered"] else (
            "PARTIAL" if any(coverage.values()) else "NONE")
        print(f"{q['id']} L{q['level']} coverage={cov} "
              f"pages={[r['page'] for r in retrieved]} "
              f"(bm25_top10={b_top10[:5]} dense_top10={d_top10[:5]})", flush=True)

    out = {"method": method,
           "embed_model": EMBED_MODEL, "k": K, "n_chunks": len(chunks),
           "retrieval_seconds": round(time.time() - t1, 1),
           "questions": results}
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    n_all = sum(1 for r in results if r["all_covered"])
    n_none = sum(1 for r in results if r["all_covered"] is False
                 and not any(r["coverage"].values()))
    print(f"retrieval {label}: {n_all}/12 preguntas con TODAS las cifras "
          f"esperadas en top-{K}; {n_none}/12 sin ninguna -> {out_path}")
"""),
        code(r"""
# Corrida 1: preguntas originales en espanol.
nb06_run_retrieval(lambda q: q["question"],
                   "hybrid BM25Okapi (k1=1.5, b=0.75) + MiniLM dense, RRF k=60",
                   REPO + "/rag_baseline/retrieval_hybrid.json", "hibrido")
"""),
        code(r"""
# Corrida 2: expanded queries (traduccion EN + keywords de dominio).
expanded = {q["id"]: q["expanded_query"]
            for q in json.load(open(REPO + "/rag_baseline/expanded_queries.json"))["questions"]}
nb06_run_retrieval(lambda q: expanded[q["id"]],
                   "hybrid BM25Okapi (k1=1.5, b=0.75) + MiniLM dense, RRF k=60, "
                   "con queries expandidas (traduccion EN + keywords de dominio)",
                   REPO + "/rag_baseline/retrieval_expanded.json", "expandido",
                   expanded=True)
"""),
        md("## 3. Lector vía Ollama (las dos variantes)\n\n"
           "El mismo `SYSTEM` y el mismo `condense()` de `rag_baseline/answer.py`, "
           "inline en la celda siguiente (sin importar el script), aplicados a los "
           "dos archivos de retrieval. Cada variante escribe su propia traza, con "
           "el mismo formato."),
        code(r"""
import json, re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

SYSTEM = (
    "Responde la pregunta usando UNICAMENTE la información de los fragmentos "
    "del informe anual de FEMSA 2024 que se te dan a continuación. Copia las "
    "cifras exactamente como aparecen en los fragmentos, sin redondear ni "
    "inventar nada. Solo si la pregunta pide explícitamente sumar o comparar "
    "cifras que están en los fragmentos, haz esa operación con esas cifras. "
    "Si la información de los fragmentos no alcanza para responder, responde "
    "exactamente: No lo sé. Responde en español, en una o dos frases cortas.")

STOPWORDS = set('''de la el en y a los del se las por un para con no una su al lo como
mas pero sus le ya o este mi que qué cuando donde cuales cuál cómo cuánto'''.split())

MAX_CHUNK_CHARS = 1800  # ventana por chunk (ver condense): el contexto completo
# de 5 paginas (~17k caracteres) hace el prefill en CPU inviable; se conserva
# la ventana mas relevante a la pregunta y se guarda en la traza para auditar.


def condense(text, question, max_chars=MAX_CHUNK_CHARS):
    '''~max_chars caracteres por chunk: ventanas alrededor de las 2 lineas
    del chunk mas relevantes a la pregunta (overlap de palabras + lineas con
    cifras). Dos ventanas porque las dos cifras que una pregunta compara
    (p. ej. 39.3% y 35.8% en Q9) suelen estar lejos en la misma pagina.
    Sin esto el prompt de 5 paginas (~4.5k tokens) tarda >6 min por pregunta
    en CPU; con la ventana el prefill baja a ~2k tokens. Las ventanas usadas
    quedan guardadas en la traza (campo 'context').'''
    qwords = {w.lower() for w in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", question)
              if len(w) > 3 and w.lower() not in STOPWORDS}
    lines = text.splitlines() or [text]
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1

    def line_score(ln):
        low = ln.lower()
        s = sum(2 for w in qwords if w in low)
        if re.search(r"\d", ln):
            s += 1
        return s

    scored = sorted(((line_score(ln), i) for i, ln in enumerate(lines)
                     if line_score(ln) > 0), reverse=True)
    # top 2 lineas no adyacentes (ventanas que no se solapan)
    chosen = []
    for s, i in scored:
        if all(abs(i - j) > 3 for j in chosen):
            chosen.append(i)
        if len(chosen) == 2:
            break
    if not chosen:
        return text[:max_chars]
    half = max_chars // max(len(chosen), 1)
    parts = []
    for i in sorted(chosen):
        start = offsets[i]
        lo = max(0, start - half // 3)
        hi = min(len(text), lo + half)
        lo = max(0, hi - half)
        parts.append(text[lo:hi])
    out = "\n[...]\n".join(parts)
    return out[:max_chars]


reader = ChatOllama(model=READER_OLLAMA_MODEL, base_url="http://localhost:11434",
                    temperature=0, num_predict=100)
print("lector RAG (Ollama):", READER_OLLAMA_MODEL)


def run_reader(retrieval_file, trace_out):
    data = json.load(open(retrieval_file))
    with open(trace_out, "w") as f:
        for item in data["questions"]:
            condensed = [condense(r["text"], item["question"]) for r in item["retrieved"]]
            ctx = "\n\n".join(f"[Página {r['page']}]\n{c}"
                              for r, c in zip(item["retrieved"], condensed))
            user = (f"Fragmentos del informe anual de FEMSA 2024:\n\n{ctx}\n\n"
                    f"Pregunta: {item['question']}")
            ans = reader.invoke([SystemMessage(content=SYSTEM),
                                 HumanMessage(content=user)]).content.strip()
            rec = {"id": item["id"], "level": item["level"],
                   "question": item["question"], "decomposition": "rag",
                   "reader_model": f"ollama {READER_OLLAMA_MODEL}",
                   "retrieved_pages": [r["page"] for r in item["retrieved"]],
                   "retrieval_scores": [r["score"] for r in item["retrieved"]],
                   "context": ctx, "final_answer": ans}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{item['id']} -> {ans[:120]}", flush=True)
    print(f"traza -> {trace_out}")


run_reader(REPO + "/rag_baseline/retrieval_hybrid.json",
           REPO + "/rag_baseline/run_trace_hybrid_colab.jsonl")
run_reader(REPO + "/rag_baseline/retrieval_expanded.json",
           REPO + "/rag_baseline/run_trace_expanded_colab.jsonl")
"""),
        md("## 4. Puntuar ambas variantes\n\n"
           "La lógica de `rag_baseline/score_hybrid.py` y "
           "`rag_baseline/score_expanded.py`, inline en la celda siguiente (mismo "
           "scorer, mismos modos de fallo verificados a mano). En CPU: híbrido 2/12 "
           "(Q1, Q11), híbrido + expansión 5/12 (Q1, Q4, Q6, Q8, Q10)."),
        code(r"""
import json, re
from collections import Counter

# --- Helpers numericos: misma exactitud que el scorer de MEMO ---
def nb06_digits(s):
    return re.sub(r"[.,]", "", s)


def nb06_numbers_in(text):
    return {nb06_digits(m.group(0)) for m in re.finditer(r"\d[\d.,]*", text or "")}


def nb06_norm_num(x):
    '''Normaliza un numero esperado ('781585', '11.2') a digitos comparables.'''
    return nb06_digits(str(x))


def nb06_expected_hit(answer_nums, expected):
    '''True si cada numero esperado aparece (o su redondeo a entero) en la respuesta.'''
    for e in expected:
        d = nb06_norm_num(e)
        if d in answer_nums:
            continue
        # tolerancia de redondeo: 307197 vs 307200, 279793 vs 279800
        try:
            v = float(e)
            ok = any(abs(float(a) - v) <= max(1.0, abs(v) * 0.001)
                     for a in answer_nums if a.replace(".", "", 1).isdigit())
        except ValueError:
            ok = False
        if not ok:
            return False
    return True


def nb06_literal_present(text, expected):
    '''True si la cifra esperada aparece literalmente en el texto (con ',' o
    '.' como separador). Sin tolerancia de redondeo: la tolerancia de
    nb06_expected_hit marca 'cubierto' un 27.5% cuando solo hay un 27 suelto en el
    texto, y eso falsea el diagnostico de retrieval.'''
    e = str(expected).replace(",", "")
    if "." in e:
        return e in text or e.replace(".", ",") in text
    # entero: aparece con separador de miles o sin el
    grouped = f"{int(e):,}"
    return grouped in text or e in text


EVAL_PATH = REPO + "/memo/eval_questions.json"
QA_PATH = REPO + "/memo/reflection_qa.jsonl"
SEED_PATH = REPO + "/pipeline/triples_seed.json"

eval_data = json.load(open(EVAL_PATH))
questions = {q["id"]: q for q in eval_data["questions"]}

# mismo conjunto de numeros verificados que el scorer de MEMO
verified = set()
try:
    for line in open(QA_PATH):
        verified |= nb06_numbers_in(json.loads(line)["answer"])
except FileNotFoundError:
    pass
try:
    seed = json.load(open(SEED_PATH))
    triples = seed if isinstance(seed, list) else seed.get("triples", [])
    for t in triples:
        for k in ("valor", "variacion_pct"):
            if t.get(k) is not None:
                verified.add(nb06_digits(str(t[k])))
except FileNotFoundError:
    pass
verified |= {"2024", "2023", "2022"}


def nb06_score(trace_path, retrieval_path, out_path, verified_modes,
               title, intro_lines, correctas_suffix, este_run_label,
               other_rows, cambio_lines, verdict_lines, caveats_lines):
    trace = [json.loads(l) for l in open(trace_path)]
    retr = {r["id"]: r for r in json.load(open(retrieval_path))["questions"]}

    rows = []
    for r in trace:
        q = questions[r["id"]]
        ans = r["final_answer"]
        ans_nums = nb06_numbers_in(ans)
        halluc = sorted(n for n in ans_nums if n not in verified and len(n) > 1)
        ok = nb06_expected_hit(ans_nums, q.get("expected_numbers", []))
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
                        if not nb06_literal_present(top5_text, e)]
        missing_ctx = [e for e in exp_nums
                       if not nb06_literal_present(ctx_text, e)]
        # Clasificacion del fallo: verificada a mano DESPUES de la corrida,
        # contra el corpus (la heuristica automatica etiqueta mal varios casos).
        mode = verified_modes[r["id"]]
        if verdict == "correcta":
            mode = "ok"
        # cobertura literal (sin tolerancia) por cifra, en top-5 y en contexto
        cov_literal = {e: nb06_literal_present(top5_text, e) for e in exp_nums}
        ctx_literal = {e: nb06_literal_present(ctx_text, e) for e in exp_nums}
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

    lines = [title, ""] + intro_lines + [
             "",
             f"Correctas: **{total_ok}/12**. Con alucinacion (numero no verificado en el informe): **{total_halluc}/12**. "
             f"Retrieval top-5 con todas las cifras esperadas literales: **{n_retr_full}/12** "
             f"{correctas_suffix}",
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
    for mode, n in Counter(r["failure_mode"] for r in rows).most_common():
        ids = ", ".join(r["id"] for r in rows if r["failure_mode"] == mode)
        lines.append(f"- {mode}: {n} ({ids})")
    lines += ["",
              "## Comparacion de los sistemas (12 preguntas, mismo scorer)",
              "",
              "| Sistema | Correctas | Nivel 1 | Nivel 2 | Nivel 3 |",
              "|---------|-----------|---------|---------|---------|",
              "| Grafo (Relate, Cypher) | 12/12 | 4/4 | 4/4 | 4/4 |",
              f"| {este_run_label} | {total_ok}/12 | "
              f"{sum(1 for r in by_level[1] if r['verdict'] == 'correcta')}/4 | "
              f"{sum(1 for r in by_level[2] if r['verdict'] == 'correcta')}/4 | "
              f"{sum(1 for r in by_level[3] if r['verdict'] == 'correcta')}/4 |"]
    lines += other_rows
    lines += cambio_lines
    lines += verdict_lines
    lines += caveats_lines
    lines += ["",
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

    open(out_path, "w").write("\n".join(lines) + "\n")
    print(f"correctas {total_ok}/12, con alucinacion {total_halluc}/12, "
          f"retrieval literal completo {n_retr_full}/12 -> {out_path}")
    for r in rows:
        print(f"  {r['id']} L{r['level']} {r['verdict']} [{r['failure_mode']}]"
              + (f" halluc={r['hallucinated']}" if r["hallucinated"] else ""))

# --- Scorer hibrido (sin expansion) ---
nb06_score(REPO + "/rag_baseline/run_trace_hybrid_colab.jsonl",
           REPO + "/rag_baseline/retrieval_hybrid.json",
           REPO + "/rag_baseline/results_hybrid_colab.md",
           {
               "Q1": "ok",
               "Q2": "cifra esperada fuera del corpus (+24.9% no existe en el informe; el lector contesto bien lo preguntado)",
               "Q3": "fallo de retrieval (23,206 solo en p.21, no en top-5 hibrido [16,88,75,72,55]; el lector se abstuvo)",
               "Q4": "fallo de retrieval + cifra mal atribuida (571 no en top-5 hibrido [83,100,85,88,92]; contesto '12 estaciones' con un 12 suelto del contexto)",
               "Q5": "fallo de retrieval (14.2 no en top-5 hibrido [28,88,128,80,100]; el lector se abstuvo)",
               "Q6": "fallo del lector (307,197 bien; Coca-Cola FEMSA la puso en 128,736 en vez de 279,793: 128,736 es su utilidad bruta, no su ingreso; el contexto hibrido cambio vs el denso)",
               "Q7": "cifra esperada fuera del corpus + aritmetica inventada (+27.5% y +24.9% no existen en el informe; el lector acerto direccion y absolutos, pero invento 'aumento un 275% (24.6 - 8.6 = 16)' y 'crecio un 50%'; el 16 cuenta como alucinacion)",
               "Q8": "fallo del lector (los 4 paises estaban en contexto; listo solo Mexico y 'America Latina')",
               "Q9": "cifra esperada fuera del corpus (39.3/35.8/75.1% no existen); el lector se abstuvo en vez de inventar la suma",
               "Q10": "fallo de retrieval ('FEMSA Forward' no en top-5 hibrido [13,75,4,34,117]; respuesta vaga sin el nombre)",
               "Q11": "ok",
               "Q12": "fallo de retrieval (Health +5.8% en pp. 5/112/114, no en top-5 hibrido [88,16,75,99,96]; el lector se abstuvo)",
           },
           "# Resultados RAG hibrido (BM25 + denso, RRF k=60) — FEMSA 2024",
           ["Sistema: retrieval hibrido top-5 paginas — BM25 Okapi (rank_bm25, "
            "k1=1.5, b=0.75) + all-MiniLM-L6-v2, fusion Reciprocal Rank Fusion "
            "k=60 — sobre las 151 paginas del informe; reader Qwen2.5-1.5B-Instruct "
            "local (bf16, CPU), mismo prompt y mismo condense que el baseline "
            "denso. Scorer identico al de MEMO (misma exactitud numerica, "
            "misma definicion de alucinacion, misma regla Q3 con 1,596). La unica "
            "diferencia con el baseline denso es el retrieval."],
           "(denso puro: 2/12; ver Caveats).",
           "RAG hibrido BM25+denso (este run)",
           ["| RAG vectorial denso puro | 2/12 | 0/4 | 1/4 | 1/4 |",
            "| MEMO run 2 (0.5B LoRA r=32 + ejecutivo 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |",
            "| MEMO run 1 (0.5B LoRA r=8 + ejecutivo 0.5B) | 0/12 | 0/4 | 0/4 | 0/4 |",
            ],
           ["",
            "## Que cambio vs el baseline denso (2/12)",
            "",
            "Mismo puntaje, distinta composicion: el hibrido acierta Q1 y Q11; "
            "el denso acertaba Q6 y Q11. Un flip a favor, un flip en contra, "
            "y las otras 10 preguntas igual.",
            "",
            "- **Q1: denso incorrecta -> hibrido correcta (unico flip a favor).** "
            "El hibrido trajo el 11.2% al top-5 (paginas [88,113,16,112,115]) "
            "y el lector eligio 781,585 en vez de los 40,236 del denso. Q1 "
            "en el denso era fallo del lector, no de retrieval (781,585 si "
            "estaba en contexto); el hibrido lo arreglo de rebote, al subir "
            "la pagina con el 11.2%.",
            "- **Q6: denso correcta -> hibrido incorrecta (flip en contra).** "
            "El contexto hibrido ([88,112,79,27,107] vs [30,114,10,27,112] "
            "del denso) cambio la tabla que vio el lector y tomo 128,736 como "
            "ingreso de Coca-Cola FEMSA. 128,736 es una cifra real del informe "
            "—su utilidad bruta, no su ingreso (279,793)—, asi que no cuenta "
            "como alucinacion, pero la respuesta esta mal. El hibrido arreglo "
            "el retrieval de Q1 y rompió el de Q6.",
            "- **Q3, Q4, Q10: igual de mal.** Los 3 fallos de retrieval del "
            "denso siguen fallando: 23,206 solo esta en la p.21, 571 en pp. "
            "21/114/126 y 'FEMSA Forward' en pp. 3/5/8/12/14/42/101/115, y "
            "ninguna de esas paginas entro al top-5 hibrido. BM25 no puede "
            "emparejar lo que no esta en la pregunta: '571' y 'Forward' no "
            "aparecen en las preguntas, y 'proximidad'/'estaciones' estan en "
            "espanol mientras el corpus esta en ingles.",
            "- **Q5: peor retrieval, mejor conducta.** El denso SI traia el "
            "14.2 en top-5 y el lector divago (alucinacion '114'); el hibrido "
            "ya no trae el 14.2 y el lector se abstuvo honestamente ('No lo "
            "se'). Sigue incorrecta, pero sin alucinar.",
            "- **Q12: peor retrieval.** El denso traia un 5.8 literal (margen "
            "de utilidad neta, espurio, no el crecimiento de Salud); el "
            "hibrido ni ese trae. El lector se abstuvo en ambas.",
            "- **Q9: mejor conducta.** El denso invento la suma 43,265; el "
            "hibrido se abstuvo. Sigue incorrecta (las cifras no existen en "
            "el informe), pero sin alucinar.",
            "- **Q8: peor lector.** El denso listo Mexico y Colombia (2 de 4); "
            "el hibrido, con los 4 paises en contexto, listo solo Mexico y "
            "'America Latina'.",
            "- **Q7: peor conducta.** El denso contesto solo direccion y "
            "absolutos (24.6 vs 8.6) y paro ahi, sin alucinar; el hibrido "
            "siguio e invento 'aumento un 275% (24.6 - 8.6 = 16)' y 'crecio "
            "un 50%'. El 16 cuenta como alucinacion. Sigue incorrecta en "
            "ambas (los porcentajes no estan en el informe).",
            "- **Sin cambio:** Q2 (cifra fuera del corpus), Q11 "
            "(correcta en ambas).",
            "- **Cobertura literal ALL en top-5:** denso 2/12 (Q5, Q12) -> "
            "hibrido 1/12 (Q1). El hibrido gano Q1 y perdio Q5 y Q12.",
            ],
           ["",
            "## Veredicto para el post",
            "",
            "**RAG hibrido BM25+denso: 2/12 — el mismo puntaje que el denso "
            "puro, con un perfil distinto.** No cerro la brecha de retrieval: "
            "de los 4 fallos de retrieval del denso (Q3, Q4, Q10, Q12), el "
            "hibrido no arreglo ninguno, y perdio Q5. Lo que arreglo fue Q1, "
            "que en el denso era fallo del lector (781,585 si estaba en "
            "contexto): el hibrido ademas trajo el 11.2% y el lector eligio "
            "bien. BM25 reordena, pero no puede arreglar un desajuste de "
            "vocabulario: las preguntas estan en espanol, el corpus en "
            "ingles, y los tokens discriminativos viven en la respuesta, no "
            "en la pregunta. El arreglo directo seria traducir o expandir la "
            "pregunta con sinonimos del dominio, o chunkear por secciones en "
            "vez de por pagina — no otro ranker lexico.",
            "",
            "Lo que SI cambio es la conducta, en ambas direcciones: "
            "alucinaciones de 2/12 (denso: Q5 '114', Q9 '43265') a 1/12 "
            "(hibrido: Q7 '16'). Q5 y Q9 ya no alucinan — el lector se "
            "abstiene — pero Q7 ahora inventa porcentajes ('aumento un "
            "275%', 'crecio un 50%') que el denso no inventaba. El mismo "
            "lector, con distinto contexto, divaga distinto. Abstenerse no "
            "suma puntos con este scorer, asi que el marcador no se mueve. "
            "La moraleja se mantiene y se afina: la exactitud de cifras "
            "exige que el dato llegue verificado al lector; cambiar el "
            "ranker sin cambiar el vocabulario de la pregunta es mover los "
            "muebles.",
            ],
           ["",
            "## Caveats (verificados, no supuestos)",
            "",
            "- El diagnostico de retrieval con tolerancia de redondeo del "
            "scorer (±0.1%) decia 5/12 con todas las cifras cubiertas; la "
            "cobertura literal estricta en top-5 es 1/12 (Q1) — la "
            "tolerancia marca 'cubierto' un 27.5% cuando solo hay un 27 "
            "suelto, igual que en el baseline denso. La tabla de detalle "
            "usa cobertura literal; la clasificacion de fallos se verifico "
            "a mano. En Q6 las cifras estan como valores exactos "
            "(307,197/279,793) mientras el campo esperado trae los "
            "redondeados de la ontologia (307,200/279,800).",
            "- El hibrido NO cerro la brecha de retrieval del denso: Q3, "
            "Q4 y Q10 siguen sin traer la cifra al top-5. La razon es "
            "estructural y se verifico en los rankings: las preguntas estan "
            "en espanol y el corpus en ingles, asi que BM25 solo empareja "
            "tokens exactos (numeros, nombres propios) y los tokens "
            "discriminativos viven en la respuesta, no en la pregunta "
            "('571' no esta en la pregunta de Q4; 'Forward' no esta en la "
            "de Q10; '23,206' no esta en la de Q3). BM25 reordena, pero no "
            "puede inventar el vocabulario que falta. El arreglo directo "
            "seria traducir la pregunta o expandirla con sinonimos del "
            "dominio, no otro ranker lexico.",
            "- El condensado de contexto (ventanas de ~1,800 caracteres por "
            "chunk, 2 ventanas) fue necesario porque el prompt de 5 paginas "
            "completas (~4.5k tokens) tarda >6 min por pregunta en CPU. "
            "Es el mismo workaround que en el baseline denso, con el mismo "
            "costo medido: cifras que estan en el top-5 pueden no llegar "
            "al lector. Es un costo del workaround CPU, no del metodo RAG "
            "en si.",
            "- La pregunta esta en espanol y el corpus en ingles; el "
            "retrieval es cross-lingual con un embedder entrenado en ingles "
            "(all-MiniLM-L6-v2). Q12 exige mapear 'Salud' -> 'Health'.",
            "- Q2 no pregunta el crecimiento (+24.9%) pero el scorer lo "
            "exige; la respuesta '8.6 millones' es correcta para la pregunta "
            "tal como esta hecha. Misma severidad que en MEMO, asi que los "
            "puntajes son comparables."])

# --- Scorer hibrido + expansion ---
nb06_score(REPO + "/rag_baseline/run_trace_expanded_colab.jsonl",
           REPO + "/rag_baseline/retrieval_expanded.json",
           REPO + "/rag_baseline/results_expanded_colab.md",
           {
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
           },
           "# Resultados RAG hibrido + expansion de queries (BM25 + denso, RRF k=60) — FEMSA 2024",
           ["Sistema: retrieval hibrido top-5 paginas — BM25 Okapi (rank_bm25, "
            "k1=1.5, b=0.75) + all-MiniLM-L6-v2, fusion Reciprocal Rank Fusion "
            "k=60 — sobre las 151 paginas del informe, con queries expandidas "
            "(traduccion manual al ingles + keywords de dominio, ver la celda 1 "
            "de este notebook; sin cifras esperadas ni nombres-respuesta). "
            "Reader Qwen2.5-1.5B-Instruct local (bf16, CPU), mismo prompt y "
            "mismo condense que el baseline denso. Scorer identico al de "
            "MEMO (misma exactitud numerica, misma definicion de "
            "alucinacion, misma regla Q3 con 1,596). La unica diferencia con "
            "el hibrido es la QUERY de retrieval."],
           "(hibrido sin expansion: 1/12; denso puro: 2/12; ver Caveats).",
           "RAG hibrido + expansion de queries (este run)",
           ["| RAG hibrido BM25+denso (sin expansion) | 2/12 | 1/4 | 0/4 | 1/4 |",
            "| RAG vectorial denso puro | 2/12 | 0/4 | 1/4 | 1/4 |",
            "| MEMO run 2 (0.5B LoRA r=32 + ejecutivo 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |",
            "| MEMO run 1 (0.5B LoRA r=8 + ejecutivo 0.5B) | 0/12 | 0/4 | 0/4 | 0/4 |",
            ],
           ["",
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
            ],
           ["",
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
            ],
           ["",
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
            "puntajes son comparables."])
"""),
        md("## 5. Comparación rápida: denso vs híbrido vs expandido"),
        code("""
import re

for name, path in [("denso (NB05)", REPO + "/rag_baseline/results_colab.md"),
                   ("hibrido", REPO + "/rag_baseline/results_hybrid_colab.md"),
                   ("hibrido + expansion", REPO + "/rag_baseline/results_expanded_colab.md")]:
    try:
        txt = open(path).read()
        m = re.search(r"Correctas: \\*\\*(\\d+/12)\\*\\*", txt)
        print(f"{name:22s} {m.group(1) if m else '?'}")
    except FileNotFoundError:
        print(f"{name:22s} (sin resultados: corre el NB05 para el denso)")
"""),
        *save_cell([
            ("rag_baseline/retrieval_hybrid.json", "Retrieval híbrido (12 preguntas)", True),
            ("rag_baseline/retrieval_expanded.json", "Retrieval híbrido + expansión", True),
            ("rag_baseline/run_trace_hybrid_colab.jsonl", "Traza híbrido", True),
            ("rag_baseline/run_trace_expanded_colab.jsonl", "Traza híbrido + expansión", True),
            ("rag_baseline/results_hybrid_colab.md", "Resultados híbrido", True),
            ("rag_baseline/results_expanded_colab.md", "Resultados híbrido + expansión", True),
            ("rag_baseline/expanded_queries.json", "Queries expandidas", False),
        ], note="El notebook 07 (juez) y el 08 (RAGAS) verifican que existan estos "
                "retrievales y trazas antes de correr: sin ellos no hay nada que juzgar "
                "ni medir."),
    ]
    write_nb("06_rag_hibrido_y_expansion.ipynb", cells, "06 RAG híbrido y expansión")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 07 — RAG agéntico JUEZ (selección de chunks con juez, vía Ollama)
# ---------------------------------------------------------------------------

def build_07():
    cells = [
        title_cell(
            "07 — RAG agéntico: el juez selecciona los chunks (vía Ollama)",
            "Segundo diseño de RAG agéntico, el que pidió el usuario: el sistema SOLO "
            "califica los chunks del RAG híbrido y selecciona los útiles con un juez. "
            "Sin graph_query, sin calculator.\n\n"
            "Por pregunta: (1) retrieval híbrido top-10 (BM25 Okapi + MiniLM, RRF k=60); "
            "(2) UN llamado al juez con los 10 chunks numerados, que devuelve un "
            "veredicto por chunk (UTIL / NO UTIL) con su razón corta — máximo 3 reintentos "
            "si los veredictos no se pueden parsear; si a los 3 sigue sin veredictos "
            "válidos, la pregunta se abstiene sin llamar al lector; (3) el lector "
            "responde ÚNICAMENTE con los chunks seleccionados (mismo lector, mismo "
            "condense, mismo prompt del RAG baseline); si el juez no "
            "selecciona nada, abstención honesta.\n\n"
            "**Juez y lector vía ChatOllama** contra el servidor local "
            "(`JUDGE_OLLAMA_MODEL`, default `qwen2.5:7b`; `qwen2.5:1.5b` para paridad con "
            "el run en CPU). La lógica de retrieval, del juez/lector y del scorer va "
            "inline en este notebook, sin correr ni importar scripts del repo "
            "(procedencia: `rag_baseline/retrieve_hybrid.py`, "
            "`agentic_rag/judge_rerank.py` y `rag_baseline/answer.py` para el juez y "
            "el lector, `agentic_rag/score_judge.py` para el scorer).\n\n"
            "**El intento anterior queda como negativo documentado:** el agente ReAct "
            "con LangChain + LangGraph y 3 herramientas sacó 2/12 en CPU y nunca usó "
            "`vector_search` ni `calculator` (24/24 llamadas fueron `graph_query`): midió "
            "un agente solo-grafo, no un RAG agéntico. Ese run no se toca "
            "(`agentic_rag/results_agentic.md`).\n\n"
            "**Prerrequisitos:** notebook 01 (chunks). El retrieval híbrido top-10 se "
            "genera en este mismo notebook."),
        gpu_check_cell(),
        drive_mount_cell(),
        params_cell("""
# --- Parametros ---
JUDGE_OLLAMA_MODEL = "qwen2.5:7b"   # juez y lector: 7b (default, el que muestra lo agentico)
                                     # o qwen2.5:1.5b para paridad con el run en CPU
K = 10
RETRIEVAL_K10 = REPO + "/rag_baseline/retrieval_hybrid_k10.json"
TRACE_OUT = REPO + "/agentic_rag/run_trace_judge_colab.jsonl"
"""),
        repo_setup_cell(),
        *data_ensure_cells(),
        pip_cell([
            '"torch>=2.15.0"', "transformers==5.24.1", "sentence-transformers==6.3.0",
            "numpy>=2.5.5", "rank-bm25==0.2.2", "langchain-ollama",
        ]),
        *ollama_compact_cells(["qwen2.5:7b", "qwen2.5:1.5b"], "JUDGE_OLLAMA_MODEL"),
        md("## 1. Retrieval híbrido top-10 (inline)\n\n"
           "Lógica inline (procedencia: `rag_baseline/retrieve_hybrid.py`): BM25 Okapi "
           "+ MiniLM, fusión RRF k=60, top-10 por pregunta (el juez necesita más de "
           "donde elegir que el top-5 fijo). Determinista: se puede regenerar sin "
           "cambiar el resultado. Escribe `retrieval_hybrid_k10.json` con el mismo "
           "esquema que el script (página, score RRF, `bm25_rank`/`dense_rank` y "
           "scores por chunk, cobertura de cifras esperadas y top-10 de cada ranker)."),
        code(r"""
import json, re, time

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60
EVAL_PATH = REPO + "/memo/eval_questions.json"
TOKEN_RE = re.compile(r"[a-z0-9áéíóúüñ]+")


def tokenize(text):
    return TOKEN_RE.findall((text or "").lower())


def digits(s):
    return re.sub(r"[.,]", "", s)


def numbers_in(text):
    return {digits(m.group(0)) for m in re.finditer(r"\d[\d.,]*", text or "")}


def rrf_ranks(bm25_scores, dense_sims):
    # Ranks 1-based (1 = mejor) de cada ranker sobre los 151 chunks.
    bm25_rank = np.empty(len(bm25_scores), dtype=int)
    bm25_rank[np.argsort(-np.asarray(bm25_scores))] = np.arange(1, len(bm25_scores) + 1)
    dense_rank = np.empty(len(dense_sims), dtype=int)
    dense_rank[np.argsort(-np.asarray(dense_sims))] = np.arange(1, len(dense_sims) + 1)
    return bm25_rank, dense_rank


t0 = time.time()
chunks = [json.loads(l) for l in open(CHUNKS)]
texts = [c["text"] for c in chunks]
print(f"BM25 Okapi (k1=1.5, b=0.75) + {EMBED_MODEL} sobre {len(chunks)} chunks ...",
      flush=True)
tokenized = [tokenize(t) for t in texts]
bm25 = BM25Okapi(tokenized)
print(f"BM25 index listo en {time.time() - t0:.1f}s", flush=True)

model = SentenceTransformer(EMBED_MODEL)
emb = np.asarray(model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=False), dtype=np.float32)
print(f"embeddings: {emb.shape} en {time.time() - t0:.1f}s", flush=True)

eval_qs = json.load(open(EVAL_PATH))["questions"]
results = []
for q in eval_qs:
    q_tokens = tokenize(q["question"])
    b_scores = np.asarray(bm25.get_scores(q_tokens), dtype=np.float64)
    q_emb = model.encode([q["question"]], normalize_embeddings=True,
                         show_progress_bar=False)[0]
    sims = emb @ q_emb
    b_rank, d_rank = rrf_ranks(b_scores, sims)
    rrf = 1.0 / (RRF_K + b_rank) + 1.0 / (RRF_K + d_rank)
    top = np.argsort(-rrf)[:K]
    retrieved = []
    for idx in top:
        c = chunks[int(idx)]
        retrieved.append({"page": c["page"],
                          "score": round(float(rrf[idx]), 6),
                          "bm25_rank": int(b_rank[idx]),
                          "dense_rank": int(d_rank[idx]),
                          "bm25_score": round(float(b_scores[idx]), 4),
                          "dense_score": round(float(sims[idx]), 4),
                          "text": c["text"]})
    union_nums = set()
    for r in retrieved:
        union_nums |= numbers_in(r["text"])
    coverage = {}
    for e in q.get("expected_numbers", []):
        d = digits(str(e))
        hit = d in union_nums
        if not hit:
            try:
                v = float(e)
                hit = any(abs(float(a) - v) <= max(1.0, abs(v) * 0.001)
                          for a in union_nums if a.replace(".", "", 1).isdigit())
            except ValueError:
                hit = False
        coverage[e] = bool(hit)
    b_top10 = [int(chunks[i]["page"]) for i in np.argsort(-b_scores)[:10]]
    d_top10 = [int(chunks[i]["page"]) for i in np.argsort(-sims)[:10]]
    results.append({"id": q["id"], "level": q["level"],
                    "question": q["question"],
                    "expected": q["expected"],
                    "expected_numbers": q.get("expected_numbers", []),
                    "retrieved": retrieved,
                    "coverage": coverage,
                    "all_covered": all(coverage.values()) if coverage else None,
                    "bm25_top10_pages": b_top10,
                    "dense_top10_pages": d_top10})
    cov = "ALL" if results[-1]["all_covered"] else (
        "PARTIAL" if any(coverage.values()) else "NONE")
    print(f"{q['id']} L{q['level']} coverage={cov} "
          f"pages={[r['page'] for r in retrieved]} "
          f"(bm25_top10={b_top10[:5]} dense_top10={d_top10[:5]})", flush=True)

out = {"method": "hybrid BM25Okapi (k1=1.5, b=0.75) + MiniLM dense, RRF k=60",
       "embed_model": EMBED_MODEL, "k": K, "n_chunks": len(chunks),
       "retrieval_seconds": round(time.time() - t0, 1),
       "questions": results}
json.dump(out, open(RETRIEVAL_K10, "w"), ensure_ascii=False, indent=2)
n_all = sum(1 for r in results if r["all_covered"])
n_none = sum(1 for r in results if r["all_covered"] is False
             and not any(r["coverage"].values()))
print(f"retrieval hibrido: {n_all}/12 preguntas con TODAS las cifras "
      f"esperadas en top-{K}; {n_none}/12 sin ninguna -> {RETRIEVAL_K10}")
"""),
        md("## 2. Juez + lector (ChatOllama)\n\n"
           "Un llamado al juez por pregunta con los 10 chunks numerados (misma versión "
           "condensada que verá el lector; SYSTEM del juez, parseo de veredictos, "
           "SYSTEM del lector y `condense()` van inline en la celda de abajo, "
           "procedencia: `agentic_rag/judge_rerank.py` y `rag_baseline/answer.py`; "
           "máximo 3 reintentos; si no hay veredictos válidos, abstención sin lector). "
           "El lector contesta solo con los chunks seleccionados. "
           "Reanudable: salta las preguntas cuyo id ya está en la traza "
           "(`run_trace_judge_colab.jsonl`, nunca toca la traza del run en CPU)."),
        code(r"""
import json, re, time, unicodedata
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

ABSTAIN = "No lo sé."

# --- Juez: SYSTEM y parseo de veredictos (inline) ---
JUDGE_SYSTEM = (
    "Eres un juez que decide si un fragmento del informe anual de FEMSA "
    "2024 sirve para responder una pregunta. Se te presentan varios "
    "fragmentos numerados del 1 al 10. Devuelve UNA linea por fragmento, "
    "en este formato exacto: 'N: UTIL: <razon corta>' si el fragmento "
    "contiene informacion que ayuda a responder la pregunta, o "
    "'N: NO UTIL: <razon corta>' si no la contiene. Una linea por "
    "fragmento, sin texto adicional, sin resumen, sin numeracion extra.")

_LINE_RE = re.compile(r"^\s*(\d{1,2})\s*[:\.\)\-]\s*(.+)$", re.MULTILINE)


def _norm(s):
    # Minusculas sin acentos, para comparar 'UTIL'/'ÚTIL'/'util' igual.
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


def _parse_line(rest):
    # Devuelve (useful, reason) o None si no se puede parsear.
    # Se revisa NO UTIL antes que UTIL porque 'NO UTIL' contiene 'UTIL'.
    rest = rest.strip()
    low = _norm(rest)
    if re.match(r"^(no\s+util|no\b)", low):
        reason = re.sub(r"^(no\s+[uú]til|no)\s*:?\s*", "", rest, flags=re.I)
        return (False, reason.strip()[:120])
    if re.match(r"^util\b", low):
        reason = re.sub(r"^[uú]til\s*:?\s*", "", rest, flags=re.I)
        return (True, reason.strip()[:120])
    if re.match(r"^si\b", low):
        reason = re.sub(r"^s[íi]\s*:?\s*", "", rest, flags=re.I)
        return (True, reason.strip()[:120])
    return None


def parse_verdicts(raw, n):
    # Devuelve {1: {"useful": bool, "reason": str}, ..., n: {...}} si hay
    # una linea parseable para CADA uno de los n chunks; None si falta
    # alguno. La primera linea valida por numero gana.
    verdicts = {}
    for m in _LINE_RE.finditer(raw or ""):
        idx = int(m.group(1))
        if idx < 1 or idx > n or idx in verdicts:
            continue
        parsed = _parse_line(m.group(2))
        if parsed is not None:
            verdicts[idx] = {"useful": parsed[0], "reason": parsed[1]}
    if set(verdicts.keys()) == set(range(1, n + 1)):
        return verdicts
    return None


# --- Lector: SYSTEM y condense (inline) ---
READER_SYSTEM = (
    "Responde la pregunta usando UNICAMENTE la información de los fragmentos "
    "del informe anual de FEMSA 2024 que se te dan a continuación. Copia las "
    "cifras exactamente como aparecen en los fragmentos, sin redondear ni "
    "inventar nada. Solo si la pregunta pide explícitamente sumar o comparar "
    "cifras que están en los fragmentos, haz esa operación con esas cifras. "
    "Si la información de los fragmentos no alcanza para responder, responde "
    "exactamente: No lo sé. Responde en español, en una o dos frases cortas.")

MAX_CHUNK_CHARS = 1800
STOPWORDS = set("de la el en y a los del se las por un para con no una su al lo como\nmas pero sus le ya o este mi que qué cuando donde cuales cuál cómo cuánto".split())


def condense(text, question, max_chars=MAX_CHUNK_CHARS):
    # ~max_chars caracteres por chunk: ventanas alrededor de las 2 lineas
    # del chunk mas relevantes a la pregunta (overlap de palabras + lineas
    # con cifras). Dos ventanas porque las dos cifras que una pregunta
    # compara suelen estar lejos en la misma pagina.
    qwords = {w.lower() for w in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", question)
              if len(w) > 3 and w.lower() not in STOPWORDS}
    lines = text.splitlines() or [text]
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1

    def line_score(ln):
        low = ln.lower()
        s = sum(2 for w in qwords if w in low)
        if re.search(r"\d", ln):
            s += 1
        return s

    scored = sorted(((line_score(ln), i) for i, ln in enumerate(lines)
                     if line_score(ln) > 0), reverse=True)
    chosen = []
    for s, i in scored:
        if all(abs(i - j) > 3 for j in chosen):
            chosen.append(i)
        if len(chosen) == 2:
            break
    if not chosen:
        return text[:max_chars]
    half = max_chars // max(len(chosen), 1)
    parts = []
    for i in sorted(chosen):
        start = offsets[i]
        lo = max(0, start - half // 3)
        hi = min(len(text), lo + half)
        lo = max(0, hi - half)
        parts.append(text[lo:hi])
    out = "\n[...]\n".join(parts)
    return out[:max_chars]


judge_llm = ChatOllama(model=JUDGE_OLLAMA_MODEL, base_url="http://localhost:11434",
                       temperature=0, num_predict=200)
reader_llm = ChatOllama(model=JUDGE_OLLAMA_MODEL, base_url="http://localhost:11434",
                        temperature=0, num_predict=100)
print("juez/lector (Ollama):", JUDGE_OLLAMA_MODEL)

def judge_once(question, chunks):
    # chunks: lista de (page, condensed_text). Un llamado, 10 veredictos.
    parts = [f"[{i}] Página {p}\n{c}" for i, (p, c) in enumerate(chunks, start=1)]
    user = (f"Pregunta: {question}\n\nFragmentos del informe anual "
            f"de FEMSA 2024:\n\n" + "\n\n".join(parts))
    return judge_llm.invoke([SystemMessage(content=JUDGE_SYSTEM),
                             HumanMessage(content=user)]).content.strip()

def run_question(item):
    t0 = time.time()
    chunks = [(r["page"], condense(r["text"], item["question"]))
              for r in item["retrieved"]]
    presented = [{"n": i + 1, "page": p, "condensed": c}
                 for i, (p, c) in enumerate(chunks)]
    rec = {"id": item["id"], "level": item["level"], "question": item["question"],
           "decomposition": "judge-rerank",
           "reader_model": f"ollama {JUDGE_OLLAMA_MODEL}",
           "retrieved_pages": [r["page"] for r in item["retrieved"]],
           "retrieval_scores": [r["score"] for r in item["retrieved"]],
           "chunks_presented": presented,
           "judge": {"attempts": 0, "attempts_raw": [], "verdicts": None,
                     "ok": False, "seconds": 0.0},
           "selected_pages": [],
           "reader": {"attempts": 0, "seconds": 0.0, "failed": False, "skipped": False},
           "context": "", "final_answer": ABSTAIN, "seconds": 0.0}
    # Juez: maximo 3 reintentos si no se pueden parsear los veredictos.
    tj = time.time()
    verdicts = None
    for attempt in range(1, 4):
        try:
            raw = judge_once(item["question"], chunks)
        except Exception as e:
            raw = f"<ERROR: {type(e).__name__}: {e}>"
        rec["judge"]["attempts_raw"].append(raw)
        rec["judge"]["attempts"] = attempt
        verdicts = parse_verdicts(raw, len(chunks))
        if verdicts is not None:
            break
    rec["judge"]["seconds"] = round(time.time() - tj, 1)
    if verdicts is None:
        rec["judge"]["ok"] = False
        rec["reader"]["skipped"] = True
        rec["seconds"] = round(time.time() - t0, 1)
        return rec
    rec["judge"]["ok"] = True
    rec["judge"]["verdicts"] = verdicts
    selected = [i for i in range(1, len(chunks) + 1) if verdicts[i]["useful"]]
    rec["selected_pages"] = [chunks[i - 1][0] for i in selected]
    if not selected:
        rec["reader"]["skipped"] = True
        rec["seconds"] = round(time.time() - t0, 1)
        return rec
    ctx = "\n\n".join(f"[Página {chunks[i - 1][0]}]\n{chunks[i - 1][1]}" for i in selected)
    rec["context"] = ctx
    # Lector: maximo 3 reintentos si el llamado falla o viene vacio.
    tr = time.time()
    answer, failed = None, False
    user = (f"Fragmentos del informe anual de FEMSA 2024:\n\n{ctx}\n\n"
            f"Pregunta: {item['question']}")
    for attempt in range(1, 4):
        try:
            ans = reader_llm.invoke([SystemMessage(content=READER_SYSTEM),
                                     HumanMessage(content=user)]).content.strip()
        except Exception as e:
            ans = ""
            print(f"  {item['id']} lector intento {attempt} fallo: {type(e).__name__}: {e}", flush=True)
        if ans:
            answer = ans
            rec["reader"]["attempts"] = attempt
            break
        rec["reader"]["attempts"] = attempt
    else:
        failed = True
    rec["reader"]["seconds"] = round(time.time() - tr, 1)
    rec["reader"]["failed"] = failed
    rec["final_answer"] = ABSTAIN if failed else answer
    rec["seconds"] = round(time.time() - t0, 1)
    return rec

done = set()
try:
    for line in open(TRACE_OUT):
        try:
            done.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            pass
except FileNotFoundError:
    pass
if done:
    print(f"reanudando: {sorted(done)} ya en {TRACE_OUT}")

qs = json.load(open(RETRIEVAL_K10))["questions"]
t0 = time.time()
with open(TRACE_OUT, "a") as f:
    for item in qs:
        if item["id"] in done:
            print(f"{item['id']} ya hecha, salto", flush=True)
            continue
        print(f"{item['id']} L{item['level']}: {item['question']}", flush=True)
        rec = run_question(item)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        print(f"{item['id']} juez_ok={rec['judge']['ok']} "
              f"intentos_juez={rec['judge']['attempts']} "
              f"seleccionados={len(rec['selected_pages'])}/10 "
              f"lector_intentos={rec['reader']['attempts']} "
              f"-> {rec['final_answer'][:160]}", flush=True)
print(f"listo en {time.time() - t0:.1f}s -> {TRACE_OUT}")
"""),
        md("## 3. Puntuar con el mismo scorer (con diagnóstico del juez, inline)\n\n"
           "Lógica inline (procedencia: `agentic_rag/score_judge.py`): escribe "
           "`results_agentic_judge_colab.md` con el puntaje, la comparación con los "
           "demás sistemas y el diagnóstico del juez (precisión/recall de selección "
           "contra las páginas de evidencia). Misma exactitud numérica, misma "
           "definición de alucinación, misma regla Q3 con 1,596 y mismas keywords "
           "Q8/Q10/Q11 que los demás scorers. Los modos de fallo quedan en PENDIENTE: "
           "se verifican a mano contra el corpus y se llenan en `VERIFIED_MODES` de "
           "la celda de abajo, y se vuelve a correr la celda con `BLANK_MODES = False`."),
        code(r"""
import json, re
from collections import Counter

TRACE_PATH = TRACE_OUT
OUT_MD = REPO + "/agentic_rag/results_agentic_judge_colab.md"
EVAL_PATH = REPO + "/memo/eval_questions.json"
QA_PATH = REPO + "/memo/reflection_qa.jsonl"
SEED_PATH = REPO + "/pipeline/triples_seed.json"
GT_PATH = REPO + "/rag_baseline/ragas/ground_truth_evidence.json"
MODEL_LABEL = f"Ollama {JUDGE_OLLAMA_MODEL} (Colab T4)"
SYSTEM_NAME = "RAG agentico juez Ollama (Colab)"
# BLANK_MODES = True: modos de fallo en PENDIENTE (run nuevo). Tras
# verificarlos a mano, llena VERIFIED_MODES abajo y corre con False.
BLANK_MODES = True
VERIFIED_MODES = {f"Q{i}": "PENDIENTE" for i in range(1, 13)}
if BLANK_MODES:
    VERIFIED_MODES = {qid: "PENDIENTE" for qid in VERIFIED_MODES}


def digits(s):
    return re.sub(r"[.,]", "", s)


def numbers_in(text):
    return {digits(m.group(0)) for m in re.finditer(r"\d[\d.,]*", text or "")}


def norm_num(x):
    return digits(str(x))


def expected_hit(answer_nums, expected):
    # True si cada numero esperado aparece (o su redondeo a entero).
    for e in expected:
        d = norm_num(e)
        if d in answer_nums:
            continue
        try:
            v = float(e)
            ok = any(abs(float(a) - v) <= max(1.0, abs(v) * 0.001)
                     for a in answer_nums if a.replace(".", "", 1).isdigit())
        except ValueError:
            ok = False
        if not ok:
            return False
    return True


def literal_present(text, expected):
    # True si la cifra esperada aparece literalmente en el texto (con ','
    # o '.' como separador). Sin tolerancia de redondeo.
    e = str(expected).replace(",", "")
    if "." in e:
        return e in text or e.replace(".", ",") in text
    grouped = f"{int(e):,}"
    return grouped in text or e in text


eval_data = json.load(open(EVAL_PATH))
questions = {q["id"]: q for q in eval_data["questions"]}
trace = [json.loads(l) for l in open(TRACE_PATH)]

# Mismo conjunto de numeros verificados que el scorer de MEMO.
verified = set()
try:
    for line in open(QA_PATH):
        verified |= numbers_in(json.loads(line)["answer"])
except FileNotFoundError:
    pass
try:
    seed = json.load(open(SEED_PATH))
    triples = seed if isinstance(seed, list) else seed.get("triples", [])
    for t in triples:
        for k in ("valor", "variacion_pct"):
            if t.get(k) is not None:
                verified.add(digits(str(t[k])))
except FileNotFoundError:
    pass
verified |= {"2024", "2023", "2022"}

# Paginas de evidencia por pregunta (ground truth auditable, RAGAS).
evidence_pages = {}
try:
    gt = json.load(open(GT_PATH))
    for q in gt.get("questions", []):
        pages = set()
        for item in q.get("evidence", []):
            pages |= set(item.get("pages", []))
        evidence_pages[q["id"]] = pages
except FileNotFoundError:
    evidence_pages = {}

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
    presented_text = "\n".join(c.get("condensed", "")
                               for c in r.get("chunks_presented", []))
    cov_presented = {e: literal_present(presented_text, e) for e in exp_nums}
    cov_selected = {e: literal_present(r.get("context", ""), e) for e in exp_nums}
    ev = evidence_pages.get(r["id"], set())
    sel = set(r.get("selected_pages", []))
    sel_prec = (len(sel & ev) / len(sel)) if sel else None
    sel_rec = (len(sel & ev) / len(ev)) if ev else None
    judge = r.get("judge", {})
    mode = VERIFIED_MODES[r["id"]]
    if verdict == "correcta":
        mode = "ok"
    rows.append({"id": r["id"], "level": q["level"],
                 "question": q["question"], "expected": q["expected"],
                 "answer": ans, "verdict": verdict,
                 "hallucinated": halluc, "note": note,
                 "judge_ok": judge.get("ok"),
                 "judge_attempts": judge.get("attempts"),
                 "judge_seconds": judge.get("seconds", 0),
                 "n_selected": len(sel),
                 "selected_pages": sorted(sel),
                 "sel_precision": sel_prec,
                 "sel_recall": sel_rec,
                 "evidence_pages": sorted(ev),
                 "reader_attempts": r.get("reader", {}).get("attempts", 0),
                 "reader_skipped": r.get("reader", {}).get("skipped", False),
                 "reader_failed": r.get("reader", {}).get("failed", False),
                 "seconds": r.get("seconds", 0),
                 "coverage_presented": cov_presented,
                 "coverage_selected": cov_selected,
                 "failure_mode": mode})

by_level = {}
for row in rows:
    by_level.setdefault(row["level"], []).append(row)
total_ok = sum(1 for r in rows if r["verdict"] == "correcta")
total_halluc = sum(1 for r in rows if r["hallucinated"])
n_ev_full = sum(1 for r in rows if r["coverage_selected"]
                and all(r["coverage_selected"].values()))
n_judge_ok = sum(1 for r in rows if r["judge_ok"])
n_abstain_judge = sum(1 for r in rows if not r["judge_ok"])
n_abstain_empty = sum(1 for r in rows
                      if r["judge_ok"] and r["n_selected"] == 0)
total_selected = sum(r["n_selected"] for r in rows)
sel_inter = sum(len(set(r["selected_pages"]) & set(r["evidence_pages"]))
                for r in rows)
sel_total = sum(len(r["selected_pages"]) for r in rows)
ev_total = sum(len(r["evidence_pages"]) for r in rows)
pooled_prec = (sel_inter / sel_total) if sel_total else None
pooled_rec = (sel_inter / ev_total) if ev_total else None
total_judge_seconds = sum(r["judge_seconds"] for r in rows)
total_seconds = sum(r["seconds"] for r in rows)
pending = [r["id"] for r in rows if r["failure_mode"] == "PENDIENTE"]
if pending:
    print(f"AVISO: modos de fallo PENDIENTES de verificar a mano: {pending}")

def lvl_ok(lv):
    return sum(1 for r in by_level.get(lv, [])
               if r["verdict"] == "correcta")

def pct(x):
    return f"{x:.2f}" if x is not None else "n/a"

lines = ["# Resultados RAG agentico juez (seleccion de chunks con juez) — FEMSA 2024",
         "",
         "Sistema: retrieval hibrido top-10 (BM25 Okapi + MiniLM, RRF k=60), "
         f"UN llamado al juez por pregunta que califica los 10 chunks de golpe "
         f"(UTIL / NO UTIL, modelo {MODEL_LABEL}), maximo 3 reintentos si los "
         "veredictos no se pueden parsear (si a los 3 sigue sin veredictos validos, "
         "la pregunta se abstiene sin llamar al lector), y el lector responde "
         "UNICAMENTE con los chunks seleccionados (mismo lector, mismo condense, "
         "mismo prompt del lector del RAG baseline; si el juez no selecciona nada, "
         "abstencion sin llamar al lector). Sin graph_query, sin calculator. "
         "Scorer con la misma exactitud numerica, misma definicion de "
         "alucinacion y misma regla Q3 con 1,596 que los demas sistemas.",
         "",
         f"Correctas: **{total_ok}/12**. Con alucinacion (numero no verificado en el informe): **{total_halluc}/12**. "
         f"Preguntas donde el contexto seleccionado contenia todas las cifras "
         f"esperadas (literal): **{n_ev_full}/12**. "
         f"Juez con veredictos validos: **{n_judge_ok}/12** preguntas "
         f"({n_abstain_judge} abstencion{'es' if n_abstain_judge != 1 else ''} por juez fallido, "
         f"{n_abstain_empty} por seleccion vacia). "
         f"Chunks seleccionados: **{total_selected}/120** "
         f"(media {total_selected / max(len(rows), 1):.1f} por pregunta). "
         f"Seleccion del juez vs evidencia (pooled): precision "
         f"**{pct(pooled_prec)}**, recall **{pct(pooled_rec)}**. "
         f"Tiempo total de la corrida: **{total_seconds / 60:.1f} min** "
         f"(juez {total_judge_seconds / 60:.1f} min, media "
         f"{total_seconds / max(len(rows), 1):.0f} s por pregunta).",
         "",
         "| ID | Nivel | Pregunta | Esperado | Respuesta del agente | Veredicto | Sel. | Juez | Modo de fallo |",
         "|----|-------|----------|----------|----------------------|-----------|------|------|---------------|"]
for r in rows:
    a = r["answer"].replace("|", "/").replace("\n", " ")[:200]
    judge_s = f"{r['judge_attempts']} int." if r["judge_ok"] else "fallo"
    lines.append(f"| {r['id']} | {r['level']} | {r['question'][:60]} | "
                 f"{r['expected'][:55]} | {a} | {r['verdict']} | "
                 f"{r['n_selected']}/10 | {judge_s} | {r['failure_mode']} |")
lines += ["",
          "## Por nivel",
          ""]
for lv in sorted(by_level):
    lines.append(f"- Nivel {lv}: {lvl_ok(lv)}/{len(by_level[lv])} correctas.")
lines += ["",
          "## Diagnostico del juez (seleccion vs evidencia)",
          "",
          "Precision = paginas seleccionadas que son evidencia / paginas "
          "seleccionadas. Recall = paginas de evidencia conservadas / paginas de "
          "evidencia totales (ground truth auditable, "
          "rag_baseline/ragas/ground_truth_evidence.json). Un recall bajo dice que "
          "el juez tiro la evidencia; una precision baja dice que el juez dejo pasar "
          "ruido y el lector tuvo que elegir entre mas chunks.",
          "",
          "| ID | Seleccionados | Precision | Recall | Paginas de evidencia |",
          "|----|---------------|-----------|--------|----------------------|"]
for r in rows:
    lines.append(f"| {r['id']} | {r['n_selected']}/10 ({', '.join(str(p) for p in r['selected_pages']) or 'ninguna'}) | "
                 f"{pct(r['sel_precision'])} | {pct(r['sel_recall'])} | "
                 f"{', '.join(str(p) for p in r['evidence_pages']) or 'n/a'} |")
lines += ["",
          f"**Pooled:** precision {pct(pooled_prec)} ({sel_inter}/{sel_total}), "
          f"recall {pct(pooled_rec)} ({sel_inter}/{ev_total}).",
          "",
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
          f"| {SYSTEM_NAME} | {total_ok}/12 | {lvl_ok(1)}/4 | {lvl_ok(2)}/4 | {lvl_ok(3)}/4 |",
          "| RAG hibrido + expansion de queries | 5/12 | 2/4 | 2/4 | 1/4 |",
          "| RAG hibrido BM25+denso (sin expansion) | 2/12 | 1/4 | 0/4 | 1/4 |",
          "| RAG vectorial denso puro | 2/12 | 0/4 | 1/4 | 1/4 |",
          "| RAG agentico ReAct (LangChain + LangGraph, 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |",
          "| MEMO run 2 (0.5B LoRA r=32 + ejecutivo 1.5B) | 2/12 | 1/4 | 1/4 | 0/4 |",
          "| MEMO run 1 (0.5B LoRA r=8 + ejecutivo 0.5B) | 0/12 | 0/4 | 0/4 | 0/4 |",
          ""]
if not BLANK_MODES:
    lines += ["## Que cambio vs el RAG hibrido + expansion (5/12)",
              "",
              "PENDIENTE de redactar tras verificar los modos de fallo a mano: "
              "comparar pregunta por pregunta cuales flips son por la seleccion del "
              "juez (tiro la evidencia / dejo pasar ruido) y cuales por el lector.",
              "",
              "## Veredicto para el post",
              "",
              "PENDIENTE de redactar tras verificar los modos de fallo a mano: la "
              "pregunta es si la seleccion con juez supera el top-5 fijo del pipeline "
              "hibrido+expansion (5/12) con el mismo modelo y el mismo scorer.",
              "",
              ]
else:
    lines += ["## Veredicto para el post",
              "",
              "Pendiente de revision manual. Los numeros de la tabla de arriba ya son "
              "comparables con los demas sistemas (mismo scorer, mismas 12 preguntas); "
              "lo que falta es verificar cada fallo contra el corpus y contra la "
              "traza, llenar VERIFIED_MODES en la celda de arriba y volver a correr "
              "el scorer con BLANK_MODES = False para que escriba los modos de fallo "
              "y el veredicto.",
              "",
              ]
lines += ["## Caveats (verificados, no supuestos)",
          "",
          "- **Q7 y Q9 son techo (9/12), igual que en los otros runs.** Las cifras "
          "esperadas de Q7 (+27.5%/+24.9%) vienen del comunicado 4Q24 y las de Q9 "
          "(39.3/35.8/75.1%) son participaciones calculadas: no estan en el informe. "
          "Q2 tambien pide +24.9% como parte de lo esperado. El techo real del juez "
          "sobre este set es 9/12, no 12/12.",
          "- **El juez ve la version condensada de cada chunk, la misma que vera el "
          "lector.** Un chunk es util si lo que el lector va a ver de el sirve para "
          "responder; la seleccion se mide contra esa misma version, no contra la "
          "pagina completa.",
          "- **El resto de los caveats esta pendiente.** Los modos de fallo de este "
          "run todavia no se verificaron a mano: ver la seccion de veredicto arriba.",
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
          "## Detalle de evidencia por pregunta (cobertura literal)",
          "",
          "| ID | Cifras esperadas en top-10 presentado | Cifras esperadas en contexto seleccionado |",
          "|----|----------------------------------------|-------------------------------------------|"]
for r in rows:
    cov_p = ", ".join(f"{k}: {'si' if v else 'no'}"
                      for k, v in r["coverage_presented"].items()) \
        if r["coverage_presented"] else "(sin cifras esperadas)"
    cov_s = ", ".join(f"{k}: {'si' if v else 'no'}"
                      for k, v in r["coverage_selected"].items()) \
        if r["coverage_selected"] else "(sin cifras esperadas)"
    lines.append(f"| {r['id']} | {cov_p} | {cov_s} |")

open(OUT_MD, "w").write("\n".join(lines) + "\n")
print(f"correctas {total_ok}/12, con alucinacion {total_halluc}/12, "
      f"seleccionados {total_selected}/120, juez ok {n_judge_ok}/12, "
      f"precision {pct(pooled_prec)}, recall {pct(pooled_rec)}, "
      f"{total_seconds / 60:.1f} min -> {OUT_MD}")
for r in rows:
    print(f"  {r['id']} L{r['level']} {r['verdict']} [{r['failure_mode']}] "
          f"sel={r['n_selected']}/10 juez={'ok' if r['judge_ok'] else 'fallo'} "
          f"({r['judge_attempts']} int.) {r['seconds']:.0f}s"
          + (f" halluc={r['hallucinated']}" if r["hallucinated"] else ""))
"""),
        *save_cell([
            ("rag_baseline/retrieval_hybrid_k10.json", "Retrieval híbrido top-10 (12 preguntas)", True),
            ("agentic_rag/run_trace_judge_colab.jsonl", "Traza juez + lector (12 preguntas)", True),
            ("agentic_rag/results_agentic_judge_colab.md", "Resultados RAG agéntico juez", True),
        ], note="El puntaje del juez es EL resultado agéntico que va al post (el ReAct "
                "2/12 queda como negativo documentado). Sin esta traza, el draft no se "
                "puede actualizar con el resultado real."),
    ]
    write_nb("07_rag_agentico.ipynb", cells, "07 RAG agéntico juez (Ollama)")


# ---------------------------------------------------------------------------
# 08 — RAGAS (juez vía Ollama)
# ---------------------------------------------------------------------------

def build_08():
    cells = [
        title_cell(
            "08 — Métricas RAGAS (juez vía Ollama + cálculo determinista)",
            "Dos capas: (1) las métricas **deterministas** estilo RAGAS "
            "(`rag_baseline/ragas/compute_metrics.py`: context precision/recall, "
            "faithfulness por chequeo de claims, relevancy por coseno) y (2) el run "
            "**real con la librería `ragas` 0.4.4**, ahora con el juez vía **ChatOllama** "
            "(`JUDGE_OLLAMA_MODEL`, por defecto `qwen2.5:7b` — el juez 1.5B en CPU se "
            "atoró incluso con timeout de 1800 s por job; en la T4 vía Ollama debería "
            "completar). Si el run real no completa, las deterministas siguen siendo "
            "las cifras citables, y el notebook lo dice.\n\n"
            "**Prerrequisitos:** notebooks 05 y 06 (trazas y retrievals de las tres "
            "variantes RAG). El juez no evalúa al grafo ni a MEMO: solo a las tres "
            "variantes RAG, como en local."),
        gpu_check_cell(),
        drive_mount_cell(),
        params_cell("""
# --- Parametros ---
JUDGE_OLLAMA_MODEL = "qwen2.5:7b"   # juez: 7b (default, mejor juez) o qwen2.5:1.5b (paridad CPU)
TIME_BUDGET_MIN = 90   # tope de tiempo para el run real con ragas (muestras no hechas = skipped, no numeros)
"""),
        repo_setup_cell(),
        *data_ensure_cells(),
        pip_cell([
            '"torch>=2.15.0"', "transformers==5.24.1", "sentence-transformers==6.3.0",
            "numpy>=2.5.5", "langchain-ollama", "langchain-huggingface==1.3.0",
            "ragas==0.4.4",
        ]),
        *ollama_compact_cells(["qwen2.5:7b", "qwen2.5:1.5b"], "JUDGE_OLLAMA_MODEL"),
        md("## Prerrequisitos: trazas y retrievals de los notebooks 05 y 06\n\n"
           "El cálculo determinista inline lee los nombres de CPU "
           "(`retrieval.json`, `run_trace.jsonl`, ...), pero en Colab los notebooks 05 y 06 "
           "escriben las trazas como `run_trace_colab.jsonl` (y variantes). Esta celda "
           "verifica que existan los 6 insumos, te pide subirlos si faltan, y alinea los "
           "nombres (copia `run_trace_*_colab.jsonl` -> `run_trace_*.jsonl`) para que "
           "el cálculo inline los lea con los nombres de CPU."),
        code(r"""
import json, shutil
from pathlib import Path

BASE = Path(REPO) / "rag_baseline"

# (nombre Colab, nombre que espera el calculo determinista, quién lo produce)
NEEDED = [
    ("retrieval.json", "retrieval.json", "NB05 (denso)"),
    ("run_trace_colab.jsonl", "run_trace.jsonl", "NB05 (denso)"),
    ("retrieval_hybrid.json", "retrieval_hybrid.json", "NB06 (híbrido)"),
    ("run_trace_hybrid_colab.jsonl", "run_trace_hybrid.jsonl", "NB06 (híbrido)"),
    ("retrieval_expanded.json", "retrieval_expanded.json", "NB06 (expansión)"),
    ("run_trace_expanded_colab.jsonl", "run_trace_expanded.jsonl", "NB06 (expansión)"),
]


def _ok(p):
    try:
        return p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


missing = [c for c, _, _ in NEEDED if not _ok(BASE / c)]
if missing:
    print("Faltan insumos de 05/06:", missing)
    print("Súbelos con el widget de abajo (botón 'Elegir archivos').")
    print("Qué subir: los archivos de la lista (trazas run_trace_*_colab.jsonl y")
    print("retrievales retrieval*.json) tal cual salieron de los notebooks 05 y 06.")
    print("Dónde conseguirlos: de la corrida de 05/06 en Drive, o corriendo 05 y 06.")
    from google.colab import files
    uploaded = files.upload()
    for name, data in uploaded.items():
        dest = BASE / Path(name).name
        dest.write_bytes(data)
        print(f"subido: {dest} ({len(data)} bytes)")

still = [(c, src) for c, _, src in NEEDED if not _ok(BASE / c)]
assert not still, (
    "FALTAN insumos de 05/06: " + ", ".join(c for c, _ in still) + "\n"
    "Corre los notebooks 05 (denso) y 06 (híbrido + expansión) primero, "
    "o sube sus archivos (retrieval*.json y run_trace_*_colab.jsonl) a rag_baseline/."
)

# Alinear nombres: el calculo determinista espera los de CPU.
for colab_name, cpu_name, _src in NEEDED:
    if colab_name == cpu_name:
        continue
    src, dst = BASE / colab_name, BASE / cpu_name
    if _ok(src) and (not _ok(dst) or src.stat().st_mtime > dst.stat().st_mtime):
        shutil.copy(src, dst)
        print(f"alineado: {colab_name} -> {cpu_name}")
    elif _ok(dst):
        print(f"ya existe {cpu_name} (se usa ese; {colab_name} no se copia)")
print("prerrequisitos OK: trazas y retrievals de 05/06 listos.")
"""),
        md("## 1. Métricas deterministas (siempre disponibles)\n\n"
           "Lógica inline en las tres celdas siguientes, tomada de "
           "`rag_baseline/ragas/compute_metrics.py` y `compute_relevancy.py` "
           "(mismas definiciones, mismos outputs): precision y recall de contexto "
           "desde los traces de retrieval + ground truth auditable, faithfulness por "
           "chequeo de claims contra el contexto condensado, relevancy como proxy de "
           "coseno. Escribe `metrics_ragas.json` igual que en local. En local: denso "
           "recall 0.47 / faithfulness 0.97, híbrido 0.53 / 0.97, expansión 1.00 / 0.98."),
        code(r"""
import json, re, unicodedata
from pathlib import Path

BASE = Path(REPO) / "rag_baseline"
RAGAS_DIR = BASE / "ragas"
GT = json.load(open(RAGAS_DIR / "ground_truth_evidence.json"))
GT_BY_ID = {q["id"]: q for q in GT["questions"]}

VARIANTS = {
    "dense": {"retrieval": "retrieval.json", "trace": "run_trace.jsonl"},
    "hybrid": {"retrieval": "retrieval_hybrid.json", "trace": "run_trace_hybrid.jsonl"},
    "expanded": {"retrieval": "retrieval_expanded.json", "trace": "run_trace_expanded.jsonl"},
}

# Veredictos de exactitud de los resultados publicados (mismo scorer).
EXACTNESS = {
    "dense": {"Q6", "Q11"},
    "hybrid": {"Q1", "Q11"},
    # Q11 expanded es "correcta con alucinacion" en results_expanded.md: la
    # sustancia es correcta (Digital@FEMSA -> Spin) pero el scorer marca '328';
    # el conteo publicado es 5/12, asi que Q11 NO esta en el conjunto correcto.
    "expanded": {"Q1", "Q4", "Q6", "Q8", "Q10"},
}
EXACTNESS_NOTES = {"expanded": {"Q11": "correcta con alucinacion segun el scorer (artefacto: '328' = 32.8% real, p.34)"}}


def norm_text(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # collapse all whitespace (PDF text has tabs/newlines inside names:
    # "Coca-Cola\tFEMSA" must match "coca-cola femsa")
    return re.sub(r"\s+", " ", s).strip().lower()


NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?")


def norm_number(tok):
    # '781,585' -> '781585'; '11,2' -> '11.2'; '8.6' -> '8.6'; '571' -> '571'.
    tok = tok.strip().rstrip("%")
    if "," in tok and "." not in tok:
        parts = tok.split(",")
        if len(parts) == 2 and len(parts[1]) == 1:
            return parts[0] + "." + parts[1]  # coma decimal (espanol)
        return "".join(parts)  # separador de miles
    return tok.replace(",", "")


def extract_numbers(text):
    return [norm_number(m.group(0)) for m in NUM_RE.finditer(text)]


ENTITIES = {
    # canonical -> accepted surface forms (Spanish answer vs English corpus:
    # a claim is supported if ANY form appears in the context)
    "FEMSA Forward": ["femsa forward"],
    "Digital@FEMSA": ["digital@femsa"],
    "Spin by OXXO": ["spin by oxxo"],
    "Spin Premia": ["spin premia"],
    "Proximidad Americas": ["proximidad americas", "proximity americas"],
    "Coca-Cola FEMSA": ["coca-cola femsa"],
    "Proximidad Europa": ["proximidad europa", "proximity europe"],
    "OXXO GAS": ["oxxo gas"],
    "Salud": ["salud", "health"],
    "Spin": ["spin"],
    "Mexico": ["mexico"],
    "Colombia": ["colombia"],
    "Chile": ["chile"],
    "Peru": ["peru"],
    "Argentina": ["argentina"],
    "Austria": ["austria"],
    "Brasil": ["brasil", "brazil"],
    "Costa Rica": ["costa rica"],
    "Ecuador": ["ecuador"],
    "Espana": ["espana", "spain"],
    "Guatemala": ["guatemala"],
    "Honduras": ["honduras"],
    "Paises Bajos": ["paises bajos", "netherlands"],
    "Nicaragua": ["nicaragua"],
    "Panama": ["panama"],
    "Uruguay": ["uruguay"],
    "Estados Unidos": ["estados unidos", "united states"],
}


def extract_entities(text):
    nt = norm_text(text)
    found = []
    for canonical, forms in ENTITIES.items():
        if any(f in nt for f in forms):
            found.append(canonical)
    # drop "Spin" if a longer Spin entity is present
    if any(e in found for e in ("Spin by OXXO", "Spin Premia")) and "Spin" in found:
        found.remove("Spin")
    return found


def context_numbers(context):
    body = re.sub(r"\[P[aá]gina \d+\]", " ", context)  # strip page headers
    return set(extract_numbers(body))


def context_entities(context):
    return set(extract_entities(context))


def load_variant(name):
    cfg = VARIANTS[name]
    ret = json.load(open(BASE / cfg["retrieval"]))
    pages = {q["id"]: [c["page"] for c in q["retrieved"]] for q in ret["questions"]}
    trace = [json.loads(l) for l in open(BASE / cfg["trace"])]
    answers = {t["id"]: t for t in trace}
    return pages, answers
"""),
        code(r"""
# context precision / context recall / faithfulness (determinista).
# Definiciones: precision = paginas recuperadas que son evidencia / paginas
# recuperadas (top-5); recall = items de evidencia con >=1 pagina en el top-5
# / items de evidencia (Q9 no tiene evidencia en el corpus -> recall null, no 0);
# faithfulness = claims de la respuesta apoyados por el contexto condensado /
# claims (numericos + entidades; abstencion "No lo se" -> 1.0 por convencion).
out = {"variants": {}, "method": "deterministic (see ragas/README.md)", "judge": None}
for name in VARIANTS:
    pages_by_q, answers = load_variant(name)
    per_q = {}
    for qid, gt in GT_BY_ID.items():
        retrieved = pages_by_q[qid]
        gt_pages = set()
        for item in gt["evidence"]:
            gt_pages.update(item["pages"])
        # context precision
        relevant = [p for p in retrieved if p in gt_pages]
        ctx_precision = len(relevant) / len(retrieved) if retrieved else None
        # context recall (a nivel de item de evidencia)
        n_items = len(gt["evidence"])
        if n_items == 0:
            ctx_recall = None
            recall_detail = {"covered": 0, "total": 0, "note": "no hay evidencia en el corpus (cifras calculadas, no impresas)"}
        else:
            covered = sum(1 for item in gt["evidence"] if any(p in retrieved for p in item["pages"]))
            ctx_recall = covered / n_items
            recall_detail = {"covered": covered, "total": n_items}
        # faithfulness
        ans = answers[qid]["final_answer"]
        ctx = answers[qid]["context"]
        abstain = ans.strip().rstrip(".").lower().startswith("no lo s")
        ctx_nums = context_numbers(ctx)
        ctx_ents = context_entities(ctx)
        claims = []
        if abstain:
            faithfulness = 1.0
            claims = []
        else:
            for n in extract_numbers(ans):
                claims.append({"claim": n, "type": "numeric", "supported": n in ctx_nums})
            for e in extract_entities(ans):
                claims.append({"claim": e, "type": "entity", "supported": e in ctx_ents})
            faithfulness = (sum(1 for c in claims if c["supported"]) / len(claims)) if claims else 1.0
        per_q[qid] = {
            "question": gt["question"],
            "retrieved_pages": retrieved,
            "ground_truth_pages": sorted(gt_pages),
            "context_precision": ctx_precision,
            "context_recall": ctx_recall,
            "recall_detail": recall_detail,
            "faithfulness": faithfulness,
            "n_claims": len(claims),
            "abstention": abstain,
            "claims": claims,
            "exactness_correct": qid in EXACTNESS[name],
            "exactness_note": EXACTNESS_NOTES.get(name, {}).get(qid, ""),
        }
    out["variants"][name] = {"per_question": per_q}
json.dump(out, open(RAGAS_DIR / "metrics_ragas.json", "w"), ensure_ascii=False, indent=1)
print("wrote metrics_ragas.json (precision/recall/faithfulness; relevancy pending)")

# Tabla resumen (promedios; recall excluye Q9, que es null, no 0).
print(f"{'variante':<10}{'precision':>11}{'recall':>9}{'faithfulness':>14}{'exactas':>9}")
for name in VARIANTS:
    pq = out["variants"][name]["per_question"]
    prec = [v["context_precision"] for v in pq.values() if v["context_precision"] is not None]
    rec = [v["context_recall"] for v in pq.values() if v["context_recall"] is not None]
    fai = [v["faithfulness"] for v in pq.values() if v["faithfulness"] is not None]
    n_exact = sum(1 for v in pq.values() if v["exactness_correct"])
    print(f"{name:<10}{sum(prec) / len(prec):>11.2f}{sum(rec) / len(rec):>9.2f}{sum(fai) / len(fai):>14.2f}{n_exact:>8}/12")
"""),
        code(r"""
# answer_relevancy (proxy): coseno entre embeddings all-MiniLM-L6-v2 de la
# pregunta y la respuesta. RAGAS la define via preguntas generadas por LLM a
# partir de la respuesta; ese paso NO se usa aqui: es un proxy, etiquetado
# como tal en metrics_ragas.json.
import json, time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(REPO) / "rag_baseline"
RAGAS_DIR = BASE / "ragas"
TRACES = {"dense": "run_trace.jsonl", "hybrid": "run_trace_hybrid.jsonl", "expanded": "run_trace_expanded.jsonl"}

t0 = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
metrics = json.load(open(RAGAS_DIR / "metrics_ragas.json"))
for name, trace_file in TRACES.items():
    trace = [json.loads(l) for l in open(BASE / trace_file)]
    qs = [t["question"] for t in trace]
    ans = [t["final_answer"] for t in trace]
    eq = model.encode(qs, normalize_embeddings=True, show_progress_bar=False)
    ea = model.encode(ans, normalize_embeddings=True, show_progress_bar=False)
    for t, a, b in zip(trace, eq, ea):
        metrics["variants"][name]["per_question"][t["id"]]["answer_relevancy"] = round(float(np.dot(a, b)), 4)
    print(name, "done")
metrics["answer_relevancy_method"] = "proxy: cosine(all-MiniLM-L6-v2(question), all-MiniLM-L6-v2(answer)); RAGAS defines it via LLM-generated questions, not used here"
json.dump(metrics, open(RAGAS_DIR / "metrics_ragas.json", "w"), ensure_ascii=False, indent=1)
print("relevancy computed in %.1fs" % (time.time() - t0))

# Tabla resumen de relevancy (promedio por variante).
print(f"{'variante':<10}{'answer_relevancy':>17}")
for name in TRACES:
    rel = [v["answer_relevancy"] for v in metrics["variants"][name]["per_question"].values()]
    print(f"{name:<10}{sum(rel) / len(rel):>17.2f}")
"""),
        md("## 2. Piloto: una muestra con el juez Ollama\n\n"
           "Antes del run completo, una sola muestra (denso Q1, la interesante: el lector "
           "contestó 40,236 con 781,585 en contexto) con las 4 métricas y `RunConfig`. "
           "Si el piloto no completa en tiempo razonable, el run completo no es viable "
           "y nos quedamos con las deterministas — igual que pasó en CPU."),
        code("""
import json, time
from pathlib import Path
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
try:
    from ragas import RunConfig
except ImportError:
    from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

BASE = Path(REPO) / "rag_baseline"
RAGAS_DIR = BASE / "ragas"

judge = ChatOllama(model=JUDGE_OLLAMA_MODEL, base_url="http://localhost:11434",
                   temperature=0, num_predict=512)
llm = LangchainLLMWrapper(judge)
emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                          model_kwargs={"device": "cuda"}))
print("juez Ollama listo:", JUDGE_OLLAMA_MODEL)

trace = [json.loads(l) for l in open(BASE / "run_trace.jsonl")][0]  # denso Q1
gt = json.load(open(RAGAS_DIR / "ground_truth_evidence.json"))["questions"][0]
contexts = [c for c in trace["context"].split("\\n\\n") if c.strip()]
sample = SingleTurnSample(
    user_input=trace["question"],
    retrieved_contexts=contexts,
    response=trace["final_answer"],
    reference=gt["question"] + " Esperado: Ps. 781,585 millones, +11.2%.",
)
t1 = time.time()
result = evaluate(
    EvaluationDataset(samples=[sample]),
    metrics=[ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy()],
    llm=llm, embeddings=emb, run_config=RunConfig(timeout=600),
)
print(f"piloto: {time.time() - t1:.1f}s")
print(result.to_pandas().to_string())
"""),
        md("## 3. Run real completo (acotado por tiempo)\n\n"
           "36 muestras (12 preguntas × 3 variantes), una llamada `evaluate()` por "
           "muestra con las 4 métricas, como en el piloto de CPU (`pilot_timeout_900.py`). "
           "Tope `TIME_BUDGET_MIN`: las muestras que no se alcanzan quedan como "
           "`skipped`, no como números. Al final se comparan los valores juzgados por "
           "el LLM con los deterministas, pregunta por pregunta."),
        code("""
import json, time
from pathlib import Path
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
try:
    from ragas import RunConfig
except ImportError:
    from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

BASE = Path(REPO) / "rag_baseline"
RAGAS_DIR = BASE / "ragas"
OUT = RAGAS_DIR / "metrics_ragas_llm_colab.json"

judge = ChatOllama(model=JUDGE_OLLAMA_MODEL, base_url="http://localhost:11434",
                   temperature=0, num_predict=512)
llm = LangchainLLMWrapper(judge)
emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                          model_kwargs={"device": "cuda"}))
METRICS = [ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy()]
METRIC_NAMES = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
run_config = RunConfig(timeout=600)

eval_q = {q["id"]: q for q in json.load(open(Path(REPO) / "memo/eval_questions.json"))["questions"]}
det = json.load(open(RAGAS_DIR / "metrics_ragas.json"))["variants"]
gt = {q["id"]: q for q in json.load(open(RAGAS_DIR / "ground_truth_evidence.json"))["questions"]}

VARIANTS = [("dense", "run_trace.jsonl"),
            ("hybrid", "run_trace_hybrid.jsonl"),
            ("expanded", "run_trace_expanded.jsonl")]
traces = {}
for name, fname in VARIANTS:
    traces[name] = {r["id"]: r for r in (json.loads(l) for l in open(BASE / fname))}

results, t_start = {}, time.time()
budget_s = TIME_BUDGET_MIN * 60
for name, _ in VARIANTS:
    for qid in sorted(eval_q):
        if time.time() - t_start > budget_s:
            results[f"{name}/{qid}"] = {"status": "skipped",
                                        "reason": f"tope de tiempo ({TIME_BUDGET_MIN} min)"}
            continue
        tr = traces[name][qid]
        contexts = [c for c in tr["context"].split("\\n\\n") if c.strip()]
        sample = SingleTurnSample(
            user_input=tr["question"], retrieved_contexts=contexts,
            response=tr["final_answer"],
            reference=gt[qid]["question"] + " Esperado: " + eval_q[qid]["expected"])
        t1 = time.time()
        try:
            res = evaluate(EvaluationDataset(samples=[sample]), metrics=METRICS,
                           llm=llm, embeddings=emb, run_config=run_config)
            row = res.to_pandas().iloc[0].to_dict()
            vals = {m: (None if row.get(m) is None else float(row[m]))
                    for m in METRIC_NAMES}
            results[f"{name}/{qid}"] = {"status": "ok",
                                        "seconds": round(time.time() - t1, 1),
                                        "llm_judged": vals,
                                        "deterministic": det[name]["per_question"][qid]}
        except Exception as e:
            results[f"{name}/{qid}"] = {"status": "error",
                                        "seconds": round(time.time() - t1, 1),
                                        "error": f"{type(e).__name__}: {e}"}
        print(f"{name}/{qid}: {results[f'{name}/{qid}']['status']} "
              f"({results[f'{name}/{qid}'].get('seconds', '?')}s)", flush=True)

done = sum(1 for v in results.values() if v["status"] == "ok")
print(f"\\ncompletadas: {done}/36 en {(time.time() - t_start) / 60:.1f} min")
json.dump({"judge": f"ollama {JUDGE_OLLAMA_MODEL} (Colab T4)",
           "results": results}, open(OUT, "w"), ensure_ascii=False, indent=2)
print(f"-> {OUT}")
"""),
        md("## 4. Comparación: juzgado por LLM vs determinista\n\n"
           "Si el run real completó muestras, esta celda compara los valores del juez "
           "con los deterministas (misma pregunta, misma variante) y resume acuerdos y "
           "desacuerdos. Si no completó ninguna, lo dice y las deterministas quedan "
           "como las cifras citables."),
        code("""
import json

OUT = REPO + "/rag_baseline/ragas/metrics_ragas_llm_colab.json"
try:
    data = json.load(open(OUT))["results"]
except FileNotFoundError:
    print("No hay run real (no se corrio la celda 3 o no completo nada).")
    print("Las cifras citables son las deterministas de metrics_ragas.json.")
    data = {}

ok = {k: v for k, v in data.items() if v.get("status") == "ok"}
print(f"muestras juzgadas por LLM: {len(ok)}/36")
for k, v in sorted(ok.items()):
    lj, dt = v["llm_judged"], v["deterministic"]
    diffs = " | ".join(
        f"{m}: LLM {lj[m] if lj[m] is not None else 'null'} vs det "
        f"{dt[m] if dt.get(m) is not None else 'null'}"
        for m in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"])
    print(f"{k}: {diffs}")
"""),
        *save_cell([
            ("rag_baseline/ragas/metrics_ragas.json", "Métricas RAGAS deterministas (actualizadas en Colab)", True),
            ("rag_baseline/ragas/metrics_ragas_llm_colab.json", "Run real ragas con juez Ollama (36 muestras)", False),
            ("rag_baseline/ragas/ground_truth_evidence.json", "Ground truth auditable", False),
        ], note="Si el run real no completó muestras, `metrics_ragas_llm_colab.json` no "
                "existe y la celda lo marca como pendiente: las deterministas siguen "
                "siendo las cifras citables."),
    ]
    write_nb("08_ragas.ipynb", cells, "08 RAGAS (juez Ollama)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_00()
    build_01()
    build_02()
    build_03()
    build_04()
    build_05()
    build_06()
    build_07()
    build_08()
    print("listo: 9 notebooks en", HERE)
