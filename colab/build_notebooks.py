#!/usr/bin/env python3
"""Genera los notebooks de Google Colab del experimento FEMSA 2024.

Uso:  python3 build_notebooks.py   (escribe colab/*.ipynb).
El README.md de colab/ se mantiene a mano (no lo genera este script).

Cada notebook es autocontenido: monta Drive, clona/actualiza el repo,
instala sus dependencias y (donde hace falta) instala/arranca Ollama con
celdas idempotentes. Los notebooks son envoltorios delgados sobre el
codigo del repo (pipeline/, memo/, rag_baseline/, agentic_rag/): los
scripts se corren como subprocesos o se importan, salvo el entrenamiento
MEMO y la inferencia, que van adaptados a GPU en el propio notebook
(los scripts originales cargan el modelo en CPU sin device_map).

Datos procesados: cada notebook que consume chunks.jsonl lleva su celda
de datos (Drive -> subir con el widget de Colab -> generar con extract.py
como ultimo recurso). Los insumos que produce una etapa anterior (adaptador
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

_DATA_MD = (
    "## Datos procesados: primero en Drive, si no, súbelos\n\n"
    "Este notebook necesita `pipeline/data/chunks.jsonl` (las 151 páginas del informe "
    "extraídas). El flujo es:\n\n"
    "1. **Buscarlo en Drive** (ruta estándar, bajo el repo clonado).\n"
    "2. **Si no está, subirlo** con el widget de Colab de la celda siguiente: el archivo "
    "es `chunks.jsonl`. Lo consigues de una corrida anterior en Drive o generándolo del "
    "repo con `make extract` (`python pipeline/extract.py`).\n"
    "3. **Solo si no lo subes**, generarlo aquí corriendo `extract.py` (descarga el PDF "
    "~19 MB y extrae las 151 páginas, ~1–2 min). Es el último recurso, no el camino "
    "principal.\n\n"
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
    print("Dónde conseguirlo: de una corrida anterior en Drive, o del repo con `make extract`.")
    print("Si no lo subes, el notebook lo genera con extract.py (último recurso).")
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
        print("Sin subida válida: último recurso, generar con extract.py...")
        print("(descarga el PDF ~19 MB y extrae las 151 páginas, ~1-2 min)")
        try:
            import pypdf
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "pypdf>=7.1.1"], check=True)
        r = subprocess.run([sys.executable, "extract.py"], cwd=REPO + "/pipeline")
        print("extract.py exit:", r.returncode)
        assert _chunks_ok(CHUNKS), "No se pudo obtener chunks.jsonl."
    print(f"chunks.jsonl listo: {CHUNKS}")
"""


def data_ensure_cells():
    """Datos procesados, en orden: Drive -> subir con widget -> generar (último recurso).
    La lleva cada notebook que consume chunks.jsonl (01-08)."""
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
    cells = [
        title_cell(
            "01 — Setup y extracción del informe FEMSA 2024",
            "Prepara todo lo demás: monta Drive, clona el repo, instala dependencias con "
            "las versiones fijadas del proyecto, descarga el informe anual FEMSA 2024 "
            "(edición en inglés, 151 páginas) y extrae el texto por página + las tripletas "
            "por reglas. **No necesita Ollama** (la extracción es determinista, sin LLM).\n\n"
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
           "`pipeline/load_relate.py` importa `neo4j` (lazy import): sin el paquete, la "
           "carga por Bolt falla. Se instala aquí, antes de cargar el grafo."),
        pip_cell(["neo4j>=6.4.1"]),
        *data_ensure_cells(),
        md("## Generar el Cypher de carga\n\n"
           "Corre `pipeline/build_cypher.py` tal cual: genera `load_graph.cypher` "
           "(MERGE idempotentes) desde `triples_seed.json`. No se cargan tripletas "
           "extraídas sin revisar — esa regla sigue vigente aquí."),
        code("""
import subprocess, sys
r = subprocess.run([sys.executable, "build_cypher.py"], cwd=REPO + "/pipeline")
print("exit:", r.returncode)
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
           "Corre `pipeline/load_relate.py` tal cual (driver `neo4j` de Python, Bolt a "
           "localhost:7687). Esperado, verificado: **27 nodos, 39 relaciones** "
           "(31 semilla + 8 `CRECIO_EN` derivadas). La carga es idempotente: correrla "
           "dos veces no duplica nada."),
        code("""
import subprocess, sys
r = subprocess.run([sys.executable, "load_relate.py"], cwd=REPO + "/pipeline")
print("exit:", r.returncode)
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
            "Genera el dataset de reflexiones **expandido** (objetivo 1,500–2,000 pares, "
            "con más variación de paráfrasis por hecho que el run 2) y entrena MEMORY "
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
           "Corre `memo/generate_reflections.py` tal cual: lee los chunks, genera los "
           "pares curados + programáticos, excluye las 12 preguntas de evaluación "
           "(Jaccard ≥ 0.80) y verifica cada cifra contra el texto citado. Salida: "
           "`memo/reflection_qa.jsonl` (446 pares)."),
        code("""
import subprocess, sys
r = subprocess.run([sys.executable, "generate_reflections.py"], cwd=REPO + "/memo")
print("exit:", r.returncode)
"""),
        md("## 2. Dataset v2: 446 → 1,414 (paráfrasis de pregunta)\n\n"
           "Corre `memo/augment_v2.py` tal cual: paráfrasis solo de la pregunta "
           "(la respuesta se copia byte-idéntica), con chequeo de fuga por Jaccard "
           "contra las 12 preguntas de evaluación y verificación de que ninguna "
           "respuesta cambió sus números."),
        code("""
import subprocess, sys
r = subprocess.run([sys.executable, "augment_v2.py"], cwd=REPO + "/memo")
print("exit:", r.returncode)
"""),
        md("## 3. Dataset v3: más variación por hecho (objetivo 1,500–2,000)\n\n"
           "Una segunda pasada de aumento, en el notebook: reutiliza los helpers de "
           "`augment_v2` (`tokens`, `jaccard`, `candidates`) y añade más patrones de "
           "paráfrasis (más marcos y más trueques). Mismas reglas: solo cambia la "
           "pregunta, la respuesta es byte-idéntica; Jaccard ≥ 0.80 contra las 12 de "
           "evaluación descarta la paráfrasis; cada cifra clave debe aparecer ≥3 veces; "
           "sin duplicados (pregunta, respuesta). Si el total queda por debajo de 1,500, "
           "la celda lo dice y no inventa pares para llegar."),
        code(r"""
import json, sys
sys.path.insert(0, REPO + "/memo")
import augment_v2 as aug

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
        out.append(aug.framing(q, prefix))
    for old, new in EXTRA_SWAPS:
        s = aug.swap(q, old, new)
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
eval_toks = [aug.tokens(q["question"]) for q in eval_qs]

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
        if max((aug.jaccard(aug.tokens(c), et) for et in eval_toks), default=0) >= aug.JACCARD_MAX:
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

# Multiplicidad de cifras clave (mismo criterio que augment_v2).
from collections import Counter
key_nums = set()
for q in eval_qs:
    key_nums.update(aug.norm_num_tokens(str(n)) for n in q.get("expected_numbers", []))
fact_counts = Counter()
for r in out_rows:
    for n in aug.norm_num_tokens(r["answer"]) & key_nums:
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
"""),
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
            ("memo/reflection_qa_v3.jsonl", "Dataset v3 (1,500–2,000 pares)", True),
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
            "12 preguntas y con el mismo scorer (`memo/evaluate.py`).\n\n"
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
           "`RUN_CONTROL`). EXECUTIVE = ChatOllama contra el servidor local. Se "
           "reutilizan los prompts y la descomposición por reglas de `memo/executive_v2.py`."),
        code("""
import json, sys, torch
sys.path.insert(0, REPO + "/memo")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from executive_v2 import DECOMP_SYSTEM, DECOMP_EXAMPLES, rule_decompose

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

exec_llm = ChatOllama(model=EXEC_OLLAMA_MODEL, base_url="http://localhost:11434",
                      temperature=0, num_predict=150)
print("EXECUTIVE (Ollama):", EXEC_OLLAMA_MODEL)


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
        md("## 2. Protocolo MEMO (adaptado de `executive_v2.py`)\n\n"
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
        md("## 5. Puntuar con el mismo scorer\n\n"
           "Corre `memo/evaluate.py` tal cual (mismo scorer, misma regla Q3 con 1,596, "
           "mismas keywords). Compara con el run 2 en CPU (2/12) y el run 1 (0/12)."),
        code("""
import subprocess, sys
for trace, out in [(TRACE_OUT, REPO + "/memo/results_colab.md"),
                   (TRACE_CONTROL_OUT, REPO + "/memo/results_colab_control.md")]:
    if RUN_CONTROL or "control" not in trace:
        r = subprocess.run([sys.executable, "evaluate.py", "--trace", trace,
                            "--eval", REPO + "/memo/eval_questions.json",
                            "--qa", REPO + "/memo/reflection_qa_v3.jsonl",
                            "--out", out], cwd=REPO + "/memo")
        print("exit:", r.returncode, "->", out)
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
import json, re, sys
sys.path.insert(0, REPO + "/memo")
from evaluate import digits

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

READER_CELL = r"""
import json, sys
sys.path.insert(0, REPO + "/rag_baseline")
from answer import SYSTEM, condense
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

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


def build_05():
    cells = [
        title_cell(
            "05 — RAG clásico (baseline denso, lector vía Ollama)",
            "Retrieval denso top-5 con all-MiniLM-L6-v2 + lector vía **ChatOllama** "
            "(`READER_OLLAMA_MODEL`: `qwen2.5:1.5b` para paridad con el baseline en CPU, "
            "`qwen2.5:7b` para el run con modelo más grande) + mismo scorer. El retrieval "
            "corre igual que en local (`rag_baseline/retrieve.py`); el lector es el "
            "mismo prompt y el mismo `condense()` de `answer.py`, solo cambia el modelo "
            "que contesta (Ollama en vez de transformers en CPU).\n\n"
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
           "Corre `rag_baseline/retrieve.py` tal cual: embebe las 151 páginas con "
           "all-MiniLM-L6-v2 y recupera top-5 por pregunta, con el diagnóstico de "
           "cobertura de cifras esperadas (separa fallos de retrieval de fallos de lector)."),
        code("""
import subprocess, sys
r = subprocess.run([sys.executable, "retrieve.py", "--k", str(K),
                    "--out", RETRIEVAL_FILE], cwd=REPO + "/rag_baseline")
print("exit:", r.returncode)
"""),
        md("## 2. Lector vía Ollama\n\n"
           "Mismo sistema y mismo `condense()` que `rag_baseline/answer.py`: el contexto "
           "son los 5 chunks recuperados (ventanas de ~1,800 caracteres por chunk, 2 "
           "ventanas) y el modelo responde solo desde ellos. La traza tiene el mismo "
           "formato que la de CPU, así que el scorer no cambia."),
        code(READER_CELL),
        md("## 3. Puntuar con el mismo scorer\n\n"
           "Corre `rag_baseline/score.py` tal cual: misma exactitud numérica, misma "
           "definición de alucinación, misma regla Q3 con 1,596. Compara con el baseline "
           "en CPU (2/12: Q6 y Q11)."),
        code("""
import subprocess, sys
r = subprocess.run([sys.executable, "score.py", "--trace", TRACE_OUT,
                    "--retrieval", RETRIEVAL_FILE,
                    "--out", REPO + "/rag_baseline/results_colab.md"],
                   cwd=REPO + "/rag_baseline")
print("exit:", r.returncode)
"""),
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
           "Corre `rag_baseline/expand_queries.py` tal cual: traduce cada pregunta al "
           "inglés y añade términos de dominio del `TERM_MAP` (solo términos presentes "
           "en la pregunta; el script aborta si se cuela una cifra esperada o un nombre "
           "de respuesta). Salida: `expanded_queries.json`."),
        code("""
import subprocess, sys
r = subprocess.run([sys.executable, "expand_queries.py"], cwd=REPO + "/rag_baseline")
print("exit:", r.returncode)
"""),
        md("## 2. Retrieval híbrido (sin expansión) y con expansión\n\n"
           "Dos corridas del mismo retrieval híbrido (BM25 Okapi k1=1.5, b=0.75 + MiniLM, "
           "RRF k=60): una con las preguntas originales y otra con las expandidas."),
        code("""
import subprocess, sys
for script, out in [("retrieve_hybrid.py", "rag_baseline/retrieval_hybrid.json"),
                    ("retrieve_expanded.py", "rag_baseline/retrieval_expanded.json")]:
    r = subprocess.run([sys.executable, script, "--k", str(K), "--out", out],
                       cwd=REPO + "/rag_baseline")
    print(script, "exit:", r.returncode)
"""),
        md("## 3. Lector vía Ollama (las dos variantes)\n\n"
           "La misma función lectora del notebook 05, aplicada a los dos archivos de "
           "retrieval. Cada variante escribe su propia traza, con el mismo formato."),
        code("""
import json, sys
sys.path.insert(0, REPO + "/rag_baseline")
from answer import SYSTEM, condense
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

reader = ChatOllama(model=READER_OLLAMA_MODEL, base_url="http://localhost:11434",
                    temperature=0, num_predict=100)
print("lector RAG (Ollama):", READER_OLLAMA_MODEL)


def run_reader(retrieval_file, trace_out):
    data = json.load(open(retrieval_file))
    with open(trace_out, "w") as f:
        for item in data["questions"]:
            condensed = [condense(r["text"], item["question"]) for r in item["retrieved"]]
            ctx = "\\n\\n".join(f"[Página {r['page']}]\\n{c}"
                              for r, c in zip(item["retrieved"], condensed))
            user = (f"Fragmentos del informe anual de FEMSA 2024:\\n\\n{ctx}\\n\\n"
                    f"Pregunta: {item['question']}")
            ans = reader.invoke([SystemMessage(content=SYSTEM),
                                 HumanMessage(content=user)]).content.strip()
            rec = {"id": item["id"], "level": item["level"],
                   "question": item["question"], "decomposition": "rag",
                   "reader_model": f"ollama {READER_OLLAMA_MODEL}",
                   "retrieved_pages": [r["page"] for r in item["retrieved"]],
                   "retrieval_scores": [r["score"] for r in item["retrieved"]],
                   "context": ctx, "final_answer": ans}
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
            print(f"{item['id']} -> {ans[:120]}", flush=True)
    print(f"traza -> {trace_out}")


run_reader(REPO + "/rag_baseline/retrieval_hybrid.json",
           REPO + "/rag_baseline/run_trace_hybrid_colab.jsonl")
run_reader(REPO + "/rag_baseline/retrieval_expanded.json",
           REPO + "/rag_baseline/run_trace_expanded_colab.jsonl")
"""),
        md("## 4. Puntuar ambas variantes\n\n"
           "Corre `score_hybrid.py` y `score_expanded.py` tal cual (mismo scorer, "
           "mismos modos de fallo verificados). En CPU: híbrido 2/12 (Q1, Q11), "
           "híbrido + expansión 5/12 (Q1, Q4, Q6, Q8, Q10)."),
        code("""
import subprocess, sys
jobs = [("score_hybrid.py", "run_trace_hybrid_colab.jsonl",
         "retrieval_hybrid.json", "results_hybrid_colab.md"),
        ("score_expanded.py", "run_trace_expanded_colab.jsonl",
         "retrieval_expanded.json", "results_expanded_colab.md")]
for script, trace, retr, out in jobs:
    r = subprocess.run([sys.executable, script, "--trace", trace,
                        "--retrieval", retr, "--out", out],
                       cwd=REPO + "/rag_baseline")
    print(script, "exit:", r.returncode, "->", out)
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
            "condense, mismo prompt que `rag_baseline/answer.py`); si el juez no "
            "selecciona nada, abstención honesta.\n\n"
            "**Juez y lector vía ChatOllama** contra el servidor local "
            "(`JUDGE_OLLAMA_MODEL`, default `qwen2.5:7b`; `qwen2.5:1.5b` para paridad con "
            "el run en CPU). El código de referencia vive en el repo "
            "(`agentic_rag/judge_rerank.py`: SYSTEM y parseo de veredictos; "
            "`agentic_rag/run_judge.py`: pipeline completo en CPU; "
            "`agentic_rag/score_judge.py`: scorer con diagnóstico del juez).\n\n"
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
        md("## 1. Retrieval híbrido top-10\n\n"
           "Corre `rag_baseline/retrieve_hybrid.py` con `--k 10`: BM25 Okapi + MiniLM, "
           "fusión RRF k=60, top-10 por pregunta (el juez necesita más de donde elegir "
           "que el top-5 fijo). Determinista: se puede regenerar sin cambiar el resultado."),
        code("""
import subprocess, sys
r = subprocess.run([sys.executable, "retrieve_hybrid.py", "--k", str(K),
                    "--out", RETRIEVAL_K10], cwd=REPO + "/rag_baseline")
print("exit:", r.returncode)
"""),
        md("## 2. Juez + lector (ChatOllama)\n\n"
           "Un llamado al juez por pregunta con los 10 chunks numerados (misma versión "
           "condensada que verá el lector, mismo SYSTEM y mismo parseo de veredictos "
           "que `agentic_rag/judge_rerank.py`; máximo 3 reintentos; si no hay veredictos "
           "válidos, abstención sin lector). El lector contesta solo con los chunks "
           "seleccionados (mismo SYSTEM y `condense()` de `rag_baseline/answer.py`). "
           "Reanudable: salta las preguntas cuyo id ya está en la traza "
           "(`run_trace_judge_colab.jsonl`, nunca toca la traza del run en CPU)."),
        code("""
import json, sys, time
sys.path.insert(0, REPO + "/agentic_rag")
sys.path.insert(0, REPO + "/rag_baseline")
from judge_rerank import SYSTEM as JUDGE_SYSTEM, parse_verdicts
from answer import SYSTEM as READER_SYSTEM, condense
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

ABSTAIN = "No lo sé."
judge_llm = ChatOllama(model=JUDGE_OLLAMA_MODEL, base_url="http://localhost:11434",
                       temperature=0, num_predict=200)
reader_llm = ChatOllama(model=JUDGE_OLLAMA_MODEL, base_url="http://localhost:11434",
                        temperature=0, num_predict=100)
print("juez/lector (Ollama):", JUDGE_OLLAMA_MODEL)

def judge_once(question, chunks):
    # chunks: lista de (page, condensed_text). Un llamado, 10 veredictos.
    parts = [f"[{i}] Página {p}\\n{c}" for i, (p, c) in enumerate(chunks, start=1)]
    user = (f"Pregunta: {question}\\n\\nFragmentos del informe anual "
            f"de FEMSA 2024:\\n\\n" + "\\n\\n".join(parts))
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
    ctx = "\\n\\n".join(f"[Página {chunks[i - 1][0]}]\\n{chunks[i - 1][1]}" for i in selected)
    rec["context"] = ctx
    # Lector: maximo 3 reintentos si el llamado falla o viene vacio.
    tr = time.time()
    answer, failed = None, False
    user = (f"Fragmentos del informe anual de FEMSA 2024:\\n\\n{ctx}\\n\\n"
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
        f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        f.flush()
        print(f"{item['id']} juez_ok={rec['judge']['ok']} "
              f"intentos_juez={rec['judge']['attempts']} "
              f"seleccionados={len(rec['selected_pages'])}/10 "
              f"lector_intentos={rec['reader']['attempts']} "
              f"-> {rec['final_answer'][:160]}", flush=True)
print(f"listo en {time.time() - t0:.1f}s -> {TRACE_OUT}")
"""),
        md("## 3. Puntuar con el mismo scorer (con diagnóstico del juez)\n\n"
           "Corre `agentic_rag/score_judge.py` con `--blank-modes`: escribe "
           "`results_agentic_judge_colab.md` con el puntaje, la comparación con los "
           "demás sistemas y el diagnóstico del juez (precisión/recall de selección "
           "contra las páginas de evidencia). Los modos de fallo quedan en PENDIENTE: "
           "se verifican a mano contra el corpus y se llenan en VERIFIED_MODES de "
           "`score_judge.py`, y se vuelve a correr sin `--blank-modes`."),
        code("""
import subprocess, sys
r = subprocess.run([sys.executable, "score_judge.py", "--trace", TRACE_OUT,
                    "--out", REPO + "/agentic_rag/results_agentic_judge_colab.md",
                    "--model-label", f"Ollama {JUDGE_OLLAMA_MODEL} (Colab T4)",
                    "--system-name", "RAG agentico juez Ollama (Colab)",
                    "--blank-modes"],
                   cwd=REPO + "/agentic_rag")
print("exit:", r.returncode)
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
           "`compute_metrics.py` y `compute_relevancy.py` leen los nombres de CPU "
           "(`retrieval.json`, `run_trace.jsonl`, ...), pero en Colab los notebooks 05 y 06 "
           "escriben las trazas como `run_trace_colab.jsonl` (y variantes). Esta celda "
           "verifica que existan los 6 insumos, te pide subirlos si faltan, y alinea los "
           "nombres (copia `run_trace_*_colab.jsonl` -> `run_trace_*.jsonl`) para que "
           "los scripts del repo los encuentren tal cual."),
        code(r"""
import json, shutil
from pathlib import Path

BASE = Path(REPO) / "rag_baseline"

# (nombre Colab, nombre que espera compute_metrics.py, quién lo produce)
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

# Alinear nombres: los scripts del repo esperan los de CPU.
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
           "Corre `compute_metrics.py` y `compute_relevancy.py` tal cual: precision y "
           "recall de contexto desde los traces de retrieval + ground truth auditable, "
           "faithfulness por chequeo de claims contra el contexto condensado, relevancy "
           "como proxy de coseno. En local: denso recall 0.47 / faithfulness 0.97, "
           "híbrido 0.53 / 0.97, expansión 1.00 / 0.98."),
        code("""
import subprocess, sys
for script in ["compute_metrics.py", "compute_relevancy.py"]:
    r = subprocess.run([sys.executable, script], cwd=REPO + "/rag_baseline/ragas")
    print(script, "exit:", r.returncode)
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
