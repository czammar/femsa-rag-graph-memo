#!/usr/bin/env python3
"""RAG agéntico (FEMSA 2024) — LangChain + LangGraph, modelo local.

Sistema: grafo LangGraph plan -> act -> observe -> answer, con 3
herramientas LangChain (@tool) que envuelven código ya verificado:
  1. vector_search: retrieval híbrido BM25 Okapi + MiniLM, RRF k=60,
     top-5 páginas (misma lógica que rag_baseline/retrieve_hybrid.py).
  2. graph_query: Cypher contra el store Relate pipeline/data/femsa.rl
     (vía CLI `relate`, mismo store que pipeline/load_relate.py).
  3. calculator: aritmética segura por AST (sin exec).

LLM: por defecto Qwen2.5-1.5B-Instruct local (bf16, CPU), mismo modelo
que hizo de EXECUTIVE en MEMO run 2 y de reader en el RAG. Con
`configure_llm("ollama", model="qwen2.5:7b-instruct")` el mismo grafo
corre contra un modelo servido por Ollama (ver agentic_rag/OLLAMA.md):
mismo grafo, mismas herramientas, mismos prompts, mismo scorer; solo
cambia el cerebro. La funcion `call_llm` vive en llm_backends.py.

NOTA SOBRE TOOL CALLING NATIVO (verificado, no supuesto):
Se probó `chat.bind_tools(...)` de LangChain con este modelo
(agentic_rag/test_toolcall.py y test_toolcall2.py, con
return_full_text True y False): en ambas pruebas el modelo NO emitió
ningún tool_call — contestó en texto libre y, en la primera prueba,
además devolvió eco del prompt completo (`<|im_start|>system...`) y
tardó 235.8 s en un solo invoke. Por eso el agente usa ReAct por texto:
el modelo escribe una línea `ACCIÓN: herramienta(argumento)` o
`RESPUESTA: <texto>`, y el nodo `act` la parsea y ejecuta la @tool
correspondiente. El grafo, los nodos y las herramientas sí son
LangChain/LangGraph; solo el formato de llamada es texto, porque el
modelo de 1.5B no sigue el formato estructurado de forma fiable.

Uso:
  agentic_rag/.venv/bin/python agentic_rag/run_agentic.py
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rag_baseline"))
sys.path.insert(0, str(ROOT / "memo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from llm_backends import backend_name, call_llm  # noqa: E402
from llm_backends import configure as configure_llm  # noqa: E402

CHUNKS = ROOT / "pipeline/data/chunks.jsonl"
RELATE_DB = ROOT / "pipeline/data/femsa.rl"
RELATE_BIN = Path.home() / ".local/bin/relate"
MAX_TOOL_CALLS = 3
MAX_NEW_TOKENS = 150

# --- retrieval híbrido (índice construido una sola vez) -------------------
from retrieve_hybrid import (  # noqa: E402
    EMBED_MODEL, RRF_K, rrf_ranks, tokenize)

_BM25 = None
_EMB = None
_CHUNKS = None
_ST_MODEL = None


def _ensure_index():
    global _BM25, _EMB, _CHUNKS, _ST_MODEL
    if _BM25 is not None:
        return
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _CHUNKS = [json.loads(l) for l in open(CHUNKS)]
    texts = [c["text"] for c in _CHUNKS]
    _BM25 = BM25Okapi([tokenize(t) for t in texts])
    _ST_MODEL = SentenceTransformer(EMBED_MODEL)
    _EMB = np.asarray(_ST_MODEL.encode(texts, normalize_embeddings=True,
                                        show_progress_bar=False),
                      dtype=np.float32)
    print(f"[tools] índice híbrido listo: {len(_CHUNKS)} chunks", flush=True)


def _hybrid_topk(query, k=5):
    import numpy as np
    _ensure_index()
    q_tokens = tokenize(query)
    b_scores = np.asarray(_BM25.get_scores(q_tokens), dtype=np.float64)
    q_emb = _ST_MODEL.encode([query], normalize_embeddings=True,
                             show_progress_bar=False)[0]
    sims = _EMB @ q_emb
    b_rank, d_rank = rrf_ranks(b_scores, sims)
    rrf = 1.0 / (RRF_K + b_rank) + 1.0 / (RRF_K + d_rank)
    return [int(i) for i in np.argsort(-rrf)[:k]]


# condense de rag_baseline/answer.py (mismo workaround CPU)
from answer import condense  # noqa: E402


@tool
def vector_search(query: str) -> str:
    """Busca en el informe anual de FEMSA 2024 (inglés) y devuelve los 5
    fragmentos más relevantes. Escribe la query en inglés con palabras
    clave del dominio (p. ej. 'OXXO GAS stations', 'FEMSA total revenue
    2024'). Devuelve cada fragmento con su número de página."""
    idxs = _hybrid_topk(query, k=5)
    parts = []
    for i in idxs:
        c = _CHUNKS[i]
        parts.append(f"[Página {c['page']}]\n{condense(c['text'], query)}")
    return "\n\n".join(parts)


@tool
def graph_query(cypher: str) -> str:
    """Ejecuta una consulta Cypher de solo lectura contra el grafo de
    conocimiento de FEMSA 2024 (Relate) y devuelve las filas en JSON.
    Esquema: nodos Empresa, Division, Formato, IndicadorFinanciero,
    MetricaOperativa, Pais, Iniciativa, Riesgo. Relaciones: REPORTA,
    MIDE, TIENE_DIVISION, TIENE_FORMATO, OPERA_EN, CRECIO_EN, IMPULSA,
    EVOLUCIONO_A. Los valores viven en la relación: REPORTA/MIDE tienen
    propiedades valor (número), variacion_pct, periodo, unidad,
    fuente_pagina. Los nodos tienen propiedad nombre.
    Ejemplo: MATCH (e:Empresa {nombre: 'FEMSA'})-[r:REPORTA]->
    (i:IndicadorFinanciero {nombre: 'Ingresos Totales'}) RETURN r.valor
    AS ingresos, r.variacion_pct AS crecimiento."""
    try:
        p = subprocess.run(
            [str(RELATE_BIN), "-d", str(RELATE_DB), "query"],
            input=cypher.encode(), capture_output=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        return f"ERROR al ejecutar Cypher: {e}"
    out = p.stdout.decode(errors="replace").strip()
    err = p.stderr.decode(errors="replace").strip()
    if p.returncode != 0:
        return f"ERROR de Cypher: {err or out}"
    # relate imprime una línea de telemetría y luego un JSON por fila
    rows = [l for l in out.splitlines() if l.startswith("{")]
    if not rows:
        return ("Sin resultados. Revisa el esquema: nodos Empresa, Division, "
                "Formato, IndicadorFinanciero, MetricaOperativa, Pais, "
                "Iniciativa, Riesgo; relaciones REPORTA, MIDE, OPERA_EN, "
                "IMPULSA, EVOLUCIONO_A; valores en r.valor / r.variacion_pct.")
    try:
        parsed = [json.loads(r) for r in rows]
        if all(all(v in (0, 0.0, None, "") for v in row.values())
               for row in parsed):
            return ("Sin resultados útiles (todo en cero). Revisa el esquema: "
                    "los valores viven en la relación REPORTA/MIDE "
                    "(r.valor, r.variacion_pct), no en el nodo.")
    except (json.JSONDecodeError, AttributeError):
        pass
    return "\n".join(rows)


@tool
def calculator(expression: str) -> str:
    """Evalúa una expresión aritmética simple con + - * / y paréntesis,
    p. ej. '307197 + 279793' o '(307197 + 279793) * 100 / 781585'.
    Úsala para cualquier suma, diferencia o porcentaje: no hagas la
    cuenta de cabeza."""
    import ast
    import operator as op
    allowed = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
               ast.Div: op.truediv, ast.Mod: op.mod, ast.USub: op.neg}
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Num):
            return n.n
        if isinstance(n, ast.BinOp) and type(n.op) in allowed:
            return allowed[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in allowed:
            return allowed[type(n.op)](ev(n.operand))
        raise ValueError("expresión no soportada")
    try:
        v = ev(ast.parse(expression, mode="eval"))
        return f"{expression} = {v}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR en calculator: {e}"


TOOLS = {"vector_search": vector_search,
         "graph_query": graph_query,
         "calculator": calculator}

# --- LLM ------------------------------------------------------------------
# call_llm vive en llm_backends.py (backend local 1.5B por defecto, u
# Ollama con configure_llm("ollama", model=...)). Solo se delega aqui.


PLAN_SYSTEM = (
    "Eres un agente que responde preguntas sobre el informe anual de "
    "FEMSA 2024. Tienes 3 herramientas:\n"
    "- vector_search(query): busca fragmentos del informe (escribe la "
    "query en inglés, con palabras clave).\n"
    "- graph_query(cypher): consulta el grafo de conocimiento para "
    "cifras exactas verificadas. Esquema del grafo: nodos Empresa, "
    "Division, Formato, IndicadorFinanciero, MetricaOperativa, Pais, "
    "Iniciativa, Riesgo (todos con propiedad nombre). Relaciones: "
    "REPORTA, MIDE, TIENE_DIVISION, TIENE_FORMATO, OPERA_EN, CRECIO_EN, "
    "IMPULSA, EVOLUCIONO_A. Los valores viven en la relación REPORTA o "
    "MIDE: r.valor (número), r.variacion_pct, r.periodo. REGLAS DE "
    "RELACIONES (no las mezcles): REPORTA conecta (Empresa o Division) "
    "con IndicadorFinanciero, para ingresos y utilidad. MIDE conecta "
    "(Division) con MetricaOperativa, para tiendas, aperturas, "
    "estaciones, usuarios y volumen. OPERA_EN conecta (Division) con "
    "Pais, solo para países. EVOLUCIONO_A conecta (Iniciativa) con otra "
    "Iniciativa o Formato, para evolución. IMPULSA conecta (Iniciativa) "
    "con Division, para qué articula una estrategia. Toda consulta que "
    "pida un valor TERMINA con RETURN r.valor, r.variacion_pct. "
    "PLANTILLAS (copia la que corresponda y cambia solo los nombres "
    "entre comillas): ingresos o utilidad: MATCH (d)-[r:REPORTA]->"
    "(i:IndicadorFinanciero {nombre: 'Ingresos Totales'}) WHERE "
    "d.nombre = '...' RETURN r.valor, r.variacion_pct (para FEMSA usa "
    "(e:Empresa {nombre: 'FEMSA'})); métrica operativa: MATCH "
    "(d:Division {nombre: '...'})-[r:MIDE]->(m:MetricaOperativa "
    "{nombre: '...'}) RETURN r.valor, r.variacion_pct; países: MATCH "
    "(d:Division {nombre: '...'})-[:OPERA_EN]->(p:Pais) RETURN p.nombre; "
    "evolución de iniciativa: MATCH (i:Iniciativa {nombre: '...'})-"
    "[r:EVOLUCIONO_A]->(x) RETURN x.nombre; qué articula una iniciativa: "
    "MATCH (i:Iniciativa {nombre: '...'})-[:IMPULSA]->(d:Division) "
    "RETURN d.nombre. NUNCA escribas m.valor, i.valor, s.valor ni "
    "d.valor: el valor SIEMPRE es r.valor (vive en la relación, no en "
    "el nodo). "
    "que existen en el grafo: Divisiones: Coca-Cola FEMSA, Combustibles, "
    "Proximidad Américas, Proximidad Europa, Salud, Spin. Formatos: "
    "Juntos+, OXXO GAS, OXXO México, Spin Premia, Spin by OXXO. "
    "IndicadorFinanciero: Ingresos Totales, Utilidad de Operación. "
    "MetricaOperativa: Aperturas, Clientes Juntos+, Estaciones, Tiendas, "
    "Usuarios Activos, Volumen. Iniciativa: Digital@FEMSA, FEMSA Forward. "
    "Pais: Chile, Colombia, México, Perú. OJO: OXXO GAS es un Formato de "
    "la División Combustibles; sus estaciones se consultan como "
    "(d:Division {nombre: 'Combustibles'})-[r:MIDE]->(m:MetricaOperativa "
    "{nombre: 'Estaciones'}) RETURN r.valor. Ejemplo: "
    "MATCH (e:Empresa {nombre: 'FEMSA'})-[r:REPORTA]->(i:IndicadorFinanciero "
    "{nombre: 'Ingresos Totales'}) RETURN r.valor AS ingresos, "
    "r.variacion_pct AS crecimiento. NUNCA inventes etiquetas, relaciones "
    "ni nombres: usa solo los de este esquema.\n"
    "- calculator(expresion): SOLO expresiones aritméticas con números y "
    "+ - * / ( ), p. ej. '307197 + 279793'. Nunca le pases texto, "
    "palabras ni consultas Cypher.\n"
    "En cada turno escribe UNA sola línea con EXACTAMENTE uno de estos "
    "dos formatos, sin nada más:\n"
    "ACCIÓN: vector_search(\"tu query en inglés\")\n"
    "ACCIÓN: graph_query(\"MATCH ... RETURN ...\")\n"
    "ACCIÓN: calculator(\"307197 + 279793\")\n"
    "RESPUESTA: <tu respuesta final en español, con las cifras exactas "
    "de la evidencia>\n"
    "Si la evidencia no alcanza, escribe RESPUESTA: No lo sé. "
    "No inventes cifras. En el primer turno SIEMPRE escribe una ACCIÓN, "
    "nunca una RESPUESTA: primero busca, luego responde. Empieza SIEMPRE "
    "con vector_search. Solo si los fragmentos no traen la cifra exacta, "
    "usa graph_query con una plantilla. Si la pregunta pide un cálculo "
    "(suma, diferencia, porcentaje), usa calculator con las cifras de la "
    "evidencia, nunca de cabeza. Si una consulta devuelve cero o un "
    "error, corrige UNA cosa (el nombre, la relación o el RETURN) y "
    "reintenta; no te rindas al primer error.\n"
    "Ejemplo de un turno correcto (pregunta que NO es del examen):\n"
    "Pregunta: ¿En qué año se fundó FEMSA?\n"
    "ACCIÓN: vector_search(\"FEMSA founded year history\")\n"
    "Observación: [Página 8] FEMSA fue fundada en 1890 en Monterrey.\n"
    "RESPUESTA: FEMSA se fundó en 1890."
)

ANSWER_SYSTEM = (
    "Responde la pregunta usando UNICAMENTE la evidencia reunida por las "
    "herramientas (fragmentos del informe y filas del grafo), que se te "
    "muestra a continuación. Copia las cifras exactamente como aparecen "
    "en la evidencia, sin redondear ni inventar nada. Si la evidencia no "
    "alcanza para responder, responde exactamente: No lo sé. Responde en "
    "español, en una o dos frases cortas."
)

ACTION_RE = re.compile(
    r"ACCI[OÓ]N:\s*(vector_search|graph_query|calculator)\s*\(\s*(.*)\s*\)\s*$",
    re.DOTALL)
RESPONSE_RE = re.compile(r"RESPUESTA:\s*(.*)", re.DOTALL)


def _parse_action(text):
    """Devuelve (tool_name, arg) o (None, None). Acepta comillas simples,
    dobles o triples en el argumento."""
    m = ACTION_RE.search(text.strip())
    if not m:
        return None, None
    arg = m.group(2).strip()
    if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in "\"'":
        arg = arg[1:-1]
    elif arg.startswith('"""') and arg.endswith('"""'):
        arg = arg[3:-3]
    arg = arg.replace('\\"', '"').replace("\\'", "'")
    return m.group(1), arg


# --- Grafo LangGraph ------------------------------------------------------
def plan_node(state):
    msgs = [{"role": "system", "content": PLAN_SYSTEM}]
    hist = f"Pregunta: {state['question']}\n"
    for i, s in enumerate(state["steps"], 1):
        hist += (f"\nPaso {i}: {s['tool']}({s['arg']})\n"
                 f"Observación: {s['observation']}\n")
    msgs.append({"role": "user", "content": hist})
    raw = call_llm(msgs, stop_strings=["\n"])
    state["last_raw"] = raw
    state.setdefault("plan_raws", []).append(raw)
    m = RESPONSE_RE.search(raw)
    if m:
        # Sin evidencia aún: no se permite responder; se le exige buscar.
        if not state["steps"]:
            state["wants_answer"] = False
            state["pending_tool"] = None
            state["pending_arg"] = None
            state["steps"].append({
                "tool": "system_nudge", "arg": "",
                "observation": "Aún no has usado ninguna herramienta. "
                "Escribe una ACCIÓN con vector_search o graph_query."})
            return state
        state["wants_answer"] = True
        state["draft_answer"] = m.group(1).strip()
    else:
        name, arg = _parse_action(raw)
        state["wants_answer"] = False
        state["pending_tool"] = name
        state["pending_arg"] = arg
    return state


def act_node(state):
    name, arg = state.get("pending_tool"), state.get("pending_arg")
    if not name or name not in TOOLS:
        obs = ("ERROR: no se entendió la acción. Vuelve a intentarlo con "
               "el formato exacto ACCIÓN: herramienta(\"argumento\").")
        state["steps"].append({"tool": "parse_error",
                              "arg": state.get("last_raw", "")[:200],
                              "observation": obs})
        return state
    # Guardia anti-bucle: repetir la misma llamada exacta no aporta nada.
    for s in state["steps"]:
        if s.get("tool") == name and s.get("arg") == arg:
            state["steps"].append({
                "tool": name, "arg": arg,
                "observation": ("Ya ejecutaste exactamente esta misma "
                                "llamada y dio: " + str(s["observation"])[:500] +
                                " Cambia de herramienta o cambia la "
                                "consulta; no la repitas.")})
            return state
    t0 = time.time()
    try:
        obs = TOOLS[name].invoke(arg)
    except Exception as e:  # noqa: BLE001
        obs = f"ERROR al ejecutar {name}: {e}"
    state["steps"].append({"tool": name, "arg": arg,
                          "observation": str(obs)[:4000],
                          "seconds": round(time.time() - t0, 1)})
    return state


def answer_node(state):
    if state.get("draft_answer"):
        state["final_answer"] = state["draft_answer"]
        return state
    ev = f"Pregunta: {state['question']}\n"
    for i, s in enumerate(state["steps"], 1):
        ev += (f"\nEvidencia {i} ({s['tool']}):\n{s['observation']}\n")
    msgs = [{"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": ev}]
    state["final_answer"] = call_llm(
        msgs, max_new_tokens=150, stop_strings=["\n\n", "Pregunta:"])
    return state


def _route(state):
    if state.get("wants_answer"):
        return "answer"
    if len(state["steps"]) >= MAX_TOOL_CALLS:
        return "answer"
    if len(state.get("plan_raws", [])) >= 5:
        # Demasiados intentos de planificación sin acción válida.
        return "answer"
    if not state.get("pending_tool"):
        # El plan no produjo una acción válida (o fue un nudge):
        # volver a planificar sin gastar una llamada a herramienta.
        return "plan"
    return "act"


def build_graph():
    g = StateGraph(dict)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("answer", answer_node)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", _route,
                            {"plan": "plan", "act": "act",
                             "answer": "answer"})
    g.add_edge("act", "plan")
    g.add_edge("answer", END)
    return g.compile()


def run_question(question, qid, level, graph=None):
    graph = graph or build_graph()
    t0 = time.time()
    state = {"question": question, "id": qid, "level": level,
             "steps": [], "last_raw": "", "plan_raws": [],
             "wants_answer": False,
             "draft_answer": "", "pending_tool": None, "pending_arg": None,
             "final_answer": ""}
    out = graph.invoke(state)
    return {"id": qid, "level": level, "question": question,
            "decomposition": "agentic",
            "llm_backend": backend_name(),
            "final_answer": out["final_answer"],
            "steps": out["steps"],
            "plan_raw_outputs": out.get("plan_raws", []),
            "seconds": round(time.time() - t0, 1)}
