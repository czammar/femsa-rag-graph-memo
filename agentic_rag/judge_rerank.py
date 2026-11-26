#!/usr/bin/env python3
"""RAG agentico (FEMSA 2024) -- el JUEZ: califica chunks en bloque.

El agente ReAct (agent.py, 2/12) midio un agente solo-grafo: el 1.5B
ignoro vector_search y calculator y fue directo al grafo. Este es el
segundo diseno, el que pidio el usuario: el sistema agentico SOLO
califica los chunks del RAG hibrido y selecciona los utiles con un juez.
Sin graph_query, sin calculator.

Esquema (1 llamado al juez por pregunta, NO 1 por chunk): se presentan
los 10 chunks del retrieval hibrido numerados del 1 al 10 (version
condensada, la misma que vera el lector) y el juez devuelve UNA linea
por chunk: 'N: UTIL: <razon corta>' o 'N: NO UTIL: <razon corta>'.

Regla de reintentos (del usuario): si los veredictos no se pueden
parsear (o el llamado falla), se reintenta hasta 3 veces con la misma
pregunta; si a los 3 sigue sin veredictos validos, la pregunta se
abstiene ("No lo se", sin contestar desde chunks sin juzgar) y se
registra en la traza. Un parseo es valido solo si hay una linea
parseable para CADA uno de los 10 chunks presentados.

Uso (libreria; el pipeline esta en run_judge.py):
  from judge_rerank import Judge
  judge = Judge.load("memo/models/Qwen2.5-1.5B-Instruct")
  res = judge.judge(question, condensed_chunks)  # lista de 10 textos
  # res = {"ok": bool, "verdicts": {1: {"useful": bool, "reason": str}, ...}
  #        | None, "attempts": int, "attempts_raw": [str, ...],
  #        "seconds": float}
"""
import re
import time
import unicodedata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_MODEL = "memo/models/Qwen2.5-1.5B-Instruct"
MAX_ATTEMPTS = 3
MAX_NEW = 200  # ~10 lineas de veredicto, 10-20 tokens cada una

SYSTEM = (
    "Eres un juez que decide si un fragmento del informe anual de FEMSA "
    "2024 sirve para responder una pregunta. Se te presentan varios "
    "fragmentos numerados del 1 al 10. Devuelve UNA linea por fragmento, "
    "en este formato exacto: 'N: UTIL: <razon corta>' si el fragmento "
    "contiene informacion que ayuda a responder la pregunta, o "
    "'N: NO UTIL: <razon corta>' si no la contiene. Una linea por "
    "fragmento, sin texto adicional, sin resumen, sin numeracion extra.")

_LINE_RE = re.compile(r"^\s*(\d{1,2})\s*[:\.\)\-]\s*(.+)$", re.MULTILINE)


def _norm(s):
    """Minusculas sin acentos, para comparar 'UTIL'/'ÚTIL'/'util' igual."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


def _parse_line(rest):
    """Parsea el resto de una linea de veredicto.

    Devuelve (useful, reason) o None si no se puede parsear.
    Se revisa NO UTIL antes que UTIL porque 'NO UTIL' contiene 'UTIL'.
    El modelo puede escribir UTIL con o sin acento; se normaliza antes
    de comparar.
    """
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
    """Parsea la salida cruda del juez.

    Devuelve {1: {"useful": bool, "reason": str}, ..., n: {...}} si hay
    una linea parseable para CADA uno de los n chunks; None si falta
    alguno o sobra ambiguedad. La primera linea valida por numero gana
    (si el modelo repite un numero, se conserva la primera).
    """
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


class Judge:
    """Juez local Qwen2.5-1.5B-Instruct (bf16, CPU)."""

    def __init__(self, model, tok):
        self.model = model
        self.tok = tok

    @classmethod
    def load(cls, model_path=JUDGE_MODEL):
        tok = AutoTokenizer.from_pretrained(model_path)
        print(f"loading judge {model_path} (bf16, CPU)...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16)
        model.eval()
        return cls(model, tok)

    def _one_call(self, question, chunks):
        """Un llamado al juez con los chunks numerados. Devuelve texto crudo."""
        parts = []
        for i, (page, text) in enumerate(chunks, start=1):
            parts.append(f"[{i}] Página {page}\n{text}")
        user = (f"Pregunta: {question}\n\nFragmentos del informe anual "
                f"de FEMSA 2024:\n\n" + "\n\n".join(parts))
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=MAX_NEW,
                                      do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()

    def judge(self, question, chunks):
        """Califica los chunks en bloque, con hasta 3 intentos.

        chunks: lista de (page, condensed_text). Devuelve un dict con
        "ok", "verdicts" (o None), "attempts", "attempts_raw" y "seconds".
        """
        t0 = time.time()
        attempts_raw = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                raw = self._one_call(question, chunks)
            except Exception as e:  # llamado fallido (OOM, etc.)
                raw = f"<ERROR: {type(e).__name__}: {e}>"
            attempts_raw.append(raw)
            verdicts = parse_verdicts(raw, len(chunks))
            if verdicts is not None:
                return {"ok": True, "verdicts": verdicts,
                        "attempts": attempt, "attempts_raw": attempts_raw,
                        "seconds": round(time.time() - t0, 1)}
        return {"ok": False, "verdicts": None, "attempts": MAX_ATTEMPTS,
                "attempts_raw": attempts_raw,
                "seconds": round(time.time() - t0, 1)}
