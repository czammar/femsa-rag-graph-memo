#!/usr/bin/env python3
"""Backends de LLM para el agente (FEMSA 2024).

El agente (agent.py) solo habla con `call_llm(messages, max_new_tokens,
stop_strings)`. Que modelo contesta detras de esa funcion depende del
backend, que se elige con `configure()`:

  - "local": Qwen2.5-1.5B-Instruct en disco (transformers, bf16, CPU),
    el backend de la corrida original (2/12).
  - "ollama": un modelo servido por Ollama (por defecto
    qwen2.5:7b-instruct), via langchain-ollama ChatOllama. Mismo grafo,
    mismas herramientas, mismos prompts, mismo scorer: solo cambia el
    cerebro. Ver agentic_rag/OLLAMA.md para correrlo en tu maquina.

El import de langchain_ollama es perezoso (dentro de OllamaBackend):
el paquete `ollama` lee variables de proxy al importarse y en algunas
VMs eso rompe la importacion aunque Ollama no se use. El backend local
tampoco importa torch/transformers hasta que se carga el modelo, asi
que `agent.py` se puede importar sin el stack ML pesado cuando se usa
Ollama.
"""
import os
import time
from pathlib import Path

DEFAULT_LOCAL_MODEL = "memo/models/Qwen2.5-1.5B-Instruct"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

_BACKEND = None


class LocalBackend:
    """Qwen2.5-1.5B-Instruct local (transformers, bf16, CPU).

    Mismo patron probado de rag_baseline/answer.py y
    memo/executive_v2.py: chat template + generate con do_sample=False.
    """

    def __init__(self, root, model=DEFAULT_LOCAL_MODEL):
        self.root = Path(root)
        self.model_path = self.root / model
        self._tok = None
        self._model = None
        self.name = f"local {model}"

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(str(self.model_path))
        print(f"[llm] cargando {self.model_path} (bf16, CPU)...", flush=True)
        t0 = time.time()
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path), torch_dtype=torch.bfloat16)
        self._model.eval()
        print(f"[llm] modelo cargado en {time.time() - t0:.1f}s", flush=True)

    def __call__(self, messages, max_new_tokens=150, stop_strings=None):
        import torch
        self._ensure()
        text = self._tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        ids = self._tok(text, return_tensors="pt")
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                      pad_token_id=self._tok.eos_token_id)
        if stop_strings:
            kwargs["stop_strings"] = stop_strings
            kwargs["tokenizer"] = self._tok
        with torch.no_grad():
            out = self._model.generate(**ids, **kwargs)
        return self._tok.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()


class OllamaBackend:
    """Un modelo servido por Ollama, via langchain-ollama ChatOllama.

    ChatOllama aplica el chat template del modelo en el servidor, asi
    que los mensajes (system/user) se pasan tal cual, igual que en el
    backend local. temperature=0: determinista, como el greedy local.
    num_predict = max_new_tokens: mismo tope de generacion que el
    backend local. stop_strings se pasan como `stop` del modelo.
    """

    def __init__(self, model=DEFAULT_OLLAMA_MODEL, host=DEFAULT_OLLAMA_HOST):
        self.model = model
        self.host = host
        self.name = f"ollama {model} @ {host}"

    def __call__(self, messages, max_new_tokens=150, stop_strings=None):
        # Todo dentro del try: hasta el import de langchain_ollama puede
        # fallar si la terminal tiene variables de proxy raras (el
        # paquete `ollama` las lee al importarse), y eso tambien debe
        # salir como el error amable de abajo, no como un traceback.
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_ollama import ChatOllama
            lc_messages = [
                SystemMessage(content=m["content"]) if m["role"] == "system"
                else HumanMessage(content=m["content"])
                for m in messages
            ]
            llm = ChatOllama(model=self.model, base_url=self.host,
                             temperature=0, num_predict=max_new_tokens,
                             stop=list(stop_strings) if stop_strings else None)
            resp = llm.invoke(lc_messages)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Ollama no responde en {self.host} ({e}). "
                "¿Esta corriendo? `ollama serve` (o abre la app de "
                "Ollama) y ¿ya bajaste el modelo? "
                f"`ollama pull {self.model}`") from e
        return str(resp.content).strip()


def configure(backend="local", model=None, host=None, root=None):
    """Elige el backend del agente. Llamar antes del primer call_llm.

    backend: "local" | "ollama".
    model: local -> ruta relativa a la raiz (por defecto 1.5B);
           ollama -> nombre del modelo en Ollama (por defecto
           qwen2.5:7b-instruct).
    host: solo ollama; por defecto $OLLAMA_HOST o
          http://localhost:11434.
    root: raiz del repo (solo local).
    """
    global _BACKEND
    if backend == "ollama":
        _BACKEND = OllamaBackend(
            model=model or os.environ.get("OLLAMA_MODEL",
                                          DEFAULT_OLLAMA_MODEL),
            host=host or os.environ.get("OLLAMA_HOST",
                                        DEFAULT_OLLAMA_HOST))
    elif backend == "local":
        _BACKEND = LocalBackend(
            root or Path(__file__).resolve().parent.parent,
            model=model or DEFAULT_LOCAL_MODEL)
    else:
        raise ValueError(f"backend desconocido: {backend} (local|ollama)")
    print(f"[llm] backend: {_BACKEND.name}", flush=True)
    return _BACKEND


def call_llm(messages, max_new_tokens=150, stop_strings=None):
    """Una llamada al LLM con el backend configurado.

    Sin configure() previo: backend local 1.5B, como la corrida
    original. Devuelve solo el texto generado, sin eco del prompt.
    stop_strings corta la generacion (el modelo tiende a seguir
    escribiendo: repite el prompt o añade turnos inventados).
    """
    if _BACKEND is None:
        configure()
    return _BACKEND(messages, max_new_tokens, stop_strings)


def backend_name():
    """Nombre legible del backend activo (para la traza)."""
    return _BACKEND.name if _BACKEND is not None else "local (por defecto)"
