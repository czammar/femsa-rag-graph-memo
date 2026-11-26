#!/usr/bin/env python3
"""Prueba 2 de tool calling nativo: con return_full_text=False.
Si tampoco emite tool_calls, el agente usará ReAct por texto."""
import time

from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

MODEL = "memo/models/Qwen2.5-1.5B-Instruct"


@tool
def calculator(expression: str) -> str:
    """Evalúa una expresión aritmética simple, p. ej. '307197 + 279793'."""
    import ast
    import operator as op
    allowed = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
               ast.Div: op.truediv, ast.Mod: op.mod}
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Num):
            return n.n
        if isinstance(n, ast.BinOp) and type(n.op) in allowed:
            return allowed[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("no soportado")
    return str(ev(ast.parse(expression, mode="eval")))


@tool
def vector_search(query: str) -> str:
    """Busca en el informe anual de FEMSA 2024 y devuelve fragmentos."""
    return ("[Página 21] OXXO GAS: 571 estaciones al cierre de 2024. "
            "[Página 112] Ingresos totales consolidados Ps. 781,585 millones.")


def main():
    t0 = time.time()
    llm = HuggingFacePipeline.from_model_id(
        model_id=MODEL, task="text-generation",
        pipeline_kwargs={"max_new_tokens": 150, "do_sample": False,
                         "return_full_text": False},
        model_kwargs={"torch_dtype": "bfloat16"})
    chat = ChatHuggingFace(llm=llm)
    print(f"modelo cargado en {time.time() - t0:.1f}s", flush=True)
    bound = chat.bind_tools([calculator, vector_search])
    msgs = [
        ("system", "Eres un agente que responde preguntas sobre el informe "
         "anual de FEMSA 2024. Usa las herramientas cuando necesites datos. "
         "Responde en español."),
        ("human", "¿Cuántas estaciones tenía OXXO GAS al cierre de 2024? "
         "Usa vector_search para buscar el dato."),
    ]
    t1 = time.time()
    out = bound.invoke(msgs)
    print(f"invoke en {time.time() - t1:.1f}s")
    print("content:", repr(out.content))
    print("tool_calls:", out.tool_calls)
    print("invalid_tool_calls:", getattr(out, "invalid_tool_calls", None))


if __name__ == "__main__":
    main()
