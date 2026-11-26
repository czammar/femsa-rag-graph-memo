#!/usr/bin/env python3
"""Diagnóstico: ¿puede Qwen2.5-1.5B escribir el Cypher correcto para Q4
con un prompt corto y enfocado (sin el bucle ReAct)?"""
import sys
sys.path.insert(0, "agentic_rag")
from agent import call_llm

SCHEMA = """El grafo de FEMSA 2024 tiene estos nodos (propiedad nombre) y relaciones:
- Division: Coca-Cola FEMSA, Combustibles, Proximidad Américas, Proximidad Europa, Salud, Spin
- Formato: Juntos+, OXXO GAS, OXXO México, Spin Premia, Spin by OXXO
- IndicadorFinanciero: Ingresos Totales, Utilidad de Operación
- MetricaOperativa: Aperturas, Clientes Juntos+, Estaciones, Tiendas, Usuarios Activos, Volumen
- Iniciativa: Digital@FEMSA, FEMSA Forward
- Pais: Chile, Colombia, México, Perú
Relaciones: REPORTA, MIDE, OPERA_EN, IMPULSA, EVOLUCIONO_A.
Los valores viven en la relación REPORTA o MIDE: r.valor, r.variacion_pct.
OXXO GAS es un Formato de la División Combustibles; sus estaciones se consultan como
(d:Division {nombre: 'Combustibles'})-[r:MIDE]->(m:MetricaOperativa {nombre: 'Estaciones'}) RETURN r.valor.

Ejemplo:
Pregunta: ¿Cuáles fueron los ingresos totales de FEMSA en 2024?
Cypher: MATCH (e:Empresa {nombre: 'FEMSA'})-[r:REPORTA]->(i:IndicadorFinanciero {nombre: 'Ingresos Totales'}) RETURN r.valor, r.variacion_pct"""

msgs = [
    {"role": "system", "content": SCHEMA + "\n\nEscribe SOLO la consulta Cypher, sin explicaciones."},
    {"role": "user", "content": "Pregunta: ¿Cuántas estaciones tenía OXXO GAS al cierre de 2024?\nCypher:"},
]
print(call_llm(msgs, max_new_tokens=120, stop_strings=["\n"]))
