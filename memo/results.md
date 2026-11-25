# Resultados MEMO (demo) — FEMSA 2024

Sistema: EXECUTIVE = Qwen2.5-0.5B-Instruct base (congelado); MEMORY = base + adaptador LoRA entrenado en reflexiones del informe. El EXECUTIVE nunca ve el corpus; solo las sub-respuestas de MEMORY.

Correctas: **0/12**. Con alucinacion (numero no verificado en el informe): **6/12**.

| ID | Nivel | Pregunta | Esperado | Respuesta MEMO | Veredicto |
|----|-------|----------|----------|----------------|-----------|
| Q1 | 1 | ¿Cuáles fueron los ingresos totales consolidados de FEMSA en 2024 y cu | Ps. 781,585 millones, +11.2%. | Los ingresos totales consolidados de FEMSA en 2024 fueron ps. 157,968 millones, un crecimiento de 11.4%. | incorrecta |
| Q2 | 1 | ¿Cuántos usuarios activos tuvo Spin by OXXO en 2024? | 8.6 millones (+24.9%). | 15 millones de usuarios activos. | incorrecta |
| Q3 | 1 | ¿Cuántas tiendas sumó Proximidad Américas al cierre de 2024 y cuántas  | 23,206 tiendas; 1,236 aperturas. | La respuesta es:  137 tiendas sumaron Proximidad Américas al cierre de 2024. 59 aperturas hubieron en el mismo período. | incorrecta |
| Q4 | 1 | ¿Cuántas estaciones tenía OXXO GAS al cierre de 2024? | 571. | 139 estaciones | incorrecta |
| Q5 | 2 | ¿Qué división tuvo el mayor crecimiento porcentual de ingresos en 2024 | Empate nominal: ambas +14.2% (trampa deliberada). | Coca-Cola FEMSA. | incorrecta |
| Q6 | 2 | ¿Qué división aportó más ingresos en 2024, Proximidad Américas o Coca- | Proximidad Américas, Ps. 307,200 millones vs Ps. 279,800 mil | Proximidad Américas aportó más ingresos en 2024, con un total de Ps. 179,536 millones frente al de Coca-Cola FEMSA, que sumó Ps. 179,536 millones. | incorrecta |
| Q7 | 2 | ¿Qué creció más en usuarios, Spin by OXXO o Spin Premia? | Spin Premia en absoluto (24.6 millones vs 8.6 millones) y en | Spin Premia, con 13.5% creció más en usuarios que Spin by OXXO, con 9.7%. | incorrecta |
| Q8 | 2 | ¿En qué países opera Proximidad Américas según el informe? | México, Colombia, Chile, Perú (además de negocios en Estados | Proximidad Américas opera en 13 países. | incorrecta |
| Q9 | 3 | ¿Qué porcentaje de los ingresos consolidados 2024 representan Proximid | ~75.1% (39.3% + 35.8%, aproximado por redondeo). | 100% | incorrecta |
| Q10 | 3 | ¿Qué estrategia articula Retail, Bebidas y el ecosistema digital de FE | FEMSA Forward; el ecosistema digital centrado en el cliente  | La estrategia articulada por Retail, Bebidas y el ecosistema digital de FEMSA es la "Ecosistema Digital". | incorrecta |
| Q11 | 3 | ¿Cómo evolucionó la iniciativa digital de FEMSA en 2024? | Digital@FEMSA evolucionó a Spin. | El uso de iniciativas digitales aumentó 13% y se consolidaron las principales iniciativas digitales. | incorrecta |
| Q12 | 3 | ¿Qué división tuvo el año más complicado en 2024 según el propio infor | Salud: 'año complicado', +5.8% (el menor crecimiento de las  | El año más complicado para la División de Transportes fue 2024, con un crecimiento de ingresos de 13.5%. | incorrecta |

## Por nivel

- Nivel 1: 0/4 correctas.
- Nivel 2: 0/4 correctas.
- Nivel 3: 0/4 correctas.

## Comparacion con el control (base sin entrenar)

- Control (MEMORY = base Qwen2.5-0.5B sin entrenar): **0/12 correctas,
  4/12 con alucinacion** (ver `results_control.md` y
  `run_trace_control.jsonl`). La base responde con numeros genericos
  («13 estaciones», «13 tiendas, 57 aperturas», «15 millones»).
- Conclusion: 1 epoca de SFT sobre 446 QA cambio el estilo de las
  respuestas (numeros con pinta de informe) pero no la exactitud: 0/12 en
  ambos casos. La memoria parametrica a esta escala no fija cifras exactas.

## Comparacion con el grafo (declarada, no medida con numeros inventados)

- Grafo (Cypher sobre tripletas semilla): ~perfecto en nivel 1-2 **por construccion**, porque las respuestas son las tripletas verificadas. En nivel 3 depende de que el agente sepa agregar sin inventar.
- MEMO (demo): ver tabla. La memoria es un modelo entrenado, no una base de datos: puede equivocar cifras aunque haya visto el hecho en entrenamiento. Esa es justo la diferencia que el post debe mostrar.
- RAG vectorial: **pendiente** (no se corrio en este demo; no se inventan cifras).

## Notas de ground-truth

- Q3: Nota ground-truth: Segun el MD&A (p.112): 1,596 tiendas netas nuevas en 2024; 23,206 es OXXO Mexico (la division Proximidad Americas suma 26,322 puntos de venta). La cifra '1,236 aperturas' no aparece en el informe. Si el modelo responde 1,596, es mas correcto que la respuesta esperada.
- Q6: Nota ground-truth: Valores exactos del MD&A: 307,197 y 279,793 (la ontologia los redondea). Se acepta el valor exacto o el redondeado.

## Alucinaciones detectadas

- Q1: numeros no verificados en el informe: 114, 157968
- Q3: numeros no verificados en el informe: 137, 59
- Q4: numeros no verificados en el informe: 139
- Q6: numeros no verificados en el informe: 179536
- Q7: numeros no verificados en el informe: 135, 97
- Q12: numeros no verificados en el informe: 135
