# Resultados MEMO — Run 2, control (FEMSA 2024)

Sistema (control run 2): EXECUTIVE = **Qwen2.5-1.5B-Instruct** base
(congelado); MEMORY = base **sin entrenar**. El EXECUTIVE nunca ve el
corpus; solo las sub-respuestas de MEMORY. Descomposición: LLM en 12/12.

Correctas: **0/12**. Con alucinacion (numero no verificado en el informe): **6/12**.

| ID | Nivel | Pregunta | Esperado | Respuesta MEMO | Veredicto |
|----|-------|----------|----------|----------------|-----------|
| Q1 | 1 | ¿Cuáles fueron los ingresos totales consolidados de FEMSA en 2024 y cu | Ps. 781,585 millones, +11.2%. | Los ingresos totales consolidados de FEMSA en 2024 fueron de $18.934 millones. Esto representa un crecimiento de 29% con respecto al año anterior (2023), ya que el ingreso en 2023 fue de $14.240 millones. | incorrecta |
| Q2 | 1 | ¿Cuántos usuarios activos tuvo Spin by OXXO en 2024? | 8.6 millones (+24.9%). | Spin by OXXO tuvo un total de 15 millones de usuarios activos en 2024. | incorrecta |
| Q3 | 1 | ¿Cuántas tiendas sumó Proximidad Américas al cierre de 2024 y cuántas  | 23,206 tiendas; 1,236 aperturas. | Proximidad Américas sumó 135 tiendas al cierre de 2024. Se abrieron 137 nuevas tiendas para fines de año 2024. | incorrecta |
| Q4 | 1 | ¿Cuántas estaciones tenía OXXO GAS al cierre de 2024? | 571. | El número total de estaciones de OXXO GAS en 2024 fue de 13. | incorrecta |
| Q5 | 2 | ¿Qué división tuvo el mayor crecimiento porcentual de ingresos en 2024 | Empate nominal: ambas +14.2% (trampa deliberada). | Para determinar cuál división tuvo el mayor crecimiento porcentual de ingresos en 2024, necesitamos comparar las tasas de crecimiento de cada una:  1. Proximidad Europa: Ingresos totales en 2024 fueron de $13,5 biliones. | incorrecta |
| Q6 | 2 | ¿Qué división aportó más ingresos en 2024, Proximidad Américas o Coca- | Proximidad Américas, Ps. 307,200 millones vs Ps. 279,800 mil | En 2024, Proximidad Américas aportó un ingreso de $13.5 biliones, mientras que Coca-Cola FEMSA aportó $15.367 millones. Por lo tanto, Proximidad Américas aportó más ingresos en 2024 con un total de $13.5 biliones frente  | incorrecta |
| Q7 | 2 | ¿Qué creció más en usuarios, Spin by OXXO o Spin Premia? | Spin Premia en absoluto (24.6 millones vs 8.6 millones) y en | Spin by OXXO creció más en usuarios que Spin Premia en 2024. Spin by OXXO aumentó 13%, mientras que Spin Premia creció solo 5%. | incorrecta |
| Q8 | 2 | ¿En qué países opera Proximidad Américas según el informe? | México, Colombia, Chile, Perú (además de negocios en Estados | Según el informe anual de FEMSA 2024, Proximidad Américas opera en varios países del Caribe Central y otros mercados relevantes. Sin embargo, no tengo acceso a detalles específicos sobre los países exactos o a una lista  | incorrecta |
| Q9 | 3 | ¿Qué porcentaje de los ingresos consolidados 2024 representan Proximid | ~75.1% (39.3% + 35.8%, aproximado por redondeo). | El 73% de los ingresos consolidados de FEMSA en 2024 se derivan de Proximidad Américas y Coca-Cola FEMSA juntos. | incorrecta |
| Q10 | 3 | ¿Qué estrategia articula Retail, Bebidas y el ecosistema digital de FE | FEMSA Forward; el ecosistema digital centrado en el cliente  | Retail, Bebidas y el ecosistema digital de FEMSA utilizan la Estrategia Retail + Bebidas + Ecosistemas Digitales para impulsar su crecimiento. Este ecosistema se crea a través de la interacción entre organismos y su ento | incorrecta |
| Q11 | 3 | ¿Cómo evolucionó la iniciativa digital de FEMSA en 2024? | Digital@FEMSA evolucionó a Spin. | FEMSA implementó nuevas tecnologías y estrategias para mejorar la digitalización en 2024, pero no proporciona detalles sobre los resultados específicos ni cómo se adaptaron a las tendencias actuales del mercado digital. | incorrecta |
| Q12 | 3 | ¿Qué división tuvo el año más complicado en 2024 según el propio infor | Salud: 'año complicado', +5.8% (el menor crecimiento de las  | La división que tuvo el año más complicado en 2024 fue la de Energía, con una crecimiento de sus ingresos a $13,596 millones. | incorrecta |

## Por nivel

- Nivel 1: 0/4 correctas.
- Nivel 2: 0/4 correctas.
- Nivel 3: 0/4 correctas.

## Comparacion con el grafo (declarada, no medida con numeros inventados)

- Grafo (Cypher sobre tripletas semilla): ~perfecto en nivel 1-2 **por construccion**, porque las respuestas son las tripletas verificadas. En nivel 3 depende de que el agente sepa agregar sin inventar.
- MEMO (demo): ver tabla. La memoria es un modelo entrenado, no una base de datos: puede equivocar cifras aunque haya visto el hecho en entrenamiento. Esa es justo la diferencia que el post debe mostrar.
- RAG vectorial: **pendiente** (no se corrio en este demo; no se inventan cifras).

## Notas de ground-truth

- Q3: Nota ground-truth: Segun el MD&A (p.112): 1,596 tiendas netas nuevas en 2024; 23,206 es OXXO Mexico (la division Proximidad Americas suma 26,322 puntos de venta). La cifra '1,236 aperturas' no aparece en el informe. Si el modelo responde 1,596, es mas correcto que la respuesta esperada.
- Q6: Nota ground-truth: Valores exactos del MD&A: 307,197 y 279,793 (la ontologia los redondea). Se acepta el valor exacto o el redondeado.

## Alucinaciones detectadas

- Q1: numeros no verificados en el informe: 14240, 18934, 29
- Q3: numeros no verificados en el informe: 135, 137
- Q5: numeros no verificados en el informe: 135
- Q6: numeros no verificados en el informe: 135, 15367
- Q9: numeros no verificados en el informe: 73
- Q12: numeros no verificados en el informe: 13596
