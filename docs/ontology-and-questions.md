# Ontología + set de preguntas — FEMSA 2024 (RAG clásico vs agéntico)

Borrador de trabajo, preparado 2026-11-24 por la noche. No es el post; es la pieza que faltaba para correr la comparación. Fuente: Informe Anual Integrado 2024 de FEMSA, «Evolucionando» (PDF público, ~150 pp., https://informeanual.femsa.com/pdf/FEMSA_Informe-Anual-2024.pdf). Cifras verificadas contra el informe del director general y el comunicado de resultados 4T24 / año completo 2024 (FEMSA, 27 feb 2025).

## 1. Ontología (v1, cerrada: el extractor solo puede usar estos tipos)

### Tipos de entidad

| Tipo | Qué cubre | Ejemplos reales del informe |
|---|---|---|
| `Empresa` | La matriz y sus compañías | FEMSA, Coca-Cola FEMSA |
| `Division` | Segmentos reportables | Proximidad Américas, Proximidad Europa, Salud, Combustibles, Coca-Cola FEMSA, Spin |
| `Formato` | Unidades / marcas dentro de una división | OXXO México, OXXO Colombia / Chile / Perú, OXXO GAS, Spin by OXXO, Spin Premia, Juntos+ |
| `IndicadorFinanciero` | Métrica de resultados | Ingresos Totales, Utilidad de Operación, Utilidad Bruta, Ventas Mismas-Tiendas |
| `MetricaOperativa` | Métrica no financiera | Tiendas, Aperturas, Estaciones, Usuarios Activos, Volumen |
| `Periodo` | Ventana de medición | 2024, 4T24, 2023 (comparativo) |
| `Pais` | Mercado donde opera | México, Colombia, Chile, Perú, Suiza, Estados Unidos |
| `Iniciativa` | Estrategia o programa | FEMSA Forward, Digital@FEMSA → Spin, Juntos+ 4.0 |
| `Riesgo` | Solo si el informe lo nombra en gobierno/sostenibilidad | (sin instancias inventadas: se llena al extraer) |

### Relaciones (únicas permitidas)

- `tiene_division`: Empresa → Division
- `tiene_formato`: Division → Formato
- `opera_en`: Division / Formato → Pais
- `reporta`: Division → IndicadorFinanciero (con `valor`, `variacion_pct`, `periodo`)
- `mide`: Division → MetricaOperativa (con `valor`, `periodo`)
- `crecio_en`: IndicadorFinanciero → Periodo (variación vs año anterior)
- `impulsa`: Iniciativa → Division
- `evoluciono_a`: Iniciativa → Iniciativa (Digital@FEMSA → Spin)

### Reglas de extracción

1. Cada tripleta lleva `fuente`: sección y página del informe. Sin fuente, no entra al grafo.
2. Las tablas de estados financieros se extraen aparte; los números del texto narrativo no sustituyen a la tabla.
3. Ningún valor inventado: si el informe no da el dato, la tripleta no existe y la pregunta cae al RAG vectorial (ese es justo el caso que se evalúa).
4. Todo valor monetario en Ps. millones, con su periodo. Las variaciones son vs 2023 salvo que se diga 4T24.

## 2. Tripletas semilla (verificadas, listas para cargar)

```text
(FEMSA, tiene_division, Proximidad Américas)
(FEMSA, tiene_division, Proximidad Europa)
(FEMSA, tiene_division, Salud)
(FEMSA, tiene_division, Combustibles)
(FEMSA, tiene_division, Coca-Cola FEMSA)
(FEMSA, tiene_division, Spin)
(Proximidad Américas, tiene_formato, OXXO México)
(Proximidad Américas, opera_en, México)
(Proximidad Américas, opera_en, Colombia)
(Proximidad Américas, opera_en, Chile)
(Proximidad Américas, opera_en, Perú)
(Combustibles, tiene_formato, OXXO GAS)
(Spin, tiene_formato, Spin by OXXO)
(Spin, tiene_formato, Spin Premia)
(Coca-Cola FEMSA, tiene_formato, Juntos+)
(FEMSA, reporta, Ingresos Totales = Ps. 781,585 M, +11.2%, 2024)
(FEMSA, reporta, Utilidad de Operación = +19.8%, 2024)
(Proximidad Américas, reporta, Ingresos Totales = Ps. 307,200 M, +10.3%, 2024)
(Proximidad Américas, mide, Tiendas = 23,206, 2024)
(Proximidad Américas, mide, Aperturas = 1,236, 2024)
(Proximidad Europa, reporta, Ingresos Totales = Ps. 49,800 M, +14.2%, 2024)
(Salud, reporta, Ingresos Totales = Ps. 79,700 M, +5.8%, 2024)
(Combustibles, reporta, Ingresos Totales = Ps. 65,400 M, +11.7%, 2024)
(Combustibles, mide, Estaciones = 571, 2024)
(Coca-Cola FEMSA, reporta, Ingresos Totales = Ps. 279,800 M, +14.2%, 2024)
(Coca-Cola FEMSA, mide, Volumen = +4.4%, 2024)
(Coca-Cola FEMSA, mide, Clientes Juntos+ = 1.3 M, 2024)
(Spin by OXXO, mide, Usuarios Activos = 8.6 M, +24.9%, 2024)
(Spin Premia, mide, Usuarios Activos = 24.6 M, +27.5%, 2024)
(FEMSA Forward, impulsa, Proximidad Américas)
(Digital@FEMSA, evoluciono_a, Spin)
```

Nota de consistencia: las cinco divisiones con ingreso reportado suman ~Ps. 781,900 M, prácticamente el total consolidado (Ps. 781,585 M); la diferencia es redondeo. Participación aproximada: Proximidad Américas 39.3%, Coca-Cola FEMSA 35.8%, Salud 10.2%, Combustibles 8.4%, Proximidad Europa 6.4%.

## 3. Set de preguntas (12, en 3 niveles: ahí se ve la diferencia)

### Nivel 1 — Hecho directo (el grafo debería ganar por precisión y cita)
1. ¿Cuáles fueron los ingresos totales consolidados de FEMSA en 2024 y cuánto crecieron vs 2023? → Ps. 781,585 M, +11.2%.
2. ¿Cuántos usuarios activos tuvo Spin by OXXO en 2024? → 8.6 M (+24.9%).
3. ¿Cuántas tiendas sumó Proximidad Américas al cierre de 2024 y cuántas aperturas hubo? → 23,206 tiendas; 1,236 aperturas.
4. ¿Cuántas estaciones tenía OXXO GAS al cierre de 2024? → 571.

### Nivel 2 — Comparación / multi-salto (el grafo debería ganar; el RAG clásico tiende a mezclar cifras)
5. ¿Qué división tuvo el mayor crecimiento porcentual de ingresos en 2024: Proximidad Europa o Coca-Cola FEMSA? → Empate nominal: ambas +14.2% (trampa deliberada: el RAG suele inventar un ganador).
6. ¿Qué división aportó más ingresos en 2024, Proximidad Américas o Coca-Cola FEMSA, y por cuánto? → Proximidad Américas, Ps. 307,200 M vs Ps. 279,800 M.
7. ¿Qué creció más en usuarios, Spin by OXXO o Spin Premia? → Spin Premia en absoluto (24.6 M vs 8.6 M) y en % (+27.5% vs +24.9%).
8. ¿En qué países opera Proximidad Américas según el informe? → México, Colombia, Chile, Perú (además de negocios en Estados Unidos).

### Nivel 3 — Agregación / síntesis (aquí el híbrido grafo+RAG debería ganar; el RAG solo alucina el cálculo)
9. ¿Qué porcentaje de los ingresos consolidados 2024 representan Proximidad Américas y Coca-Cola FEMSA juntas? → ~75.1% (39.3% + 35.8%, aproximado por redondeo).
10. ¿Qué estrategia articula Retail, Bebidas y el ecosistema digital de FEMSA, y a través de qué se crea ese ecosistema? → FEMSA Forward; el ecosistema digital centrado en el cliente se crea a través de Spin.
11. ¿Cómo evolucionó la iniciativa digital de FEMSA en 2024? → Digital@FEMSA evolucionó a Spin.
12. ¿Qué división tuvo el año más complicado en 2024 según el propio informe, y cuánto crecieron sus ingresos? → Salud: «año complicado», +5.8% (el menor crecimiento de las divisiones reportadas).

## 4. Cómo se corre la comparación (diseño del experimento)

- Tres sistemas, mismas 12 preguntas: (a) RAG vectorial solo, (b) grafo solo (Cypher sobre esta ontología), (c) híbrido: grafo primero, RAG solo si el grafo no cubre.
- Métricas por pregunta: exactitud del número, cita correcta (sección/página), alucinación (número no presente en el informe = fallo). Con grafo, la alucinación en Nivel 1–2 debería tender a cero; el Nivel 3 mide si el agente sabe agregar sin inventar.
- Ejemplo de consulta grafo (pregunta 6): `MATCH (d:Division)-[:reporta]->(i:IndicadorFinanciero {nombre:'Ingresos Totales', periodo:'2024'}) RETURN d.nombre, i.valor ORDER BY i.valor DESC LIMIT 2`.
- Para el post: con NetworkX basta para el demo y las figuras; Neo4j solo si quiere escalar. Cada nodo guarda valor + cita.

## 5. Lo que falta (decisión de Cesar)

- ¿Le sirve esta ontología v1 así, o la ajustamos (p. ej. añadir `Directivo` con nombres de la carta, o `Riesgo` con instancias reales)?
- ¿Corro la extracción completa sobre el PDF para poblar el grafo, o lo hace él con su pipeline?
- El Nivel 2, pregunta 5, es una trampa intencional (+14.2% en ambas): ¿la dejamos como está para el post?

## 6. Set de evaluación ampliado: 100 preguntas (2026-11-25)

El set original de 12 preguntas (sección 3) se amplía a **100**: las 12
originales sin cambios (Q1–Q12) + **88 nuevas** (Q13–Q100), cada cifra
verificada por grep literal en `pipeline/data/chunks.jsonl` (151 páginas,
edición en inglés del informe anual FEMSA 2024). Generador y verificación
reproducible: `memo/build_eval_100.py` (falla si alguna cifra no aparece
en el corpus; las de nivel 3 se verifican por sus insumos + fórmula).
Archivo: `memo/eval_questions_100.json`; evidencia por pregunta para RAGAS:
`rag_baseline/ragas/ground_truth_evidence_100.json`.

**Reparto final (12 originales + 88 nuevas): Nivel 1: 40 · Nivel 2: 30 · Nivel 3: 30.**
Las listas de abajo son solo las 88 nuevas.

**Trampas nuevas** (empates o cifras parecidas reales, nunca inventadas):
Q49 (Spin +32.8% vs Premia +31.5% según el informe — es la cara del
conflicto de Q7, cuyas cifras +27.5%/+24.9% vienen del comunicado 4Q24),
Q62 (fatalidades 2024: empate 2 directos vs 2 contratistas; en 2023 era 9 vs 4),
Q86 (energía renovable 36.2% del total vs 65.3% de electricidad renovable).

**Conflictos de ground-truth nuevos** (además de Q3 y Q7): Q15 (utilidad neta:
tablas dicen Ps. 40,236 M, narrativa del MD&A p. 113 dice Ps. 41,687 M — se usa
la tabla) y Q17 (utilidad bruta: tablas Ps. 321,513 M, narrativa p. 112 dice
Ps. 321,417 M — se usa la tabla). La suma de ingresos por división (Q75,
Ps. 781,865 M) difiere del consolidado (Ps. 781,585 M) por 280 de
eliminaciones y otros.

**Solape con el entrenamiento MEMO:** 10 de las 88 nuevas (Q14, Q15, Q17, Q18,
Q21, Q22, Q23, Q33, Q41, Q45) tienen casi-duplicados (Jaccard ≥ 0.80) en
`reflection_qa.jsonl`/`v2` — son preguntas naturales sobre cifras titulares.
El notebook 03 regenera los datasets con el chequeo de fuga contra el set de
100, así que esos pares de entrenamiento se excluyen al re-entrenar en Colab.

### Nivel 1 — Hecho directo
13. ¿Cuál fue la utilidad de operación consolidada de FEMSA en 2024 y cuánto creció vs 2023? → Ps. 70,667 millones, +19.8%.
14. ¿Cuál fue el EBITDA consolidado de FEMSA en 2024? → Ps. 115,593 millones.
15. ¿Cuál fue la utilidad neta consolidada de FEMSA en 2024? → Ps. 40,236 millones. [Nota: La tabla del estado de resultados (pp. 108-109) dice Ps. 40,236 millones; la narrativa del MD&A (p. 113) dice Ps. 41,687 millones. Se usa el valor de la tabla.]
16. ¿Cuáles fueron los ingresos de cada división de FEMSA en 2024? → Proximidad Américas Ps. 307,197 millones; Coca-Cola FEMSA Ps. 279,793 millones; Salud Ps. 79,755 millones; Combustibles Ps. 65,365 millones; Proximidad Europa Ps. 49,755 millones.
17. ¿Cuál fue la utilidad bruta consolidada de FEMSA en 2024? → Ps. 321,513 millones. [Nota: El estado de resultados y la tabla del MD&A (pp. 109, 112) dicen Ps. 321,513 millones; la narrativa del MD&A (p. 112) dice Ps. 321,417 millones. Se usa el valor de la tabla.]
18. ¿Cuál fue el margen bruto consolidado de FEMSA en 2024? → 41.1%.
19. ¿Cuánto sumaron los activos totales de FEMSA al cierre de 2024? → Ps. 851,536 millones.
20. ¿Cuánto invirtió FEMSA en CAPEX en 2024 y cuánto creció vs 2023? → Ps. 51,074 millones, +31.1%.
21. ¿Cuánto efectivo y equivalentes tenía FEMSA al cierre de 2024? → Ps. 139,834 millones.
22. ¿Cuál fue la utilidad neta de la participación controladora en 2024? → Ps. 26,735 millones.
23. ¿Cuál fue el volumen de Coca-Cola FEMSA en 2024? → 4,224.6 millones de cajas unidad.
24. ¿Cuál fue la utilidad bruta de Coca-Cola FEMSA y de Proximidad Américas en 2024? → Coca-Cola FEMSA Ps. 128,736 millones; Proximidad Américas Ps. 136,993 millones.
25. ¿Cuánto crecieron las ventas mismas tiendas de Proximidad Américas en 2024? → +4.2%.
26. ¿Cuánto crecieron las ventas mismas estaciones de Combustibles (OXXO GAS) en 2024? → +9.9%.
27. ¿Cuántas tiendas OXXO había en total al cierre de 2024? → 24,462 tiendas.
28. ¿Cuántos puntos de venta tenía Proximidad Américas al cierre de 2024? → 26,322 puntos de venta.
29. ¿Cuántas farmacias tenía FEMSA Salud al cierre de 2024? → 4,661 farmacias.
30. ¿Cuántos puntos de venta tenía Proximidad Europa al cierre de 2024? → 2,778 puntos de venta.
31. ¿Cuántos colaboradores tenía FEMSA en total en 2024? → 388,999 colaboradores.
32. ¿En cuántos países opera FEMSA según el informe anual 2024? → 18 países.
33. ¿Cuántos clientes tenía Juntos+ en 2024? → 1.3 millones de clientes.
34. ¿Cuántos usuarios tenía Premia Juntos+ en 2024? → Más de 1.1 millones de usuarios.
35. ¿Cuántas plantas embotelladoras y cuántos centros de distribución tenía Coca-Cola FEMSA en 2024? → 56 plantas embotelladoras y 256 centros de distribución.
36. ¿Cuántas líneas de producción nuevas instaló Coca-Cola FEMSA en 2024? → 7 líneas nuevas.
37. ¿Cuál fue la tasa efectiva de impuestos de FEMSA en 2024? → 35.9%.
38. ¿Cuántas transacciones mensuales procesaba Spin by OXXO en 2024? → Más de 63 millones de transacciones mensuales.
39. ¿Cuántos clientes nuevos sumó Spin by OXXO en 2024? → Más de 13 millones de clientes.
40. ¿Cuántos usuarios nuevos sumó Spin Premia en 2024? → Más de 52 millones de usuarios.
41. ¿Cuántas tazas de café andatti se servían al día en 2024? → Más de 811,000 tazas al día.
42. ¿Cuántas tiendas OXXO había en Colombia al cierre de 2024? → 569 tiendas.
43. ¿Cuántas tiendas OXXO había en México al cierre de 2024? → 23,206 tiendas.
44. ¿Cuál fue el margen de utilidad de operación consolidado de FEMSA en 2024? → 9.0%.
45. ¿Cuál fue el margen EBITDA consolidado de FEMSA en 2024? → 14.8%.
46. ¿Cuánto fueron las ventas netas consolidadas de FEMSA en 2024? → Ps. 775,551 millones.
47. ¿Cuánto fue el costo de ventas consolidado de FEMSA en 2024? → Ps. 460,072 millones.
48. ¿Cuánto sumaron los pasivos totales de FEMSA al cierre de 2024? → Ps. 470,405 millones.

### Nivel 2 — Comparación / multi-salto
49. ¿Cuál creció más en usuarios en 2024 según el informe anual, Spin by OXXO o Spin Premia, y cuánto creció cada uno? → Spin by OXXO creció más: +32.8% vs +31.5% de Spin Premia. [Nota: TRAMPA DOCUMENTADA: estas son las cifras del informe anual (p. 34). La Q7 espera +27.5% / +24.9%, que vienen del comunicado 4Q24, no del informe. Ambas preguntas se mantienen para medir el conflicto.]
50. ¿Qué división tuvo el mayor margen bruto en 2024? → Coca-Cola FEMSA, 46.0%.
51. ¿Qué división tuvo el menor margen bruto en 2024? → Combustibles, 12.1%.
52. ¿Qué división tuvo los menores ingresos en 2024? → Proximidad Europa, Ps. 49,755 millones.
53. ¿Qué división tuvo el mayor crecimiento de gastos de administración en 2024? → Salud, +56.0%.
54. ¿En qué país tenía FEMSA más colaboradores en 2024? → México, 268,407 colaboradores.
55. ¿Quién tenía más puntos de venta, Coca-Cola FEMSA o Proximidad Américas? → Coca-Cola FEMSA, con ~2.2 millones de puntos de venta, frente a 26,322 de Proximidad Américas.
56. ¿En qué país consumió FEMSA más energía renovable (MWh) en 2024? → México, 1,724,031 MWh.
57. ¿Qué material de empaque tenía el mayor porcentaje reciclado en 2024? → Aluminio, 80% reciclado.
58. ¿En qué tema se impartieron más horas de capacitación en 2024? → Conocimiento técnico, 4,242,222 horas.
59. ¿Cuál fue la desinversión más grande de FEMSA en 2024? → Imbera / Torrey, Ps. 8,000 millones.
60. ¿Cuál fue la recompra acelerada de acciones (ASR) más grande de 2024, la de marzo o la de junio? → La de junio, hasta US$ 600 millones (la de marzo fue de US$ 400 millones).
61. ¿Hubo más horas de capacitación en 2023 o en 2024, y cuántas en cada año? → Más en 2023: 9,787,020 horas vs 6,181,088 en 2024.
62. ¿Hubo más fatalidades entre colaboradores directos o entre contratistas en 2024? → Empate: 2 fatalidades de colaboradores directos y 2 de contratistas. [Nota: TRAMPA: la respuesta es un empate (2 y 2). En 2023 no lo era (9 directos vs 4 contratistas): quien responda con las cifras de 2023 falla.]
63. ¿Qué fue mayor en 2024, las emisiones de Alcance 1 o las de Alcance 2? → Alcance 1: 1,003,012 t CO2eq vs 444,618 t de Alcance 2 (base de mercado).
64. ¿En qué país tenía FEMSA Salud más farmacias en 2024? → México, 1,739 farmacias.
65. ¿En qué país tenía Valora más tiendas en 2024? → Alemania, 1,376 tiendas.
66. ¿Qué indicador financiero creció más en 2024, los ingresos (+11.2%), la utilidad de operación (+19.8%) o el EBITDA (+20.6%)? → El EBITDA, +20.6%.
67. ¿Qué cayó más en 2024, la utilidad neta consolidada (-47.5%) o la de la participación controladora (-59.3%)? → La de la participación controladora, -59.3% (vs -47.5% consolidada).
68. ¿Quién tenía más clientes en 2024, Juntos+ o Premia Juntos+? → Juntos+, 1.3 millones de clientes, frente a más de 1.1 millones de Premia Juntos+.
69. ¿Qué porcentaje del Consejo de Administración era independiente y qué porcentaje eran mujeres en 2024? → 46% independientes y 40% mujeres.
70. ¿Qué fue mayor en Spin by OXXO en 2024, los clientes nuevos (+13 millones) o los usuarios activos (8.6 millones)? → Los clientes nuevos: más de 13 millones, frente a 8.6 millones de usuarios activos.
71. ¿En qué país tenía FEMSA más colaboradores después de México, Brasil o Colombia? → Brasil, 46,016 colaboradores (Colombia: 25,563).
72. ¿Qué era mayor en 2024, las tiendas OXXO en México (23,206) o las farmacias de FEMSA Salud en total (4,661)? → Las tiendas OXXO en México: 23,206 frente a 4,661 farmacias.
73. ¿En qué pilar invirtió más FEMSA en 2024 según el informe de sostenibilidad? → En Personas (People), Ps. 3,855 millones.
74. ¿Qué meta 2030 estaba más cerca de cumplirse en 2024, la de compromiso de colaboradores (90%) o la de proveedores locales (90%)? → La de compromiso de colaboradores: 88.5% en 2024 (proveedores locales: 77%).

### Nivel 3 — Agregación / síntesis
75. ¿Cuánto suman los ingresos de las cinco divisiones de FEMSA en 2024? → Ps. 781,865 millones (la suma de las divisiones; difiere del consolidado de Ps. 781,585 millones por eliminaciones y otros). [La suma (781,865) difiere del consolidado (781,585) por 280: eliminaciones y otros. El modelo debe dar la suma de las divisiones.]
76. ¿Cuántas tiendas OXXO había en total si sumas las de México, Colombia, Chile, Perú y Estados Unidos? → 24,462 tiendas.
77. ¿Cuántas farmacias tenía FEMSA Salud en total si sumas las de México, Chile, Ecuador y Colombia? → 4,661 farmacias.
78. ¿Cuántos puntos de venta tenía Proximidad Europa en total si sumas Alemania, Suiza, Luxemburgo, Países Bajos y Austria? → 2,778 puntos de venta.
79. ¿Cuántos usuarios activos tenía Spin en total en 2024 (Spin by OXXO + Spin Premia)? → 33.2 millones de usuarios activos.
80. ¿Cuántos clientes sumaban Juntos+ y Premia Juntos+ en 2024? → 2.4 millones de clientes.
81. ¿Qué porcentaje de la utilidad bruta consolidada aportó Coca-Cola FEMSA en 2024? → 40.0% (128,736 de 321,513). [TRAMPA DE PORCENTAJE: el 40.0% es la participación en la utilidad bruta, no un margen.]
82. ¿Por cuánto superaron los ingresos de Proximidad Américas a los de Coca-Cola FEMSA en 2024? → Ps. 27,404 millones.
83. ¿Cuánto suman las ventas netas y las otras partidas que llegan a los ingresos totales de 2024 (781,585 menos 775,551)? → Ps. 6,034 millones.
84. ¿Cuánto suman los pasivos y el capital para llegar a los activos totales de 2024? → Ps. 851,536 millones (470,405 + 381,131).
85. ¿Cuánto suman los pasivos corrientes y no corrientes de FEMSA en 2024? → Ps. 470,405 millones (202,930 + 267,475).
86. ¿Qué porcentaje de la energía total consumida por FEMSA en 2024 era renovable? → 36.2% (7,588,556 de 20,948,731 GJ). [TRAMPA: el 65.3% del informe es electricidad renovable, NO energía total. La respuesta correcta es 36.2%.]
87. ¿Qué porcentaje de los materiales usados por FEMSA en 2024 era reciclado? → 36.9% (214,902 de 582,453 toneladas).
88. ¿Qué porcentaje del agua extraída por FEMSA en 2024 era subterránea? → 52.6% (22,494 de 42,734 miles de m³).
89. ¿Cuánto sumaron las emisiones de Alcance 1 y Alcance 2 de FEMSA en 2024? → 1,447,630 t CO2eq (1,003,012 + 444,618).
90. ¿Cuánta agua extrajo FEMSA en total en 2024 si sumas la subterránea, la de terceros y la superficial? → 42,734 miles de m³ (22,494 + 19,464 + 776).
91. ¿Cuántos residuos generó FEMSA en total en 2024 (peligrosos + no peligrosos)? → 326,038 toneladas (4,705 + 321,333).
92. ¿Cuánta energía consumió FEMSA en total en 2024 (renovable + no renovable)? → 20,948,731 GJ (7,588,556 + 13,360,175).
93. ¿Cuántos materiales usó FEMSA en total en 2024 (vírgenes + reciclados)? → 582,453 toneladas (367,551 + 214,902).
94. ¿Cuántos colaboradores sumaban los tres países con más personal de FEMSA en 2024 (México, Brasil y Colombia)? → 339,986 colaboradores (268,407 + 46,016 + 25,563).
95. ¿Cuántos colaboradores tenía FEMSA fuera de México en 2024? → 120,592 colaboradores (388,999 − 268,407).
96. ¿Cuánto sumaron las tres desinversiones de FEMSA en 2024 (Imbera/Torrey, Solistica y plásticos)? → Ps. 15,225 millones (8,000 + 4,060 + 3,165).
97. ¿Cuánto sumaron las dos recompras aceleradas de acciones (ASR) de FEMSA en 2024? → US$ 1,000 millones (400 + 600).
98. ¿Por cuántos millones superaba Spin Premia a Spin by OXXO en usuarios activos en 2024? → 16.0 millones (24.6 − 8.6).
99. ¿Cuántas horas de capacitación sumaron cultura/liderazgo y conocimiento técnico en 2024, y qué porcentaje del total representan? → 4,979,608 horas (737,386 + 4,242,222), el 80.6% del total de 6,181,088. [Dos resultados: la suma (4,979,608) y su participación (80.6%).]
100. ¿Cuántas tiendas OXXO había en Chile y Perú juntas al cierre de 2024? → 438 tiendas (235 + 203).
