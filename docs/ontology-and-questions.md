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
