// queries.cypher — las 12 preguntas del set de evaluación (ontología v1)
// traducidas a Cypher. Cada consulta lleva encima la pregunta en español
// y la respuesta esperada. Corre con: relate (ver README.md) o Neo4j.
//
// Convención: relaciones en UPPER_SNAKE_CASE (ontología: minúsculas).
// Los valores (valor, variacion_pct) viven en la RELACIÓN REPORTA/MIDE,
// no en el nodo del indicador.

// ── Nivel 1 — Hecho directo ──────────────────────────────────────────────

// Q1. ¿Cuáles fueron los ingresos totales consolidados de FEMSA en 2024
//     y cuánto crecieron vs 2023?
//     Esperado: Ps. 781,585 M, +11.2%.
MATCH (e:Empresa {nombre: 'FEMSA'})-[r:REPORTA]->(i:IndicadorFinanciero {nombre: 'Ingresos Totales'})
RETURN r.valor AS ingresos_totales_Ps_millones, r.unidad AS unidad,
       r.variacion_pct AS crecimiento_pct_vs_2023, r.periodo AS periodo,
       r.fuente_pagina AS pagina_fuente;

// Q2. ¿Cuántos usuarios activos tuvo Spin by OXXO en 2024?
//     Esperado: 8.6 M (+24.9%).
MATCH (f:Formato {nombre: 'Spin by OXXO'})-[r:MIDE]->(m:MetricaOperativa {nombre: 'Usuarios Activos'})
RETURN r.valor AS usuarios_activos_millones, r.variacion_pct AS crecimiento_pct,
       r.periodo AS periodo, r.fuente_pagina AS pagina_fuente;

// Q3. ¿Cuántas tiendas sumó Proximidad Américas al cierre de 2024
//     y cuántas aperturas hubo?
//     Esperado: 23,206 tiendas; 1,236 aperturas.
//     (Nota: 23,206 es OXXO México; ver la nota de la tripleta en
//     triples_seed.json.)
MATCH (d:Division {nombre: 'Proximidad Américas'})-[r:MIDE]->(m:MetricaOperativa)
WHERE m.nombre IN ['Tiendas', 'Aperturas']
RETURN m.nombre AS metrica, r.valor AS valor, r.unidad AS unidad,
       r.periodo AS periodo;

// Q4. ¿Cuántas estaciones tenía OXXO GAS al cierre de 2024?
//     Esperado: 571.
MATCH (d:Division {nombre: 'Combustibles'})-[r:MIDE]->(m:MetricaOperativa {nombre: 'Estaciones'})
RETURN r.valor AS estaciones, r.periodo AS periodo, r.fuente_pagina AS pagina_fuente;

// ── Nivel 2 — Comparación / multi-salto ──────────────────────────────────

// Q5. ¿Qué división tuvo el mayor crecimiento porcentual de ingresos en
//     2024: Proximidad Europa o Coca-Cola FEMSA?
//     Esperado: EMPATE nominal, ambas +14.2% (trampa deliberada: el RAG
//     clásico suele inventar un ganador).
MATCH (d:Division)-[r:REPORTA]->(i:IndicadorFinanciero {nombre: 'Ingresos Totales'})
WHERE d.nombre IN ['Proximidad Europa', 'Coca-Cola FEMSA']
RETURN d.nombre AS division, r.variacion_pct AS crecimiento_pct
ORDER BY r.variacion_pct DESC;

// Q6. ¿Qué división aportó más ingresos en 2024, Proximidad Américas o
//     Coca-Cola FEMSA, y por cuánto?
//     Esperado: Proximidad Américas, Ps. 307,200 M vs Ps. 279,800 M
//     (diferencia Ps. 27,400 M).
MATCH (d:Division)-[r:REPORTA]->(i:IndicadorFinanciero {nombre: 'Ingresos Totales'})
WHERE d.nombre IN ['Proximidad Américas', 'Coca-Cola FEMSA']
RETURN d.nombre AS division, r.valor AS ingresos_Ps_millones
ORDER BY r.valor DESC;

// Q7. ¿Qué creció más en usuarios, Spin by OXXO o Spin Premia?
//     Esperado: Spin Premia en absoluto (24.6 M vs 8.6 M)
//     y en % (+27.5% vs +24.9%).
MATCH (f:Formato)-[r:MIDE]->(m:MetricaOperativa {nombre: 'Usuarios Activos'})
WHERE f.nombre IN ['Spin by OXXO', 'Spin Premia']
RETURN f.nombre AS formato, r.valor AS usuarios_millones,
       r.variacion_pct AS crecimiento_pct
ORDER BY r.valor DESC;

// Q8. ¿En qué países opera Proximidad Américas según el informe?
//     Esperado: México, Colombia, Chile, Perú
//     (además de negocios en Estados Unidos).
MATCH (d:Division {nombre: 'Proximidad Américas'})-[:OPERA_EN]->(p:Pais)
RETURN p.nombre AS pais
ORDER BY p.nombre;

// ── Nivel 3 — Agregación / síntesis ──────────────────────────────────────

// Q9. ¿Qué porcentaje de los ingresos consolidados 2024 representan
//     Proximidad Américas y Coca-Cola FEMSA juntas?
//     Esperado: ~75.1% (39.3% + 35.8%, aproximado por redondeo).
MATCH (e:Empresa {nombre: 'FEMSA'})-[rc:REPORTA]->(ic:IndicadorFinanciero {nombre: 'Ingresos Totales'})
WITH rc.valor AS total_consolidado
MATCH (d:Division)-[r:REPORTA]->(i:IndicadorFinanciero {nombre: 'Ingresos Totales'})
WHERE d.nombre IN ['Proximidad Américas', 'Coca-Cola FEMSA']
RETURN total_consolidado AS total_Ps_millones,
       sum(r.valor) AS suma_divisiones_Ps_millones,
       sum(r.valor) * 100.0 / total_consolidado AS pct_del_total;

// Q10. ¿Qué estrategia articula Retail, Bebidas y el ecosistema digital
//      de FEMSA, y a través de qué se crea ese ecosistema?
//      Esperado: FEMSA Forward; el ecosistema digital centrado en el
//      cliente se crea a través de Spin.
MATCH (i:Iniciativa {nombre: 'FEMSA Forward'})-[:IMPULSA]->(d:Division)
OPTIONAL MATCH (i2:Iniciativa {nombre: 'Digital@FEMSA'})-[:EVOLUCIONO_A]->(x)
RETURN collect(DISTINCT d.nombre) AS divisiones_articuladas,
       x.nombre AS ecosistema_creado_a_traves_de;

// Q11. ¿Cómo evolucionó la iniciativa digital de FEMSA en 2024?
//      Esperado: Digital@FEMSA evolucionó a Spin.
MATCH (i:Iniciativa {nombre: 'Digital@FEMSA'})-[r:EVOLUCIONO_A]->(x)
RETURN i.nombre AS de, x.nombre AS a, labels(x) AS tipo_destino;

// Q12. ¿Qué división tuvo el año más complicado en 2024 según el propio
//      informe, y cuánto crecieron sus ingresos?
//      Esperado: Salud, +5.8% (el menor crecimiento de las divisiones).
//      (El calificativo "año complicado" es del texto del informe; el
//      grafo solo demuestra que Salud tuvo el menor crecimiento.)
MATCH (d:Division)-[r:REPORTA]->(i:IndicadorFinanciero {nombre: 'Ingresos Totales'})
RETURN d.nombre AS division, r.variacion_pct AS crecimiento_pct
ORDER BY r.variacion_pct ASC
LIMIT 1;
