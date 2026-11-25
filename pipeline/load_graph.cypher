// load_graph.cypher — generado por build_cypher.py
// Fuente: triples_seed.json (ontología v1, cifras verificadas)
// Todo es MERGE: se puede correr varias veces sin duplicar.
// Relaciones en UPPER_SNAKE_CASE (ontología: minúsculas).

// (Digital@FEMSA, evoluciono_a, Spin)
MERGE (s1:Iniciativa {nombre: 'Digital@FEMSA'});
MERGE (o1:Division {nombre: 'Spin'});
MERGE (s1)-[:EVOLUCIONO_A {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 5, nota: 'La carta dice literalmente \'Digital@FEMSA becomes Spin\'. El objeto es la división Spin (la ontología v1 lo modela así).'}]->(o1);

// (FEMSA Forward, impulsa, Proximidad Américas)
MERGE (s2:Iniciativa {nombre: 'FEMSA Forward'});
MERGE (o2:Division {nombre: 'Proximidad Américas'});
MERGE (s2)-[:IMPULSA {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 5, nota: 'Relación de la ontología v1: el informe presenta FEMSA Forward como la estrategia que articula las verticales (carta del director general).'}]->(o2);

// (Coca-Cola FEMSA, mide, Clientes Juntos+)
MERGE (s3:Division {nombre: 'Coca-Cola FEMSA'});
MERGE (o3:MetricaOperativa {nombre: 'Clientes Juntos+'});
MERGE (s3)-[:MIDE {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 19, valor: 1.3, unidad: 'millones de clientes', periodo: '2024'}]->(o3);

// (Coca-Cola FEMSA, mide, Volumen)
MERGE (s4:Division {nombre: 'Coca-Cola FEMSA'});
MERGE (o4:MetricaOperativa {nombre: 'Volumen'});
MERGE (s4)-[:MIDE {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 27, valor: 4.4, unidad: '% crecimiento vs 2023', periodo: '2024'}]->(o4);

// (Combustibles, mide, Estaciones)
MERGE (s5:Division {nombre: 'Combustibles'});
MERGE (o5:MetricaOperativa {nombre: 'Estaciones'});
MERGE (s5)-[:MIDE {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 21, valor: 571, unidad: 'estaciones', periodo: '2024', nota: '571 estaciones OXXO GAS en México (key data, p. 21).'}]->(o5);

// (Proximidad Américas, mide, Aperturas)
MERGE (s6:Division {nombre: 'Proximidad Américas'});
MERGE (o6:MetricaOperativa {nombre: 'Aperturas'});
MERGE (s6)-[:MIDE {fuente_documento: 'Ontología v1 (verificación de página pendiente)', valor: 1236, unidad: 'aperturas', periodo: '2024', nota: 'Cifra de la ontología v1 (comunicado 4T24). NO localizada en el texto de la edición en inglés: el informe reporta 1,596 aperturas netas en 2024 (p. 23/112/114). Verificar contra la edición en español antes de citarla como \'aperturas\' sin más.'}]->(o6);

// (Proximidad Américas, mide, Tiendas)
MERGE (s7:Division {nombre: 'Proximidad Américas'});
MERGE (o7:MetricaOperativa {nombre: 'Tiendas'});
MERGE (s7)-[:MIDE {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 21, valor: 23206, unidad: 'tiendas', periodo: '2024', nota: '23,206 = tiendas OXXO en México (key data, p. 21). Total Proximidad Américas: 26,322 puntos de venta; total OXXO: 24,462 tiendas (p. 23).'}]->(o7);

// (Spin Premia, mide, Usuarios Activos)
MERGE (s8:Formato {nombre: 'Spin Premia'});
MERGE (o8:MetricaOperativa {nombre: 'Usuarios Activos'});
MERGE (s8)-[:MIDE {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 33, valor: 24.6, unidad: 'millones de usuarios', variacion_pct: 27.5, periodo: '2024', nota: 'El +27.5% viene del comunicado de resultados 4T24 (vs 4T23); no aparece en el texto del informe anual.'}]->(o8);

// (Spin by OXXO, mide, Usuarios Activos)
MERGE (s9:Formato {nombre: 'Spin by OXXO'});
MERGE (o9:MetricaOperativa {nombre: 'Usuarios Activos'});
MERGE (s9)-[:MIDE {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 33, valor: 8.6, unidad: 'millones de usuarios', variacion_pct: 24.9, periodo: '2024', nota: 'El +24.9% viene del comunicado de resultados 4T24 (vs 4T23); no aparece en el texto del informe anual.'}]->(o9);

// (Proximidad Américas, opera_en, Chile)
MERGE (s10:Division {nombre: 'Proximidad Américas'});
MERGE (o10:Pais {nombre: 'Chile'});
MERGE (s10)-[:OPERA_EN {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 21}]->(o10);

// (Proximidad Américas, opera_en, Colombia)
MERGE (s11:Division {nombre: 'Proximidad Américas'});
MERGE (o11:Pais {nombre: 'Colombia'});
MERGE (s11)-[:OPERA_EN {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 21}]->(o11);

// (Proximidad Américas, opera_en, México)
MERGE (s12:Division {nombre: 'Proximidad Américas'});
MERGE (o12:Pais {nombre: 'México'});
MERGE (s12)-[:OPERA_EN {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 21}]->(o12);

// (Proximidad Américas, opera_en, Perú)
MERGE (s13:Division {nombre: 'Proximidad Américas'});
MERGE (o13:Pais {nombre: 'Perú'});
MERGE (s13)-[:OPERA_EN {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 21}]->(o13);

// (Coca-Cola FEMSA, reporta, Ingresos Totales)
MERGE (s14:Division {nombre: 'Coca-Cola FEMSA'});
MERGE (o14:IndicadorFinanciero {nombre: 'Ingresos Totales'});
MERGE (s14)-[:REPORTA {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 112, valor: 279800, unidad: 'Ps. millones', variacion_pct: 14.2, periodo: '2024', valor_exacto_informe: 279793}]->(o14);

// (Combustibles, reporta, Ingresos Totales)
MERGE (s15:Division {nombre: 'Combustibles'});
MERGE (o15:IndicadorFinanciero {nombre: 'Ingresos Totales'});
MERGE (s15)-[:REPORTA {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 112, valor: 65400, unidad: 'Ps. millones', variacion_pct: 11.7, periodo: '2024', valor_exacto_informe: 65365}]->(o15);

// (FEMSA, reporta, Ingresos Totales)
MERGE (s16:Empresa {nombre: 'FEMSA'});
MERGE (o16:IndicadorFinanciero {nombre: 'Ingresos Totales'});
MERGE (s16)-[:REPORTA {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 112, valor: 781585, unidad: 'Ps. millones', variacion_pct: 11.2, periodo: '2024'}]->(o16);

// (FEMSA, reporta, Utilidad de Operación)
MERGE (s17:Empresa {nombre: 'FEMSA'});
MERGE (o17:IndicadorFinanciero {nombre: 'Utilidad de Operación'});
MERGE (s17)-[:REPORTA {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 108, unidad: 'Ps. millones', variacion_pct: 19.8, periodo: '2024', valor_exacto_informe: 70667, nota: 'La ontología v1 solo registra la variación (+19.8%). El valor absoluto Ps. 70,667 M es del MD&A (p. 108).'}]->(o17);

// (Proximidad Américas, reporta, Ingresos Totales)
MERGE (s18:Division {nombre: 'Proximidad Américas'});
MERGE (o18:IndicadorFinanciero {nombre: 'Ingresos Totales'});
MERGE (s18)-[:REPORTA {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 112, valor: 307200, unidad: 'Ps. millones', variacion_pct: 10.3, periodo: '2024', valor_exacto_informe: 307197}]->(o18);

// (Proximidad Europa, reporta, Ingresos Totales)
MERGE (s19:Division {nombre: 'Proximidad Europa'});
MERGE (o19:IndicadorFinanciero {nombre: 'Ingresos Totales'});
MERGE (s19)-[:REPORTA {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 112, valor: 49800, unidad: 'Ps. millones', variacion_pct: 14.2, periodo: '2024', valor_exacto_informe: 49755}]->(o19);

// (Salud, reporta, Ingresos Totales)
MERGE (s20:Division {nombre: 'Salud'});
MERGE (o20:IndicadorFinanciero {nombre: 'Ingresos Totales'});
MERGE (s20)-[:REPORTA {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 112, valor: 79700, unidad: 'Ps. millones', variacion_pct: 5.8, periodo: '2024', valor_exacto_informe: 79755}]->(o20);

// (FEMSA, tiene_division, Coca-Cola FEMSA)
MERGE (s21:Empresa {nombre: 'FEMSA'});
MERGE (o21:Division {nombre: 'Coca-Cola FEMSA'});
MERGE (s21)-[:TIENE_DIVISION {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 21}]->(o21);

// (FEMSA, tiene_division, Combustibles)
MERGE (s22:Empresa {nombre: 'FEMSA'});
MERGE (o22:Division {nombre: 'Combustibles'});
MERGE (s22)-[:TIENE_DIVISION {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 21}]->(o22);

// (FEMSA, tiene_division, Proximidad Américas)
MERGE (s23:Empresa {nombre: 'FEMSA'});
MERGE (o23:Division {nombre: 'Proximidad Américas'});
MERGE (s23)-[:TIENE_DIVISION {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 21}]->(o23);

// (FEMSA, tiene_division, Proximidad Europa)
MERGE (s24:Empresa {nombre: 'FEMSA'});
MERGE (o24:Division {nombre: 'Proximidad Europa'});
MERGE (s24)-[:TIENE_DIVISION {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 21}]->(o24);

// (FEMSA, tiene_division, Salud)
MERGE (s25:Empresa {nombre: 'FEMSA'});
MERGE (o25:Division {nombre: 'Salud'});
MERGE (s25)-[:TIENE_DIVISION {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 21}]->(o25);

// (FEMSA, tiene_division, Spin)
MERGE (s26:Empresa {nombre: 'FEMSA'});
MERGE (o26:Division {nombre: 'Spin'});
MERGE (s26)-[:TIENE_DIVISION {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 33}]->(o26);

// (Coca-Cola FEMSA, tiene_formato, Juntos+)
MERGE (s27:Division {nombre: 'Coca-Cola FEMSA'});
MERGE (o27:Formato {nombre: 'Juntos+'});
MERGE (s27)-[:TIENE_FORMATO {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 27}]->(o27);

// (Combustibles, tiene_formato, OXXO GAS)
MERGE (s28:Division {nombre: 'Combustibles'});
MERGE (o28:Formato {nombre: 'OXXO GAS'});
MERGE (s28)-[:TIENE_FORMATO {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 21}]->(o28);

// (Proximidad Américas, tiene_formato, OXXO México)
MERGE (s29:Division {nombre: 'Proximidad Américas'});
MERGE (o29:Formato {nombre: 'OXXO México'});
MERGE (s29)-[:TIENE_FORMATO {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 23}]->(o29);

// (Spin, tiene_formato, Spin Premia)
MERGE (s30:Division {nombre: 'Spin'});
MERGE (o30:Formato {nombre: 'Spin Premia'});
MERGE (s30)-[:TIENE_FORMATO {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 33}]->(o30);

// (Spin, tiene_formato, Spin by OXXO)
MERGE (s31:Division {nombre: 'Spin'});
MERGE (o31:Formato {nombre: 'Spin by OXXO'});
MERGE (s31)-[:TIENE_FORMATO {fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 33}]->(o31);

// --- CRECIO_EN derivadas de REPORTA/MIDE (mismo hecho) ---
// Nota: si dos divisiones comparten indicador y variación
// (p. ej. Proximidad Europa y Coca-Cola FEMSA, ambas +14.2%
// en Ingresos Totales), MERGE las colapsa en UNA sola
// relación: es el mismo hecho a nivel de indicador.

MERGE (s1008:MetricaOperativa {nombre: 'Usuarios Activos'});
MERGE (o1008:Periodo {nombre: '2024'});
MERGE (s1008)-[:CRECIO_EN {variacion_pct: 27.5, fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 33}]->(o1008);
MERGE (s1009:MetricaOperativa {nombre: 'Usuarios Activos'});
MERGE (o1009:Periodo {nombre: '2024'});
MERGE (s1009)-[:CRECIO_EN {variacion_pct: 24.9, fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 33}]->(o1009);
MERGE (s1014:IndicadorFinanciero {nombre: 'Ingresos Totales'});
MERGE (o1014:Periodo {nombre: '2024'});
MERGE (s1014)-[:CRECIO_EN {variacion_pct: 14.2, fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 112}]->(o1014);
MERGE (s1015:IndicadorFinanciero {nombre: 'Ingresos Totales'});
MERGE (o1015:Periodo {nombre: '2024'});
MERGE (s1015)-[:CRECIO_EN {variacion_pct: 11.7, fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 112}]->(o1015);
MERGE (s1016:IndicadorFinanciero {nombre: 'Ingresos Totales'});
MERGE (o1016:Periodo {nombre: '2024'});
MERGE (s1016)-[:CRECIO_EN {variacion_pct: 11.2, fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 112}]->(o1016);
MERGE (s1017:IndicadorFinanciero {nombre: 'Utilidad de Operación'});
MERGE (o1017:Periodo {nombre: '2024'});
MERGE (s1017)-[:CRECIO_EN {variacion_pct: 19.8, fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 108}]->(o1017);
MERGE (s1018:IndicadorFinanciero {nombre: 'Ingresos Totales'});
MERGE (o1018:Periodo {nombre: '2024'});
MERGE (s1018)-[:CRECIO_EN {variacion_pct: 10.3, fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 112}]->(o1018);
MERGE (s1020:IndicadorFinanciero {nombre: 'Ingresos Totales'});
MERGE (o1020:Periodo {nombre: '2024'});
MERGE (s1020)-[:CRECIO_EN {variacion_pct: 5.8, fuente_documento: 'FEMSA 2024 Integrated Annual Report (EN)', fuente_pagina: 112}]->(o1020);