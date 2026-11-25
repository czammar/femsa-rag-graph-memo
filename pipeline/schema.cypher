// schema.cypher — restricciones e índices para la ontología v1 (FEMSA 2024)
//
// OpenCypher estándar: corre en Neo4j 5+ y en Relate (relatedb.ai).
// Una restricción de unicidad por etiqueta sobre `nombre`: es lo que hace
// que los MERGE de load_graph.cypher sean idempotentes sin duplicar nodos.
//
// Nota sobre Relate: si tu versión de Relate aún no soporta CREATE
// CONSTRAINT, load_relate.py lo detecta, avisa y sigue (los MERGE
// siguen funcionando; solo pierdes la garantía de unicidad a nivel de
// base). En Neo4j las restricciones se crean sin problema.

CREATE CONSTRAINT empresa_nombre_unique IF NOT EXISTS
FOR (n:Empresa) REQUIRE n.nombre IS UNIQUE;

CREATE CONSTRAINT division_nombre_unique IF NOT EXISTS
FOR (n:Division) REQUIRE n.nombre IS UNIQUE;

CREATE CONSTRAINT formato_nombre_unique IF NOT EXISTS
FOR (n:Formato) REQUIRE n.nombre IS UNIQUE;

CREATE CONSTRAINT indicador_nombre_unique IF NOT EXISTS
FOR (n:IndicadorFinanciero) REQUIRE n.nombre IS UNIQUE;

CREATE CONSTRAINT metrica_nombre_unique IF NOT EXISTS
FOR (n:MetricaOperativa) REQUIRE n.nombre IS UNIQUE;

CREATE CONSTRAINT periodo_nombre_unique IF NOT EXISTS
FOR (n:Periodo) REQUIRE n.nombre IS UNIQUE;

CREATE CONSTRAINT pais_nombre_unique IF NOT EXISTS
FOR (n:Pais) REQUIRE n.nombre IS UNIQUE;

CREATE CONSTRAINT iniciativa_nombre_unique IF NOT EXISTS
FOR (n:Iniciativa) REQUIRE n.nombre IS UNIQUE;

CREATE CONSTRAINT riesgo_nombre_unique IF NOT EXISTS
FOR (n:Riesgo) REQUIRE n.nombre IS UNIQUE;
