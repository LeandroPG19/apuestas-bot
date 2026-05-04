"""Agents on-demand para análisis profundo de partidos individuales.

A diferencia del flow batch `deep_analysis_flow` que procesa 200+ events en
modo "best-effort low-latency", los agentes aquí son intensivos por evento:
30-60 s de análisis con todas las señales disponibles fusionadas.

Casos de uso:
    /analizar PSG vs Bayern  →  reporte completo con picks recomendados
    /analizar 116617         →  por match_id directo
"""
