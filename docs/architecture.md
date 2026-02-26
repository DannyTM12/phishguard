# Arquitectura — PhishGuard Híbrido (Late Fusion)

Flujo:
1) Parseo email (MIME/.eml) → subject/body/headers/urls
2) Modelo de metadatos → score_meta
3) Gating (umbral θ_meta)
4) Texto (solo si aplica) → score_text
5) Fusión tardía: score_final = α*score_meta + (1-α)*score_text
6) Decisión + Reporte
