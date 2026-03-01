# Política de datos (PhishGuard)

## Datasets en el repositorio
Los archivos en `data/raw/*.csv` se versionan mediante **Git LFS** (Large File Storage) para evitar el límite de tamaño de GitHub.

### Requisitos para clonar correctamente
1) Instalar Git LFS
2) Ejecutar:
   - `git lfs install`
   - `git lfs pull`

## Qué NO se versiona
- `data/processed/` (salidas intermedias/limpias/featurizadas)
- artefactos temporales, caches, modelos intermedios sin release

## Notas
Dependiendo de licencias/uso académico, se recomienda mantener el repositorio privado o documentar el origen/licenciamiento de los datasets.
