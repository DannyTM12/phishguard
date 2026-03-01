# src/phishguard/config.py
"""
Cargador de configuración centralizado para PhishGuard.

Responsabilidad única: leer configs/model_config.yaml y exponerlo
como un objeto Pydantic validado. Ningún otro módulo debe leer
el YAML directamente; todos deben importar `get_config()`.

Uso:
    from phishguard.config import get_config
    cfg = get_config()
    print(cfg.gating.metadata_threshold)   # 0.70
    print(cfg.fusion.alpha)                 # 0.60
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sección: sub-modelos de configuración
# ---------------------------------------------------------------------------


class GatingConfig(BaseModel):
    """
    Parámetros del mecanismo de gating por metadatos.

    Si score_meta >= metadata_threshold el correo se clasifica
    directamente como phishing sin invocar el submodelo de texto,
    reduciendo el costo computacional.
    """

    metadata_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="θ_meta: umbral sobre score_meta para activar el gating.",
    )

    @field_validator("metadata_threshold")
    @classmethod
    def threshold_is_finite(cls, v: float) -> float:
        import math

        if not math.isfinite(v):
            raise ValueError("metadata_threshold debe ser un número finito.")
        return v


class FusionConfig(BaseModel):
    """
    Parámetros de la etapa de fusión tardía (late fusion).

    score_final = alpha * score_meta + (1 - alpha) * score_text
    La decisión final es: phishing si score_final >= decision_threshold.
    """

    alpha: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Peso del submodelo de metadatos en la fusión ponderada.",
    )
    decision_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="θ_final: umbral sobre score_final para emitir etiqueta phishing.",
    )

    @model_validator(mode="after")
    def alpha_and_threshold_are_finite(self) -> "FusionConfig":
        import math

        for name, val in [("alpha", self.alpha), ("decision_threshold", self.decision_threshold)]:
            if not math.isfinite(val):
                raise ValueError(f"{name} debe ser un número finito.")
        return self


class ExplainabilityConfig(BaseModel):
    """Parámetros para el módulo de explicabilidad SHAP/LIME."""

    enabled: bool = Field(
        default=True,
        description="Activar/desactivar el cómputo de explicaciones en inferencia.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Número de features más importantes a reportar por predicción.",
    )


class PhishGuardConfig(BaseModel):
    """
    Configuración raíz de PhishGuard.

    Agrupa todos los sub-modelos y expone helpers de validación
    cruzada entre secciones.
    """

    gating: GatingConfig = Field(default_factory=GatingConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)

    @model_validator(mode="after")
    def gating_threshold_above_fusion_threshold(self) -> "PhishGuardConfig":
        """
        Restricción de coherencia: el umbral de gating debe ser
        más restrictivo que el umbral de decisión final, ya que el
        gating es un atajo para casos de alta confianza.
        """
        if self.gating.metadata_threshold <= self.fusion.decision_threshold:
            raise ValueError(
                f"gating.metadata_threshold ({self.gating.metadata_threshold}) "
                f"debe ser mayor que fusion.decision_threshold "
                f"({self.fusion.decision_threshold}) para que el gating "
                f"tenga sentido semántico."
            )
        return self


# ---------------------------------------------------------------------------
# Sección: carga del archivo YAML
# ---------------------------------------------------------------------------


def _resolve_config_path(config_path: str | Path | None) -> Path:
    """
    Resuelve la ruta al YAML de configuración.

    Orden de prioridad:
      1. `config_path` explícito (argumento de la función).
      2. Variable de entorno PHISHGUARD_CONFIG.
      3. Ruta por defecto relativa a la raíz del repositorio.
    """
    if config_path is not None:
        return Path(config_path).resolve()

    env_path = os.environ.get("PHISHGUARD_CONFIG")
    if env_path:
        return Path(env_path).resolve()

    # Sube desde este archivo → src/phishguard/ → src/ → raíz del repo
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "configs" / "model_config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Lee el YAML y devuelve un dict sin validar. Aísla el I/O."""
    if not path.exists():
        raise FileNotFoundError(
            f"Archivo de configuración no encontrado: {path}\n"
            f"Crea el archivo o ajusta PHISHGUARD_CONFIG."
        )
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    return raw


def load_config(config_path: str | Path | None = None) -> PhishGuardConfig:
    """
    Carga y valida la configuración desde el YAML.

    Args:
        config_path: Ruta explícita al YAML. Si es None, se usa
                     la resolución automática (env var → default).

    Returns:
        Instancia validada de PhishGuardConfig.

    Raises:
        FileNotFoundError: si el YAML no existe en ninguna ruta resuelta.
        pydantic.ValidationError: si algún valor viola las restricciones.
    """
    path = _resolve_config_path(config_path)
    raw = _load_yaml(path)
    return PhishGuardConfig(**raw)


@lru_cache(maxsize=1)
def get_config() -> PhishGuardConfig:
    """
    Singleton cacheado de la configuración global.

    Llama a `load_config()` la primera vez y devuelve el mismo
    objeto en llamadas subsiguientes dentro del mismo proceso.

    Para forzar la recarga (útil en tests), llama a:
        get_config.cache_clear()
    """
    return load_config()