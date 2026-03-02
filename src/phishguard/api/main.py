# src/phishguard/api/main.py
"""
API REST de PhishGuard — Microservicio de detección de phishing.

Endpoints
---------
  GET  /health      → Estado del servicio y metadatos del engine.
  POST /classify    → Clasifica un correo como phishing o legítimo.

Ejecución local
---------------
  uvicorn phishguard.api.main:app --reload --port 8000

Ejecución en Docker / producción
----------------------------------
  uvicorn phishguard.api.main:app --host 0.0.0.0 --port 8000 --workers 4

Decisiones de diseño
--------------------
  • El engine se inicializa UNA sola vez en el evento `lifespan` de la
    app (no en cada request) usando PhishGuardEngine.get_instance().
  • Los modelos se inyectan en `lifespan`: en el Hito 2, reemplazar los
    dummies por modelos reales es cambiar 2 líneas en ese bloque.
  • Todos los schemas de request/response son Pydantic v2 para
    validación automática y generación de OpenAPI.
  • Los errores de negocio (engine no iniciado, input inválido) usan
    HTTPException con códigos semánticos.
"""

from __future__ import annotations

import logging
import platform
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from phishguard.models.fusion_engine import (
    PhishGuardEngine,
    PredictionResult,
    SubModel,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemas de request / response (contratos públicos de la API)
# ---------------------------------------------------------------------------


class ClassifyRequest(BaseModel):
    """
    Cuerpo del request para POST /classify.

    Todos los campos de texto se aceptan vacíos (el engine los maneja),
    pero `body` es obligatorio ya que es la fuente principal de evidencia.
    """

    subject: str = Field(
        default="",
        description="Asunto del correo electrónico.",
        examples=["URGENT: Verify your PayPal account immediately"],
        max_length=1_000,
    )
    body: str = Field(
        description="Cuerpo del correo (puede contener HTML).",
        examples=["Dear Customer, click here to verify: http://evil.tk/login"],
        max_length=100_000,
    )
    sender: str = Field(
        default="",
        description=(
            "Dirección del remitente (e.g., support@paypal.com). "
            "Reservado para features de cabecera en el Hito 2."
        ),
        examples=["noreply@paypa1-security.tk"],
        max_length=500,
    )

    @field_validator("body")
    @classmethod
    def body_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("El campo 'body' no puede estar vacío.")
        return v

    model_config = {"str_strip_whitespace": False}  # el engine limpia por sí mismo


class GatingDetail(BaseModel):
    """Detalles de la etapa de gating incluidos en la respuesta."""

    activated: bool = Field(description="True si el gating cortocircuitó la decisión.")
    score_meta: float = Field(description="Score del submodelo de metadatos ∈ [0, 1].")
    threshold: float = Field(description="Umbral θ_meta configurado.")


class ClassifyResponse(BaseModel):
    """
    Respuesta del endpoint POST /classify.

    Devuelve la decisión final, todos los scores intermedios y metadatos
    de trazabilidad para facilitar auditorías y depuración.
    """

    label: str = Field(
        description="Etiqueta de clasificación: 'phishing' | 'legitimate'.",
        examples=["phishing"],
    )
    is_phishing: bool = Field(
        description="Booleano equivalente a label == 'phishing'.",
    )
    score_final: float = Field(
        description=(
            "Score de decisión final ∈ [0, 1]. "
            "Igual a score_meta si el gating se activó; "
            "de lo contrario, α·score_meta + (1-α)·score_text."
        ),
    )
    score_meta: float = Field(
        description="Score del submodelo de metadatos ∈ [0, 1].",
    )
    score_text: float | None = Field(
        default=None,
        description=(
            "Score del submodelo de texto ∈ [0, 1], "
            "o null si el gating se activó (texto no evaluado)."
        ),
    )
    gating: GatingDetail = Field(
        description="Detalle de la etapa de gating por metadatos.",
    )
    metadata_features: dict[str, Any] = Field(
        description="Las 32 características técnicas extraídas del correo.",
    )
    latency_ms: float = Field(
        description="Latencia de inferencia en milisegundos.",
    )
    meta_model_name: str = Field(
        description="Nombre/versión del submodelo de metadatos usado.",
    )
    text_model_name: str | None = Field(
        default=None,
        description="Nombre/versión del submodelo de texto usado, o null.",
    )
    warning: str | None = Field(
        default=None,
        description=(
            "Advertencia opcional si el engine usa modelos dummy/heurísticos. "
            "Ausente en producción con modelos entrenados."
        ),
    )
    # Hito 3: explicación SHAP top-K de las features de metadatos.
    # Presente solo si config.explainability.enabled = true y SHAP está instalado.
    # Ejemplo: {"has_ip_url": 0.312, "has_url_shortener": 0.201, ...}
    explanation: dict[str, float] | None = Field(
        default=None,
        description=(
            "Top features que explican la predicción según SHAP (SHapley Additive "
            "exPlanations) sobre el submodelo de metadatos. Cada clave es el nombre "
            "de la feature y el valor es su contribución SHAP a la clase phishing. "
            "Null si la explicabilidad está desactivada o SHAP no está instalado."
        ),
    )


class HealthResponse(BaseModel):
    """Respuesta del endpoint GET /health."""

    status: str = Field(description="'ok' si el servicio está operativo.")
    version: str = Field(description="Versión del paquete PhishGuard.")
    engine_ready: bool = Field(description="True si el engine está inicializado.")
    using_dummy_models: bool = Field(
        description="True si los submodelos son heurísticos (no entrenados)."
    )
    meta_model: str = Field(description="Nombre del submodelo de metadatos activo.")
    text_model: str = Field(description="Nombre del submodelo de texto activo.")
    python_version: str = Field(description="Versión de Python del intérprete.")
    uptime_seconds: float = Field(description="Segundos desde que arrancó el proceso.")


class ErrorResponse(BaseModel):
    """Schema estándar para respuestas de error."""

    detail: str
    error_code: str | None = None


# ---------------------------------------------------------------------------
# Estado de aplicación
# ---------------------------------------------------------------------------

# Almacena el engine y el timestamp de inicio entre requests.
# El uso de un dict simple (no una clase) mantiene el estado legible
# y es suficiente para este microservicio single-process.
_app_state: dict[str, Any] = {
    "engine": None,
    "startup_time": None,
}

# Versión del paquete — en producción esto vendría de importlib.metadata
_PHISHGUARD_VERSION = "0.1.0-hito1"


# ---------------------------------------------------------------------------
# Lifespan: inicialización y cierre del engine
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Gestiona el ciclo de vida del engine de PhishGuard.

    El bloque ANTES del `yield` se ejecuta al arrancar el servidor.
    El bloque DESPUÉS del `yield` se ejecuta al apagarlo.

    ─────────────────────────────────────────────────────────────────
    HITO 2: Para inyectar modelos entrenados, reemplaza las líneas
    marcadas con [HITO 2] por la carga real de los pickles:

        from phishguard.models.meta_submodel import MetaSubModel
        from phishguard.models.text_submodel import TextSubModel

        meta = MetaSubModel.load("models/meta.pkl")      # [HITO 2]
        text = TextSubModel.load("models/text.pkl")      # [HITO 2]
    ─────────────────────────────────────────────────────────────────
    """
    log.info("Iniciando PhishGuard API...")

    # ── [HITO 2] Modelos reales ───────────────────────────────────────────
    from phishguard.models.meta_submodel import MetaSubModel
    from phishguard.models.text_submodel import TextSubModel
    from pathlib import Path

    try:
        meta_model: SubModel | None = MetaSubModel.load(Path("artifacts/meta_model.pkl"))
    except FileNotFoundError:
        log.warning("Modelo de metadatos no encontrado. Usando Dummy.")
        meta_model = None

    try:
        text_model: SubModel | None = TextSubModel.load(Path("artifacts/text_model.pkl"))
    except FileNotFoundError:
        log.warning("Modelo de texto no encontrado. Usando Dummy.")
        text_model = None
    # ──────────────────────────────────────────────────────────────────────

    try:
        engine = PhishGuardEngine.get_instance(
            meta_model=meta_model,
            text_model=text_model,
        )
        _app_state["engine"] = engine
        _app_state["startup_time"] = time.time()
        log.info("Engine inicializado: %s", engine)
    except Exception as exc:
        log.error("Fallo al inicializar el engine: %s", exc, exc_info=True)
        # El servidor arranca de todas formas; el endpoint /health
        # reportará engine_ready=False.

    yield  # ← La app sirve requests entre yield y el cierre

    # Limpieza al apagar
    log.info("Apagando PhishGuard API. Liberando resources.")
    PhishGuardEngine.reset_instance()
    _app_state["engine"] = None


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------


app = FastAPI(
    title="PhishGuard API",
    description=(
        "Microservicio de detección de phishing basado en fusión tardía "
        "de señales semánticas (NLP) y técnicas (metadatos/URLs).\n\n"
        "**Hito 1**: Los submodelos son heurísticos (dummy). "
        "El flujo de gating y fusión está completamente implementado."
    ),
    version=_PHISHGUARD_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    responses={
        422: {"model": ErrorResponse, "description": "Error de validación"},
        500: {"model": ErrorResponse, "description": "Error interno del servidor"},
    },
)


# ---------------------------------------------------------------------------
# Dependencias inyectables
# ---------------------------------------------------------------------------


def get_engine() -> PhishGuardEngine:
    """
    Dependencia FastAPI que provee el engine al handler del endpoint.

    Lanza HTTP 503 si el engine no está disponible (fallo en startup).
    """
    engine: PhishGuardEngine | None = _app_state.get("engine")
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El motor de clasificación no está disponible. "
                "Revisa los logs de inicio del servidor."
            ),
        )
    return engine


# Tipo anotado para usar en las firmas de los handlers
EngineDepends = Annotated[PhishGuardEngine, Depends(get_engine)]


# ---------------------------------------------------------------------------
# Manejadores de errores globales
# ---------------------------------------------------------------------------


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Convierte ValueError de la lógica de negocio en HTTP 400."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_code": "INVALID_INPUT"},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Captura cualquier excepción no manejada y devuelve HTTP 500."""
    log.error("Error no manejado en %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Error interno del servidor. Revisa los logs.",
            "error_code": "INTERNAL_ERROR",
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio",
    description=(
        "Verifica que el servicio está operativo y devuelve información "
        "sobre el estado del engine y los modelos activos."
    ),
    tags=["infraestructura"],
)
async def health() -> HealthResponse:
    """
    Endpoint de healthcheck.

    Devuelve HTTP 200 en todos los casos (incluso si el engine no está
    listo), para permitir que los orquestadores distingan entre
    "el proceso está vivo" (200) y "el engine está listo" (engine_ready=True).
    """
    engine: PhishGuardEngine | None = _app_state.get("engine")
    startup_time: float | None = _app_state.get("startup_time")
    uptime = (time.time() - startup_time) if startup_time else 0.0

    if engine is not None:
        return HealthResponse(
            status="ok",
            version=_PHISHGUARD_VERSION,
            engine_ready=True,
            using_dummy_models=engine.is_using_dummy_models,
            meta_model=engine._meta_model.model_name,
            text_model=engine._text_model.model_name,
            python_version=platform.python_version(),
            uptime_seconds=round(uptime, 2),
        )

    return HealthResponse(
        status="degraded",
        version=_PHISHGUARD_VERSION,
        engine_ready=False,
        using_dummy_models=True,
        meta_model="unavailable",
        text_model="unavailable",
        python_version=platform.python_version(),
        uptime_seconds=round(uptime, 2),
    )


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Clasificar un correo electrónico",
    description=(
        "Recibe el asunto, cuerpo y remitente de un correo y devuelve "
        "la clasificación (phishing / legitimate) junto con scores "
        "intermedios y detalles del gating.\n\n"
        "El campo `warning` estará presente si se están usando modelos "
        "heurísticos (Hito 1) en lugar de modelos entrenados reales."
    ),
    tags=["clasificación"],
    responses={
        200: {"description": "Clasificación exitosa"},
        400: {"model": ErrorResponse, "description": "Input inválido"},
        503: {"model": ErrorResponse, "description": "Engine no disponible"},
    },
)
async def classify(
    request_body: ClassifyRequest,
    engine: EngineDepends,
) -> ClassifyResponse:
    """
    Clasifica un correo como phishing o legítimo.

    Invoca el pipeline completo del engine:
      features → score_meta → gating → (score_text) → fusión → decisión.

    La respuesta incluye todos los scores intermedios para facilitar
    la depuración, auditoría y generación futura de explicaciones SHAP.
    """
    log.info(
        "POST /classify — subject=%r | body_len=%d | sender=%r",
        request_body.subject[:60] if request_body.subject else "",
        len(request_body.body),
        request_body.sender[:60] if request_body.sender else "",
    )

    result: PredictionResult = engine.predict(
        subject=request_body.subject,
        body=request_body.body,
        sender=request_body.sender,
    )

    # Advertencia si los modelos son dummy (solo en Hito 1)
    warning: str | None = None
    if engine.is_using_dummy_models:
        warning = (
            "Los submodelos activos son heurísticos (Hito 1). "
            "Los scores no reflejan un modelo entrenado. "
            "Sustituye meta_model y text_model en lifespan() para producción."
        )

    return ClassifyResponse(
        label=result.label,
        is_phishing=result.is_phishing,
        score_final=result.score_final,
        score_meta=result.score_meta,
        score_text=result.score_text,
        gating=GatingDetail(
            activated=result.gating.activated,
            score_meta=result.gating.score_meta,
            threshold=result.gating.threshold,
        ),
        metadata_features=result.metadata_features,
        latency_ms=result.latency_ms,
        meta_model_name=result.meta_model_name,
        text_model_name=result.text_model_name,
        warning=warning,
        explanation=result.explanation,
    )