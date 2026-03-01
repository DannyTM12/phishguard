# src/phishguard/features/extractor.py
"""
Extracción de características para los dos submodelos de PhishGuard.

Responsabilidad única: dado texto crudo de un correo, producir
representaciones listas para ser consumidas por los submodelos.

  • extract_metadata_features() → dict[str, float | int | bool]
        Características numéricas rápidas para el submodelo técnico
        (Random Forest / XGBoost) y para el gating.

  • extract_text_features() → str
        Texto preparado para TF-IDF o Transformer (BERT/DistilBERT).

Dependencias
------------
Solo biblioteca estándar + el módulo text_cleaner del mismo paquete.
Sin sklearn, sin transformers — la extracción es independiente del
modelo que consuma los features.

Notas de diseño
---------------
  • FeatureExtractor es una clase (no funciones sueltas) para permitir
    inyección de configuración futura (umbrales, listas negras, etc.)
    sin cambiar las firmas públicas.
  • Todos los métodos son deterministas y sin efectos secundarios.
  • Los nombres de features en el dict son snake_case estables —
    no deben cambiar entre versiones para no romper modelos ya
    entrenados que referencian columnas por nombre.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Final

from phishguard.preprocessing.text_cleaner import (
    clean_subject,
    clean_text,
    count_urgency_words,
    extract_domains,
    extract_urls,
    has_urgency_words,
)

# ---------------------------------------------------------------------------
# Constantes del extractor
# ---------------------------------------------------------------------------

# Extensiones de archivo típicas en adjuntos maliciosos
_MALICIOUS_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".exe", ".bat", ".cmd", ".scr", ".pif", ".com",
        ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh",
        ".ps1", ".psm1", ".psd1",
        ".doc", ".docm", ".xls", ".xlsm", ".ppt", ".pptm",
        ".zip", ".rar", ".7z", ".gz", ".tar",
        ".pdf",   # legítimo pero usado en phishing con frecuencia
        ".html", ".htm",  # phishing pages distribuidas como adjuntos
    }
)

# Acortadores de URLs conocidos (feature de riesgo elevado)
_URL_SHORTENERS: Final[frozenset[str]] = frozenset(
    {
        "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
        "buff.ly", "short.link", "is.gd", "v.gd", "rebrand.ly",
        "cutt.ly", "shorte.st", "adf.ly", "bc.vc", "tiny.cc",
        "lnkd.in", "fb.me", "youtu.be",
    }
)

# TLDs de alto riesgo frecuentes en dominios de phishing
_SUSPICIOUS_TLDS: Final[frozenset[str]] = frozenset(
    {".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".cc", ".xyz",
     ".top", ".club", ".online", ".site", ".fun", ".live",
     ".stream", ".download", ".bid", ".win"}
)

# Palabras de marca comúnmente suplantadas en phishing
_BRAND_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "paypal", "apple", "microsoft", "amazon", "netflix",
        "google", "facebook", "instagram", "twitter", "linkedin",
        "chase", "wells fargo", "citibank", "bankofamerica",
        "fedex", "ups", "dhl", "usps",
        "irs", "gov", "gobierno", "sat", "imss", "bbva",
        "bancomer", "banamex", "santander", "hsbc",
    }
)

# Separador usado al concatenar asunto y cuerpo para el modelo de texto
_TEXT_SEPARATOR: Final[str] = " [SEP] "

# Número máximo de caracteres para el feature de texto (evitar tokens gigantes)
_MAX_TEXT_LENGTH: Final[int] = 10_000

# ---------------------------------------------------------------------------
# Helpers internos (funciones privadas del módulo, no forman parte del API)
# ---------------------------------------------------------------------------


def _shannon_entropy(text: str) -> float:
    """
    Calcula la entropía de Shannon del texto (en bits por carácter).

    Alta entropía en una URL o en el cuerpo indica strings aleatorios
    (hashes, tokens, dominios generados algorítmicamente) — señal de
    phishing.

    Args:
        text: Cadena de caracteres.

    Returns:
        Entropía en bits/carácter. 0.0 si el texto está vacío o es un
        solo carácter único.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
        if c > 0
    )
    return round(entropy, 4)


def _count_digits(text: str) -> int:
    """Cuenta los dígitos en el texto."""
    return sum(1 for c in text if c.isdigit())


def _count_special_chars(text: str) -> int:
    """Cuenta caracteres especiales (no alfanuméricos, no whitespace)."""
    return sum(1 for c in text if not c.isalnum() and not c.isspace())


def _has_ip_in_url(urls: list[str]) -> bool:
    """
    Detecta si alguna URL usa una dirección IP en lugar de un dominio.

    URLs como http://192.168.1.1/login son una señal clara de phishing
    (los sitios legítimos raramente usan IPs directas en enlaces).
    """
    _RE_IP_URL = re.compile(
        r"^https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:[:/]|$)",
        re.IGNORECASE,
    )
    return any(_RE_IP_URL.match(url) for url in urls)


def _has_url_shortener(domains: list[str]) -> bool:
    """Detecta si algún dominio pertenece a un servicio acortador conocido."""
    return any(d.lower() in _URL_SHORTENERS for d in domains)


def _count_suspicious_tlds(domains: list[str]) -> int:
    """Cuenta dominios con TLDs de alto riesgo."""
    count = 0
    for domain in domains:
        for tld in _SUSPICIOUS_TLDS:
            if domain.lower().endswith(tld):
                count += 1
                break
    return count


def _max_url_entropy(urls: list[str]) -> float:
    """
    Devuelve la entropía máxima entre todas las URLs de la lista.

    Una URL con entropía muy alta suele ser un dominio generado
    algorítmicamente (DGA) o contener parámetros obfuscados.
    """
    if not urls:
        return 0.0
    return max(_shannon_entropy(url) for url in urls)


def _avg_url_length(urls: list[str]) -> float:
    """Longitud media de las URLs (0.0 si no hay URLs)."""
    if not urls:
        return 0.0
    return round(sum(len(u) for u in urls) / len(urls), 2)


def _count_brand_mentions(text: str) -> int:
    """
    Cuenta menciones de marcas conocidas suplantadas en phishing.

    Detectar que el asunto o cuerpo menciona "PayPal", "Apple",
    "Microsoft" etc. es una señal de suplantación cuando se combina
    con URLs de dominios no relacionados.
    """
    if not text:
        return 0
    lower = text.lower()
    return sum(1 for brand in _BRAND_KEYWORDS if brand in lower)


def _ratio_uppercase(text: str) -> float:
    """
    Proporción de letras en mayúscula respecto al total de letras.

    Correos de phishing suelen usar MAYÚSCULAS para crear urgencia.
    """
    if not text:
        return 0.0
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    upper = sum(1 for c in alpha if c.isupper())
    return round(upper / len(alpha), 4)


def _count_exclamations(text: str) -> int:
    """Cuenta signos de exclamación (!) — indicador de tono alarmista."""
    return text.count("!")


def _has_html_in_body(body: str) -> bool:
    """Detecta si el cuerpo del correo contiene marcado HTML."""
    return bool(re.search(r"<[a-zA-Z][^>]{0,200}>", body))


def _has_form_in_body(body: str) -> bool:
    """
    Detecta presencia de etiquetas <form> en el cuerpo HTML.

    Formularios embebidos que envían datos fuera del dominio del
    remitente son una señal fuerte de phishing.
    """
    return bool(re.search(r"<form\b", body, re.IGNORECASE))


def _has_iframe_in_body(body: str) -> bool:
    """Detecta iframes ocultos en el cuerpo HTML."""
    return bool(re.search(r"<iframe\b", body, re.IGNORECASE))


def _count_redirect_params(urls: list[str]) -> int:
    """
    Cuenta URLs con parámetros de redirección sospechosos.

    Patrones como ?redirect=, ?url=, ?next=, ?goto= se usan en
    open-redirect attacks para encadenar dominios legítimos con
    páginas de phishing.
    """
    _RE_REDIRECT = re.compile(
        r"[?&](?:redirect|url|next|goto|return|redir|continue|target)"
        r"(?:_url|_to)?=",
        re.IGNORECASE,
    )
    return sum(1 for url in urls if _RE_REDIRECT.search(url))


def _subject_has_re_fwd(subject: str) -> bool:
    """
    Detecta si el asunto simula ser una respuesta o reenvío.

    "Re:", "Fw:", "Fwd:" en asuntos de phishing intentan crear
    falsa familiaridad o contexto de conversación previa.
    """
    return bool(re.match(r"^\s*(re|fw|fwd)\s*:", subject, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Configuración del extractor (extensible sin romper el API)
# ---------------------------------------------------------------------------


@dataclass
class ExtractorConfig:
    """
    Parámetros de configuración del FeatureExtractor.

    Todos los campos tienen defaults razonables. Pasar una instancia
    personalizada al constructor de FeatureExtractor permite ajustar
    el comportamiento sin subclasear.
    """

    normalize_unicode: bool = True
    """Si True, normaliza homoglyphs Unicode a ASCII en clean_text."""

    max_text_length: int = _MAX_TEXT_LENGTH
    """Longitud máxima del string para extract_text_features."""

    text_separator: str = _TEXT_SEPARATOR
    """Separador entre asunto y cuerpo en extract_text_features."""

    include_subject_in_text: bool = True
    """Si True, prepende el asunto al texto de entrada al modelo NLP."""


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """
    Extractor de características para los submodelos de PhishGuard.

    Métodos públicos
    ----------------
    extract_metadata_features(subject, body, urls) → dict
        Características numéricas para el submodelo de gating/metadatos.

    extract_text_features(subject, body) → str
        Texto preparado para TF-IDF o Transformer.

    Uso típico
    ----------
    >>> extractor = FeatureExtractor()
    >>> urls = extractor.get_urls_from_body(body_raw)
    >>> meta = extractor.extract_metadata_features(subject, body_raw, urls)
    >>> text = extractor.extract_text_features(subject, body_raw)
    """

    def __init__(self, config: ExtractorConfig | None = None) -> None:
        """
        Args:
            config: Configuración opcional. Si es None, se usan los defaults.
        """
        self._cfg = config or ExtractorConfig()

    # ── Método auxiliar público ─────────────────────────────────────────────

    def get_urls_from_body(self, body: str) -> list[str]:
        """
        Extrae URLs del cuerpo crudo del correo.

        Expuesto como método público para que el caller pueda reutilizar
        la lista de URLs tanto en extract_metadata_features como en
        otras lógicas sin reextraer dos veces.

        Args:
            body: Cuerpo del correo, con o sin HTML.

        Returns:
            Lista de URLs únicas en orden de aparición.
        """
        return extract_urls(body)

    # ── Submodelo de metadatos ──────────────────────────────────────────────

    def extract_metadata_features(
        self,
        subject: str,
        body: str,
        urls: list[str],
    ) -> dict[str, float | int | bool]:
        """
        Extrae características rápidas (sin modelos pesados) para el
        submodelo de gating y el clasificador de metadatos.

        Todas las features son numéricas o booleanas para ser directamente
        consumibles por scikit-learn (no requieren encoding adicional).

        Los nombres de feature son estables: cambiarlos rompe modelos
        ya entrenados que referencian columnas por nombre.

        Args:
            subject: Asunto del correo (puede ser crudo o ya limpio).
            body:    Cuerpo del correo (puede contener HTML).
            urls:    Lista de URLs ya extraídas (usar get_urls_from_body()).

        Returns:
            Diccionario con 30 features agrupadas en 5 categorías:
              - Longitudes y conteos básicos
              - Señales de URLs
              - Señales de HTML
              - Señales de urgencia / persuasión
              - Señales del asunto
        """
        # Limpiar inputs una sola vez
        subject_clean = clean_subject(subject)
        body_clean = clean_text(
            body, normalize_unicode=self._cfg.normalize_unicode
        )
        domains = extract_domains(urls)
        body_lower = body_clean.lower()

        # ── Grupo 1: Longitudes y conteos básicos ──────────────────────────
        features: dict[str, float | int | bool] = {
            "subject_length": len(subject_clean),
            "body_length": len(body_clean),
            "body_word_count": len(body_clean.split()),
            "body_digit_count": _count_digits(body_clean),
            "body_special_char_count": _count_special_chars(body_clean),
            "body_entropy": _shannon_entropy(body_clean),
            "ratio_uppercase_body": _ratio_uppercase(body_clean),
            "ratio_uppercase_subject": _ratio_uppercase(subject_clean),
            "exclamation_count": _count_exclamations(body_clean)
                                 + _count_exclamations(subject_clean),
        }

        # ── Grupo 2: Señales de URLs ────────────────────────────────────────
        features.update(
            {
                "num_urls": len(urls),
                "num_unique_domains": len(set(domains)),
                "has_url": len(urls) > 0,
                "has_ip_url": _has_ip_in_url(urls),
                "has_url_shortener": _has_url_shortener(domains),
                "num_suspicious_tlds": _count_suspicious_tlds(domains),
                "max_url_entropy": _max_url_entropy(urls),
                "avg_url_length": _avg_url_length(urls),
                "num_redirect_params": _count_redirect_params(urls),
                "max_url_length": max((len(u) for u in urls), default=0),
            }
        )

        # ── Grupo 3: Señales de HTML en el cuerpo ──────────────────────────
        features.update(
            {
                "has_html": _has_html_in_body(body),
                "has_form": _has_form_in_body(body),
                "has_iframe": _has_iframe_in_body(body),
            }
        )

        # ── Grupo 4: Señales de urgencia / persuasión ──────────────────────
        combined_text = subject_clean + " " + body_clean
        features.update(
            {
                "has_urgency_words": has_urgency_words(combined_text),
                "urgency_word_count": count_urgency_words(combined_text),
                "brand_mention_count": _count_brand_mentions(combined_text),
                "has_brand_mention": _count_brand_mentions(combined_text) > 0,
            }
        )

        # ── Grupo 5: Señales del asunto ─────────────────────────────────────
        features.update(
            {
                "subject_is_empty": len(subject_clean) == 0,
                "subject_has_re_fwd": _subject_has_re_fwd(subject_clean),
                "subject_entropy": _shannon_entropy(subject_clean),
                "subject_exclamations": _count_exclamations(subject_clean),
                "subject_has_urgency": has_urgency_words(subject_clean),
                "subject_has_brand": _count_brand_mentions(subject_clean) > 0,
            }
        )

        return features

    # ── Submodelo de texto ──────────────────────────────────────────────────

    def extract_text_features(self, subject: str, body: str) -> str:
        """
        Prepara el texto del correo para el submodelo NLP.

        Aplica la limpieza completa (strip_html, normalize_unicode,
        normalize_whitespace) y concatena asunto + cuerpo con un
        separador reconocible por los tokenizadores de Transformer
        (compatible con el token [SEP] de BERT).

        El resultado es un string listo para:
          • sklearn TfidfVectorizer.transform([text])
          • transformers.AutoTokenizer(text, ...)
          • Cualquier otro modelo que consuma texto como entrada

        Args:
            subject: Asunto del correo (crudo).
            body:    Cuerpo del correo (crudo, puede tener HTML).

        Returns:
            String limpio con formato:
              "<asunto_limpio> [SEP] <cuerpo_limpio>"
            o solo "<cuerpo_limpio>" si include_subject_in_text=False.

            Truncado a max_text_length caracteres para evitar inputs
            desproporcionados en modelos con límite de tokens.
        """
        subject_clean = clean_subject(subject)
        body_clean = clean_text(
            body, normalize_unicode=self._cfg.normalize_unicode
        )

        if self._cfg.include_subject_in_text and subject_clean:
            combined = subject_clean + self._cfg.text_separator + body_clean
        else:
            combined = body_clean

        # Truncar a max_text_length caracteres
        # (para BERT: el tokenizador trunca a 512 tokens internamente,
        #  pero limitar aquí evita procesamiento innecesario)
        if len(combined) > self._cfg.max_text_length:
            combined = combined[: self._cfg.max_text_length]

        return combined

    # ── Representación de depuración ───────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"FeatureExtractor("
            f"normalize_unicode={self._cfg.normalize_unicode}, "
            f"max_text_length={self._cfg.max_text_length}, "
            f"separator={self._cfg.text_separator!r})"
        )


# ---------------------------------------------------------------------------
# Nombres de features de metadatos (contrato estable para el modelo)
# ---------------------------------------------------------------------------

def get_metadata_feature_names() -> list[str]:
    """
    Devuelve la lista ordenada y estable de nombres de features de metadatos.

    Usar esta función (en lugar de list(dict.keys())) garantiza que el
    orden de columnas sea siempre el mismo independientemente de la versión
    de Python, lo cual es crítico para serializar/deserializar modelos
    sklearn que asumen un orden fijo de columnas.

    Returns:
        Lista de 30 nombres de features en el mismo orden en que
        extract_metadata_features() los devuelve.
    """
    return [
        # Grupo 1: básicos
        "subject_length",
        "body_length",
        "body_word_count",
        "body_digit_count",
        "body_special_char_count",
        "body_entropy",
        "ratio_uppercase_body",
        "ratio_uppercase_subject",
        "exclamation_count",
        # Grupo 2: URLs
        "num_urls",
        "num_unique_domains",
        "has_url",
        "has_ip_url",
        "has_url_shortener",
        "num_suspicious_tlds",
        "max_url_entropy",
        "avg_url_length",
        "num_redirect_params",
        "max_url_length",
        # Grupo 3: HTML
        "has_html",
        "has_form",
        "has_iframe",
        # Grupo 4: urgencia
        "has_urgency_words",
        "urgency_word_count",
        "brand_mention_count",
        "has_brand_mention",
        # Grupo 5: asunto
        "subject_is_empty",
        "subject_has_re_fwd",
        "subject_entropy",
        "subject_exclamations",
        "subject_has_urgency",
        "subject_has_brand",
    ]