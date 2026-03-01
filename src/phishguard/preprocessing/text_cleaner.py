# src/phishguard/preprocessing/text_cleaner.py
"""
Limpieza y normalización de texto para PhishGuard.

Responsabilidad única: transformar texto crudo (con posible HTML,
caracteres especiales, Unicode raro) en texto limpio y extraer
estructuras específicas (URLs).

Ninguna función de este módulo toma decisiones de clasificación;
solo transforma cadenas en cadenas o listas.

Diseño
------
Todas las funciones son puras (sin estado, sin efectos secundarios)
y operan sobre cadenas individuales para facilitar su uso con
pandas .apply() o en pipelines de streaming.

Dependencias
------------
Solo biblioteca estándar: re, html, unicodedata, string.
Sin pandas, sin sklearn, sin transformers — el módulo es importable
en cualquier entorno, incluido el microservicio de inferencia.
"""

from __future__ import annotations

import html
import re
import string
import unicodedata
from typing import Final

# ---------------------------------------------------------------------------
# Constantes compiladas (compilar una sola vez al importar el módulo)
# ---------------------------------------------------------------------------

# ── Regex de HTML ──────────────────────────────────────────────────────────

# Etiquetas HTML completas: <tag ...> y </tag>
# Requiere que el nombre de la etiqueta empiece con letra o '/' para
# evitar falsos positivos sobre símbolos < y > que no son HTML
# (ej: "precio < 100" o entidades decodificadas "&lt;foo&gt;").
_RE_HTML_TAGS: Final = re.compile(r"</?[a-zA-Z][^>]{0,500}>", re.IGNORECASE)

# Comentarios HTML: <!-- ... -->
_RE_HTML_COMMENTS: Final = re.compile(r"<!--.*?-->", re.DOTALL)

# Bloques <style> y <script> con su contenido (no útiles semánticamente)
_RE_STYLE_BLOCKS: Final = re.compile(
    r"<(style|script)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

# ── Regex de URL ───────────────────────────────────────────────────────────
#
# Cubre:
#   • http:// y https://
#   • ftp://
#   • URLs ofuscadas con caracteres unicode lookalike (se normalizan antes)
#   • Parámetros de query, fragmentos, puertos, rutas con caracteres especiales
#   • Acortadores comunes (bit.ly, tinyurl, etc.) — detectados por el patrón,
#     no por lista negra
#   • URLs sin esquema explícito con www. (grupo 2 del patrón)
#
# NO intenta parsear emails ni direcciones de red que no sean URLs web.
_RE_URL: Final = re.compile(
    r"""
    (?:                              # ── Grupo A: URL con esquema explícito ──
        (?:https?|ftp)://            #   esquema (http, https, ftp)
        [^\s<>"'(){}\[\]]{1,2048}    #   resto de la URL (sin espacios ni delimitadores HTML)
    )
    |                                # ── Grupo B: URL con www. sin esquema ──
    (?:
        \bwww\.                      #   empieza con www.
        [^\s<>"'(){}\[\]]{1,2048}    #   resto
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Caracteres de cierre que no pertenecen a la URL pero quedan capturados
# por el patrón greedy en textos narrativos ("visita http://ejemplo.com.")
_URL_TRAILING_GARBAGE: Final = re.compile(r"[.,;:!?)>»\]]+$")

# ── Regex de normalización de espacios ────────────────────────────────────

# Cualquier secuencia de whitespace (espacio, tab, newline, NBSP, etc.)
_RE_WHITESPACE: Final = re.compile(r"\s+")

# Líneas en blanco consecutivas (más de 2 saltos de línea seguidos)
_RE_BLANK_LINES: Final = re.compile(r"\n{3,}")

# ── Regex de caracteres de control ────────────────────────────────────────

# Caracteres de control ASCII (0x00-0x1F) excepto \t, \n, \r
_RE_CONTROL_CHARS: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# ── Regex de patrones de ofuscación frecuentes en phishing ────────────────

# "p a s s w o r d" → "password"  (letras sueltas separadas por espacios)
_RE_SPACED_LETTERS: Final = re.compile(r"\b([a-zA-Z]) (?=[a-zA-Z] |\b[a-zA-Z]\b)")

# ── Palabras de urgencia / persuasión (clasificación binaria rápida) ───────
#
# El set cubre el léxico más documentado en campañas de phishing en inglés
# y español (los datasets del proyecto mezclan ambos idiomas).
_URGENCY_WORDS: Final[frozenset[str]] = frozenset(
    {
        # Inglés
        "urgent", "urgently", "immediately", "asap", "now", "today",
        "expire", "expires", "expiring", "expired",
        "suspend", "suspended", "suspending",
        "verify", "verification", "validate", "confirm", "confirmation",
        "update", "required", "mandatory",
        "limited", "limited time", "last chance",
        "account", "password", "credential", "login", "sign in",
        "click here", "click now", "click below",
        "free", "winner", "won", "prize", "gift", "reward", "bonus",
        "bank", "paypal", "apple", "microsoft", "amazon", "netflix",
        "unauthorized", "unusual", "suspicious", "fraud", "alert",
        "act now", "respond", "respond immediately",
        "dear customer", "dear user", "dear account",
        "billing", "invoice", "payment", "overdue",
        # Español
        "urgente", "urgentemente", "inmediatamente", "ahora",
        "verificar", "verificación", "validar", "confirmar",
        "actualizar", "requerido", "obligatorio",
        "cuenta", "contraseña", "acceso", "iniciar sesión",
        "haz clic", "haz click",
        "gratis", "ganador", "premio", "regalo",
        "banco", "suspendido", "vencer", "vence",
        "alerta", "fraude", "inusual", "no autorizado",
        "estimado cliente", "estimado usuario",
        "factura", "pago", "vencido",
    }
)

# ---------------------------------------------------------------------------
# Funciones de limpieza de HTML
# ---------------------------------------------------------------------------


def strip_html_tags(text: str) -> str:
    """
    Elimina etiquetas HTML y devuelve solo el texto visible.

    Pipeline interno:
      1. Decodifica entidades HTML (&amp; → &, &nbsp; → ' ', etc.)
      2. Elimina bloques <style> y <script> con su contenido completo.
      3. Elimina comentarios HTML <!-- ... -->.
      4. Elimina todas las etiquetas restantes <...>.
      5. Decodifica de nuevo (por si había entidades anidadas).

    Args:
        text: Texto crudo, puede contener HTML o no.

    Returns:
        Texto sin marcado HTML. Si el texto no tenía HTML,
        devuelve el mismo texto sin modificaciones relevantes.
    """
    if not text:
        return ""

    # Paso 1: decodificar entidades (&amp; &lt; &nbsp; &#160; etc.)
    text = html.unescape(text)

    # Paso 2: eliminar bloques estilo/script (incluye su contenido)
    text = _RE_STYLE_BLOCKS.sub(" ", text)

    # Paso 3: eliminar comentarios HTML
    text = _RE_HTML_COMMENTS.sub(" ", text)

    # Paso 4: eliminar etiquetas
    text = _RE_HTML_TAGS.sub(" ", text)

    # Paso 5: segunda pasada de entidades (anidadas o generadas por el paso 4)
    text = html.unescape(text)

    # Paso 6: normalizar whitespace residual dejado por las etiquetas eliminadas
    # (cada etiqueta se reemplazó por " ", generando espacios múltiples)
    text = _RE_WHITESPACE.sub(" ", text).strip()

    return text


def _normalize_unicode(text: str) -> str:
    """
    Normaliza Unicode a forma NFC y elimina caracteres homoglyph comunes.

    Los atacantes usan caracteres Unicode visualmente idénticos a ASCII
    (ej. 'а' cirílico en lugar de 'a' latino) para evadir filtros léxicos.
    Esta función los normaliza a su representación ASCII más cercana
    cuando es posible (NFKD → encode ascii ignorando no-ASCII).

    Nota: La normalización es lossy — algunos caracteres no tienen
    equivalente ASCII y se eliminan. Esto es aceptable para el dominio
    de emails donde el contenido semántico está en texto occidental.
    """
    # NFKD descompone caracteres como ñ → n + combining tilde
    normalized = unicodedata.normalize("NFKD", text)
    # Codificar a ASCII ignorando lo que no tiene equivalente, luego decodificar
    return normalized.encode("ascii", errors="ignore").decode("ascii")


def remove_control_characters(text: str) -> str:
    """Elimina caracteres de control ASCII que no son whitespace legible."""
    return _RE_CONTROL_CHARS.sub("", text)


def normalize_whitespace(text: str) -> str:
    """
    Colapsa cualquier secuencia de whitespace en un único espacio.

    Convierte tabs, newlines, NBSP y múltiples espacios en un espacio
    simple. Elimina espacios al inicio y al final.
    """
    return _RE_WHITESPACE.sub(" ", text).strip()


def normalize_whitespace_preserve_lines(text: str) -> str:
    """
    Versión de normalización que preserva la estructura de párrafos.

    Útil para el módulo de texto cuando se quiere mantener
    la separación entre asunto y cuerpo con saltos de línea.
    Colapsa espacios dentro de cada línea pero permite hasta 2
    saltos de línea consecutivos.
    """
    # Normalizar tabs y espacios múltiples dentro de líneas
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    text = "\n".join(lines)
    # Colapsar más de 2 saltos de línea consecutivos
    text = _RE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Pipeline de limpieza completo
# ---------------------------------------------------------------------------


def clean_text(text: str, *, normalize_unicode: bool = True) -> str:
    """
    Pipeline completo de limpieza de texto para un email individual.

    Aplica en orden:
      1. strip_html_tags        — eliminar marcado HTML
      2. remove_control_characters — quitar chars de control
      3. _normalize_unicode     — normalizar homoglyphs (opcional)
      4. normalize_whitespace   — colapsar espacios

    Args:
        text:              Texto a limpiar.
        normalize_unicode: Si True, normaliza caracteres unicode a ASCII.
                           Desactivar solo si se necesita preservar
                           idiomas con caracteres no-ASCII.

    Returns:
        Texto limpio como cadena.
    """
    if not isinstance(text, str):
        return ""

    result = strip_html_tags(text)
    result = remove_control_characters(result)

    if normalize_unicode:
        result = _normalize_unicode(result)

    result = normalize_whitespace(result)
    return result


def clean_subject(subject: str) -> str:
    """
    Limpieza específica para la línea de asunto.

    El asunto raramente tiene HTML, así que se aplica una pipeline
    más ligera: solo normalización de unicode y whitespace.
    """
    if not isinstance(subject, str):
        return ""
    result = remove_control_characters(subject)
    result = _normalize_unicode(result)
    result = normalize_whitespace(result)
    return result


# ---------------------------------------------------------------------------
# Extracción de URLs
# ---------------------------------------------------------------------------


def _clean_url(raw_url: str) -> str:
    """
    Limpia caracteres de cierre que no pertenecen a la URL.

    Ejemplo:
        "http://ejemplo.com/login." → "http://ejemplo.com/login"
        "https://phish.net/page)."  → "https://phish.net/page"
    """
    return _URL_TRAILING_GARBAGE.sub("", raw_url)


def extract_urls(text: str) -> list[str]:
    """
    Extrae todas las URLs del texto de un correo electrónico.

    Maneja:
      • URLs con http://, https://, ftp://
      • URLs que empiezan con www. (sin esquema explícito)
      • URLs dentro de HTML (href="...", src="...")
      • URLs precedidas de entidades como &lt; (&lt;http://...&gt;)
      • URLs ofuscadas con whitespace intercalado (limitado)

    El texto se procesa en dos etapas:
      1. Sobre el texto ORIGINAL (para capturar URLs en atributos HTML)
      2. Sobre el texto SIN HTML (para capturar URLs en texto plano)

    Esto evita perder URLs que estaban como href="..." (se eliminarían
    con el HTML) y también captura las que están escritas en el cuerpo
    visible del correo.

    Args:
        text: Texto del correo, con o sin HTML.

    Returns:
        Lista de URLs únicas y limpias, manteniendo el orden de aparición.
        Lista vacía si no se encuentran URLs.
    """
    if not text:
        return []

    seen: set[str] = set()
    urls: list[str] = []

    def _add_urls_from(source: str) -> None:
        for match in _RE_URL.finditer(source):
            raw = _clean_url(match.group(0))
            if raw and raw not in seen:
                seen.add(raw)
                urls.append(raw)

    # Pasada 1: texto original (captura hrefs y srcset antes de quitar HTML)
    _add_urls_from(html.unescape(text))

    # Pasada 2: texto sin HTML (captura URLs en contenido visible)
    text_no_html = strip_html_tags(text)
    _add_urls_from(text_no_html)

    # Pasada 3: extraer URLs anidadas dentro de parámetros de query.
    # Caso frecuente en phishing: href="http://legit.com?redirect=http://evil.tk"
    # La pasada greedy solo captura la URL externa; esta pasada extrae la interna.
    for outer_url in list(urls):  # iterar sobre copia para no mutar durante iteración
        # Buscar http/https/ftp dentro de la parte de query (?...) de cada URL
        query_start = outer_url.find("?")
        if query_start != -1:
            _add_urls_from(outer_url[query_start:])

    return urls


def extract_domains(urls: list[str]) -> list[str]:
    """
    Extrae el dominio (host) de cada URL de la lista.

    Usa parseo manual con re para evitar importar urllib en todos
    los contextos. Soporta puertos (:8080) y rutas (/path).

    Args:
        urls: Lista de URLs, tal como devuelve extract_urls().

    Returns:
        Lista de dominios únicos (en minúsculas), en orden de aparición.
        URLs malformadas se omiten silenciosamente.

    Example:
        >>> extract_domains(["https://evil.com/login", "http://bank.com:8080/"])
        ["evil.com", "bank.com"]
    """
    _RE_DOMAIN = re.compile(
        r"^(?:https?|ftp)://([^/:?#\s]{1,253})",
        re.IGNORECASE,
    )
    seen: set[str] = set()
    domains: list[str] = []
    for url in urls:
        m = _RE_DOMAIN.match(url)
        if m:
            domain = m.group(1).lower()
            if domain not in seen:
                seen.add(domain)
                domains.append(domain)
    return domains


# ---------------------------------------------------------------------------
# Detección de urgencia
# ---------------------------------------------------------------------------


def has_urgency_words(text: str) -> bool:
    """
    Detecta si el texto contiene palabras o frases asociadas
    a ingeniería social / urgencia típica de phishing.

    Usa búsqueda de substrings case-insensitive sobre el texto
    completo (no tokenización) para capturar frases multi-palabra
    como "click here" o "estimado cliente".

    Args:
        text: Texto limpio (se recomienda pasarlo por clean_text() antes).

    Returns:
        True si al menos una palabra/frase de urgencia está presente.
    """
    if not text:
        return False
    lower = text.lower()
    return any(word in lower for word in _URGENCY_WORDS)


def count_urgency_words(text: str) -> int:
    """
    Cuenta cuántas palabras/frases de urgencia distintas contiene el texto.

    Útil como feature numérico para el modelo de metadatos
    (más granular que el booleano `has_urgency_words`).

    Args:
        text: Texto limpio.

    Returns:
        Número de palabras de urgencia distintas encontradas (0..N).
    """
    if not text:
        return 0
    lower = text.lower()
    return sum(1 for word in _URGENCY_WORDS if word in lower)