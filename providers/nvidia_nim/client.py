"""
NVIDIA NIM provider implementation with built-in rate limiter (39 requests/min).
"""

import json
import threading
import time
from typing import Any

import openai
from loguru import logger

from config.nim import NimSettings
from providers.base import ProviderConfig
from providers.defaults import NVIDIA_NIM_DEFAULT_BASE
from providers.transports.openai_chat import OpenAIChatTransport

from .request_options import build_nim_request_body
from .retry import (
    clone_body_without_chat_template,
    clone_body_without_reasoning_budget,
    clone_body_without_reasoning_content,
)
from .tool_schema import (
    body_without_nim_tool_argument_aliases,
    nim_tool_argument_aliases_from_body,
)


class RateLimiter:
    """
    Ventana fija simple: máximo `max_requests` por cada minuto (según reloj).
    Si se alcanza el límite, bloquea hasta que comience el siguiente minuto.
    """

    def __init__(self, max_requests: int = 39):
        self.max_requests = max_requests
        self._lock = threading.Lock()
        self._window_start = 0.0
        self._count = 0

    def acquire(self):
        """Adquiere un cupo; se bloquea si se superó el límite en el minuto actual."""
        with self._lock:
            now = time.time()
            current_minute = int(now // 60)

            # Reiniciar contador si cambió de minuto
            if current_minute != int(self._window_start // 60):
                self._window_start = now
                self._count = 0

            # Si ya llegamos al límite, esperar hasta el siguiente minuto
            if self._count >= self.max_requests:
                seconds_to_next_minute = 60 - (now % 60)
                logger.warning(
                    f"RateLimiter: límite de {self.max_requests} peticiones/minuto alcanzado. "
                    f"Esperando {seconds_to_next_minute:.1f}s hasta la nueva ventana."
                )
                # Fuera del lock mientras dormimos para no bloquear otros hilos
                time.sleep(seconds_to_next_minute + 0.1)  # pequeño margen de seguridad
                # Volver a intentar (recursivo) con la nueva ventana
                return self.acquire()

            # Cupo disponible
            self._count += 1
            logger.debug(
                f"RateLimiter: petición aceptada ({self._count}/{self.max_requests} en esta ventana)"
            )


class NvidiaNimProvider(OpenAIChatTransport):
    """NVIDIA NIM provider using official OpenAI client, with rate limiting."""

    def __init__(self, config: ProviderConfig, *, nim_settings: NimSettings):
        super().__init__(
            config,
            provider_name="NIM",
            base_url=config.base_url or NVIDIA_NIM_DEFAULT_BASE,
            api_key=config.api_key,
        )
        self._nim_settings = nim_settings
        # Instanciamos el limitador (39 peticiones por minuto)
        self._rate_limiter = RateLimiter(max_requests=39)

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Internal helper for tests and shared building."""
        return build_nim_request_body(
            request,
            self._nim_settings,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _prepare_create_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Strip private request metadata before calling NVIDIA NIM."""
        return body_without_nim_tool_argument_aliases(body)

    def _tool_argument_aliases(self, body: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Return NIM tool argument aliases captured while building this request."""
        return nim_tool_argument_aliases_from_body(body)

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Retry once with a downgraded body when NIM rejects a known field."""
        status_code = getattr(error, "status_code", None)
        if not isinstance(error, openai.BadRequestError) and status_code != 400:
            return None

        error_text = str(error)
        error_body = getattr(error, "body", None)
        if error_body is not None:
            error_text = f"{error_text} {json.dumps(error_body, default=str)}"
        error_text = error_text.lower()

        if "reasoning_budget" in error_text:
            retry_body = clone_body_without_reasoning_budget(body)
            if retry_body is None:
                return None
            logger.warning(
                "NIM_STREAM: retrying without reasoning_budget after 400 error"
            )
            return retry_body

        if "chat_template" in error_text:
            retry_body = clone_body_without_chat_template(body)
            if retry_body is None:
                return None
            logger.warning("NIM_STREAM: retrying without chat_template after 400 error")
            return retry_body

        if "reasoning_content" in error_text:
            retry_body = clone_body_without_reasoning_content(body)
            if retry_body is None:
                return None
            logger.warning(
                "NIM_STREAM: retrying without reasoning_content after 400 error"
            )
            return retry_body

        return None

    # ------------------------------------------------------------------
    # Rate limiting hook – intercepta cada llamada HTTP real
    # ------------------------------------------------------------------
    def _send_request(self, *args, **kwargs) -> Any:
        """
        Envuelve la petición HTTP original para aplicar rate limiting.
        IMPORTANTE: Si OpenAIChatTransport usa otro nombre de método (ej. _request,
        _post, _make_request), cambia el nombre de este override.
        """
        self._rate_limiter.acquire()          # espera si es necesario
        return super()._send_request(*args, **kwargs)
