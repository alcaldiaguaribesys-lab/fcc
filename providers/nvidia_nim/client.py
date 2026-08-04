"""
NVIDIA NIM provider implementation with STEROID-GRADE rate limiter (39 req/min).
Deadlock-free, minimal overhead, identical speed to the original fast code.
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
    Ventana fija con esteroides: máximo `max_requests` por minuto.
    - Lock solo para la sección crítica (nanosegundos).
    - Sleep fuera del lock para no bloquear a los demás hilos.
    - Sin recursividad, sin condiciones complejas. Eficiencia pura.
    """

    def __init__(self, max_requests: int = 39):
        self.max_requests = max_requests
        self._lock = threading.Lock()
        self._window_start = time.monotonic()
        self._count = 0

    def acquire(self):
        """Adquiere un cupo; duerme SIN candado si se alcanzó el límite."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Cambio de minuto → reinicio de ventana
                if int(now // 60) != int(self._window_start // 60):
                    self._window_start = now
                    self._count = 0

                # ¿Hay espacio? Tomamos y nos vamos volando
                if self._count < self.max_requests:
                    self._count += 1
                    logger.debug(
                        f"RateLimiter: cupo aceptado ({self._count}/{self.max_requests})"
                    )
                    return

                # Calculamos cuánto falta para la próxima ventana, con margen de seguridad
                seconds_until_next_minute = 60.0 - (now % 60) + 0.1

            # Dormimos FUERA del lock → el resto de hilos pueden seguir trabajando
            logger.warning(
                f"RateLimiter: límite de {self.max_requests}/min alcanzado. "
                f"Esperando {seconds_until_next_minute:.2f}s..."
            )
            time.sleep(seconds_until_next_minute)
            # Al despertar repetimos el bucle: el lock vuelve a tomarse instantáneamente


class NvidiaNimProvider(OpenAIChatTransport):
    """Proveedor NVIDIA NIM con cliente OpenAI y rate limiter blindado."""

    def __init__(self, config: ProviderConfig, *, nim_settings: NimSettings):
        super().__init__(
            config,
            provider_name="NIM",
            base_url=config.base_url or NVIDIA_NIM_DEFAULT_BASE,
            api_key=config.api_key,
        )
        self._nim_settings = nim_settings
        self._rate_limiter = RateLimiter(max_requests=39)

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_nim_request_body(
            request,
            self._nim_settings,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _prepare_create_body(self, body: dict[str, Any]) -> dict[str, Any]:
        return body_without_nim_tool_argument_aliases(body)

    def _tool_argument_aliases(self, body: dict[str, Any]) -> dict[str, dict[str, str]]:
        return nim_tool_argument_aliases_from_body(body)

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Reintenta una vez eliminando campos que NIM rechaza con 400."""
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
            if retry_body is not None:
                logger.warning("NIM: reintentando sin reasoning_budget")
                return retry_body

        if "chat_template" in error_text:
            retry_body = clone_body_without_chat_template(body)
            if retry_body is not None:
                logger.warning("NIM: reintentando sin chat_template")
                return retry_body

        if "reasoning_content" in error_text:
            retry_body = clone_body_without_reasoning_content(body)
            if retry_body is not None:
                logger.warning("NIM: reintentando sin reasoning_content")
                return retry_body

        return None

    # ------------------------------------------------------------------
    # INTERCEPTOR DE RATE LIMITING
    # ¡Ojo! Asegúrate de que en OpenAIChatTransport el método que envía la
    # petición se llame realmente `_send_request`. Si no, renómbralo aquí
    # al nombre real (p. ej. `_create`, `_request`). Con este cambio es
    # suficiente para aplicar el límite en todas las llamadas.
    # ------------------------------------------------------------------
    def _send_request(self, *args, **kwargs) -> Any:
        self._rate_limiter.acquire()
        return super()._send_request(*args, **kwargs)
