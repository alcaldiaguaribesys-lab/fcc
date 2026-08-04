"""
NVIDIA NIM provider implementation with steroid‑grade rate limiter (39 req/min).
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
    Ventana fija turbo: 39 peticiones por minuto.
    Usa threading.Condition para despertar a todos los hilos exactamente al abrir
    una nueva ventana → cero polling, cero cuellos de botella.
    """

    def __init__(self, max_requests: int = 39):
        self.max_requests = max_requests
        self._cond = threading.Condition()
        self._window_start = time.monotonic()
        self._count = 0

    def acquire(self):
        """Bloquea hasta conseguir un cupo, con la mínima latencia posible."""
        with self._cond:
            while True:
                now = time.monotonic()
                current_minute = int(now // 60)
                window_minute = int(self._window_start // 60)

                # ¿Nueva ventana? Reiniciamos contador y despertamos a los que esperan
                if current_minute != window_minute:
                    self._window_start = now
                    self._count = 0
                    self._cond.notify_all()   # 🚀 todos a la vez

                # Si hay sitio, lo tomamos y salimos
                if self._count < self.max_requests:
                    self._count += 1
                    logger.debug(
                        f"RateLimiter: cupo {self._count}/{self.max_requests}"
                    )
                    return

                # Calculamos exactamente cuándo empieza el siguiente minuto
                seconds_until_next = 60 - (now % 60)
                if seconds_until_next <= 0:
                    seconds_until_next = 0.1  # margen de seguridad

                logger.info(
                    f"RateLimiter: límite alcanzado, durmiendo {seconds_until_next:.2f}s"
                )
                # wait() libera el lock y espera hasta que:
                #   - alguien llame notify_all() (nueva ventana), o
                #   - se cumpla el timeout
                self._cond.wait(timeout=seconds_until_next)
                # Al despertar, el bucle reevalúa inmediatamente


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
    # Interceptor de rate limiting – se engancha a la petición HTTP real
    # ------------------------------------------------------------------
    def _send_request(self, *args, **kwargs) -> Any:
        """
        Aplica el límite antes de enviar la petición.
        ⚠️ Si el método de la clase base se llama distinto (ej. _request, _post),
        renombra este override y la llamada a super().
        """
        self._rate_limiter.acquire()
        return super()._send_request(*args, **kwargs)
