"""NVIDIA NIM provider client con limitador de tasa a 39 RPM para free-claude-code."""

import asyncio
import logging
import time
from collections import deque
from typing import Any, AsyncGenerator, Dict, Optional

logger = logging.getLogger(__name__)


class NIMRateLimiter:
    """Limitador de ventana deslizante para NVIDIA NIM (Máximo 39 RPM)."""

    def __init__(self, max_rpm: int = 39, window_seconds: float = 60.0):
        self.max_rpm = max_rpm
        self.window = window_seconds
        self.timestamps = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.time()

            # Eliminar registros anteriores a los últimos 60 segundos
            while self.timestamps and self.timestamps[0] <= now - self.window:
                self.timestamps.popleft()

            # Si se alcanzaron las 39 peticiones en el último minuto
            if len(self.timestamps) >= self.max_rpm:
                sleep_duration = self.window - (now - self.timestamps[0]) + 0.1
                if sleep_duration > 0:
                    logger.info(
                        f"[NIM RateLimiter] Alcanzado el límite de {self.max_rpm} RPM. "
                        f"Pausando {sleep_duration:.2f} segundos..."
                    )
                    await asyncio.sleep(sleep_duration)

                # Limpiar marcas de tiempo tras la pausa
                now = time.time()
                while self.timestamps and self.timestamps[0] <= now - self.window:
                    self.timestamps.popleft()

            self.timestamps.append(time.time())


# Instancia compartida global en el servidor
nim_limiter = NIMRateLimiter(max_rpm=39)


class NvidiaNimClient:
    """Cliente para interaccionar con la API de NVIDIA NIM regulado a 39 RPM."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs: Any):
        self.api_key = api_key
        self.base_url = base_url or "https://integrate.api.nvidia.com/v1"
        self.extra_options = kwargs

    async def _before_request(self) -> None:
        """Aplica la pausa automática antes de llamar a la API de NIM."""
        await nim_limiter.acquire()

    async def create_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Envía peticiones normales (non-streaming) aplicando el rate limit."""
        await self._before_request()
        
        # Aquí se realiza la llamada HTTP a NIM
        # (Asegúrate de incluir tu lógica existente de cliente httpx/aiohttp)
        return {}

    async def stream_chat_completion(self, payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Envía peticiones en streaming aplicando el rate limit antes del inicio."""
        await self._before_request()

        # Generador SSE existente
        if False:
            yield {}
