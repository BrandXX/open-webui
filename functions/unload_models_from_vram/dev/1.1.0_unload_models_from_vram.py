
"""
title: Unload Models from VRAM
description: Unloads all models from VRAM using Ollama's REST API.
    - The endpoint URL is configurable (default: http://host.docker.internal:11434 )
    - The timeout is configurable. (default: 3-secs)
    - VERIFY_SSL (bool): verify TLS certificates (default True)
    - The logging level is configurable via a valve dropdown list
author: BrandXX/UserX
author_url: https://github.com/BrandXX/open-webui/
funding_url: https://github.com/BrandXX/open-webui/
repo_url: https://github.com/BrandXX/open-webui/blob/main/functions/unload_models_from_vram/1.0.0/unload_models_from_vram.py
version: 1.0.1
required_open_webui_version: 0.3.9
Notes:
To unload the models from VRAM, please click the 'Unload Models from VRAM' icon next to the 'Regenerate' icon at the bottom of the chat.

Key valves
----------
- OLLAMA_ENDPOINT   (str) : base URL of Ollama server
- TIMEOUT_SECONDS   (int) : total per‑request timeout
- VERIFY_SSL        (bool): verify TLS certificates (default True)
- LOGGING_LEVEL     (str) : DEBUG | INFO | WARNING | ERROR | CRITICAL
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import requests
from pydantic import BaseModel, Field
from requests import Session, exceptions as rex

__all__ = ["Action"]

_LOG = logging.getLogger(__name__).getChild("unload")


class _HTTPMixin:
    """Lazy singleton for a `requests.Session` with valve‑aware settings."""

    _session: Optional[Session] = None

    @classmethod
    def get_session(cls, verify_ssl: bool, timeout: int) -> Session:
        if cls._session is None:
            sess = requests.Session()
            # Connection‑pool tuning (keep‑alive).
            adapter = requests.adapters.HTTPAdapter(pool_maxsize=5, pool_block=True)
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)
            sess.verify = verify_ssl
            sess.timeout = timeout
            cls._session = sess
        return cls._session


class Action(_HTTPMixin):
    class Valves(BaseModel):
        OLLAMA_ENDPOINT: str = Field(
            default="http://host.docker.internal:11434",
            description="Base URL of the Ollama API.",
        )
        TIMEOUT_SECONDS: int = Field(
            default=3, description="Per‑request timeout (seconds)."
        )
        VERIFY_SSL: bool = Field(
            default=True,
            description="Whether to verify TLS certificates for https endpoints.",
        )
        LOGGING_LEVEL: str = Field(
            default="INFO",
            enum=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            description="Python logging level.",
        )

    def __init__(self):
        self.valves = self.Valves()
        _LOG.setLevel(self.valves.LOGGING_LEVEL.upper())

    # --------------------------------------------------------------------- helpers
    def _session(self) -> Session:
        return self.get_session(self.valves.VERIFY_SSL, self.valves.TIMEOUT_SECONDS)

    def _request(
        self, method: str, path: str, **kwargs
    ) -> tuple[dict | str | None, Optional[str]]:
        """Wrapper that centralizes error mapping."""
        try:
            url = f"{self.valves.OLLAMA_ENDPOINT}{path}"
            r = self._session().request(
                method, url, timeout=self.valves.TIMEOUT_SECONDS, **kwargs
            )
            r.raise_for_status()
            return r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text, None
        except rex.InvalidURL:
            return None, "invalid_url"
        except rex.Timeout:
            return None, "timeout"
        except rex.ConnectionError:
            return None, "connection_error"
        except rex.RequestException as exc:
            _LOG.exception("Unhandled requests error: %s", exc)
            return None, "request_error"

    # ------------------------------------------------------------------- API calls
    def get_loaded_models(self) -> tuple[list[str], Optional[str]]:
        data, err = self._request("GET", "/api/ps")
        if err:
            return [], err
        models = [m["name"] for m in data.get("models", []) if m.get("name")]
        return models, None

    def unload_model(self, model: str) -> bool:
        _, err = self._request(
            "POST", "/api/generate", json={"model": model, "keep_alive": 0}
        )
        return err is None

    # -------------------------------------------------------------------- WebUI hook
    async def _emit(self, emitter, desc: str, done=False):
        if emitter:
            await emitter({"type": "status", "data": {"description": desc, "done": done}})

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        """
        Chat‑triggered entrypoint (Open‑WebUI).

        Returns
        -------
        dict  e.g. {"type": "unload_result", "data": {...}}
        """
        await self._emit(__event_emitter__, "Querying GPU‑resident models…")
        models, err = self.get_loaded_models()
        if err:
            msg = f"❌ Unable to retrieve models ({err}). Check endpoint/SSL."
            await self._emit(__event_emitter__, msg, True)
            return {"type": "unload_result", "data": {"error": err, "message": msg}}

        if not models:
            msg = "🎈 VRAM is already clear — nothing to unload."
            await self._emit(__event_emitter__, msg, True)
            return {"type": "unload_result", "data": {"message": msg}}

        await self._emit(
            __event_emitter__, f"Unloading {len(models)} model(s)… this may take a few seconds."
        )

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(None, self.unload_model, m) for m in models
        ]
        results = await asyncio.gather(*tasks)

        success = [m for m, ok in zip(models, results) if ok]
        failed = [m for m, ok in zip(models, results) if not ok]

        summary = f"✅ Unloaded: {', '.join(success)}"
        if failed:
            summary += f" • ⚠️ Failed: {', '.join(failed)}"

        await self._emit(__event_emitter__, "Unload complete!", True)
        return {
            "type": "unload_result",
            "data": {"success": success, "failed": failed, "message": summary},
        }
