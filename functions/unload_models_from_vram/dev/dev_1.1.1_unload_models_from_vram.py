"""
title: Unload Models from VRAM
description: Unloads/frees all models from GPU's VRAM using Ollama's REST API.
    Key valves
    ----------
        - OLLAMA_ENDPOINT   (str) : base URL of Ollama server
        - REQUEST_TIMEOUT   (int) : total per‑request timeout
        - VERIFY_SSL        (bool): verify TLS certificates (default True)
        - LOG_LEVEL     (str) : DEBUG | INFO | WARNING | ERROR | CRITICAL
author: BrandXX/UserX
author_url: https://github.com/BrandXX/open-webui/
funding_url: https://github.com/BrandXX/open-webui/
repo_url: https://github.com/BrandXX/open-webui/blob/main/functions/unload_models_from_vram/1.0.0/unload_models_from_vram.py
version: 1.1.1
required_open_webui_version: 0.3.9
Notes:
To unload the models from VRAM, please click the 'Unload Models from VRAM' icon next to the 'Regenerate' icon at the bottom of the chat.
"""

import asyncio
import logging
import requests
from typing import List, Optional
from pydantic import BaseModel, Field

logging.basicConfig(format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)


class Action:
    # ------------------------------------------------------------------ #
    class Valves(BaseModel):
        OLLAMA_ENDPOINT: str = Field(
            default="http://host.docker.internal:11434",
            description="Base URL of the Ollama REST API",
        )
        REQUEST_TIMEOUT: int = Field(
            default=3,
            description="HTTP timeout in seconds",
        )
        VERIFY_SSL: bool = Field(
            default=True,
            description="Verify HTTPS certificates",
        )
        LOG_LEVEL: str = Field(
            default="INFO",
            enum=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            description="Python logging verbosity",
        )

    # ------------------------------------------------------------------ #
    def __init__(self):
        self.valves = self.Valves()
        log.setLevel(self.valves.LOG_LEVEL)

    # ============ helper methods (sync, run in a thread) =============== #
    def _call_api(self, method: str, path: str, **kw):
        url = f"{self.valves.OLLAMA_ENDPOINT}{path}"
        kw.setdefault("timeout", self.valves.REQUEST_TIMEOUT)
        kw.setdefault("verify", self.valves.VERIFY_SSL)
        return requests.request(method, url, **kw)

    def _list_models(self) -> List[str]:
        resp = self._call_api("GET", "/api/ps")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def _unload(self, name: str) -> str:
        payload = {"model": name, "prompt": "", "keep_alive": 0}
        resp = self._call_api("POST", "/api/generate", json=payload)
        resp.raise_for_status()
        return f"✅ Unloaded **{name}**"

    # ====================== entry point for WebUI ====================== #
    async def action(
        self,
        body: dict,
        __event_emitter__=None,
        **_,
    ) -> dict:
        """Unload one or more models; if none specified, unload all."""
        wanted: Optional[List[str]] = body.get("models")

        async def emit(msg, done=False, **extra):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": msg, "done": done, **extra},
                    }
                )

        await emit("🔍 Gathering loaded models…")
        try:
            models = await asyncio.to_thread(self._list_models)
        except Exception as e:
            await emit(f"⚠️ Failed to query Ollama: {e}", done=True)
            return {"type": "unload_result", "data": {"message": str(e)}}

        targets = models if not wanted else [m for m in models if m in wanted]
        if not targets:
            await emit("🎈 Nothing to unload.", done=True)
            return {
                "type": "unload_result",
                "data": {"message": "No matching models in VRAM."},
            }

        msgs, total = [], len(targets)
        for i, model in enumerate(targets, 1):
            await emit(f"⏏️ Unloading {model} ({i}/{total})…", progress=i / total)
            try:
                msgs.append(await asyncio.to_thread(self._unload, model))
            except Exception as e:
                msgs.append(f"⚠️ {model}: {e}")

        await emit("🏁 All done!", done=True, progress=1.0)
        return {"type": "unload_result", "data": {"message": "\n".join(msgs)}}
