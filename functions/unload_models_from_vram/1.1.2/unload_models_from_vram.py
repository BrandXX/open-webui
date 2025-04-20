"""
title: Unload Models from VRAM
icon_url: data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjI1IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0xMiAxM1Y3Ii8+PHBhdGggZD0ibTE1IDEwLTMgMy0zLTMiLz48cmVjdCB3aWR0aD0iMjAiIGhlaWdodD0iMTQiIHg9IjIiIHk9IjMiIHJ4PSIyIi8+PHBhdGggZD0iTTEyIDE3djQiLz48cGF0aCBkPSJNOCAyMWg4Ii8+PC9zdmc+
description: Unloads all models from GPU VRAM via the Ollama REST API.
author: BrandXX/UserX
author_url: https://github.com/BrandXX/open-webui/
funding_url: https://github.com/BrandXX/open-webui/
repo_url: https://github.com/BrandXX/open-webui/blob/main/functions/unload_models_from_vram/1.1.2/unload_models_from_vram.py
version: 1.1.2
required_open_webui_version: 0.6.0
Notes:
  Key valves:
    - OLLAMA_ENDPOINT (str): base URL of Ollama server
    - REQUEST_TIMEOUT (int): per-request timeout in seconds
    - VERIFY_SSL (bool): verify TLS certificates
    - LOG_LEVEL (str): Python logging verbosity (DEBUG|INFO|WARNING|ERROR|CRITICAL)
  To unload the models from VRAM, click the 'Unload Models from VRAM' icon next to 'Regenerate'.
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
        payload = {
            "model": name,
            "prompt": "",
            "keep_alive": 0,
            "stream": False,  # <‑‑ get one JSON blob, no stream
        }
        resp = self._call_api("POST", "/api/generate", json=payload)
        resp.raise_for_status()
        time.sleep(0.2)  # give the emitter time to show progress
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
