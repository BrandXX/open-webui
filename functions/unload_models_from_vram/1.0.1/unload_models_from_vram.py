"""
title: Unload Models from VRAM
description: Unloads all models from VRAM using Ollama's REST API.
    - The endpoint URL is configurable (default: http://host.docker.internal:11434 )
    - The timeout is configurable. (default: 3-secs)
author: BrandXX/UserX
author_url: https://github.com/BrandXX/open-webui/
funding_url: https://github.com/BrandXX/open-webui/
repo_url: https://github.com/BrandXX/open-webui/blob/main/functions/unload_models_from_vram/1.0.0/unload_models_from_vram.py
version: 1.0.1
required_open_webui_version: 0.3.9
Notes:
To unload the models from VRAM, please click the 'Unload Models from VRAM' icon next to the 'Regenerate' icon at the bottom of the chat.
"""

import asyncio
import time
import logging
import requests
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Action:
    class Valves(BaseModel):
        OLLAMA_ENDPOINT: str = Field(
            default="http://host.docker.internal:11434",
            description="URL to the Ollama API endpoint",
            advanced=False,
        )
        TIMEOUT_SECONDS: int = Field(
            default=3,
            description="Timeout in seconds for API requests (set to 3 seconds)",
            advanced=False,
        )
        LOGGING_LEVEL: str = Field(
            default="INFO",
            description="Set the logging verbosity level.",
            advanced=True,
            enum=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        )

    def __init__(self):
        self.valves = self.Valves()
        logger.setLevel(getattr(logging, self.valves.LOGGING_LEVEL))

    def get_loaded_models(self) -> Tuple[List[str], Optional[str], str]:
        start_time = time.time()
        try:
            url = f"{self.valves.OLLAMA_ENDPOINT}/api/ps"
            response = requests.get(
                url, verify=False, timeout=self.valves.TIMEOUT_SECONDS
            )
            response.raise_for_status()
            data = response.json()
            models = [
                item.get("name") for item in data.get("models", []) if item.get("name")
            ]
            logger.debug("Models retrieved via REST API: %s", models)
            return models, None, ""
        except (requests.exceptions.InvalidURL, requests.exceptions.MissingSchema) as e:
            elapsed = time.time() - start_time
            if elapsed < self.valves.TIMEOUT_SECONDS:
                time.sleep(self.valves.TIMEOUT_SECONDS - elapsed)
            error_msg = f"Endpoint error: The URL '{self.valves.OLLAMA_ENDPOINT}' appears to be misconfigured."
            logger.error("%s Exception details: %s", error_msg, e)
            return [], "invalid_url", error_msg
        except (requests.Timeout, requests.ConnectionError) as e:
            elapsed = time.time() - start_time
            if elapsed < self.valves.TIMEOUT_SECONDS:
                time.sleep(self.valves.TIMEOUT_SECONDS - elapsed)
            error_status = (
                "timeout" if isinstance(e, requests.Timeout) else "connection_error"
            )
            error_msg = (
                f"{error_status.capitalize()} error: Could not reach the endpoint."
            )
            logger.error("%s Exception details: %s", error_msg, e)
            return [], error_status, error_msg
        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed < self.valves.TIMEOUT_SECONDS:
                time.sleep(self.valves.TIMEOUT_SECONDS - elapsed)
            error_msg = f"Unexpected error retrieving models: {e}"
            logger.exception(error_msg)
            return [], "exception", error_msg

    def unload_model(self, model: str) -> str:
        try:
            payload = {"model": model, "keep_alive": 0}
            url = f"{self.valves.OLLAMA_ENDPOINT}/api/generate"
            response = requests.post(
                url, json=payload, verify=False, timeout=self.valves.TIMEOUT_SECONDS
            )
            response.raise_for_status()
            result_text = response.text.strip()
            logger.debug("Unload response for %s: %s", model, result_text)
            return f"Model {model} unloaded successfully: {result_text}"
        except Exception as e:
            logger.exception("Error unloading model %s:", model)
            return f"Error unloading model {model}: {e}"

    async def _emit(self, emitter, description: str, done: bool):
        if emitter:
            await emitter(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        """
        Asynchronous action triggered from the chat.
        Prompts the user for confirmation, retrieves loaded models, and then unloads each model.
        Provides clear emitter messages for misconfigured endpoints, timeouts, and connection issues.
        """
        try:
            await self._emit(__event_emitter__, "Retrieving loaded models...", False)
            retrieval_start = time.time()
            models, error_status, error_msg = self.get_loaded_models()
            retrieval_elapsed = time.time() - retrieval_start
            if retrieval_elapsed < self.valves.TIMEOUT_SECONDS:
                await asyncio.sleep(self.valves.TIMEOUT_SECONDS - retrieval_elapsed)

            if error_status:
                final_msg = f"Error retrieving models: {error_msg}"
                await self._emit(__event_emitter__, final_msg, True)
                return {"type": "unload_result", "data": {"message": final_msg}}

            if not models:
                no_models_msg = "No models loaded into VRAM. Nothing to unload! 🎈"
                await self._emit(__event_emitter__, no_models_msg, True)
                return {"type": "unload_result", "data": {"message": no_models_msg}}

            messages = []
            for model in models:
                await self._emit(
                    __event_emitter__, f"Unloading model {model}...", False
                )
                messages.append(self.unload_model(model))

            final_message = "\n".join(messages)
            await self._emit(__event_emitter__, "Unload complete! ✅", True)
            return {"type": "unload_result", "data": {"message": final_message}}

        except Exception as e:
            error_msg = f"Unexpected error during unload: {e}"
            logger.exception(error_msg)
            await self._emit(__event_emitter__, error_msg, True)
            return {"type": "unload_result", "data": {"message": error_msg}}
