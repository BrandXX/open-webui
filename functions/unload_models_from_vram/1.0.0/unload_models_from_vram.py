"""
title: Unload Models from VRAM
description: |
  - Unloads all models from VRAM using Ollama's REST API.
  - The endpoint URL is configurable (default: http://host.docker.internal:11434 )
  - The timeout is configurable. (default: 3-secs)
author: BrandXX/UserX
author_url: https://github.com/BrandXX/open-webui/
funding_url: https://github.com/BrandXX/open-webui/
repo_url: https://github.com/BrandXX/open-webui/blob/main/functions/unload_models_from_vram/.07/unload_models_from_vram.yaml
version: 1.0.0
required_open_webui_version: 0.3.9
Notes:
To unload the models from VRAM, please click the 'Unload Models from VRAM' icon next to the 'Regenerate' icon at the bottom of the chat. Type 'yes' to confirm.
"""

import asyncio
import time
import requests
from pydantic import BaseModel, Field
from typing import Optional, List


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
        pass

    def __init__(self):
        self.valves = self.Valves()

    def get_loaded_models(self) -> List[str]:
        """
        Retrieves a list of loaded models via the configured REST API endpoint.
        It calls {OLLAMA_ENDPOINT}/api/ps and expects a JSON response.
        A timeout of 3 seconds is applied. If an error occurs,
        a delay is enforced so the error message isn’t instantaneous.
        """
        start_time = time.time()
        try:
            url = f"{self.valves.OLLAMA_ENDPOINT}/api/ps"
            response = requests.get(
                url, verify=False, timeout=self.valves.TIMEOUT_SECONDS
            )
            response.raise_for_status()
            data = response.json()
            models = []
            for item in data.get("models", []):
                model_name = item.get("name")
                if model_name:
                    models.append(model_name)
            print("DEBUG: Models retrieved via REST API:", models)
            return models
        except (requests.Timeout, requests.ConnectionError) as e:
            elapsed = time.time() - start_time
            if elapsed < self.valves.TIMEOUT_SECONDS:
                time.sleep(self.valves.TIMEOUT_SECONDS - elapsed)
            if isinstance(e, requests.Timeout):
                error_msg = (
                    f"Request timed out after {self.valves.TIMEOUT_SECONDS} seconds."
                )
            else:
                error_msg = (
                    "Connection error: Could not reach the endpoint. "
                    "This might be due to an endpoint URL misconfiguration."
                )
            print("DEBUG:", error_msg)
            return []
        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed < self.valves.TIMEOUT_SECONDS:
                time.sleep(self.valves.TIMEOUT_SECONDS - elapsed)
            print("DEBUG: Error retrieving models via REST API:", e)
            return []

    def unload_model(self, model: str) -> str:
        """
        Unloads the given model via REST API by sending a POST request.
        It calls {OLLAMA_ENDPOINT}/api/generate with the payload {"model": model, "keep_alive": 0}.
        A timeout of 3 seconds is applied.
        """
        try:
            payload = {"model": model, "keep_alive": 0}
            url = f"{self.valves.OLLAMA_ENDPOINT}/api/generate"
            response = requests.post(
                url, json=payload, verify=False, timeout=self.valves.TIMEOUT_SECONDS
            )
            response.raise_for_status()
            result_text = response.text.strip()
            print(f"DEBUG: Unload response for {model}: {result_text}")
            return f"Model {model} unloaded successfully: {result_text}"
        except (requests.Timeout, requests.ConnectionError) as e:
            if isinstance(e, requests.Timeout):
                error_msg = f"Request timed out after {self.valves.TIMEOUT_SECONDS} seconds while unloading {model}."
            else:
                error_msg = f"Connection error while unloading {model}. Check the endpoint configuration."
            print("DEBUG:", error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error unloading model {model}: {e}"
            print("DEBUG:", error_msg)
            return error_msg

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        """
        Asynchronous action triggered from the chat.
        Prompts the user for confirmation once and retrieves loaded models via the REST API.
        If no models are retrieved (e.g., due to a timeout or unreachable endpoint),
        it displays a clear error message and does NOT re-prompt for confirmation.
        Then it unloads each model via the configured unload REST API.
        """
        # Request confirmation from the user (only once)
        confirmation = await __event_call__(
            {
                "type": "input",
                "data": {
                    "title": "Unload Models Confirmation",
                    "message": "Type 'yes' to confirm unloading all loaded models via REST API.",
                    "placeholder": "yes/no",
                },
            }
        )

        if confirmation.lower() != "yes":
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Unload cancelled", "done": True},
                    }
                )
            return {
                "type": "unload_result",
                "data": {"message": "Unload cancelled by user."},
            }

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Retrieving loaded models via REST API...",
                        "done": False,
                    },
                }
            )

        models = self.get_loaded_models()
        if not models:
            error_msg = (
                "No models were retrieved. This might be due to a timeout, nothing loaded"
                ", or an endpoint URL misconfiguration."
            )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            # Return the error message without re-prompting for input.
            return {"type": "unload_result", "data": {"message": error_msg}}

        messages = []
        for model in models:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Unloading model {model}...",
                            "done": False,
                        },
                    }
                )
            messages.append(self.unload_model(model))

        final_message = "\n".join(messages)
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Unload complete", "done": True},
                }
            )

        return {"type": "unload_result", "data": {"message": final_message}}
