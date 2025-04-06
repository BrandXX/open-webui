"""
title: Gemini Manifold Pipe
author: justinh-rahb
author_url: https://github.com/justinh-rahb
funding_url: https://github.com/open-webui
version: 0.1.4
license: MIT
"""

import os
import json
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, GenerateContentResponse
from typing import List, Union, Iterator

# Set DEBUG to True to enable detailed logging
DEBUG = False


class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(default="")
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)

    def __init__(self):
        self.id = "google_genai"
        self.type = "manifold"
        self.name = "Google: "
        self.valves = self.Valves(
            **{
                "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
                "USE_PERMISSIVE_SAFETY": False,
            }
        )

    def get_google_models(self):
        if not self.valves.GOOGLE_API_KEY:
            return [
                {
                    "id": "error",
                    "name": "GOOGLE_API_KEY is not set. Please update the API Key in the valves.",
                }
            ]
        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            models = genai.list_models()
            return [
                {
                    "id": model.name[7:],  # remove the "models/" part
                    "name": model.display_name,
                }
                for model in models
                if "generateContent" in model.supported_generation_methods
                if model.name.startswith("models/")
            ]
        except Exception as e:
            if DEBUG:
                print(f"Error fetching Google models: {e}")
            return [
                {"id": "error", "name": f"Could not fetch models from Google: {str(e)}"}
            ]

    def pipes(self) -> List[dict]:
        return self.get_google_models()

    def pipe(self, body: dict) -> Union[str, Iterator[str]]:
        if not self.valves.GOOGLE_API_KEY:
            return "Error: GOOGLE_API_KEY is not set"
        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            model_id = body["model"]

            if model_id.startswith("google_genai."):
                model_id = model_id[12:]

            model_id = model_id.lstrip(".")

            if not model_id.startswith("gemini-"):
                return f"Error: Invalid model name format: {model_id}"

            messages = body["messages"]
            stream = body.get("stream", False)

            if DEBUG:
                print("Incoming body:", str(body))

            system_message = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), None
            )

            contents = []
            for message in messages:
                if message["role"] != "system":
                    if isinstance(message.get("content"), list):
                        parts = []
                        for content in message["content"]:
                            if content["type"] == "text":
                                parts.append({"text": content["text"]})
                            elif content["type"] == "image_url":
                                image_url = content["image_url"]["url"]
                                if image_url.startswith("data:image"):
                                    image_data = image_url.split(",")[1]
                                    parts.append(
                                        {
                                            "inline_data": {
                                                "mime_type": "image/jpeg",
                                                "data": image_data,
                                            }
                                        }
                                    )
                                else:
                                    parts.append({"image_url": image_url})
                        contents.append({"role": message["role"], "parts": parts})
                    else:
                        contents.append(
                            {
                                "role": (
                                    "user" if message["role"] == "user" else "model"
                                ),
                                "parts": [{"text": message["content"]}],
                            }
                        )

            if system_message:
                contents.insert(
                    0,
                    {"role": "user", "parts": [{"text": f"System: {system_message}"}]},
                )

            if "gemini-1.5" in model_id:
                model = genai.GenerativeModel(
                    model_name=model_id, system_instruction=system_message
                )
            else:
                model = genai.GenerativeModel(model_name=model_id)

            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )

            # Safety settings omitted for brevity...
            if self.valves.USE_PERMISSIVE_SAFETY:
                safety_settings = {
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            else:
                safety_settings = body.get("safety_settings")

            if DEBUG:
                print("Google API request:")
                print("  Model:", model_id)
                print("  Contents:", str(contents))
                print("  Generation Config:", generation_config)
                print("  Safety Settings:", safety_settings)
                print("  Stream:", stream)

            if stream:

                def stream_generator():
                    response = model.generate_content(
                        contents,
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                        stream=True,
                    )
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text

                return stream_generator()
            else:
                response = model.generate_content(
                    contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=False,
                )
                return response.text
        except Exception as e:
            if DEBUG:
                print(f"Error in pipe method: {e}")
            return f"Error: {e}"
