# Component Templates for Open-WebUI

This document provides templates and examples for creating Functions, Tools, and Pipelines in Open-WebUI.

## Function Template

Functions extend Open-WebUI itself by adding new capabilities to the platform.

### Standard Metadata Header

All components should include a standardized metadata header:

```python
"""
title: Function Name
author: YourName
author_url: https://github.com/yourusername
git_url: https://github.com/yourusername/your-repo
description: A brief description of what this function does
requirements: package1,package2  # Optional: comma-separated list of required packages
version: 0.1.0
license: MIT
"""
```

### Basic Function Structure

```python
"""
title: Example Function
author: YourName
version: 0.1.0
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, List

class Function:
    """
    A function that extends Open-WebUI capabilities.

    This function [brief description of what the function does].
    """

    class Config(BaseModel):
        """Configuration parameters for the function."""
        PARAMETER_NAME: str = Field(
            default="default_value",
            description="Description of what this parameter does"
        )
        ANOTHER_PARAMETER: int = Field(
            default=42,
            description="Description of another parameter"
        )
        # Add more parameters as needed

    def __init__(self):
        """Initialize the function with default configuration."""
        self.config = self.Config()

    async def execute(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Any] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Execute the function.

        Args:
            body: The request body containing input parameters
            __user__: User information (automatically injected)
            __request__: Request object (automatically injected)

        Returns:
            The function result as a string or dictionary
        """
        # Implementation goes here
        result = f"Function executed with parameters: {self.config.PARAMETER_NAME}"
        return result
```

### Example: Google Gemini Integration Function

```python
"""
title: Gemini Manifold Pipe
author: YourName
version: 0.1.0
license: MIT
"""

import os
import json
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from typing import List, Union, Iterator, Dict, Any, Optional

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
        """Retrieve available Google Gemini models."""
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
            return [
                {"id": "error", "name": f"Could not fetch models from Google: {str(e)}"}
            ]

    def pipes(self) -> List[dict]:
        """Return available models for the UI."""
        return self.get_google_models()

    def pipe(self, body: Dict[str, Any]) -> Union[str, Iterator[str]]:
        """Process a request through the Google Gemini API.

        Args:
            body: The request body containing model, messages, and parameters

        Returns:
            The model's response as a string or a stream of strings
        """
        if not self.valves.GOOGLE_API_KEY:
            return "Error: GOOGLE_API_KEY is not set"
        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            model_id = body["model"]

            # Handle model ID format
            if model_id.startswith("google_genai."):
                model_id = model_id[12:]
            model_id = model_id.lstrip(".")

            if not model_id.startswith("gemini-"):
                return f"Error: Invalid model name format: {model_id}"

            messages = body["messages"]
            stream = body.get("stream", False)

            # Extract system message if present
            system_message = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), None
            )

            # Format messages for Gemini API
            contents = self._format_messages(messages)

            # Initialize the model
            if "gemini-1.5" in model_id:
                model = genai.GenerativeModel(
                    model_name=model_id, system_instruction=system_message
                )
            else:
                model = genai.GenerativeModel(model_name=model_id)

            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )

            # Configure safety settings
            safety_settings = self._get_safety_settings(body)

            # Handle streaming vs. non-streaming responses
            if stream:
                return self._generate_stream(model, contents, generation_config, safety_settings)
            else:
                response = model.generate_content(
                    contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=False,
                )
                return response.text
        except Exception as e:
            return f"Error: {e}"

    def _format_messages(self, messages):
        """Format messages for the Gemini API."""
        contents = []
        for message in messages:
            if message["role"] != "system":
                if isinstance(message.get("content"), list):
                    # Handle multimodal content (text + images)
                    parts = []
                    for content in message["content"]:
                        if content["type"] == "text":
                            parts.append({"text": content["text"]})
                        elif content["type"] == "image_url":
                            # Handle base64 images
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
                    # Handle text-only content
                    contents.append(
                        {
                            "role": (
                                "user" if message["role"] == "user" else "model"
                            ),
                            "parts": [{"text": message["content"]}],
                        }
                    )
        return contents

    def _get_safety_settings(self, body):
        """Configure safety settings for the Gemini API."""
        if self.valves.USE_PERMISSIVE_SAFETY:
            return {
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT:
                    genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH:
                    genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
                    genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                    genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
        else:
            return body.get("safety_settings")

    def _generate_stream(self, model, contents, generation_config, safety_settings):
        """Generate a streaming response."""
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
```

## Tool Template

Tools extend LLM capabilities by allowing them to access external data and services.

### Basic Tool Structure

```python
"""
title: Example Tool
author: YourName
author_url: https://github.com/yourusername
git_url: https://github.com/yourusername/your-repo
description: A brief description of what this tool does
requirements: package1,package2  # Optional: comma-separated list of required packages
version: 0.1.0
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union, List, Callable

# Helper class for standardized event emission
class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def progress_update(self, description):
        await self.emit(description)

    async def error_update(self, description):
        await self.emit(description, "error", True)

    async def success_update(self, description):
        await self.emit(description, "success", True)

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )

class Tools:
    """
    A tool that extends LLM capabilities.

    This tool [brief description of what the tool does].
    """

    class Valves(BaseModel):
        """Configuration parameters for the tool."""
        API_KEY: str = Field(
            default="",
            description="API key for the external service"
        )
        ENDPOINT_URL: str = Field(
            default="https://api.example.com",
            description="API endpoint URL"
        )
        DEBUG: bool = Field(
            default=False,
            description="Enable debug mode for detailed logging"
        )
        # Add more parameters as needed

    def __init__(self):
        """Initialize the tool with default configuration."""
        self.valves = self.Valves()

    async def execute_tool(
        self,
        parameters: Dict[str, Any],
        __user__: Dict[str, Any] = None,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.

        Args:
            parameters: Tool-specific parameters
            __user__: User information (automatically injected)
            __event_emitter__: Event emitter for progress updates

        Returns:
            The tool execution result
        """
        # Create event emitter for progress updates
        emitter = EventEmitter(__event_emitter__)

        try:
            # Validate parameters
            await emitter.progress_update("Validating parameters")
            if not self._validate_parameters(parameters):
                await emitter.error_update("Invalid parameters provided")
                return {"error": "Invalid parameters"}

            # Execute the main logic
            await emitter.progress_update("Processing request")
            result = await self._process_request(parameters)

            # Return success
            await emitter.success_update("Request processed successfully")
            return {"status": "success", "data": result}

        except Exception as e:
            # Handle errors
            error_message = f"Error: {str(e)}"
            await emitter.error_update(error_message)
            return {"error": error_message}

    def _validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate the input parameters."""
        # Implementation goes here
        return True

    async def _process_request(self, parameters: Dict[str, Any]) -> Any:
        """Process the request and return results."""
        # Implementation goes here
        return "Tool executed successfully"
```

### Example: YouTube Transcript Tool

```python
"""
title: Youtube Transcript Provider
author: YourName
version: 0.1.0
requirements: langchain-yt-dlp
license: MIT
"""

from typing import Any, Callable, Dict, List

from langchain_community.document_loaders import YoutubeLoader
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
from pydantic import BaseModel, Field


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def progress_update(self, description):
        await self.emit(description)

    async def error_update(self, description):
        await self.emit(description, "error", True)

    async def success_update(self, description):
        await self.emit(description, "success", True)

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class Tools:
    class Valves(BaseModel):
        CITATION: bool = Field(
            default=True, description="True or false for citation"
        )

    class UserValves(BaseModel):
        TRANSCRIPT_LANGUAGE: str = Field(
            default="en,en_auto",
            description="A comma-separated list of languages from highest priority to lowest.",
        )
        TRANSCRIPT_TRANSLATE: str = Field(
            default="en",
            description="The language you want the transcript to auto-translate to, if it does not already exist.",
        )
        GET_VIDEO_DETAILS: bool = Field(
            default=True, description="Grab video details, such as title and author"
        )

    def __init__(self):
        self.valves = self.Valves()

    async def get_youtube_transcript(
        self,
        url: str,
        __event_emitter__: Callable[[dict], Any] = None,
        __user__: dict = {},
    ) -> str:
        """
        Provides the title and full transcript of a YouTube video in English.
        Only use if the user supplied a valid YouTube URL.
        Examples of valid YouTube URLs: https://youtu.be/dQw4w9WgXcQ, https://www.youtube.com/watch?v=dQw4w9WgXcQ

        :param url: The URL of the youtube video that you want the transcript for.
        :return: The full transcript of the YouTube video in English, or an error message.
        """
        emitter = EventEmitter(__event_emitter__)

        # Initialize user valves if not present
        if "valves" not in __user__:
            __user__["valves"] = self.UserValves()

        try:
            await emitter.progress_update(f"Validating URL: {url}")

            # Check if the URL is valid
            if not url or url == "":
                raise Exception(f"Invalid YouTube URL: {url}")
            # Prevent common test URL
            elif "dQw4w9WgXcQ" in url:
                raise Exception("Rick Roll URL provided... is that what you want?).")

            # Get video details if the user wants them
            title = ""
            author = ""
            if __user__["valves"].GET_VIDEO_DETAILS:
                await emitter.progress_update("Getting video details")
                details = await YoutubeLoaderDL.from_youtube_url(
                    url, add_video_info=True
                ).aload()

                if len(details) == 0:
                    raise Exception("Failed to get video details")

                title = details[0].metadata["title"]
                author = details[0].metadata["author"]
                await emitter.progress_update(
                    f"Grabbed details for {title} by {author}"
                )

            # Parse language preferences
            languages = [
                item.strip()
                for item in __user__["valves"].TRANSCRIPT_LANGUAGE.split(",")
            ]

            # Get transcript
            transcript = await YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False,
                language=languages,
                translation=__user__["valves"].TRANSCRIPT_TRANSLATE,
            ).aload()

            if len(transcript) == 0:
                raise Exception(
                    f"Failed to find transcript for {title if title else url}"
                )

            # Format transcript
            transcript_text = "\n".join([document.page_content for document in transcript])

            # Add title and author if available
            if title and author:
                transcript_text = f"{title}\nby {author}\n\n{transcript_text}"

            await emitter.success_update(f"Transcript for video {title} retrieved!")
            return transcript_text

        except Exception as e:
            error_message = f"Error: {str(e)}"
            await emitter.error_update(error_message)
            return error_message
```

## Pipeline Template

Pipelines create complex workflows by chaining multiple tools and functions.

### Basic Pipeline Structure

```python
"""
title: Example Pipeline
author: YourName
version: 0.1.0
license: MIT
"""

import asyncio
import time
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union, Generator, Iterator, Callable

class Pipeline:
    """
    A pipeline that creates a complex workflow.

    This pipeline [brief description of what the pipeline does].
    """

    class Valves(BaseModel):
        """Configuration parameters for the pipeline."""
        MODEL_ID: str = Field(
            default="gpt-4",
            description="The model to use for this pipeline"
        )
        MAX_TOKENS: int = Field(
            default=2000,
            description="Maximum number of tokens to generate"
        )
        EMIT_INTERVAL: float = Field(
            default=0.5,
            description="Interval in seconds between status emissions"
        )
        ENABLE_STATUS_INDICATOR: bool = Field(
            default=True,
            description="Enable status indicator emissions"
        )
        # Add more parameters as needed

    def __init__(self):
        """Initialize the pipeline with default configuration."""
        self.valves = self.Valves()
        self.last_emit_time = 0

    def pipe(
        self,
        body: Dict[str, Any],
        messages: List[Dict[str, Any]],
        user_message: str,
        model_id: str,
    ) -> Union[str, Generator, Iterator]:
        """
        Execute the pipeline.

        Args:
            body: The request body
            messages: The conversation history
            user_message: The latest user message
            model_id: The model ID to use

        Returns:
            The pipeline result as a string or generator
        """
        # Create an event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.pipe_async(body, messages, user_message, model_id))
        loop.close()
        return result

    async def pipe_async(
        self,
        body: Dict[str, Any],
        messages: List[Dict[str, Any]],
        user_message: str,
        model_id: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """Asynchronous implementation of the pipeline."""
        try:
            # Emit start status
            await self.emit_status(
                __event_emitter__, "start", "Starting pipeline process"
            )

            # Step 1: Parse input
            await self.emit_status(
                __event_emitter__, "progress", "Parsing input", progress=10
            )
            parsed_input = await self._parse_input(user_message)

            # Step 2: Process data
            await self.emit_status(
                __event_emitter__, "progress", "Processing data", progress=40
            )
            processed_data = await self._process_data(parsed_input)

            # Step 3: Generate output
            await self.emit_status(
                __event_emitter__, "progress", "Generating output", progress=70
            )
            output = await self._generate_output(processed_data, model_id)

            # Complete
            await self.emit_status(
                __event_emitter__, "complete", "Pipeline completed", progress=100
            )

            return output
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            await self.emit_status(__event_emitter__, "error", error_msg)
            return f"Error: {error_msg}"

    async def _parse_input(self, user_message: str) -> Dict[str, Any]:
        """Parse the user message into structured data."""
        # Implementation goes here
        return {"query": user_message}

    async def _process_data(self, parsed_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process the parsed input data."""
        # Implementation goes here
        return {"processed": parsed_input["query"]}

    async def _generate_output(self, processed_data: Dict[str, Any], model_id: str) -> str:
        """Generate the final output."""
        # Implementation goes here
        return f"Result for '{processed_data['processed']}' using model {model_id}"

    async def emit_status(self, emitter, event_type: str, message: str, **kwargs):
        """Emit status updates with rate limiting."""
        if not emitter or not self.valves.ENABLE_STATUS_INDICATOR:
            return

        payload = {
            "type": f"pipeline_{event_type}",
            "data": {
                "timestamp": time.time(),
                "message": message,
                **kwargs
            }
        }

        if time.time() - self.last_emit_time >= self.valves.EMIT_INTERVAL:
            await emitter(payload)
            self.last_emit_time = time.time()
```

### Example: Mixture of Agents Pipeline

```python
"""
title: Mixture of Agents Pipeline
author: YourName
version: 0.1.0
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union, Generator, Iterator, Callable
import asyncio
import aiohttp
import random
import time
import os

class Pipeline:
    """
    A pipeline that implements the Mixture of Agents (MoA) pattern.

    This pipeline:
    1. Distributes a task to multiple LLM agents
    2. Processes responses in multiple layers
    3. Aggregates results for a higher-quality final answer
    """

    class Valves(BaseModel):
        models: List[str] = Field(
            default=["llama3", "mixtral"],
            description="List of models to use in the MoA architecture."
        )
        aggregator_model: str = Field(
            default="mixtral",
            description="Model to use for aggregation tasks."
        )
        openai_api_base: str = Field(
            default="http://localhost:11434/v1/api",
            description="Base URL for Ollama API compatible with OpenWebUI 0.5+"
        )
        num_layers: int = Field(
            default=2,
            description="Number of MoA layers."
        )
        num_agents_per_layer: int = Field(
            default=3,
            description="Number of agents to use in each layer."
        )
        emit_interval: float = Field(
            default=0.5,
            description="Interval in seconds between status emissions"
        )
        enable_status_indicator: bool = Field(
            default=True,
            description="Enable status indicator emissions"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.last_emit_time = 0
        self.active_models = []

    def pipe(
        self,
        body: Dict[str, Any],
        messages: List[Dict[str, Any]],
        user_message: str,
        model_id: str,
    ) -> str:
        """Execute the MoA pipeline."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.pipe_async(body, messages, user_message, model_id))
        loop.close()
        return result

    async def pipe_async(
        self,
        body: Dict[str, Any],
        messages: List[Dict[str, Any]],
        user_message: str,
        model_id: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """Asynchronous implementation of the MoA pipeline."""
        await self.emit_status(
            __event_emitter__, "start", "Starting Mixture of Agents process"
        )

        try:
            # Validate models
            await self.emit_status(__event_emitter__, "status", "Validating models")
            valid_models = await self.validate_models(__event_emitter__)
            if not valid_models:
                error_msg = "No valid models available"
                await self.emit_status(__event_emitter__, "error", error_msg)
                return f"Error: {error_msg}"

            # Process through layers
            layer_outputs = []
            for layer in range(self.valves.num_layers):
                await self.emit_status(
                    __event_emitter__,
                    "progress",
                    f"Layer {layer+1}/{self.valves.num_layers}",
                    progress=(layer+1)/self.valves.num_layers*100
                )

                # Select random agents for this layer
                agents = random.sample(self.valves.models, self.valves.num_agents_per_layer)

                # Process in parallel
                tasks = [self.process_agent(user_message, agent, layer, i, layer_outputs, __event_emitter__)
                        for i, agent in enumerate(agents)]

                layer_results = await asyncio.gather(*tasks)
                valid_outputs = [res for res in layer_results if not res.startswith("Error:")]

                if not valid_outputs:
                    error_msg = f"Layer {layer+1} failed: No valid responses"
                    await self.emit_status(__event_emitter__, "error", error_msg)
                    return f"Error: {error_msg}"

                layer_outputs.append(valid_outputs)

            # Final aggregation
            await self.emit_status(
                __event_emitter__,
                "progress",
                "Performing final aggregation",
                progress=90
            )

            final_prompt = self.create_final_prompt(user_message, layer_outputs)
            final_response = await self.query_model(
                self.valves.aggregator_model, final_prompt, __event_emitter__
            )

            await self.emit_status(__event_emitter__, "complete", "Process completed", progress=100)
            return final_response if not final_response.startswith("Error:") else "Error: Final aggregation failed"

        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            await self.emit_status(__event_emitter__, "error", error_msg)
            return f"Error: {error_msg}"

    async def validate_models(self, __event_emitter__) -> List[str]:
        """Validate that models are available and working."""
        try:
            valid_models = []
            for model in self.valves.models:
                test_result = await self.check_model_availability(model, __event_emitter__)
                if test_result:
                    valid_models.append(model)
            return valid_models
        except Exception as e:
            await self.emit_status(__event_emitter__, "error", f"Model validation error: {str(e)}")
            return []

    async def check_model_availability(self, model: str, __event_emitter__):
        """Check if a model is available and working."""
        test_prompt = "Respond with 'OK' if operational"
        response = await self.query_model(model, test_prompt, __event_emitter__)
        return model if response.strip() == "OK" else None

    async def process_agent(self, prompt, agent, layer, idx, layer_outputs, __event_emitter__):
        """Process a single agent in the MoA architecture."""
        await self.emit_status(
            __event_emitter__,
            "status",
            f"Agent {idx+1} in layer {layer+1} processing",
            active_models=[agent]
        )

        if layer == 0:
            # First layer processes the original prompt
            return await self.query_model(agent, prompt, __event_emitter__)
        else:
            # Subsequent layers process aggregated results from previous layer
            agg_prompt = self.create_agg_prompt(prompt, layer_outputs[-1])
            return await self.query_model(self.valves.aggregator_model, agg_prompt, __event_emitter__)

    def create_agg_prompt(self, original: str, responses: List[str]) -> str:
        """Create a prompt for aggregating responses."""
        return f"""Synthesize an improved response from these inputs:
        Original: {original}
        Previous Responses:
        {chr(10).join(f'- {r}' for r in responses)}
        Combined Answer:"""

    def create_final_prompt(self, original: str, all_responses: List[List[str]]) -> str:
        """Create the final aggregation prompt."""
        layer_responses = "\n\n".join(
            f"Layer {i+1}:\n" + "\n".join(f'- {r}' for r in layer)
            for i, layer in enumerate(all_responses)
        )
        return f"""Integrate insights from all layers into a final answer:
        Original: {original}
        {layer_responses}
        Final Comprehensive Answer:"""

    async def query_model(self, model: str, prompt: str, __event_emitter__) -> str:
        """Query an LLM model with the given prompt."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENWEBUI_TOKEN', '')}"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.valves.openai_api_base}/chat/completions",
                    headers=headers,
                    json=data
                ) as resp:
                    if resp.status != 200:
                        error = f"API Error {resp.status}: {await resp.text()}"
                        await self.emit_status(__event_emitter__, "error", error)
                        return f"Error: {error}"

                    result = await resp.json()
                    return result["choices"][0]["message"]["content"]

        except Exception as e:
            error = f"Network error: {str(e)}"
            await self.emit_status(__event_emitter__, "error", error)
            return f"Error: {error}"

    async def emit_status(self, emitter, event_type: str, message: str, **kwargs):
        """Emit status updates with rate limiting."""
        if not emitter or not self.valves.enable_status_indicator:
            return

        payload = {
            "type": f"moa_{event_type}",
            "data": {
                "timestamp": time.time(),
                "message": message,
                **kwargs
            }
        }

        if time.time() - self.last_emit_time >= self.valves.emit_interval:
            await emitter(payload)
            self.last_emit_time = time.time()
```

## Action Template

Actions create interactive UI elements within chat messages.

### Basic Action Structure

```python
"""
title: Example Action
author: YourName
version: 0.1.0
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Callable, Awaitable

class Action:
    """
    An action that creates interactive UI elements in chat messages.

    This action [brief description of what the action does].
    """

    class Valves(BaseModel):
        """Configuration parameters for the action."""
        BUTTON_TEXT: str = Field(
            default="Click Me",
            description="Text to display on the button"
        )
        CONFIRMATION_REQUIRED: bool = Field(
            default=False,
            description="Whether to require confirmation before executing"
        )
        # Add more parameters as needed

    def __init__(self):
        """Initialize the action with default configuration."""
        self.valves = self.Valves()
        self.last_emit_time = 0

    async def action(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the action.

        Args:
            body: The request body
            __user__: User information (automatically injected)
            __event_emitter__: Event emitter for sending events (automatically injected)
            __event_call__: Function to call interactive events (automatically injected)

        Returns:
            The action result
        """
        if not __event_call__:
            return {"error": "Event call function not available"}

        try:
            # Show confirmation if required
            if self.valves.CONFIRMATION_REQUIRED:
                confirmed = await self._show_confirmation(
                    __event_call__,
                    "Are you sure you want to perform this action?"
                )
                if not confirmed:
                    return {"status": "cancelled", "message": "Action cancelled by user"}

            # Execute the main action logic
            result = await self._execute_action(body, __user__, __event_call__)

            # Show success notification
            await __event_call__({
                "type": "notification",
                "data": {
                    "type": "success",
                    "message": "Action completed successfully"
                }
            })

            return {"status": "success", "data": result}

        except Exception as e:
            # Handle errors
            error_message = f"Error: {str(e)}"

            # Show error notification
            if __event_call__:
                await __event_call__({
                    "type": "notification",
                    "data": {
                        "type": "error",
                        "message": error_message
                    }
                })

            return {"error": error_message}

    async def _show_confirmation(self, event_call, message: str) -> bool:
        """Show a confirmation dialog."""
        response = await event_call({
            "type": "confirm",
            "data": {
                "title": "Confirmation",
                "message": message,
                "confirmLabel": "Confirm",
                "cancelLabel": "Cancel"
            }
        })

        return response == True

    async def _execute_action(self, body: Dict[str, Any], user: Dict[str, Any], event_call) -> Any:
        """Execute the main action logic."""
        # Implementation goes here
        return {"message": "Action executed successfully"}
```

### Example: Memory Action Button

```python
"""
title: Add to Memories Action Button
author: YourName
version: 0.1.0
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Callable, Awaitable

class Action:
    """
    An action that adds user memories through an interactive button in chat messages.
    """

    class Valves(BaseModel):
        BUTTON_TEXT: str = Field(
            default="Add to Memories",
            description="Text to display on the memory button"
        )
        CONFIRMATION_REQUIRED: bool = Field(
            default=True,
            description="Whether to require confirmation before adding memories"
        )
        MAX_MEMORY_LENGTH: int = Field(
            default=500,
            description="Maximum character length for a memory"
        )

    def __init__(self):
        self.valves = self.Valves()

    async def action(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Dict[str, Any]:
        """
        Add a memory through an interactive UI.

        Args:
            body: The request body containing the message to potentially save as a memory
            __user__: User information (automatically injected)
            __event_emitter__: Event emitter for sending events (automatically injected)
            __event_call__: Function to call interactive events (automatically injected)

        Returns:
            Dictionary with the result of the memory addition
        """
        if not __event_call__:
            return {"error": "Event call function not available"}

        if not __user__ or "id" not in __user__:
            return {"error": "User information not available"}

        try:
            # Extract the message to save as memory
            message_to_save = body.get("message", "")

            # If no message provided, prompt the user for input
            if not message_to_save:
                input_response = await __event_call__({
                    "type": "input",
                    "data": {
                        "title": "Add to Memories",
                        "placeholder": "Enter the memory you want to save",
                        "multiline": True
                    }
                })

                if not input_response:
                    return {"status": "cancelled", "message": "Memory addition cancelled"}

                message_to_save = input_response

            # Validate memory length
            if len(message_to_save) > self.valves.MAX_MEMORY_LENGTH:
                await __event_call__({
                    "type": "notification",
                    "data": {
                        "type": "error",
                        "message": f"Memory exceeds maximum length of {self.valves.MAX_MEMORY_LENGTH} characters"
                    }
                })
                return {"error": "Memory too long"}

            # Show confirmation if required
            if self.valves.CONFIRMATION_REQUIRED:
                confirmed = await __event_call__({
                    "type": "confirm",
                    "data": {
                        "title": "Confirm Memory Addition",
                        "message": f"Add this to your memories?\n\n{message_to_save}",
                        "confirmLabel": "Add Memory",
                        "cancelLabel": "Cancel"
                    }
                })

                if not confirmed:
                    return {"status": "cancelled", "message": "Memory addition cancelled"}

            # Add the memory using the memory tool
            memory_result = await self._add_memory(message_to_save, __user__["id"])

            # Show success notification
            await __event_call__({
                "type": "notification",
                "data": {
                    "type": "success",
                    "message": "Memory added successfully"
                }
            })

            return {
                "status": "success",
                "message": "Memory added successfully",
                "memory": message_to_save
            }

        except Exception as e:
            error_message = f"Error adding memory: {str(e)}"

            # Show error notification
            if __event_call__:
                await __event_call__({
                    "type": "notification",
                    "data": {
                        "type": "error",
                        "message": error_message
                    }
                })

            return {"error": error_message}

    async def _add_memory(self, memory_text: str, user_id: str) -> Dict[str, Any]:
        """Add a memory to the user's memory store."""
        # This would typically call the memory API or database
        # Implementation depends on how memories are stored in Open-WebUI

        # Example implementation (placeholder):
        from open_webui.models.memories import Memories

        new_memory = Memories.insert_new_memory(user_id, memory_text)
        if not new_memory:
            raise Exception("Failed to add memory")

        return {"id": new_memory.id, "content": memory_text}
```

These templates provide a starting point for creating various components in Open-WebUI. Adapt them to your specific needs and requirements.
