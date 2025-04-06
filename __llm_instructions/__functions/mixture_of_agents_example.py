"""
title: Mixture of Agents Action
author: MaxKerkula
version: 0.5
required_open_webui_version: 0.5.2
"""

"""
MODERATION TEAM NOTE:
WE'VE HAD SEVERAL REPORTS THAT THIS FUNCTION NO LONGER WORKS ON LATER VERSIONS OF OPENWEBUI.
WE INVITE THE AUTHOR TO PLEASE UPDATE THIS FUNCTION OR IT WILL BE REMOVED.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Callable, Awaitable
import aiohttp
import random
import asyncio
import time
import os

class Action:
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

    class DeploymentPresets:
        SMALL_SCALE = Valves(
            num_layers=2,
            num_agents_per_layer=2,
            models=["llama3", "mistral"],
            aggregator_model="mixtral"
        )
        
        LARGE_SCALE = Valves(
            num_layers=3,
            num_agents_per_layer=4,
            models=["llama3-70b", "mixtral", "qwen2-72b"],
            aggregator_model="gpt-4"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.last_emit_time = 0
        self.active_models = []

    async def action(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Optional[dict]:
        await self.emit_status(
            __event_emitter__, "moa_start", "Starting Mixture of Agents process"
        )

        try:
            await self.validate_models(__event_emitter__)
        except ValueError as e:
            await self.emit_status(__event_emitter__, "moa_error", str(e))
            return {"error": str(e)}

        messages = body.get("messages", [])
        if not messages:
            error_msg = "No messages found in request"
            await self.emit_status(__event_emitter__, "moa_error", error_msg)
            return {"error": error_msg}

        last_message = messages[-1]["content"]
        moa_response = await self.moa_process(last_message, __event_emitter__)

        if moa_response.startswith("Error:"):
            await self.emit_status(__event_emitter__, "moa_error", moa_response)
            return {"error": moa_response}

        body["messages"].append({"role": "assistant", "content": moa_response})
        await self.emit_status(__event_emitter__, "moa_complete", "Process completed")
        return body

    async def validate_models(self, __event_emitter__: Callable[[dict], Awaitable[None]]):
        await self.emit_status(__event_emitter__, "moa_status", "Validating models")
        
        try:
            async with asyncio.timeout(30):
                valid_models = await asyncio.gather(*[
                    self.check_model_availability(model, __event_emitter__) 
                  for model in self.valves.models
                ])
            self.valves.models = [m for m in valid_models if m]
        except TimeoutError:
            error_msg = "Model validation timed out"
            await self.emit_status(__event_emitter__, "moa_error", error_msg)
            raise ValueError(error_msg)

        if not self.valves.models:
            error_msg = "No valid models available"
            await self.emit_status(__event_emitter__, "moa_error", error_msg)
            raise ValueError(error_msg)

    async def check_model_availability(self, model: str, __event_emitter__):
        test_prompt = "Respond with 'OK' if operational"
        response = await self.query_ollama(model, test_prompt, __event_emitter__)
        return model if response.strip() == "OK" else None

    async def moa_process(self, prompt: str, __event_emitter__) -> str:
        layer_outputs = []
        for layer in range(self.valves.num_layers):
            await self.emit_status(
                __event_emitter__,
                "moa_progress",
                f"Layer {layer+1}/{self.valves.num_layers}",
                progress=(layer+1)/self.valves.num_layers*100
            )

            agents = random.sample(self.valves.models, self.valves.num_agents_per_layer)
            tasks = [self.process_agent(prompt, agent, layer, i, layer_outputs, __event_emitter__) 
                    for i, agent in enumerate(agents)]
            
            layer_results = await asyncio.gather(*tasks)
            valid_outputs = [res for res in layer_results if not res.startswith("Error:")]
            
            if not valid_outputs:
                error_msg = f"Layer {layer+1} failed: No valid responses"
                await self.emit_status(__event_emitter__, "moa_error", error_msg)
                return f"Error: {error_msg}"
                
            layer_outputs.append(valid_outputs)

        final_prompt = self.create_final_prompt(prompt, layer_outputs)
        final_response = await self.query_ollama(
            self.valves.aggregator_model, final_prompt, __event_emitter__
        )
        
        return final_response if not final_response.startswith("Error:") else "Error: Final aggregation failed"

    async def process_agent(self, prompt, agent, layer, idx, layer_outputs, __event_emitter__):
        await self.emit_status(
            __event_emitter__,
            "moa_status",
            f"Agent {idx+1} in layer {layer+1} processing",
            active_models=[agent]
        )

        if layer == 0:
            return await self.query_ollama(agent, prompt, __event_emitter__)
        else:
            agg_prompt = self.create_agg_prompt(prompt, layer_outputs[-1])
            return await self.query_ollama(self.valves.aggregator_model, agg_prompt, __event_emitter__)

    def create_agg_prompt(self, original: str, responses: List[str]) -> str:
        return f"""Synthesize an improved response from these inputs:
        Original: {original}
        Previous Responses:
        {chr(10).join(f'- {r}' for r in responses)}
        Combined Answer:"""

    def create_final_prompt(self, original: str, all_responses: List[List[str]]) -> str:
        layer_responses = "\n\n".join(
            f"Layer {i+1}:\n" + "\n".join(f'- {r}' for r in layer) 
            for i, layer in enumerate(all_responses)
        )
        return f"""Integrate insights from all layers into a final answer:
        Original: {original}
        {layer_responses}
        Final Comprehensive Answer:"""

    async def query_ollama(self, model: str, prompt: str, __event_emitter__) -> str:
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
                        await self.emit_status(__event_emitter__, "moa_error", error)
                        return f"Error: {error}"
                        
                    result = await resp.json()
                    return result["choices"][0]["message"]["content"]
                    
        except Exception as e:
            error = f"Network error: {str(e)}"
            await self.emit_status(__event_emitter__, "moa_error", error)
            return f"Error: {error}"

    async def emit_status(self, emitter, event_type: str, message: str, **kwargs):
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

    async def on_start(self):
        print("MOA Service: Active")

    async def on_stop(self):
        print("MOA Service: Terminated")
