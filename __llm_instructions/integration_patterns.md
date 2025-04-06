# Integration Patterns and Best Practices for Open-WebUI

This document outlines common integration patterns and best practices for developing Functions, Tools, and Pipelines in Open-WebUI.

## Common Integration Patterns

### 1. Mixture of Agents (MoA) Pattern

A sophisticated pattern where multiple LLM agents work together in layers to produce better results.

#### Pattern:

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Callable, Awaitable
import asyncio
import random
import time

class MixtureOfAgentsPipeline:
    """Pipeline implementing the Mixture of Agents pattern."""

    class Valves(BaseModel):
        MODELS: List[str] = Field(
            default=["gpt-4", "claude-3", "llama-3"],
            description="Models to use in the mixture"
        )
        AGGREGATOR_MODEL: str = Field(
            default="gpt-4",
            description="Model to use for aggregation"
        )
        NUM_LAYERS: int = Field(
            default=2,
            description="Number of layers in the MoA architecture"
        )
        AGENTS_PER_LAYER: int = Field(
            default=3,
            description="Number of agents per layer"
        )

    def __init__(self):
        self.valves = self.Valves()

    async def pipe_async(
        self,
        query: str,
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]] = None
    ) -> str:
        """Process a query through multiple layers of agents."""
        # Track progress
        await self._emit_status(__event_emitter__, "Starting Mixture of Agents process")

        # Process through layers
        layer_outputs = []
        for layer in range(self.valves.NUM_LAYERS):
            await self._emit_status(
                __event_emitter__,
                f"Processing layer {layer+1}/{self.valves.NUM_LAYERS}"
            )

            # Select random agents for this layer
            agents = random.sample(
                self.valves.MODELS,
                min(self.valves.AGENTS_PER_LAYER, len(self.valves.MODELS))
            )

            # Process in parallel
            tasks = []
            for i, agent in enumerate(agents):
                if layer == 0:
                    # First layer processes the original query
                    tasks.append(self._query_agent(agent, query, __event_emitter__))
                else:
                    # Subsequent layers process aggregated results from previous layer
                    agg_prompt = self._create_aggregation_prompt(query, layer_outputs[-1])
                    tasks.append(self._query_agent(agent, agg_prompt, __event_emitter__))

            # Gather results
            layer_results = await asyncio.gather(*tasks)
            layer_outputs.append(layer_results)

        # Final aggregation
        await self._emit_status(__event_emitter__, "Performing final aggregation")
        final_prompt = self._create_final_prompt(query, layer_outputs)
        final_response = await self._query_agent(
            self.valves.AGGREGATOR_MODEL,
            final_prompt,
            __event_emitter__
        )

        await self._emit_status(__event_emitter__, "Process complete", done=True)
        return final_response

    async def _query_agent(self, model: str, prompt: str, __event_emitter__) -> str:
        """Query a specific agent (model) with the given prompt."""
        await self._emit_status(
            __event_emitter__,
            f"Querying {model}",
            metadata={"active_model": model}
        )

        # Implementation would call the actual model API
        # This is a simplified example
        await asyncio.sleep(1)  # Simulate API call
        return f"Response from {model} to: {prompt[:50]}..."

    def _create_aggregation_prompt(self, original_query: str, responses: List[str]) -> str:
        """Create a prompt for aggregating responses from a layer."""
        return f"""Synthesize an improved response from these inputs:
        Original query: {original_query}
        Responses:
        {chr(10).join(f'- {r}' for r in responses)}

        Synthesized response:"""

    def _create_final_prompt(self, original_query: str, all_layers: List[List[str]]) -> str:
        """Create the final aggregation prompt using all layer outputs."""
        layers_text = ""
        for i, layer in enumerate(all_layers):
            layers_text += f"Layer {i+1}:\n"
            layers_text += "\n".join(f"- {response}" for response in layer)
            layers_text += "\n\n"

        return f"""Create a comprehensive final answer based on these inputs:
        Original query: {original_query}

        {layers_text}

        Final comprehensive answer:"""

    async def _emit_status(self, __event_emitter__, message: str, done: bool = False, metadata: Dict[str, Any] = None):
        """Emit status updates."""
        if __event_emitter__:
            data = {
                "status": "complete" if done else "in_progress",
                "description": message,
                "done": done
            }
            if metadata:
                data.update(metadata)

            await __event_emitter__({
                "type": "status",
                "data": data
            })
```

#### Best Practices:

1. **Model Diversity**: Use a diverse set of models with different strengths
2. **Parallel Processing**: Process agents in parallel for efficiency
3. **Structured Aggregation**: Use clear prompts for aggregating responses
4. **Progress Tracking**: Provide detailed status updates throughout the process
5. **Error Handling**: Implement fallbacks if specific models fail
6. **Configurable Architecture**: Allow customization of layers and agents

### 2. External API Integration

Connecting Open-WebUI to external services and APIs.

#### Pattern:

```python
import aiohttp
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class ExternalAPITool:
    """Tool that integrates with an external API."""

    class Config(BaseModel):
        API_KEY: str = Field(default="", description="API key for the service")
        API_URL: str = Field(default="https://api.example.com", description="API endpoint")
        TIMEOUT: int = Field(default=30, description="Request timeout in seconds")

    def __init__(self):
        self.config = self.Config()

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a request to the external API."""
        headers = {
            "Authorization": f"Bearer {self.config.API_KEY}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.config.API_URL,
                    json=parameters,
                    headers=headers,
                    timeout=self.config.TIMEOUT
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "status": "error",
                            "code": response.status,
                            "message": f"API returned error: {error_text}"
                        }

                    result = await response.json()
                    return {
                        "status": "success",
                        "data": result
                    }
            except aiohttp.ClientError as e:
                return {
                    "status": "error",
                    "message": f"Request failed: {str(e)}"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Unexpected error: {str(e)}"
                }
```

#### Best Practices:

1. **Use async/await**: For network operations to prevent blocking
2. **Implement timeouts**: Prevent hanging on slow responses
3. **Handle errors gracefully**: Catch and process different types of errors
4. **Validate responses**: Ensure the API returned valid data
5. **Implement rate limiting**: Respect API rate limits
6. **Cache responses**: When appropriate to reduce API calls

### 2. Data Processing Pipeline

Processing and transforming data through multiple stages.

#### Pattern:

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union

class DataProcessingPipeline:
    """Pipeline for processing and transforming data through multiple stages."""

    class Valves(BaseModel):
        INPUT_FORMAT: str = Field(default="json", description="Format of input data")
        OUTPUT_FORMAT: str = Field(default="json", description="Format of output data")
        PROCESSING_STEPS: List[str] = Field(
            default=["validate", "transform", "enrich", "format"],
            description="Steps to perform in the pipeline"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the input data."""
        # Implementation details
        return {"status": "valid", "data": data}

    def _transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the data into a standardized format."""
        # Implementation details
        return {"transformed": True, "data": data}

    def _enrich_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich the data with additional information."""
        # Implementation details
        return {"enriched": True, "data": data}

    def _format_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format the output according to the specified format."""
        # Implementation details
        return {"formatted": True, "data": data}

    def pipe(
        self,
        body: Dict[str, Any],
        messages: List[Dict[str, Any]],
        user_message: str,
        model_id: str,
    ) -> Dict[str, Any]:
        """Process data through the pipeline."""
        data = body.get("data", {})
        result = {"original": data}

        # Execute each step in the pipeline
        for step in self.valves.PROCESSING_STEPS:
            if step == "validate":
                validation_result = self._validate_data(data)
                if validation_result.get("status") != "valid":
                    return {"error": "Validation failed", "details": validation_result}
                result["validation"] = validation_result

            elif step == "transform":
                transformation_result = self._transform_data(data)
                data = transformation_result.get("data", data)
                result["transformation"] = transformation_result

            elif step == "enrich":
                enrichment_result = self._enrich_data(data)
                data = enrichment_result.get("data", data)
                result["enrichment"] = enrichment_result

            elif step == "format":
                formatting_result = self._format_output(data)
                data = formatting_result.get("data", data)
                result["formatting"] = formatting_result

        result["final_data"] = data
        return result
```

#### Best Practices:

1. **Modular design**: Break processing into discrete steps
2. **Configurable pipeline**: Allow steps to be enabled/disabled
3. **Data validation**: Validate data at each stage
4. **Error handling**: Handle errors at each processing step
5. **Logging**: Log progress and results for debugging
6. **Idempotency**: Ensure operations can be safely repeated

### 3. Event Emission Pattern

Implementing standardized event emission for progress tracking and user feedback.

#### Pattern:

```python
from typing import Dict, Any, Optional, Callable, Awaitable
import time

class EventEmitter:
    """Helper class for standardized event emission."""

    def __init__(self, event_emitter: Callable[[Dict[str, Any]], Awaitable[None]] = None):
        self.event_emitter = event_emitter
        self.last_emit_time = 0
        self.emit_interval = 0.5  # Seconds between emissions

    async def progress_update(self, description: str, progress: int = None):
        """Emit a progress update."""
        await self.emit(description, "in_progress", False, progress=progress)

    async def error_update(self, description: str):
        """Emit an error update."""
        await self.emit(description, "error", True)

    async def success_update(self, description: str):
        """Emit a success update."""
        await self.emit(description, "success", True)

    async def emit(self, description: str, status: str = "in_progress", done: bool = False, **kwargs):
        """Emit a status update with rate limiting."""
        if not self.event_emitter:
            return

        # Rate limiting to prevent too many events
        current_time = time.time()
        if current_time - self.last_emit_time < self.emit_interval and not done:
            return

        self.last_emit_time = current_time

        # Prepare the event data
        data = {
            "status": status,
            "description": description,
            "done": done,
            "timestamp": current_time
        }

        # Add any additional data
        if kwargs:
            data.update(kwargs)

        # Emit the event
        await self.event_emitter({
            "type": "status",
            "data": data
        })

class ToolWithEvents:
    """Example tool implementing the event emission pattern."""

    async def execute_tool(
        self,
        parameters: Dict[str, Any],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]] = None
    ) -> Dict[str, Any]:
        """Execute the tool with progress tracking."""
        # Create the event emitter helper
        emitter = EventEmitter(__event_emitter__)

        try:
            # Step 1: Validate parameters
            await emitter.progress_update("Validating parameters")
            if not self._validate_parameters(parameters):
                await emitter.error_update("Invalid parameters provided")
                return {"error": "Invalid parameters"}

            # Step 2: Initialize processing
            await emitter.progress_update("Initializing processing", progress=10)
            # Implementation details...

            # Step 3: Process data
            await emitter.progress_update("Processing data", progress=30)
            # Implementation details...

            # Step 4: Finalize results
            await emitter.progress_update("Finalizing results", progress=80)
            # Implementation details...

            # Success
            await emitter.success_update("Processing completed successfully")
            return {"status": "success", "data": "Result data"}

        except Exception as e:
            # Handle errors
            error_message = f"Error: {str(e)}"
            await emitter.error_update(error_message)
            return {"error": error_message}

    def _validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate the input parameters."""
        # Implementation details...
        return True
```

#### Best Practices:

1. **Standardized Events**: Use consistent event types and data structures
2. **Rate Limiting**: Prevent overwhelming the UI with too many events
3. **Progress Tracking**: Include numeric progress indicators when possible
4. **Descriptive Messages**: Provide clear, user-friendly status messages
5. **Error Handling**: Emit appropriate error events with helpful messages
6. **Completion Events**: Always emit a final event when processing is complete

### 4. LLM-Powered Tool

Using an LLM to enhance tool capabilities.

#### Pattern:

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import json

class LLMPoweredTool:
    """Tool that uses an LLM to enhance its capabilities."""

    class Config(BaseModel):
        MODEL_ID: str = Field(default="gpt-4", description="LLM model to use")
        SYSTEM_PROMPT: str = Field(
            default="You are a helpful assistant that processes and analyzes data.",
            description="System prompt for the LLM"
        )
        MAX_TOKENS: int = Field(default=1000, description="Maximum tokens for LLM response")

    def __init__(self):
        self.config = self.Config()

    async def _call_llm(self, messages: List[Dict[str, Any]]) -> str:
        """Call the LLM with the given messages."""
        # This would typically use the Open-WebUI API to call the LLM
        # Implementation details would depend on how Open-WebUI exposes this functionality

        # Placeholder implementation
        return "LLM response would be here"

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool using LLM capabilities."""
        input_data = parameters.get("input", "")
        task = parameters.get("task", "analyze")

        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": self.config.SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {task}\nInput data: {input_data}"}
        ]

        # Call the LLM
        llm_response = await self._call_llm(messages)

        # Process the LLM response
        try:
            # Attempt to parse as JSON if the response looks like JSON
            if llm_response.strip().startswith("{") and llm_response.strip().endswith("}"):
                result = json.loads(llm_response)
                return {
                    "status": "success",
                    "format": "json",
                    "data": result
                }
            else:
                return {
                    "status": "success",
                    "format": "text",
                    "data": llm_response
                }
        except json.JSONDecodeError:
            return {
                "status": "success",
                "format": "text",
                "data": llm_response
            }

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for this tool."""
        return {
            "name": "llm_processor",
            "description": "Process and analyze data using an LLM",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The input data to process"
                    },
                    "task": {
                        "type": "string",
                        "description": "The task to perform (analyze, summarize, extract, etc.)"
                    }
                },
                "required": ["input"]
            }
        }
```

#### Best Practices:

1. **Clear instructions**: Provide clear instructions to the LLM
2. **Structured output**: Request structured output when possible
3. **Error handling**: Handle unexpected LLM responses
4. **Prompt engineering**: Craft effective prompts for the LLM
5. **Context management**: Manage context length for efficient token usage
6. **Fallback mechanisms**: Implement fallbacks for when the LLM fails

### 4. Memory Enhancement Pattern

Implementing tools and actions for managing and enhancing user memories.

#### Pattern:

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Callable, Awaitable
import time
import uuid

class MemoryEnhancementTool:
    """Tool for enhancing and managing user memories."""

    class Valves(BaseModel):
        MAX_MEMORIES: int = Field(
            default=1000,
            description="Maximum number of memories to store per user"
        )
        MAX_MEMORY_LENGTH: int = Field(
            default=500,
            description="Maximum character length for a memory"
        )
        ENABLE_SEARCH: bool = Field(
            default=True,
            description="Enable memory search functionality"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.memory_store = {}  # In a real implementation, this would be a database

    async def execute_tool(
        self,
        parameters: Dict[str, Any],
        __user__: Dict[str, Any] = None,
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]] = None
    ) -> Dict[str, Any]:
        """Execute the memory enhancement tool."""
        # Create event emitter for progress updates
        emitter = self._create_emitter(__event_emitter__)

        # Validate user information
        if not __user__ or "id" not in __user__:
            await emitter.error_update("User information not available")
            return {"error": "User information not available"}

        user_id = __user__["id"]
        action = parameters.get("action", "list")

        try:
            # Handle different memory actions
            if action == "add":
                return await self._add_memory(parameters, user_id, emitter)
            elif action == "list":
                return await self._list_memories(parameters, user_id, emitter)
            elif action == "search":
                return await self._search_memories(parameters, user_id, emitter)
            elif action == "delete":
                return await self._delete_memory(parameters, user_id, emitter)
            else:
                await emitter.error_update(f"Unknown action: {action}")
                return {"error": f"Unknown action: {action}"}
        except Exception as e:
            await emitter.error_update(f"Error: {str(e)}")
            return {"error": str(e)}

    async def _add_memory(self, parameters: Dict[str, Any], user_id: str, emitter) -> Dict[str, Any]:
        """Add a new memory for the user."""
        await emitter.progress_update("Adding new memory")

        # Get memory content
        content = parameters.get("content")
        if not content:
            await emitter.error_update("Memory content is required")
            return {"error": "Memory content is required"}

        # Validate memory length
        if len(content) > self.valves.MAX_MEMORY_LENGTH:
            await emitter.error_update(f"Memory exceeds maximum length of {self.valves.MAX_MEMORY_LENGTH} characters")
            return {"error": "Memory too long"}

        # Initialize user's memory store if needed
        if user_id not in self.memory_store:
            self.memory_store[user_id] = []

        # Check if user has reached the memory limit
        if len(self.memory_store[user_id]) >= self.valves.MAX_MEMORIES:
            await emitter.error_update(f"Maximum number of memories ({self.valves.MAX_MEMORIES}) reached")
            return {"error": "Memory limit reached"}

        # Create and store the memory
        memory_id = str(uuid.uuid4())
        timestamp = time.time()

        memory = {
            "id": memory_id,
            "content": content,
            "created_at": timestamp,
            "tags": parameters.get("tags", [])
        }

        self.memory_store[user_id].append(memory)

        await emitter.success_update("Memory added successfully")
        return {"status": "success", "memory": memory}

    async def _list_memories(self, parameters: Dict[str, Any], user_id: str, emitter) -> Dict[str, Any]:
        """List memories for the user."""
        await emitter.progress_update("Retrieving memories")

        # Get pagination parameters
        limit = min(parameters.get("limit", 10), 100)  # Cap at 100
        offset = max(parameters.get("offset", 0), 0)  # Ensure non-negative

        # Get memories for the user
        memories = self.memory_store.get(user_id, [])

        # Apply pagination
        paginated_memories = memories[offset:offset+limit]

        await emitter.success_update(f"Retrieved {len(paginated_memories)} memories")
        return {
            "status": "success",
            "memories": paginated_memories,
            "total": len(memories),
            "limit": limit,
            "offset": offset
        }

    async def _search_memories(self, parameters: Dict[str, Any], user_id: str, emitter) -> Dict[str, Any]:
        """Search user memories."""
        if not self.valves.ENABLE_SEARCH:
            await emitter.error_update("Search functionality is disabled")
            return {"error": "Search functionality is disabled"}

        await emitter.progress_update("Searching memories")

        # Get search parameters
        query = parameters.get("query")
        if not query:
            await emitter.error_update("Search query is required")
            return {"error": "Search query is required"}

        # Get memories for the user
        memories = self.memory_store.get(user_id, [])

        # Perform simple text search (in a real implementation, this would use vector search)
        results = []
        for memory in memories:
            if query.lower() in memory["content"].lower():
                results.append(memory)

        await emitter.success_update(f"Found {len(results)} matching memories")
        return {"status": "success", "results": results}

    async def _delete_memory(self, parameters: Dict[str, Any], user_id: str, emitter) -> Dict[str, Any]:
        """Delete a memory."""
        await emitter.progress_update("Deleting memory")

        # Get memory ID
        memory_id = parameters.get("memory_id")
        if not memory_id:
            await emitter.error_update("Memory ID is required")
            return {"error": "Memory ID is required"}

        # Get memories for the user
        memories = self.memory_store.get(user_id, [])

        # Find and remove the memory
        for i, memory in enumerate(memories):
            if memory["id"] == memory_id:
                del memories[i]
                await emitter.success_update("Memory deleted successfully")
                return {"status": "success", "message": "Memory deleted successfully"}

        await emitter.error_update("Memory not found")
        return {"error": "Memory not found"}

    def _create_emitter(self, __event_emitter__):
        """Create an event emitter helper."""
        class Emitter:
            def __init__(self, event_emitter):
                self.event_emitter = event_emitter

            async def progress_update(self, description):
                await self.emit(description)

            async def error_update(self, description):
                await self.emit(description, "error", True)

            async def success_update(self, description):
                await self.emit(description, "success", True)

            async def emit(self, description, status="in_progress", done=False):
                if self.event_emitter:
                    await self.event_emitter({
                        "type": "status",
                        "data": {
                            "status": status,
                            "description": description,
                            "done": done
                        }
                    })

        return Emitter(__event_emitter__)
```

#### Best Practices:

1. **User Scoping**: Always scope memories to specific users
2. **Validation**: Validate memory content and length
3. **Pagination**: Implement pagination for listing memories
4. **Search Capabilities**: Provide efficient search functionality
5. **Error Handling**: Handle edge cases like missing memories
6. **Progress Updates**: Provide clear status updates during operations

### 5. Interactive UI Component

Creating interactive UI elements in chat messages.

#### Pattern:

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class InteractiveUIComponent:
    """Action that creates interactive UI elements in chat messages."""

    class Config(BaseModel):
        BUTTON_LABEL: str = Field(default="Interact", description="Label for the button")
        CONFIRMATION_REQUIRED: bool = Field(
            default=False,
            description="Whether to require confirmation before executing"
        )

    def __init__(self):
        self.config = self.Config()

    async def _show_form(self, event_call, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Show a form to collect user input."""
        fields = form_data.get("fields", [])
        form_fields = []

        for field in fields:
            form_fields.append({
                "name": field["name"],
                "label": field.get("label", field["name"]),
                "type": field.get("type", "text"),
                "placeholder": field.get("placeholder", ""),
                "required": field.get("required", False),
                "options": field.get("options", [])
            })

        response = await event_call({
            "type": "form",
            "data": {
                "title": form_data.get("title", "Input Required"),
                "fields": form_fields,
                "submitLabel": form_data.get("submitLabel", "Submit")
            }
        })

        return response

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

    async def _show_notification(self, event_call, message: str, type: str = "info") -> None:
        """Show a notification."""
        await event_call({
            "type": "notification",
            "data": {
                "message": message,
                "type": type
            }
        })

    async def action(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Any] = None,
        __event_call__: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute the interactive UI component."""
        if not __event_call__:
            return {"error": "Event call function not available"}

        action_type = body.get("action_type", "default")

        # Show confirmation if required
        if self.config.CONFIRMATION_REQUIRED:
            confirmed = await self._show_confirmation(
                __event_call__,
                f"Are you sure you want to perform this {action_type} action?"
            )
            if not confirmed:
                return {"status": "cancelled", "message": "Action cancelled by user"}

        # Handle different action types
        if action_type == "collect_input":
            form_data = body.get("form_data", {
                "title": "Input Required",
                "fields": [
                    {
                        "name": "input",
                        "label": "Input",
                        "type": "text",
                        "placeholder": "Enter your input",
                        "required": True
                    }
                ]
            })

            user_input = await self._show_form(__event_call__, form_data)
            return {
                "status": "success",
                "action": "collect_input",
                "data": user_input
            }

        elif action_type == "display_result":
            result_data = body.get("result_data", {})

            # Show notification
            await self._show_notification(
                __event_call__,
                body.get("message", "Operation completed successfully"),
                body.get("notification_type", "success")
            )

            return {
                "status": "success",
                "action": "display_result",
                "data": result_data
            }

        else:
            return {
                "status": "error",
                "message": f"Unknown action type: {action_type}"
            }
```

#### Best Practices:

1. **User-friendly interfaces**: Design intuitive and accessible UI components
2. **Progressive disclosure**: Reveal complexity progressively
3. **Validation feedback**: Provide clear feedback for validation errors
4. **Responsive design**: Ensure components work on different devices
5. **Consistent styling**: Follow Open-WebUI design patterns
6. **Accessibility**: Ensure components are accessible to all users

## General Best Practices

### Code Organization

1. **Modular Design**: Break functionality into small, focused modules
2. **Separation of Concerns**: Separate configuration, business logic, and I/O
3. **Consistent Naming**: Use clear, consistent naming conventions
4. **Documentation**: Document code thoroughly with docstrings and comments

### Error Handling

1. **Graceful Degradation**: Fail gracefully when errors occur
2. **Detailed Error Messages**: Provide helpful error messages
3. **Error Classification**: Distinguish between different types of errors
4. **Retry Mechanisms**: Implement retries for transient failures
5. **Fallback Strategies**: Have fallback strategies for critical functionality

### Performance

1. **Asynchronous Operations**: Use async/await for I/O-bound operations
2. **Caching**: Cache expensive operations and responses
3. **Pagination**: Implement pagination for large datasets
4. **Resource Management**: Properly manage resources like connections and file handles
5. **Monitoring**: Add instrumentation for performance monitoring

### Security

1. **Input Validation**: Validate all inputs thoroughly
2. **Output Sanitization**: Sanitize outputs to prevent injection attacks
3. **Secure Credentials**: Never hardcode credentials
4. **Principle of Least Privilege**: Request only necessary permissions
5. **Rate Limiting**: Implement rate limiting to prevent abuse

### Testing

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test interactions between components
3. **Edge Cases**: Test boundary conditions and edge cases
4. **Error Scenarios**: Test error handling and recovery
5. **Mocking**: Use mocks for external dependencies

## Component-Specific Best Practices

### Functions

1. **Clear Purpose**: Each function should have a single, clear purpose
2. **Configurable**: Make functions configurable through the Config class
3. **Stateless**: Design functions to be stateless when possible
4. **Idempotent**: Ensure functions can be safely called multiple times
5. **Versioning**: Support versioning for backward compatibility

### Tools

1. **Clear Documentation**: Document tool purpose, inputs, and outputs
2. **Schema Definition**: Define clear JSON schemas for tool parameters
3. **Error Handling**: Handle and report errors in a standardized way
4. **Rate Limiting**: Respect rate limits for external services
5. **Caching**: Implement caching for expensive operations

### Pipelines

1. **Step Isolation**: Design pipeline steps to be isolated and reusable
2. **State Management**: Manage state carefully between pipeline steps
3. **Error Recovery**: Implement error recovery mechanisms
4. **Progress Tracking**: Track and report pipeline progress
5. **Configurability**: Make pipelines configurable through valves

### Actions

1. **User Experience**: Focus on creating a smooth user experience
2. **Feedback**: Provide clear feedback for user actions
3. **Confirmation**: Confirm potentially destructive actions
4. **Progressive Disclosure**: Reveal complexity progressively
5. **Accessibility**: Ensure actions are accessible to all users

## Migration and Compatibility

1. **Backward Compatibility**: Maintain backward compatibility when possible
2. **Deprecation Notices**: Provide clear deprecation notices
3. **Migration Guides**: Create migration guides for breaking changes
4. **Version Support**: Support multiple versions during transition periods
5. **Feature Flags**: Use feature flags for gradual rollouts

By following these integration patterns and best practices, you can create high-quality, maintainable, and effective components for Open-WebUI.
