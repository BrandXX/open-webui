# Component Templates for Open-WebUI

This document provides templates and examples for creating Functions, Tools, and Pipelines in Open-WebUI.

## Function Template

Functions extend Open-WebUI itself by adding new capabilities to the platform.

### Basic Function Structure

```python
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

### Example: Custom Model Integration Function

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union
import aiohttp

class AnthropicIntegration:
    """
    A function that integrates Anthropic Claude models into Open-WebUI.
    """
    
    class Config(BaseModel):
        ANTHROPIC_API_KEY: str = Field(
            default="",
            description="API key for Anthropic Claude"
        )
        ANTHROPIC_API_URL: str = Field(
            default="https://api.anthropic.com/v1/messages",
            description="Anthropic API endpoint URL"
        )
        DEFAULT_MODEL: str = Field(
            default="claude-3-opus-20240229",
            description="Default Anthropic model to use"
        )

    def __init__(self):
        self.config = self.Config()
    
    async def execute(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Send a request to Anthropic Claude API and return the response.
        
        Args:
            body: The request body containing the messages
            
        Returns:
            The response from Anthropic API
        """
        if not self.config.ANTHROPIC_API_KEY:
            return {"error": "Anthropic API key not configured"}
        
        messages = body.get("messages", [])
        model = body.get("model", self.config.DEFAULT_MODEL)
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": body.get("max_tokens", 1024)
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.ANTHROPIC_API_URL,
                json=payload,
                headers=headers
            ) as response:
                result = await response.json()
                return result
```

## Tool Template

Tools extend LLM capabilities by allowing them to access external data and services.

### Basic Tool Structure

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union, List

class Tool:
    """
    A tool that extends LLM capabilities.
    
    This tool [brief description of what the tool does].
    """
    
    class Config(BaseModel):
        """Configuration parameters for the tool."""
        API_KEY: str = Field(
            default="",
            description="API key for the external service"
        )
        ENDPOINT_URL: str = Field(
            default="https://api.example.com",
            description="API endpoint URL"
        )
        # Add more parameters as needed

    def __init__(self):
        """Initialize the tool with default configuration."""
        self.config = self.Config()
    
    async def execute(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.
        
        Args:
            parameters: Tool-specific parameters
            
        Returns:
            The tool execution result
        """
        # Implementation goes here
        result = {"status": "success", "data": "Tool executed successfully"}
        return result
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Return the JSON schema for this tool.
        
        This schema will be used by the LLM to understand how to use the tool.
        
        Returns:
            The tool's JSON schema
        """
        return {
            "name": "example_tool",
            "description": "This tool does something useful",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to process"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return"
                    }
                },
                "required": ["query"]
            }
        }
```

### Example: Weather Information Tool

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import aiohttp

class WeatherTool:
    """
    A tool that retrieves current weather information for a location.
    """
    
    class Config(BaseModel):
        WEATHER_API_KEY: str = Field(
            default="",
            description="API key for the weather service"
        )
        WEATHER_API_URL: str = Field(
            default="https://api.weatherapi.com/v1/current.json",
            description="Weather API endpoint URL"
        )

    def __init__(self):
        self.config = self.Config()
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve weather information for the specified location.
        
        Args:
            parameters: Dictionary containing:
                - location: The city or location to get weather for
                
        Returns:
            Dictionary containing weather information
        """
        location = parameters.get("location")
        if not location:
            return {"error": "Location parameter is required"}
        
        if not self.config.WEATHER_API_KEY:
            return {"error": "Weather API key not configured"}
        
        params = {
            "key": self.config.WEATHER_API_KEY,
            "q": location,
            "aqi": "no"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.config.WEATHER_API_URL,
                params=params
            ) as response:
                if response.status != 200:
                    return {"error": f"API returned status code {response.status}"}
                
                data = await response.json()
                
                # Extract relevant information
                result = {
                    "location": data["location"]["name"],
                    "country": data["location"]["country"],
                    "temperature_c": data["current"]["temp_c"],
                    "temperature_f": data["current"]["temp_f"],
                    "condition": data["current"]["condition"]["text"],
                    "humidity": data["current"]["humidity"],
                    "wind_kph": data["current"]["wind_kph"]
                }
                
                return result
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for this tool."""
        return {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city or location to get weather for"
                    }
                },
                "required": ["location"]
            }
        }
```

## Pipeline Template

Pipelines create complex workflows by chaining multiple tools and functions.

### Basic Pipeline Structure

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union, Generator, Iterator
import asyncio

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
        # Add more parameters as needed

    def __init__(self):
        """Initialize the pipeline with default configuration."""
        self.valves = self.Valves()
    
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
        # Implementation goes here
        result = f"Pipeline executed with model: {model_id}"
        return result
```

### Example: Research Pipeline

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union, Generator, Iterator
import asyncio
import aiohttp
from bs4 import BeautifulSoup

class ResearchPipeline:
    """
    A pipeline that performs comprehensive research on a topic.
    
    This pipeline:
    1. Generates search queries based on the user's question
    2. Searches for relevant information
    3. Extracts and summarizes content from top results
    4. Synthesizes a comprehensive answer
    """
    
    class Valves(BaseModel):
        SEARCH_API_KEY: str = Field(
            default="",
            description="API key for the search service"
        )
        SEARCH_API_URL: str = Field(
            default="https://api.searchservice.com/search",
            description="Search API endpoint URL"
        )
        MODEL_ID: str = Field(
            default="gpt-4",
            description="The model to use for this pipeline"
        )
        MAX_RESULTS: int = Field(
            default=5,
            description="Maximum number of search results to process"
        )

    def __init__(self):
        self.valves = self.Valves()
    
    async def _generate_search_queries(self, question: str, model_id: str) -> List[str]:
        """Generate search queries based on the user's question."""
        # Implementation would use an LLM to generate effective search queries
        return [f"best information about {question}", f"{question} explained", f"{question} latest research"]
    
    async def _search(self, query: str) -> List[Dict[str, Any]]:
        """Perform a search using the configured search API."""
        # Implementation would call a search API and return results
        return [{"title": "Result 1", "url": "https://example.com/1"}, {"title": "Result 2", "url": "https://example.com/2"}]
    
    async def _extract_content(self, url: str) -> str:
        """Extract relevant content from a URL."""
        # Implementation would fetch and parse webpage content
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                # Extract main content (simplified)
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
                return content[:1000]  # Truncate for example
    
    async def _synthesize_answer(self, question: str, contents: List[str], model_id: str) -> str:
        """Synthesize a comprehensive answer based on the extracted contents."""
        # Implementation would use an LLM to synthesize information
        combined_content = "\n\n".join(contents)
        return f"Based on my research about '{question}', I found: {combined_content[:500]}..."
    
    async def pipe_async(
        self,
        body: Dict[str, Any],
        messages: List[Dict[str, Any]],
        user_message: str,
        model_id: str,
    ) -> str:
        """Asynchronous implementation of the pipeline."""
        # 1. Generate search queries
        search_queries = await self._generate_search_queries(user_message, model_id)
        
        # 2. Perform searches
        all_results = []
        for query in search_queries:
            results = await self._search(query)
            all_results.extend(results)
        
        # Deduplicate and limit results
        unique_results = {result["url"]: result for result in all_results}
        top_results = list(unique_results.values())[:self.valves.MAX_RESULTS]
        
        # 3. Extract content from each result
        contents = []
        for result in top_results:
            content = await self._extract_content(result["url"])
            contents.append(content)
        
        # 4. Synthesize answer
        answer = await self._synthesize_answer(user_message, contents, model_id)
        
        return answer
    
    def pipe(
        self,
        body: Dict[str, Any],
        messages: List[Dict[str, Any]],
        user_message: str,
        model_id: str,
    ) -> str:
        """
        Execute the research pipeline.
        
        This method creates an event loop and runs the asynchronous implementation.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.pipe_async(body, messages, user_message, model_id))
        loop.close()
        return result
```

## Action Template

Actions create interactive UI elements within chat messages.

### Basic Action Structure

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class Action:
    """
    An action that creates interactive UI elements in chat messages.
    
    This action [brief description of what the action does].
    """
    
    class Config(BaseModel):
        """Configuration parameters for the action."""
        BUTTON_TEXT: str = Field(
            default="Click Me",
            description="Text to display on the button"
        )
        # Add more parameters as needed

    def __init__(self):
        """Initialize the action with default configuration."""
        self.config = self.Config()
    
    async def action(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Any] = None,
        __event_call__: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute the action.
        
        Args:
            body: The request body
            __user__: User information (automatically injected)
            __event_emitter__: Event emitter for sending events (automatically injected)
            __event_call__: Function to call events (automatically injected)
            
        Returns:
            The action result
        """
        # Implementation goes here
        if __event_call__:
            response = await __event_call__({
                "type": "notification",
                "data": {
                    "message": "Action executed successfully"
                }
            })
        
        return {"status": "success", "message": "Action completed"}
```

### Example: Data Visualization Action

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import json
import base64
import matplotlib.pyplot as plt
import io

class DataVisualizationAction:
    """
    An action that creates interactive data visualization buttons in chat messages.
    """
    
    class Config(BaseModel):
        BUTTON_TEXT: str = Field(
            default="Visualize Data",
            description="Text to display on the visualization button"
        )
        DEFAULT_CHART_TYPE: str = Field(
            default="bar",
            description="Default chart type (bar, line, pie, scatter)"
        )

    def __init__(self):
        self.config = self.Config()
    
    def _generate_chart(self, data: Dict[str, Any], chart_type: str) -> str:
        """Generate a chart from the provided data and return as base64 image."""
        plt.figure(figsize=(10, 6))
        
        labels = data.get("labels", [])
        values = data.get("values", [])
        
        if chart_type == "bar":
            plt.bar(labels, values)
        elif chart_type == "line":
            plt.plot(labels, values)
        elif chart_type == "pie":
            plt.pie(values, labels=labels, autopct='%1.1f%%')
        elif chart_type == "scatter":
            plt.scatter(labels, values)
        else:
            plt.bar(labels, values)  # Default to bar
        
        plt.title(data.get("title", "Data Visualization"))
        plt.xlabel(data.get("x_label", ""))
        plt.ylabel(data.get("y_label", ""))
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    async def action(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Any] = None,
        __event_call__: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Create and display a data visualization.
        
        Args:
            body: The request body containing:
                - data: Dictionary with labels, values, title, x_label, y_label
                - chart_type: Type of chart to generate (optional)
            
        Returns:
            Dictionary with the generated visualization
        """
        if not __event_call__:
            return {"error": "Event call function not available"}
        
        # First, prompt the user for data if not provided
        if "data" not in body:
            data_response = await __event_call__({
                "type": "input",
                "data": {
                    "title": "Enter Data for Visualization",
                    "placeholder": '{"labels": ["A", "B", "C"], "values": [10, 20, 30], "title": "My Chart", "x_label": "Categories", "y_label": "Values"}'
                }
            })
            
            try:
                data = json.loads(data_response)
            except:
                return {"error": "Invalid JSON data provided"}
        else:
            data = body["data"]
        
        # Then, prompt for chart type if not provided
        chart_type = body.get("chart_type")
        if not chart_type:
            chart_type_response = await __event_call__({
                "type": "select",
                "data": {
                    "title": "Select Chart Type",
                    "options": [
                        {"value": "bar", "label": "Bar Chart"},
                        {"value": "line", "label": "Line Chart"},
                        {"value": "pie", "label": "Pie Chart"},
                        {"value": "scatter", "label": "Scatter Plot"}
                    ]
                }
            })
            chart_type = chart_type_response or self.config.DEFAULT_CHART_TYPE
        
        # Generate the visualization
        image_data = self._generate_chart(data, chart_type)
        
        # Display the visualization
        await __event_call__({
            "type": "display",
            "data": {
                "type": "image",
                "src": image_data,
                "alt": f"{chart_type.capitalize()} chart of {data.get('title', 'data')}"
            }
        })
        
        return {
            "status": "success",
            "message": f"Generated {chart_type} chart",
            "image": image_data
        }
```

These templates provide a starting point for creating various components in Open-WebUI. Adapt them to your specific needs and requirements.
