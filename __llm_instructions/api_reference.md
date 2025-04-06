# Open-WebUI API Reference

This document provides a comprehensive reference for the Open-WebUI API endpoints and interfaces that are relevant for developing Functions, Tools, and Pipelines.

## Core API Endpoints

### Chat Completion

The primary endpoint for generating chat completions.

**Endpoint:** `/api/chat/completions`

**Method:** POST

**Request Body:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "model": "gpt-3.5-turbo",
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 1000,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-3.5-turbo",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "I'm doing well, thank you for asking! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```

### Tool Registration

Register a new tool with the system.

**Endpoint:** `/api/tools/register`

**Method:** POST

**Request Body:**
```json
{
  "name": "get_weather",
  "description": "Get the current weather in a given location",
  "schema": {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          }
        },
        "required": ["location"]
      }
    }
  },
  "implementation": {
    "type": "python",
    "code": "async def execute(parameters):\n    location = parameters.get('location')\n    # Implementation details\n    return {'temperature': 72, 'condition': 'sunny'}"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Tool registered successfully",
  "tool_id": "tool_123456"
}
```

### Function Registration

Register a new function with the system.

**Endpoint:** `/api/functions/register`

**Method:** POST

**Request Body:**
```json
{
  "name": "custom_model_integration",
  "description": "Integrates a custom model provider",
  "config_schema": {
    "type": "object",
    "properties": {
      "API_KEY": {
        "type": "string",
        "description": "API key for the model provider"
      },
      "API_URL": {
        "type": "string",
        "description": "API endpoint URL"
      }
    },
    "required": ["API_KEY"]
  },
  "implementation": {
    "type": "python",
    "code": "async def execute(body, __user__, __request__):\n    # Implementation details\n    return {'status': 'success'}"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Function registered successfully",
  "function_id": "function_123456"
}
```

### Pipeline Registration

Register a new pipeline with the system.

**Endpoint:** `/api/pipelines/register`

**Method:** POST

**Request Body:**
```json
{
  "name": "research_pipeline",
  "description": "Performs comprehensive research on a topic",
  "valves_schema": {
    "type": "object",
    "properties": {
      "MODEL_ID": {
        "type": "string",
        "description": "The model to use for this pipeline"
      },
      "MAX_RESULTS": {
        "type": "integer",
        "description": "Maximum number of search results to process"
      }
    }
  },
  "implementation": {
    "type": "python",
    "code": "def pipe(body, messages, user_message, model_id):\n    # Implementation details\n    return 'Research results'"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Pipeline registered successfully",
  "pipeline_id": "pipeline_123456"
}
```

## WebSocket Events

Open-WebUI uses WebSocket connections for real-time communication. Here are the relevant events for component development:

### Chat Message Event

Sent when a new chat message is received.

```json
{
  "type": "chat_message",
  "data": {
    "message_id": "msg_123456",
    "conversation_id": "conv_123456",
    "role": "user",
    "content": "Hello, how are you?",
    "timestamp": 1677652288
  }
}
```

### Tool Call Event

Sent when a tool is called by the LLM.

```json
{
  "type": "tool_call",
  "data": {
    "message_id": "msg_123456",
    "conversation_id": "conv_123456",
    "tool_name": "get_weather",
    "parameters": {
      "location": "San Francisco, CA"
    },
    "timestamp": 1677652288
  }
}
```

### Tool Response Event

Sent when a tool returns a response.

```json
{
  "type": "tool_response",
  "data": {
    "message_id": "msg_123456",
    "conversation_id": "conv_123456",
    "tool_name": "get_weather",
    "result": {
      "temperature": 72,
      "condition": "sunny"
    },
    "timestamp": 1677652288
  }
}
```

## Component Interfaces

### Tool Interface

Tools must implement the following interface:

```python
class Tool:
    def __init__(self):
        # Initialize the tool
        pass
    
    async def execute(self, parameters: dict) -> dict:
        # Execute the tool with the given parameters
        pass
    
    def get_schema(self) -> dict:
        # Return the JSON schema for this tool
        pass
```

### Function Interface

Functions must implement the following interface:

```python
class Function:
    def __init__(self):
        # Initialize the function
        pass
    
    async def execute(self, body: dict, __user__: dict = None, __request__ = None) -> dict:
        # Execute the function with the given parameters
        pass
```

### Pipeline Interface

Pipelines must implement the following interface:

```python
class Pipeline:
    def __init__(self):
        # Initialize the pipeline
        self.valves = self.Valves()
    
    def pipe(self, body: dict, messages: list, user_message: str, model_id: str) -> str:
        # Execute the pipeline
        pass
```

### Action Interface

Actions must implement the following interface:

```python
class Action:
    def __init__(self):
        # Initialize the action
        pass
    
    async def action(self, body: dict, __user__: dict = None, __event_emitter__ = None, __event_call__ = None) -> dict:
        # Execute the action
        pass
```

## Authentication and Authorization

Components can access user information through the `__user__` parameter, which contains:

```json
{
  "id": "user_123456",
  "username": "johndoe",
  "email": "john@example.com",
  "roles": ["user", "admin"],
  "permissions": ["read", "write", "execute"]
}
```

Use this information to implement proper authorization in your components.

## Error Handling

Components should handle errors gracefully and return appropriate error responses:

```json
{
  "status": "error",
  "message": "An error occurred",
  "details": "Detailed error information"
}
```

## Rate Limiting

Components should respect rate limits for external APIs and implement appropriate backoff strategies.

## Best Practices

1. **Validation**: Always validate input parameters before processing
2. **Error Handling**: Implement comprehensive error handling
3. **Async Operations**: Use async/await for I/O-bound operations
4. **Documentation**: Document your component thoroughly
5. **Security**: Never expose sensitive information in responses
6. **Testing**: Test your component with various inputs and edge cases

This API reference provides the essential information needed to develop components for Open-WebUI. Refer to the specific documentation for each component type for more detailed information.
