# 🚰 Pipes in Open-WebUI

## Overview

**Pipes** allow you to create custom models and workflows within Open-WebUI, offering detailed control over data processing.

## Pipe Structure

A basic Pipe consists of:

- **Valves**: Configurable parameters controlling Pipe behavior.
- **Pipe Function**: Main processing logic handling inputs and outputs.

### Example Structure

```python
from pydantic import BaseModel, Field

class Pipe:
    class Valves(BaseModel):
        MODEL_ID: str = Field(default="")

    def __init__(self):
        self.valves = self.Valves()

    def pipe(self, body: dict):
        print(self.valves, body)
        return "Hello, World!"
```

## Creating Multiple Models

Use the `pipes` function to create multiple models within one Pipe:

```python
def pipes(self):
    return [
        {"id": "model_1", "name": "Model One"},
        {"id": "model_2", "name": "Model Two"}
    ]
```

## Practical Uses
- Retrieval Augmented Generation (RAG)
- Proxying requests to external APIs like OpenAI, Anthropic, Azure OpenAI, and Google.

## Extending Pipes

Customize pipes further by integrating Open-WebUI's internal functions for advanced functionality.

Enjoy crafting your custom Pipes!

