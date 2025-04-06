# 🚚 Migration Guide: Open-WebUI 0.4 to 0.5

## Overview of Changes

Open-WebUI 0.5 introduces a simplified architecture, transitioning from separate apps to a unified router-based structure, enhancing scalability and maintainability.

## Key Updates

- **Apps ➡️ Routers:** Apps moved from `open_webui.apps` to `open_webui.routers`.
- **Unified API Endpoint:** New `chat_completion` function under `open_webui.main`.
- **Function Signatures:** Now require a `Request` object for better integration.

## Quick Migration Steps

### Update Import Statements

Old:
```python
from open_webui.apps.ollama import main as ollama
```

New:
```python
from open_webui.routers.ollama import generate_chat_completion
# OR use unified endpoint
from open_webui.main import chat_completion
```

### Function Signature Adjustment

Old:
```python
async def pipe(self, body: dict, __user__: dict) -> str:
```

New:
```python
from fastapi import Request

async def pipe(self, body: dict, __user__: dict, __request__: Request) -> str:
```

## Recommendations

- Use the unified endpoint (`chat_completion`) for simplicity.
- Refactor custom functions to include the required `Request` object.

Happy migrating to Open-WebUI 0.5!

