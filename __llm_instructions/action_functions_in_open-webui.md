# 🎯 Action Functions in Open-WebUI

## Overview

**Action Functions** allow the creation of interactive buttons directly within chat messages, enabling dynamic user interactions.

## What Actions Do

- Create interactive UI elements (buttons) in the message toolbar.
- Allow users to trigger actions such as granting permissions, generating visualizations, or downloading data.

## Example of an Action Function

### Basic Structure

```python
async def action(
    self,
    body: dict,
    __user__=None,
    __event_emitter__=None,
    __event_call__=None,
) -> dict:
    response = await __event_call__({
        "type": "input",
        "data": {
            "title": "Enter your message",
            "placeholder": "Your message here..."
        }
    })
    print(response)
```

## Practical Examples
- User prompts for additional data entry.
- Generating visual graphs or charts.
- Downloading conversation snippets as audio.

Enjoy crafting interactive and engaging experiences with Action Functions!

