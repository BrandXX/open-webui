# 🪄 Open-WebUI: Filters Guide

## 🌊 What Are Filters?
Filters modify or enhance data entering (inputs) and leaving (outputs) your Large Language Models (LLMs). They act as checkpoints to optimize context, readability, and security.

---

## 📍 Basic Filter Structure

```python
from pydantic import BaseModel

class Filter:
    class Valves(BaseModel):
        pass

    def inlet(self, body: dict) -> dict:
        print(f"Inlet called: {body}")
        return body

    def stream(self, event: dict) -> dict:
        print(f"Stream event: {event}")
        return event

    def outlet(self, body: dict) -> dict:
        print(f"Outlet called: {body}")
        return body
```

---

## 🔍 Filter Function Types:

### 📥 Inlet
- **Purpose:** Pre-process user inputs for improved clarity and context.
- **Use Case:** Sanitizing messages, appending crucial context.

```python
def inlet(self, body: dict) -> dict:
    context_message = {"role": "system", "content": "You're a software troubleshooting assistant."}
    body.setdefault("messages", []).insert(0, context_message)
    return body
```

### 🔄 Stream
- **Purpose:** Modify LLM output chunks in real-time.
- **Use Case:** Real-time censorship, emoji removal.

```python
def stream(self, event: dict) -> dict:
    for choice in event.get("choices", []):
        delta = choice.get("delta", {})
        if "content" in delta:
            delta["content"] = delta["content"].replace("😊", "")
    return event
```

### 📤 Outlet
- **Purpose:** Post-process LLM outputs for refined delivery.
- **Use Case:** Formatting outputs, removing sensitive data.

```python
def outlet(self, body: dict) -> dict:
    for message in body["messages"]:
        message["content"] = message["content"].replace("<API_KEY>", "[REDACTED]")
    return body
```

---

## 🚀 Benefits of Using Filters:
- Enhance data clarity and contextual accuracy.
- Real-time data interception and modification.
- Maintain security and compliance standards.

---

## 🌟 Start Experimenting:
Get hands-on with Filters to streamline your Open-WebUI interactions and enhance your AI workflows today!

