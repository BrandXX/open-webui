# 🌟 Open-WebUI Tools, Functions, and Pipelines: A Friendly Guide

Feeling a little lost with Open-WebUI's talk about "Tools", "Functions", and "Pipelines"? Don't worry! Let's simplify everything, step by step. By the end of this, you'll confidently understand what these terms mean and how to use them to boost your Open-WebUI experience!

## 🚀 Quick Summary (TL;DR)
- **Tools**: Enhance LLMs, allowing real-world interactions like retrieving live weather or stocks.
- **Functions**: Extend Open-WebUI itself, adding new AI models or improving usability (like custom buttons).
- **Pipelines**: Advanced tools turning Open-WebUI features into API-compatible workflows, typically used to offload heavy processing.

## 🛠️ Tools & Functions Explained

### 🔧 Tools
Tools give LLMs abilities beyond their default text processing. They're like plugins allowing real-time interactions with the outside world.

**Examples:**
- 🌤️ Real-time weather data
- 📈 Stock market updates
- ✈️ Flight tracking

### ⚙️ Functions
Functions modify or enhance Open-WebUI itself—like adding ingredients to a recipe or changing how the kitchen runs!

**Examples:**
- Adding support for Anthropic or Vertex AI
- Creating custom buttons on your interface
- Filtering spam or inappropriate messages

### 📌 Key Differences
| Aspect | Tools | Functions |
| ------ | ----- | --------- |
| **Purpose** | Extend LLM capabilities | Modify WebUI behavior |
| **Example** | Fetching real-time data | Adding new AI models or UI components |

## 🚰 Pipes: Creating Custom Agents/Models

Think of Pipes as customizable plumbing systems within Open-WebUI, letting you define exactly how data flows and processes.

### 🧩 Pipe Structure
A basic Pipe consists of:
- **Valves**: Settings controlling behavior (like knobs on a pipe).
- **Pipe Method**: Main logic processing inputs and returning outputs.

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

### 🌈 Multiple Models (Manifold)
You can define multiple models in a single Pipe:

```python
def pipes(self):
    return [
        {"id": "model_id_1", "name": "model_1"},
        {"id": "model_id_2", "name": "model_2"}
    ]
```

## 🪄 Filter Functions: Modify Inputs & Outputs
Filters modify data entering (inlet) and leaving (outlet) the LLM, improving context or readability.

### 🌊 Basic Filter Skeleton

```python
from pydantic import BaseModel

class Filter:
    class Valves(BaseModel):
        pass

    def inlet(self, body: dict) -> dict:
        print(f"Inlet called: {body}")
        return body

    def outlet(self, body: dict) -> dict:
        print(f"Outlet called: {body}")
        return body
```

### 🥗 Practical Examples
- **Adding Context:** Automatically append important context to messages.
- **Sanitizing Input:** Clean user input for clarity.

## 🎯 Action Functions: Interactive Buttons
Action Functions add custom interactive buttons directly under chat messages.

### 🔘 Example Action

```python
async def action(self, body: dict, __user__=None, __event_emitter__=None, __event_call__=None) -> dict:
    response = await __event_call__({
        "type": "input",
        "data": {"title": "Enter Message", "placeholder": "Your message here..."}
    })
    print(response)
```

## 🚚 Migration Guide: Open WebUI 0.4 ➡️ 0.5
Open-WebUI 0.5 introduced a simplified architecture, moving from "apps" to "routers" for better scalability and unified API endpoints.

### 🔄 Quick Migration Tips:
- Update imports from `open_webui.apps` ➡️ `open_webui.routers`.
- Use the unified endpoint `chat_completion` from `open_webui.main`.
- New function signatures now require a `Request` object:

```python
async def pipe(self, body: dict, __user__: dict, __request__: Request):
    # Function logic here
```

## 📌 Important Concepts to Remember
- **Pipelines** are for advanced API integrations and heavy computational tasks.
- **Valves** provide configurable settings for Pipes.

---

🌟 **You're ready to dive into Open-WebUI!** 🎉

Start exploring Tools, Functions, and Pipes today, and discover how flexible and powerful your Open-WebUI can be! Happy coding! 🚀✨

