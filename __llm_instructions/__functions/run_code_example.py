from pydantic import BaseModel, Field
from typing import Optional
from fastapi.requests import Request
from io import StringIO
import sys

"""
MODERATOR COMMENT: This function should be utilized with EXTREME caution.
Do not expose to untrusted users or deploy on secure networks unless you are sure you have considered all risks.
"""


class Action:
    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    def execute_python_code(self, code: str) -> str:
        """Executes Python code and returns the output."""
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        try:
            exec(code, {})
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            sys.stdout = old_stdout
        return redirected_output.getvalue()

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        print(f"action:{__name__}")

        user_valves = __user__.get("valves")
        if not user_valves:
            user_valves = self.UserValves()

        if __event_emitter__:
            last_assistant_message = body["messages"][-1]

            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Processing your input", "done": False},
                    }
                )

            # Execute Python code if the input is detected as code
            input_text = last_assistant_message["content"]
            if input_text.startswith("```python") and input_text.endswith("```"):
                code = input_text[9:-3].strip()  # Remove the ```python and ``` markers
                output = self.execute_python_code(code)
                return {"type": "code_execution_result", "data": {"output": output}}

            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "No valid Python code detected",
                            "done": True,
                        },
                    }
                )
