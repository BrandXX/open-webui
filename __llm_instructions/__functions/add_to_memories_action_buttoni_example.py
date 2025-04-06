"""
title: Add to Memory Action Button
author: Peter De-Ath
author_url: https://github.com/Peter-De-Ath
funding_url: https://github.com/open-webui
version: 0.1.3
icon_url: data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDMyIDMyIj4KICA8ZyB0cmFuc2Zvcm09InJvdGF0ZSgtOTAgMTYgMTYpIj4KICAgIDxwYXRoIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRjNGM0YyIgc3Ryb2tlLXdpZHRoPSIxLjUiIGQ9Ik03LDE2YzAtNC40LDMuNi04LDgtOGMzLjMsMCw2LjIsMiw3LjQsNC44YzIuMSwwLjMsMy42LDIsMy42LDQuMmMwLDEuNC0wLjcsMi42LTEuNywzLjQKICAgICAgYzEsMC44LDEuNywyLDEuNywzLjRjMCwyLjQtMS45LDQuMy00LjMsNC4zYy0wLjUsMS45LTIuMiwzLjMtNC4yLDMuM2MtMS41LDAtMi44LTAuNy0zLjYtMS44Yy0wLjgsMS4xLTIuMSwxLjgtMy42LDEuOAogICAgICBjLTIuNSwwLTQuNS0yLTQuNS00LjVjMC0xLjQsMC42LTIuNiwxLjYtMy40QzYuNiwyMi42LDYsMjEuNCw2LDIwQzYsMTguMiw3LjIsMTYuNiw5LDE2LjIiLz4KICAgIDxwYXRoIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRjNGM0YyIgc3Ryb2tlLXdpZHRoPSIxLjUiIGQ9Ik0xMSwxNGMwLjUtMSwxLjUtMiwyLjUtMi41YzEtMC41LDItMC41LDMtMC41Ii8+CiAgICA8cGF0aCBmaWxsPSJub25lIiBzdHJva2U9IiM0YzRjNGMiIHN0cm9rZS13aWR0aD0iMS41IiBkPSJNMTMsMTljMC0xLjUsMC41LTMsMi00Ii8+CiAgICA8cGF0aCBmaWxsPSJub25lIiBzdHJva2U9IiM0YzRjNGMiIHN0cm9rZS13aWR0aD0iMS41IiBkPSJNMTgsMTVjMSwwLjUsMiwxLjUsMi41LDIuNWMwLjUsMSwwLjUsMiwwLjUsMyIvPgogICAgPHBhdGggZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGM0YzRjIiBzdHJva2Utd2lkdGg9IjEuNSIgZD0iTTE1LDIyYzAsMS41LDAuNSwzLDIsNCIvPgogIDwvZz4KPC9zdmc+
required_open_webui_version: 0.5.0
"""

from pydantic import BaseModel, Field
from typing import Optional

from fastapi.requests import Request
from open_webui.routers.users import Users
from open_webui.routers.memories import add_memory, AddMemoryForm


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

    async def action(
        self,
        body: dict,
        __request__: Request,
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
            user = Users.get_user_by_id(__user__["id"])

            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Adding to Memories", "done": False},
                    }
                )

            # add the assistant response to memories
            try:
                await add_memory(
                    request=__request__,
                    form_data=AddMemoryForm(content=last_assistant_message["content"]),
                    user=user,
                )
                
                if user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "Memory Saved", "done": True},
                        }
                    )
            except Exception as e:
                print(f"Error adding memory {str(e)}")
                if user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Error Adding Memory",
                                "done": True,
                            },
                        }
                    )

                    # add a citation to the message with the error
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "source": {"name": "Error:adding memory"},
                                "document": [str(e)],
                                "metadata": [{"source": "Add to Memory Action Button"}],
                            },
                        }
                    )
