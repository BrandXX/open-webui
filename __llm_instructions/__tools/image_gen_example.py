"""
title: Image Gen
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1
required_open_webui_version: 0.5.3
"""

import os
import requests
from datetime import datetime
from typing import Callable
from fastapi import Request

from open_webui.routers.images import image_generations, GenerateImageForm
from open_webui.models.users import Users


class Tools:
    def __init__(self):
        pass

    async def generate_image(
        self, prompt: str, __request__: Request, __user__: dict, __event_emitter__=None
    ) -> str:
        """
        Generate an image given a prompt

        :param prompt: prompt to use for image generation
        """

        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Generating an image", "done": False},
            }
        )

        try:
            images = await image_generations(
                request=__request__,
                form_data=GenerateImageForm(**{"prompt": prompt}),
                user=Users.get_user_by_id(__user__["id"]),
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Generated an image", "done": True},
                }
            )

            for image in images:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {"content": f"![Generated Image]({image['url']})"},
                    }
                )

            return f"Notify the user that the image has been successfully generated"

        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"An error occured: {e}", "done": True},
                }
            )

            return f"Tell the user: {e}"
