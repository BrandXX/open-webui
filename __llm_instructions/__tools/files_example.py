"""
title: Files
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1.0
"""

import os
import requests
from datetime import datetime
from typing import List


class Tools:
    def __init__(self):
        # If set to true it will prevent default RAG pipeline
        self.file_handler = True
        self.citation = True
        pass

    def get_files(self, __files__: List[dict] = []) -> str:
        """
        Get the files
        """

        print(__files__)
        return (
            """Show the file content directly using: `/api/v1/files/{file_id}/content`
If the file is video content render the video directly using the following template: {{VIDEO_FILE_ID_[file_id]}}
If the file is html file render the html directly as iframe using the following template: {{HTML_FILE_ID_[file_id]}}"""
            + f"Files: {str(__files__)}"
        )
