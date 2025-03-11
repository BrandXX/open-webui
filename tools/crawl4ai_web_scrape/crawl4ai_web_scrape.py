import aiohttp
import asyncio
import re
from typing import Callable, Optional, Awaitable, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import time
import urllib.parse


class Event(Enum):
    START = "start"
    WAITING = "waiting"
    FINISHED = "finished"
    ERROR = "error"


class Crawler:
    def __init__(
        self, base_url: str, token: str = None, timeout=300, poll_interval: int = 2
    ):
        self.base_url = base_url
        self.token = token
        self.timeout = timeout
        self.poll_interval = poll_interval  # Valve for polling interval

    async def submit_and_wait(
        self,
        request_data: dict,
        token: str = None,
        timeout: int = 0,
        hook: Optional[Callable[[Event, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> dict:
        task_id = None
        if token is None:
            token = self.token
        if timeout == 0:
            timeout = self.timeout
        headers = {"Authorization": f"Bearer {token}"}

        async with aiohttp.ClientSession(headers=headers) as session:
            endpoint = self.get_crawler_url("crawl")

            try:
                if hook:
                    await hook(Event.START, {"url": request_data["urls"]})

                async with session.post(endpoint, json=request_data) as response:
                    if response.status != 200:
                        raise Exception(
                            f"Failed to submit task: HTTP {response.status}"
                        )

                    response_json = await response.json()
                    task_id = response_json.get("task_id")

                    if not task_id:
                        raise Exception(f"task_id missing in response: {response_json}")

                start_time = time.time()
                if hook:
                    await hook(Event.WAITING, {"task_id": task_id})

                while True:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Task {task_id} timeout")

                    async with session.get(
                        self.get_crawler_url(f"task/{task_id}")
                    ) as result:
                        status = await result.json()

                    if status["status"] == "completed":
                        if hook:
                            await hook(
                                Event.FINISHED,
                                {"task_id": task_id, "status": status["status"]},
                            )
                        return status

                    # Use the poll_interval valve here
                    await asyncio.sleep(self.poll_interval)

            except Exception as e:
                if hook:
                    await hook(Event.ERROR, {"task_id": task_id, "exception": e})
                raise

    def get_crawler_url(self, sub_dir):
        return urllib.parse.urljoin(self.base_url, sub_dir)


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def progress_update(self, description):
        await self.emit(description)

    async def error_update(self, description):
        await self.emit(description, "error", True)

    async def success_update(self, description):
        await self.emit(description, "success", True)

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class Tools:
    class Valves(BaseModel):
        # Connection & Authentication Settings (Basic)
        CRAWL4AI_URL: str = Field(
            default="http://crawl4ai:11235/",
            description="Crawl4ai server URL",
            advanced=False,
        )
        CRAWL4AI_TOKEN: str = Field(
            default="123456", description="Optional Crawl4ai token", advanced=False
        )

        # Scraping Options (Advanced)
        CSS_SELECTOR: Optional[str] = Field(
            default=None,
            description="CSS selector to target specific content",
            advanced=True,
            enum=[
                None,
                "main",
                "article",
                "div.prose",
                "div[class*='article']",
                "section",
                "div.post-content",
            ],
        )
        CSS_SELECTOR_OVERRIDE: Optional[str] = Field(
            default=None,
            description="Custom CSS selector override (takes precedence over dropdown selection)",
            advanced=True,
        )

        # Browser & Simulation Settings
        HEADLESS_MODE: bool = Field(
            default=True, description="Run browser in headless mode", advanced=False
        )
        USER_AGENT: str = Field(
            default="Mozilla/5.0",
            description="Browser User-Agent string",
            advanced=False,
        )
        BROWSER_TYPE: str = Field(
            default="chromium",
            description="Browser type for crawling",
            enum=["chromium", "firefox", "webkit"],
            advanced=True,
        )
        SIMULATE_USER: bool = Field(
            default=True, description="Simulate human browsing behavior", advanced=True
        )
        ENABLE_MAGIC_MODE: bool = Field(
            default=True, description="Advanced anti-detection features", advanced=True
        )
        OVERRIDE_NAVIGATOR: bool = Field(
            default=True,
            description="Override browser navigator properties",
            advanced=True,
        )

        # Performance & Timing Controls
        TIMEOUT_SECONDS: int = Field(
            default=120,
            description="Timeout for scraper tasks in seconds",
            advanced=False,
        )
        POLL_INTERVAL_SECONDS: int = Field(
            default=3,
            description="Polling interval for crawl status checks",
            advanced=True,
        )

        # Content Filtering & Output Options
        CLEANUP_REGEX: Optional[str] = Field(
            default="Menu|Home|About|Contact|Save|Copy|Δ|Recent Posts|Leave a Comment|Reply|Related Posts|Share(?: this)?|Sponsored|Advertisement|Subscribe|Follow us|Back to top|Privacy Policy|Terms of Service|Copyright.*",
            description="Regex for cleaning up unwanted markdown content",
            advanced=True,
        )
        INCLUDE_IMAGES: bool = Field(
            default=False,
            description="Include images in markdown output",
            advanced=False,
        )
        MAX_CONTENT_LENGTH: int = Field(
            default=0,
            description="Max length of output markdown (0 for no limit)",
            advanced=True,
        )
        SKIP_INTERNAL_LINKS: bool = Field(
            default=False,
            description="If True, omit #localAnchors or internal links referencing the same page from markdown output",
            advanced=True,
        )
        EXCLUDE_EXTERNAL_LINKS: bool = Field(
            default=False,
            description="If True, remove external links (those pointing to other domains) from the final markdown output",
            advanced=True,
        )
        EXCLUDE_SOCIAL_MEDIA_LINKS: bool = Field(
            default=False,
            description="If True, remove links pointing to social media platforms from the final markdown output",
            advanced=True,
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True

    async def web_scrape(
        self,
        url: str,
        __event_emitter__: Callable[[dict], Any] = None,
        __user__: dict = {},
    ) -> str:
        emitter = EventEmitter(__event_emitter__)

        async def hook(event: Event, data: Dict[str, Any]):
            if event == Event.START:
                await emitter.progress_update(f"✅ Started scraping {data['url']}")
            elif event == Event.WAITING:
                await emitter.progress_update(
                    f"⏳ Waiting for results... (Task ID: {data['task_id']})"
                )
            elif event == Event.FINISHED:
                await emitter.success_update(
                    f"🎉 Scraping complete! (Task ID: {data['task_id']})"
                )
            elif event == Event.ERROR:
                await emitter.error_update(
                    f"❌ Error during scrape: {data.get('exception')}"
                )

        # Pass in the poll_interval from valves to the Crawler!
        crawler = Crawler(
            self.valves.CRAWL4AI_URL,
            self.valves.CRAWL4AI_TOKEN,
            self.valves.TIMEOUT_SECONDS,
            poll_interval=self.valves.POLL_INTERVAL_SECONDS,
        )

        request_data = {
            "urls": url,
            "crawler_params": {
                "headless": self.valves.HEADLESS_MODE,
                "browser_type": self.valves.BROWSER_TYPE,
                "user_agent": self.valves.USER_AGENT,
                "simulate_user": self.valves.SIMULATE_USER,
                "magic": self.valves.ENABLE_MAGIC_MODE,
                "override_navigator": self.valves.OVERRIDE_NAVIGATOR,
            },
            "extra": {"only_text": not self.valves.INCLUDE_IMAGES},
        }

        selector = self.valves.CSS_SELECTOR_OVERRIDE or self.valves.CSS_SELECTOR
        if selector:
            request_data["css_selector"] = selector

        result = await crawler.submit_and_wait(request_data, hook=hook)
        markdown = result["result"].get("markdown")
        if markdown and self.valves.CLEANUP_REGEX:
            markdown = re.sub(self.valves.CLEANUP_REGEX, "", markdown, flags=re.DOTALL)

        # Apply the skip_internal_links valve if enabled
        if markdown and self.valves.SKIP_INTERNAL_LINKS:
            # Remove local anchor links (e.g., [text](#anchor))
            markdown = re.sub(r"\[([^\]]+)\]\((#.*?)\)", r"\1", markdown)
            # Remove links that point exactly to the original URL
            markdown = re.sub(
                r"\[([^\]]+)\]\((" + re.escape(url) + r")\)", r"\1", markdown
            )

        # Apply the exclude_external_links valve if enabled
        if markdown and self.valves.EXCLUDE_EXTERNAL_LINKS:
            # Parse the domain of the original URL
            original_domain = urllib.parse.urlparse(url).netloc

            # Remove markdown links that point to an external domain
            def remove_external(match):
                text = match.group(1)
                link = match.group(2)
                link_domain = urllib.parse.urlparse(link).netloc
                if link_domain and link_domain != original_domain:
                    return text
                return match.group(0)

            markdown = re.sub(
                r"\[([^\]]+)\]\((https?://[^\)]+)\)", remove_external, markdown
            )

        # Apply the exclude_social_media_links valve if enabled
        if markdown and self.valves.EXCLUDE_SOCIAL_MEDIA_LINKS:
            social_domains = [
                "facebook.com",
                "twitter.com",
                "instagram.com",
                "linkedin.com",
                "pinterest.com",
                "tiktok.com",
                "snapchat.com",
            ]

            def remove_social(match):
                text = match.group(1)
                link = match.group(2)
                link_domain = urllib.parse.urlparse(link).netloc
                for domain in social_domains:
                    if domain in link_domain:
                        return text
                return match.group(0)

            markdown = re.sub(
                r"\[([^\]]+)\]\((https?://[^\)]+)\)", remove_social, markdown
            )

        if self.valves.MAX_CONTENT_LENGTH > 0:
            markdown = markdown[: self.valves.MAX_CONTENT_LENGTH]
        return markdown
