import aiohttp
import asyncio
import re
import json
from typing import Callable, Optional, Awaitable, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field
import time
import urllib.parse
import gc
from urllib.parse import urlparse


class Event(Enum):
    START = "start"
    WAITING = "waiting"
    FINISHED = "finished"
    ERROR = "error"
    QUERY_GENERATION = "query_generation"


class Crawler:
    def __init__(
        self, base_url: str, token: str = None, timeout=300, poll_interval: int = 2
    ):
        self.base_url = base_url
        self.token = token
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._session = None

    def cleanup(self):
        self._session = None
        gc.collect()

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
        CRAWL4AI_URL: str = Field(
            default="http://crawl4ai:11235/",
            description="Crawl4ai server URL",
            advanced=False,
        )
        CRAWL4AI_TOKEN: str = Field(
            default="123456", description="Optional Crawl4ai token", advanced=False
        )
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
            description="Custom CSS selector override (takes precedence)",
            advanced=True,
        )
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
        CLEANUP_REGEX: Optional[str] = Field(
            default=(
                "Menu|Home|About|Contact|Save|Copy|Δ|Recent Posts|Leave a Comment|Reply|"
                "Related Posts|Share(?: this)?|Sponsored|Advertisement|Subscribe|Follow us|"
                "Back to top|Privacy Policy|Terms of Service|Copyright.*"
            ),
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
            default=False, description="Omit internal page links", advanced=True
        )
        EXCLUDE_EXTERNAL_LINKS: bool = Field(
            default=False, description="Remove external links", advanced=True
        )
        EXCLUDE_SOCIAL_MEDIA_LINKS: bool = Field(
            default=False, description="Remove social media links", advanced=True
        )
        SEARXNG_URL: str = Field(
            default="http://host.docker.internal:8080/search?q=<query>&format=json",
            description="SearXNG URL",
            advanced=True,
        )
        NUM_RESULTS: int = Field(
            default=5,
            description="Number of URLs to scrape from SearXNG",
            advanced=True,
        )
        ENABLE_RANKING: bool = Field(
            default=True,
            description="Enable content quality ranking before scraping",
            advanced=False,
        )
        MIN_RANK_THRESHOLD: float = Field(
            default=0.6,
            description="Minimum rank score (0-1) to scrape content",
            advanced=True,
        )
        SKIP_LOW_QUALITY: bool = Field(
            default=True, description="Skip scraping low quality content", advanced=True
        )
        MAX_CONCURRENT_SCRAPES: int = Field(
            default=5,
            description="Maximum number of concurrent scraping operations",
            advanced=True,
        )
        INCLUDE_SOURCES: bool = Field(
            default=True,
            description="Include source information in results",
            advanced=False,
        )
        CITATION_STYLE: str = Field(
            default="inline",
            description="How to include source citations",
            enum=["inline", "footnote", "endnotes"],
            advanced=True,
        )
        OLLAMA_ENDPOINT: str = Field(
            default="http://host.docker.internal:11434",
            description="Ollama endpoint URL",
            advanced=False,
        )

    def __init__(self, llm=None):
        self.valves = self.Valves()
        self.citation = True
        self.llm = None
        self.ollama_model = None
        self._ollama_checked = False

    async def setup_ollama(self) -> bool:
        if self._ollama_checked:
            return self.ollama_model is not None

        self._ollama_checked = True
        try:
            async with aiohttp.ClientSession() as session:
                ps_url = f"{self.valves.OLLAMA_ENDPOINT}/api/ps"
                async with session.get(ps_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models_list = data.get("models", [])
                        if models_list and isinstance(models_list, list):
                            first_model = models_list[0].get("model")
                            if first_model:
                                self.ollama_model = first_model
                                return True
        except Exception as e:
            print(f"Ollama model detection failed: {e}")

        self.ollama_model = None
        return False

    async def call_ollama(self, system_prompt: str, user_query: str) -> str:
        """
        Put instructions in the system prompt so the model remains broad
        and avoids domain-specific knowledge. The user prompt is minimal.
        """
        if not self.ollama_model:
            return ""

        body = {
            "system": system_prompt,
            "prompt": user_query,
            "model": self.ollama_model,
            "stream": False,
        }
        try:
            async with aiohttp.ClientSession() as session:
                generate_url = f"{self.valves.OLLAMA_ENDPOINT}/api/generate"
                async with session.post(generate_url, json=body) as resp:
                    if resp.status != 200:
                        print(f"Ollama generate call failed: HTTP {resp.status}")
                        return ""
                    data = await resp.json()
                    return data.get("response", "")
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""

    async def generate_queries(self, user_query: str) -> List[str]:
        # Attempt Ollama detection
        has_model = await self.setup_ollama()
        if not has_model:
            return [user_query, f"{user_query} info", f"{user_query} research"]

        # System prompt to keep expansions broad and general:
        system_instructions = (
            "You have no external knowledge. You are a broad search-query generator. "
            "The user gave a topic, but you do NOT know any specifics. "
            "You only produce 3 to 5 short, generic queries separated by semicolons, with no disclaimers. "
            "Focus on synonyms, broader categories, or sub-topics related to the user's request. "
            "Avoid referencing any unmentioned or detailed events or data. "
            "Example: If user says 'Elon Musk latest political news', respond with queries like: "
            "'Elon Musk political overview; Elon Musk public statements; Elon Musk controversies; Elon Musk policy timeline'."
        )

        # The user_query is minimal
        raw_response = await self.call_ollama(system_instructions, user_query)

        # In case disclaimers appear, we parse out only semicolon-delimited strings
        parts = [r.strip() for r in raw_response.split(";") if r.strip()]
        if not parts:
            return [user_query]  # fallback

        # De-dupe
        final_queries = []
        seen = set()
        for p in parts:
            if p not in seen:
                seen.add(p)
                final_queries.append(p)

        return final_queries

    async def search_searxng(self, query: str) -> list:
        url = self.valves.SEARXNG_URL.replace("<query>", urllib.parse.quote(query))
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(
                        f"SearXNG search failed with status {response.status}"
                    )
                data = await response.json()
                seen, links = set(), []
                for r in data.get("results", []):
                    link = r.get("url")
                    if link and link not in seen:
                        links.append(link)
                        seen.add(link)
                return links

    async def rank_url(self, url: str) -> float:
        score = 0.5
        domain = urlparse(url).netloc
        if any(premium in domain for premium in [".edu", ".gov", ".org"]):
            score += 0.2
        if re.search(r"news|blog|article|research|paper|study", domain):
            score += 0.1
        if re.search(r"spam|click|tracker|ad\.|ads\.|advert", domain):
            score -= 0.3

        path = urlparse(url).path
        if re.search(r"article|post|blog|research|paper|study", path):
            score += 0.1
        if re.search(r"download|signup|register|login", path):
            score -= 0.1

        return max(0.0, min(1.0, score))

    async def search_and_scrape(
        self,
        user_query: str,
        __event_emitter__: Callable[[dict], Any] = None,
        __user__: dict = {},
    ) -> str:
        emitter = EventEmitter(__event_emitter__)
        try:
            await emitter.progress_update(
                f"🔄 Generating search queries from: {user_query}"
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "event",
                        "data": {
                            "event": Event.QUERY_GENERATION.value,
                            "description": f"Generating queries from user input: {user_query}",
                        },
                    }
                )

            expanded_queries = await self.generate_queries(user_query)

            all_urls = []
            for idx, q in enumerate(expanded_queries, start=1):
                await emitter.progress_update(
                    f"🔍 Searching SearXNG for query #{idx}: {q}"
                )
                new_urls = await self.search_searxng(q)
                all_urls.extend(new_urls)

            all_urls = list(dict.fromkeys(all_urls))

            if not all_urls:
                return "No search results found from any generated query."

            ranked_urls = []
            if self.valves.ENABLE_RANKING:
                await emitter.progress_update(f"⚖️ Ranking {len(all_urls)} results")
                tasks = [self.rank_url(u) for u in all_urls]
                ranks = await asyncio.gather(*tasks)
                ranked_urls = list(zip(all_urls, ranks))
                ranked_urls.sort(key=lambda x: x[1], reverse=True)

                if self.valves.SKIP_LOW_QUALITY:
                    ranked_urls = [
                        (url, rank)
                        for url, rank in ranked_urls
                        if rank >= self.valves.MIN_RANK_THRESHOLD
                    ]

                ranked_urls = ranked_urls[: self.valves.NUM_RESULTS]
            else:
                ranked_urls = [(u, 0.0) for u in all_urls[: self.valves.NUM_RESULTS]]

            if not ranked_urls:
                return "No quality content found matching your criteria."

            semaphore = asyncio.Semaphore(self.valves.MAX_CONCURRENT_SCRAPES)

            async def scrape(url, rank):
                async with semaphore:
                    rank_display = (
                        f"[Quality: {rank:.2f}]" if self.valves.ENABLE_RANKING else ""
                    )
                    await emitter.progress_update(f"🌐 Scraping: {url} {rank_display}")
                    markdown = await self.web_scrape(
                        url, __event_emitter__=__event_emitter__, __user__=__user__
                    )

                    if self.valves.INCLUDE_SOURCES:
                        domain = urlparse(url).netloc
                        hdr = f"## Source: {domain} {rank_display}\n\n"
                        if self.valves.CITATION_STYLE == "inline":
                            foot = f"\n\n*Source: [{domain}]({url})*"
                            return f"---\n\n{hdr}{markdown.strip()}{foot}"
                        elif self.valves.CITATION_STYLE == "footnote":
                            return (
                                f"---\n\n{hdr}{markdown.strip()}\n\n[^{domain}]: {url}"
                            )
                        else:
                            return f"---\n\n{hdr}{markdown.strip()}"
                    else:
                        return f"---\n\n{markdown.strip()}"

            tasks = [scrape(u, r) for u, r in ranked_urls]
            results = await asyncio.gather(*tasks)
            results = [r for r in results if r and not r.endswith("\n\n")]

            final_out = "\n\n".join(results) if results else "No quality content found."
            if (
                self.valves.INCLUDE_SOURCES
                and self.valves.CITATION_STYLE == "endnotes"
                and results
            ):
                src_list = "\n".join(
                    [f"- [{urlparse(u).netloc}]({u})" for u, _ in ranked_urls if u]
                )
                final_out += f"\n\n## Sources\n{src_list}"

            return final_out

        except Exception as e:
            await emitter.error_update(f"❌ Failed during search and scrape: {str(e)}")
            return f"Error: {str(e)}"

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

        crawler = Crawler(
            self.valves.CRAWL4AI_URL,
            self.valves.CRAWL4AI_TOKEN,
            self.valves.TIMEOUT_SECONDS,
            poll_interval=self.valves.POLL_INTERVAL_SECONDS,
        )

        req_data = {
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
            req_data["css_selector"] = selector

        try:
            result = await crawler.submit_and_wait(req_data, hook=hook)
            markdown = result.get("result", {}).get("markdown", "")
            if not markdown:
                return "No content was retrieved from the webpage."

            # Truncate large content
            max_size = 10 * 1024 * 1024
            if len(markdown) > max_size:
                markdown = markdown[:max_size] + "\n\n[Content truncated due to size]"

            # Cleanup
            if markdown and self.valves.CLEANUP_REGEX:
                try:
                    markdown = re.sub(
                        self.valves.CLEANUP_REGEX, "", markdown, flags=re.DOTALL
                    )
                except re.error as e:
                    await emitter.error_update(f"Error in regex pattern: {str(e)}")

            # Remove internal links
            if markdown and self.valves.SKIP_INTERNAL_LINKS:
                markdown = re.sub(r"\[([^\]]+)\]\((#.*?)\)", r"\1", markdown)
                markdown = re.sub(
                    r"\[([^\]]+)\]\((" + re.escape(url) + r")\)", r"\1", markdown
                )

            # Remove external links
            if markdown and self.valves.EXCLUDE_EXTERNAL_LINKS:
                original_domain = urlparse(url).netloc

                def remove_external(m):
                    txt, link = m.group(1), m.group(2)
                    ldom = urlparse(link).netloc
                    return txt if ldom and ldom != original_domain else m.group(0)

                markdown = re.sub(
                    r"\[([^\]]+)\]\((https?://[^\)]+)\)", remove_external, markdown
                )

            # Remove social links
            if markdown and self.valves.EXCLUDE_SOCIAL_MEDIA_LINKS:
                social = [
                    "facebook.com",
                    "twitter.com",
                    "instagram.com",
                    "linkedin.com",
                    "pinterest.com",
                    "tiktok.com",
                    "snapchat.com",
                ]

                def remove_social(m):
                    txt, link = m.group(1), m.group(2)
                    ldom = urlparse(link).netloc
                    return txt if any(d in ldom for d in social) else m.group(0)

                markdown = re.sub(
                    r"\[([^\]]+)\]\((https?://[^\)]+)\)", remove_social, markdown
                )

            # Enforce max content length
            if self.valves.MAX_CONTENT_LENGTH > 0:
                markdown = markdown[: self.valves.MAX_CONTENT_LENGTH]

            return markdown
        except Exception as e:
            await emitter.error_update(f"❌ Error during web scrape: {str(e)}")
            return f"Error scraping web page: {str(e)}"
