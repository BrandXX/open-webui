"""
title: AI Researcher
description: A powerful web scraping and research tool integrating Crawl4AI, Ollama, and SearXNG to deliver high-quality, ranked content. Generates targeted search queries, fetches relevant URLs, and scrapes content concurrently with robust anti-detection features and customizable extraction settings.
author: BrandXX/UserX
author_url: https://github.com/BrandXX/open-webui/
funding_url: https://github.com/BrandXX/open-webui/
repo_url: https://github.com/BrandXX/open-webui/edit/main/tools/crawl4ai_web_scrape/
version: 1.4.1
required_open_webui_version: 0.3.9
Notes:
Thanks to 'focuses' over at the Open-WebUI community for providing the initial code @ https://openwebui.com/t/focuses/crawl4ai_web_scrape

Crawl4AI Web Scraper Tool
-------------------------
This tool integrates with SearXNG for search, uses Crawl4AI for scraping,
and utilizes Ollama to generate short, semicolon-delimited search queries.

Key functionality:
- Valves for user configuration (including Ollama endpoint/model).
- Query generation from user input.
- SearXNG searching using the generated queries.
- Optional ranking of search results.
- Scraping each URL via Crawl4AI with concurrency.
"""

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
    """
    Enumeration of events for logging and tracking the scraping/search process.
    """

    START = "start"
    WAITING = "waiting"
    FINISHED = "finished"
    ERROR = "error"
    QUERY_GENERATION = "query_generation"


class Crawler:
    """
    A lightweight wrapper around the Crawl4AI endpoint that manages
    asynchronous scraping tasks, polling for completion.
    """

    def __init__(
        self, base_url: str, token: str = None, timeout=300, poll_interval: int = 2
    ):
        """
        :param base_url: Base URL for the Crawl4AI server.
        :param token: Optional auth token for Crawl4AI.
        :param timeout: Overall timeout in seconds for a scrape task.
        :param poll_interval: How frequently (in seconds) to poll the status of a scrape task.
        """
        self.base_url = base_url
        self.token = token
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._session = None

    def cleanup(self):
        """
        Cleanup references and run garbage collection.
        """
        self._session = None
        gc.collect()

    async def submit_and_wait(
        self,
        request_data: dict,
        token: str = None,
        timeout: int = 0,
        hook: Optional[Callable[[Event, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> dict:
        """
        Submit a scraping job to Crawl4AI, then poll until it's completed or times out.
        :param request_data: JSON data to post to the 'crawl' endpoint.
        :param token: Optional auth token.
        :param timeout: Optional override for the operation's timeout.
        :param hook: Optional async callback to receive event updates.
        :return: The final JSON status from Crawl4AI, which should include 'result' data.
        """
        task_id = None
        if token is None:
            token = self.token
        if timeout == 0:
            timeout = self.timeout
        headers = {"Authorization": f"Bearer {token}"}

        async with aiohttp.ClientSession(headers=headers) as session:
            endpoint = self.get_crawler_url("crawl")

            try:
                # Optional event hook for start
                if hook:
                    await hook(Event.START, {"url": request_data["urls"]})

                # Submit the crawl job
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

                # Optional event hook for waiting
                if hook:
                    await hook(Event.WAITING, {"task_id": task_id})

                # Poll for completion until timeout
                while True:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Task {task_id} timeout")

                    async with session.get(
                        self.get_crawler_url(f"task/{task_id}")
                    ) as result:
                        status = await result.json()

                    if status["status"] == "completed":
                        # Optional event hook for finished
                        if hook:
                            await hook(
                                Event.FINISHED,
                                {"task_id": task_id, "status": status["status"]},
                            )
                        return status

                    await asyncio.sleep(self.poll_interval)

            except Exception as e:
                # Optional event hook for errors
                if hook:
                    await hook(Event.ERROR, {"task_id": task_id, "exception": e})
                raise

    def get_crawler_url(self, sub_dir: str) -> str:
        """
        Build a full URL for a given subpath using the base URL.
        :param sub_dir: Subpath like 'crawl' or 'task/<id>'.
        :return: The absolute URL.
        """
        return urllib.parse.urljoin(self.base_url, sub_dir)


class EventEmitter:
    """
    Helper class to unify event emission. Tools can call progress_update,
    error_update, etc., and the relevant data is sent to a provided emitter callback.
    """

    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        """
        :param event_emitter: Callback that receives dictionary events for logging or UI updates.
        """
        self.event_emitter = event_emitter

    async def progress_update(self, description: str):
        """
        Emit a progress event with 'in_progress' status.
        """
        await self.emit(description)

    async def error_update(self, description: str):
        """
        Emit an error event with 'error' status.
        """
        await self.emit(description, "error", True)

    async def success_update(self, description: str):
        """
        Emit a success event with 'success' status.
        """
        await self.emit(description, "success", True)

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        """
        Generic method to dispatch an event with the given state.
        :param description: Human-readable description or message.
        :param status: The status string (e.g. 'in_progress', 'error', 'success').
        :param done: Boolean indicating whether this event marks a final state.
        """
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
    """
    Main class that holds:
    - Valves (user-configurable settings) for both crawling and query generation.
    - Methods to generate queries via Ollama, search with SearXNG, rank results, and
      scrape pages via Crawl4AI.
    """

    class Valves(BaseModel):
        """
        Pydantic model specifying user-configurable valves/parameters for the tool.
        """

        # Ollama valves
        OLLAMA_ENDPOINT: str = Field(
            default="http://host.docker.internal:11434",
            description="Ollama endpoint URL",
            advanced=False,
        )
        OLLAMA_QUERY_MODEL: str = Field(
            default="llama3.2:latest",
            description="Name of the Ollama LLM to use for query generation. If empty, fallback expansions are used.",
            advanced=False,
        )

        # Crawl4AI + SearxNG valves
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

    def __init__(self, llm=None):
        """
        :param llm: Deprecated parameter (kept for legacy compatibility).
                    Query generation is handled by Ollama calls instead.
        """
        self.valves = self.Valves()
        self.citation = True
        self.llm = None

    async def call_ollama(
        self, model_name: str, system_prompt: str, user_query: str
    ) -> str:
        """
        Call the Ollama API at self.valves.OLLAMA_ENDPOINT with the specified model_name,
        system instructions, and user prompt. Non-streaming call, returning the final text.
        :param model_name: The user-specified (or default) model name for Ollama.
        :param system_prompt: A system-level instruction string.
        :param user_query: The user’s actual query/topic.
        :return: The 'response' string from Ollama's JSON.
        """
        body = {
            "system": system_prompt,
            "prompt": user_query,
            "model": model_name,
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
        """
        Generate short, semicolon-delimited queries from user input via Ollama.
        - If OLLAMA_QUERY_MODEL is empty, return trivial expansions.
        - Otherwise, call `call_ollama` using system instructions for narrow output.
        :param user_query: The user’s topic or request.
        :return: A list of 3-5 short queries or fallback expansions.
        """
        model_name = (self.valves.OLLAMA_QUERY_MODEL or "").strip()
        # If no model is provided, fallback to trivial expansions
        if not model_name:
            return [user_query, f"{user_query} info", f"{user_query} research"]

        # New system instructions focusing on user-supplied topic only
        system_instructions = (
            "You have no external knowledge beyond what the user provides. "
            "The user has given you a topic or request, and you must base your search queries strictly on that topic. "
            "If the user references something unknown or ambiguous, do not guess or invent details. "
            "Simply reflect the user’s request in your queries.\n\n"
            "You will produce exactly 3 to 5 short search queries, separated by semicolons, with no disclaimers or commentary. "
            "Avoid referencing any events, data, or specifics not explicitly mentioned by the user. "
            "If you do not recognize the user’s topic, you still create queries focusing on the user’s exact request, without speculation.\n\n"
            "Your entire output must consist only of these 3 to 5 queries. No disclaimers, no extra words."
        )

        # Call Ollama for the generation
        raw_response = await self.call_ollama(
            model_name, system_instructions, user_query
        )

        # Parse the semicolon-delimited output
        parts = [r.strip() for r in raw_response.split(";") if r.strip()]
        if not parts:
            return [user_query, f"{user_query} info", f"{user_query} research"]

        # De-duplicate any repeated queries
        final_queries = []
        seen = set()
        for p in parts:
            if p not in seen:
                seen.add(p)
                final_queries.append(p)

        return final_queries

    async def search_searxng(self, query: str) -> list:
        """
        Use SearXNG to get search results for a given query.
        :param query: A single search query string.
        :return: A list of unique result URLs from the JSON response.
        """
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
        """
        Simple heuristic rank from 0 to 1, based on domain name and path.
        Higher is presumably better quality.
        :param url: The URL to rank.
        :return: A float 0.0 - 1.0 indicating the rank.
        """
        score = 0.5
        domain = urlparse(url).netloc

        # Domain hints
        if any(premium in domain for premium in [".edu", ".gov", ".org"]):
            score += 0.2
        if re.search(r"news|blog|article|research|paper|study", domain):
            score += 0.1
        if re.search(r"spam|click|tracker|ad\.|ads\.|advert", domain):
            score -= 0.3

        # Path hints
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
        """
        Main entry point:
        1) Generate queries from user input.
        2) Search SearXNG for each query, gather URLs.
        3) Optionally rank & filter results.
        4) Scrape the final set of URLs concurrently.
        5) Return combined markdown from all scrapes.

        :param user_query: The user’s entire question/keyword.
        :param __event_emitter__: Optional callback for status updates.
        :param __user__: Optional user context (not used in this example).
        :return: Combined markdown from all scrapes, or an error/empty message.
        """
        emitter = EventEmitter(__event_emitter__)
        try:
            # Step 1: Query generation
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

            # Step 2: Search SearXNG for each expanded query
            all_urls = []
            for idx, q in enumerate(expanded_queries, start=1):
                await emitter.progress_update(
                    f"🔍 Searching SearXNG for query #{idx}: {q}"
                )
                new_urls = await self.search_searxng(q)
                all_urls.extend(new_urls)

            # Remove duplicates
            all_urls = list(dict.fromkeys(all_urls))
            if not all_urls:
                return "No search results found from any generated query."

            # Step 3: (Optional) rank and filter
            ranked_urls = []
            if self.valves.ENABLE_RANKING:
                await emitter.progress_update(f"⚖️ Ranking {len(all_urls)} results")
                tasks = [self.rank_url(u) for u in all_urls]
                ranks = await asyncio.gather(*tasks)
                ranked_urls = list(zip(all_urls, ranks))

                # Sort descending by rank
                ranked_urls.sort(key=lambda x: x[1], reverse=True)

                # Filter out low quality if requested
                if self.valves.SKIP_LOW_QUALITY:
                    ranked_urls = [
                        (url, rank)
                        for url, rank in ranked_urls
                        if rank >= self.valves.MIN_RANK_THRESHOLD
                    ]

                # Truncate to the top N results
                ranked_urls = ranked_urls[: self.valves.NUM_RESULTS]
            else:
                # If ranking disabled, just slice first N
                ranked_urls = [(u, 0.0) for u in all_urls[: self.valves.NUM_RESULTS]]

            if not ranked_urls:
                return "No quality content found matching your criteria."

            # Step 4: Scrape the final set of URLs (with concurrency)
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

                    # Optionally include source info
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

            # Filter out empty or blank results
            results = [r for r in results if r and not r.endswith("\n\n")]

            # Combine them
            final_out = "\n\n".join(results) if results else "No quality content found."

            # If endnotes, build a final list
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
            # Any unhandled exception is reported as an error
            await emitter.error_update(f"❌ Failed during search and scrape: {str(e)}")
            return f"Error: {str(e)}"

    async def web_scrape(
        self,
        url: str,
        __event_emitter__: Callable[[dict], Any] = None,
        __user__: dict = {},
    ) -> str:
        """
        Scrape a single URL using Crawl4AI. Supports optional concurrency limit from the caller.
        :param url: The webpage to scrape.
        :param __event_emitter__: Optional callback for logging.
        :param __user__: Optional user context (not used in this example).
        :return: Extracted markdown or an error message.
        """
        emitter = EventEmitter(__event_emitter__)

        async def hook(event: Event, data: Dict[str, Any]):
            """
            Hook to adapt Crawler events to the general event emitter.
            """
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

        # Respect optional CSS selector override
        selector = self.valves.CSS_SELECTOR_OVERRIDE or self.valves.CSS_SELECTOR
        if selector:
            req_data["css_selector"] = selector

        try:
            # Submit the scraping job to Crawl4AI and wait for completion
            result = await crawler.submit_and_wait(req_data, hook=hook)
            markdown = result.get("result", {}).get("markdown", "")
            if not markdown:
                return "No content was retrieved from the webpage."

            # Truncate overly large content
            max_size = 10 * 1024 * 1024
            if len(markdown) > max_size:
                markdown = markdown[:max_size] + "\n\n[Content truncated due to size]"

            # Cleanup regex
            if markdown and self.valves.CLEANUP_REGEX:
                try:
                    markdown = re.sub(
                        self.valves.CLEANUP_REGEX, "", markdown, flags=re.DOTALL
                    )
                except re.error as e:
                    await emitter.error_update(f"Error in regex pattern: {str(e)}")

            # Remove internal links if requested
            if markdown and self.valves.SKIP_INTERNAL_LINKS:
                markdown = re.sub(r"\[([^\]]+)\]\((#.*?)\)", r"\1", markdown)
                markdown = re.sub(
                    r"\[([^\]]+)\]\((" + re.escape(url) + r")\)", r"\1", markdown
                )

            # Remove external links if requested
            if markdown and self.valves.EXCLUDE_EXTERNAL_LINKS:
                original_domain = urlparse(url).netloc

                def remove_external(m):
                    txt, link = m.group(1), m.group(2)
                    ldom = urlparse(link).netloc
                    return txt if ldom and ldom != original_domain else m.group(0)

                markdown = re.sub(
                    r"\[([^\]]+)\]\((https?://[^\)]+)\)", remove_external, markdown
                )

            # Remove social media links if requested
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

            # Enforce a maximum content length if configured
            if self.valves.MAX_CONTENT_LENGTH > 0:
                markdown = markdown[: self.valves.MAX_CONTENT_LENGTH]

            return markdown

        except Exception as e:
            # Any exception is reported as an error state
            await emitter.error_update(f"❌ Error during web scrape: {str(e)}")
            return f"Error scraping web page: {str(e)}"
