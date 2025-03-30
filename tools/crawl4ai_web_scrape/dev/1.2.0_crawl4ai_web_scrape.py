# Updated Crawl4AI Web Scraper Tool with SearXNG integration, search_and_scrape functionality,
# AND LLM-based query generation. The generate_queries method now calls self.llm to produce expansions.
"""
title: Crawl4AI Web Scraper
description: Robust and configurable web scraping tool using Crawl4AI server with advanced options
author: BrandXX/UserX
author_url: https://github.com/BrandXX/open-webui/
funding_url: https://github.com/BrandXX/open-webui/
repo_url: https://github.com/BrandXX/open-webui/edit/main/tools/crawl4ai_web_scrape/
version: 1.1.0
required_open_webui_version: 0.3.9
Notes:
Thanks to 'focuses' over at the Open-WebUI community for providing the initial code @ https://openwebui.com/t/focuses/crawl4ai_web_scrape
"""

import aiohttp
import asyncio
import re
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
    QUERY_GENERATION = (
        "query_generation"  # Signals when we generate queries (LLM-based now)
    )


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
            description="Custom CSS selector override (takes precedence over dropdown selection)",
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
        :param llm: Optional reference to an LLM or chain. Could be:
                    - A LangChain LLMChain instance
                    - An Open-WebUI Python client
                    - A local method that calls the model
        """
        self.valves = self.Valves()
        self.citation = True
        self.llm = llm  # We'll store a reference to the LLM if provided

    async def generate_queries(self, user_query: str) -> List[str]:
        """
        Use an LLM to generate expansions from the user's input.
        The actual code for calling your LLM depends on your environment.
        Below is a simple synchronous call example. If your LLM is purely sync,
        you can wrap it in an executor or adapt accordingly.
        """
        # If there's no LLM, fall back to a trivial expansion
        if not self.llm:
            return [user_query, f"{user_query} information", f"{user_query} research"]

        # Create a prompt or input template for the LLM
        # This is just an example. Tweak to your liking.
        prompt = (
            f"You are a helpful assistant. The user has asked about: '{user_query}'.\n\n"
            "Generate a short list of search queries (3 to 5) that expand on this topic. "
            "Return them as a single line, separated by semicolons."
        )

        try:
            # Example: if your LLM is something like a synchronous function call
            # We'll do this in a thread executor. If your LLM is async, you can await it directly.

            # For demonstration, let's assume self.llm is a synchronous call returning a string:
            # "user_query; user_query info; user_query latest news"
            import concurrent.futures

            loop = asyncio.get_running_loop()
            response_str = await loop.run_in_executor(None, self.llm, prompt)

            # Now parse the expansions from the response
            expansions = [r.strip() for r in response_str.split(";") if r.strip()]
            if not expansions:
                expansions = [user_query]  # fallback if LLM returns nothing

            # De-dup for sanity
            final_queries = []
            seen = set()
            for q in expansions:
                if q not in seen:
                    seen.add(q)
                    final_queries.append(q)

            return final_queries

        except Exception as e:
            # If something goes wrong, we can either raise or just fall back
            print(f"LLM query generation failed: {e}")
            return [user_query]

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
        """Rank URL quality based on various heuristics (0-1 scale)"""
        score = 0.5  # Default neutral score

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
            # 1) Emit a progress update
            await emitter.progress_update(
                f"🔄 Generating search queries from: {user_query}"
            )

            # 2) Emit the new event type for query generation
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "event",
                        "data": {
                            "event": Event.QUERY_GENERATION.value,
                            "description": f"Generating expanded queries from user input: {user_query}",
                        },
                    }
                )

            # 3) Use the LLM to generate expansions
            expanded_queries = await self.generate_queries(user_query)

            # 4) For each generated query, search SearXNG
            all_urls = []
            for idx, q in enumerate(expanded_queries, start=1):
                await emitter.progress_update(
                    f"🔍 Searching SearXNG for query #{idx}: {q}"
                )
                new_urls = await self.search_searxng(q)
                all_urls.extend(new_urls)

            # Deduplicate
            all_urls = list(dict.fromkeys(all_urls))

            if not all_urls:
                return "No search results found from any generated query."

            # (Optional) Ranking
            ranked_urls = []
            if self.valves.ENABLE_RANKING:
                await emitter.progress_update(
                    f"⚖️ Ranking {len(all_urls)} combined search results"
                )
                ranking_tasks = [self.rank_url(url) for url in all_urls]
                ranks = await asyncio.gather(*ranking_tasks)
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
                ranked_urls = [
                    (url, 0.0) for url in all_urls[: self.valves.NUM_RESULTS]
                ]

            if not ranked_urls:
                return "No quality content found matching your criteria."

            # Scraping concurrency
            semaphore = asyncio.Semaphore(self.valves.MAX_CONCURRENT_SCRAPES)

            async def scrape_with_semaphore(url, rank):
                async with semaphore:
                    rank_indicator = (
                        f"[Quality: {rank:.2f}]" if self.valves.ENABLE_RANKING else ""
                    )
                    await emitter.progress_update(
                        f"🌐 Scraping: {url} {rank_indicator}"
                    )
                    markdown = await self.web_scrape(
                        url, __event_emitter__=__event_emitter__, __user__=__user__
                    )

                    if self.valves.INCLUDE_SOURCES:
                        domain = urlparse(url).netloc
                        source_header = f"## Source: {domain} {rank_indicator}\n\n"

                        if self.valves.CITATION_STYLE == "inline":
                            source_footer = f"\n\n*Source: [{domain}]({url})*"
                            return f"---\n\n{source_header}{markdown.strip()}{source_footer}"
                        elif self.valves.CITATION_STYLE == "footnote":
                            return f"---\n\n{source_header}{markdown.strip()}\n\n[^{domain}]: {url}"
                        else:  # endnotes
                            return f"---\n\n{source_header}{markdown.strip()}"
                    else:
                        return f"---\n\n{markdown.strip()}"

            scraping_tasks = [
                scrape_with_semaphore(url, rank) for url, rank in ranked_urls
            ]
            results = await asyncio.gather(*scraping_tasks)
            results = [r for r in results if r and not r.endswith("\n\n")]

            final_output = (
                "\n\n".join(results)
                if results
                else "No quality content found matching your criteria."
            )

            # If endnotes style
            if (
                self.valves.INCLUDE_SOURCES
                and self.valves.CITATION_STYLE == "endnotes"
                and results
            ):
                sources_list = "\n".join(
                    [
                        f"- [{urlparse(url).netloc}]({url})"
                        for url, _ in ranked_urls
                        if url
                    ]
                )
                final_output += f"\n\n## Sources\n{sources_list}"

            return final_output

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

        try:
            result = await crawler.submit_and_wait(request_data, hook=hook)
            markdown = result.get("result", {}).get("markdown", "")
            if not markdown:
                return "No content was retrieved from the webpage."

            # Truncate large content
            max_size_for_processing = 10 * 1024 * 1024
            if len(markdown) > max_size_for_processing:
                markdown = (
                    markdown[:max_size_for_processing]
                    + "\n\n[Content truncated due to size limitations]"
                )

            # Cleanup via regex
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
                original_domain = urllib.parse.urlparse(url).netloc

                def remove_external(match):
                    text, link = match.group(1), match.group(2)
                    link_domain = urllib.parse.urlparse(link).netloc
                    return (
                        text
                        if link_domain and link_domain != original_domain
                        else match.group(0)
                    )

                markdown = re.sub(
                    r"\[([^\]]+)\]\((https?://[^\)]+)\)", remove_external, markdown
                )

            # Remove social media links
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
                    text, link = match.group(1), match.group(2)
                    link_domain = urllib.parse.urlparse(link).netloc
                    return (
                        text
                        if any(domain in link_domain for domain in social_domains)
                        else match.group(0)
                    )

                markdown = re.sub(
                    r"\[([^\]]+)\]\((https?://[^\)]+)\)", remove_social, markdown
                )

            # Finally, enforce a maximum content length if needed
            if self.valves.MAX_CONTENT_LENGTH > 0:
                markdown = markdown[: self.valves.MAX_CONTENT_LENGTH]

            return markdown
        except Exception as e:
            await emitter.error_update(f"❌ Error during web scrape: {str(e)}")
            return f"Error scraping web page: {str(e)}"
