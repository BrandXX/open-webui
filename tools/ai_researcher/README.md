# AI Researcher

Welcome to the **AI Researcher** tool for Open-WebUI! Originally created as a simple web-scraping utility, this tool is now **repurposed and expanded** to harness the power of Crawl4AI, Ollama, and SearXNG for comprehensive AI-driven research. From generating precise search queries to concurrent scraping and ranking, AI Researcher securely handles complex research workflows. Ready to level up your web scraping and research capabilities? Let’s dive in!

---

## Overview

**AI Researcher** provides:

- **AI-Driven Query Generation**  
  Leveraging Ollama to transform your prompt into precise search queries.

- **SearXNG Integration**  
  Collects URLs from multiple search engines with user-defined ranking thresholds.

- **Concurrent Web Scraping**  
  Uses Crawl4AI with asynchronous scraping to efficiently gather page data.

- **Quality Ranking & Filtering**  
  Optionally rank and filter out low-quality URLs before scraping.

- **Real-Time Updates**  
  Provides live feedback on progress, errors, and completion.

- **Advanced Browser Simulation**  
  Simulates human browsing (headless, user-agent overrides, anti-detection features).

- **Content Cleaning & Formatting**  
  Cleans, formats, and optionally truncates output markdown.

---

## Features

- **Asynchronous Efficiency**  
  Swiftly scrapes multiple URLs without pausing your workflow.

- **AI-Powered Queries**  
  Ollama-generated search queries boost relevance and precision.

- **Configurable Ranking**  
  Simple domain-based heuristics prioritize higher-quality sources.

- **Customizable Valves**  
  Fine-tune scraping, formatting, concurrency, and more.

- **Safe & Stable**  
  Memory safeguards and robust error handling prevent crashes.

- **Live Progress Tracking**  
  Monitors each step of the research pipeline.

---

## Configuration

Below are key configuration options in the `Tools.Valves` class:

### Ollama Settings

- **`OLLAMA_ENDPOINT`**  
  - *Default:* `http://host.docker.internal:11434`  
  - *Description:* Ollama server URL.
- **`OLLAMA_QUERY_MODEL`**  
  - *Default:* `llama3.2:latest`  
  - *Description:* Model name for generating search queries.

### Crawl4AI & Token

- **`CRAWL4AI_URL`**  
  - *Default:* `http://crawl4ai:11235/`
- **`CRAWL4AI_TOKEN`**  
  - *Default:* `123456`

### Search Settings

- **`SEARXNG_URL`**  
  - *Default:* `http://host.docker.internal:8080/search?q=<query>&format=json`
- **`NUM_RESULTS`**  
  - *Default:* `5`
  - *Description:* Maximum URLs to scrape per set of queries.
- **`ENABLE_RANKING`**  
  - *Default:* `True`
  - *Description:* Enables domain-based URL ranking.
- **`MIN_RANK_THRESHOLD`**  
  - *Default:* `0.6`
  - *Description:* Scrape only URLs scoring above this threshold if ranking is on.

### Browser & Simulation

- **`HEADLESS_MODE`**  
  - *Default:* `True`
  - *Description:* Runs the browser in headless mode.
- **`USER_AGENT`**  
  - *Default:* `Mozilla/5.0`
- **`BROWSER_TYPE`**  
  - *Options:* `chromium`, `firefox`, `webkit`  
  - *Default:* `chromium`
- **`SIMULATE_USER`**  
  - *Default:* `True`
  - *Description:* Simulates human browsing (clicks, random delays).
- **`ENABLE_MAGIC_MODE`**  
  - *Default:* `True`
  - *Description:* Additional anti-detection features.

### Performance Controls

- **`TIMEOUT_SECONDS`**  
  - *Default:* `120`
- **`POLL_INTERVAL_SECONDS`**  
  - *Default:* `3`
- **`MAX_CONCURRENT_SCRAPES`**  
  - *Default:* `5`

### Content Filtering & Output

- **`CLEANUP_REGEX`**  
  - *Default:* A regex for cleaning markdown (removes menus, footers, ads, etc.).
- **`INCLUDE_IMAGES`**  
  - *Default:* `False`
  - *Description:* Whether to include images in the scraped markdown.
- **`MAX_CONTENT_LENGTH`**  
  - *Default:* `0` (No limit)
- **`CITATION_STYLE`**  
  - *Options:* `inline`, `footnote`, `endnotes`  
  - *Default:* `inline`

---

## How It Works

1. **AI Query Generation**  
   Convert your prompt into multiple semicolon-delimited search queries.
2. **Search & Rank**  
   Retrieve URLs with SearXNG and optionally rank them by domain quality.
3. **Concurrent Scraping**  
   Use Crawl4AI to scrape pages asynchronously, respecting concurrency limits.
4. **Content Processing**  
   Clean up and format the scraped markdown using regex and other filters.
5. **Compile & Cite**  
   Combine results, with optional domain-based citations or footnotes.

---

## Usage Instructions

1. **Install or Integrate**  
   Place this tool within your Open-WebUI environment.

2. **Review & Adjust Settings**  
   Override valves in `Tools.Valves` as needed (e.g., changing default endpoints or concurrency).

3. **Initiate a Research Task**  
   Provide a query, URL, or topic. For example:

   ```plaintext
   AI trends in healthcare 2025
   ```

   Or include instructions:
   ```plaintext
   AI trends in healthcare 2025. Summarize major findings and cite sources.
   ```

4. **Monitor Progress**  
   You’ll see real-time status updates about query generation, ranking, and scraping tasks.

5. **Review Final Output**  
   The tool returns cleaned markdown, often with citations or footnotes.

---

## Code Overview

- **`aiohttp` / `asyncio`:** Facilitates non-blocking HTTP requests and scraping.  
- **`pydantic`:** Manages user-configurable valves.  
- **`EventEmitter` & Hooks:** Emits real-time status updates or errors.  
- **`Crawler` (Crawl4AI Wrapper):** Coordinates asynchronous scraping tasks.  
- **`Ollama` Integration:** Creates short, refined search queries from user input.

---

## Changelog

### Version 1.0.0 (Initial Release)
- Introduced AI-based query generation with Ollama.
- Added SearXNG integration for broader search coverage.
- Enabled concurrent page scraping via Crawl4AI.
- Provided ranking capabilities to filter out low-value URLs.
- Implemented robust error handling and real-time event hooks.

---

## Contributing

All contributions, including bug reports and feature requests, are welcome. Feel free to open a pull request or issue on our [GitHub Repository](https://github.com/BrandXX/open-webui/).

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Acknowledgments

Big thanks to the Open-WebUI community and especially **focuses**, whose early code and feedback helped shape the AI Researcher tool.

---
