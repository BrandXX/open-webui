# AI Researcher (Previously known as 'Crawl4AI Web Scraper')

Welcome to the *AI Researcher* tool for Open-WebUI! This advanced, robust, and configurable web scraping tool integrates Crawl4AI, Ollama, and SearXNG to perform targeted research tasks effortlessly. From generating precise search queries to concurrent scraping and content ranking, AI Researcher securely handles complex research workflows. Ready to level up your web scraping? Let's dive in!  
  - Notes
     - This tool is being ranmed and branded as "AI Researcher". Further iterations of this tool are going to be located under 'AI Researcher' within the tools section.

---

## Overview

AI Researcher performs the following:
- **Query Generation:** Uses Ollama to create precise, targeted search queries from user input.
- **SearXNG Integration:** Performs searches and collects high-quality URLs.
- **Concurrent Web Scraping:** Leverages Crawl4AI with asynchronous scraping for efficient data retrieval.
- **Ranking & Filtering:** Ranks search results to prioritize quality content.
- **Real-Time Updates:** Provides continuous feedback on scraping progress, errors, and completion status.
- **Advanced Browser Simulation:** Simulates realistic browsing with headless operation and anti-detection measures.
- **Content Cleaning & Formatting:** Uses advanced regex and customizable rules to clean markdown outputs.

---

## Features

- **Asynchronous Efficiency:** Rapid scraping without workflow interruptions.
- **AI-Powered Queries:** Ollama-generated queries improve search relevance.
- **Advanced Ranking:** Intelligent URL ranking prioritizes high-quality sources.
- **Customizable Valves:** Tailor scraping and output parameters easily.
- **Memory & Stability Optimizations:** Safeguards prevent excessive memory usage and gracefully handle errors.
- **Real-Time Feedback:** Instant updates keep you informed throughout.

---

## Configuration

- **Ollama Settings:**
  - `OLLAMA_ENDPOINT`: Default: `http://host.docker.internal:11434`
  - `OLLAMA_QUERY_MODEL`: Default: `llama3.2:latest`
    - (specifically used for generating search queries)

- **Connection Settings:**  
  - `CRAWL4AI_URL`: Default: `http://crawl4ai:11235/`
  - `CRAWL4AI_TOKEN`: Default: `123456`

- **Search Settings:**
  - `SEARXNG_URL`: Default: `http://host.docker.internal:8080/search?q=<query>&format=json`
  - `NUM_RESULTS`: Default: `5`
  - `ENABLE_RANKING`: Default: `True`
  - `MIN_RANK_THRESHOLD`: Default: `0.6`

- **Browser & Simulation Settings:**  
  - `HEADLESS_MODE`: Default: `True`
  - `USER_AGENT`: Default: `Mozilla/5.0`
  - `BROWSER_TYPE`: Options: `chromium`, `firefox`, `webkit` (Default: `chromium`)
  - `SIMULATE_USER`: Default: `True`
  - `ENABLE_MAGIC_MODE`: Default: `True`

- **Performance Controls:**  
  - `TIMEOUT_SECONDS`: Default: `120`
  - `POLL_INTERVAL_SECONDS`: Default: `3`
  - `MAX_CONCURRENT_SCRAPES`: Default: `5`

- **Content Filtering & Output:**
  - `CLEANUP_REGEX`: Default regex provided for markdown cleaning.
  - `INCLUDE_IMAGES`: Default: `False`
  - `MAX_CONTENT_LENGTH`: Default: `0` (No limit)
  - `CITATION_STYLE`: Options: `inline`, `footnote`, `endnotes` (Default: `inline`)

---

## How It Works

1. **Query Generation:** AI-generated search queries ensure targeted content retrieval.
2. **Search & Ranking:** Collects and ranks URLs from SearXNG based on quality.
3. **Concurrent Scraping:** Efficiently scrapes multiple URLs simultaneously.
4. **Content Processing:** Cleans and formats scraped markdown.
5. **Result Compilation:** Provides markdown with optional source citations.

---

## Usage Instructions

1. **Installation:** Copy and paste into your Open-WebUI environment.

2. **Customization (Optional):** Adjust valves in the `Tools.Valves` class.

3. **Initiate Research:** Enter a query or URL.

### Example Requests:

- **Simple query:**
```
AI trends in healthcare 2025
```

- **Detailed Instructions:**
```
AI trends in healthcare 2025
Summarize major findings and include citations.
```

- **Specific URL with Formatting:**
```
https://example.com/research-paper
Extract key points as bullet points and cite sources.
```

4. **Monitor Progress:** Receive live status updates as scraping progresses.

---

## Code Overview

Built with:
- **`aiohttp` & `asyncio`:** Async web requests.
- **`pydantic`:** Easy configuration management.
- **Custom Event Hooks:** Real-time progress updates.
- **Advanced Safety Measures:** Robust error handling and memory controls.

---

## Changelog

- **Version 1.4.1**
  - Integrated Ollama for query generation.
  - Added URL ranking
  - Strengthened the SearXNG integration.
  - Improved concurrent scraping and memory optimization.
  - Changed name due to functionality changes
  - Clearly marked code for easy updating

- **Version 1.1.0**
  - Improved resource management and stability.
  - Added markdown content limits and error checks.

- **Version 1.0.0**
  - Initial async scraping capabilities.

---

## Contributions

We welcome your contributions! Submit pull requests or issues on [GitHub](https://github.com/BrandXX/open-webui/).

---

## License

Licensed under MIT. See LICENSE for details.

---

## Acknowledgments

Special thanks to the Open-WebUI community and **focuses** for their foundational support and inspiration!

---

Happy researching, and may your findings be ever insightful! 🚀😄

