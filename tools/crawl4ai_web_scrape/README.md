# Crawl4AI Web Scraper

Welcome to the *Crawl4AI Web Scraper* tool for Open-WebUI! This robust, configurable tool lets you scrape web pages effortlessly, complete with advanced settings and safety measures to handle various scraping scenarios. Ready to dive in and harvest some data securely and efficiently? Let's get scraping!

---

## Overview

This tool performs the following:
- **Asynchronous Web Scraping:** Leverages `aiohttp` and `asyncio` to fetch content without blocking your workflow.
- **Real-Time Updates:** Emits progress messages at every key stage—start, waiting, finished, and error.
- **Configurable Options:** Use CSS selectors or overrides to pinpoint exactly what you need.
- **Browser Simulation:** Emulates real user behavior with headless mode, configurable browser types, and advanced anti-detection measures.
- **Content Filtering:** Cleans your markdown output with customizable regex filters.
- **Link Management:** Optionally skip internal links and exclude external or social media links for tidier results.
- **Enhanced Stability:** Improved error handling, memory management, and safety checks.

---

## Features

- **Asynchronous Operation:** Smooth, non-blocking scraping powered by `aiohttp` and `asyncio`.
- **Advanced Configurations:** Adjust connection details, browser behavior, scraping parameters, and regex safely.
- **Emitter Feedback:** Clear real-time updates during scraping.
- **Advanced Content Filtering:** Efficient removal of unwanted content.
- **Memory Safety:** Limits markdown size to prevent excessive memory usage.
- **Error Handling:** Graceful management of errors, including regex and network issues.
- **User-Friendly Setup:** Copy-paste into your Open-WebUI installation—no extra setup required.

---

## Configuration

- **Connection & Authentication Settings:**  
  - **CRAWL4AI_URL:**  
    Default: `http://crawl4ai:11235/`
  - **CRAWL4AI_TOKEN:**  
    Default: `123456`

- **Scraping Options:**  
  - **CSS_SELECTOR:**  
    Choose from predefined selectors (`main`, `article`, `div.prose`, etc.)
  - **CSS_SELECTOR_OVERRIDE:**  
    Custom selector overriding defaults.

- **Browser & Simulation Settings:**  
  - **HEADLESS_MODE:**  
    Default: `True`
  - **USER_AGENT:**  
    Default: `Mozilla/5.0`
  - **BROWSER_TYPE:**  
    Options: `chromium`, `firefox`, `webkit` (Default: `chromium`)
  - **SIMULATE_USER:**  
    Default: `True`
  - **ENABLE_MAGIC_MODE:**  
    Default: `True`
  - **OVERRIDE_NAVIGATOR:**  
    Default: `True`

- **Performance & Timing Controls:**  
  - **TIMEOUT_SECONDS:**  
    Default: `120`
  - **POLL_INTERVAL_SECONDS:**  
    Default: `3`

- **Content Filtering & Output Options:**  
  - **CLEANUP_REGEX:**  
    Default regex to clean markdown.
  - **INCLUDE_IMAGES:**  
    Default: `False`
  - **MAX_CONTENT_LENGTH:**  
    Default: `0` (No limit)
  - **SKIP_INTERNAL_LINKS:**  
    Default: `False`
  - **EXCLUDE_EXTERNAL_LINKS:**  
    Default: `False`
  - **EXCLUDE_SOCIAL_MEDIA_LINKS:**  
    Default: `False`

---

## How It Works

1. **Initiate Scraping:**  
   Sends an asynchronous request to Crawl4AI server with your URL.

2. **Progress Updates:**  
   Real-time status:
   - *Start*: Scraping initiated.
   - *Waiting*: Processing ongoing.
   - *Finished*: Scraping complete.
   - *Error*: Detailed error messages.

3. **Content Processing:**  
   Applies filters and link rules.

4. **Output:**  
   Returns cleaned markdown.

---

## Usage Instructions

1. **Copy & Paste:**  
   Paste into Open-WebUI.

2. **Customize (Optional):**  
   Adjust settings in the `Tools.Valves` class.

3. **Trigger Tool:**  
   Provide URL and initiate scraping.

4. **Monitor Progress:**  
   Watch real-time emitter updates.

---

## Code Overview

The tool is implemented in Python using:
- **`aiohttp` & `asyncio`:** Asynchronous scraping.
- **`pydantic`:** Configurable settings.
- **Custom Event Hooks:** Real-time feedback.
- **Enhanced Safety:** Improved error handling and memory safeguards.

---

## Changelog

- **Version 1.1.0**
  - Improved resource cleanup and memory management.
  - Added markdown content size limitations.
  - Enhanced regex error handling and stability.
  - Added safety checks for empty markdown retrieval.
  - Updated emitter to provide detailed error messages.

- **Version 1.0.0**
  - Initial asynchronous scraping release.
  - Basic configuration and emitter feedback.

---

## Contributions

Contributions welcome! Fork and submit pull requests at [GitHub](https://github.com/BrandXX/open-webui/). Help us improve this tool further!

---

## License

Licensed under the MIT License. See LICENSE file for details.

---

## Acknowledgments

Thanks to the Open-WebUI community, especially **focuses**, for initial inspiration. Your support makes this tool shine!

---

Happy scraping, and may your data be ever abundant! 😄
