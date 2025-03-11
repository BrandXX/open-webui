# Crawl4AI Web Scrape

Welcome to the *Crawl4AI Web Scrape* tool for Open-WebUI! This nifty tool lets you scrape web pages with ease, giving you plenty of configurable valves to fine-tune your scraping adventure. Ready to dive in and harvest some data? Let’s get scraping!

---

## Overview

This tool performs the following:
- **Asynchronous Web Scraping:** Leverages `aiohttp` and `asyncio` to fetch content without blocking your workflow.
- **Real-Time Updates:** Emits progress messages at every key stage—start, waiting, finished, and error.
- **Configurable Options:** Use CSS selectors or overrides to pinpoint exactly what you need.
- **Browser Simulation:** Emulates real user behavior with headless mode, configurable browser types, and anti-detection measures.
- **Content Filtering:** Cleans up your markdown output with customizable regex filters.
- **Link Management:** Optionally skip internal links and exclude external or social media links for a tidier result.

---

## Features

- **Asynchronous Operation:** Enjoy smooth, non-blocking scraping powered by `aiohttp` and `asyncio`.
- **Customizable Settings:** Easily adjust connection details, browser behavior, and scraping parameters.
- **Detailed Emitter Feedback:** Stay informed with clear progress updates during the scraping process.
- **Advanced Content Filtering:** Remove unwanted text elements with a pre-configured regex.
- **User-Friendly:** Simply copy and paste the code into a new tool in your Open-WebUI installation—no cloning or extra setup required!

---

## Configuration

- **Connection & Authentication Settings:**  
  - **CRAWL4AI_URL:**  
    Default: `http://crawl4ai:11235/`
  - **CRAWL4AI_TOKEN:**  
    Default: `123456`

- **Scraping Options:**  
  - **CSS_SELECTOR:**  
    Choose from options like `main`, `article`, `div.prose`, etc.
  - **CSS_SELECTOR_OVERRIDE:**  
    A custom selector that takes precedence over the default selection.

- **Browser & Simulation Settings:**  
  - **HEADLESS_MODE:**  
    Default: `True` (Runs the browser in headless mode)
  - **USER_AGENT:**  
    Default: `Mozilla/5.0`
  - **BROWSER_TYPE:**  
    Options: `chromium`, `firefox`, or `webkit` (Default: `chromium`)
  - **SIMULATE_USER:**  
    Default: `True`
  - **ENABLE_MAGIC_MODE:**  
    Default: `True` (Enables advanced anti-detection features)
  - **OVERRIDE_NAVIGATOR:**  
    Default: `True`

- **Performance & Timing Controls:**  
  - **TIMEOUT_SECONDS:**  
    Default: `120` seconds
  - **POLL_INTERVAL_SECONDS:**  
    Default: `3` seconds

- **Content Filtering & Output Options:**  
  - **CLEANUP_REGEX:**  
    Default regex to remove unwanted markdown artifacts.
  - **INCLUDE_IMAGES:**  
    Default: `False`
  - **MAX_CONTENT_LENGTH:**  
    Default: `0` (No length limit)
  - **SKIP_INTERNAL_LINKS:**  
    Default: `False`
  - **EXCLUDE_EXTERNAL_LINKS:**  
    Default: `False`
  - **EXCLUDE_SOCIAL_MEDIA_LINKS:**  
    Default: `False`

---

## How It Works

1. **Initiate Scraping:**  
   When you trigger the tool, it sends an asynchronous request to the Crawl4AI server to start scraping your specified URL.

2. **Progress Updates:**  
   The tool emits messages to keep you in the loop:
   - *Start:* Notification that the scraping has begun.
   - *Waiting:* Updates while the tool is processing your request.
   - *Finished:* Confirmation that scraping is complete.
   - *Error:* Clear error messages if something goes awry.

3. **Content Processing:**  
   After scraping, the tool processes the returned markdown, applying your custom filters and link management rules.

4. **Output:**  
   The final, cleaned markdown output is returned for you to use in your projects.

---

## Usage Instructions

1. **Copy & Paste:**  
   Simply copy the Crawl4AI Web Scrape code into a new tool within your Open-WebUI installation.

2. **Customize (Optional):**  
   Adjust the settings in the `Tools.Valves` class to perfectly suit your scraping needs.

3. **Trigger the Tool:**  
   Provide the URL you wish to scrape and let the tool work its magic.

4. **Monitor Progress:**  
   Keep an eye on the emitter messages for real-time updates during the scraping process.

---

## Code Overview

The tool is implemented in Python and leverages:
- **`aiohttp` & `asyncio`:** For asynchronous web scraping.
- **`pydantic`:** For managing configurable settings.
- **Custom Event Hooks:** To provide clear, real-time progress feedback.

---

## Changelog

- **Version 1.0.0**  
  - Initial release with comprehensive asynchronous scraping functionality.
  - Configurable options for browser simulation and content filtering.
  - Real-time emitter feedback for seamless user experience.

---

## Contributions

Contributions are always welcome! If you have ideas, bug fixes, or improvements, feel free to fork the repository and submit a pull request via the [GitHub repository](https://github.com/BrandXX/open-webui/). We love seeing new ideas to make this tool even better!

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

A huge thanks to the Open-WebUI community and especially to **focuses** for the initial code inspiration. Your contributions and support make this tool shine!

---

Happy scraping, and may your data be ever abundant! 😄
