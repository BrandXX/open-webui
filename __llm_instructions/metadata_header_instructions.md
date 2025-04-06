```markdown
# 🛠️ Metadata Header Instructions for Open-WebUI Python Scripts

All Python scripts created for Open-WebUI (**Tools**, **Functions**, and **Pipelines**) must include a standardized metadata header at the top of each script for easy identification, documentation, and maintainability.

---

## 🔖 Metadata Format

Include the following metadata header at the very top of each Python script:

```python
"""
title: [Descriptive Title of Tool/Function/Pipeline]
description: [Concise yet detailed summary of what the script does and its key capabilities.]
author: [Your GitHub Username or Name]
author_url: [Link to your GitHub Profile or Organization Page]
funding_url: [Optional: Link to your funding or sponsorship page]
repo_url: [Direct Link to edit/view this specific script on GitHub]
version: [Semantic versioning, e.g., 1.0.0]
required_open_webui_version: [Minimum compatible version of Open-WebUI, e.g., 0.3.9]

Notes:
[Optional notes or acknowledgments. Mention contributions, community members, or related resources.]

[Name of Tool/Function/Pipeline]
-------------------------
Briefly summarize key integrations, workflow, and features here in bullet points:

Key functionality:
- [Feature or integration #1]
- [Feature or integration #2]
- [Feature or integration #3]
"""
```

---

## 🚩 Example Metadata Header

Here’s an example using the AI Researcher Tool:

```python
"""
title: AI Researcher
description: A powerful web scraping and research tool integrating Crawl4AI, Ollama, and SearXNG to deliver high-quality, ranked content. Generates targeted search queries, fetches relevant URLs, and scrapes content concurrently with robust anti-detection features and customizable extraction settings.
author: BrandXX/UserX
author_url: https://github.com/BrandXX/open-webui/
funding_url: https://github.com/BrandXX/open-webui/
repo_url: https://github.com/BrandXX/open-webui/edit/main/tools/crawl4ai_web_scrape/
version: 1.0.0
required_open_webui_version: 0.3.9

Notes:
Thanks to 'focuses' over at the Open-WebUI community for providing the initial code @ https://openwebui.com/t/focuses/crawl4ai_web_scrape

AI Researcher Tool
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
```

---

## ✅ **Instructions for the Local Coding LLM**

When creating or modifying Python scripts for Tools, Functions, or Pipelines within Open-WebUI:

- **Always insert this metadata header** at the top of every Python script.
- Clearly and succinctly describe the tool's purpose and capabilities.
- Include accurate links to relevant GitHub repositories or user profiles.
- Keep the version updated following semantic versioning principles.
- Always verify and update the compatibility with the specified Open-WebUI version.

🎯 *Consistently including metadata enhances readability, maintainability, and community collaboration!*

```
