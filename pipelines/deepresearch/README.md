# SmolAgents Deep Research Pipeline (Currently in Development)

**A pipeline designed to initiate and manage comprehensive research tasks using Open-WebUI integrated with SmolAgents.**

---

## Version
**1.0.0** - Initial release of the pipeline.

---

## Overview
The SmolAgents Deep Research Pipeline provides an advanced, easy-to-configure environment for deep internet research, web browsing, document inspection, and visual Q&A tasks. Leveraging powerful OpenAI models and intuitive web tools, this pipeline is perfect for researchers, developers, and AI enthusiasts.

---

## Features
- **Web Browsing:** Navigate, search, and analyze online content directly through automated agents.
- **Document Analysis:** Inspect and summarize textual and document-based content effectively.
- **Tool Integration:** Integrated browser tools include web-page navigation, archival search, text extraction, and more.
- **Model Flexibility:** Supports various OpenAI-compatible models through configurable API settings.

---

## Prerequisites
- Docker
- Docker Compose (optional but recommended)
- API Keys (OpenAI, SERP API, Bing, Google Search API)

---

## Installation
### Docker Containers (Currently in Development, invite only)
Pull the container using Docker:

```sh
docker pull ghcr.io/brandxx/pipelines:main
```
Or for GPU-enabled systems:

```sh
docker pull ghcr.io/brandxx/pipelines:latest-cuda
```

---

## Configuration
Configure pipeline variables by setting environment variables or editing directly in your pipeline's settings:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `SERPAPI_API_KEY` *(optional)*
- `BING_API_KEY` *(optional)*
- `GOOGLE_API_KEY` *(optional)*
- `GOOGLE_API_CX` *(optional)*

Example `.env` file:
```dotenv
OPENAI_BASE_URL="https://api.openai.com/v1"
OPENAI_API_KEY="your-api-key"
OPENAI_MODEL="gpt-4"
SERPAPI_API_KEY="your-serpapi-key"
BING_API_KEY="your-bing-key"
GOOGLE_API_KEY="your-google-api-key"
GOOGLE_API_CX="your-google-cx"
```

---

## Usage
Run your pipeline within Open-WebUI by selecting **SmolAgents Deep Research Pipeline** and entering your prompts. The pipeline automates complex research tasks, providing summarized results from multiple sources.

---

## Changelog
- **Version 1.0.0**
  - Initial release.
  - Integrated web browsing tools.
  - Document inspection and analysis.
  - Configurable model and API settings.

---

## Acknowledgements
Special thanks to [elabbarw](https://github.com/elabbarw) and the Open-WebUI community for the original repository. You can find the original implementation [here](https://github.com/elabbarw/aiagent_playground/tree/main/openwebui/pipelines/deepresearch).

---

## Contributing
Feel free to open issues, fork this repository, and submit pull requests. Contributions are highly welcome!

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Happy researching!* 🚀🔍✨

