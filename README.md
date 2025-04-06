# Open-WebUI Functions, Tools, Pipelines, and Instructions
------------------------------
Welcome to the repository for storing functions, tools, pipelines, and LLM instructions for Open-WebUI. This repository is dedicated to enhancing the capabilities of Open-WebUI by providing a collection of useful components and development resources.

## Overview
This repository contains:

- **Functions**: Reusable code snippets that perform specific tasks within Open-WebUI.
- **Tools**: Standalone utilities callable directly by LLMs to extend functionality (like web scraping, data retrieval).
- **Pipelines**: Automated workflows that chain together multiple components for complex processes.
- **LLM Instructions**: Documentation and examples to help LLMs understand and generate code for Open-WebUI.
- **(Coming Soon)** Setup and configuration instructions for:
  - Ollama
  - Open-WebUI
  - SearXNG
  - Caddy
  - OpenedAI Speech
  - ComfyUI
  - Crawl4AI
  - n8n
  - TailScale
  - Docker Run and Compose
  - WSL2
  - Ubuntu

## Component Directories

### Functions
Functions are internal logic units that can be reused across tools and pipelines. They encapsulate specific functionality like model management or data processing.

### Tools
Tools extend LLM capabilities with real-world interactions like retrieving weather data, stock prices, or performing web searches. Each tool has its own directory with documentation.

### Pipelines
Pipelines are sequences of integrated tools and functions designed for complex workflows, such as the SmolAgents Deep Research Pipeline.

### LLM Instructions
The `__llm_instructions` directory contains resources for AI coding assistants to understand Open-WebUI's architecture and generate appropriate code:

- Documentation templates
- Code examples
- Best practices
- Migration guides
- Metadata standards

## Usage
Each function, tool, or pipeline has its own README with detailed usage instructions. Please refer to the respective folders for specific information.

## Contributing
Contributions are welcome! If you have a function, tool, or pipeline that you think would be useful for the community, please feel free to submit a pull request. Here are some guidelines to get you started:

1. Fork the repository.

2. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

3. Make your changes and commit them:
```bash
git commit -m "Add some feature"
```

4. Push to the branch:
```bash
git push origin feature/your-feature-name
```

5. Open a pull request.

## Documentation Standards
All components should include:
- A clear README.md following the template in `__llm_instructions/documentation.md`
- Proper metadata headers for Python scripts as outlined in `__llm_instructions/metadata_header_instructions.md`
- Comprehensive inline documentation

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For any questions or suggestions, feel free to reach out via GitHub `Issues` or via Open-WebUI's Discord `@userx`.

## Open-WebUI Links
- <a href="https://openwebui.com/" target="_blank">Open-WebUI Community Site</a>
- <a href="https://docs.openwebui.com/" target="_blank">Open-WebUI Documentation</a>
- <a href="https://github.com/open-webui/open-webui" target="_blank">GitHub Repository</a>
- <a href="https://discord.gg/5rJgQTnV4s" target="_blank">Discord Community</a>

## Disclaimer
I am not a developer. My primary expertise lies in IT infrastructure, where I have served as a Senior System Administrator and currently as the IT Infrastructure Manager for a government organization. While I am involved in writing tools, functions, and pipelines for Open-WebUI, it is important to understand that software development is not my primary area of expertise.
