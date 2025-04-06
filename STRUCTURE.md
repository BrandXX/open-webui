# Repository Structure

This document provides an overview of the Open-WebUI repository structure, explaining the organization and purpose of each directory.

## Root Directory

```
/
├── __llm_instructions/     # Resources for AI coding assistants
├── functions/              # Reusable code components
├── pipelines/              # Automated workflows
├── tools/                  # Standalone utilities
├── LICENSE.md              # License information
├── README.md               # Project overview
└── STRUCTURE.md            # This file
```

## __llm_instructions

This directory contains resources for AI coding assistants to understand Open-WebUI's architecture and generate appropriate code.

```
/__llm_instructions/
├── __functions/            # Examples and documentation for functions
├── __prompts/              # Prompt templates and examples
├── __tools/                # Documentation for tool development
├── api_reference.md        # Comprehensive API reference documentation
├── component_templates.md  # Templates and examples for components
├── integration_patterns.md # Integration patterns and best practices
├── technical_architecture.md # System architecture overview
├── troubleshooting_guide.md # Solutions for common issues
└── README.md               # Overview of LLM instructions
```

Purpose: Provides documentation templates, code examples, best practices, migration guides, and metadata standards for AI assistants. The directory also includes comprehensive guides for developing Functions, Tools, and Pipelines, including technical architecture, API references, component templates, integration patterns, and troubleshooting information.

## functions

This directory contains reusable code components that can be integrated into tools and pipelines.

```
/functions/
├── unload_models_from_vram/    # Function to free up VRAM
│   ├── 1.0.0/                  # Version 1.0.0
│   ├── 1.0.1/                  # Version 1.0.1
│   └── README.md               # Documentation for this function
└── [future functions]/         # Additional functions will be added here
```

Purpose: Houses internal logic units that can be reused across tools and pipelines, encapsulating specific functionality like model management or data processing.

## pipelines

This directory contains automated workflows that chain together multiple components for complex processes.

```
/pipelines/
├── deepresearch/               # Deep research pipeline
│   ├── 1.0.0/                  # Version 1.0.0
│   └── README.md               # Documentation for this pipeline
└── [future pipelines]/         # Additional pipelines will be added here
```

Purpose: Provides sequences of integrated tools and functions designed for complex workflows.

## tools

This directory contains standalone utilities that can be called directly by LLMs to extend functionality.

```
/tools/
├── ai_researcher/              # AI research assistant tool
│   ├── 1.0.0/                  # Version 1.0.0
│   ├── dev/                    # Development version
│   └── README.md               # Documentation for this tool
├── convert_to_json/            # Tool for converting data to JSON
│   ├── 1.0.1/                  # Version 1.0.1
│   ├── 1.0.2/                  # Version 1.0.2
│   ├── 1.0.3/                  # Version 1.0.3
│   ├── 1.0.4/                  # Version 1.0.4
│   └── DEV/                    # Development version
├── crawl4ai_web_scrape/        # Web scraping tool
│   ├── 1.0.0/                  # Version 1.0.0
│   ├── 1.1.0/                  # Version 1.1.0
│   ├── 1.4.1/                  # Version 1.4.1
│   ├── dev/                    # Development version
│   └── README.md               # Documentation for this tool
└── [future tools]/             # Additional tools will be added here
```

Purpose: Extends LLM capabilities with real-world interactions like retrieving data, performing web searches, or processing information.

## Versioning

Each component follows a versioning system:
- Components are organized in directories by name
- Each version is contained in a subdirectory (e.g., `1.0.0`, `1.1.0`)
- Development versions are typically in a `dev` or `DEV` directory
- Each component has its own README.md with detailed documentation

## Documentation Standards

All components should include:
- A clear README.md following the template in `__llm_instructions/documentation.md`
- Proper metadata headers for Python scripts as outlined in `__llm_instructions/metadata_header_instructions.md`
- Comprehensive inline documentation

## Contributing

When contributing new components:
1. Follow the existing directory structure
2. Create appropriate version directories
3. Include comprehensive documentation
4. Follow the contribution guidelines in the main README.md
