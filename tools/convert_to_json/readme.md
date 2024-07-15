# Convert to JSON

**Author:** BrandXX/UserX  
**Version:** 1.0.3 
**License:** MIT  

## Description
This tool converts data provided within the chat into JSON format and returns it in a markdown code block.

## Requirements
- Open-WebUI v0.3.5 or above
- No additional Python libraries required
  - Only uses `json` (standard library)
- Ability to follow instructions
- **Note:** I have only tested locally with Ollama

## Tested with
- **llama3:8b-instruct-fp16**
  - Most Reliable
- **llava-llama3:8b-v1.1-fp16**
  - Fairly reliable
- **dolphin-mistral:7b-v2.8-fp16**
  - Hit and miss
- All testing done with models served locally through an Ollama inference server.
- I have not tested on any externally hosted models or services.


## Usage
#### Import the Latest Stable Release
  - https://openwebui.com/t/userx/convert_to_json/

#### Latest Dev (Copy and Paste method)
  - [tools/convert_to_json/1.0.3/convert_to_json.yaml](https://github.com/BrandXX/open-webui/tree/main/tools/convert_to_json/DEV)

#### Instructions
- Import the tool from the link above or copy and paste the code from the repo.
- Set the indent level, and JSON format.
  - Workspace>Tools>Valves
    - `Gear` icon on the right side of the tool.
- Toggle `Convert to JSON` tool on within the chat window `+` menu.
- Start the chat with `Convert to JSON` followed by the data to be converted to JSON.
- EXAMPLE:
```text
Convert to JSON

id,name,age,email,phone,address,join_date,salary,department
1,John Doe,28,johndoe@example.com,555-1234,123 Maple St,2021-05-15,55000,Engineering
2,Jane Smith,34,janesmith@example.com,555-5678,456 Oak St,2019-03-22,65000,Marketing
3,Bob Johnson,45,bobjohnson@example.com,555-8765,789 Pine St,2018-11-30,72000,Sales
4,Alice Williams,23,alicewilliams@example.com,555-4321,135 Cedar St,2022-07-19,48000,Support
5,Charlie Brown,31,charliebrown@example.com,555-9876,246 Birch St,2020-01-10,61000,HR
```
- Optional, for models that do not reliably call the tool or have difficulty rendering the JSON, I have had some success using the following system prompt.
  - SYSTEM PROMPT:
```text
You are an expert at calling tools and functions to accomplish the task at hand. You should read and follow all instructions before returning anything to the user. Please be sure that you read and follow the instructions to ensure quality and accuracy.
```
  
- The System Prompt can be accessed on a per chat basis through the `Controls` menu located at the top right of the chat window.
- On occasion, you may need to resubmit the data to get the proper responce.
  - If you do not see a citation when the LLM returns the JSON, then the tool was not called.
- There are currently no `UserValves`.
  - I will implement `UserValve` once I get it working.

## Changelog
#### Version 1.0.3
- Added JSON_INDENT Valve with a default indent value of 4
- Refactored the code
- Added descriptive and useful comments (Feel free to delete the comments to preserve context length)
- Added pydantic and Logging librabries
- Enabled Logging
- Thanks to @atgehrhardt, an editor over at https://openwebui.com for the suggestions.
#### Version 1.0.2
- Removed Examples as they are not relevant
#### Version 1.0.1
- Added comments to the function for better understanding.
- Added GitHub Repo (under construction)
#### Version 1.0
- Initial release.

## Development
#### Version 1.0.4-RC
- Improved the reliability of the LLM to call the 'Convert to JSON' tool
  - Contemplating changing the name to 'JSON Tool'
- Added 'COMPACT_PRINT' Valve
  - OFF: Pretty-Printed JSON
    - Toggle 'COMPACT_PRINT' Valve OFF for Pretty-Printed JSON
  - ON: Compact JSON
    - Toggle 'COMPACT_PRINT' Valve ON for one Array per line
- Added 'SINGLE_LINE' Valve
  - Toggle 'SINGLE_LINE' Valve ON for a single line of JSON
- Increased the reliability of the JSON output to be in a properly formatted markdown code block
- Refactored code to increase dependability and reliability
- Restructured and provided a more detailed and reliable instruction set to the LLM
- (Undecided Change) Contemplating changing the name from 'Convert to JSON' to 'JSON Tool''
- Updated Description
- Updated documentation at https://github.com/BrandXX/open-webui/edit/main/tools/convert_to_json/readme.md

## Contact
- Email: N/A
- GitHub: https://github.com/BrandXX/open-webui

## Dependencies
- None (uses Python standard library)

## Future Enhancements
- (DONE) Add GitHub Repo
- (DONE) Add Documentation
- Add support for more complex data structures.
- Add support for imported datasets via RAG
- Implement validation for input data.
- Add SaveAs feature (JSON, XML, etc.)
- (IN-PROGRESS) Add new Valves and UserValves to extend the features and capabilities.

## Additional Notes
- A citation is added as an indicator of successful execution
  - If you do not see a citation when the LLM returns the JSON, then the tool was not called.
```python
        self.citation = (
            True  # Attribute to trigger a citation if a return occurs within the tool
        )
```
## OpenWebUI Links
- <a href="https://openwebui.com/t/userx/convert_to_json/" target="_blank">Convert to JSON @OpenWebUI</a>
- <a href="https://openwebui.com/" target="_blank">Open-WebUI Community Site @OpenWebUI</a>
- <a href="https://docs.openwebui.com/" target="_blank">Open-WebUI Docs @OpenWebUI</a>
- <a href="https://github.com/open-webui/open-webui" target="_blank">GitHub @OpenWebUI</a>
- <a href="https://discord.gg/5rJgQTnV4s" target="_blank">Discordb @OpenWebUI</a>

## Disclaimer
I am not a developer. My primary expertise lies in IT infrastructure, where I have served as a Senior System Administrator and currently as the IT Infrastructure Manager for a government organization. While I am involved in writing tools, functions, and pipelines for Open-WebUI, it is important to understand that software development is not my primary area of expertise.
