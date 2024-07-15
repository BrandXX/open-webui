# Convert to JSON

**Author:** BrandXX/UserX  
**Version:** 1.0.3 
**License:** MIT  

## Description
This tool converts data provided from the chat into JSON format and returns it in a markdown code block.

## Requirements
- Python 3.7+
- `json` (standard library)

## Tested with
- llama3:8b-instruct-fp16
- llava-llama3:8b-v1.1-fp16
- dolphin-mistral:7b-v2.8-fp16

## Usage
Call the function `convert_to_json(data)` with your data as an argument.  
Example:
```python
data = {"name": "John", "age": 30}
print(convert_to_json(data))
```
Set the indent level in Workspace>Tools>Valves (Gear icon next to the tool)
I will implement UserValve once I get it working.

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
- Improved the reliability of the LLM to call the Convert to JSON tool
        - Contemplating changing the name to 'JSON Tool'
- Added 'COMPACT_PRINT' Valve
        - OFF: Pretty-Printed JSON
                - Toggle 'COMPACT_PRINT' <OFF> for Pretty-Printed JSON
        - ON: Compact JSON
                - Toggle 'COMPACT_PRINT' <ON> for one Array per line
- Added 'SINGLE_LINE' Valve
        - Toggle 'SINGLE_LINE' <ON> for a single line of JSON
- Increased the reliability of the JSON output to be in a properly formatted markdown code block
- Code refactored
- Restructured and provided a more detailed and reliable instruction set to the LLM
- (Undecided Change) Contemplating changing the name from 'Convert to JSON' to 'JSON Tool''
- Updated Description

## Contact
- Email: N/A
- GitHub: https://github.com/BrandXX/open-webui

## Dependencies
- None (uses Python standard library)

## Future Enhancements
- Add GitHub Repo
- Add Documentation
- Add support for more complex data structures.
- Add support for imported datasets via RAG
- Implement validation for input data.
- Add SaveAs feature (JSON, XML, etc.)

## Additional Notes
- A citation is added as an indicator of successful execution
```python
        self.citation = (
            True  # Attribute to trigger a citation if a return occurs within the tool
        )
```
## OpenWebUI Link
<a href="https://openwebui.com/t/userx/convert_to_json/" target="_blank">Convert to JSON @OpenWebUI</a>
