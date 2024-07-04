# Convert to JSON

**Author:** BrandXX/UserX  
**Version:** 1.0.1  
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

## Changelog
#### Version 1.0.1
- Added comments to the function for better understanding.
- Added GitHub Repo (under construction)
#### Version 1.0
- Initial release.

## Contact
- Email: N/A
- GitHub: https://github.com/BrandXX/open-webui

## Dependencies
- None (uses Python standard library)

## Future Enhancements
- Add GitHub repo
- Add Documentation
- Add support for more complex data structures.
- Implement validation for input data.
- Add SaveAs feature

## Additional Notes
- A citation is added as an indicator of successful execution
```python
        self.citation = (
            True  # Attribute to trigger a citation if a return occurs within the tool
        )
```

