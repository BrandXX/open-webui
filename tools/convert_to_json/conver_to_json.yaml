import json


class Tools:
    def __init__(self):
        self.citation = (
            True  # Attribute to trigger a citation if a return occurs within the tool
        )

    def convert_to_json(self, data) -> str:
        """
        Convert provided data to JSON format and return it with an instruction to format it in a ```code block```.

        :param data: The data to be converted to JSON. This can be a dictionary or any serializable Python object.
        :return: The JSON string with an instruction for the LLM to format it in a ```code block```.
        """
        try:
            # Convert the provided data to a JSON string with indentation for readability
            json_data = json.dumps(data, indent=4)
            # Instruction for the LLM to format the JSON in a code block
            instruction = "Please format the following JSON in a ```code block```:\n\n"
            formatted_json = instruction + json_data
            print(f"Debug: Instruction with JSON data:\n{formatted_json}")
            return formatted_json
        except (TypeError, ValueError) as e:
            # Return an error message if the data cannot be converted to JSON
            error_message = f"Error converting data to JSON: {str(e)}"
            print(f"Debug: Error message:\n{error_message}")
            return error_message


# Example usage:
tools = Tools()
data = {"name": "John Doe", "age": 30, "city": "New York"}
output = tools.convert_to_json(data)
print(output)
