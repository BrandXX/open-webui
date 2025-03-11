# Unload Models from VRAM

Welcome to the *Unload Models from VRAM* function for Open-WebUI! This neat little tool lets you effortlessly unload all models from your VRAM using Ollama's REST API. If your VRAM is feeling a bit too crowded, this is your chance to give it a breather—just like cleaning out your digital closet. 😉

---

## Overview

This function performs the following:
- **Retrieves Loaded Models:** It fetches a list of models currently loaded in VRAM by calling the `/api/ps` endpoint.
- **Unloads Each Model:** For each model retrieved, it sends a POST request to `/api/generate` to unload it.
- **User Confirmation:** It prompts you for confirmation before proceeding—because we know you're cautious about losing your hard work.
- **Configurable Settings:** Endpoint URL and timeout are configurable, with sensible defaults that work in most scenarios.

---

## Features

- **REST API Integration:** Seamlessly interacts with Ollama's API.
- **Configurable Endpoint & Timeout:** Default endpoint is `http://host.docker.internal:11434` and timeout is 3 seconds.
- **Graceful Error Handling:** Manages timeouts and connection errors without leaving you in the dark.
- **Debug Logging:** Prints helpful debug messages to track operations.
- **Asynchronous Operation:** Uses async functions for smooth integration within chat-based interactions.

---

## Configuration

- **Ollama API Endpoint:**  
  Default: `http://host.docker.internal:11434`  
  *(Change this if your setup uses a different URL.)*

- **Timeout:**  
  Default: `3` seconds  
  *(Set to a higher value if your network is on the slower side.)*

- **Version:**  
  `1.0.0`

- **Required Open-WebUI Version:**  
  `0.3.9`

---

## How It Works

1. **User Confirmation:**  
   When you trigger the function (by clicking the 'Unload Models from VRAM' icon near the 'Regenerate' icon), you'll be prompted with a simple message. Type `yes` to confirm you want to unload all loaded models.  
   *Tip: Type "yes" like you really mean it!*

2. **Retrieving Models:**  
   The function makes a GET request to `{OLLAMA_ENDPOINT}/api/ps` to get the list of currently loaded models. If nothing is found (or if there's a hiccup), it stops right there with a clear error message.

3. **Unloading Process:**  
   For every model retrieved, a POST request is sent to `{OLLAMA_ENDPOINT}/api/generate` with the payload `{ "model": model, "keep_alive": 0 }` to unload that model.

4. **Error Handling:**  
   If there's a timeout or connection issue, the function gracefully handles it, providing you with a concise debug message so you know what went wrong (because we all need a little help sometimes!).

5. **Feedback:**  
   Progress updates and final results are communicated back to you, so you’re always in the loop.

---

## Usage Instructions

1. **Trigger the Function:**  
   Click the 'Unload Models from VRAM' icon located next to the 'Regenerate' icon at the bottom of your chat interface.

2. **Confirm Action:**  
   When prompted, type `yes` to confirm you want to unload the models.

3. **Watch the Magic:**  
   The function will retrieve the list of loaded models and proceed to unload them one by one. Sit back, relax, and enjoy the clean VRAM!

---

## Code Overview

The function is implemented in Python and leverages:
- **`requests`** for HTTP communications.
- **`pydantic`** for configuration management.
- **`asyncio`** for handling asynchronous operations and user interactions via event callbacks.

The source code is neatly organized to ensure that each step—from model retrieval to unloading—is handled efficiently, with clear debug logging for easy troubleshooting.

---

## Contributions

Contributions are more than welcome! If you have ideas, bug fixes, or improvements, feel free to fork the repository and submit a pull request. For more details, check out the [GitHub repository](https://github.com/BrandXX/open-webui/).

---

## License

This project is licensed under the MIT License. See the LICENSE file for further details.

---

## Acknowledgments

A big shout-out to the Open-WebUI community and all contributors who continuously improve the tool. Thanks for making this project as awesome as it is!

---

Happy unloading, and may your VRAM be ever light! 😄
