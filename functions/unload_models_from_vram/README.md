# Unload Models from VRAM

Welcome to the *Unload Models from VRAM* function for Open-WebUI! This neat little tool lets you effortlessly unload all models from your VRAM using Ollama's REST API. If your VRAM is feeling a bit too crowded, this is your chance to give it a breather—just like cleaning out your digital closet. 😉

---

## Overview

This function performs the following:
- **Retrieves Loaded Models:** Fetches a list of models currently loaded in VRAM by calling the `/api/ps` endpoint.
- **Unloads Each Model:** Sends a POST request to `/api/generate` for every retrieved model to unload it.
- **User Confirmation:** Prompts for confirmation before proceeding—because we know you're cautious about losing your hard work.
- **Configurable Settings:** The endpoint URL and timeout are configurable with sensible defaults.
- **Robust Error Handling:** Provides clear, concise emitter messages (and detailed logs for developers) when issues occur, such as misconfigured endpoints, timeouts, or connection errors.

---

## Features

- **REST API Integration:** Seamlessly interacts with Ollama's API.
- **Configurable Endpoint & Timeout:** Default endpoint is `http://host.docker.internal:11434` and timeout is 3 seconds.
- **Graceful Error Handling & Emitter Feedback:** Clear, one-line messages guide you when issues occur.
- **Debug Logging:** Detailed logs help diagnose problems without overwhelming the end user.
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
   When you trigger the function (by clicking the 'Unload Models from VRAM' icon near the 'Regenerate' icon), you'll be prompted with a message. Type `yes` to confirm you want to unload all loaded models.  
   *Tip: Type "yes" like you really mean it!*

2. **Retrieving Models:**  
   The function makes a GET request to `{OLLAMA_ENDPOINT}/api/ps` to fetch the list of loaded models. If nothing is found or an error occurs, it stops and displays a clear error message.

3. **Unloading Process:**  
   For each retrieved model, a POST request is sent to `{OLLAMA_ENDPOINT}/api/generate` with the payload `{ "model": model, "keep_alive": 0 }` to unload that model.

4. **Error Handling:**  
   If issues occur (such as a misconfigured endpoint, timeout, or connection error), the function emits a concise error message—like:  
   > "Endpoint error: The URL '...' appears to be misconfigured."  
   This guides you to quickly resolve the issue, while detailed logs are kept for troubleshooting.

5. **Feedback:**  
   Progress updates and final results are communicated back to you, so you’re always in the loop.

---

## Usage Instructions

1. **Trigger the Function:**  
   Click the 'Unload Models from VRAM' icon located next to the 'Regenerate' icon at the bottom of your chat interface.

2. **Confirm Action:**  
   When prompted, type `yes` to confirm you want to unload the models.

3. **Watch the Magic:**  
   The function retrieves the list of loaded models and proceeds to unload them one by one. Sit back, relax, and enjoy the clean VRAM!

---

## Code Overview

The function is implemented in Python and leverages:
- **`requests`** for HTTP communications.
- **`pydantic`** for configuration management.
- **`asyncio`** for handling asynchronous operations and user interactions via event callbacks.

The source code is neatly organized to ensure that each step—from model retrieval to unloading—is handled efficiently, with clear debug logging for easy troubleshooting.

---

## Changelog

- **Version 1.0.0**  
  - Initial release.
  - Robust error handling with clear emitter feedback.
  - Detailed logging for troubleshooting.
  - Asynchronous operation for seamless integration in chat-based interfaces.

---

## Contributions 
Contributions are more than welcome! If you have ideas, bug fixes, or improvements, feel free to fork the repository and submit a pull request. For more details, check out the [GitHub repository](https://github.com/BrandXX/open-webui/).  

- Note: I normally run solo so I'm new to contributors. If you’re interested in contributing, please bear with me while I work to better understand the process.


---

## License

This project is licensed under the MIT License. See the LICENSE file for further details.

---

## Acknowledgments

A big shout-out to the Open-WebUI community and all contributors who continuously improve the tool. Thanks for making this project as awesome as it is!

---

Happy unloading, and may your VRAM be ever light! 😄
