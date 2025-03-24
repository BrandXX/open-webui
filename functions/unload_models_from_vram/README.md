# Unload Models from VRAM

Welcome to the *Unload Models from VRAM* function for Open-WebUI! This streamlined tool effortlessly unloads all models from your VRAM using Ollama's REST API. If your VRAM feels a bit crowded, here's your chance to clear it out with a simple click—think of it as spring cleaning for your digital workspace. 😉

---

## Overview

This function performs the following:
- **Retrieves Loaded Models:** Fetches a list of models currently loaded in VRAM by calling the `/api/ps` endpoint.
- **Unloads Each Model:** Sends a POST request to `/api/generate` for each retrieved model to unload it.
- **One-click Action:** Unloads instantly upon clicking—no confirmation needed, for ultimate convenience.
- **Configurable Settings:** The endpoint URL and timeout are configurable with sensible defaults.
- **Robust Error Handling:** Provides clear emitter messages and detailed logs for troubleshooting endpoint issues, timeouts, or connection errors.

---

## Features

- **REST API Integration:** Seamlessly interacts with Ollama's API.
- **Configurable Endpoint & Timeout:** Default endpoint is `http://host.docker.internal:11434` and timeout is 3 seconds.
- **Graceful Error Handling & Emitter Feedback:** Clear, concise messages guide you if issues occur.
- **Debug Logging:** Detailed logs for diagnosing problems without overwhelming the user.
- **Asynchronous Operation:** Smoothly integrates within chat-based interactions.

---

## Configuration

- **Ollama API Endpoint:**  
  Default: `http://host.docker.internal:11434`  
  *(Change this if your setup uses a different URL.)*

- **Timeout:**  
  Default: `3` seconds  
  *(Adjust higher if needed.)*

- **Version:**  
  `1.0.1`

- **Required Open-WebUI Version:**  
  `0.3.9`

---

## How It Works

1. **Trigger Action:**  
   Click the 'Unload Models from VRAM' button near the 'Regenerate' icon to immediately initiate unloading.

2. **Retrieving Models:**  
   Makes a GET request to `{OLLAMA_ENDPOINT}/api/ps` to fetch loaded models. If none are found or an error occurs, a clear message is displayed.

3. **Unloading Process:**  
   Sends a POST request to `{OLLAMA_ENDPOINT}/api/generate` with payload `{ "model": model, "keep_alive": 0 }` for each retrieved model to unload them.

4. **Error Handling:**  
   Issues concise error messages for misconfigured endpoints, timeouts, or connection issues. Detailed logs are available for troubleshooting.

5. **Feedback:**  
   Progress updates and final results keep you informed every step of the way.

---

## Usage Instructions

1. **Click and Go:**  
   Simply click the 'Unload Models from VRAM' icon next to the 'Regenerate' button at the bottom of your chat interface.

2. **Sit Back and Relax:**  
   Models unload immediately. No confirmation is required—watch your VRAM clear instantly!

---

## Code Overview

The function is written in Python, utilizing:
- **`requests`** for HTTP communication.
- **`pydantic`** for managing configuration.
- **`asyncio`** for asynchronous operations and event-driven interactions.

Each step, from retrieving to unloading models, is handled efficiently and clearly logged for easy troubleshooting.

---

## Changelog

- **Version 1.0.1**  
  - Removed confirmation prompt for streamlined operation.
  - Maintained robust error handling and user feedback.

- **Version 1.0.0**  
  - Initial release.
  - Robust error handling with emitter feedback.
  - Detailed logging.
  - Asynchronous design for chat integration.

---

## Contributions

Contributions are warmly welcomed! For improvements, bug fixes, or feature requests, please visit the [GitHub repository](https://github.com/BrandXX/open-webui/) to submit pull requests.

*(Note: New to collaborating? No worries—let's learn and grow together!)*

---

## License

Licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

A special thank-you to the Open-WebUI community and contributors for your continuous enhancements. You're awesome!

---

Happy unloading, and enjoy your tidy VRAM! 😄🚀
