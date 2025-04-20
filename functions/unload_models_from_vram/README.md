# Unload Models from VRAM

Welcome to the **Unload Models from VRAM** function for Open‑WebUI!  
This streamlined tool effortlessly frees GPU memory by unloading models through Ollama’s REST API. If your VRAM feels a bit crowded, clear it out with a single click—think of it as spring‑cleaning for your digital workspace. 😉

---

## Overview

This function:

- **Retrieves Loaded Models:** Calls `/api/ps` to list models currently in VRAM.  
- **Unloads Each Model:** Sends a POST to `/api/generate` (`prompt:""`, `keep_alive:0`, `stream:false`) for every model found.  
- **One‑Click Action:** Just hit the “Unload Models from VRAM” button—no extra confirmation screens.  
- **Configurable Settings:** Endpoint URL, time‑outs, SSL verification, delay, and logging level are all adjustable.  
- **Robust Error Handling:** Clear emitter messages and detailed logs help you troubleshoot misconfigurations, time‑outs, or connection issues.  
- **Progress Feedback:** Real‑time status updates (with a progress bar) keep you informed while each model unloads.

---

## Features

| Feature | Description |
|---------|-------------|
| **REST API Integration** | Seamlessly interacts with Ollama’s API. |
| **Configurable Endpoint & Time‑out** | Defaults: `http://host.docker.internal:11434`, `3 s`. |
| **SSL Verification Toggle** | `VERIFY_SSL` valve lets you enable or disable TLS cert checks. |
| **Adjustable Logging Level** | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| **Selective Unload** | Pass a `models` list in the request body to unload just one or two heavyweights. |
| **Delay Between Unloads** | Optional (default 200 ms) delay ensures the UI renders progress nicely. |
| **Graceful Error Handling** | Concise user messages + verbose logs when you need them. |
| **Asynchronous Operation** | Non‑blocking design keeps the chat UI responsive. |

---

## Configuration

| Valve | Default | Purpose |
|-------|---------|---------|
| `OLLAMA_ENDPOINT` | `http://host.docker.internal:11434` | Base URL of the Ollama REST API. |
| `REQUEST_TIMEOUT` | `3` seconds | HTTP request time‑out. |
| `UNLOAD_DELAY_MS` | `200` | Wait time between model unloads (set `0` for maximum speed). |
| `VERIFY_SSL` | `true` | Toggle TLS certificate verification. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |

- **Version:** `1.1.2`  
- **Required Open‑WebUI Version:** `0.3.9`

---

## How It Works

1. **Trigger Action**  
   Click the *Unload Models from VRAM* button (next to “Regenerate”) to start.

2. **Retrieve Models**  
   A GET request to `{OLLAMA_ENDPOINT}/api/ps` fetches the current VRAM residents.

3. **Unload Loop**  
   For each model, a POST to `{OLLAMA_ENDPOINT}/api/generate` with payload  
   ```json
   { "model": "<name>", "prompt": "", "keep_alive": 0, "stream": false }
   ```  
   frees the GPU memory. A short delay (`UNLOAD_DELAY_MS`) lets the UI breathe.

4. **Error & Progress Reporting**  
   Status events show which model is unloading and how far you’ve progressed. Problems (timeouts, bad URLs, etc.) surface in‑chat and in logs.

---

## Usage Instructions

1. **Click and Go**  
   Hit the *Unload Models from VRAM* icon → models start unloading immediately.

2. **Watch the Progress Bar**  
   A live counter shows how many models remain. When it hits 100 %, you’re done.

---

## Checking Logs

```bash
# Docker example
docker logs <open-webui_container> --follow
```

Set `LOG_LEVEL=DEBUG` for the deepest insights, or stick with `INFO` for day‑to‑day use.

---

## Code Overview

Built with:

- **`requests`** – HTTP calls to Ollama.  
- **`asyncio`** – Keeps the chat UI fluid.  
- **`pydantic`** – Validates and documents valves.  
- **`logging`** – Structured logs for easy troubleshooting.

---

## Changelog

- **Version 1.1.2**  
  - Added `UNLOAD_DELAY_MS` valve to control a short pause between unloads (default 200 ms) so progress updates aren’t skipped.  
  - Introduced custom toolbar icon via `icon_url` header (Lucide “monitor‑down” SVG embedded as Base‑64 data URI, fully offline‑friendly).  

- **Version 1.1.1** **("DEV")**  
  - Restored inner `Valves` class (action button visible again).  
  - Introduced `VERIFY_SSL` valve for secure API calls.  
  - Added `stream:false` to unload payload for faster single‑JSON responses.  
  - Implemented progress emission and optional *selective unload* via request body.  
  - Upgraded logging initialization (`logging.basicConfig`).  

- **Version 1.0.1**  
  - Removed confirmation prompt for faster operation.  
  - Added configurable logging level valve.  

- **Version 1.0.0**  
  - Initial release with robust error handling, emitter feedback, and async design.

---

## Contributions

Have an idea? Found a bug? Open a pull request or issue on [GitHub](https://github.com/BrandXX/open-webui/).  
First‑timers welcome—let’s build something great together!

---

## License

MIT License — see `LICENSE` for details.

---

## Acknowledgments

Huge thanks to the Open‑WebUI community and contributors for your constant improvements. You rock!

---

Happy unloading, and enjoy your tidy VRAM! 😄🚀

