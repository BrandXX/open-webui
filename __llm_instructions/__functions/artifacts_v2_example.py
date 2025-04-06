"""
Changes: 
- Improved formatting for CSS and JavaScript code.
- Structured functions for better readability.
- Cleaned up indentation and spacing for clarity.
- Enhanced CSS styles for better responsiveness, including mobile, tablet, and desktop buttons.
- Improved script functionality for handling multiple artifacts and toggling between views.

author: open-webui, helloworldwastaken, atgehrhardt
author_url:https://github.com/helloworldxdwastaken
orignal_coder_author_url: https://github.com/atgehrhardt 

funding_url: https://github.com/open-webui
version: 2.0.0
required_open_webui_version: 0.3.10 or above
"""

import os
import re
import uuid
import html
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from open_webui.apps.webui.models.files import Files, FileForm
from open_webui.config import UPLOAD_DIR


class MiddlewareHTMLGenerator:
    @staticmethod
    def generate_style():
        return """
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .header {
            height: 40px;
            background-color: #2d2d2d;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 10px;
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        .content-wrapper {
            padding: 20px;
        }
        .content-item {
            width: 100%;
            margin-bottom: 20px;
            border: 1px solid #444;
            background-color: #2d2d2d;
        }
        .content-item.code-view {
            padding: 10px;
        }
        .render-view .rendered-content {
            margin: 0;
            padding: 0;
        }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            background-color: #1e1e1e;
            padding: 10px;
            border-radius: 5px;
            margin: 0;
        }
        code {
            font-family: 'Courier New', Courier, monospace;
        }
        .hidden {
            display: none;
        }
        h2 {
            margin: 0;
            padding: 10px;
            background-color: #3d3d3d;
        }
        .iframe-wrapper {
            width: 100%;
            height: 600px;
            overflow: hidden;
            position: relative;
            resize: both;
            background-color: transparent;
        }
        .content-frame {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: none;
            background-color: transparent;
        }
        .resize-handle {
            position: absolute;
            bottom: 0;
            right: 0;
            width: 20px;
            height: 20px;
            cursor: se-resize;
        }
        .responsive-controls {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
            margin-top: 15px;
        }
        .device-button {
            margin: 0 5px;
            padding: 5px 10px;
            background-color: transparent;
            color: #ffffff;
            border: 1px solid #ffffff;
            cursor: pointer;
            border-radius: 4px;
            transition: background-color 0.3s, color 0.3s;
        }
        .device-button:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        .device-button.active {
            background-color: rgba(255, 255, 255, 0.2);
            font-weight: bold;
        }
        .switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 24px;
        }
        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 24px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        input:checked + .slider {
            background-color: #2196F3;
        }
        input:checked + .slider:before {
            transform: translateX(36px);
        }
        .slider-text {
            position: absolute;
            color: white;
            top: 50%;
            transform: translateY(-50%);
            text-align: center;
            left: 0;
            right: 0;
            font-size: 12px;
        }
        .nav-buttons {
            display: flex;
            align-items: center;
        }
        .nav-button, .select-button, .fullscreen-button {
            background-color: transparent;
            border: none;
            color: #ffffff;
            cursor: pointer;
            font-size: 18px;
            padding: 5px;
            margin: 0 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            transition: background-color 0.3s ease;
        }
        .select-button svg {
            width: 30px;
            height: 30px;
        }
        .nav-button:hover, .select-button:hover, .fullscreen-button:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        .nav-button:disabled {
            color: #666666;
            cursor: not-allowed;
        }
        .nav-button:disabled:hover {
            background-color: transparent;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1001;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.4);
        }
        .modal-content {
            background-color: #2d2d2d;
            margin: 5% auto;
            padding: 20px;
            border: 1px solid #888;
            width: 90%;
            max-width: 800px;
            border-radius: 5px;
            max-height: 80vh;
            overflow-y: auto;
        }
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        .close:hover, .close:focus {
            color: #fff;
            text-decoration: none;
            cursor: pointer;
        }
        .artifact-list {
            list-style-type: none;
            padding: 0;
        }
        .artifact-list li {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #444;
            cursor: pointer;
        }
        .artifact-list li:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        .artifact-info {
            flex: 1;
            margin-right: 10px;
        }
        .artifact-preview {
            width: 200px;
            height: 120px;
            overflow: hidden;
            background-color: transparent;
        }
        .artifact-preview iframe {
            width: 400px;
            height: 240px;
            border: none;
            transform: scale(0.5);
            transform-origin: top left;
            pointer-events: none;
        }
        .editor {
            width: 100%;
            height: 300px;
            font-family: monospace;
            font-size: 14px;
            border: 1px solid #444;
            background-color: #1e1e1e;
            color: #ffffff;
            padding: 10px;
            box-sizing: border-box;
            overflow: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .copy-button {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: #5E5B5A;
            border: none;
            color: white;
            padding: 5px 10px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        .copy-button:hover {
            background-color: #45a049;
        }
        .code-container {
            position: relative;
        }
        .iframe-wrapper:-webkit-full-screen,
        .iframe-wrapper:-moz-full-screen,
        .iframe-wrapper:-ms-fullscreen,
        .iframe-wrapper:fullscreen {
            width: 100%;
            height: 100%;
        }
        """

    @staticmethod
    def generate_script():
        return """
        const totalArtifacts = document.querySelectorAll('.render-view').length;
        let currentArtifact = 1;
        let isCodeView = false;

        const modal = document.getElementById("artifactModal");
        const selectButton = document.getElementById("selectArtifact");
        const closeButton = document.getElementsByClassName("close")[0];
        const artifactList = document.getElementById("artifactList");
        const fullscreenButton = document.getElementById('fullscreenButton');
        const body = document.body;

        window.addEventListener('load', () => {
            for (let i = 0; i < totalArtifacts; i++) {
                ['html', 'css', 'js'].forEach(type => {
                    const storedContent = localStorage.getItem(`artifact_${i}_${type}`);
                    if (storedContent) {
                        const editor = document.getElementById(`${type}-editor-${i}`);
                        if (editor) {
                            editor.value = storedContent;
                        }
                    }
                });
            }
            reloadCurrentArtifact();
        });

        function applyStoredChanges(artifactNumber) {
            ['html', 'css', 'js'].forEach(type => {
                const storedContent = localStorage.getItem(`artifact_${artifactNumber - 1}_${type}`);
                if (storedContent) {
                    updateContent(type, artifactNumber - 1, true);
                }
            });
        }

        document.getElementById('toggleView').addEventListener('change', function() {
            isCodeView = this.checked;
            const sliderText = document.querySelector('.slider-text');
            sliderText.textContent = isCodeView ? 'Code' : 'Render';
            updateArtifactVisibility();
        });

        function updateArtifactVisibility() {
            document.querySelectorAll('.content-item').forEach(item => {
                const isCorrectArtifact = item.dataset.artifact == currentArtifact;
                const isCorrectView = (item.classList.contains('render-view') && !isCodeView) || 
                                      (item.classList.contains('code-view') && isCodeView);
                item.classList.toggle('hidden', !(isCorrectArtifact && isCorrectView));
            });
            document.getElementById('prevArtifact').disabled = currentArtifact === 1;
            document.getElementById('nextArtifact').disabled = currentArtifact === totalArtifacts;
        }

        function navigateToArtifact(artifactNumber) {
            currentArtifact = artifactNumber;
            updateArtifactVisibility();
            reloadCurrentArtifact();
            modal.style.display = "none";
        }

        function reloadCurrentArtifact() {
            const frame = document.querySelector(`.content-item[data-artifact="${currentArtifact}"] .content-frame`);
            if (frame) {
                const currentSrcdoc = frame.getAttribute('data-original-content');
                frame.srcdoc = '';
                setTimeout(() => {
                    frame.srcdoc = currentSrcdoc;
                }, 0);
            }
        }

        document.getElementById('prevArtifact').addEventListener('click', () => {
            if (currentArtifact > 1) {
                currentArtifact--;
                updateArtifactVisibility();
                reloadCurrentArtifact();
            }
        });

        document.getElementById('nextArtifact').addEventListener('click', () => {
            if (currentArtifact < totalArtifacts) {
                currentArtifact++;
                updateArtifactVisibility();
                reloadCurrentArtifact();
            }
        });

        function updateContent(type, index, skipReload = false) {
            const frame = document.querySelector(`.content-item[data-artifact="${index + 1}"] .content-frame`);
            const editor = document.getElementById(`${type}-editor-${index}`);
            const content = editor.value;
            
            let updatedSrcdoc = frame.getAttribute('data-original-content');
            const parser = new DOMParser();
            const doc = parser.parseFromString(updatedSrcdoc, 'text/html');
            
            if (type === 'html') {
                doc.body.innerHTML = content;
            } else if (type === 'css') {
                let styleTag = doc.querySelector('style');
                if (!styleTag) {
                    styleTag = doc.createElement('style');
                    doc.head.appendChild(styleTag);
                }
                styleTag.textContent = content;
            } else if (type === 'js') {
                let scriptTag = doc.querySelector('script:not([src])');
                if (!scriptTag) {
                    scriptTag = doc.createElement('script');
                    doc.body.appendChild(scriptTag);
                }
                scriptTag.textContent = content;
            }
        
            updatedSrcdoc = new XMLSerializer().serializeToString(doc);
            
            frame.setAttribute('data-original-content', updatedSrcdoc);
            
            if (!skipReload) {
                frame.srcdoc = '';
                setTimeout(() => {
                    frame.srcdoc = updatedSrcdoc;
                }, 0);
            }
        
            localStorage.setItem(`artifact_${index}_${type}`, content);
            console.log(`Content updated for artifact ${index + 1}, type ${type}`);
        }

        function copyToClipboard(button, elementId) {
            const codeElement = document.getElementById(elementId);
            const textArea = document.createElement('textarea');
            textArea.value = codeElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            const originalText = button.textContent;
            button.textContent = 'Copied!';
            setTimeout(() => {
                button.textContent = originalText;
            }, 2000);
        }

        selectButton.onclick = function() {
            const makeTransparent = (doc) => {
                doc.body.style.background = 'transparent';
                const styleEl = doc.createElement('style');
                styleEl.textContent = 'body { background: transparent !important; }';
                doc.head.appendChild(styleEl);
            };

            artifactList.innerHTML = '';
            document.querySelectorAll('.content-frame').forEach((frame, index) => {
                const li = document.createElement('li');
                const previewContent = frame.getAttribute('srcdoc');
                
                li.innerHTML = `
                    <div class="artifact-info">
                        <strong>Artifact ${index + 1}</strong>
                    </div>
                    <div class="artifact-preview">
                        <iframe sandbox="allow-scripts allow-same-origin"></iframe>
                    </div>
                `;
                li.onclick = function() { navigateToArtifact(index + 1); };
                artifactList.appendChild(li);

                const previewIframe = li.querySelector('.artifact-preview iframe');
                previewIframe.onload = function() {
                    makeTransparent(this.contentDocument);
                    this.contentDocument.body.style.transform = 'scale(0.5)';
                    this.contentDocument.body.style.transformOrigin = 'top left';
                    this.style.pointerEvents = 'none';
                };
                previewIframe.srcdoc = previewContent;
            });
            modal.style.display = "block";
        }

        closeButton.onclick = function() {
            modal.style.display = "none";
        }

        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = "none";
            }
        }

        document.querySelectorAll('.device-button').forEach(button => {
            button.addEventListener('click', function() {
                const width = this.getAttribute('data-width');
                const wrapper = this.closest('.content-item').querySelector('.iframe-wrapper');
                const iframe = wrapper.querySelector('.content-frame');
                
                if (width === '100%') {
                    wrapper.style.width = '100%';
                    wrapper.style.height = '600px';
                    iframe.style.width = '100%';
                    iframe.style.height = '100%';
                } else {
                    wrapper.style.width = width;
                    wrapper.style.height = '80vh';
                    iframe.style.width = width;
                    iframe.style.height = '100%';
                }
                
                this.closest('.responsive-controls').querySelectorAll('.device-button').forEach(btn => {
                    btn.classList.remove('active');
                });
                this.classList.add('active');
            });
        });

        document.querySelectorAll('.resize-handle').forEach(handle => {
            handle.addEventListener('mousedown', initResize, false);
        });

        function initResize(e) {
            window.addEventListener('mousemove', resize, false);
            window.addEventListener('mouseup', stopResize, false);
        }

        function resize(e) {
            if (!body.classList.contains('fullscreen')) {
                const wrapper = e.target.closest('.iframe-wrapper');
                wrapper.style.width = (e.clientX - wrapper.offsetLeft) + 'px';
                wrapper.style.height = (e.clientY - wrapper.offsetTop) + 'px';
            }
        }

        function stopResize(e) {
            window.removeEventListener('mousemove', resize, false);
            window.removeEventListener('mouseup', stopResize, false);
        }

        function toggleFullscreen() {
            const currentFrame = document.querySelector(`.content-item[data-artifact="${currentArtifact}"] .iframe-wrapper`);
            
            if (!document.fullscreenElement) {
                if (currentFrame.requestFullscreen) {
                    currentFrame.requestFullscreen();
                } else if (currentFrame.mozRequestFullScreen) {
                    currentFrame.mozRequestFullScreen();
                } else if (currentFrame.webkitRequestFullscreen) {
                    currentFrame.webkitRequestFullscreen();
                } else if (currentFrame.msRequestFullscreen) {
                    currentFrame.msRequestFullscreen();
                }
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.mozCancelFullScreen) {
                    document.mozCancelFullScreen();
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                } else if (document.msExitFullscreen) {
                    document.msExitFullscreen();
                }
            }
        }
        
        fullscreenButton.addEventListener('click', toggleFullscreen);

        document.addEventListener('fullscreenchange', updateFullscreenButtonIcon);
        document.addEventListener('webkitfullscreenchange', updateFullscreenButtonIcon);
        document.addEventListener('mozfullscreenchange', updateFullscreenButtonIcon);
        document.addEventListener('MSFullscreenChange', updateFullscreenButtonIcon);

        function updateFullscreenButtonIcon() {
            if (document.fullscreenElement) {
                fullscreenButton.innerHTML = `
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3 12h7v2H5v5H3v-7zm18 0h-7v2h5v5h2v-7zM3 7h2V5h5V3H3v4zm18 0h-2V5h-5V3h7v4z" fill="currentColor"/>
                    </svg>
                `;
            } else {
                fullscreenButton.innerHTML = `
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3 3h7v2H5v5H3V3zm18 0h-7v2h5v5h2V3zM3 21h7v-2H5v-5H3v7zm18 0h-7v-2H5v-5H3v7z" fill="currentColor"/>
                    </svg>
                `;
            }
        }

        updateArtifactVisibility();
        """

    @staticmethod
    def generate_content_item(i, page):
        html_content = page.get("html", "")
        raw_html = page.get("raw_html", "")
        css_content = page.get("css", "")
        js_content = page.get("js", "")

        escaped_html = html.escape(raw_html)
        escaped_css = html.escape(css_content)
        escaped_js = html.escape(js_content)

        base_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    {css_content}
                </style>
            </head>
            <body>
                {html_content}
                <script>
                    {js_content}
                </script>
            </body>
            </html>
        """
        escaped_base_html = html.escape(base_html)

        return f"""
            <div class='content-item render-view' data-artifact="{i+1}">
                <h2>HTML Content {i+1}</h2>
                <div class="responsive-controls">
                    <button class="device-button" data-width="320px">Mobile</button>
                    <button class="device-button" data-width="768px">Tablet</button>
                    <button class="device-button active" data-width="100%">Desktop</button>
                </div>
                <div class="iframe-wrapper">
                    <iframe class="content-frame" sandbox="allow-scripts" srcdoc="{escaped_base_html}" data-original-content="{escaped_base_html}"></iframe>
                    <div class="resize-handle"></div>
                </div>
            </div>
            <div class='content-item code-view hidden' data-artifact="{i+1}">
                <h2>HTML Content {i+1}</h2>
                <div class="code-container">
                    <button class="copy-button" onclick="copyToClipboard(this, 'html-editor-{i}')">Copy</button>
                    <pre class="editor" id="html-editor-{i}">{escaped_html}</pre>
                </div>
            </div>
            {"" if not css_content else f'''
            <div class='content-item code-view hidden' data-artifact="{i+1}">
                <h2>CSS Content {i+1}</h2>
                <div class="code-container">
                    <button class="copy-button" onclick="copyToClipboard(this, 'css-editor-{i}')">Copy</button>
                    <pre class="editor" id="css-editor-{i}">{escaped_css}</pre>
                </div>
            </div>
            '''}
            {"" if not js_content else f'''
            <div class='content-item code-view hidden' data-artifact="{i+1}">
                <h2>JavaScript Content {i+1}</h2>
                <div class="code-container">
                    <button class="copy-button" onclick="copyToClipboard(this, 'js-editor-{i}')">Copy</button>
                    <pre class="editor" id="js-editor-{i}">{escaped_js}</pre>
                </div>
            </div>
            '''}
        """

    @classmethod
    def create_middleware_html(cls, pages):
        content_items = "".join(
            cls.generate_content_item(i, page) for i, page in enumerate(pages)
        )

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Generated Content</title>
            <style>
                {cls.generate_style()}
            </style>
        </head>
        <body>
            <div class="header">
                <label class="switch">
                    <input type="checkbox" id="toggleView">
                    <span class="slider">
                        <span class="slider-text">Render</span>
                    </span>
                </label>
                <div class="nav-buttons">
                    <button id="prevArtifact" class="nav-button" aria-label="Previous artifact">&#10094;</button>
                    <button id="selectArtifact" class="select-button" aria-label="Select artifact">
                        <svg width="30" height="30" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                            <rect x="90" y="20" width="20" height="20" fill="#FFFFFF"/>
                            <rect x="70" y="60" width="20" height="20" fill="#FFFFFF"/>
                            <rect x="110" y="60" width="20" height="20" fill="#FFFFFF"/>
                            <rect x="50" y="100" width="20" height="20" fill="#FFFFFF"/>
                            <rect x="70" y="100" width="20" height="20" fill="#FFFFFF"/>
                            <rect x="90" y="100" width="20" height="20" fill="#FFFFFF"/>
                            <rect x="110" y="100" width="20" height="20" fill="#FFFFFF"/>
                            <rect x="130" y="100" width="20" height="20" fill="#FFFFFF"/>
                            <rect x="30" y="140" width="20" height="20" fill="#FFFFFF"/>
                            <rect x="150" y="140" width="20" height="20" fill="#FFFFFF"/>
                        </svg>
                    </button>
                    <button id="nextArtifact" class="nav-button" aria-label="Next artifact">&#10095;</button>
                    <button id="fullscreenButton" class="fullscreen-button" aria-label="Toggle fullscreen">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M3 3h7v2H5v5H3V3zm18 0h-7v2h5v5h2V3zM3 21h7v-2H5v-5H3v7zm18 0h-7v-2h5v-5h2v7z" fill="currentColor"/>
                        </svg>
                    </button>
                </div>
            </div>
            <div class='content-wrapper'>
                {content_items}
            </div>
            <div id="artifactModal" class="modal">
                <div class="modal-content">
                    <span class="close">&times;</span>
                    <h2>Select Artifact</h2>
                    <ul class="artifact-list" id="artifactList"></ul>
                </div>
            </div>
            <script>
                {cls.generate_script()}
            </script>
        </body>
        </html>
        """


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.viz_dir = "visualizations"
        self.html_dir = "html"
        self.middleware_file = "middleware.html"
        self.current_artifact = None

    def ensure_chat_directory(self, chat_id, content_type):
        chat_dir = os.path.join(UPLOAD_DIR, self.viz_dir, content_type, chat_id)
        os.makedirs(chat_dir, exist_ok=True)
        return chat_dir

    def extract_content(self, content, pattern):
        return re.findall(pattern, content, re.IGNORECASE | re.DOTALL)

    def write_content_to_file(self, content, user_id, chat_id, content_type):
        chat_dir = self.ensure_chat_directory(chat_id, content_type)
        filename = f"{content_type}_{uuid.uuid4()}.html"
        file_path = os.path.join(chat_dir, filename)

        with open(file_path, "w") as f:
            f.write(content)

        relative_path = os.path.join(self.viz_dir, content_type, chat_id, filename)
        file_form = FileForm(
            id=str(uuid.uuid4()),
            filename=relative_path,
            meta={
                "name": filename,
                "content_type": "text/html",
                "size": len(content),
                "path": file_path,
            },
        )
        return Files.insert_new_file(user_id, file_form).id

    def parse_content(self, content):
        html_pattern = r"```(?:html|xml)\s*([\s\S]*?)\s*```"
        css_pattern = r"```css\s*([\s\S]*?)\s*```"
        js_pattern = r"```javascript\s*([\s\S]*?)\s*```"
        svg_pattern = r"<svg[\s\S]*?</svg>"

        html_blocks = self.extract_content(content, html_pattern)
        css_blocks = self.extract_content(content, css_pattern)
        js_blocks = self.extract_content(content, js_pattern)
        standalone_svg_blocks = self.extract_content(content, svg_pattern)

        if not self.current_artifact:
            self.current_artifact = {"html": "", "css": "", "js": "", "raw_html": ""}

        if html_blocks:
            self.current_artifact["html"] = html_blocks[0]
            self.current_artifact["raw_html"] = html_blocks[0]

        if css_blocks:
            self.current_artifact["css"] = css_blocks[0]

        if js_blocks:
            self.current_artifact["js"] = js_blocks[0]

        if standalone_svg_blocks:
            self.current_artifact["html"] = standalone_svg_blocks[0]
            self.current_artifact["raw_html"] = standalone_svg_blocks[0]

        return [self.current_artifact] if any(self.current_artifact.values()) else []

    def create_middleware_html(self, pages):
        return MiddlewareHTMLGenerator.create_middleware_html(pages)

    def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if "messages" in body and body["messages"] and __user__ and "id" in __user__:
            last_message = body["messages"][-1]["content"]
            chat_id = body.get("chat_id")

            if chat_id:
                try:
                    pages = self.parse_content(last_message)

                    if pages:
                        middleware_content = self.create_middleware_html(pages)
                        middleware_id = self.write_content_to_file(
                            middleware_content,
                            __user__["id"],
                            chat_id,
                            self.html_dir,
                        )

                        body["messages"][-1][
                            "content"
                        ] += f"\n\n{{{{HTML_FILE_ID_{middleware_id}}}}}"

                except Exception as e:
                    error_msg = (
                        f"Error processing content: {str(e)}\n{traceback.format_exc()}"
                    )
                    print(error_msg)
                    body["messages"][-1][
                        "content"
                    ] += f"\n\nError: Failed to process content. Details: {error_msg}"
            else:
                print("chat_id is missing")

        return body
