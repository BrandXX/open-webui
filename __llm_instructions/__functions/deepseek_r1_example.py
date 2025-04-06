"""
title: DeepSeek R1
author: zgccrui
description: 在OpwenWebUI中显示DeepSeek R1模型的思维链 - 仅支持0.5.6及以上版本
version: 1.2.16
licence: MIT
"""

import json
import httpx
import re
from typing import AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field
import asyncio
import traceback


class Pipe:
    class Valves(BaseModel):
        DEEPSEEK_API_BASE_URL: str = Field(
            default="https://api.deepseek.com/v1",
            description="DeepSeek API的基础请求地址",
        )
        DEEPSEEK_API_KEY: str = Field(
            default="", description="用于身份验证的DeepSeek API密钥，可从控制台获取"
        )
        DEEPSEEK_API_MODEL: str = Field(
            default="deepseek-reasoner",
            description="API请求的模型名称，默认为 deepseek-reasoner，多模型名可使用`,`分隔",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.data_prefix = "data:"
        self.emitter = None

    def pipes(self):
        models = self.valves.DEEPSEEK_API_MODEL.split(",")
        return [
            {
                "id": model.strip(),
                "name": model.strip(),
            }
            for model in models
        ]

    async def pipe(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        """主处理管道（已移除缓冲）"""
        thinking_state = {"thinking": -1}  # 用于存储thinking状态
        self.emitter = __event_emitter__
        # 用于存储联网模式下返回的参考资料列表
        stored_references = []

        # 联网搜索供应商 0-无 1-火山引擎 2-PPLX引擎 3-硅基流动
        search_providers = 0
        waiting_for_reference = False

        # 用于处理硅基的 [citation:1] 的栈
        citation_stack_reference = [
            "[",
            "c",
            "i",
            "t",
            "a",
            "t",
            "i",
            "o",
            "n",
            ":",
            "",
            "]",
        ]
        citation_stack = []
        # 临时保存的未处理的字符串
        unprocessed_content = ""

        # 验证配置
        if not self.valves.DEEPSEEK_API_KEY:
            yield json.dumps({"error": "未配置API密钥"}, ensure_ascii=False)
            return
        # 准备请求参数
        headers = {
            "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            # 模型ID提取
            model_id = body["model"].split(".", 1)[-1]
            payload = {**body, "model": model_id}
            # 处理消息以防止连续的相同角色
            messages = payload["messages"]
            i = 0
            while i < len(messages) - 1:
                if messages[i]["role"] == messages[i + 1]["role"]:
                    # 插入具有替代角色的占位符消息
                    alternate_role = (
                        "assistant" if messages[i]["role"] == "user" else "user"
                    )
                    messages.insert(
                        i + 1,
                        {"role": alternate_role, "content": "[Unfinished thinking]"},
                    )
                i += 1

            # 发起API请求
            async with httpx.AsyncClient(http2=True) as client:
                async with client.stream(
                    "POST",
                    f"{self.valves.DEEPSEEK_API_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=300,
                ) as response:

                    # 错误处理
                    if response.status_code != 200:
                        error = await response.aread()
                        yield self._format_error(response.status_code, error)
                        return

                    # 流式处理响应
                    async for line in response.aiter_lines():
                        if not line.startswith(self.data_prefix):
                            continue

                        # 截取 JSON 字符串
                        json_str = line[len(self.data_prefix) :].strip()

                        # 去除首尾空格后检查是否为结束标记
                        if json_str == "[DONE]":
                            return
                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            error_detail = f"解析失败 - 内容：{json_str}，原因：{e}"
                            yield self._format_error("JSONDecodeError", error_detail)
                            return

                        if search_providers == 0:
                            # 检查 delta 中的搜索结果
                            choices = data.get("choices")
                            if not choices or len(choices) == 0:
                                continue  # 跳过没有 choices 的数据块
                            delta = choices[0].get("delta", {})
                            if delta.get("type") == "search_result":
                                search_results = delta.get("search_results", [])
                                if search_results:
                                    ref_count = len(search_results)
                                    yield '<details type="search">\n'
                                    yield f"<summary>已搜索 {ref_count} 个网站</summary>\n"
                                    for idx, result in enumerate(search_results, 1):
                                        yield f'> {idx}. [{result["title"]}]({result["url"]})\n'
                                    yield "</details>\n"
                                    search_providers = 3
                                    stored_references = search_results
                                    continue

                            # 处理参考资料
                            stored_references = data.get("references", []) + data.get(
                                "citations", []
                            )
                            if stored_references:
                                ref_count = len(stored_references)
                                yield '<details type="search">\n'
                                yield f"<summary>已搜索 {ref_count} 个网站</summary>\n"
                            # 如果data中有references，则说明是火山引擎的返回结果
                            if data.get("references"):
                                for idx, reference in enumerate(stored_references, 1):
                                    yield f'> {idx}. [{reference["title"]}]({reference["url"]})\n'
                                yield "</details>\n"
                                search_providers = 1
                            # 如果data中有citations，则说明是PPLX引擎的返回结果
                            elif data.get("citations"):
                                for idx, reference in enumerate(stored_references, 1):
                                    yield f"> {idx}. {reference}\n"
                                yield "</details>\n"
                                search_providers = 2

                        # 方案 A: 检查 choices 是否存在且非空
                        choices = data.get("choices")
                        if not choices or len(choices) == 0:
                            continue  # 跳过没有 choices 的数据块
                        choice = choices[0]

                        # 结束条件判断
                        if choice.get("finish_reason"):
                            return

                        # 状态机处理
                        state_output = await self._update_thinking_state(
                            choice.get("delta", {}), thinking_state
                        )
                        if state_output:
                            yield state_output
                            if state_output == "<think>":
                                yield "\n"

                        # 处理并立即发送内容
                        content = self._process_content(choice["delta"])
                        if content:
                            # 处理思考状态标记
                            if content.startswith("<think>"):
                                content = re.sub(r"^<think>", "", content)
                                yield "<think>"
                                await asyncio.sleep(0.1)
                                yield "\n"
                            elif content.startswith("</think>"):
                                content = re.sub(r"^</think>", "", content)
                                yield "</think>"
                                await asyncio.sleep(0.1)
                                yield "\n"

                            # 处理参考资料
                            if search_providers == 1:
                                # 火山引擎的参考资料处理
                                # 如果文本中包含"摘要"，设置等待标志
                                if "摘要" in content:
                                    waiting_for_reference = True
                                    yield content
                                    continue

                                # 如果正在等待参考资料的数字
                                if waiting_for_reference:
                                    # 如果内容仅包含数字或"、"
                                    if re.match(r"^(\d+|、)$", content.strip()):
                                        numbers = re.findall(r"\d+", content)
                                        if numbers:
                                            num = numbers[0]
                                            ref_index = int(num) - 1
                                            if 0 <= ref_index < len(stored_references):
                                                ref_url = stored_references[ref_index][
                                                    "url"
                                                ]
                                            else:
                                                ref_url = ""
                                            content = f"[[{num}]]({ref_url})"
                                        # 保持等待状态继续处理后续数字
                                    # 如果遇到非数字且非"、"的内容且不含"摘要"，停止等待
                                    elif not "摘要" in content:
                                        waiting_for_reference = False
                            elif search_providers == 2:
                                # PPLX引擎的参考资料处理
                                def replace_ref(m):
                                    idx = int(m.group(1)) - 1
                                    if 0 <= idx < len(stored_references):
                                        return f"[[{m.group(1)}]]({stored_references[idx]})"
                                    return f"[[{m.group(1)}]]()"

                                content = re.sub(r"\[(\d+)\]", replace_ref, content)
                            elif search_providers == 3:
                                skip_outer = False

                                if len(unprocessed_content) > 0:
                                    content = unprocessed_content + content
                                    unprocessed_content = ""

                                for i in range(len(content)):
                                    # 检查 content[i] 是否可访问
                                    if i >= len(content):
                                        break
                                    # 检查 citation_stack_reference[len(citation_stack)] 是否可访问
                                    if len(citation_stack) >= len(
                                        citation_stack_reference
                                    ):
                                        break
                                    if (
                                        content[i]
                                        == citation_stack_reference[len(citation_stack)]
                                    ):
                                        citation_stack.append(content[i])
                                        # 如果 citation_stack 的位数等于 citation_stack_reference 的位数，则修改为 URL 格式返回
                                        if len(citation_stack) == len(
                                            citation_stack_reference
                                        ):
                                            # 检查 citation_stack[10] 是否可访问
                                            if len(citation_stack) > 10:
                                                ref_index = int(citation_stack[10]) - 1
                                                # 检查 stored_references[ref_index] 是否可访问
                                                if (
                                                    0
                                                    <= ref_index
                                                    < len(stored_references)
                                                ):
                                                    ref_url = stored_references[
                                                        ref_index
                                                    ]["url"]
                                                else:
                                                    ref_url = ""

                                                # 将content中剩余的部分保存到unprocessed_content中
                                                unprocessed_content = "".join(
                                                    content[i + 1 :]
                                                )

                                                content = f"[[{citation_stack[10]}]]({ref_url})"
                                                citation_stack = []
                                                skip_outer = False
                                                break
                                        else:
                                            skip_outer = True
                                    elif (
                                        citation_stack_reference[len(citation_stack)]
                                        == ""
                                    ):
                                        # 判断是否为数字
                                        if content[i].isdigit():
                                            citation_stack.append(content[i])
                                            skip_outer = True
                                        else:
                                            # 将 citation_stack 中全部元素拼接成字符串
                                            content = "".join(citation_stack) + content
                                            citation_stack = []
                                    elif (
                                        citation_stack_reference[len(citation_stack)]
                                        == "]"
                                    ):
                                        # 判断前一位是否为数字
                                        if citation_stack[-1].isdigit():
                                            citation_stack[-1] += content[i]
                                            skip_outer = True
                                        else:
                                            content = "".join(citation_stack) + content
                                            citation_stack = []
                                    else:
                                        if len(citation_stack) > 0:
                                            # 将 citation_stack 中全部元素拼接成字符串
                                            content = "".join(citation_stack) + content
                                            citation_stack = []

                                if skip_outer:
                                    continue

                            yield content
        except Exception as e:
            yield self._format_exception(e)

    async def _update_thinking_state(self, delta: dict, thinking_state: dict) -> str:
        """更新思考状态机（简化版）"""
        state_output = ""
        if thinking_state["thinking"] == -1 and delta.get("reasoning_content"):
            thinking_state["thinking"] = 0
            state_output = "<think>"
        elif (
            thinking_state["thinking"] == 0
            and not delta.get("reasoning_content")
            and delta.get("content")
        ):
            thinking_state["thinking"] = 1
            state_output = "\n</think>\n\n"
        return state_output

    def _process_content(self, delta: dict) -> str:
        """直接返回处理后的内容"""
        return delta.get("reasoning_content", "") or delta.get("content", "")

    def _emit_status(self, description: str, done: bool = False) -> Awaitable[None]:
        """发送状态更新"""
        if self.emitter:
            return self.emitter(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": done,
                    },
                }
            )
        return None

    def _format_error(self, status_code: int, error: bytes) -> str:
        if isinstance(error, str):
            error_str = error
        else:
            error_str = error.decode(errors="ignore")
        try:
            err_msg = json.loads(error_str).get("message", error_str)[:200]
        except Exception:
            err_msg = error_str[:200]
        return json.dumps(
            {"error": f"HTTP {status_code}: {err_msg}"}, ensure_ascii=False
        )

    def _format_exception(self, e: Exception) -> str:
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        detailed_error = "".join(tb_lines)
        return json.dumps({"error": detailed_error}, ensure_ascii=False)
