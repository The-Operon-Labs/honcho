"""
src/llm/providers/nvidia_provider.py
-------------------------------------------
NVIDIA API Inference Driver for text completions (Nemotron).
"""

from typing import AsyncGenerator, Optional, Dict, Any, List
from openai import AsyncOpenAI
import time
import logging

logger = logging.getLogger(__name__)

class InferenceRequest:
    def __init__(self, messages: list[dict[str, Any]], model_name: str, max_tokens: int, temperature: float = 0.7, top_p: float = 0.95, extra_body: dict = None, tools: list = None, task_type: str = "general"):
        self.messages = messages
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.extra_body = extra_body
        self.tools = tools
        self.task_type = task_type


class InferenceResponse:
    def __init__(self, content: str, tool_calls: list = None, usage: dict = None, provider_name: str = "", model_used: str = "", execution_time_s: float = 0.0, input_kb: float = 0.0, output_kb: float = 0.0, tps: float = 0.0):
        self.content = content
        self.tool_calls = tool_calls
        self.usage = usage
        self.provider_name = provider_name
        self.model_used = model_used
        self.execution_time_s = execution_time_s
        self.input_kb = input_kb
        self.output_kb = output_kb
        self.tps = tps


class StreamChunk:
    def __init__(self, delta_content: str = None, finish_reason: str = None):
        self.delta_content = delta_content
        self.finish_reason = finish_reason


class NVIDIAInferenceProvider:
    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._base_url = base_url
        self._default_model = "nvidia/nemotron-4-340b-instruct"
        self._async_client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

    @property
    def provider_name(self) -> str:
        return "nvidia"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        model = request.model_name or self._default_model
        start_time = time.perf_counter()

        input_str = str(request.messages)
        input_kb = round(len(input_str.encode('utf-8')) / 1024.0, 2)
        logger.info(
            "[LLM ITERATION START] Task='%s' | Provider='nvidia' | Model='%s' | Input Size=%.2f KB",
            request.task_type, model, input_kb
        )
        logger.info("[LLM INPUT PROMPT CONTENT]:\n%s", input_str)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p or 0.95,
        }
        if request.extra_body:
            kwargs["extra_body"] = request.extra_body
        if request.tools:
            kwargs["tools"] = request.tools

        try:
            resp = await self._async_client.chat.completions.create(**kwargs)
            execution_time_s = round(time.perf_counter() - start_time, 3)
            choice = resp.choices[0].message

            tool_calls = None
            if choice.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in choice.tool_calls
                ]

            content_text = choice.content or ""
            output_kb = round(len(content_text.encode('utf-8')) / 1024.0, 2)

            usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            tps = 0.0
            if resp.usage:
                usage_dict = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens
                }
                if execution_time_s > 0:
                    tps = round(resp.usage.completion_tokens / execution_time_s, 2)

            logger.info(
                "[LLM ITERATION END] Task='%s' | Provider='nvidia' | Model='%s' | Latency=%.3fs | Prompt Tokens=%d (%.2f KB) | Completion Tokens=%d (%.2f KB) | Speed=%.2f tokens/sec",
                request.task_type, model, execution_time_s, usage_dict["prompt_tokens"], input_kb, usage_dict["completion_tokens"], output_kb, tps
            )
            logger.info("[LLM FULL OUTPUT RESPONSE CONTENT]:\n%s", content_text)

            return InferenceResponse(
                content=choice.content,
                tool_calls=tool_calls,
                usage=usage_dict,
                provider_name=self.provider_name,
                model_used=model,
                execution_time_s=execution_time_s,
                input_kb=input_kb,
                output_kb=output_kb,
                tps=tps
            )
        except Exception as e:
            logger.error("[NVIDIAProvider] Error executing text completion: %s", e)
            raise

    async def generate_stream(self, request: InferenceRequest) -> AsyncGenerator[StreamChunk, None]:
        model = request.model_name or self._default_model
        logger.info("[NVIDIAProvider] Streaming request for task '%s' with model '%s'", request.task_type, model)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p or 0.95,
            "stream": True,
        }
        if request.extra_body:
            kwargs["extra_body"] = request.extra_body

        try:
            stream = await self._async_client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                
                yield StreamChunk(
                    delta_content=delta.content,
                    finish_reason=finish_reason
                )
        except Exception as e:
            logger.error("[NVIDIAProvider] Error streaming completion: %s", e)
            raise
